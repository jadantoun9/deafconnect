"""
landmark_dataset.py — landmark-track dataset on top of cached MediaPipe output.

Track 1 doesn't run MediaPipe at training time (too slow). Instead, the owner
runs `extract_landmarks.py` once, which writes `<video_id>.npz` files into
`landmarks_cache/` with two arrays:

    landmarks  shape [F, K, 3]   float32  — F frames, K keypoints (x, y, z)
    valid      shape [F, K]      bool     — True if a detection was made

This dataset reads those caches, samples T frames to match the video pipeline,
and returns:

    (landmarks, valid_mask, label, video_id, meta)

Where landmarks are normalised so the (x, y) span across all keypoints in a
frame fits in [-1, 1] (translation- and scale-invariant). Z is left raw —
MediaPipe's depth estimate is noisy but useful as a coarse signal.

If a clip's landmark cache is missing, the dataset raises immediately rather
than zero-filling — that would silently train the model on garbage.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.registry import register_dataset


def _normalise_landmarks(lm: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-frame zero-centre + scale to [-1, 1] over (x, y); leave z raw."""
    lm = lm.copy()
    F, K, _ = lm.shape
    for f in range(F):
        m = valid[f]
        if m.sum() < 2:
            continue
        xy = lm[f, m, :2]
        cx, cy = xy.mean(axis=0)
        lm[f, :, 0] -= cx
        lm[f, :, 1] -= cy
        max_abs = np.max(np.abs(lm[f, m, :2]))
        if max_abs > 1e-6:
            lm[f, :, :2] /= max_abs
    return lm


@register_dataset("landmark")
class LandmarkDataset(Dataset):
    """
    Args:
        manifest_path:  wlasl<N>_<split>.json from prepare_wlasl.py.
        landmarks_root: directory of <video_id>.npz files from
                        extract_landmarks.py.
        num_frames:     T (per-clip frame budget after sampling).
        sampling:       'uniform' | 'random' | 'dense' (same semantics as
                        VideoDataset).
        normalise:      apply per-frame translation + scale normalisation.
        held_out_signers / held_out_mode: same as VideoDataset.
    """

    def __init__(
        self,
        manifest_path: Path | str,
        landmarks_root: Path | str,
        num_frames: int = 32,
        sampling: str = "uniform",
        normalise: bool = True,
        transform: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None,
        augment: bool | dict[str, Any] = False,
        keypoint_subset: str = "all",
        held_out_signers: set[Any] | None = None,
        held_out_mode: str | None = None,
        seed: int = 0,
    ):
        self.manifest_path = Path(manifest_path)
        self.landmarks_root = Path(landmarks_root)
        self.num_frames = num_frames
        self.sampling = sampling
        self.normalise = normalise

        # Keypoint slicing — caches are always K=75; if a subset is requested,
        # we select those indices at __getitem__ time (zero re-extraction).
        from data.keypoint_subsets import get_subset
        subset = get_subset(keypoint_subset)
        self.keypoint_subset = keypoint_subset
        self.keypoint_indices = list(subset["indices"])

        # YAML-friendly: `augment: true` (default config) or a dict mapping to
        # LandmarkAugmentConfig fields. `transform=` still wins if explicitly
        # passed, so callers building a custom pipeline aren't surprised.
        if transform is None and augment:
            from data.landmark_augmentations import (
                LandmarkAugment, LandmarkAugmentConfig,
            )
            aug_kwargs = augment if isinstance(augment, dict) else {}
            transform = LandmarkAugment(
                LandmarkAugmentConfig(**aug_kwargs),
                keypoint_subset=keypoint_subset,
            )
        self.transform = transform
        self.rng = random.Random(seed)

        items: list[dict[str, Any]] = json.loads(self.manifest_path.read_text())
        if held_out_signers and held_out_mode:
            if held_out_mode == "reserve":
                items = [it for it in items if it.get("signer_id") in held_out_signers]
            elif held_out_mode == "exclude":
                items = [it for it in items if it.get("signer_id") not in held_out_signers]
            else:
                raise ValueError(f"unknown held_out_mode={held_out_mode!r}")

        # Pre-filter to clips that actually have a landmark cache. Missing
        # caches are skipped with a warning rather than crashing later.
        self.items: list[dict[str, Any]] = []
        missing = 0
        for it in items:
            if (self.landmarks_root / f"{it['video_id']}.npz").exists():
                self.items.append(it)
            else:
                missing += 1
        if missing:
            print(f"[LandmarkDataset] {missing} clips missing landmark caches and will be skipped.")

    def __len__(self) -> int:
        return len(self.items)

    def _sample_indices(self, total: int) -> list[int]:
        T = self.num_frames
        if total <= 0:
            return [0] * T
        if self.sampling == "uniform":
            if total == 1:
                return [0] * T
            return list(np.linspace(0, total - 1, T, dtype=int))
        if self.sampling == "random":
            if total <= T:
                return list(range(total)) + [total - 1] * (T - total)
            start = self.rng.randint(0, total - T)
            return list(range(start, start + T))
        if self.sampling == "dense":
            return list(range(min(T, total))) + [total - 1] * max(0, T - total)
        raise ValueError(f"unknown sampling={self.sampling!r}")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, str, dict[str, Any]]:
        item = self.items[idx]
        cache_path = self.landmarks_root / f"{item['video_id']}.npz"
        z = np.load(cache_path)
        lm = z["landmarks"]      # [F, K, 3] float32
        valid = z["valid"]       # [F, K] bool

        indices = self._sample_indices(lm.shape[0])
        lm = lm[indices]
        valid = valid[indices]

        # Slice down to the requested keypoint subset BEFORE normalisation, so
        # the per-frame translate-and-scale uses only the kept keypoints.
        if self.keypoint_indices and len(self.keypoint_indices) != lm.shape[1]:
            kp_idx = np.asarray(self.keypoint_indices, dtype=np.int64)
            lm = lm[:, kp_idx, :]
            valid = valid[:, kp_idx]

        if self.normalise:
            lm = _normalise_landmarks(lm, valid)

        lm_t = torch.from_numpy(lm).float()
        valid_t = torch.from_numpy(valid).bool()
        if self.transform is not None:
            lm_t, valid_t = self.transform(lm_t, valid_t)

        meta = {
            "gloss": item.get("gloss"),
            "signer_id": item.get("signer_id"),
            "fps": item.get("fps"),
        }
        return lm_t, valid_t, int(item["label"]), str(item["video_id"]), meta
