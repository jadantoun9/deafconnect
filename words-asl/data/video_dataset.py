"""
video_dataset.py — shared base dataset for raw-video tracks.

Used by:
  - Track 2 (end-to-end video transformer)
  - Track 3 (distilled hybrid)
  - Robustness evaluation (wraps this with corruptions)

Returns a 4-tuple per item:
    (frames, label, video_id, meta)

Where:
    frames    : torch.Tensor [T, 3, H, W], float32, normalised to [0, 1]
    label     : int (class index)
    video_id  : str (so eval scripts can join back to per-clip metadata)
    meta      : dict with 'gloss', 'signer_id', 'split_origin' for downstream
                analysis (held-out-signer eval, per-class metrics).

Three temporal sampling strategies:
  - 'uniform' : T frames evenly spaced over the clip (default for eval).
  - 'random'  : T contiguous frames from a random start (default for train).
  - 'dense'   : every frame from start to start + T (used for benchmarking).

Frame decoding is via decord, which is much faster than OpenCV for this
workload because it returns NumPy arrays directly without per-frame Python
overhead.

The dataset deliberately holds no global state about augmentations — the
caller passes in a callable transform that operates on `[T, 3, H, W]`. See
`augmentations.py`.
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


SamplingStrategy = str  # one of: 'uniform', 'random', 'dense'


@register_dataset("video")
class VideoDataset(Dataset):
    """
    Args:
        manifest_path:  path to a wlasl<N>_<split>.json produced by
                        prepare_wlasl.py.
        videos_root:    root directory under which manifest's video_path
                        entries resolve.
        num_frames:     T, frames per clip.
        sampling:       'uniform' | 'random' | 'dense'.
        resolution:     output H = W. Frames are bilinearly resized.
        transform:      optional callable [T, 3, H, W] float -> [T, 3, H, W].
        held_out_signers: optional set of signer IDs. If `held_out_mode` is
                          'reserve', clips with these signers are *kept* (so
                          you get the held-out-signer eval set). If 'exclude',
                          they are dropped (so you get the standard split
                          minus held-out signers).
        held_out_mode:  'reserve' | 'exclude' | None.
    """

    def __init__(
        self,
        manifest_path: Path | str,
        videos_root: Path | str,
        num_frames: int = 16,
        sampling: SamplingStrategy = "uniform",
        resolution: int = 224,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        held_out_signers: set[Any] | None = None,
        held_out_mode: str | None = None,
        seed: int = 0,
    ):
        self.manifest_path = Path(manifest_path)
        self.videos_root = Path(videos_root)
        self.num_frames = num_frames
        self.sampling = sampling
        self.resolution = resolution
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
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def _resolve_path(self, item: dict[str, Any]) -> Path:
        candidate = (self.videos_root / item["video_path"]).resolve()
        if candidate.exists():
            return candidate
        # Fallback: bare filename in videos_root.
        return self.videos_root / Path(item["video_path"]).name

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
                # Pad by repeating the last frame; keep things deterministic.
                return list(range(total)) + [total - 1] * (T - total)
            start = self.rng.randint(0, total - T)
            return list(range(start, start + T))
        if self.sampling == "dense":
            return list(range(min(T, total))) + [total - 1] * max(0, T - total)
        raise ValueError(f"unknown sampling={self.sampling!r}")

    def _load_frames(self, video_path: Path, indices: list[int]) -> torch.Tensor:
        # decord is the fast path; fall back to OpenCV so the dataset still
        # works on machines where decord isn't installed (CI smoke tests).
        try:
            import decord
            decord.bridge.set_bridge("native")
            vr = decord.VideoReader(str(video_path), width=self.resolution, height=self.resolution)
            indices = [min(i, len(vr) - 1) for i in indices]
            frames = vr.get_batch(indices).asnumpy()  # [T, H, W, 3] uint8
        except (ImportError, RuntimeError):
            frames = self._load_frames_opencv(video_path, indices)

        # NHWC uint8 -> NCHW float32 in [0, 1]
        t = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        return t

    def _load_frames_opencv(self, video_path: Path, indices: list[int]) -> np.ndarray:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out = []
        wanted = sorted(set(min(i, max(total - 1, 0)) for i in indices))
        idx = 0
        cache: dict[int, np.ndarray] = {}
        for w in wanted:
            cap.set(cv2.CAP_PROP_POS_FRAMES, w)
            ok, frame = cap.read()
            if not ok:
                # Use a previously-read frame or a black image as a fallback.
                cache[w] = next(iter(cache.values())) if cache else np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.resolution, self.resolution))
                cache[w] = frame
        cap.release()
        out = np.stack([cache[min(i, max(total - 1, 0))] for i in indices], axis=0)
        return out

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str, dict[str, Any]]:
        item = self.items[idx]
        path = self._resolve_path(item)

        # Cheap upper bound on length: rely on decord/cv2 internally; we don't
        # store frame counts in the manifest because YouTube re-encodes change
        # them. _sample_indices receives the actual count once decoded.
        try:
            import decord
            vr = decord.VideoReader(str(path))
            total = len(vr)
        except Exception:
            import cv2
            cap = cv2.VideoCapture(str(path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

        indices = self._sample_indices(total)
        frames = self._load_frames(path, indices)
        if self.transform is not None:
            frames = self.transform(frames)

        meta = {
            "gloss": item.get("gloss"),
            "signer_id": item.get("signer_id"),
            "fps": item.get("fps"),
        }
        return frames, int(item["label"]), str(item["video_id"]), meta
