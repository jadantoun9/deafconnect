"""
landmark_augmentations.py — temporally-consistent augmentations for landmark
sequences (Track 1).

Operates on the post-normalisation tensors produced by LandmarkDataset:
    lm:    torch.float32 [T, 75, 3], (x, y) ~ centred and scaled to [-1, 1]
    valid: torch.bool    [T, 75]

Critical property (mirrors /data/augmentations.py): every random
parameter is sampled once per clip and applied to all T frames. Per-frame
randomness destroys temporal coherence and trains the model on a stream of
unrelated motions.

The mirror op is the subtle one: flipping the x coordinate alone is wrong
because MediaPipe Holistic stores hand keypoints in two anatomical slots
(left/right hand) and pose keypoints have left/right pairs (shoulder, elbow,
hip, etc.). A faithful mirror has to flip x AND swap the left/right slots —
otherwise a right-handed sign becomes a structurally inconsistent half-mirror
that the model can never learn from.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch


NUM_POSE = 33
NUM_HAND = 21
NUM_KEYPOINTS = NUM_POSE + 2 * NUM_HAND  # 75
LEFT_HAND_SLICE = slice(NUM_POSE, NUM_POSE + NUM_HAND)
RIGHT_HAND_SLICE = slice(NUM_POSE + NUM_HAND, NUM_KEYPOINTS)

# MediaPipe pose left/right symmetry pairs. A mirror has to swap these to
# keep "left shoulder" on the signer's left after flipping the image.
POSE_LR_PAIRS: list[tuple[int, int]] = [
    (1, 4), (2, 5), (3, 6),     # eyes
    (7, 8),                      # ears
    (9, 10),                     # mouth corners
    (11, 12),                    # shoulders
    (13, 14),                    # elbows
    (15, 16),                    # wrists
    (17, 18), (19, 20), (21, 22),  # hands (pose-side: pinky/index/thumb)
    (23, 24),                    # hips
    (25, 26), (27, 28),          # knees / ankles
    (29, 30), (31, 32),          # heels / foot index
]


@dataclass
class LandmarkAugmentConfig:
    """All probabilities and ranges. Set any to 0 to disable that op."""
    mirror_p: float = 0.5
    rotate_max_deg: float = 15.0
    scale_range: tuple[float, float] = (0.85, 1.15)
    translate_max: float = 0.05            # fraction of [-1, 1] coord range
    coord_noise_std: float = 0.01          # Gaussian std on (x, y)
    joint_dropout_p: float = 0.05          # per-keypoint probability per clip
    temporal_jitter_std: float = 1.5       # frame-index jitter (used by dataset sampler)


def _build_mirror_perm(num_keypoints: int = NUM_KEYPOINTS) -> torch.Tensor:
    """Permutation that swaps left/right hand slots and pose L/R pairs (full
    K=75 layout)."""
    perm = list(range(num_keypoints))
    for i in range(NUM_HAND):
        a = NUM_POSE + i
        b = NUM_POSE + NUM_HAND + i
        perm[a], perm[b] = perm[b], perm[a]
    for a, b in POSE_LR_PAIRS:
        perm[a], perm[b] = perm[b], perm[a]
    return torch.tensor(perm, dtype=torch.long)


def _build_mirror_perm_for_subset(subset_name: str) -> torch.Tensor:
    """Subset-aware mirror permutation. Reads the L/R pair list from
    data.keypoint_subsets and applies the swaps within the subset's
    own index space (so [0,1] = left/right shoulder swap, etc.)."""
    from data.keypoint_subsets import get_subset
    subset = get_subset(subset_name)
    n = len(subset["indices"])
    perm = list(range(n))
    for a, b in subset["mirror_pairs"]:
        perm[a], perm[b] = perm[b], perm[a]
    return torch.tensor(perm, dtype=torch.long)


_MIRROR_PERM = _build_mirror_perm()


class LandmarkAugment:
    """Callable augmentation pipeline matching LandmarkDataset's `transform`
    contract: (lm, valid) -> (lm, valid).

    Sampling is per-clip — every __call__ draws fresh parameters that apply
    uniformly across all T frames in the clip.

    `keypoint_subset` selects the matching mirror permutation; defaults to the
    full K=75 layout for backward compatibility.
    """

    def __init__(
        self,
        cfg: LandmarkAugmentConfig | None = None,
        keypoint_subset: str = "all",
    ):
        self.cfg = cfg or LandmarkAugmentConfig()
        self.keypoint_subset = keypoint_subset
        self._mirror_perm: torch.Tensor = (
            _MIRROR_PERM if keypoint_subset == "all"
            else _build_mirror_perm_for_subset(keypoint_subset)
        )

    def __call__(
        self, lm: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        T, K, C = lm.shape
        assert C == 3, f"unexpected coord_dims {C}, expected 3"
        if K != self._mirror_perm.shape[0]:
            raise ValueError(
                f"input K={K} doesn't match mirror perm size "
                f"{self._mirror_perm.shape[0]} for subset {self.keypoint_subset!r}"
            )

        # 1. Mirror — flip x AND permute L/R slots.
        if cfg.mirror_p > 0 and random.random() < cfg.mirror_p:
            lm = lm.clone()
            lm[..., 0] = -lm[..., 0]
            lm = lm.index_select(dim=1, index=self._mirror_perm.to(lm.device))
            valid = valid.index_select(dim=1, index=self._mirror_perm.to(valid.device))

        # 2. Rotation around (0, 0) in (x, y); leave z raw.
        if cfg.rotate_max_deg > 0:
            angle = math.radians(random.uniform(-cfg.rotate_max_deg, cfg.rotate_max_deg))
            cos, sin = math.cos(angle), math.sin(angle)
            x = lm[..., 0]
            y = lm[..., 1]
            lm = lm.clone()
            lm[..., 0] = cos * x - sin * y
            lm[..., 1] = sin * x + cos * y

        # 3. Isotropic scale on (x, y).
        if cfg.scale_range and cfg.scale_range != (1.0, 1.0):
            s = random.uniform(*cfg.scale_range)
            lm = lm.clone() if not lm.is_contiguous() else lm
            lm = lm.clone()
            lm[..., 0] *= s
            lm[..., 1] *= s

        # 4. Translation on (x, y).
        if cfg.translate_max > 0:
            dx = random.uniform(-cfg.translate_max, cfg.translate_max)
            dy = random.uniform(-cfg.translate_max, cfg.translate_max)
            lm = lm.clone()
            lm[..., 0] += dx
            lm[..., 1] += dy

        # 5. Per-frame Gaussian noise on (x, y). Resampled every frame because
        # detection wobble is genuinely per-frame; this is the one exception
        # to the "same noise across T" rule.
        if cfg.coord_noise_std > 0:
            noise = torch.randn(T, K, 2, dtype=lm.dtype) * cfg.coord_noise_std
            lm = lm.clone()
            lm[..., :2] += noise

        # 6. Joint dropout — pick random keypoints and mark invalid for the
        # whole clip (simulates a hand the detector lost across the entire
        # sign). Per-clip rather than per-frame.
        if cfg.joint_dropout_p > 0:
            drop_mask = torch.rand(K) < cfg.joint_dropout_p
            if drop_mask.any():
                valid = valid.clone()
                valid[:, drop_mask] = False
                # Zero the dropped landmarks so the model can't accidentally
                # learn from invalid coordinates.
                lm = lm.clone()
                lm[:, drop_mask, :] = 0.0

        return lm, valid


def build_default_augment() -> LandmarkAugment:
    return LandmarkAugment(LandmarkAugmentConfig())
