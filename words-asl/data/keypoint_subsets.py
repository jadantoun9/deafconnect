"""
keypoint_subsets.py — keypoint subset definitions for Track 1 experiments.

The cached MediaPipe Holistic landmark files store K=75 keypoints
(33 pose + 21 left hand + 21 right hand). For some experiments we don't
want all of them — e.g. v2 of Track 1 prunes the head, hips, lower body, and
pose-level hand landmarks (which are redundant with the full hand mesh).

Each subset specifies:
    indices       — selection from the K=75 cache, in the desired output order
    mirror_pairs  — (i, j) pairs to swap when horizontally mirroring, expressed
                    in the **subset's** index space (not the original 75)

The dataset slices its caches by these indices; the augmentation module reads
the mirror_pairs to build a permutation tensor that flips the L/R anatomy
correctly after the x coordinate is flipped.
"""

from __future__ import annotations

from typing import TypedDict

NUM_POSE = 33
NUM_HAND = 21
NUM_KEYPOINTS_FULL = NUM_POSE + 2 * NUM_HAND  # 75


class _Subset(TypedDict):
    indices: list[int]
    mirror_pairs: list[tuple[int, int]]


def _full_pose_lr_pairs() -> list[tuple[int, int]]:
    return [
        (1, 4), (2, 5), (3, 6),
        (7, 8),
        (9, 10),
        (11, 12),
        (13, 14),
        (15, 16),
        (17, 18), (19, 20), (21, 22),
        (23, 24),
        (25, 26), (27, 28),
        (29, 30), (31, 32),
    ]


def _full_subset() -> _Subset:
    """All 75 keypoints — same as before pruning was introduced."""
    pairs = list(_full_pose_lr_pairs())
    # Hand block swap: pose-end..pose-end+21 <-> pose-end+21..end
    pairs.extend((NUM_POSE + i, NUM_POSE + NUM_HAND + i) for i in range(NUM_HAND))
    return {"indices": list(range(NUM_KEYPOINTS_FULL)), "mirror_pairs": pairs}


def _upper_body_hands_subset() -> _Subset:
    """v2 — shoulders/elbows/wrists + both full hand meshes only.

    Output order:
       [0] L shoulder   [1] R shoulder
       [2] L elbow      [3] R elbow
       [4] L wrist      [5] R wrist
       [6..26]   left hand  (21 keypoints)
       [27..47]  right hand (21 keypoints)
    Total: 48.
    """
    pose_keep = [11, 12, 13, 14, 15, 16]
    left_hand = list(range(NUM_POSE, NUM_POSE + NUM_HAND))
    right_hand = list(range(NUM_POSE + NUM_HAND, NUM_KEYPOINTS_FULL))
    indices = pose_keep + left_hand + right_hand

    pairs: list[tuple[int, int]] = [(0, 1), (2, 3), (4, 5)]
    pairs.extend((6 + i, 27 + i) for i in range(NUM_HAND))
    return {"indices": indices, "mirror_pairs": pairs}


SUBSETS: dict[str, _Subset] = {
    "all": _full_subset(),
    "upper_body_hands": _upper_body_hands_subset(),
}


def get_subset(name: str) -> _Subset:
    if name not in SUBSETS:
        raise ValueError(
            f"unknown keypoint subset {name!r}. Known: {list(SUBSETS)}"
        )
    return SUBSETS[name]
