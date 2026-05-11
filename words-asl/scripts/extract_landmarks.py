"""
extract_landmarks.py — offline MediaPipe Holistic extraction for Track 1.

Reads one or more prepared WLASL manifests (output of prepare_wlasl.py) and
writes per-clip landmark caches to disk. Track 1 trains on these caches; the
trainer never runs MediaPipe itself.

Cache schema (frozen contract — /data/landmark_dataset.py reads this
shape directly):
    landmarks  shape [F, 75, 3]   float32  (33 pose + 21 left + 21 right hand)
    valid      shape [F, 75]      bool

We do NOT include the 468-point face mesh by default. Word-level signs in
WLASL don't lean heavily on facial non-manual markers, and skipping the mesh
keeps K small (model fits ANE) and extraction ~2x faster. The per-clip index
file records `face=False`; if accuracy plateaus, re-extract with --include-face
and the dataset's num_keypoints config can grow without code changes.

Run:
    .venv/bin/python scripts/extract_landmarks.py \\
        --manifests wlasl/prepared/wlasl100_train.json \\
                    wlasl/prepared/wlasl100_val.json \\
                    wlasl/prepared/wlasl100_test.json \\
        --videos-root wlasl \\
        --out-dir /landmarks_cache \\
        --skip-existing \\
        --report-md docs/landmark-pipeline.md
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

# Constants tied to MediaPipe Holistic's published landmark schema.
# The dataset code at /data/landmark_dataset.py expects exactly these
# K=75 keypoints in this order: pose (33), left hand (21), right hand (21).
NUM_POSE = 33
NUM_HAND = 21
NUM_KEYPOINTS = NUM_POSE + 2 * NUM_HAND  # 75


def _load_manifests(paths: list[Path]) -> list[dict[str, Any]]:
    """Concatenate manifests; each entry keeps its `split` field implicitly via
    where it came from but we don't actually need it for extraction."""
    items: list[dict[str, Any]] = []
    for p in paths:
        if not p.exists():
            raise SystemExit(f"manifest not found: {p}")
        items.extend(json.loads(p.read_text()))
    return items


def _resolve_video_path(item: dict[str, Any], videos_root: Path) -> Path:
    # `video_path` in the manifest is relative to videos_root.parent
    # (the wlasl/ folder). Try that first; fall back to videos_root /
    # <video_id>.mp4 for older manifests.
    p = (videos_root.parent / item["video_path"]).resolve()
    if p.exists():
        return p
    fallback = videos_root / f"{item['video_id']}.mp4"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"video missing for {item['video_id']}: tried {p} and {fallback}")


def _slice_frames(frame_start: int | None, frame_end: int | None, total: int) -> tuple[int, int]:
    """Honor manifest frame_start/frame_end. WLASL uses 1-indexed inclusive
    starts; -1 means "to end of clip"."""
    s = max(0, (frame_start or 1) - 1)
    e = total if (frame_end is None or frame_end == -1) else min(total, frame_end)
    if e <= s:
        return 0, total
    return s, e


def _decode_frames(video_path: Path, start: int, end: int, max_frames: int | None) -> list[np.ndarray]:
    """Read frames [start, end) as RGB uint8 arrays via OpenCV.

    Apple Silicon CPU decode of WLASL clips runs ~150-300 fps, fast enough
    that we don't need decord here.
    """
    import cv2  # local import: opencv is a heavy dep, only needed when extracting

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    frames: list[np.ndarray] = []
    idx = 0
    target = end if max_frames is None else min(end, start + max_frames)
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx >= target:
            break
        if idx >= start:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
        idx += 1
    cap.release()
    return frames


def _build_holistic(model_complexity: int):
    """Construct a MediaPipe Holistic solution. Imported lazily because the
    mediapipe wheel itself is several hundred MB."""
    import mediapipe as mp

    return mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=model_complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _extract_clip_landmarks(holistic, rgb_frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Returns (landmarks [F, 75, 3] float32, valid [F, 75] bool)."""
    F = len(rgb_frames)
    landmarks = np.zeros((F, NUM_KEYPOINTS, 3), dtype=np.float32)
    valid = np.zeros((F, NUM_KEYPOINTS), dtype=bool)

    for f, rgb in enumerate(rgb_frames):
        rgb.flags.writeable = False  # mediapipe wants this
        result = holistic.process(rgb)

        if result.pose_landmarks is not None:
            for k, lm in enumerate(result.pose_landmarks.landmark[:NUM_POSE]):
                landmarks[f, k] = (lm.x, lm.y, lm.z)
                valid[f, k] = True

        if result.left_hand_landmarks is not None:
            for k, lm in enumerate(result.left_hand_landmarks.landmark[:NUM_HAND]):
                landmarks[f, NUM_POSE + k] = (lm.x, lm.y, lm.z)
                valid[f, NUM_POSE + k] = True

        if result.right_hand_landmarks is not None:
            for k, lm in enumerate(result.right_hand_landmarks.landmark[:NUM_HAND]):
                landmarks[f, NUM_POSE + NUM_HAND + k] = (lm.x, lm.y, lm.z)
                valid[f, NUM_POSE + NUM_HAND + k] = True

    return landmarks, valid


def _frame_has_hand(valid: np.ndarray) -> np.ndarray:
    """Per-frame flag: at least one hand keypoint detected. Useful audit
    metric since a clip with no hand detections at all is unrecoverable for
    Track 1."""
    return valid[:, NUM_POSE:].any(axis=1)


def _write_index(out_dir: Path, entries: list[dict[str, Any]], extras: dict[str, Any]) -> None:
    """One _index.json per cache directory. Cheaper than opening 600 npz files."""
    payload = {"entries": entries, **extras}
    (out_dir / "_index.json").write_text(json.dumps(payload, indent=2))


def _write_report(report_path: Path, entries: list[dict[str, Any]], extras: dict[str, Any]) -> None:
    """Owner-facing summary the plan asks for."""
    n = len(entries)
    if n == 0:
        report_path.write_text("# Landmark pipeline\n\nNo clips processed.\n")
        return

    avg_runtime = sum(e["runtime_s"] for e in entries) / n
    avg_valid_frame_pct = sum(e["hand_frame_pct"] for e in entries) / n
    no_hand = [e for e in entries if e["hand_frame_pct"] < 0.5]
    avg_frames = sum(e["frames"] for e in entries) / n
    total_frames = sum(e["frames"] for e in entries)

    lines = [
        "# Landmark pipeline",
        "",
        "Auto-generated by `scripts/extract_landmarks.py`. Re-run the",
        "script to refresh.",
        "",
        f"- Clips processed:           **{n}**",
        f"- Total frames decoded:      **{total_frames:,}**",
        f"- Avg frames per clip:       **{avg_frames:.1f}**",
        f"- Avg per-clip runtime:      **{avg_runtime:.2f}s**",
        f"- Mean hand-detected-frame%: **{avg_valid_frame_pct*100:.1f}%**",
        f"- Clips with <50% hand frames: **{len(no_hand)}** "
        + (
            "(may be unrecoverable for Track 1 — re-record or drop)"
            if no_hand else ""
        ),
        "",
        "## Pipeline configuration",
        "",
        f"- MediaPipe version:         `{extras.get('mediapipe_version', '?')}`",
        f"- model_complexity:          `{extras.get('model_complexity')}`",
        f"- include face mesh:         `{extras.get('include_face')}`",
        f"- min_detection_confidence:  `0.5`",
        f"- min_tracking_confidence:   `0.5`",
        f"- Landmark schema:           `[F, {NUM_KEYPOINTS}, 3]` float32 + `[F, {NUM_KEYPOINTS}]` bool",
        f"- Keypoint layout:           33 pose + 21 left hand + 21 right hand",
        f"- Platform:                  `{extras.get('platform', '?')}`",
        "",
        "## Cache layout",
        "",
        "```",
        "landmarks_cache/",
        "  <video_id>.npz       # {landmarks: [F,75,3] f32, valid: [F,75] bool}",
        "  _index.json          # this report's underlying data",
        "```",
        "",
        "Cache is gitignored. To regenerate from scratch, delete the directory",
        "and re-run the extractor. To resume an interrupted run, re-invoke",
        "with `--skip-existing`.",
        "",
    ]
    report_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifests", nargs="+", type=Path, required=True)
    parser.add_argument("--videos-root", type=Path, default=Path("wlasl/videos"))
    parser.add_argument("--out-dir", type=Path, default=Path("landmarks_cache"))
    parser.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test cap on clip count.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Per-clip frame cap. Defaults to all decoded frames.")
    parser.add_argument("--fail-on-missing", action="store_true",
                        help="Exit non-zero if any video file is missing (for CI).")
    parser.add_argument(
        "--report-md", type=Path, default=Path("docs/landmark-pipeline.md"),
        help="Markdown summary written after extraction completes."
    )
    parser.add_argument(
        "--include-face", action="store_true",
        help="Also store the 468 face-mesh keypoints. Default off; see module docstring."
    )
    args = parser.parse_args()

    if args.include_face:
        # Cache schema would change; keep the door open but explicitly out of
        # scope until the dataset class supports it.
        raise SystemExit("--include-face is reserved; LandmarkDataset uses K=75. Wire the dataset first.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    items = _load_manifests(args.manifests)
    if args.limit:
        items = items[: args.limit]

    # De-dup by video_id — train/val/test manifests can share none, but the
    # extractor is invoked across all three so guarding is cheap.
    seen: set[str] = set()
    unique_items: list[dict[str, Any]] = []
    for it in items:
        vid = str(it["video_id"])
        if vid in seen:
            continue
        seen.add(vid)
        unique_items.append(it)
    print(f"[extract] {len(unique_items)} unique clips to process "
          f"(after dedup of {len(items)} manifest rows)")

    import mediapipe as mp  # imported here so --help works without the dep
    holistic = _build_holistic(args.model_complexity)

    entries: list[dict[str, Any]] = []
    skipped = 0
    missing_video = 0
    decode_failed = 0
    counts = Counter()

    try:
        for i, item in enumerate(unique_items):
            video_id = str(item["video_id"])
            cache_path = args.out_dir / f"{video_id}.npz"
            if args.skip_existing and cache_path.exists():
                skipped += 1
                continue

            try:
                video_path = _resolve_video_path(item, args.videos_root)
            except FileNotFoundError as e:
                missing_video += 1
                if args.fail_on_missing:
                    raise SystemExit(str(e))
                continue

            # Read raw video to get total frame count, then slice per manifest.
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if total <= 0:
                decode_failed += 1
                continue
            s, e = _slice_frames(item.get("frame_start"), item.get("frame_end"), total)

            t0 = time.time()
            rgb_frames = _decode_frames(video_path, s, e, args.max_frames)
            if len(rgb_frames) == 0:
                decode_failed += 1
                continue

            landmarks, valid = _extract_clip_landmarks(holistic, rgb_frames)
            np.savez_compressed(cache_path, landmarks=landmarks, valid=valid)
            elapsed = time.time() - t0

            hand_frame_pct = float(_frame_has_hand(valid).mean())
            counts["written"] += 1
            entries.append({
                "video_id": video_id,
                "frames": int(landmarks.shape[0]),
                "hand_frame_pct": hand_frame_pct,
                "runtime_s": elapsed,
                "model_complexity": args.model_complexity,
            })
            if (i + 1) % 25 == 0 or i == len(unique_items) - 1:
                print(f"[extract] {i+1}/{len(unique_items)} written={counts['written']} "
                      f"skipped={skipped} missing={missing_video} decode_failed={decode_failed}")
    finally:
        holistic.close()

    extras = {
        "mediapipe_version": mp.__version__,
        "model_complexity": args.model_complexity,
        "include_face": False,
        "platform": platform.platform(),
        "skipped_existing": skipped,
        "missing_video": missing_video,
        "decode_failed": decode_failed,
    }
    _write_index(args.out_dir, entries, extras)

    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    _write_report(args.report_md, entries, extras)

    print(f"[extract] done. "
          f"written={counts['written']} skipped={skipped} "
          f"missing_video={missing_video} decode_failed={decode_failed}")
    print(f"[extract] cache  -> {args.out_dir}")
    print(f"[extract] report -> {args.report_md}")
    if missing_video and args.fail_on_missing:
        sys.exit(2)


if __name__ == "__main__":
    main()
