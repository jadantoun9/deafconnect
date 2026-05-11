"""
dataset_stats.py — descriptive stats over the prepared WLASL manifests.

Reads the train/val/test JSONs produced by prepare_wlasl.py and computes:
  - per-class clip counts (per split)
  - signer distribution (per split)
  - frame-count distribution
  - resolution distribution
  - duration distribution

Output: a single `stats.json` in the same prepared/ directory plus a printed
summary. The JSON is consumed by docs/datasets.md auto-fill (manual paste
for now) and by build_signer_split.py.

Frames + resolution come from decord. If decord can't open a clip, that clip
is counted under `unreadable` and the script continues — corrupt clips happen
with YouTube downloads.

Run:
    python scripts/dataset_stats.py --subset 100
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

# Reading every clip is expensive on a slow disk. Default to a sample so a
# full WLASL-300 stats run finishes in minutes; the owner can bump this for
# the final report.
DEFAULT_SAMPLE = 64


def safe_video_meta(path: Path) -> dict[str, Any] | None:
    """Open with decord, return frame count + (h, w). None on failure."""
    try:
        import decord
    except ImportError:
        return None
    try:
        vr = decord.VideoReader(str(path))
        n = len(vr)
        h, w, _ = vr[0].shape
        return {"frames": n, "height": int(h), "width": int(w), "fps": float(vr.get_avg_fps())}
    except Exception:
        return None


def summarize_split(
    name: str,
    items: list[dict[str, Any]],
    videos_root: Path,
    sample: int,
) -> dict[str, Any]:
    classes = Counter(it["gloss"] for it in items)
    signers = Counter(it.get("signer_id") for it in items)

    # Sample for the expensive metadata pass.
    import random
    rng = random.Random(0)
    sampled = rng.sample(items, k=min(sample, len(items))) if items else []

    frames: list[int] = []
    resolutions: list[tuple[int, int]] = []
    fpss: list[float] = []
    unreadable = 0
    for it in sampled:
        # video_path in the manifest is relative to videos_root.parent (the
        # `wlasl/` folder); resolve from there.
        candidate = (videos_root.parent / it["video_path"]).resolve()
        if not candidate.exists():
            candidate = videos_root / Path(it["video_path"]).name
        meta = safe_video_meta(candidate)
        if meta is None:
            unreadable += 1
            continue
        frames.append(meta["frames"])
        resolutions.append((meta["height"], meta["width"]))
        fpss.append(meta["fps"])

    def describe_int(values: list[int]) -> dict[str, Any]:
        if not values:
            return {"count": 0}
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        }

    summary = {
        "name": name,
        "n_clips": len(items),
        "n_classes": len(classes),
        "n_signers": len(signers),
        "class_balance": dict(classes.most_common()),
        "signer_distribution": {str(k): v for k, v in signers.most_common()},
        "sampled_for_metadata": len(sampled),
        "unreadable": unreadable,
        "frames": describe_int(frames),
        "fps_mean": statistics.mean(fpss) if fpss else None,
        "resolution_modes": Counter(resolutions).most_common(5),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared-dir", type=Path, default=Path("wlasl/prepared"))
    parser.add_argument("--videos-dir", type=Path, default=Path("wlasl/videos"))
    parser.add_argument("--subset", type=int, default=100)
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE, help="Per-split sample for the metadata pass. Use -1 to read every clip.")
    args = parser.parse_args()

    sample = 10**9 if args.sample == -1 else args.sample
    out: dict[str, Any] = {"subset": args.subset, "splits": {}}
    for split in ("train", "val", "test"):
        manifest_path = args.prepared_dir / f"wlasl{args.subset}_{split}.json"
        if not manifest_path.exists():
            print(f"skipping {split}: {manifest_path} not found")
            continue
        items = json.loads(manifest_path.read_text())
        print(f"summarising {split} ({len(items)} clips, sampling {min(sample, len(items))} for video metadata) ...")
        out["splits"][split] = summarize_split(split, items, args.videos_dir, sample)

    # Quick console summary.
    for split, summary in out["splits"].items():
        f = summary.get("frames", {})
        print(f"\n[{split}] clips={summary['n_clips']} classes={summary['n_classes']} signers={summary['n_signers']}")
        if f.get("count"):
            print(f"  frames median={f['median']:.0f} (min={f['min']}, max={f['max']}, mean={f['mean']:.1f})")
        print(f"  top resolutions: {summary['resolution_modes']}")

    out_path = args.prepared_dir / f"wlasl{args.subset}_stats.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
