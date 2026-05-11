"""
prepare_wlasl.py — turn the raw WLASL annotation file into train/val/test
JSON manifests that downstream code can consume without ever touching the
original schema.

The WLASL_v0.3.json file is a list of glosses, each containing a list of
instances. We:
  1. Sort glosses alphabetically and take the first N for the requested
     subset (WLASL-100 / WLASL-300). The original WLASL paper defines the
     subsets the same way; sorting alphabetically is reproducible without an
     extra dependency on the upstream `nslt_<N>.json` split file.
  2. Keep only instances whose video file exists on disk (YouTube takedowns
     mean 20-40% are dead — see docs/datasets.md).
  3. Group surviving instances by their per-instance `split` field
     (train/val/test) and write three manifest JSONs.
  4. Warn if any class drops below MIN_PER_CLASS — that class is unusable.

Output layout (under --videos-root):
    prepared/
      wlasl<N>_train.json
      wlasl<N>_val.json
      wlasl<N>_test.json
      wlasl<N>_classes.json    # gloss -> integer label
      wlasl<N>_drops.json      # per-gloss missing/dropped counts

Each manifest entry looks like:
    {
      "video_id": "12345",
      "video_path": "videos/12345.mp4",
      "gloss": "book",
      "label": 7,
      "signer_id": 42,
      "frame_start": 1, "frame_end": 100,
      "fps": 25,
      "bbox": [x1, y1, x2, y2]   // null if missing
    }

Run:
    python scripts/prepare_wlasl.py --subset 100
    python scripts/prepare_wlasl.py --subset 300 --videos-root wlasl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# A class needs at least this many surviving samples per split to be usable.
# Below this, train/val/test become statistically meaningless. The script
# warns rather than auto-dropping the class — owner decides whether to retry
# the YouTube downloader or accept the loss.
MIN_PER_CLASS_PER_SPLIT = 1


def load_wlasl(annotation_path: Path) -> list[dict[str, Any]]:
    if not annotation_path.exists():
        raise SystemExit(f"WLASL annotation file not found: {annotation_path}")
    return json.loads(annotation_path.read_text())


def select_subset(entries: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    sorted_entries = sorted(entries, key=lambda e: e["gloss"])
    return sorted_entries[:n]


def scan(
    entries: list[dict[str, Any]],
    videos_dir: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, int]]]:
    """
    Returns (manifests_by_split, drop_counts_by_gloss).

    manifests_by_split: {'train': [...], 'val': [...], 'test': [...]}
    drop_counts_by_gloss: {'book': {'requested': N, 'missing': M, 'kept': K}}
    """
    manifests: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    drops: dict[str, dict[str, int]] = {}

    for label_idx, entry in enumerate(entries):
        gloss = entry["gloss"]
        requested = 0
        missing = 0
        kept = 0
        for inst in entry.get("instances", []):
            requested += 1
            video_id = inst.get("video_id")
            if not video_id:
                missing += 1
                continue
            video_path = videos_dir / f"{video_id}.mp4"
            if not video_path.exists():
                missing += 1
                continue
            split = inst.get("split", "train").lower()
            if split not in manifests:
                # Unexpected split name — coerce to train and let stats
                # surface the surprise.
                split = "train"
            manifests[split].append({
                "video_id": video_id,
                "video_path": str(video_path.relative_to(videos_dir.parent)) if video_path.is_relative_to(videos_dir.parent) else str(video_path),
                "gloss": gloss,
                "label": label_idx,
                "signer_id": inst.get("signer_id"),
                "frame_start": inst.get("frame_start", 1),
                "frame_end": inst.get("frame_end", -1),
                "fps": inst.get("fps", 25),
                "bbox": inst.get("bbox"),
            })
            kept += 1
        drops[gloss] = {"requested": requested, "missing": missing, "kept": kept}
    return manifests, drops


def warn_thin_classes(manifests: dict[str, list[dict[str, Any]]]) -> None:
    counts = {split: Counter(m["gloss"] for m in items) for split, items in manifests.items()}
    all_classes = set().union(*[set(c) for c in counts.values()])
    thin = []
    for gloss in sorted(all_classes):
        for split in ("train", "val", "test"):
            if counts[split].get(gloss, 0) < MIN_PER_CLASS_PER_SPLIT:
                thin.append((gloss, split, counts[split].get(gloss, 0)))
    if thin:
        print("\nWARNING: classes below MIN_PER_CLASS_PER_SPLIT in some split:")
        for gloss, split, n in thin[:20]:
            print(f"  {gloss:<24} split={split:<5} n={n}")
        if len(thin) > 20:
            print(f"  ... {len(thin) - 20} more")
        print(
            "  These classes will not have a usable train/val/test triplet. Re-run\n"
            "  the YouTube downloader, or accept the loss and document it in\n"
            "  docs/datasets.md.\n"
        )


def write_outputs(
    out_dir: Path,
    subset: int,
    manifests: dict[str, list[dict[str, Any]]],
    classes: list[str],
    drops: dict[str, dict[str, int]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, items in manifests.items():
        path = out_dir / f"wlasl{subset}_{split}.json"
        path.write_text(json.dumps(items, indent=2))
        print(f"  wrote {len(items):>5} entries -> {path}")
    classes_path = out_dir / f"wlasl{subset}_classes.json"
    classes_path.write_text(json.dumps({"gloss_to_label": {g: i for i, g in enumerate(classes)},
                                        "label_to_gloss": classes}, indent=2))
    print(f"  wrote class map         -> {classes_path}")
    drops_path = out_dir / f"wlasl{subset}_drops.json"
    drops_path.write_text(json.dumps(drops, indent=2))
    print(f"  wrote drop counts       -> {drops_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotation", type=Path, default=Path("wlasl/WLASL_v0.3.json"))
    parser.add_argument("--videos-dir", type=Path, default=Path("wlasl/videos"))
    parser.add_argument("--out-dir", type=Path, default=Path("wlasl/prepared"))
    parser.add_argument("--subset", type=int, default=100, choices=[100, 300, 1000, 2000], help="Number of glosses (alphabetical).")
    args = parser.parse_args()

    print(f"loading {args.annotation} ...")
    entries = load_wlasl(args.annotation)
    print(f"  {len(entries)} total glosses")

    selected = select_subset(entries, args.subset)
    print(f"selected first {len(selected)} glosses alphabetically (subset = WLASL-{args.subset})")

    print(f"scanning {args.videos_dir} for clip availability ...")
    manifests, drops = scan(selected, args.videos_dir)
    total_kept = sum(len(v) for v in manifests.values())
    total_requested = sum(d["requested"] for d in drops.values())
    yield_pct = 100.0 * total_kept / max(total_requested, 1)
    print(f"  kept {total_kept}/{total_requested} clips ({yield_pct:.1f}% yield)")

    write_outputs(args.out_dir, args.subset, manifests, [e["gloss"] for e in selected], drops)
    warn_thin_classes(manifests)


if __name__ == "__main__":
    main()
