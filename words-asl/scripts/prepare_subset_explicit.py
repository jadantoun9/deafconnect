"""
prepare_subset_explicit.py — build a WLASL subset from an explicit gloss list.

Unlike prepare_wlasl.py (which sorts alphabetically and takes the first N — a
known bug for our purposes; see [wlasl/prepared/wlasl100_classes.json]
where every class starts with 'a'), this script lets us hand-pick glosses by
name. It also drops any instance whose video file isn't on disk, which is
common for WLASL since many YouTube links have gone dead.

Output: same schema as prepare_wlasl.py — `<name>_train.json`, `_val.json`,
`_test.json`, `_classes.json` — so the existing trainer + dataset code reads
it without changes.

Run:
    .venv/bin/python scripts/prepare_subset_explicit.py \\
        --name wlasl8 \\
        --glosses drink go help who yes no before computer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def find_local_video(video_id: str, video_dirs: list[Path]) -> Path | None:
    for d in video_dirs:
        for ext in (".mp4", ".mkv", ".webm"):
            p = d / f"{video_id}{ext}"
            if p.exists():
                return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True,
                        help="Output prefix (e.g. wlasl8 -> wlasl8_train.json).")
    parser.add_argument("--glosses", nargs="+", required=True,
                        help="Explicit list of glosses to include, in order.")
    parser.add_argument("--master", type=Path,
                        default=Path("archive (6)/WLASL_v0.3.json"))
    parser.add_argument("--video-dirs", nargs="+", type=Path,
                        default=[Path("wlasl/videos"),
                                 Path("archive (6)/videos")])
    parser.add_argument("--out-dir", type=Path,
                        default=Path("wlasl/prepared"))
    args = parser.parse_args()

    master = json.loads(args.master.read_text())
    by_gloss = {e["gloss"]: e for e in master}

    classes: list[str] = []
    gloss_to_label: dict[str, int] = {}
    by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    drops: dict[str, dict[str, int]] = {}

    for label, gloss in enumerate(args.glosses):
        if gloss not in by_gloss:
            raise SystemExit(f"gloss not in master: {gloss!r}")
        classes.append(gloss)
        gloss_to_label[gloss] = label

        instances = by_gloss[gloss]["instances"]
        kept = missing = 0
        for inst in instances:
            vid = str(inst["video_id"])
            video_path = find_local_video(vid, args.video_dirs)
            if video_path is None:
                missing += 1
                continue
            split = inst.get("split", "train")
            if split not in by_split:
                continue
            # Match the existing convention from prepare_wlasl.py:
            # video_path is relative to `wlasl/` (i.e. videos_root.parent),
            # so the trainer/extractor can resolve it the same way.
            try:
                rel_path = video_path.relative_to(Path("wlasl").resolve())
                stored_path = str(Path("videos") / rel_path.name) if rel_path.parts and rel_path.parts[0] == "videos" else f"videos/{vid}{video_path.suffix}"
            except ValueError:
                # Falls back to bare videos/<id>.<ext> — extract_landmarks's
                # fallback path will resolve via videos_root / <video_id>.mp4.
                stored_path = f"videos/{vid}{video_path.suffix}"
            by_split[split].append({
                "video_id": vid,
                "video_path": stored_path,
                "gloss": gloss,
                "label": label,
                "signer_id": inst.get("signer_id"),
                "split": split,
                "fps": inst.get("fps"),
                "frame_start": inst.get("frame_start"),
                "frame_end": inst.get("frame_end"),
                "source": inst.get("source"),
            })
            kept += 1
        drops[gloss] = {"requested": len(instances), "missing": missing, "kept": kept}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        path = args.out_dir / f"{args.name}_{split}.json"
        path.write_text(json.dumps(by_split[split], indent=2))
        print(f"  wrote {path} ({len(by_split[split])} clips)")

    classes_path = args.out_dir / f"{args.name}_classes.json"
    classes_path.write_text(json.dumps({
        "gloss_to_label": gloss_to_label,
        "label_to_gloss": classes,
    }, indent=2))
    print(f"  wrote {classes_path} ({len(classes)} classes)")

    drops_path = args.out_dir / f"{args.name}_drops.json"
    drops_path.write_text(json.dumps(drops, indent=2))
    print(f"  wrote {drops_path}")

    total_kept = sum(d["kept"] for d in drops.values())
    total_req = sum(d["requested"] for d in drops.values())
    print()
    print(f"Total: {total_kept}/{total_req} clips kept "
          f"({100*total_kept/max(total_req,1):.0f}%)")


if __name__ == "__main__":
    main()
