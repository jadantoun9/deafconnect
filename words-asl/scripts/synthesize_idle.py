"""
synthesize_idle.py — generate a synthetic "idle" class for the Track-1 model.

Without this class, the 8-way softmax always sums to 1.0 — so when the user
isn't signing the model is forced to confidently pick one of the 8 known
glosses (the "always predicts go/computer when idle" failure mode).

Approach:
  - For each training clip, take the **first detected frame** of the cached
    landmark sequence (the signer's resting pose before the sign begins) and
    replicate it across T=32 frames. The resulting clip is a "frozen still"
    — exactly what we want the idle class to mean.
  - Save under /landmarks_cache/idle_<source_id>.npz so the existing
    LandmarkDataset reads it identically.
  - Generate a balanced number of idle samples per split (matching real-class
    counts) so class weighting does the rest.

Augmentation at training time still varies these per-epoch (rotation, mirror,
noise), so the model doesn't memorise one specific pose — it learns "no
inter-frame motion" as the idle signature.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np


def synthesize_idle_cache(source_video_id: str, cache_dir: Path,
                           num_frames: int = 32) -> str | None:
    """Return the new video_id of the synthesized idle clip, or None if the
    source had no detected frames."""
    src = cache_dir / f"{source_video_id}.npz"
    if not src.exists():
        return None
    z = np.load(src)
    lm = z["landmarks"]
    valid = z["valid"]

    # Pick the first frame with at least one valid keypoint.
    has_any = valid.any(axis=1)
    if not has_any.any():
        return None
    first = int(np.argmax(has_any))
    src_lm = lm[first]            # [K, 3]
    src_valid = valid[first]      # [K]

    out_lm = np.broadcast_to(src_lm[None, ...], (num_frames, *src_lm.shape)).copy().astype(np.float32)
    out_valid = np.broadcast_to(src_valid[None, ...], (num_frames, *src_valid.shape)).copy()

    new_id = f"idle_{source_video_id}"
    np.savez_compressed(cache_dir / f"{new_id}.npz", landmarks=out_lm, valid=out_valid)
    return new_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-prefix", default="wlasl9",
                        help="Read wlasl9_{train,val,test}.json and wlasl9_classes.json.")
    parser.add_argument("--out-prefix", default="wlasl9idle",
                        help="Write {prefix}_{train,val,test}.json + classes.json.")
    parser.add_argument("--cache-dir", type=Path, default=Path("landmarks_cache"))
    parser.add_argument("--prepared-dir", type=Path, default=Path("wlasl/prepared"))
    parser.add_argument("--n-train", type=int, default=12)
    parser.add_argument("--n-val", type=int, default=3)
    parser.add_argument("--n-test", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=32)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    classes = json.loads((args.prepared_dir / f"{args.source_prefix}_classes.json").read_text())
    glosses = list(classes["label_to_gloss"]) + ["idle"]
    gloss_to_label = {g: i for i, g in enumerate(glosses)}
    idle_label = gloss_to_label["idle"]

    counts_per_split = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    summary = {}

    for split, n_idle in counts_per_split.items():
        manifest_path = args.prepared_dir / f"{args.source_prefix}_{split}.json"
        items = json.loads(manifest_path.read_text())
        # Sample n_idle source clips to derive idle samples from. If n_idle
        # >= len(items), use all of them.
        sources = list(items)
        rng.shuffle(sources)
        chosen = sources[:n_idle]

        new_items = list(items)
        synthesised = 0
        for it in chosen:
            new_id = synthesize_idle_cache(it["video_id"], args.cache_dir,
                                            num_frames=args.num_frames)
            if new_id is None:
                continue
            new_items.append({
                "video_id": new_id,
                "video_path": it.get("video_path", ""),
                "gloss": "idle",
                "label": idle_label,
                "signer_id": it.get("signer_id"),
                "split": split,
                "fps": it.get("fps"),
                "frame_start": 1,
                "frame_end": -1,
                "source": "synthesized_idle",
            })
            synthesised += 1

        out_path = args.prepared_dir / f"{args.out_prefix}_{split}.json"
        out_path.write_text(json.dumps(new_items, indent=2))
        summary[split] = (len(items), synthesised, len(new_items))
        print(f"  {split}: {len(items)} real + {synthesised} idle = {len(new_items)} total -> {out_path.name}")

    classes_path = args.prepared_dir / f"{args.out_prefix}_classes.json"
    classes_path.write_text(json.dumps({
        "gloss_to_label": gloss_to_label,
        "label_to_gloss": glosses,
    }, indent=2))
    print(f"  classes: {len(glosses)} -> {classes_path.name}")
    print(f"  summary: {summary}")


if __name__ == "__main__":
    main()
