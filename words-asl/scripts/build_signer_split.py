"""
build_signer_split.py — choose a held-out-signer test set.

WLASL's default `split` field mixes signers across train/test, so a model can
shortcut by recognising a signer's hands rather than the gloss. This script
selects a subset of signer IDs to reserve for a separate held-out-signer
evaluation, used by `eval.py` to produce the cross-signer generalisation
result that goes into the comparison tables (Phase E).

Selection algorithm:
  - Pool all signers across the prepared train/val/test manifests of a given
    subset.
  - Greedy pick: until we have reserved ~target_fraction of all clips, pick
    the signer that adds the fewest *new* glosses to the reserved set.
    Rationale: a held-out set with broad gloss coverage gives a fairer
    evaluation than one biased towards a narrow set of classes.
  - Output the chosen signer IDs and the gloss coverage they imply.

The result is written to `configs/signer_split.json` so the split is
reproducible (eval.py loads it). The actual data partitioning happens at
load time — we don't move clips between manifests.

Run:
    python scripts/build_signer_split.py --subset 100 --target-fraction 0.15
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_all_clips(prepared_dir: Path, subset: int) -> list[dict[str, Any]]:
    clips: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        path = prepared_dir / f"wlasl{subset}_{split}.json"
        if not path.exists():
            print(f"  warning: {path} not found")
            continue
        clips.extend(json.loads(path.read_text()))
    return clips


def select_signers(
    clips: list[dict[str, Any]],
    target_fraction: float,
    seed: int = 0,
) -> dict[str, Any]:
    by_signer: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for c in clips:
        s = c.get("signer_id")
        if s is None:
            continue
        by_signer[s].append(c)

    if not by_signer:
        raise SystemExit("No signer IDs in manifest. Cannot build held-out-signer split.")

    total_clips = sum(len(v) for v in by_signer.values())
    target_clips = int(round(total_clips * target_fraction))
    print(f"  total clips with signer IDs: {total_clips}")
    print(f"  target reserved clips:       {target_clips} ({target_fraction:.0%})")

    reserved: set[Any] = set()
    reserved_glosses: set[str] = set()
    reserved_clips_count = 0

    # Greedy: at each step, pick the signer that adds the fewest *new* glosses
    # (so we drift towards broad gloss coverage without grabbing a single
    # high-volume signer that monopolises one class).
    candidates = sorted(by_signer.keys(), key=lambda s: (len(by_signer[s]), s))
    while reserved_clips_count < target_clips and candidates:
        best = None
        best_overlap = -1
        for s in candidates:
            if s in reserved:
                continue
            new_glosses = {c["gloss"] for c in by_signer[s]} - reserved_glosses
            overlap = len({c["gloss"] for c in by_signer[s]}) - len(new_glosses)
            # Prefer high overlap (= adds few new glosses) so coverage stays broad
            # rather than concentrated in one signer's specialty.
            if overlap > best_overlap:
                best_overlap = overlap
                best = s
        if best is None:
            break
        reserved.add(best)
        reserved_glosses.update(c["gloss"] for c in by_signer[best])
        reserved_clips_count += len(by_signer[best])

    coverage = sorted(reserved_glosses)
    return {
        "seed": seed,
        "target_fraction": target_fraction,
        "reserved_signers": sorted(reserved, key=lambda x: str(x)),
        "reserved_clip_count": reserved_clips_count,
        "total_clips": total_clips,
        "actual_fraction": reserved_clips_count / max(total_clips, 1),
        "gloss_coverage": coverage,
        "gloss_coverage_count": len(coverage),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared-dir", type=Path, default=Path("wlasl/prepared"))
    parser.add_argument("--subset", type=int, default=100)
    parser.add_argument("--target-fraction", type=float, default=0.15, help="Fraction of total clips to reserve for held-out evaluation.")
    parser.add_argument("--out", type=Path, default=Path("configs/signer_split.json"))
    args = parser.parse_args()

    clips = load_all_clips(args.prepared_dir, args.subset)
    print(f"loaded {len(clips)} clips from {args.prepared_dir}")

    decision = select_signers(clips, args.target_fraction)
    decision["subset"] = args.subset
    decision["source_manifests"] = sorted(p.name for p in args.prepared_dir.glob(f"wlasl{args.subset}_*.json"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(decision, indent=2, default=str))

    print(f"\nreserved {len(decision['reserved_signers'])} signers covering "
          f"{decision['gloss_coverage_count']} glosses "
          f"({decision['reserved_clip_count']} / {decision['total_clips']} clips, "
          f"{decision['actual_fraction']:.1%} of total).")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
