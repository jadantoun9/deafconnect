"""
eval.py — single entry point for all model evaluation.

Loads a checkpoint, runs the configured eval split through the model, and
writes a Markdown results block + a machine-readable JSON.

Reports:
  - top-1 / top-5 / per-class accuracy
  - parameter count
  - FLOPs (via fvcore on a single representative input)
  - the dataset name + the held-out-signer mode used (so the output is
    self-describing in the comparison tables)

Two ways to invoke:
  python eval.py --config configs/track1_landmark.yaml --checkpoint .../best.pt
  python eval.py --config ... --checkpoint ... --corruption blur --severity 3
  python eval.py --config ... --checkpoint ... --held-out-signers configs/signer_split.json

The corruption flag wraps the dataset's frame transform with a corruption
from data.robustness_corruptions. The held-out flag flips the
dataset to `reserve` mode so we evaluate only on signers in the split file.

Output:
  --out-md      docs/<track>-results.md   (overwritten with new run)
  --out-json    docs/<track>-results.json (machine readable)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from utils import checkpoint as ckpt_utils
from utils.registry import build_dataset, build_model


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _wrap_with_corruption(dataset_cfg: dict[str, Any], name: str, severity: int) -> dict[str, Any]:
    # We can't modify the dataset transform after construction because the
    # registry returns a built instance. Instead, we tell the dataset to use
    # an `eval_args.transform` callable that the caller composes. Track-level
    # configs already expose `eval_args.transform_chain` which is a list of
    # named transforms; we append the corruption.
    cfg = json.loads(json.dumps(dataset_cfg))  # deep copy via JSON
    eval_args = cfg.setdefault("eval_args", {})
    chain = eval_args.setdefault("corruption_chain", [])
    chain.append({"name": name, "severity": severity})
    return cfg


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    correct1 = correct5 = total = 0
    per_class_correct: dict[int, int] = defaultdict(int)
    per_class_total: dict[int, int] = defaultdict(int)
    loss_sum = 0.0

    for batch in loader:
        if len(batch) == 4:
            x, labels, _, _ = batch
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(x)
        else:
            lm, valid, labels, _, _ = batch
            lm = lm.to(device); valid = valid.to(device); labels = labels.to(device)
            try:
                logits = model(lm, valid_mask=valid)
            except TypeError:
                logits = model(lm)

        loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        _, top5 = logits.topk(min(5, logits.size(-1)), dim=-1)
        correct1 += (top5[:, 0] == labels).sum().item()
        correct5 += (top5 == labels.unsqueeze(-1)).any(dim=-1).sum().item()
        total += labels.numel()
        for pred, t in zip(top5[:, 0].tolist(), labels.tolist()):
            per_class_total[t] += 1
            if pred == t:
                per_class_correct[t] += 1

    per_class = {
        cls: per_class_correct[cls] / max(per_class_total[cls], 1)
        for cls in per_class_total
    }
    return {
        "loss": loss_sum / max(total, 1),
        "top1": correct1 / max(total, 1),
        "top5": correct5 / max(total, 1),
        "n_samples": total,
        "per_class": per_class,
    }


def count_flops(model: torch.nn.Module, sample_input: torch.Tensor) -> dict[str, Any]:
    """fvcore-based FLOP / param counts. Returns gracefully on import failure."""
    n_params = sum(p.numel() for p in model.parameters())
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, sample_input).total()
    except ImportError:
        flops = None
    return {"params": n_params, "flops": flops}


def write_results(
    results: dict[str, Any],
    out_md: Path,
    out_json: Path,
    track: str,
    cfg: dict[str, Any],
) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))

    md = [f"# {track} — eval results\n"]
    md.append(f"- dataset: {cfg.get('dataset_name')}")
    md.append(f"- model:   {cfg.get('model', {}).get('name')}")
    md.append(f"- samples: {results['top_level']['n_samples']}")
    md.append("")
    md.append("| metric | value |")
    md.append("|--------|-------|")
    for k in ("top1", "top5", "loss"):
        md.append(f"| {k} | {results['top_level'][k]:.4f} |")
    if results.get("flops") is not None:
        md.append(f"| FLOPs | {results['flops']:,} |")
    md.append(f"| params | {results['params']:,} |")
    if results.get("variants"):
        md.append("\n## Robustness variants")
        md.append("| variant | top1 | top5 |")
        md.append("|---------|------|------|")
        for name, m in results["variants"].items():
            md.append(f"| {name} | {m['top1']:.4f} | {m['top5']:.4f} |")
    out_md.write_text("\n".join(md) + "\n")
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")


def build_eval_loader(
    cfg: dict[str, Any],
    held_out_signers: set[Any] | None,
    held_out_mode: str | None,
    corruption: str | None,
    severity: int,
) -> DataLoader:
    ds_cfg = cfg["dataset"]
    eval_args = dict(ds_cfg["val_args"])
    if held_out_signers is not None:
        eval_args["held_out_signers"] = held_out_signers
        eval_args["held_out_mode"] = held_out_mode or "reserve"
    if corruption:
        # We attach the corruption as a wrapper transform — only meaningful
        # for VideoDataset. LandmarkDataset has no pixel data to corrupt; if
        # the owner asks for a corruption against landmarks, refuse loudly.
        if ds_cfg["name"] != "video":
            raise SystemExit(f"corruption {corruption!r} is not meaningful for dataset {ds_cfg['name']!r}.")
        from data import robustness_corruptions as rc
        wrap = rc.build(corruption, severity=severity)
        existing = eval_args.get("transform")
        if existing is None:
            eval_args["transform"] = wrap
        else:
            def composed(x, e=existing, w=wrap):
                return w(e(x))
            eval_args["transform"] = composed
    ds = build_dataset(ds_cfg["name"], **eval_args)
    return DataLoader(
        ds,
        batch_size=cfg.get("training", {}).get("batch_size", 32),
        shuffle=False,
        num_workers=cfg.get("training", {}).get("num_workers", 4),
        pin_memory=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corruption", type=str, default=None, help="One of low_light/blur/occlusion/jpeg.")
    parser.add_argument("--severity", type=int, default=3)
    parser.add_argument("--held-out-signers", type=Path, default=None,
                        help="Path to signer_split.json. Switches dataset to 'reserve' mode.")
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    track = cfg.get("track", "unknown")
    out_md = args.out_md or Path(f"docs/{track}-results.md")
    out_json = args.out_json or Path(f"docs/{track}-results.json")

    device = torch.device(cfg.get("training", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    payload = ckpt_utils.load(args.checkpoint, map_location=device)
    train_cfg = payload.get("config", cfg)
    model = build_model(train_cfg["model"]["name"], **train_cfg["model"].get("args", {}))
    model.load_state_dict(payload["model"])
    model = model.to(device)

    held_out: set[Any] | None = None
    if args.held_out_signers and args.held_out_signers.exists():
        decision = json.loads(args.held_out_signers.read_text())
        held_out = set(decision["reserved_signers"])
        print(f"using {len(held_out)} held-out signers from {args.held_out_signers}")

    base_loader = build_eval_loader(cfg, held_out_signers=held_out, held_out_mode="reserve" if held_out else None, corruption=None, severity=0)
    base = evaluate(model, base_loader, device)
    print(f"[base] top1={base['top1']:.4f} top5={base['top5']:.4f} n={base['n_samples']}")

    # Build a sample input for FLOP counting from the first batch.
    sample = next(iter(base_loader))
    if len(sample) == 4:
        sample_input = sample[0][:1].to(device)
    else:
        sample_input = sample[0][:1].to(device)
    flop_info = count_flops(model, sample_input)

    variants: dict[str, Any] = {}
    if args.corruption:
        loader = build_eval_loader(cfg, held_out_signers=held_out, held_out_mode="reserve" if held_out else None,
                                   corruption=args.corruption, severity=args.severity)
        result = evaluate(model, loader, device)
        variants[f"{args.corruption}-s{args.severity}"] = result
        print(f"[{args.corruption}@{args.severity}] top1={result['top1']:.4f} top5={result['top5']:.4f}")

    results = {
        "top_level": base,
        "variants": variants,
        "params": flop_info["params"],
        "flops": flop_info["flops"],
        "checkpoint": str(args.checkpoint),
    }
    write_results(results, out_md, out_json, track, cfg)


if __name__ == "__main__":
    main()
