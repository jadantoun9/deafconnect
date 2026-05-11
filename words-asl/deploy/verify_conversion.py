"""
verify_conversion.py — round-trip test for converted Core ML packages.

Purpose: catch silent numerical drift introduced by tracing or FP16
conversion. We feed the same random inputs through:
  - the original PyTorch checkpoint
  - the converted Core ML package

and report:
  - max absolute error per output
  - mean absolute error
  - top-1 agreement rate
  - top-5 agreement rate

Tolerances are eyeballed defaults; the script exits with a non-zero status if
they're exceeded so it can be wired into CI later.

Run:
    python /deploy/verify_conversion.py \
        --checkpoint checkpoints/track2/best.pt \
        --mlpackage checkpoints/track2/Track2.mlpackage \
        --input-shape 1,3,16,224,224 \
        --num-trials 8
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from utils import checkpoint as ckpt_utils
from utils.registry import build_model


def parse_shape(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def load_pytorch(checkpoint_path: Path) -> torch.nn.Module:
    payload = ckpt_utils.load(checkpoint_path)
    cfg = payload.get("config", {}).get("model")
    if cfg is None:
        raise SystemExit("Checkpoint missing embedded config.")
    model = build_model(cfg["name"], **cfg.get("args", {}))
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def run_pytorch(model: torch.nn.Module, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        out = model(torch.from_numpy(x).float())
    return out.cpu().numpy()


def run_coreml(model_path: Path, x: np.ndarray) -> np.ndarray:
    import coremltools as ct
    m = ct.models.MLModel(str(model_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    # CPU_ONLY for a stable comparison — ANE FP16 paths are non-deterministic
    # in the lower-order bits and would inflate max-abs-error here.
    input_name = m.input_description._fd_spec[0].name
    out = m.predict({input_name: x})
    if len(out) == 1:
        return np.asarray(next(iter(out.values())))
    # Classifier outputs include both probabilities and the predicted label;
    # pick the largest tensor as the logits stand-in.
    biggest_key = max(out.keys(), key=lambda k: np.asarray(out[k]).size if hasattr(out[k], '__len__') else 0)
    return np.asarray(out[biggest_key])


def compare(pt_out: np.ndarray, ct_out: np.ndarray) -> dict[str, Any]:
    if pt_out.shape != ct_out.shape:
        return {
            "shape_mismatch": True,
            "pt_shape": pt_out.shape,
            "ct_shape": ct_out.shape,
        }
    err = np.abs(pt_out - ct_out)
    pt_top = np.argsort(-pt_out, axis=-1)
    ct_top = np.argsort(-ct_out, axis=-1)
    return {
        "max_abs_err": float(err.max()),
        "mean_abs_err": float(err.mean()),
        "top1_match": float(np.mean(pt_top[..., 0] == ct_top[..., 0])),
        "top5_match": float(np.mean([
            ct_top[i, 0] in pt_top[i, :5] for i in range(pt_out.shape[0])
        ])) if pt_out.ndim == 2 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mlpackage", type=Path, required=True)
    parser.add_argument("--input-shape", type=parse_shape, required=True)
    parser.add_argument("--num-trials", type=int, default=8)
    parser.add_argument("--max-abs-err-tol", type=float, default=5e-2,
                        help="FP16 conversion typically lands in the 1e-3..1e-2 range; loosen for INT8.")
    parser.add_argument("--top1-tol", type=float, default=1.0,
                        help="Required minimum top-1 agreement rate (0..1).")
    args = parser.parse_args()

    model = load_pytorch(args.checkpoint)

    rng = np.random.default_rng(0)
    aggregate = {"max_abs_err": 0.0, "mean_abs_err_sum": 0.0, "top1_matches": 0, "top5_matches": 0, "n": 0}

    for trial in range(args.num_trials):
        x = rng.uniform(0.0, 1.0, size=args.input_shape).astype(np.float32)
        pt_out = run_pytorch(model, x)
        ct_out = run_coreml(args.mlpackage, x)
        cmp = compare(pt_out, ct_out)
        if cmp.get("shape_mismatch"):
            raise SystemExit(f"shape mismatch: pt={cmp['pt_shape']} ct={cmp['ct_shape']}")
        print(f"  trial {trial}: max_abs_err={cmp['max_abs_err']:.4e} mean={cmp['mean_abs_err']:.4e} "
              f"top1={cmp['top1_match']:.3f}")
        aggregate["max_abs_err"] = max(aggregate["max_abs_err"], cmp["max_abs_err"])
        aggregate["mean_abs_err_sum"] += cmp["mean_abs_err"]
        aggregate["top1_matches"] += cmp["top1_match"]
        if cmp.get("top5_match") is not None:
            aggregate["top5_matches"] += cmp["top5_match"]
        aggregate["n"] += 1

    n = aggregate["n"]
    summary = {
        "max_abs_err": aggregate["max_abs_err"],
        "mean_abs_err": aggregate["mean_abs_err_sum"] / n,
        "top1_agreement": aggregate["top1_matches"] / n,
        "top5_agreement": aggregate["top5_matches"] / n,
        "trials": n,
    }
    print(f"\nsummary: {summary}")

    failures = []
    if summary["max_abs_err"] > args.max_abs_err_tol:
        failures.append(f"max_abs_err {summary['max_abs_err']:.4e} > tol {args.max_abs_err_tol:.4e}")
    if summary["top1_agreement"] < args.top1_tol:
        failures.append(f"top1_agreement {summary['top1_agreement']:.3f} < tol {args.top1_tol:.3f}")

    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        raise SystemExit(1)
    print("OK: round-trip within tolerance.")


if __name__ == "__main__":
    main()
