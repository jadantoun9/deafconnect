"""
convert_to_coreml.py — generic PyTorch -> ONNX -> Core ML converter.

Use this for production models (Track 1/2/3). The much smaller
`sanity/convert_sanity.py` exists separately so the sanity gate's
conversion can't be confused with a production-conversion bug.

Flow:
    PyTorch checkpoint -> torch.jit.trace -> ONNX (optional) -> Core ML

Why ONNX as an intermediate:
    Some Core ML converter passes work better when the model is already in
    ONNX form (cleaner op set, easier op-fallback inspection). The flag
    `--via onnx` enables that route; `--via torchscript` is the direct path
    used when ONNX export is breaking on a custom op.

Output:
    A single .mlpackage at --out, plus a JSON sidecar `<out>.report.json`
    listing any ops that fell back to CPU.

Run:
    python /deploy/convert_to_coreml.py \
        --checkpoint checkpoints/track2/best.pt \
        --out checkpoints/track2/Track2.mlpackage \
        --precision fp16 --via torchscript --input-shape 1,3,16,224,224
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from utils import checkpoint as ckpt_utils
from utils.registry import build_model


def parse_shape(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def trace_model(model: torch.nn.Module, input_shape: tuple[int, ...]) -> torch.jit.ScriptModule:
    model.eval()
    example = torch.zeros(*input_shape, dtype=torch.float32)
    # check_trace=False because PyTorch's verifier flags re-runs that produce
    # the same outputs but different SSA node IDs (mangled scope names) as
    # "graphs differed". For our eval-mode Transformer with dropout off, the
    # trace is deterministic by construction; the per-run mangling is noise.
    return torch.jit.trace(model, example, strict=False, check_trace=False)


def export_onnx(model: torch.nn.Module, input_shape: tuple[int, ...], path: Path) -> None:
    model.eval()
    example = torch.zeros(*input_shape, dtype=torch.float32)
    torch.onnx.export(
        model,
        example,
        str(path),
        opset_version=17,
        input_names=["input"],
        output_names=["logits"],
        # We deliberately don't pass `dynamic_axes` — Core ML's mlprogram path
        # generally produces tighter graphs with fixed shapes, and the
        # downstream iOS app uses fixed-size frame buffers anyway.
    )


def convert(
    checkpoint_path: Path,
    out_path: Path,
    precision: str,
    via: str,
    input_shape: tuple[int, ...],
    inputs_kind: str,
    classes_path: Path | None,
) -> None:
    try:
        import coremltools as ct
    except ImportError as e:
        raise SystemExit("coremltools not installed.") from e

    payload = ckpt_utils.load(checkpoint_path)
    train_cfg: dict[str, Any] = payload.get("config") or {}
    model_cfg = train_cfg.get("model")
    if model_cfg is None:
        raise SystemExit(
            f"checkpoint {checkpoint_path} does not embed its config. Re-train "
            f"with train.py (which embeds config) or pass --model-* flags."
        )

    model = build_model(model_cfg["name"], **model_cfg.get("args", {}))
    model.load_state_dict(payload["model"])
    model.eval()

    classes = None
    if classes_path and classes_path.exists():
        c = json.loads(classes_path.read_text())
        # Class file from prepare_wlasl.py has both `gloss_to_label` and
        # `label_to_gloss`; the latter is what Core ML wants.
        classes = c.get("label_to_gloss", c) if isinstance(c, dict) else c

    # ClassifierConfig + mlprogram does NOT auto-insert softmax — Core ML
    # treats the model's last tensor as the probability vector verbatim.
    # Track 1 / Track 2 / Track 3 all output raw logits from a final
    # Linear, which would surface as nonsense confidences (>100%, <0%) on
    # the iOS side. Wrap with Softmax explicitly when a classifier
    # output is requested. Same fix already in place for the sanity
    # model in /sanity/convert_sanity.py.
    if classes is not None:
        model = torch.nn.Sequential(model, torch.nn.Softmax(dim=1))
        model.eval()

    if via == "onnx":
        onnx_path = out_path.with_suffix(".onnx")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        export_onnx(model, input_shape, onnx_path)
        source = str(onnx_path)
        print(f"  exported ONNX -> {onnx_path}")
    elif via == "torchscript":
        traced = trace_model(model, input_shape)
        source = traced
    else:
        raise ValueError(f"--via must be onnx or torchscript, got {via!r}")

    # Build the Core ML input spec. Track 1 (landmarks) takes a tensor; Track
    # 2/3 (video) take a tensor of stacked frames. Image-typed inputs are
    # only used for the sanity model, not for full clips, because Core ML's
    # ImageType wants a single 3-channel HxW frame.
    if inputs_kind == "tensor":
        ct_inputs = [ct.TensorType(name="input", shape=input_shape, dtype=float)]
    elif inputs_kind == "image":
        if len(input_shape) != 4:
            raise SystemExit("inputs_kind=image requires a 4D shape (1,3,H,W).")
        ct_inputs = [ct.ImageType(name="input", shape=input_shape, scale=1.0 / 255.0,
                                  color_layout=ct.colorlayout.RGB)]
    else:
        raise ValueError(f"unknown inputs_kind={inputs_kind!r}")

    convert_kwargs: dict[str, Any] = dict(
        inputs=ct_inputs,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS16,
    )
    if classes:
        convert_kwargs["classifier_config"] = ct.ClassifierConfig(class_labels=classes)

    print(f"  converting via {via} ...")
    mlmodel = ct.convert(source, **convert_kwargs)

    # Optional INT8 weight palettization. INT8 is *separate* from compute
    # precision: weights stored as int8 with a learned scale; activations
    # still flow at fp16. Enabled by --precision int8.
    if precision == "int8":
        print("  applying INT8 weight palettization ...")
        from coremltools.optimize.coreml import (
            OpPalettizerConfig,
            OptimizationConfig,
            palettize_weights,
        )
        config = OptimizationConfig(global_config=OpPalettizerConfig(mode="kmeans", nbits=8))
        mlmodel = palettize_weights(mlmodel, config=config)

    mlmodel.short_description = f"SignReader — {model_cfg['name']} converted via {via}, precision={precision}"
    mlmodel.author = "ml-final / SignReader"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_path))
    print(f"  saved -> {out_path}")

    # Op-fallback report. coremltools doesn't expose ANE-vs-CPU directly; the
    # closest signal is the op-types in the resulting program. We dump the op
    # names so the owner can diff against docs/coreml-compat.md and spot
    # anything new.
    report = {
        "checkpoint": str(checkpoint_path),
        "out": str(out_path),
        "precision": precision,
        "via": via,
        "input_shape": list(input_shape),
        "ops": list_ops(mlmodel),
    }
    report_path = out_path.with_suffix(out_path.suffix + ".report.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(f"  wrote op report -> {report_path}")


def list_ops(mlmodel: Any) -> list[str]:
    """Best-effort extraction of unique op names from the mlprogram."""
    spec = mlmodel.get_spec()
    op_names: set[str] = set()
    try:
        program = spec.mlProgram
        for func_name in program.functions:
            func = program.functions[func_name]
            for block in func.block_specializations:
                for op in func.block_specializations[block].operations:
                    op_names.add(op.type)
    except Exception:
        pass
    return sorted(op_names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    parser.add_argument("--via", choices=["onnx", "torchscript"], default="torchscript")
    parser.add_argument("--input-shape", type=parse_shape, required=True,
                        help="Comma-separated shape, e.g. 1,3,16,224,224 for video; 1,32,75,3 for landmarks.")
    parser.add_argument("--inputs-kind", choices=["tensor", "image"], default="tensor")
    parser.add_argument("--classes", type=Path, default=None,
                        help="Optional class mapping JSON (from prepare_wlasl.py).")
    args = parser.parse_args()

    convert(
        checkpoint_path=args.checkpoint,
        out_path=args.out,
        precision=args.precision,
        via=args.via,
        input_shape=args.input_shape,
        inputs_kind=args.inputs_kind,
        classes_path=args.classes,
    )


if __name__ == "__main__":
    main()
