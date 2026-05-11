"""
realtime_demo.py — real-time WLASL-100 inference from a webcam.

Loads a Track 1 landmark checkpoint, captures frames from the webcam, runs
MediaPipe Holistic, maintains a rolling landmark buffer, and overlays the
top-K predictions on the live video.

Run:
    .venv/bin/python scripts/realtime_demo.py \\
        --checkpoint checkpoints/track1-wlasl100-s0/best.pt \\
        --classes wlasl/prepared/wlasl100_classes.json

Keys:  q = quit, r = reset rolling buffer.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

# Allow running as `python scripts/realtime_demo.py` (no -m).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import mediapipe as mp
import numpy as np
import torch

from utils.registry import build_model

NUM_POSE = 33
NUM_HAND = 21
NUM_KEYPOINTS = NUM_POSE + 2 * NUM_HAND


def normalise_landmarks(lm: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Mirror /data/landmark_dataset.py:_normalise_landmarks exactly:
    per-frame zero-centre + scale to [-1, 1] over (x, y); leave z raw."""
    lm = lm.copy()
    F = lm.shape[0]
    for f in range(F):
        m = valid[f]
        if m.sum() < 2:
            continue
        xy = lm[f, m, :2]
        cx, cy = xy.mean(axis=0)
        lm[f, :, 0] -= cx
        lm[f, :, 1] -= cy
        max_abs = float(np.max(np.abs(lm[f, m, :2])))
        if max_abs > 1e-6:
            lm[f, :, :2] /= max_abs
    return lm


def uniform_indices(total: int, T: int) -> list[int]:
    if total <= 1:
        return [0] * T
    return list(np.linspace(0, total - 1, T, dtype=int))


def hand_motion_score(buf_lm: list[np.ndarray], buf_valid: list[np.ndarray],
                       window: int = 16) -> float:
    """Mean per-frame movement of detected hand keypoints (in raw [0,1] coords)
    over the last `window` buffered frames. Returns 0 if too few frames or no
    detected hands. Used as the demo-side motion gate."""
    if len(buf_lm) < 2:
        return 0.0
    n = min(window, len(buf_lm))
    arr = np.stack(buf_lm[-n:], axis=0)            # [n, 75, 3]
    val = np.stack(buf_valid[-n:], axis=0)         # [n, 75]
    hand_xy = arr[:, NUM_POSE:, :2]                 # [n, 42, 2]
    hand_val = val[:, NUM_POSE:]                    # [n, 42]
    # Frame-to-frame displacement, masked to keypoints valid in BOTH frames.
    diffs = np.linalg.norm(hand_xy[1:] - hand_xy[:-1], axis=-1)   # [n-1, 42]
    paired_valid = hand_val[1:] & hand_val[:-1]                    # [n-1, 42]
    if not paired_valid.any():
        return 0.0
    return float(diffs[paired_valid].mean())


def extract_frame(holistic, rgb: np.ndarray):
    rgb.flags.writeable = False
    result = holistic.process(rgb)
    lm = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
    valid = np.zeros(NUM_KEYPOINTS, dtype=bool)
    if result.pose_landmarks is not None:
        for k, p in enumerate(result.pose_landmarks.landmark[:NUM_POSE]):
            lm[k] = (p.x, p.y, p.z)
            valid[k] = True
    if result.left_hand_landmarks is not None:
        for k, p in enumerate(result.left_hand_landmarks.landmark[:NUM_HAND]):
            lm[NUM_POSE + k] = (p.x, p.y, p.z)
            valid[NUM_POSE + k] = True
    if result.right_hand_landmarks is not None:
        for k, p in enumerate(result.right_hand_landmarks.landmark[:NUM_HAND]):
            lm[NUM_POSE + NUM_HAND + k] = (p.x, p.y, p.z)
            valid[NUM_POSE + NUM_HAND + k] = True
    return lm, valid, result


def draw_skeleton(bgr: np.ndarray, result) -> np.ndarray:
    drawing = mp.solutions.drawing_utils
    styles = mp.solutions.drawing_styles
    holistic_module = mp.solutions.holistic
    if result.pose_landmarks:
        drawing.draw_landmarks(
            bgr, result.pose_landmarks, holistic_module.POSE_CONNECTIONS,
            landmark_drawing_spec=styles.get_default_pose_landmarks_style(),
        )
    if result.left_hand_landmarks:
        drawing.draw_landmarks(
            bgr, result.left_hand_landmarks, holistic_module.HAND_CONNECTIONS,
            landmark_drawing_spec=styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=styles.get_default_hand_connections_style(),
        )
    if result.right_hand_landmarks:
        drawing.draw_landmarks(
            bgr, result.right_hand_landmarks, holistic_module.HAND_CONNECTIONS,
            landmark_drawing_spec=styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=styles.get_default_hand_connections_style(),
        )
    return bgr


def draw_panel(bgr, predictions, buffer_len, T, fps, hand_pct, threshold,
                motion, motion_threshold, gated_reason):
    """`gated_reason`: empty string if a prediction is shown, otherwise a short
    label explaining why the demo is suppressing it ('still', 'idle', 'low')."""
    h, w = bgr.shape[:2]
    n_rows = max(len(predictions), 1)
    panel_h = 60 + 28 * n_rows + 16
    panel = np.zeros((panel_h, 400, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)
    cv2.putText(panel, f"Top-{len(predictions) or '?'} (smoothed)",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    if predictions:
        # Big banner: the active prediction, or "—" with reason when gated.
        if gated_reason:
            banner = "—"
            banner_color = (120, 120, 120)
        else:
            banner = predictions[0][0]
            banner_color = (0, 255, 0)
        cv2.putText(panel, banner, (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, banner_color, 2)
        for i, (gloss, prob) in enumerate(predictions):
            y = 92 + i * 28
            is_top_active = (i == 0 and not gated_reason)
            color = (0, 255, 0) if is_top_active else (200, 200, 200)
            text = f"{i+1}. {gloss:<10s} {prob*100:5.1f}%"
            cv2.putText(panel, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    else:
        cv2.putText(panel, f"Filling buffer  {buffer_len}/{T}", (12, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    bgr[10:10 + panel.shape[0], 10:10 + panel.shape[1]] = panel

    status = (f"FPS {fps:4.1f}  buf {buffer_len}  hand% {hand_pct*100:4.1f}  "
              f"motion {motion*1000:5.2f}/k (gate {motion_threshold*1000:.2f})  "
              f"thresh {threshold:.2f}  "
              + (f"[gated: {gated_reason}]" if gated_reason else "[active]")
              + "   q=quit  r=reset")
    cv2.putText(bgr, status, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return bgr


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/track1-wlasl9idle-s0/best.pt"))
    parser.add_argument("--classes", type=Path,
                        default=Path("wlasl/prepared/wlasl9idle_classes.json"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--buffer-frames", type=int, default=64,
                        help="Rolling window length. Inference samples T uniformly from this.")
    parser.add_argument("--infer-every", type=int, default=4,
                        help="Run inference every N captured frames.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                        help="Only show top-1 if its (smoothed) softmax prob exceeds this.")
    parser.add_argument("--smoothing-window", type=int, default=10,
                        help="Average the last N inference probability vectors before "
                             "displaying. Stabilises the top-1 over short flips.")
    parser.add_argument("--motion-threshold", type=float, default=0.005,
                        help="Mean per-frame hand-keypoint displacement (in MediaPipe's "
                             "[0,1] coords) below which the demo gates output to '—'.")
    parser.add_argument("--idle-class-name", default="idle",
                        help="If this class is the smoothed top-1, gate output to '—'.")
    parser.add_argument("--no-mirror", action="store_true",
                        help="Disable horizontal flip of the webcam feed.")
    args = parser.parse_args()

    print(f"[demo] loading {args.checkpoint}")
    payload = torch.load(str(args.checkpoint), map_location="cpu")
    cfg = payload.get("config")
    if cfg is None:
        raise SystemExit("checkpoint missing config; cannot rebuild model")
    model_cfg = cfg["model"]
    T = cfg["dataset"]["val_args"].get("num_frames", 32)
    model = build_model(model_cfg["name"], **model_cfg.get("args", {}))
    model.load_state_dict(payload["model"])
    model.eval()
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = model.to(device)

    # If the checkpoint was trained with a keypoint subset, we need to slice
    # the live 75-keypoint MediaPipe output the same way before inference.
    keypoint_subset = cfg["dataset"]["val_args"].get("keypoint_subset", "all")
    if keypoint_subset == "all":
        keep_idx = None
    else:
        from data.keypoint_subsets import get_subset
        keep_idx = np.asarray(get_subset(keypoint_subset)["indices"], dtype=np.int64)
    print(f"[demo] device={device} T={T} val_top1@best={payload.get('metric')} "
          f"keypoint_subset={keypoint_subset} K={len(keep_idx) if keep_idx is not None else 75}")

    class_data = json.loads(args.classes.read_text())
    label_to_gloss: list[str] = class_data["label_to_gloss"]
    if len(label_to_gloss) != model.num_classes:
        raise SystemExit(
            f"class count mismatch: model has {model.num_classes}, "
            f"classes file has {len(label_to_gloss)}"
        )

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera index {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    # Warm-up: AVFoundation can take 1-2 seconds to start delivering frames
    # after VideoCapture returns, especially after a recent capture session
    # was torn down. Burn through a few failed reads before the main loop so
    # we don't bail on the first .read() of the day.
    warmup_deadline = time.time() + 6.0
    while time.time() < warmup_deadline:
        ok, _ = cap.read()
        if ok:
            break
        time.sleep(0.1)
    else:
        cap.release()
        raise SystemExit("camera opened but never delivered a frame within 6s")

    buf_lm: deque[np.ndarray] = deque(maxlen=args.buffer_frames)
    buf_valid: deque[np.ndarray] = deque(maxlen=args.buffer_frames)
    # Rolling history of the raw softmax probability vectors. Averaging these
    # is what produces the displayed top-K — single-shot inference flips too
    # quickly between similar signs to be readable.
    prob_history: deque[np.ndarray] = deque(maxlen=args.smoothing_window)

    predictions: list[tuple[str, float]] = []
    gated_reason = ""
    motion = 0.0
    frame_idx = 0
    t_last = time.time()
    fps_smooth = 0.0
    print("[demo] press q to quit, r to reset the buffer")

    try:
        consecutive_read_failures = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                # Tolerate a few transient drops; only bail if the camera
                # actually stops delivering frames for a while.
                consecutive_read_failures += 1
                if consecutive_read_failures > 30:
                    print("[demo] camera read failing for 30 frames; exiting")
                    break
                time.sleep(0.05)
                continue
            consecutive_read_failures = 0
            if not args.no_mirror:
                bgr = cv2.flip(bgr, 1)

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            lm, valid, mp_result = extract_frame(holistic, rgb)
            buf_lm.append(lm)
            buf_valid.append(valid)
            frame_idx += 1

            if frame_idx % args.infer_every == 0 and len(buf_lm) >= T:
                lm_arr = np.stack(list(buf_lm), axis=0)
                valid_arr = np.stack(list(buf_valid), axis=0)
                idx = uniform_indices(lm_arr.shape[0], T)
                lm_sampled = lm_arr[idx]
                valid_sampled = valid_arr[idx]
                # Slice down to the keypoint subset the checkpoint expects, BEFORE
                # normalisation, mirroring /data/landmark_dataset.py.
                if keep_idx is not None:
                    lm_sampled = lm_sampled[:, keep_idx, :]
                    valid_sampled = valid_sampled[:, keep_idx]
                lm_norm = normalise_landmarks(lm_sampled, valid_sampled)
                x = torch.from_numpy(lm_norm).float().unsqueeze(0).to(device)
                v = torch.from_numpy(valid_sampled).bool().unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(x, valid_mask=v)
                    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                prob_history.append(probs)

                # Smoothed display: average the last N probability vectors,
                # then take top-K of the average. Reduces inter-frame flicker.
                smoothed = np.mean(np.stack(list(prob_history), axis=0), axis=0)
                topk = smoothed.argsort()[::-1][:args.top_k]
                predictions = [(label_to_gloss[i], float(smoothed[i])) for i in topk]

                # Decide whether to gate the displayed prediction. Order:
                # (1) too little hand motion -> "still"
                # (2) the model itself thinks this is idle -> "idle"
                # (3) top-1 confidence below threshold -> "low"
                motion = hand_motion_score(list(buf_lm), list(buf_valid))
                top1_gloss, top1_prob = predictions[0]
                if motion < args.motion_threshold:
                    gated_reason = "still"
                elif top1_gloss == args.idle_class_name:
                    gated_reason = "idle"
                elif top1_prob < args.confidence_threshold:
                    gated_reason = "low"
                else:
                    gated_reason = ""

            bgr = draw_skeleton(bgr, mp_result)

            now = time.time()
            dt = now - t_last
            t_last = now
            if dt > 0:
                inst = 1.0 / dt
                fps_smooth = inst if fps_smooth == 0.0 else 0.9 * fps_smooth + 0.1 * inst

            hand_pct = (
                float(np.mean([v[NUM_POSE:].any() for v in buf_valid]))
                if buf_valid else 0.0
            )
            bgr = draw_panel(bgr, predictions, len(buf_lm), T, fps_smooth,
                             hand_pct, args.confidence_threshold,
                             motion, args.motion_threshold, gated_reason)

            cv2.imshow("WLASL-100 real-time (Track 1 baseline)", bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                buf_lm.clear()
                buf_valid.clear()
                prob_history.clear()
                predictions = []
                gated_reason = ""
    finally:
        cap.release()
        cv2.destroyAllWindows()
        holistic.close()


if __name__ == "__main__":
    main()
