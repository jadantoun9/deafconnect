import os
import time

import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "resnet_landmarks_model.h5")
LABELS = {i: chr(65 + i) for i in range(26)}
CONF_THRESHOLD = 0.6
CAM_INDEX = 0

model = load_model(MODEL_PATH)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5,
                       min_tracking_confidence=0.5, max_num_hands=2)

cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise SystemExit(f"could not open camera {CAM_INDEX}")

prev = time.time()
fps = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        best = None
        for lms in results.multi_hand_landmarks:
            pts = np.array([[p.x, p.y, p.z] for p in lms.landmark], dtype=np.float32)
            pred = model.predict(pts.reshape(1, 21, 3), verbose=0)[0]
            idx = int(np.argmax(pred))
            conf = float(pred[idx])
            if best is None or conf > best[0]:
                best = (conf, idx, lms, pts)

        conf, idx, lms, pts = best
        mp_drawing.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS,
                                  mp_styles.get_default_hand_landmarks_style(),
                                  mp_styles.get_default_hand_connections_style())
        xs = pts[:, 0] * w
        ys = pts[:, 1] * h
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        pad = 12
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

        color = (0, 200, 0) if conf >= CONF_THRESHOLD else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{LABELS[idx]}  {conf*100:.1f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
    prev = now
    cv2.putText(frame, f"FPS {fps:.1f}   q to quit", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Sign Language - letters", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
