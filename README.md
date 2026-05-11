# DeafConnect

An end-to-end American Sign Language (ASL) recognition system that combines
two complementary models — a fingerspelling alphabet classifier and a
word-level video classifier — and ships them as an on-device iOS app
(SignReader) for real-time, offline sign-to-text translation.

The project covers the full pipeline from raw video to deployed Core ML:
data preparation, model training and comparison, evaluation, conversion
to Core ML, and integration into a SwiftUI app.

---

## Repository layout

```
deafconnect/
├── letters-asl/         # ASL alphabet (A–Z) — image & landmark classifiers
│   ├── notebooks/       # data prep, training (LSTM / MobileNet / ResNet), evaluation
│   ├── trained-models/  # *.h5 checkpoints (gitignored)
│   ├── dataset/         # raw images, 26 classes (gitignored)
│   └── realtime-demo.py # OpenCV + MediaPipe webcam demo
│
├── words-asl/           # Word-level recognition on the WLASL benchmark
│   ├── configs/         # YAML recipes per track
│   ├── data/            # PyTorch Datasets, augmentations, keypoint subsets
│   ├── models/          # landmark classifier + temporal-ToMe video transformer
│   ├── scripts/         # WLASL prep, landmark extraction, signer splits, demo
│   ├── deploy/          # PyTorch → Core ML conversion + verification
│   ├── train.py / eval.py
│   └── checkpoints/     # *.pt (gitignored)
│
├── mobile-app/          # SignReader — SwiftUI iOS app (iOS 17+)
│   ├── SignReader/      # Swift sources, Core ML model bundle, avatars
│   ├── project.yml      # XcodeGen spec (single source of truth)
│   └── Podfile          # MediaPipe Tasks via CocoaPods
│
└── report/              # Paper, figures, evaluation notebooks, LaTeX template
    ├── letters-results/ # Letter-model comparison notebook (confusion, accuracy)
    ├── word-results/    # Word-model comparison notebook
    ├── app-images/      # Screenshots used in the report
    └── latex-template/  # ACL-style report sources
```

---

## Components

### 1. Letters (A–Z fingerspelling)

Two model families trained on the ASL Alphabet dataset (~78 k images, 26
classes):

- **Image-based:** MobileNetV2 and ResNet50 transfer-learning baselines.
- **Landmark-based:** 21 MediaPipe Hands keypoints fed into an LSTM and a
  ResNet-style 1D classifier.

Notebooks in [letters-asl/notebooks/](letters-asl/notebooks/) walk through
data loading, preprocessing, training each architecture, and the side-by-side
comparison in [report/letters-results/model_comparison.ipynb](report/letters-results/model_comparison.ipynb).
The best landmark model is exported to Core ML by
[letters-asl/notebooks/convert_to_coreml.py](letters-asl/notebooks/convert_to_coreml.py)
and bundled as `LetterClassifier26.mlpackage` in the mobile app.

### 2. Words (WLASL)

A research environment for video sign-language recognition on the
[WLASL](https://dxli94.github.io/WLASL/) benchmark, organised as
configurable *tracks*:

- **Track 1 — Landmark classifier:** MediaPipe Holistic keypoints (pose +
  hands) fed into a temporal model. Tiny (~MB-scale), ANE-friendly, ideal
  for on-device inference.
- **Track 2 — Video transformer:** ViT-Small frame encoder + a 3-layer
  temporal Transformer with Token Merging (ToMe). Higher accuracy, larger
  footprint.

Pipeline:

```bash
# 1. Prepare a WLASL subset
python scripts/prepare_wlasl.py --num-classes 9

# 2. Extract MediaPipe Holistic landmarks (cached to disk)
python scripts/extract_landmarks.py

# 3. Train
python train.py --config configs/track1_landmark_v2.yaml
python train.py --config configs/track2_video_baseline.yaml

# 4. Evaluate
python eval.py --checkpoint checkpoints/track1-wlasl9idle_v2-s0/best.pt

# 5. Convert to Core ML for the iOS app
python deploy/convert_to_coreml.py --checkpoint <path> --output <name>.mlpackage
python deploy/verify_conversion.py   # parity check vs. PyTorch
```

Configs live in [words-asl/configs/](words-asl/configs/). Training is tracked
with [Weights & Biases](https://wandb.ai/).

### 3. SignReader (iOS app)

A SwiftUI app targeting iOS 17+ that runs both classifiers on-device via
Core ML and stitches their outputs into a chat-style transcript. Highlights:

- **Real-time camera pipeline** ([CameraManager.swift](mobile-app/SignReader/CameraManager.swift),
  [FrameBuffer.swift](mobile-app/SignReader/FrameBuffer.swift)) — AVFoundation
  capture, frame throttling, orientation handling.
- **Landmark extraction** via MediaPipe Tasks (`holistic_landmarker.task`).
- **Pluggable prediction engine** ([PredictionEngine.swift](mobile-app/SignReader/PredictionEngine.swift),
  [ModelProtocol.swift](mobile-app/SignReader/ModelProtocol.swift)) so the
  alphabet, landmark-word, and video-word models share a single interface.
- **Persistence** of transcripts and shortcuts via SwiftData.
- **Speech I/O** for two-way conversation between a deaf signer and a
  hearing speaker.
- **Rigged glTF avatars** (text → sign demonstration) rendered through
  [GLTFKit2](https://github.com/warrenm/GLTFKit2).

The Xcode project is *generated* from [mobile-app/project.yml](mobile-app/project.yml)
with [XcodeGen](https://github.com/yonki/XcodeGen) so the repo stays diff-friendly:

```bash
cd mobile-app
brew install xcodegen
xcodegen generate
pod install
open SignReader.xcworkspace   # always the workspace, not the project
```

Drop the trained `.mlpackage` files into
`mobile-app/SignReader/Resources/` before building. They are gitignored and
regenerated from the Python pipelines above.

---

## Getting started

### Python environment

Python 3.10 or 3.11 is recommended (the `coremltools` and `mediapipe` wheels
lag the latest Python).

```bash
cd words-asl
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The same environment works for `letters-asl`; the alphabet notebooks rely on
Keras/TensorFlow which you may install separately (`tensorflow==2.15.*`).

### Datasets

Datasets are **not** committed. To reproduce:

| Component | Source |
|---|---|
| Letters | [Kaggle: ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) — unzip into `letters-asl/dataset/` |
| Words   | [WLASL](https://dxli94.github.io/WLASL/) — place `WLASL_v0.3.json` and a `videos/` directory under `words-asl/wlasl/` |

### iOS build

Requirements: macOS 14+, Xcode 15+, CocoaPods, XcodeGen, an Apple developer
team ID (fill in `DEVELOPMENT_TEAM` in `mobile-app/project.yml`).

---

## Results

See the evaluation notebooks for confusion matrices, per-class accuracy, and
model comparison plots:

- [report/letters-results/model_comparison.ipynb](report/letters-results/model_comparison.ipynb)
- [report/word-results/model_comparison.ipynb](report/word-results/model_comparison.ipynb)

The accompanying paper sources live in
[report/latex-template/](report/latex-template/).

---

## License

Released for academic and research purposes. The WLASL dataset and the ASL
Alphabet dataset retain their original licenses; please consult the original
sources before redistributing any derived artefacts.

---

## Acknowledgements

- [WLASL](https://dxli94.github.io/WLASL/) — Li et al., word-level ASL benchmark.
- [MediaPipe](https://developers.google.com/mediapipe) — hand and holistic
  landmark estimation.
- [GLTFKit2](https://github.com/warrenm/GLTFKit2) — glTF rendering on Apple
  platforms.
