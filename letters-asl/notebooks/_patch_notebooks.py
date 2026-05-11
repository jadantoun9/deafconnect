"""Patch the letters-asl notebooks so they (a) point at the new self-contained
deafconnect/letters-asl/ layout, (b) produce a train/val/test split, (c) cap
ResNet hyperopt budget, and (d) fix the meta-model train-on-test leakage in
the evaluation notebook.

Run from deafconnect/letters-asl/notebooks/ with the SLP venv python:
    .venv/bin/python3 _patch_notebooks.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent  # .../letters-asl/notebooks/


def load(nb_path):
    return json.loads(nb_path.read_text())


def save(nb_path, nb):
    nb_path.write_text(json.dumps(nb, indent=1))


def src_eq(cell, needle):
    return needle in "".join(cell.get("source", []))


def replace_source(cell, old, new):
    s = "".join(cell["source"])
    if old not in s:
        raise AssertionError(f"old string not found in cell:\n{old!r}\n---\n{s[:300]}")
    s = s.replace(old, new)
    cell["source"] = s.splitlines(keepends=True)


def set_source(cell, new):
    cell["source"] = new.splitlines(keepends=True)


# --- 1. data_loading_and_exploration.ipynb ----------------------------------
nb_path = ROOT / "data/data_loading_and_exploration.ipynb"
nb = load(nb_path)
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    if "C:\\\\Users\\\\Lenovo" in s:
        replace_source(
            c,
            'images_folder_path = "C:\\\\Users\\\\Lenovo\\Desktop\\\\sign-language-detector-python-master\\\\data\\\\asl_alphabet_train"',
            'images_folder_path = "../../dataset"',
        )
save(nb_path, nb)
print(f"patched {nb_path.relative_to(ROOT)}")


# --- 2. data_preprocessing.ipynb --------------------------------------------
nb_path = ROOT / "data/data_preprocessing.ipynb"
nb = load(nb_path)
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    if "data/processed/landmarks/landmarks.npz" in s:
        replace_source(
            c,
            "np.savez_compressed('data/processed/landmarks/landmarks.npz', X=X_landmarks, y=y_landmarks)\nnp.savez_compressed('data/processed/landmarks/images.npz', X=X_resized, y=y)",
            "np.savez_compressed('../../data/processed/landmarks.npz', X=X_landmarks, y=y_landmarks)\nnp.savez_compressed('../../data/processed/images.npz', X=X_resized, y=y)",
        )
save(nb_path, nb)
print(f"patched {nb_path.relative_to(ROOT)}")


# --- 3. data_dividing.ipynb -- add a val split ------------------------------
nb_path = ROOT / "data/data_dividing.ipynb"
nb = load(nb_path)
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    # landmarks split
    if (
        "X_train_landmarks, X_test_landmarks, y_train_landmarks, y_test_landmarks = train_test_split("
        in s
    ):
        set_source(
            c,
            "# Stratified 70/15/15 split: train, val, test.\n"
            "# random_state=42 keeps the original train/test boundaries reproducible.\n"
            "X_tmp, X_test_landmarks, y_tmp, y_test_landmarks = train_test_split(\n"
            "    X_landmarks, y_landmarks, test_size=0.15, random_state=42, stratify=y_landmarks,\n"
            ")\n"
            "X_train_landmarks, X_val_landmarks, y_train_landmarks, y_val_landmarks = train_test_split(\n"
            "    X_tmp, y_tmp, test_size=0.15/0.85, random_state=42, stratify=y_tmp,\n"
            ")\n"
            "print('landmarks shapes — train', X_train_landmarks.shape, 'val', X_val_landmarks.shape, 'test', X_test_landmarks.shape)\n",
        )
    # landmarks save
    if 'np.savez_compressed(\n    "../../data/divided/landmarks_dataset.npz",' in s:
        set_source(
            c,
            'np.savez_compressed(\n'
            '    "../../data/divided/landmarks_dataset.npz",\n'
            "    X_train=X_train_landmarks,\n"
            "    y_train=y_train_landmarks,\n"
            "    X_val=X_val_landmarks,\n"
            "    y_val=y_val_landmarks,\n"
            "    X_test=X_test_landmarks,\n"
            "    y_test=y_test_landmarks,\n"
            ")\n",
        )
    # images split
    if (
        "X_train_images, X_test_images, y_train_images, y_test_images = train_test_split("
        in s
    ):
        set_source(
            c,
            "X_tmp_i, X_test_images, y_tmp_i, y_test_images = train_test_split(\n"
            "    X_images, y_images, test_size=0.15, random_state=42, stratify=y_images,\n"
            ")\n"
            "X_train_images, X_val_images, y_train_images, y_val_images = train_test_split(\n"
            "    X_tmp_i, y_tmp_i, test_size=0.15/0.85, random_state=42, stratify=y_tmp_i,\n"
            ")\n"
            "print('images shapes — train', X_train_images.shape, 'val', X_val_images.shape, 'test', X_test_images.shape)\n",
        )
    # images save
    if 'np.savez_compressed(\n    "../../data/divided/images_dataset.npz",' in s:
        set_source(
            c,
            'np.savez_compressed(\n'
            '    "../../data/divided/images_dataset.npz",\n'
            "    X_train=X_train_images,\n"
            "    y_train=y_train_images,\n"
            "    X_val=X_val_images,\n"
            "    y_val=y_val_images,\n"
            "    X_test=X_test_images,\n"
            "    y_test=y_test_images,\n"
            ")\n",
        )
save(nb_path, nb)
print(f"patched {nb_path.relative_to(ROOT)}")


# --- 4. lstm.ipynb -- redirect save path to trained-models/ -----------------
nb_path = ROOT / "models/LSTM/lstm.ipynb"
nb = load(nb_path)
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    if "../../../results/lstm_landmarks_model.h5" in s:
        replace_source(
            c,
            "../../../results/lstm_landmarks_model.h5",
            "../../../trained-models/lstm_landmarks_model.h5",
        )
save(nb_path, nb)
print(f"patched {nb_path.relative_to(ROOT)}")


# --- 5. mobilenet.ipynb -----------------------------------------------------
nb_path = ROOT / "models/MobileNet/mobilenet.ipynb"
nb = load(nb_path)
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    if "'results/mobilenet_images_model.h5'" in s:
        replace_source(
            c,
            "'results/mobilenet_images_model.h5'",
            "'../../../trained-models/mobilenet_images_model.h5'",
        )
save(nb_path, nb)
print(f"patched {nb_path.relative_to(ROOT)}")


# --- 6. resnet.ipynb -- save paths + cap hyperopt max_evals=5 ---------------
nb_path = ROOT / "models/ResNet/resnet.ipynb"
nb = load(nb_path)
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    if "../../../results/resnet_images_model.h5" in s:
        replace_source(
            c,
            "../../../results/resnet_images_model.h5",
            "../../../trained-models/resnet_images_model.h5",
        )
    if "../../../results/resnet_landmarks_model.h5" in s:
        replace_source(
            c,
            "../../../results/resnet_landmarks_model.h5",
            "../../../trained-models/resnet_landmarks_model.h5",
        )
    if "../../../results/resnet_landmarks_tuned_model.h5" in s:
        replace_source(
            c,
            "../../../results/resnet_landmarks_tuned_model.h5",
            "../../../trained-models/resnet_landmarks_tuned_model.h5",
        )
    if "max_evals=50" in s:
        replace_source(c, "max_evals=50", "max_evals=5")
save(nb_path, nb)
print(f"patched {nb_path.relative_to(ROOT)}")


# --- 7. model_comparison.ipynb -- fix the leak, use new layout --------------
nb_path = ROOT / "evaluation/model_comparison.ipynb"
nb = load(nb_path)
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    # model load paths
    for k in (
        "resnet_landmarks_model",
        "resnet_images_model",
        "lstm_landmarks_model",
        "mobilenet_images_model",
    ):
        old = f'../../../results/{k}.h5'
        new = f'../../../trained-models/{k}.h5'
        if old in s:
            replace_source(c, old, new)
            s = "".join(c["source"])
    # add val split to the loaded data — replace the entire data-loading cell
    if 'np.load(\'../../../data/divided/landmarks_dataset.npz\')' in s and 'X_train_landmarks' in s:
        set_source(
            c,
            "import numpy as np\n"
            "\n"
            "landmarks_data = np.load('../../../data/divided/landmarks_dataset.npz')\n"
            "X_train_landmarks = landmarks_data['X_train']\n"
            "y_train_landmarks = landmarks_data['y_train']\n"
            "X_val_landmarks = landmarks_data['X_val']\n"
            "y_val_landmarks = landmarks_data['y_val']\n"
            "X_test_landmarks = landmarks_data['X_test']\n"
            "y_test_landmarks = landmarks_data['y_test']\n"
            "\n"
            "images_data = np.load('../../../data/divided/images_dataset.npz')\n"
            "X_train_images = images_data['X_train']\n"
            "y_train_images = images_data['y_train']\n"
            "X_val_images = images_data['X_val']\n"
            "y_val_images = images_data['y_val']\n"
            "X_test_images = images_data['X_test']\n"
            "y_test_images = images_data['y_test']\n"
            "\n"
            "print('landmarks — train', X_train_landmarks.shape, 'val', X_val_landmarks.shape, 'test', X_test_landmarks.shape)\n"
            "print('images   — train', X_train_images.shape, 'val', X_val_images.shape, 'test', X_test_images.shape)\n",
        )
    # the leaky ensemble — replace meta-feature build + LR-fit-on-test cells.
    # Replace the train_meta_features / test_meta_features cell.
    if "train_meta_features = np.hstack((" in s:
        set_source(
            c,
            "# LEAK-FREE ENSEMBLE: fit LR on the held-out *val* predictions,\n"
            "# evaluate on the held-out *test* predictions. The original notebook\n"
            "# trained the meta-model on the test labels themselves (`fit(..., y_test_landmarks)`)\n"
            "# and then scored it on those same labels — see the git history of this cell.\n"
            "resnet_val_preds  = resnet_landmarks_model.predict(X_val_landmarks,  verbose=0)\n"
            "lstm_val_preds    = lstm_landmarks_model.predict(X_val_landmarks,    verbose=0)\n"
            "resnet_test_preds = resnet_landmarks_model.predict(X_test_landmarks, verbose=0)\n"
            "lstm_test_preds   = lstm_landmarks_model.predict(X_test_landmarks,   verbose=0)\n"
            "\n"
            "val_meta_features  = np.hstack((resnet_val_preds,  lstm_val_preds))\n"
            "test_meta_features = np.hstack((resnet_test_preds, lstm_test_preds))\n",
        )
    if "meta_model = LogisticRegression()" in s and "meta_model.fit(train_meta_features, y_test_landmarks)" in s:
        set_source(
            c,
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.metrics import accuracy_score\n"
            "\n"
            "meta_model = LogisticRegression(max_iter=1000)\n"
            "meta_model.fit(val_meta_features, y_val_landmarks)   # honest: fit on val, never on test\n"
            "\n"
            "ensemble_preds = meta_model.predict(test_meta_features)\n"
            "ensemble_accuracy = accuracy_score(y_test_landmarks, ensemble_preds)\n"
            "print(f'Leak-free ensemble (ResNet+LSTM landmarks) test accuracy: {ensemble_accuracy:.4f}')\n"
            "ensemble_accuracy\n",
        )
save(nb_path, nb)
print(f"patched {nb_path.relative_to(ROOT)}")

print("\nAll notebooks patched.")
