"""
Convert the trained ResNet-landmarks letter classifier to a Core ML
mlpackage that the SignReader iOS app can load.

Source : results/resnet_landmarks_model.h5
        Keras MLP, input (None, 21, 3) (21 hand landmarks x xyz),
        output (None, 26) softmax over letters A..Z.

Target : ios-app/SignReader/Resources/LetterClassifier26.mlpackage
        Core ML mlprogram, fp16 weights, ClassifierConfig over
        ["A","B",...,"Z"] so the model exposes classLabel /
        classLabel_probs and Swift can read predictions the same way
        the word-classifier already does.

The model already ends with Dense(26, softmax), so no extra Softmax
layer is needed before attaching ClassifierConfig.
"""
from pathlib import Path

import numpy as np
import tensorflow as tf
import coremltools as ct

REPO = Path(__file__).resolve().parents[2]
H5_PATH = REPO / "results" / "resnet_landmarks_model.h5"
OUT_PATH = REPO.parent / "ml-final" / "ios-app" / "SignReader" / "Resources" / "LetterClassifier26.mlpackage"

LABELS = [chr(ord("A") + i) for i in range(26)]


def main() -> None:
    print(f"loading {H5_PATH}")
    src_model = tf.keras.models.load_model(str(H5_PATH), compile=False)
    src_model.summary()

    # Sanity: a zero input should produce a uniform-ish softmax (26 ~= 1/26).
    probe = src_model.predict(np.zeros((1, 21, 3), dtype=np.float32), verbose=0)
    print(f"  probe sum = {probe.sum():.4f}, argmax = {int(probe.argmax())}, max prob = {probe.max():.4f}")

    # coremltools 9 + tf 2.16 trips on Keras's `_get_save_spec` shim.
    # Workaround: wrap the loaded model in a tf.function with a fixed
    # input signature and convert from the concrete function instead of
    # the Keras object. The numerical graph is identical.
    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 21, 3), dtype=tf.float32, name="input")])
    def inference(x):
        return src_model(x, training=False)

    concrete = inference.get_concrete_function()

    print(f"converting -> {OUT_PATH}")
    mlmodel = ct.convert(
        [concrete],
        source="tensorflow",
        inputs=[ct.TensorType(name="input", shape=(1, 21, 3), dtype=np.float32)],
        classifier_config=ct.ClassifierConfig(class_labels=LABELS),
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS16,
    )

    mlmodel.short_description = "SignReader letters - ResNet landmarks (21x3 -> A..Z)"
    mlmodel.author = "ml-final / SignReader"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(OUT_PATH))
    print(f"  saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
