// LetterClassifier.swift
//
// Thin wrapper around the bundled LetterClassifier26.mlpackage — the
// ResNet-landmarks ASL alphabet model ported from
// `Sign Language Python/results/resnet_landmarks_model.h5` via
// `src/letters/convert_to_coreml.py`.
//
// Shape contract (matches the original Keras model and src/realtime.py):
//   input : [1, 21, 3]  — 21 MediaPipe hand landmarks, raw [0,1] coords
//   output: classLabel (String, "A".."Z") + classLabel_probs (Dict<String, Double>)
//
// Unlike the word LandmarkClassifier this model is single-frame and
// single-hand: no 32-frame ring, no per-frame normalisation. The
// LandmarkExtractor's `snapshotHandFrame()` returns the most recent
// best-detected hand verbatim.
import CoreML
import Foundation

@MainActor
final class LetterClassifier: ObservableObject {

    public struct Prediction: Equatable {
        public let label: String
        public let confidence: Float
        public let allConfidences: [String: Float]
    }

    @Published public private(set) var topPrediction: Prediction?
    @Published public private(set) var isModelLoaded: Bool = false
    @Published public private(set) var lastError: String?

    private var model: MLModel?
    private static let resourceName: String = "LetterClassifier26"

    public init() {
        loadModel()
    }

    private func loadModel() {
        let name = Self.resourceName
        if let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc")
            ?? Bundle.main.url(forResource: name, withExtension: "mlpackage") {
            do {
                let config = MLModelConfiguration()
                config.computeUnits = .all
                let m = try MLModel(contentsOf: url, configuration: config)
                self.model = m
                self.isModelLoaded = true
                self.lastError = nil
                NSLog("LetterClassifier: loaded \(url.lastPathComponent)")
                return
            } catch {
                self.lastError = "Model load failed: \(error.localizedDescription)"
                NSLog("LetterClassifier: load failed for \(url.lastPathComponent): \(error)")
            }
        }
        self.isModelLoaded = false
        self.lastError = "\(name).mlpackage not found in app bundle."
    }

    /// Clear the displayed prediction (e.g. on tab switch).
    public func reset() {
        topPrediction = nil
    }

    /// Run the model on a [1, 21, 3] MLMultiArray. Updates `topPrediction`.
    public func predict(input: MLMultiArray) async {
        guard let model else { return }

        let provider: MLFeatureProvider
        do {
            provider = try MLDictionaryFeatureProvider(dictionary: ["input": MLFeatureValue(multiArray: input)])
        } catch {
            lastError = "Input wrap failed: \(error.localizedDescription)"
            return
        }

        do {
            let outputs = try await model.prediction(from: provider)
            guard
                let label = outputs.featureValue(for: "classLabel")?.stringValue,
                let probsRaw = outputs.featureValue(for: "classLabel_probs")?.dictionaryValue as? [String: Double]
            else {
                lastError = "Model output missing expected fields."
                return
            }
            let probs = probsRaw.mapValues { Float($0) }
            let confidence = probs[label] ?? 0
            topPrediction = Prediction(
                label: label,
                confidence: confidence,
                allConfidences: probs
            )
            lastError = nil
        } catch {
            lastError = "Prediction failed: \(error.localizedDescription)"
        }
    }
}
