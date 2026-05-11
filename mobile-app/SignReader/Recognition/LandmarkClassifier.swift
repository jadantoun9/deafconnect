// LandmarkClassifier.swift
//
// Thin wrapper around the bundled LandmarkClassifier9.mlpackage. Exposes
// a single `predict(input:)` API that consumes the MLMultiArray
// snapshot from `LandmarkExtractor.snapshotInputTensor()` and publishes
// a top-1 (label, confidence) pair the UI can render.
//
// We deliberately don't go through the existing PredictionEngine /
// SignModel protocol — those are for the Phase 0 video-frame pipeline.
// The landmark path is structurally different (per-frame Vision pass +
// 32-frame ring buffer + tensor input) and lives on its own track.
import CoreML
import Foundation

/// The two bundled landmark-model variants. Each ships as its own
/// `.mlpackage` and consumes a different per-frame keypoint count, so
/// switching variants reloads the Core ML model AND changes the
/// `LandmarkExtractor` snapshot subset.
public enum LandmarkModelVariant: String, CaseIterable, Identifiable {
    case v1   // 75-keypoint full MediaPipe Holistic output
    case v2   // 48-keypoint pruned: shoulders/elbows/wrists + hands

    public var id: String { rawValue }

    /// Resource basename inside the app bundle (no extension).
    public var resourceName: String {
        switch self {
        case .v1: return "LandmarkClassifier9"
        case .v2: return "LandmarkClassifier9V2"
        }
    }

    /// Which `KeypointSubset` `LandmarkExtractor.snapshotInputTensor(subset:)`
    /// must use to produce inputs in this variant's layout.
    public var keypointSubset: KeypointSubset {
        switch self {
        case .v1: return .all
        case .v2: return .upperBodyHands
        }
    }

    /// Short label for UI.
    public var displayName: String {
        switch self {
        case .v1: return "v1 · 75 keypoints"
        case .v2: return "v2 · 48 keypoints"
        }
    }
}

@MainActor
final class LandmarkClassifier: ObservableObject {

    public struct Prediction: Equatable {
        public let label: String
        public let confidence: Float
        /// Full softmax distribution keyed by label name. Useful if the
        /// view wants to show top-3 instead of just top-1.
        public let allConfidences: [String: Float]
    }

    @Published public private(set) var topPrediction: Prediction?
    @Published public private(set) var isModelLoaded: Bool = false
    @Published public private(set) var lastError: String?
    @Published public private(set) var variant: LandmarkModelVariant

    private var model: MLModel?

    public init(variant: LandmarkModelVariant = .v2) {
        self.variant = variant
        loadModel()
    }

    /// Swap to a different bundled variant at runtime. Clears the current
    /// `topPrediction` so the UI doesn't show a stale value while the new
    /// model spins up.
    public func setVariant(_ newVariant: LandmarkModelVariant) {
        guard newVariant != variant else { return }
        variant = newVariant
        topPrediction = nil
        model = nil
        isModelLoaded = false
        loadModel()
    }

    private func loadModel() {
        // The .mlpackage is XcodeGen-bundled into the app at build time.
        // Look for the compiled .mlmodelc first (what Xcode emits on
        // build), then fall back to the raw .mlpackage path for dev
        // builds where the compiler has different conventions.
        let name = variant.resourceName
        if let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc")
            ?? Bundle.main.url(forResource: name, withExtension: "mlpackage") {
            do {
                let config = MLModelConfiguration()
                config.computeUnits = .all
                let m = try MLModel(contentsOf: url, configuration: config)
                self.model = m
                self.isModelLoaded = true
                self.lastError = nil
                NSLog("LandmarkClassifier: loaded \(url.lastPathComponent) (variant=\(variant.rawValue))")
                return
            } catch {
                self.lastError = "Model load failed: \(error.localizedDescription)"
                NSLog("LandmarkClassifier: load failed for \(url.lastPathComponent): \(error)")
            }
        }
        self.isModelLoaded = false
        self.lastError = "\(name).mlpackage not found in app bundle."
    }

    /// Run the model on a [1, 32, 75, 3] MLMultiArray. Updates
    /// `topPrediction` on completion.
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
            // The classifier_config in convert_to_coreml.py emits two
            // outputs: classLabel (top-1 string) and classLabel_probs
            // (dict<String, Double>). Read both.
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
