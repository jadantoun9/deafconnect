// ModelLoader.swift
//
// Centralised dispatch from a ModelKind to a concrete SignModel instance.
// Until Track 1/2/3 implementations land, every kind resolves to a
// PlaceholderModel that returns "(not loaded)" predictions, so the shell
// boots cleanly and SettingsView's model toggle works.
//
// When each track lands:
//   .landmark  -> LandmarkModel.swift   (Track 1.4)
//   .endToEnd  -> EndToEndModel.swift   (Track 2.6)
//   .distilled -> DistilledModel.swift  (Track 3.5)
// Replace the corresponding case in `load(kind:)`.
import CoreVideo
import Foundation

public enum ModelLoader {
    public static func load(kind: ModelKind) -> SignModel {
        switch kind {
        case .landmark:  return PlaceholderModel(kind: .landmark)
        case .endToEnd:  return PlaceholderModel(kind: .endToEnd)
        case .distilled: return PlaceholderModel(kind: .distilled)
        }
    }
}

/// SignModel that always returns "(not loaded)". The Phase 0.8 shell uses
/// this so the camera + UI work end-to-end before any real model exists.
public final class PlaceholderModel: SignModel {
    public let kind: ModelKind
    public let inputFrameCount: Int = 16
    public let inputResolution: Int = 224
    public let defaultConfidenceThreshold: Float = 0.0
    public let isActive: Bool = false

    public init(kind: ModelKind) { self.kind = kind }

    public func predict(frames: [CVPixelBuffer]) async throws -> Prediction {
        // Pretend to spend a few ms so the engine's `isInferring` indicator
        // flickers rather than staying constantly on. No real compute.
        try? await Task.sleep(nanoseconds: 5_000_000)
        return Prediction(
            label: "(not loaded)",
            confidence: 0.0,
            topK: [],
            latencyMs: 5,
            timestamp: Date()
        )
    }
}
