// ModelProtocol.swift
//
// The single interface every track-specific model conforms to. Track 1
// (Landmark) wraps MediaPipe Tasks + a Core ML classifier on the landmark
// vector; Tracks 2 (EndToEnd) and 3 (Distilled) wrap a Core ML video
// transformer. PredictionEngine, LiveView, PracticeView and BenchmarkRunner
// only ever talk to this protocol — none of them know which model they are
// driving.
//
// Why not split video models from landmark models with separate protocols:
//   The orchestration is identical. A pile of frames goes in, a Prediction
//   comes out. Splitting the protocol would force PredictionEngine to be
//   generic over an associated input type, which the SettingsView toggle
//   could not satisfy at runtime. Keeping the protocol input-agnostic
//   (frames in, prediction out) lets a single engine swap models freely.
import CoreVideo
import Foundation

public struct Prediction: Equatable, Identifiable {
    public let id: UUID
    public let label: String
    public let confidence: Float
    /// Ranked alternatives, label -> confidence; usually top 5.
    public let topK: [(label: String, confidence: Float)]
    /// Wall-clock latency of the inference call that produced this prediction.
    public let latencyMs: Double
    /// When the prediction was produced (host time, monotonic).
    public let timestamp: Date

    public init(label: String, confidence: Float, topK: [(String, Float)], latencyMs: Double, timestamp: Date = Date()) {
        self.id = UUID()
        self.label = label
        self.confidence = confidence
        self.topK = topK.map { (label: $0.0, confidence: $0.1) }
        self.latencyMs = latencyMs
        self.timestamp = timestamp
    }

    public static func == (lhs: Prediction, rhs: Prediction) -> Bool {
        lhs.id == rhs.id
    }
}

public enum ModelKind: String, CaseIterable, Identifiable {
    case landmark
    case endToEnd
    case distilled
    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .landmark:  return "Landmark (Track 1)"
        case .endToEnd:  return "End-to-End (Track 2)"
        case .distilled: return "Distilled (Track 3)"
        }
    }
}

public protocol SignModel: AnyObject {
    var kind: ModelKind { get }
    /// Number of frames each prediction call expects. PredictionEngine reads
    /// this to size the FrameBuffer and to skip prediction until the buffer
    /// has filled at least once.
    var inputFrameCount: Int { get }
    /// Side length of the spatial input (model expects square frames).
    var inputResolution: Int { get }
    /// Confidence below which PredictionEngine should suppress display.
    var defaultConfidenceThreshold: Float { get }
    /// When false, PredictionEngine skips the per-frame buffer resize and
    /// the inference loop. Used by PlaceholderModel so the shell doesn't
    /// burn CPU before any track has landed.
    var isActive: Bool { get }

    /// Run inference on `frames`, expected to be at least `inputFrameCount`
    /// long. The model decides whether to centre-crop the most recent N or
    /// uniformly sample. Throws if the underlying Core ML model is missing
    /// or fails.
    func predict(frames: [CVPixelBuffer]) async throws -> Prediction
}

public extension SignModel {
    var isActive: Bool { true }
}

/// Concrete error type — protocol uses `throws` so individual models can
/// surface their own; this is the common one wrappers should reach for.
public enum ModelError: LocalizedError {
    case modelMissing(name: String)
    case invalidInput(reason: String)
    case inferenceFailed(underlying: Error)

    public var errorDescription: String? {
        switch self {
        case .modelMissing(let name):
            return "Core ML model \(name) not found in app bundle. Convert + drag the .mlpackage into Resources/."
        case .invalidInput(let reason):
            return "Invalid input: \(reason)"
        case .inferenceFailed(let err):
            return "Inference failed: \(err.localizedDescription)"
        }
    }
}
