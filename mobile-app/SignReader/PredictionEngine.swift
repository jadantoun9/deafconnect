// PredictionEngine.swift
//
// Glue layer between CameraManager (frame producer) and SignModel
// (consumer). Owns:
//   - the FrameBuffer
//   - a single in-flight inference task (cancels the previous one if a new
//     prediction is requested before the previous returned)
//   - debounce + smoothing on the published `prediction` so the UI doesn't
//     flicker between low-confidence frames
//   - a scrolling history of accepted predictions (for LiveView's log)
//
// Concurrency model: the engine is @MainActor because its @Published
// properties bind to SwiftUI; inference work is detached to an awaitable
// Task with `await` semantics. CameraManager hands off frames on the main
// actor, so `feed` is synchronous from the caller's POV.
import Combine
import CoreVideo
import Foundation
import QuartzCore

@MainActor
public final class PredictionEngine: ObservableObject {

    // ── Published bindings ────────────────────────────────────────────
    @Published public private(set) var prediction: Prediction?
    @Published public private(set) var history: [Prediction] = []
    @Published public private(set) var isInferring: Bool = false
    @Published public private(set) var lastError: String?

    // ── Tuning knobs (mirrored from SettingsView) ─────────────────────
    @Published public var confidenceThreshold: Float
    @Published public var minIntervalSeconds: TimeInterval
    @Published public var historyLimit: Int = 20

    // ── Internals ─────────────────────────────────────────────────────
    private(set) public var model: SignModel
    // `buffer` is read from the camera-output queue via the nonisolated
    // `feed`; FrameBuffer is internally locked so this is safe.
    nonisolated private let buffer: FrameBuffer
    private var inflight: Task<Void, Never>?
    private var lastPredictionAt: TimeInterval = 0
    nonisolated private let feedThrottleLock = NSLock()
    nonisolated(unsafe) private var lastFeedAt: TimeInterval = 0
    /// Mirrors `model.isActive` for read from the nonisolated `feed`.
    /// Updated whenever `setModel` runs on the main actor.
    nonisolated(unsafe) private var modelIsActive: Bool = true
    /// Cap how often we accept frames into the buffer. 10 Hz keeps the
    /// resize cost low without starving the inference loop.
    nonisolated private let feedHz: Int = 10

    public init(model: SignModel, buffer: FrameBuffer? = nil, confidenceThreshold: Float? = nil, minIntervalSeconds: TimeInterval = 0.2) {
        self.model = model
        self.confidenceThreshold = confidenceThreshold ?? model.defaultConfidenceThreshold
        self.minIntervalSeconds = minIntervalSeconds
        self.modelIsActive = model.isActive
        // Default frame buffer sized to the model's expectations. Callers
        // can pass a larger buffer for sliding-window evaluation.
        self.buffer = buffer ?? FrameBuffer(
            capacity: max(model.inputFrameCount * 2, model.inputFrameCount),
            targetSize: CGSize(width: model.inputResolution, height: model.inputResolution)
        )
    }

    /// Swap the active model. Cancels in-flight inference and clears the
    /// frame buffer (the new model may expect a different resolution or
    /// frame count).
    public func setModel(_ next: SignModel) {
        inflight?.cancel()
        inflight = nil
        self.model = next
        self.modelIsActive = next.isActive
        self.confidenceThreshold = next.defaultConfidenceThreshold
        self.buffer.clear()
    }

    public func setConfidenceThreshold(_ v: Float) {
        self.confidenceThreshold = max(0, min(v, 1))
    }

    /// Append a frame and, if eligible, kick off an inference. Safe to call
    /// from any thread — the frame buffer is internally locked, and the
    /// scheduling decision hops to the main actor only briefly.
    ///
    /// Throttled to `feedHz` (default 10 Hz) regardless of camera FPS so
    /// the Core Image resize inside FrameBuffer.append doesn't run 30
    /// times per second. The buffer still covers ~1.5s of recent frames
    /// at 16 frames * 100ms, which is plenty for sign recognition.
    public nonisolated func feed(_ frame: CVPixelBuffer) {
        if !modelIsActive { return }
        let now = CACurrentMediaTime()
        feedThrottleLock.lock()
        let interval = 1.0 / Double(feedHz)
        if now - lastFeedAt < interval {
            feedThrottleLock.unlock()
            return
        }
        lastFeedAt = now
        feedThrottleLock.unlock()

        buffer.append(frame)
        Task { [weak self] in
            await self?.maybeStartInference()
        }
    }

    private func maybeStartInference() {
        let now = CACurrentMediaTime()
        if now - lastPredictionAt < minIntervalSeconds { return }
        if buffer.size < model.inputFrameCount { return }
        if inflight != nil { return }

        let frames = buffer.snapshot(model.inputFrameCount)
        guard frames.count == model.inputFrameCount else { return }

        lastPredictionAt = now
        isInferring = true
        let m = self.model
        let threshold = self.confidenceThreshold
        let limit = self.historyLimit

        inflight = Task.detached(priority: .userInitiated) { [weak self] in
            do {
                let p = try await m.predict(frames: frames)
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.lastError = nil
                    self.isInferring = false
                    self.inflight = nil
                    if p.confidence >= threshold {
                        self.prediction = p
                        self.history.insert(p, at: 0)
                        if self.history.count > limit {
                            self.history.removeLast(self.history.count - limit)
                        }
                    } else {
                        // Low-confidence — keep the previous accepted
                        // prediction visible rather than blanking the UI.
                    }
                }
            } catch {
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.lastError = error.localizedDescription
                    self.isInferring = false
                    self.inflight = nil
                }
            }
        }
    }

    public func clearHistory() { history.removeAll() }
}
