// SignToTextView.swift
//
// Live sign-to-word recognition. Uses the existing CameraManager (front
// camera) for frame capture, runs Apple Vision body+hand pose per
// frame, accumulates 32 frames in a ring buffer, and runs the bundled
// LandmarkClassifier9 Core ML model at ~5 Hz.
//
// Why Apple Vision rather than MediaPipe Tasks: native, no external
// dep, and the 9-class demo signs (drink/go/help/who/yes/no/before/walk
// /idle) are mostly hand-driven — Vision's 21-point hand layout maps
// 1-for-1 to MediaPipe's, with body's missing 14 keypoints zero-filled
// (which the training distribution already contains plenty of, since
// MediaPipe Holistic also drops keypoints on noisy inputs).
import AVFoundation
import SwiftUI

struct SignToTextView: View {
    @EnvironmentObject private var app: AppState
    @StateObject private var classifier = LandmarkClassifier(variant: .v2)
    @StateObject private var extractor = LandmarkExtractor()
    @StateObject private var videoClassifier = VideoClassifier()
    @StateObject private var letterClassifier = LetterClassifier()
    @State private var lastInferenceAt: Date = .distantPast
    @State private var lastVideoFrameAt: Date = .distantPast
    @State private var frameCount: Int = 0
    /// User-facing pick: either the ASL alphabet (single-frame, single-
    /// hand) or "Words" — which underlying classifier "Words" routes to
    /// (LM v1, LM v2, or the video Transformer) is set in Settings.
    @State private var selectedModel: ActiveModel = .words
    /// Capture start; used as the base for MediaPipe's monotonic
    /// timestamp argument (millis since session start).
    @State private var captureStart: Date = .now

    /// Suppress the "idle" class from the user-visible prediction strip
    /// — it's the model's "no sign" signal, not a real prediction. We
    /// keep it under the hood so its softmax mass doesn't bleed into the
    /// other classes.
    private let suppressedLabel: String = "idle"
    /// Don't surface predictions below this softmax confidence — they're
    /// noise before the buffer is fully primed or during transitions
    /// between signs.
    private let confidenceThreshold: Float = 0.50

    var body: some View {
        NavigationStack {
            ZStack {
                cameraLayer
                overlay
            }
            .navigationTitle("Sign to Text")
            .navigationBarTitleDisplayMode(.inline)
            // Camera content ignores safe areas to fill the screen, which
            // leaves the system bars transparent unless we force them
            // visible — match the opaque chrome on the other tabs.
            .toolbarBackground(.visible, for: .navigationBar)
            .toolbarBackground(.visible, for: .tabBar)
            .withSettingsLauncher()
            .onAppear { startCapture() }
            .onDisappear { stopCapture() }
            .onChange(of: selectedModel) { _, _ in rebindAndReset() }
            .onChange(of: app.wordsModelVariant) { _, _ in
                // Only flush if the active pick actually uses the words
                // model; switching the underlying variant while on the
                // Letters tab is a no-op.
                if selectedModel == .words { rebindAndReset() }
            }
        }
    }

    /// Switching model class flushes both ring buffers — the landmark
    /// and video pipelines hold completely independent input shapes,
    /// so frames captured under one mode aren't reusable for the other.
    private func rebindAndReset() {
        switch runningModel {
        case .landmarkV1: classifier.setVariant(.v1)
        case .landmarkV2: classifier.setVariant(.v2)
        case .video, .letters: break
        }
        extractor.reset()
        videoClassifier.reset()
        letterClassifier.reset()
        frameCount = 0
    }

    /// Resolves the user's two-way pick (Letter / Words) plus the
    /// settings-side variant into the concrete pipeline that should
    /// run this frame.
    private var runningModel: RunningModel {
        switch selectedModel {
        case .letters: return .letters
        case .words:
            switch app.wordsModelVariant {
            case .landmarkV1: return .landmarkV1
            case .landmarkV2: return .landmarkV2
            case .video:      return .video
            }
        }
    }

    // ── Camera layer ───────────────────────────────────────────────────

    private var cameraLayer: some View {
        ZStack {
            CameraPreview(session: app.camera.session)
            // Live MediaPipe landmark dots + skeleton lines drawn over
            // the camera feed. CameraManager already mirrors the output
            // connection (isVideoMirrored = true) so MediaPipe sees the
            // same mirrored frame the user sees — no extra flip in the
            // overlay or the dots end up on the wrong side of the body.
            LandmarksOverlayView(extractor: extractor, mirrorHorizontally: false)
        }
        .ignoresSafeArea()
    }

    // ── Overlays ──────────────────────────────────────────────────────

    private var overlay: some View {
        VStack {
            predictionStrip
            modelVariantPicker
            Spacer()
            footerHints
        }
    }

    /// Segmented control: just "Letter" vs "Words". Which underlying
    /// classifier powers "Words" (LM v1 / LM v2 / Video) is hidden
    /// behind a Settings picker — see `AppState.wordsModelVariant`.
    private var modelVariantPicker: some View {
        VStack(spacing: 6) {
            Text("Model")
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.white.opacity(0.7))
            Picker("Model", selection: $selectedModel) {
                ForEach(ActiveModel.allCases) { m in
                    Text(m.shortLabel).tag(m)
                }
            }
            .pickerStyle(.segmented)
            .colorScheme(.dark)
            .padding(.horizontal, 16)
        }
        .padding(.vertical, 10)
        .frame(maxWidth: .infinity)
        .background(.black.opacity(0.45))
    }

    private var predictionStrip: some View {
        VStack(spacing: 4) {
            if !modelLoaded {
                Label("Model not loaded", systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.white)
                    .padding(.horizontal, 14).padding(.vertical, 8)
                    .background(.red.opacity(0.85), in: Capsule())
            } else if let pred = activePrediction {
                Text(pred.label.uppercased())
                    .font(.system(size: 36, weight: .heavy, design: .rounded))
                    .foregroundStyle(.white)
                Text(String(format: "%.0f%%", pred.confidence * 100))
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.white.opacity(0.85))
            } else {
                Text("—")
                    .font(.system(size: 36, weight: .heavy, design: .rounded))
                    .foregroundStyle(.white.opacity(0.65))
                Text(statusText)
                    .font(.footnote)
                    .foregroundStyle(.white.opacity(0.7))
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 14)
        .background(.black.opacity(0.45))
    }

    private var footerHints: some View {
        VStack(spacing: 4) {
            Text("Recognised vocab")
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.white.opacity(0.7))
            Text(recognisedVocabText)
                .font(.caption.weight(.medium))
                .foregroundStyle(.white.opacity(0.85))
                .multilineTextAlignment(.center)
                .padding(.horizontal, 16)
        }
        .padding(.vertical, 10)
        .frame(maxWidth: .infinity)
        .background(.black.opacity(0.45))
    }

    private var recognisedVocabText: String {
        switch runningModel {
        case .landmarkV1, .landmarkV2, .video:
            return "drink · go · help · who · yes · no · before · walk"
        case .letters:
            return "A · B · C · D · E · F · G · H · I · J · K · L · M\nN · O · P · Q · R · S · T · U · V · W · X · Y · Z"
        }
    }

    // ── Derived state ──────────────────────────────────────────────────

    /// Adapter so the prediction strip can read either the landmark or
    /// video classifier without caring which is active.
    private struct UnifiedPrediction {
        let label: String
        let confidence: Float
    }

    /// What we actually show on screen — top prediction from whichever
    /// model is active, after dropping the idle class and weak picks.
    private var activePrediction: UnifiedPrediction? {
        let raw: UnifiedPrediction? = {
            switch runningModel {
            case .landmarkV1, .landmarkV2:
                guard let p = classifier.topPrediction else { return nil }
                return UnifiedPrediction(label: p.label, confidence: p.confidence)
            case .video:
                guard let p = videoClassifier.topPrediction else { return nil }
                return UnifiedPrediction(label: p.label, confidence: p.confidence)
            case .letters:
                guard let p = letterClassifier.topPrediction else { return nil }
                return UnifiedPrediction(label: p.label, confidence: p.confidence)
            }
        }()
        guard let p = raw else { return nil }
        // The 'idle' class is a word-model construct; letter outputs are
        // pure A..Z so the suppression check is a no-op for letters.
        if p.label == suppressedLabel { return nil }
        if p.confidence < confidenceThreshold { return nil }
        return p
    }

    private var modelLoaded: Bool {
        switch runningModel {
        case .landmarkV1, .landmarkV2: return classifier.isModelLoaded
        case .video: return videoClassifier.isModelLoaded
        case .letters: return letterClassifier.isModelLoaded
        }
    }

    private var statusText: String {
        switch runningModel {
        case .landmarkV1, .landmarkV2:
            if let err = classifier.lastError { return err }
            if frameCount < kFrameWindow {
                return "Buffering… \(min(frameCount, kFrameWindow))/\(kFrameWindow) frames"
            }
            return "Sign one of the words below"
        case .video:
            if let err = videoClassifier.lastError { return err }
            let n = videoClassifier.bufferFilledCount
            if n < 16 { return "Buffering… \(n)/16 frames" }
            return "Sign one of the words below"
        case .letters:
            if let err = letterClassifier.lastError { return err }
            return "Sign a letter with one hand"
        }
    }

    // ── Capture lifecycle ─────────────────────────────────────────────

    private func startCapture() {
        extractor.reset()
        videoClassifier.reset()
        frameCount = 0
        captureStart = .now
        // Make sure the LandmarkClassifier matches the user's Settings
        // choice on first appear — the @StateObject default is .v2.
        rebindAndReset()

        // Order matters: app.startCamera() internally calls
        // CameraManager.start(...), which always re-binds the camera's
        // frame handler to engine.feed. Start the session first and
        // override the handler immediately after.
        app.startCamera()
        let start = captureStart
        app.camera.setFrameHandler { pixelBuffer in
            // Branch on the active model: only run the pipeline that's
            // actually being used. Doing both in parallel costs CPU
            // (MediaPipe is ~5–15 ms/frame, video resize is ~3–5 ms)
            // and the user only ever sees one prediction.
            switch runningModel {
            case .landmarkV1, .landmarkV2:
                let tsMillis = Int(Date().timeIntervalSince(start) * 1000)
                extractor.extract(pixelBuffer: pixelBuffer, timestampMillis: tsMillis)

                Task { @MainActor in
                    frameCount += 1
                    let now = Date()
                    // ~5 Hz inference; MediaPipe runs at camera FPS.
                    guard now.timeIntervalSince(lastInferenceAt) > 0.2 else { return }
                    guard let input = extractor.snapshotInputTensor(
                        subset: classifier.variant.keypointSubset
                    ) else { return }
                    lastInferenceAt = now
                    await classifier.predict(input: input)
                }

            case .video:
                // Track 2: throttle frame ingestion to ~10 Hz so the
                // 16-frame buffer spans ~1.6 s (matches the per-clip
                // duration the model was trained on).
                let now = Date()
                if now.timeIntervalSince(lastVideoFrameAt) >= 0.1 {
                    videoClassifier.appendFrame(pixelBuffer)
                    Task { @MainActor in
                        lastVideoFrameAt = now
                    }
                }

                Task { @MainActor in
                    // ~1 Hz inference for video — each forward is much
                    // heavier than the landmark path (16 ViT passes per
                    // call). 1 Hz keeps the strip responsive without
                    // pinning the ANE.
                    let t = Date()
                    guard t.timeIntervalSince(lastInferenceAt) > 1.0 else { return }
                    guard videoClassifier.bufferFilledCount >= 16 else { return }
                    lastInferenceAt = t
                    await videoClassifier.predict()
                }

            case .letters:
                // ASL alphabet: single-frame, single-hand. Reuse the
                // MediaPipe extractor so the dots overlay still draws,
                // then read the latest hand keypoints directly (raw
                // coords; matches the Keras training pipeline).
                let tsMillis = Int(Date().timeIntervalSince(start) * 1000)
                extractor.extract(pixelBuffer: pixelBuffer, timestampMillis: tsMillis)

                Task { @MainActor in
                    let now = Date()
                    // ~5 Hz inference — same cadence as the word
                    // landmark path. The model is tiny (170k params)
                    // so this is comfortable on the ANE.
                    guard now.timeIntervalSince(lastInferenceAt) > 0.2 else { return }
                    guard let input = extractor.snapshotHandFrame() else {
                        // No hand detected this frame — clear the strip
                        // so a stale letter doesn't linger on screen.
                        letterClassifier.reset()
                        return
                    }
                    lastInferenceAt = now
                    await letterClassifier.predict(input: input)
                }
            }
        }
    }

    private func stopCapture() {
        // Restore the placeholder engine's frame handler. Without this
        // the camera would keep running while we're off-screen with our
        // (now stale) handler still attached. Calling app.startCamera
        // is idempotent — it just re-binds the engine handler if the
        // session is already running.
        app.startCamera()
    }
}

/// The two top-level choices exposed in the Sign-to-Text picker.
/// "Words" hides three underlying classifiers (LM v1, LM v2, video
/// Transformer) behind a single button — the actual variant is picked
/// in Settings (`AppState.wordsModelVariant`).
enum ActiveModel: String, CaseIterable, Identifiable {
    case letters
    case words

    var id: String { rawValue }

    /// Short label for the segmented control.
    var shortLabel: String {
        switch self {
        case .letters: return "Letter"
        case .words:   return "Words"
        }
    }
}

/// Resolved internal pipeline — i.e. what's actually fed pixel buffers
/// this frame. Computed from `ActiveModel` + `WordsModelVariant`; all
/// downstream branching switches over this so callers don't have to
/// know about the settings indirection.
enum RunningModel {
    case landmarkV1
    case landmarkV2
    case video
    case letters
}
