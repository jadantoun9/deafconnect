// SanityCheckView.swift — Phase 0.3 deployment dry-run, iOS half.
//
// What this view is for:
//   We need to confirm — before we invest weeks of training — that the full
//   PyTorch -> Core ML -> Xcode -> iPhone -> Neural Engine pipeline works.
//   Phase 0.3 trains a tiny CIFAR-10 classifier (`research/sanity/`), converts
//   it, and the owner copies the resulting `SanityModel.mlpackage` into
//   `ios-app/SignReader/Resources/`. This view loads it and runs:
//
//     1. A static-image classification (proves model loading works).
//     2. A live camera classification (proves the runtime pipeline works).
//
// What it intentionally is NOT:
//   - It does *not* import the Phase 0.8 CameraManager / FrameBuffer /
//     PredictionEngine, because those are shared infra for sign-language
//     models and Phase 0.3 has to be runnable before 0.8 lands.
//   - The camera handling here is intentionally minimal — single front-facing
//     stream, throttled inference, no buffering. Once Phase 0.8 is built,
//     SanityCheckView becomes a debug-only screen behind a tab.
//
// To make this view useful as a gate, the owner must:
//   - Drag SanityModel.mlpackage into `ios-app/SignReader/Resources/` (it's
//     gitignored on purpose; checkpoints don't belong in git).
//   - Build to a real iPhone (not the simulator — Neural Engine is hardware).
//   - Profile in Xcode Instruments -> Core ML, confirm ANE usage, and record
//     the latency in `docs/toolchain.md`.

import AVFoundation
import CoreML
import QuartzCore
import SwiftUI
import UIKit
import Vision

struct SanityCheckView: View {
    @StateObject private var classifier = SanityClassifier()
    @StateObject private var camera = SanityCamera()
    // Used to pause the shared AppState camera while this view is open.
    // SanityCamera and AppState.camera both want exclusive access to the
    // front lens, so they cannot run concurrently.
    @EnvironmentObject private var app: AppState

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                statusBanner

                ZStack {
                    if camera.isAuthorized {
                        SanityCameraPreview(session: camera.session)
                            .aspectRatio(3.0 / 4.0, contentMode: .fit)
                            .cornerRadius(12)
                    } else {
                        cameraPlaceholder
                            .aspectRatio(3.0 / 4.0, contentMode: .fit)
                    }
                }

                predictionPanel

                HStack {
                    Button("Classify static image") {
                        classifier.classifyStaticImage()
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!classifier.isModelLoaded)

                    Spacer()

                    Picker("Compute", selection: $classifier.preferredComputeUnits) {
                        Text("All").tag(MLComputeUnits.all)
                        Text("CPU+ANE").tag(MLComputeUnits.cpuAndNeuralEngine)
                        Text("CPU").tag(MLComputeUnits.cpuOnly)
                    }
                    .pickerStyle(.segmented)
                    .frame(maxWidth: 220)
                    .onChange(of: classifier.preferredComputeUnits) { _ in
                        classifier.reloadModel()
                    }
                }
                .padding(.horizontal)

                Spacer()
            }
            .padding()
            .navigationTitle("Sanity check")
            .onAppear {
                // Yield the front camera from the shared AppState session
                // so SanityCamera can claim it.
                app.stopCamera()
                classifier.loadModel()
                camera.start { sampleBuffer in
                    classifier.classify(sampleBuffer: sampleBuffer)
                }
            }
            .onDisappear {
                camera.stop()
                // Hand the front camera back to the production app.
                app.startCamera()
            }
        }
    }

    @ViewBuilder
    private var statusBanner: some View {
        if !classifier.isModelLoaded {
            Label(classifier.statusMessage, systemImage: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
                .font(.footnote)
                .multilineTextAlignment(.center)
                .padding(8)
                .frame(maxWidth: .infinity)
                .background(.orange.opacity(0.12))
                .cornerRadius(8)
        } else {
            Label("Model loaded — \(classifier.modelDescription)", systemImage: "checkmark.seal.fill")
                .foregroundStyle(.green)
                .font(.footnote)
                .padding(8)
                .frame(maxWidth: .infinity)
                .background(.green.opacity(0.12))
                .cornerRadius(8)
        }
    }

    @ViewBuilder
    private var cameraPlaceholder: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 12).fill(.gray.opacity(0.2))
            VStack(spacing: 8) {
                Image(systemName: "camera.fill").font(.largeTitle)
                Text(camera.statusMessage).font(.footnote).multilineTextAlignment(.center)
            }
            .padding()
        }
    }

    @ViewBuilder
    private var predictionPanel: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Prediction").font(.headline)
            HStack {
                Text(classifier.topLabel.isEmpty ? "—" : classifier.topLabel)
                    .font(.title2.bold())
                Spacer()
                Text(classifier.topConfidence > 0
                    ? String(format: "%.1f%%", classifier.topConfidence * 100)
                    : "")
                    .font(.title3)
                    .foregroundStyle(.secondary)
            }
            ProgressView(value: classifier.topConfidence)
                .progressViewStyle(.linear)
        }
        .padding()
        .background(.secondary.opacity(0.1))
        .cornerRadius(12)
    }
}

// MARK: - Classifier

@MainActor
final class SanityClassifier: ObservableObject {
    @Published var topLabel: String = ""
    @Published var topConfidence: Double = 0
    @Published var isModelLoaded: Bool = false
    @Published var statusMessage: String = "Model not loaded yet."
    @Published var modelDescription: String = ""
    @Published var preferredComputeUnits: MLComputeUnits = .all

    private var visionModel: VNCoreMLModel?
    private var lastInferenceAt: TimeInterval = 0
    // Cap to ~10 inferences per second on the live stream — the goal is a
    // smoke test, not a benchmark. Real per-track views run uncapped.
    private let inferenceIntervalSeconds: TimeInterval = 0.1

    func loadModel() {
        // The .mlpackage is git-ignored on purpose; the owner copies it in
        // after running the converter. Look for it under Resources/ first
        // (the Phase 0 layout) and as a top-level bundle resource as a
        // fallback in case Xcode's compile-resources step relocates it.
        let candidates = [
            "SanityModel",
        ]

        for name in candidates {
            if let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc")
                ?? Bundle.main.url(forResource: name, withExtension: "mlpackage") {
                if loadFromURL(url) { return }
            }
        }

        isModelLoaded = false
        statusMessage = "SanityModel.mlpackage not found in app bundle. Run convert_sanity.py and drag the .mlpackage into ios-app/SignReader/Resources/."
    }

    func reloadModel() {
        // Reload to re-bind compute-unit preference. Core ML resolves the
        // compute units at MLModel initialisation; you can't change them on
        // a live MLModel instance.
        loadModel()
    }

    private func loadFromURL(_ url: URL) -> Bool {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = preferredComputeUnits
            let mlModel = try MLModel(contentsOf: url, configuration: config)
            self.visionModel = try VNCoreMLModel(for: mlModel)
            self.isModelLoaded = true
            self.statusMessage = ""
            self.modelDescription = mlModel.modelDescription.metadata[.description] as? String ?? "Sanity model"
            return true
        } catch {
            self.isModelLoaded = false
            self.statusMessage = "Failed to load model: \(error.localizedDescription)"
            return false
        }
    }

    func classify(sampleBuffer: CMSampleBuffer) {
        guard let visionModel else { return }
        let now = CACurrentMediaTime()
        guard now - lastInferenceAt > inferenceIntervalSeconds else { return }
        lastInferenceAt = now

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let request = VNCoreMLRequest(model: visionModel) { [weak self] request, _ in
            guard let self else { return }
            guard let results = request.results as? [VNClassificationObservation], let top = results.first else { return }
            DispatchQueue.main.async {
                self.topLabel = top.identifier
                self.topConfidence = Double(top.confidence)
            }
        }
        // Front camera is mirrored; orient correctly for classification.
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .leftMirrored)
        DispatchQueue.global(qos: .userInitiated).async {
            try? handler.perform([request])
        }
    }

    func classifyStaticImage() {
        // Use a system symbol rendered to a UIImage as the static input. We
        // don't ship a test image asset because the sanity gate is just
        // "did the inference path run?" — the result need not be correct.
        guard let cg = makeStaticCGImage() else {
            statusMessage = "Could not build static test image."
            return
        }
        guard let visionModel else { return }
        let request = VNCoreMLRequest(model: visionModel) { [weak self] request, _ in
            guard let self else { return }
            guard let results = request.results as? [VNClassificationObservation], let top = results.first else { return }
            DispatchQueue.main.async {
                self.topLabel = "[static] \(top.identifier)"
                self.topConfidence = Double(top.confidence)
            }
        }
        let handler = VNImageRequestHandler(cgImage: cg, options: [:])
        DispatchQueue.global(qos: .userInitiated).async {
            try? handler.perform([request])
        }
    }

    private func makeStaticCGImage() -> CGImage? {
        let size = CGSize(width: 96, height: 96)
        UIGraphicsBeginImageContextWithOptions(size, true, 1)
        defer { UIGraphicsEndImageContext() }
        UIColor.systemBlue.setFill()
        UIRectFill(CGRect(origin: .zero, size: size))
        let img = UIGraphicsGetImageFromCurrentImageContext()
        return img?.cgImage
    }
}

// MARK: - Camera plumbing (intentionally minimal — replaced by 0.8 CameraManager)

@MainActor
final class SanityCamera: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    @Published var isAuthorized: Bool = false
    @Published var statusMessage: String = "Camera not started."

    private let queue = DispatchQueue(label: "sanity.camera.queue")
    private var onFrame: ((CMSampleBuffer) -> Void)?

    func start(onFrame: @escaping (CMSampleBuffer) -> Void) {
        self.onFrame = onFrame
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            isAuthorized = true
            configureAndRun()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                Task { @MainActor in
                    guard let self else { return }
                    self.isAuthorized = granted
                    if granted { self.configureAndRun() }
                    else { self.statusMessage = "Camera access denied." }
                }
            }
        default:
            isAuthorized = false
            statusMessage = "Camera access denied. Enable it in Settings."
        }
    }

    func stop() {
        queue.async { [weak self] in
            self?.session.stopRunning()
        }
    }

    private func configureAndRun() {
        queue.async { [weak self] in
            guard let self else { return }
            self.session.beginConfiguration()
            self.session.sessionPreset = .vga640x480
            for input in self.session.inputs { self.session.removeInput(input) }
            for output in self.session.outputs { self.session.removeOutput(output) }

            guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front) else {
                Task { @MainActor in self.statusMessage = "No front camera available." }
                self.session.commitConfiguration()
                return
            }
            do {
                let input = try AVCaptureDeviceInput(device: device)
                if self.session.canAddInput(input) { self.session.addInput(input) }
            } catch {
                Task { @MainActor in self.statusMessage = "Camera init failed: \(error.localizedDescription)" }
                self.session.commitConfiguration()
                return
            }

            let output = AVCaptureVideoDataOutput()
            output.alwaysDiscardsLateVideoFrames = true
            output.setSampleBufferDelegate(self, queue: self.queue)
            if self.session.canAddOutput(output) { self.session.addOutput(output) }

            self.session.commitConfiguration()
            self.session.startRunning()
            Task { @MainActor in self.statusMessage = "" }
        }
    }

    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // Hand off on the main actor so the @Published classifier stays consistent.
        let buffer = sampleBuffer
        Task { @MainActor in
            self.onFrame?(buffer)
        }
    }
}

private struct SanityCameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewView {
        let v = PreviewView()
        v.videoPreviewLayer.session = session
        v.videoPreviewLayer.videoGravity = .resizeAspectFill
        return v
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {}

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var videoPreviewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }
}

#Preview {
    SanityCheckView()
}
