// CameraManager.swift
//
// AVCaptureSession wrapper that publishes the device's environment state
// (thermal, lighting, permission, errors) plus delivers per-frame
// CVPixelBuffers via a callback.
//
// What this manager owns:
//   - AVCaptureSession lifecycle (configure / start / stop)
//   - Front-camera selection at the configured FPS
//   - Light-level monitoring via AVCaptureDevice.iso (proxy: ISO above
//     `lowLightISO` flips `isLowLight` on)
//   - ProcessInfo thermal-state observation
//   - Memory pressure observation via DispatchSource.makeMemoryPressureSource
//   - Camera permission state
//
// What it does NOT own:
//   - Frame buffering: see FrameBuffer.swift.
//   - Inference: see PredictionEngine.swift.
//   - UI: callers attach the session to an AVCaptureVideoPreviewLayer
//     themselves (the Phase 0.3 SanityCheckView shows the pattern).
//
// Threading: configure / start / stop run on `sessionQueue`. Callbacks
// (frame delivery, state changes) are dispatched to the main actor so
// SwiftUI consumers don't have to hop themselves.
import AVFoundation
import Combine
import Foundation
import UIKit

@MainActor
public final class CameraManager: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {

    public enum CameraStatus: Equatable {
        case unconfigured
        case authorized
        case denied
        case unavailable(reason: String)
        case running
    }

    public enum MemoryPressure: String { case normal, warning, critical }

    // ── Published state — bind from SwiftUI ──────────────────────────
    @Published public private(set) var status: CameraStatus = .unconfigured
    @Published public private(set) var thermalState: ProcessInfo.ThermalState = ProcessInfo.processInfo.thermalState
    @Published public private(set) var isLowLight: Bool = false
    @Published public private(set) var memoryPressure: MemoryPressure = .normal
    @Published public private(set) var lastError: String?

    nonisolated public let session = AVCaptureSession()

    // ── Internals ─────────────────────────────────────────────────────
    nonisolated private let sessionQueue = DispatchQueue(label: "signreader.camera.session")
    nonisolated private let outputQueue = DispatchQueue(label: "signreader.camera.output", qos: .userInitiated)
    // Frame handler is read from the output queue (non-main) on every
    // captured frame. Stored under a lock so callers can swap it from any
    // thread without ARC tearing. `nonisolated(unsafe)` because the lock
    // is the actual concurrency guard, not the actor.
    nonisolated private let frameHandlerLock = NSLock()
    nonisolated(unsafe) private var _onFrame: ((CVPixelBuffer) -> Void)?
    private var memoryPressureSource: DispatchSourceMemoryPressure?
    private var thermalObserver: NSObjectProtocol?

    /// ISO threshold above which we flag the scene as "low light". Tuned for
    /// the iPhone 15 Pro front camera; the absolute number matters less than
    /// the user-visible threshold pattern (app warns once per session if
    /// scene is consistently above this).
    private let lowLightISO: Float = 1000

    public override init() {
        super.init()
        observeThermal()
        observeMemoryPressure()
    }

    deinit {
        if let obs = thermalObserver {
            NotificationCenter.default.removeObserver(obs)
        }
        memoryPressureSource?.cancel()
    }

    // ── Public API ────────────────────────────────────────────────────

    /// Request authorisation, configure the pipeline, start streaming.
    /// `fps` clamped to [10, 60]. `onFrame` runs on the output queue —
    /// don't do UI work inside it; hand off to the main actor instead.
    public func start(fps: Int = 30, onFrame: @escaping (CVPixelBuffer) -> Void) {
        let clampedFPS = max(10, min(fps, 60))
        setFrameHandler(onFrame)
        // Idempotent: refreshing the frame handler is fine, but if a
        // session is already running or in the middle of configuring, do
        // not enqueue a second configureAndRun on the session queue —
        // that path tears down inputs/outputs and triggers visible lag.
        if status == .running || status == .authorized { return }
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            status = .authorized
            configureAndRun(fps: clampedFPS)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                Task { @MainActor in
                    guard let self else { return }
                    self.status = granted ? .authorized : .denied
                    if granted { self.configureAndRun(fps: clampedFPS) }
                }
            }
        default:
            status = .denied
        }
    }

    public func stop() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            self.session.stopRunning()
            Task { @MainActor in
                self.status = .unconfigured
            }
        }
    }

    /// Replace the per-frame callback without restarting the session.
    /// Safe to call from any thread.
    public nonisolated func setFrameHandler(_ onFrame: @escaping (CVPixelBuffer) -> Void) {
        frameHandlerLock.lock()
        _onFrame = onFrame
        frameHandlerLock.unlock()
    }

    // ── Configuration ─────────────────────────────────────────────────

    private func configureAndRun(fps: Int) {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            self.session.beginConfiguration()
            self.session.sessionPreset = .vga640x480

            for input in self.session.inputs { self.session.removeInput(input) }
            for output in self.session.outputs { self.session.removeOutput(output) }

            guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front) else {
                self.session.commitConfiguration()
                Task { @MainActor in self.status = .unavailable(reason: "No front camera.") }
                return
            }

            do {
                let input = try AVCaptureDeviceInput(device: device)
                if self.session.canAddInput(input) { self.session.addInput(input) }
            } catch {
                self.session.commitConfiguration()
                Task { @MainActor in self.lastError = error.localizedDescription; self.status = .unavailable(reason: error.localizedDescription) }
                return
            }

            // Lock FPS within the device's supported range.
            do {
                try device.lockForConfiguration()
                let target = CMTime(value: 1, timescale: CMTimeScale(fps))
                if let supported = device.activeFormat.videoSupportedFrameRateRanges.first {
                    let clampedFPS = max(supported.minFrameRate, min(Double(fps), supported.maxFrameRate))
                    let t = CMTime(value: 1, timescale: CMTimeScale(clampedFPS))
                    device.activeVideoMinFrameDuration = t
                    device.activeVideoMaxFrameDuration = t
                } else {
                    device.activeVideoMinFrameDuration = target
                    device.activeVideoMaxFrameDuration = target
                }
                device.unlockForConfiguration()
            } catch {
                Task { @MainActor in self.lastError = "Could not pin FPS: \(error.localizedDescription)" }
            }

            // Watch ISO for the low-light proxy. We can't subscribe to a
            // notification, so PredictionEngine pulls the device.iso each
            // frame (via the captureOutput delegate) and we flip the flag
            // there.
            Task { @MainActor in self.observeISO(device: device) }

            let output = AVCaptureVideoDataOutput()
            output.alwaysDiscardsLateVideoFrames = true
            output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            output.setSampleBufferDelegate(self, queue: self.outputQueue)
            if self.session.canAddOutput(output) { self.session.addOutput(output) }
            // Front camera is mirrored — request that the output buffer is
            // mirrored so the prediction sees the same image as the user.
            if let conn = output.connection(with: .video) {
                if conn.isVideoMirroringSupported { conn.automaticallyAdjustsVideoMirroring = false; conn.isVideoMirrored = true }
                if conn.isVideoOrientationSupported { conn.videoOrientation = .portrait }
            }

            self.session.commitConfiguration()
            self.session.startRunning()
            Task { @MainActor in self.status = .running }
        }
    }

    private var isoObservation: NSKeyValueObservation?
    private func observeISO(device: AVCaptureDevice) {
        isoObservation?.invalidate()
        isoObservation = device.observe(\.iso, options: [.new]) { [weak self] _, change in
            guard let self, let new = change.newValue else { return }
            let lowLight = new > self.lowLightISO
            Task { @MainActor in
                if self.isLowLight != lowLight { self.isLowLight = lowLight }
            }
        }
    }

    // ── Thermal + memory pressure observers ───────────────────────────

    private func observeThermal() {
        thermalObserver = NotificationCenter.default.addObserver(
            forName: ProcessInfo.thermalStateDidChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor in
                self?.thermalState = ProcessInfo.processInfo.thermalState
            }
        }
    }

    private func observeMemoryPressure() {
        let source = DispatchSource.makeMemoryPressureSource(eventMask: [.warning, .critical], queue: .main)
        source.setEventHandler { [weak self] in
            guard let self else { return }
            let event = source.data
            Task { @MainActor in
                if event.contains(.critical) {
                    self.memoryPressure = .critical
                } else if event.contains(.warning) {
                    self.memoryPressure = .warning
                } else {
                    self.memoryPressure = .normal
                }
            }
        }
        source.resume()
        memoryPressureSource = source
    }

    // ── Frame delivery ────────────────────────────────────────────────

    nonisolated public func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        // Run the handler synchronously on the output queue (background).
        // Hopping every frame to the main actor used to choke the UI under
        // thermal pressure — preview layer would freeze because the main
        // thread was busy with Core Image resizes.
        frameHandlerLock.lock()
        let handler = _onFrame
        frameHandlerLock.unlock()
        handler?(pixelBuffer)
    }
}
