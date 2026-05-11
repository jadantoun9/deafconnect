// VideoClassifier.swift
//
// Wraps the bundled VideoClassifier8.mlpackage (Track 2 baseline:
// ViT-Small + 3-layer temporal Transformer) for live sign recognition.
//
// Pipeline differs from LandmarkClassifier:
//   - Per camera frame, resize the CVPixelBuffer to 224x224 RGB and
//     ImageNet-normalize. Push the float32 tensor [3, 224, 224] into a
//     16-frame ring buffer.
//   - On predict(), build a [1, 16, 3, 224, 224] MLMultiArray from the
//     ring and run the model.
//
// Preprocessing must mirror research/data/{video_dataset,augmentations}.py:
//   pixel_u8 / 255.0 → (x - mean) / std    (ImageNet stats)
// Diverging here puts the model out of distribution.
import Accelerate
import CoreImage
import CoreML
import CoreVideo
import Foundation

private let kVideoFrameWindow: Int = 16
private let kVideoSize: Int = 224
private let kVideoChannels: Int = 3
private let kVideoFrameFloats: Int = kVideoChannels * kVideoSize * kVideoSize  // 150528

@MainActor
final class VideoClassifier: ObservableObject {

    public struct Prediction: Equatable {
        public let label: String
        public let confidence: Float
        public let allConfidences: [String: Float]
    }

    @Published public private(set) var topPrediction: Prediction?
    @Published public private(set) var isModelLoaded: Bool = false
    @Published public private(set) var lastError: String?

    /// True once the ring buffer has at least 16 distinct frames written.
    @Published public private(set) var bufferFilledCount: Int = 0

    private var model: MLModel?
    /// Marked nonisolated(unsafe) so the camera output queue can call
    /// `appendFrame(_:)` without hopping to MainActor for every frame.
    /// Safe because VideoFrameExtractor uses an internal NSLock to
    /// protect its ring buffer.
    private nonisolated(unsafe) let extractor = VideoFrameExtractor()

    public init() {
        loadModel()
    }

    private func loadModel() {
        let name = "VideoClassifier8"
        if let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc")
            ?? Bundle.main.url(forResource: name, withExtension: "mlpackage") {
            do {
                let config = MLModelConfiguration()
                config.computeUnits = .all
                let m = try MLModel(contentsOf: url, configuration: config)
                self.model = m
                self.isModelLoaded = true
                self.lastError = nil
                NSLog("VideoClassifier: loaded \(url.lastPathComponent)")
                return
            } catch {
                self.lastError = "Model load failed: \(error.localizedDescription)"
                NSLog("VideoClassifier: load failed: \(error)")
            }
        }
        self.isModelLoaded = false
        self.lastError = "VideoClassifier8.mlpackage not found in app bundle."
    }

    /// Push a camera frame into the ring buffer. Safe to call from any
    /// queue. The expensive resize+normalize happens here so predict()
    /// is just a Core ML forward pass.
    nonisolated public func appendFrame(_ pixelBuffer: CVPixelBuffer) {
        extractor.append(pixelBuffer: pixelBuffer)
        Task { @MainActor [weak self] in
            self?.bufferFilledCount = self?.extractor.framesWritten ?? 0
        }
    }

    /// Reset the ring buffer (e.g. on tab re-enter).
    public func reset() {
        extractor.reset()
        bufferFilledCount = 0
        topPrediction = nil
    }

    /// Run the model on the latest 16 frames. Returns nil silently if
    /// the buffer isn't yet primed.
    public func predict() async {
        guard let model else { return }
        guard let input = extractor.snapshotInput() else { return }

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
            topPrediction = Prediction(
                label: label,
                confidence: probs[label] ?? 0,
                allConfidences: probs
            )
            lastError = nil
        } catch {
            lastError = "Prediction failed: \(error.localizedDescription)"
        }
    }
}

// ── Frame ring buffer + preprocessing ─────────────────────────────────

private final class VideoFrameExtractor {
    /// Each entry holds the float32 [3, 224, 224] = 150 528 floats for
    /// one frame, already ImageNet-normalised. The MLMultiArray is built
    /// by stacking 16 of these in temporal order.
    private var ring: [[Float]] = Array(
        repeating: Array(repeating: 0, count: kVideoFrameFloats),
        count: kVideoFrameWindow
    )
    private var writeIndex: Int = 0
    private(set) var framesWritten: Int = 0
    private let lock = NSLock()

    /// CIContext re-used across frames so we don't pay the GPU/Metal
    /// init cost every push. Initialising once buys ~10 ms/frame.
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    private let outRect = CGRect(x: 0, y: 0, width: kVideoSize, height: kVideoSize)

    /// Per-channel ImageNet stats; matches research/data/augmentations.py
    /// `normalize_imagenet`.
    private let mean: [Float] = [0.485, 0.456, 0.406]
    private let std:  [Float] = [0.229, 0.224, 0.225]

    func append(pixelBuffer: CVPixelBuffer) {
        guard let frameFloats = preprocess(pixelBuffer: pixelBuffer) else { return }
        lock.lock()
        ring[writeIndex] = frameFloats
        writeIndex = (writeIndex + 1) % kVideoFrameWindow
        framesWritten = min(framesWritten + 1, kVideoFrameWindow)
        lock.unlock()
    }

    func reset() {
        lock.lock()
        for i in 0..<ring.count { ring[i] = Array(repeating: 0, count: kVideoFrameFloats) }
        writeIndex = 0
        framesWritten = 0
        lock.unlock()
    }

    func snapshotInput() -> MLMultiArray? {
        lock.lock()
        defer { lock.unlock() }
        guard framesWritten >= kVideoFrameWindow else { return nil }

        guard let array = try? MLMultiArray(
            shape: [1, NSNumber(value: kVideoFrameWindow), NSNumber(value: kVideoChannels),
                    NSNumber(value: kVideoSize), NSNumber(value: kVideoSize)],
            dataType: .float32
        ) else { return nil }

        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: kVideoFrameWindow * kVideoFrameFloats)
        for t in 0..<kVideoFrameWindow {
            let src = ring[(writeIndex + t) % kVideoFrameWindow]
            src.withUnsafeBufferPointer { buf in
                let dst = ptr.advanced(by: t * kVideoFrameFloats)
                memcpy(dst, buf.baseAddress!, kVideoFrameFloats * MemoryLayout<Float>.size)
            }
        }
        return array
    }

    // MARK: - Preprocess one CVPixelBuffer to [3, 224, 224] ImageNet floats

    private func preprocess(pixelBuffer: CVPixelBuffer) -> [Float]? {
        // Resize via CoreImage. Center-crop the source to a square
        // before scaling so non-square camera frames don't get squashed
        // (matches the eval pipeline's center_crop).
        let source = CIImage(cvPixelBuffer: pixelBuffer)
        let sw = source.extent.width
        let sh = source.extent.height
        let side = min(sw, sh)
        let dx = (sw - side) / 2
        let dy = (sh - side) / 2
        let cropped = source.cropped(to: CGRect(x: dx, y: dy, width: side, height: side))
        // Translate the crop back to origin, then scale to 224.
        let scaled = cropped
            .transformed(by: CGAffineTransform(translationX: -dx, y: -dy))
            .transformed(by: CGAffineTransform(scaleX: CGFloat(kVideoSize) / side,
                                               y: CGFloat(kVideoSize) / side))

        // Render to a planar BGRA buffer we can read pixel-by-pixel.
        let bytesPerRow = kVideoSize * 4
        var rgba = [UInt8](repeating: 0, count: kVideoSize * kVideoSize * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        rgba.withUnsafeMutableBytes { rawBuf in
            ciContext.render(scaled,
                             toBitmap: rawBuf.baseAddress!,
                             rowBytes: bytesPerRow,
                             bounds: outRect,
                             format: .RGBA8,
                             colorSpace: colorSpace)
        }

        // Convert RGBA8 → planar [3, 224, 224] float, divide by 255,
        // ImageNet-normalise per channel.
        var planar = Array<Float>(repeating: 0, count: kVideoFrameFloats)
        let plane = kVideoSize * kVideoSize  // 50 176 pixels per channel
        let invStd: [Float] = [1 / std[0], 1 / std[1], 1 / std[2]]
        let mean = self.mean
        rgba.withUnsafeBufferPointer { srcPtr in
            planar.withUnsafeMutableBufferPointer { dstPtr in
                let src = srcPtr.baseAddress!
                let dst = dstPtr.baseAddress!
                for i in 0..<plane {
                    let r = Float(src[i * 4 + 0]) / 255.0
                    let g = Float(src[i * 4 + 1]) / 255.0
                    let b = Float(src[i * 4 + 2]) / 255.0
                    dst[0 * plane + i] = (r - mean[0]) * invStd[0]
                    dst[1 * plane + i] = (g - mean[1]) * invStd[1]
                    dst[2 * plane + i] = (b - mean[2]) * invStd[2]
                }
            }
        }
        return planar
    }
}
