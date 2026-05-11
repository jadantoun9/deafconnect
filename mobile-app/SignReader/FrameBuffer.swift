// FrameBuffer.swift
//
// Thread-safe ring buffer of CVPixelBuffer. PredictionEngine writes one
// frame per camera callback and reads the most recent N when an inference
// slot opens up. The buffer also auto-resizes incoming frames to the model's
// expected square input dimensions, so each track's `predict(frames:)` can
// assume frames are exactly the right size.
//
// Resizing happens at append-time (rather than at snapshot-time) so the
// buffer holds smaller, already-formatted buffers — the dropped-frame path
// is much cheaper than re-resizing the same pixels every prediction.
import Accelerate
import CoreImage
import CoreVideo
import Foundation

public final class FrameBuffer {
    public let capacity: Int
    public let targetSize: CGSize

    private var storage: [CVPixelBuffer?] = []
    private var head: Int = 0
    private var count: Int = 0
    private let lock = NSLock()
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    public init(capacity: Int, targetSize: CGSize) {
        precondition(capacity > 0, "FrameBuffer capacity must be positive")
        self.capacity = capacity
        self.targetSize = targetSize
        self.storage = Array(repeating: nil, count: capacity)
    }

    public var isFull: Bool {
        lock.lock(); defer { lock.unlock() }
        return count >= capacity
    }

    public var size: Int {
        lock.lock(); defer { lock.unlock() }
        return count
    }

    public func clear() {
        lock.lock(); defer { lock.unlock() }
        for i in 0..<capacity { storage[i] = nil }
        head = 0
        count = 0
    }

    /// Resize + insert. Drops the oldest frame when full.
    public func append(_ frame: CVPixelBuffer) {
        let resized = resize(frame: frame, to: targetSize) ?? frame
        lock.lock(); defer { lock.unlock() }
        storage[head] = resized
        head = (head + 1) % capacity
        if count < capacity { count += 1 }
    }

    /// Returns up to `n` most recent frames, oldest first, suitable for
    /// passing as the `frames:` argument to a SignModel.
    public func snapshot(_ n: Int? = nil) -> [CVPixelBuffer] {
        lock.lock(); defer { lock.unlock() }
        let want = min(n ?? count, count)
        if want == 0 { return [] }
        var result = [CVPixelBuffer]()
        result.reserveCapacity(want)
        // Walk backwards from head, take the last `want` entries.
        let start = (head - want + capacity) % capacity
        for i in 0..<want {
            let idx = (start + i) % capacity
            if let b = storage[idx] { result.append(b) }
        }
        return result
    }

    // ── Resize helper ─────────────────────────────────────────────────
    //
    // Core Image is the most reliable path because it preserves the pixel
    // format (32BGRA from CameraManager). vImage would be faster but needs
    // explicit format negotiation per pixel layout — the marginal speedup
    // isn't worth the complexity for a buffer this size.
    private func resize(frame: CVPixelBuffer, to target: CGSize) -> CVPixelBuffer? {
        let srcWidth = CVPixelBufferGetWidth(frame)
        let srcHeight = CVPixelBufferGetHeight(frame)
        let dstWidth = Int(target.width)
        let dstHeight = Int(target.height)
        if srcWidth == dstWidth && srcHeight == dstHeight {
            return frame
        }

        var dst: CVPixelBuffer?
        let attrs = [
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
        ] as CFDictionary
        CVPixelBufferCreate(kCFAllocatorDefault, dstWidth, dstHeight, kCVPixelFormatType_32BGRA, attrs, &dst)
        guard let dst else { return nil }

        let ciImage = CIImage(cvPixelBuffer: frame)
        let scaleX = CGFloat(dstWidth) / CGFloat(srcWidth)
        let scaleY = CGFloat(dstHeight) / CGFloat(srcHeight)
        // Centre-crop to a square aspect first so the resize doesn't
        // distort. We assume the camera outputs a 4:3 frame and the target
        // is square — see CameraManager.sessionPreset.
        let scale = max(scaleX, scaleY)
        let scaled = ciImage.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        let dx = (scaled.extent.width - CGFloat(dstWidth)) / 2
        let dy = (scaled.extent.height - CGFloat(dstHeight)) / 2
        let cropped = scaled.cropped(to: CGRect(x: scaled.extent.minX + dx,
                                                y: scaled.extent.minY + dy,
                                                width: CGFloat(dstWidth),
                                                height: CGFloat(dstHeight)))
        ciContext.render(cropped, to: dst)
        return dst
    }
}
