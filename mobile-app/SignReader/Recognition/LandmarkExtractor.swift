// LandmarkExtractor.swift
//
// Per-frame body + hand keypoint extraction using **MediaPipe Holistic
// Landmarker** — the same feature extractor the Track 1 9-class model
// was trained against. Distribution-matched, no Apple Vision detour.
//
// Keypoint layout produced (75 total, matches the .mlpackage's input
// shape [1, 32, 75, 3]):
//   [ 0..32]  pose          (33 points; MediaPipe's full-body schema)
//   [33..53]  left hand     (21 points)
//   [54..74]  right hand    (21 points)
// (Face mesh is exposed by the holistic model but the model wasn't
// trained with it; we ignore it.)
//
// Threading:
//   - extract(...) runs synchronously on the camera output queue.
//   - The internal ring buffer is lock-protected so snapshotInputTensor()
//     can be called from the main actor without races.
//   - latestLandmarks is @Published; updates dispatch to MainActor so
//     the overlay view re-renders on each new detection.
//
// MediaPipe API note: the holistic landmarker runs in `.video` mode so
// we can pass timestamps and let it apply temporal smoothing across
// frames. Image-mode would re-detect from scratch every frame.
import CoreML
import CoreVideo
import Foundation
import MediaPipeTasksVision

/// One frame of extracted landmarks: a flat [75 * 3 = 225] array of
/// floats. Missing keypoints are zero-filled. The ring buffer stores
/// these *raw* (un-normalised) frames so the keypoint subset selected
/// at snapshot time can normalise over its own subset, not the full
/// 75-point centroid (the Python data pipeline does the same).
typealias LandmarkFrame = [Float]

/// Number of keypoints in the raw extractor output. Always 75 — even when
/// the active model variant uses fewer (we slice at snapshot time).
let kKeypointCount: Int = 75
let kCoordDims: Int = 3
let kFlatFrameSize: Int = kKeypointCount * kCoordDims  // 225

/// Number of frames in the temporal window the model expects.
let kFrameWindow: Int = 32

/// Which keypoints feed the model. Mirrors `research/data/keypoint_subsets.py`
/// — the v2 subset must match byte-for-byte or the model gets garbage input.
public enum KeypointSubset: String, CaseIterable, Identifiable {
    /// Full 75-keypoint MediaPipe Holistic output (Track 1 v1 model).
    case all
    /// 48-keypoint v2 layout: shoulders, elbows, wrists + both 21-point hand
    /// meshes. Drops head, hips, lower body, and pose-level coarse hand
    /// landmarks (which are redundant with the full hand mesh).
    case upperBodyHands

    public var id: String { rawValue }

    /// Indices into the raw 75-keypoint layout, in the order the model
    /// expects them. For `upperBodyHands` this is:
    ///   [0..5]   pose anchors  L/R shoulder, L/R elbow, L/R wrist
    ///   [6..26]  left hand     (21 points, full mesh)
    ///   [27..47] right hand    (21 points, full mesh)
    public var indices: [Int] {
        switch self {
        case .all:
            return Array(0..<75)
        case .upperBodyHands:
            return [11, 12, 13, 14, 15, 16]
                + Array(33..<54)
                + Array(54..<75)
        }
    }

    public var keypointCount: Int { indices.count }
}

/// One detection's keypoint positions in normalized [0,1] coords (top-
/// left origin). Used by the SwiftUI overlay to draw circles + skeleton
/// lines over the live camera feed. Empty arrays mean "not detected
/// this frame".
public struct LandmarkSnapshot: Equatable {
    public var pose: [CGPoint]        // up to 33
    public var leftHand: [CGPoint]    // up to 21
    public var rightHand: [CGPoint]   // up to 21

    public static let empty = LandmarkSnapshot(pose: [], leftHand: [], rightHand: [])
}

final class LandmarkExtractor: ObservableObject {

    /// Latest detection — published for the overlay drawing.
    @Published public private(set) var latestLandmarks: LandmarkSnapshot = .empty

    /// 32-frame ring buffer of flat [225] frames + parallel valid-mask ring.
    /// We store the validity mask alongside the raw landmarks so the snapshot
    /// step can normalise only over keypoints that were actually detected
    /// (matching research/data/landmark_dataset.py:_normalise_landmarks).
    /// Newest write is at `(writeIndex - 1) mod kFrameWindow`. Lock-guarded.
    private var ring: [LandmarkFrame] = Array(
        repeating: Array(repeating: 0, count: kFlatFrameSize),
        count: kFrameWindow
    )
    private var ringValid: [[Bool]] = Array(
        repeating: Array(repeating: false, count: kKeypointCount),
        count: kFrameWindow
    )
    private var writeIndex: Int = 0
    private var framesWritten: Int = 0
    private let ringLock = NSLock()

    private let landmarker: HolisticLandmarker?

    public init() {
        self.landmarker = Self.makeLandmarker()
        if landmarker == nil {
            NSLog("LandmarkExtractor: holistic_landmarker.task not found or failed to load.")
        }
    }

    private static func makeLandmarker() -> HolisticLandmarker? {
        guard let path = Bundle.main.path(forResource: "holistic_landmarker", ofType: "task") else {
            return nil
        }
        let options = HolisticLandmarkerOptions()
        options.baseOptions.modelAssetPath = path
        options.runningMode = .video
        // Confidence thresholds — defaults are fine for the demo. Lower
        // them if hands frequently fail to detect under poor lighting.
        options.minPoseDetectionConfidence = 0.5
        options.minPoseSuppressionThreshold = 0.5
        options.minPosePresenceConfidence = 0.5
        options.minHandLandmarksConfidence = 0.5
        do {
            return try HolisticLandmarker(options: options)
        } catch {
            NSLog("LandmarkExtractor: HolisticLandmarker init failed: \(error)")
            return nil
        }
    }

    /// Extract one frame's landmarks and push them into the ring buffer.
    /// Safe to call from the camera output queue.
    func extract(pixelBuffer: CVPixelBuffer, timestampMillis: Int) {
        var frame = Array<Float>(repeating: 0, count: kFlatFrameSize)
        var valid = Array<Bool>(repeating: false, count: kKeypointCount)
        var snapshot = LandmarkSnapshot.empty

        guard let landmarker else {
            ringWrite(frame, valid: valid)
            publish(snapshot)
            return
        }

        let mpImage: MPImage
        do {
            mpImage = try MPImage(pixelBuffer: pixelBuffer)
        } catch {
            ringWrite(frame, valid: valid)
            publish(snapshot)
            return
        }

        let result: HolisticLandmarkerResult?
        do {
            result = try landmarker.detect(videoFrame: mpImage, timestampInMilliseconds: timestampMillis)
        } catch {
            ringWrite(frame, valid: valid)
            publish(snapshot)
            return
        }

        if let r = result {
            // poseLandmarks / leftHandLandmarks / rightHandLandmarks are
            // each non-optional [NormalizedLandmark]; an empty array means
            // "not detected this frame".
            let pose = r.poseLandmarks
            if !pose.isEmpty {
                writeLandmarks(pose, into: &frame, valid: &valid, baseIndex: 0, count: 33)
                snapshot.pose = pose.prefix(33).map { CGPoint(x: CGFloat($0.x), y: CGFloat($0.y)) }
            }
            let left = r.leftHandLandmarks
            if !left.isEmpty {
                writeLandmarks(left, into: &frame, valid: &valid, baseIndex: 33, count: 21)
                snapshot.leftHand = left.prefix(21).map { CGPoint(x: CGFloat($0.x), y: CGFloat($0.y)) }
            }
            let right = r.rightHandLandmarks
            if !right.isEmpty {
                writeLandmarks(right, into: &frame, valid: &valid, baseIndex: 54, count: 21)
                snapshot.rightHand = right.prefix(21).map { CGPoint(x: CGFloat($0.x), y: CGFloat($0.y)) }
            }
        }

        // Ring stores RAW (un-normalised) frames + per-frame valid masks.
        // Normalisation runs at snapshot time, after slicing to the active
        // subset, so v2 normalises over only the 48 kept points (matching
        // research/data/landmark_dataset.py).
        ringWrite(frame, valid: valid)
        publish(snapshot)
    }

    /// Snapshot the ring buffer as a `[1, T, K, 3]` MLMultiArray, where K is
    /// determined by `subset` (75 for v1, 48 for v2). Slicing happens BEFORE
    /// per-frame normalisation, byte-matching the Python data pipeline.
    func snapshotInputTensor(
        subset: KeypointSubset = .all,
        minFramesWritten: Int = kFrameWindow
    ) -> MLMultiArray? {
        ringLock.lock()
        defer { ringLock.unlock() }
        guard framesWritten >= minFramesWritten else { return nil }

        let kpIdx = subset.indices
        let K = kpIdx.count
        guard let array = try? MLMultiArray(
            shape: [1, NSNumber(value: kFrameWindow),
                    NSNumber(value: K), NSNumber(value: kCoordDims)],
            dataType: .float32
        ) else {
            return nil
        }
        let ptr = array.dataPointer.bindMemory(to: Float.self,
                                               capacity: kFrameWindow * K * kCoordDims)

        // Reusable scratch space for one normalised, sliced frame.
        var sliced = Array<Float>(repeating: 0, count: K * kCoordDims)
        var slicedValid = Array<Bool>(repeating: false, count: K)

        for t in 0..<kFrameWindow {
            let srcIdx = (writeIndex + t) % kFrameWindow
            let src = ring[srcIdx]
            let srcValid = ringValid[srcIdx]

            // 1. Slice raw 75-kp frame down to the requested subset.
            for (i, rawK) in kpIdx.enumerated() {
                sliced[i * 3 + 0] = src[rawK * 3 + 0]
                sliced[i * 3 + 1] = src[rawK * 3 + 1]
                sliced[i * 3 + 2] = src[rawK * 3 + 2]
                slicedValid[i] = srcValid[rawK]
            }
            // 2. Normalise across the *sliced* keypoints (centroid + scale
            //    of the kept points only).
            normaliseSlice(&sliced, valid: slicedValid)
            // 3. Pack into the multi-array.
            let base = t * K * kCoordDims
            for k in 0..<(K * kCoordDims) {
                ptr[base + k] = sliced[k]
            }
        }
        return array
    }

    /// Snapshot the most recently written frame's hand keypoints as a
    /// `[1, 21, 3]` MLMultiArray of RAW (un-normalised) MediaPipe coords.
    /// Used by the letter classifier, which is single-frame, single-hand,
    /// and was trained on raw [0,1] hand-landmark coords (no centroid /
    /// scale normalisation — matches src/realtime.py).
    ///
    /// Hand-selection rule: prefer whichever hand has more detected
    /// keypoints in the latest frame; tie-break to the right hand
    /// (matches the right-hand-dominant ASL alphabet training set).
    /// Returns nil if neither hand was detected on the latest frame.
    func snapshotHandFrame() -> MLMultiArray? {
        ringLock.lock()
        defer { ringLock.unlock() }
        guard framesWritten > 0 else { return nil }
        let latest = (writeIndex + kFrameWindow - 1) % kFrameWindow
        let valid = ringValid[latest]
        let frame = ring[latest]

        // Left hand:  raw indices 33..53 (21 points)
        // Right hand: raw indices 54..74 (21 points)
        let leftCount  = (33..<54).reduce(0)  { $0 + (valid[$1] ? 1 : 0) }
        let rightCount = (54..<75).reduce(0)  { $0 + (valid[$1] ? 1 : 0) }
        guard leftCount > 0 || rightCount > 0 else { return nil }

        let baseIdx: Int = rightCount >= leftCount ? 54 : 33

        guard let array = try? MLMultiArray(
            shape: [1, 21, NSNumber(value: kCoordDims)],
            dataType: .float32
        ) else { return nil }
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: 21 * kCoordDims)
        for i in 0..<21 {
            let src = (baseIdx + i) * kCoordDims
            ptr[i * 3 + 0] = frame[src + 0]
            ptr[i * 3 + 1] = frame[src + 1]
            ptr[i * 3 + 2] = frame[src + 2]
        }
        return array
    }

    /// Reset the buffer (e.g. when the user re-enters the tab, or when the
    /// model variant toggles — the ring's contents from the previous variant
    /// are still raw 75-kp data, so they're fine to keep, but resetting also
    /// clears `framesWritten` so we re-buffer before the next inference).
    func reset() {
        ringLock.lock()
        for i in 0..<ring.count {
            ring[i] = Array(repeating: 0, count: kFlatFrameSize)
            ringValid[i] = Array(repeating: false, count: kKeypointCount)
        }
        writeIndex = 0
        framesWritten = 0
        ringLock.unlock()
        publish(.empty)
    }

    // MARK: - Internals

    private func ringWrite(_ frame: LandmarkFrame, valid: [Bool]) {
        ringLock.lock()
        ring[writeIndex] = frame
        ringValid[writeIndex] = valid
        writeIndex = (writeIndex + 1) % kFrameWindow
        framesWritten = min(framesWritten + 1, kFrameWindow)
        ringLock.unlock()
    }

    private func publish(_ snapshot: LandmarkSnapshot) {
        Task { @MainActor [weak self] in
            self?.latestLandmarks = snapshot
        }
    }

    /// Slot landmarks into the flat frame at the given base index. Coords
    /// are taken straight from the model in normalized [0,1] (no flip
    /// needed — MediaPipe already uses top-left origin).
    private func writeLandmarks(_ lms: [NormalizedLandmark], into frame: inout LandmarkFrame, valid: inout [Bool], baseIndex: Int, count: Int) {
        let n = min(lms.count, count)
        for i in 0..<n {
            let lm = lms[i]
            let slot = baseIndex + i
            frame[slot * 3 + 0] = lm.x
            frame[slot * 3 + 1] = lm.y
            frame[slot * 3 + 2] = lm.z
            valid[slot] = true
        }
    }

    /// Per-frame normalisation operating on an already-sliced flat buffer of
    /// `K * 3` floats and its `K`-length validity mask. Translates by the
    /// centroid of valid points then scales by `max(|x|,|y|)` of valid points.
    /// Mirrors `research/data/landmark_dataset.py:_normalise_landmarks`.
    private func normaliseSlice(_ frame: inout [Float], valid: [Bool]) {
        let K = valid.count
        var sumX: Float = 0, sumY: Float = 0
        var count: Int = 0
        for k in 0..<K where valid[k] {
            sumX += frame[k * 3 + 0]
            sumY += frame[k * 3 + 1]
            count += 1
        }
        guard count >= 2 else { return }
        let cx = sumX / Float(count)
        let cy = sumY / Float(count)
        for k in 0..<K {
            frame[k * 3 + 0] -= cx
            frame[k * 3 + 1] -= cy
        }
        var maxAbs: Float = 0
        for k in 0..<K where valid[k] {
            maxAbs = max(maxAbs, max(abs(frame[k * 3 + 0]), abs(frame[k * 3 + 1])))
        }
        guard maxAbs > 1e-6 else { return }
        for k in 0..<K {
            frame[k * 3 + 0] /= maxAbs
            frame[k * 3 + 1] /= maxAbs
        }
    }
}
