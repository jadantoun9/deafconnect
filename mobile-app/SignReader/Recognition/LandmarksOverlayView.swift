// LandmarksOverlayView.swift
//
// SwiftUI Canvas overlay that paints the latest MediaPipe Holistic
// landmarks on top of the live camera preview. Subscribes to the
// extractor's published `latestLandmarks` and re-renders on every new
// detection (~30 fps).
//
// We keep the drawing inexpensive: small filled circles for points,
// thin lines for selected skeleton edges. Heavier visualisations
// (full skeleton mesh, per-joint labels) are easy to add later but
// would compete with Vision's CPU budget on smaller devices.
import SwiftUI

struct LandmarksOverlayView: View {
    @ObservedObject var extractor: LandmarkExtractor
    /// True when the camera preview is mirrored (front camera). We
    /// flip the x-coords so the drawn dots align with what the user
    /// sees on screen instead of with the un-mirrored sensor frame.
    var mirrorHorizontally: Bool = true

    var body: some View {
        Canvas { context, size in
            let snapshot = extractor.latestLandmarks
            // Skeleton edges first so the joint dots draw over them.
            drawConnections(in: &context, size: size,
                            points: snapshot.pose, edges: Self.poseConnections,
                            color: .green.opacity(0.7), width: 2)
            drawConnections(in: &context, size: size,
                            points: snapshot.leftHand, edges: Self.handConnections,
                            color: .yellow.opacity(0.85), width: 2)
            drawConnections(in: &context, size: size,
                            points: snapshot.rightHand, edges: Self.handConnections,
                            color: .cyan.opacity(0.85), width: 2)
            drawPoints(in: &context, size: size, points: snapshot.pose,
                       color: .green, radius: 3)
            drawPoints(in: &context, size: size, points: snapshot.leftHand,
                       color: .yellow, radius: 3)
            drawPoints(in: &context, size: size, points: snapshot.rightHand,
                       color: .cyan, radius: 3)
        }
        .allowsHitTesting(false) // never block underlying camera taps
    }

    private func screenPoint(_ p: CGPoint, in size: CGSize) -> CGPoint {
        let x = mirrorHorizontally ? (1 - p.x) : p.x
        return CGPoint(x: x * size.width, y: p.y * size.height)
    }

    private func drawPoints(in context: inout GraphicsContext, size: CGSize,
                            points: [CGPoint], color: Color, radius: CGFloat) {
        for p in points {
            let pt = screenPoint(p, in: size)
            let rect = CGRect(x: pt.x - radius, y: pt.y - radius,
                              width: radius * 2, height: radius * 2)
            context.fill(Path(ellipseIn: rect), with: .color(color))
        }
    }

    private func drawConnections(in context: inout GraphicsContext, size: CGSize,
                                 points: [CGPoint], edges: [(Int, Int)],
                                 color: Color, width: CGFloat) {
        guard points.count >= 2 else { return }
        var path = Path()
        for (a, b) in edges {
            guard a < points.count, b < points.count else { continue }
            path.move(to: screenPoint(points[a], in: size))
            path.addLine(to: screenPoint(points[b], in: size))
        }
        context.stroke(path, with: .color(color), lineWidth: width)
    }

    /// Pose skeleton edges — a subset of MediaPipe Pose's 33-point graph.
    /// We omit the inner-eye / mouth-corner edges since those just clutter
    /// the upper-body view of a signing user.
    private static let poseConnections: [(Int, Int)] = [
        // Shoulders + arms
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        // Hand tip clusters (built into pose, kept for visual continuity)
        (15, 17), (15, 19), (15, 21), (17, 19),
        (16, 18), (16, 20), (16, 22), (18, 20),
        // Torso
        (11, 23), (12, 24), (23, 24),
        // Hips → legs
        (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
        (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
    ]

    /// MediaPipe Hands' 21-point standard topology (5 fingers × 4 bones
    /// + thumb chain + connections back to the wrist).
    private static let handConnections: [(Int, Int)] = [
        // Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        // Index
        (0, 5), (5, 6), (6, 7), (7, 8),
        // Middle
        (0, 9), (9, 10), (10, 11), (11, 12),
        // Ring
        (0, 13), (13, 14), (14, 15), (15, 16),
        // Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
        // Palm spans
        (5, 9), (9, 13), (13, 17),
    ]
}
