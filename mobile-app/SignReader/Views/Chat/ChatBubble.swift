// ChatBubble.swift
//
// One row in the chat list. Mirrors deafconnect's
// lib/widgets/chat/chatbox.dart visual: rounded bubble, sender colour
// brand-blue with a tail on the last message of a run, receiver colour
// soft-grey on the left. Tap the speaker icon to replay the message via
// TextToSpeech.
import SwiftUI

struct ChatBubble: View {
    let text: String
    let time: String
    let isReceived: Bool
    /// Show the bubble's tail when this row ends a run of same-side messages.
    let showTail: Bool
    /// Tap-to-play handler. Mirrors `onPlay` in the Flutter widget.
    let onPlay: () -> Void

    var body: some View {
        HStack(alignment: .bottom, spacing: 6) {
            if isReceived {
                content
                Spacer(minLength: 60)
            } else {
                Spacer(minLength: 60)
                content
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 2)
    }

    private var content: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(text)
                .foregroundStyle(isReceived ? Color.primary : .white)
                .multilineTextAlignment(.leading)
            HStack(spacing: 8) {
                Text(time)
                    .font(.caption2)
                    .foregroundStyle(isReceived ? Color.brandLightGray : Color.white.opacity(0.85))
                Button(action: onPlay) {
                    Image(systemName: "speaker.wave.2.fill")
                        .font(.caption)
                        .foregroundStyle(isReceived ? Color.brandMain : .white.opacity(0.9))
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Replay message")
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(bubbleBackground)
        .clipShape(BubbleShape(isReceived: isReceived, showTail: showTail))
    }

    private var bubbleBackground: Color {
        // Received bubbles use the same white as the chat input bar so
        // both feel like "the other person is talking back" surfaces.
        // Sent bubbles use brand blue.
        isReceived ? Color.white : Color.brandMain
    }
}

/// Rounded-rectangle bubble with an optional tail nub on the bottom edge.
/// Tail side flips with the speaker. Approximates deafconnect's
/// `flutter_chat_bubble`'s "type1" shape — close enough that the visual
/// rhythm carries over.
private struct BubbleShape: Shape {
    var isReceived: Bool
    var showTail: Bool

    func path(in rect: CGRect) -> Path {
        let radius: CGFloat = 14
        var path = Path()
        let body = CGRect(x: rect.minX, y: rect.minY, width: rect.width, height: rect.height)
        path.addRoundedRect(in: body, cornerSize: CGSize(width: radius, height: radius))
        if showTail {
            // Tail on bottom-left for received, bottom-right for sent.
            let tailWidth: CGFloat = 10
            let tailHeight: CGFloat = 8
            let baseY = rect.maxY - 2
            if isReceived {
                let baseX = rect.minX + 6
                var tail = Path()
                tail.move(to: CGPoint(x: baseX, y: baseY))
                tail.addQuadCurve(
                    to: CGPoint(x: baseX + tailWidth, y: baseY),
                    control: CGPoint(x: baseX - 4, y: baseY + tailHeight)
                )
                path.addPath(tail)
            } else {
                let baseX = rect.maxX - 6
                var tail = Path()
                tail.move(to: CGPoint(x: baseX, y: baseY))
                tail.addQuadCurve(
                    to: CGPoint(x: baseX - tailWidth, y: baseY),
                    control: CGPoint(x: baseX + 4, y: baseY + tailHeight)
                )
                path.addPath(tail)
            }
        }
        return path
    }
}
