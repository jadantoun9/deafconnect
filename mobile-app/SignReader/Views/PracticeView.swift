// PracticeView.swift
//
// Drills mode: app picks a target sign from a small curated list, the user
// signs it, and we score correct/incorrect based on the live model output.
//
// Why a curated list rather than the full WLASL gloss space:
//   The full 100/300 vocabulary mostly contains words a non-signer doesn't
//   know how to perform. Practice mode is for showing the model works;
//   real evaluation belongs in the user-study screen (Phase E.3).
import SwiftUI

public struct PracticeView: View {
    @EnvironmentObject private var app: AppState

    /// Short, well-known signs from the WLASL-100 vocabulary. Owner can
    /// extend this list once a real model is trained — the loader doesn't
    /// constrain to it.
    private let curated: [String] = [
        "hello", "thank you", "yes", "no", "please",
        "good", "bad", "school", "book", "computer",
    ]

    @State private var target: String = "hello"
    @State private var streak: Int = 0
    @State private var attempts: Int = 0
    @State private var feedback: Feedback = .none
    @State private var lockoutUntil: Date = .distantPast

    private enum Feedback {
        case none
        case correct
        case incorrect(seen: String)

        var color: Color {
            switch self {
            case .none: return .clear
            case .correct: return .green
            case .incorrect: return .orange
            }
        }
        var text: String {
            switch self {
            case .none: return ""
            case .correct: return "Correct!"
            case .incorrect(let seen): return "Saw “\(seen)”"
            }
        }
    }

    public init() {}

    public var body: some View {
        NavigationStack {
            VStack(spacing: 14) {
                targetCard

                CameraPreview(session: app.camera.session)
                    .aspectRatio(3.0 / 4.0, contentMode: .fit)
                    .cornerRadius(12)

                feedbackBanner

                statsBar

                HStack {
                    Button("Skip", systemImage: "forward") { newTarget() }
                        .buttonStyle(.bordered)
                    Button("Reset", systemImage: "arrow.counterclockwise") {
                        attempts = 0
                        streak = 0
                        feedback = .none
                        newTarget()
                    }
                    .buttonStyle(.bordered)
                }

                Spacer()
            }
            .padding()
            .navigationTitle("Practice")
            .onAppear {
                target = curated.randomElement() ?? "hello"
                // Camera lifecycle handled by scenePhase in SignReaderApp.
            }
            .onChange(of: app.engine.prediction) { newValue in
                guard let p = newValue else { return }
                evaluate(prediction: p)
            }
        }
    }

    @ViewBuilder
    private var targetCard: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Sign this:").font(.headline).foregroundStyle(.secondary)
            Text(target.uppercased())
                .font(.system(size: 36, weight: .heavy, design: .rounded))
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(.tint.opacity(0.12))
        .cornerRadius(12)
    }

    @ViewBuilder
    private var feedbackBanner: some View {
        Text(feedback.text)
            .font(.headline)
            .foregroundStyle(feedback.color == .clear ? .secondary : feedback.color)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 8)
            .background(feedback.color.opacity(0.12))
            .cornerRadius(8)
            .opacity(feedback.text.isEmpty ? 0 : 1)
            .animation(.easeInOut(duration: 0.2), value: feedback.text)
    }

    @ViewBuilder
    private var statsBar: some View {
        HStack {
            Label("Streak: \(streak)", systemImage: "flame.fill")
            Spacer()
            Label("Attempts: \(attempts)", systemImage: "number")
        }
        .font(.subheadline)
        .foregroundStyle(.secondary)
        .padding(.horizontal)
    }

    private func evaluate(prediction p: Prediction) {
        // Lockout window after a correct/incorrect verdict so the same
        // gesture isn't credited multiple times in a row.
        if Date() < lockoutUntil { return }
        guard p.confidence >= app.engine.confidenceThreshold else { return }
        attempts += 1
        if matches(predicted: p.label, target: target) {
            streak += 1
            feedback = .correct
            lockoutUntil = Date().addingTimeInterval(1.5)
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                self.newTarget()
                self.feedback = .none
            }
        } else {
            streak = 0
            feedback = .incorrect(seen: p.label)
            lockoutUntil = Date().addingTimeInterval(0.8)
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
                self.feedback = .none
            }
        }
    }

    private func matches(predicted: String, target: String) -> Bool {
        // Case-insensitive, whitespace-insensitive comparison. Models are
        // trained on space-free glosses ("thankyou"); allow either form.
        let norm: (String) -> String = { $0.lowercased().filter { !$0.isWhitespace } }
        return norm(predicted) == norm(target)
    }

    private func newTarget() {
        var t = curated.randomElement() ?? "hello"
        // Avoid repeating the same target back-to-back.
        if t == target, curated.count > 1 {
            t = curated.first { $0 != target } ?? t
        }
        target = t
    }
}
