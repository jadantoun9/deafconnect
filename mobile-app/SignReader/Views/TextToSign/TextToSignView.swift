// TextToSignView.swift
//
// Type or speak text → 3D avatar performs the sign. Mirrors deafconnect's
// lib/screens/text_to_sign.screen.dart at ~390 LOC, simplified where SwiftUI
// affordances let us replace Flutter ceremony.
//
// Translation pipeline:
//   1. Split input by space.
//   2. For each word: if a same-named animation exists in
//      SignAnimationCatalog, play it. Otherwise fall back to per-letter
//      fingerspelling.
//   3. After each animation, await its duration before queuing the next.
//   4. On completion, persist the input as an AvatarMessageEntity (Phase 7
//      surfaces these in a history sheet) and return the avatar to Idle.
import SwiftData
import SwiftUI

struct TextToSignView: View {
    @EnvironmentObject private var app: AppState
    @Environment(\.modelContext) private var modelContext

    @StateObject private var stt = SpeechRecognizer()
    @State private var avatarCommands: [Avatar3DCommand] = []
    @State private var inputText: String = ""
    @State private var indexBeingTranslated: Int = -1
    @State private var isTranslating: Bool = false
    @State private var showAvatarPicker: Bool = false
    @State private var showBackgroundPicker: Bool = false
    @State private var showHistory: Bool = false
    @FocusState private var inputFocused: Bool

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                wordBreakdownRow
                avatarContainer
                bottomBar
            }
            // .background takes a view that fills behind the modified
            // content without contributing to parent layout sizing — so
            // the VStack alone drives width/height and the background
            // image's intrinsic pixel dimensions can never leak into
            // the row above (which was pushing the input bar past the
            // visible width).
            .background {
                background
            }
            .navigationTitle("Text to Sign Language")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Menu {
                        Button { showAvatarPicker = true } label: { Label("Choose avatar", systemImage: "person.fill") }
                        Button { showBackgroundPicker = true } label: { Label("Background", systemImage: "photo.fill") }
                        Button { showHistory = true } label: { Label("History", systemImage: "clock.fill") }
                    } label: {
                        Image(systemName: "slider.horizontal.3")
                            .foregroundStyle(Color.brandMain)
                    }
                }
            }
            .sheet(isPresented: $showAvatarPicker) {
                AvatarPickerSheet()
            }
            .sheet(isPresented: $showBackgroundPicker) {
                BackgroundPickerSheet()
            }
            .sheet(isPresented: $showHistory) {
                AvatarHistorySheet { text in
                    showHistory = false
                    inputText = text
                    Task { await translate(text: text) }
                }
            }
            .onAppear {
                // Match deafconnect: idle pose ~10s after launch. We
                // shorten to a ~2s post-load beat so the avatar isn't
                // statically frozen while assets load.
                Task {
                    try? await Task.sleep(for: .seconds(2))
                    avatarCommands.append(.idle)
                }
            }
            .onChange(of: stt.transcript) { _, new in
                // Mirror partial speech recognition into the text field
                // while listening, just like deafconnect's onResult callback.
                if stt.isRecording { inputText = new }
            }
        }
    }

    // ── Background ──────────────────────────────────────────────────

    private var background: some View {
        // Constrain the layer to the ZStack's frame and clip the overflow.
        // Without these, a scaled-to-fill background image's intrinsic
        // dimensions (a multi-megapixel photo can be 4000+ pts wide)
        // propagate into the ZStack's size, which in turn pushes the
        // bottom input row past the visible width — the send button
        // disappears and the leading text scrolls off.
        Group {
            if app.textToSignBackground.isEmpty {
                Color.brandSecondary
            } else {
                Image(app.textToSignBackground)
                    .resizable()
                    .scaledToFill()
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .clipped()
        .ignoresSafeArea()
    }

    // ── Word breakdown ─────────────────────────────────────────────

    private var wordBreakdownRow: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(Array(tokensForBreakdown(inputText).enumerated()), id: \.offset) { idx, token in
                    Text(token.text)
                        .font(.system(size: 21, weight: idx == indexBeingTranslated ? .bold : .regular))
                        .foregroundStyle(idx == indexBeingTranslated ? Color.brandMain : Color.primary)
                }
            }
            .padding(.horizontal)
        }
        .frame(height: 50)
        .background(Color.white.opacity(0.0))
        // Tap anywhere along the breakdown row (or on its empty space
        // before any input is typed) to dismiss the keyboard. The avatar
        // surface itself can't carry this gesture without breaking the
        // SCNView's orbit-on-drag behaviour.
        .contentShape(Rectangle())
        .onTapGesture { inputFocused = false }
    }

    // ── Avatar ──────────────────────────────────────────────────────

    private var avatarContainer: some View {
        Avatar3DView(glbName: app.avatarKind.glbName, commands: $avatarCommands)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            // Cropping to upper body and gesture handling are owned by
            // Avatar3DView's underlying SCNView (custom camera +
            // allowsCameraControl). Don't apply SwiftUI .scaleEffect
            // here — it leaked into the bottom-bar / keyboard layout
            // when the keyboard appeared.
            //
            // Tap-to-dismiss-keyboard is wired on the word-breakdown row
            // above instead, so it doesn't compete with the SCNView's
            // pan/pinch gestures for orbiting the avatar.
    }

    // ── Bottom input bar ───────────────────────────────────────────

    private var bottomBar: some View {
        HStack(alignment: .bottom, spacing: 8) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Translate")
                    .foregroundStyle(Color.brandMain)
                    .font(.subheadline)
                TextField("Type to translate...", text: $inputText)
                    .textFieldStyle(.plain)
                    .focused($inputFocused)
                    .submitLabel(.send)
                    .onSubmit {
                        guard canSend else { return }
                        Task { await translate(text: inputText) }
                    }
            }
            // Pin the input column to take whatever horizontal space is
            // left after the two trailing buttons. Without this the
            // TextField's intrinsic width grows with the typed string
            // and pushes the leading characters (and the buttons) off
            // the visible row.
            .frame(maxWidth: .infinity, alignment: .leading)

            Button {
                Task { await toggleListening() }
            } label: {
                roundedAction(systemName: stt.isRecording ? "stop.fill" : "mic.fill")
            }
            .buttonStyle(.plain)
            Button {
                Task { await translate(text: inputText) }
            } label: {
                roundedAction(systemName: "paperplane.fill")
                    .opacity(canSend ? 1 : 0.45)
            }
            .buttonStyle(.plain)
            .disabled(!canSend)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(Color.brandSecondary)
    }

    private func roundedAction(systemName: String) -> some View {
        ZStack {
            Circle().fill(Color.brandMain)
            Image(systemName: systemName)
                .foregroundStyle(.white)
                .font(.system(size: 17, weight: .semibold))
        }
        .frame(width: 40, height: 40)
    }

    private var canSend: Bool {
        !inputText.trimmingCharacters(in: .whitespaces).isEmpty && !isTranslating
    }

    // ── Actions ────────────────────────────────────────────────────

    private func toggleListening() async {
        if stt.isRecording {
            let final = stt.stop()
            inputText = final
        } else {
            do { try await stt.start() } catch { /* alert via stt.lastError if needed */ }
            // Pipe partial recognition into the text field as it streams.
            // Done via observation in body re-render; here we just kick off.
        }
    }

    private func translate(text rawText: String) async {
        let trimmed = rawText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !isTranslating else { return }
        inputFocused = false
        isTranslating = true

        // Persist this translation request to history.
        let entry = AvatarMessageEntity(text: trimmed)
        modelContext.insert(entry)
        try? modelContext.save()

        // Compute the playback sequence + the visual breakdown indices.
        let tokens = tokensForBreakdown(trimmed)
        for (idx, token) in tokens.enumerated() {
            indexBeingTranslated = idx
            avatarCommands.append(.play(animationName: token.animationName))
            let dur = SignAnimationCatalog.duration(for: token.animationName)
            try? await Task.sleep(for: .seconds(dur))
        }
        indexBeingTranslated = -1
        inputText = ""
        avatarCommands.append(.idle)
        isTranslating = false
    }

    // ── Tokenization ───────────────────────────────────────────────

    /// One token = one animation play. Either a whole word (if a
    /// matching animation exists) or a single letter (fallback).
    private struct Token: Hashable {
        let text: String
        let animationName: String
    }

    private func tokensForBreakdown(_ raw: String) -> [Token] {
        guard !raw.isEmpty else { return [] }
        var out: [Token] = []
        let words = raw.split(separator: " ").map(String.init)
        for word in words {
            let asWord = SignAnimationCatalog.capitalizeFirstLetter(word)
            if SignAnimationCatalog.durations.keys.contains(asWord) {
                out.append(Token(text: word, animationName: asWord))
            } else {
                for ch in word {
                    let letter = String(ch).uppercased()
                    if SignAnimationCatalog.durations.keys.contains(letter) {
                        out.append(Token(text: String(ch), animationName: letter))
                    }
                }
            }
        }
        return out
    }
}

// ── Avatar picker ──────────────────────────────────────────────────

struct AvatarPickerSheet: View {
    @EnvironmentObject private var app: AppState
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            VStack(spacing: 20) {
                Text("Choose your avatar")
                    .font(.headline)
                    .padding(.top)
                HStack(spacing: 16) {
                    ForEach(AvatarKind.allCases) { kind in
                        Button {
                            app.avatarKind = kind
                            dismiss()
                        } label: {
                            VStack {
                                Image(kind.thumbnailAsset)
                                    .resizable()
                                    .scaledToFit()
                                    .frame(height: 140)
                                    .clipShape(RoundedRectangle(cornerRadius: 12))
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 12)
                                            .stroke(app.avatarKind == kind ? Color.brandMain : Color.clear, lineWidth: 3)
                                    )
                                Text(kind.displayName)
                                    .font(.subheadline)
                                    .foregroundStyle(.primary)
                            }
                        }
                        .buttonStyle(.plain)
                    }
                }
                Spacer()
            }
            .padding()
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }
                }
            }
            .presentationDetents([.height(320)])
        }
    }
}

// ── Background picker ──────────────────────────────────────────────

struct BackgroundPickerSheet: View {
    @EnvironmentObject private var app: AppState
    @Environment(\.dismiss) private var dismiss

    /// Asset names that ship in Assets.xcassets/Bg*.imageset, plus the
    /// empty-string sentinel for "no background".
    private let options: [(name: String, asset: String)] = [
        ("None",       ""),
        ("Beach",      "BgBeach"),
        ("Christmas",  "BgChristmas"),
        ("Green",      "BgGreen"),
        ("Moon",       "BgMoon"),
        ("Mountains",  "BgMountains"),
        ("Orange",     "BgOrange"),
        ("Pink",       "BgPink"),
        ("Sky blue",   "BgSkyblue"),
        ("Transparent","BgTransparent"),
    ]

    private let columns = [GridItem(.adaptive(minimum: 100), spacing: 12)]

    var body: some View {
        NavigationStack {
            ScrollView {
                LazyVGrid(columns: columns, spacing: 12) {
                    ForEach(options, id: \.asset) { option in
                        Button {
                            app.textToSignBackground = option.asset
                            dismiss()
                        } label: {
                            VStack(spacing: 6) {
                                Group {
                                    if option.asset.isEmpty {
                                        ZStack {
                                            Color.brandSecondary
                                            Image(systemName: "rectangle.dashed")
                                                .foregroundStyle(.secondary)
                                        }
                                    } else {
                                        Image(option.asset)
                                            .resizable()
                                            .scaledToFill()
                                    }
                                }
                                .frame(width: 100, height: 70)
                                .clipShape(RoundedRectangle(cornerRadius: 8))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 8)
                                        .stroke(app.textToSignBackground == option.asset ? Color.brandMain : Color.clear, lineWidth: 2)
                                )
                                Text(option.name)
                                    .font(.caption)
                                    .foregroundStyle(.primary)
                            }
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding()
            }
            .navigationTitle("Background")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }
                }
            }
            .presentationDetents([.medium])
        }
    }
}

// ── History (compact; full screen lands in Phase 7) ────────────────

struct AvatarHistorySheet: View {
    @Query(sort: \AvatarMessageEntity.date, order: .reverse)
    private var history: [AvatarMessageEntity]
    @Environment(\.modelContext) private var modelContext
    @Environment(\.dismiss) private var dismiss

    /// Replay handler — caller re-runs the translation with the picked text.
    let onReplay: (String) -> Void

    var body: some View {
        NavigationStack {
            List {
                if history.isEmpty {
                    Text("No translations yet.")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(history) { entry in
                        Button {
                            onReplay(entry.text)
                        } label: {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(entry.text)
                                    .foregroundStyle(.primary)
                                Text(entry.date.formatted(date: .abbreviated, time: .shortened))
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                    .onDelete { offsets in
                        for offset in offsets {
                            modelContext.delete(history[offset])
                        }
                        try? modelContext.save()
                    }
                }
            }
            .navigationTitle("Recent translations")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }
                }
            }
            .presentationDetents([.medium, .large])
        }
    }
}
