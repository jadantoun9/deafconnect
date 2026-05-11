// ChatView.swift
//
// Real-time chat with TTS playback and STT dictation. The "received" side
// represents what the other person signed/spoke (dictated via the mic);
// the "sent" side is what the app user typed (and is auto-spoken via TTS
// for the deaf-listening counterpart). Mirrors deafconnect's
// lib/screens/chat.screen.dart, simplified where the SwiftUI surface
// doesn't need the Flutter ceremony.
//
// Persistence flows through SwiftData @Query against TranscriptEntity
// and MessageEntity. The currently-selected transcript is tracked in
// AppState (UserDefaults-persisted), so the user comes back to the same
// conversation across launches.
import SwiftData
import SwiftUI

struct ChatView: View {
    @EnvironmentObject private var app: AppState
    @Environment(\.modelContext) private var modelContext

    @Query(sort: \TranscriptEntity.dateCreated, order: .reverse)
    private var transcripts: [TranscriptEntity]

    @StateObject private var tts = TextToSpeech()
    @StateObject private var stt = SpeechRecognizer()

    @State private var inputText: String = ""
    @State private var showTranscriptDialog: Bool = false
    @FocusState private var inputFocused: Bool

    var body: some View {
        NavigationStack {
            content
                .navigationTitle(navTitle)
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .topBarLeading) {
                        Button {
                            showTranscriptDialog = true
                        } label: {
                            Image(systemName: "plus")
                                .foregroundStyle(Color.brandMain)
                        }
                        .accessibilityLabel("New transcript")
                    }
                }
                .sheet(isPresented: $showTranscriptDialog) {
                    TranscriptDialog { name in
                        createTranscript(named: name)
                    }
                }
                .alert("Speech error", isPresented: .constant(stt.lastError != nil)) {
                    Button("OK") { stt.resetTranscript() }
                } message: {
                    Text(stt.lastError ?? "")
                }
        }
    }

    // ── Body switching: empty state vs chat ──────────────────────────

    @ViewBuilder
    private var content: some View {
        if let current = currentTranscript {
            chatBody(for: current)
        } else {
            emptyState
        }
    }

    private var emptyState: some View {
        VStack(spacing: 14) {
            Spacer()
            Image(systemName: "bubble.left.and.bubble.right")
                .font(.system(size: 56))
                .foregroundStyle(Color.brandMain.opacity(0.4))
            Text("No conversation selected")
                .font(.headline)
                .foregroundStyle(.secondary)
            Text("Tap + to start a new conversation.")
                .font(.footnote)
                .foregroundStyle(.secondary)
            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.brandSecondary.ignoresSafeArea())
    }

    private func chatBody(for transcript: TranscriptEntity) -> some View {
        VStack(spacing: 0) {
            messagesList(for: transcript)
            if stt.isRecording && !stt.transcript.isEmpty {
                recordingStrip
            }
            inputBar(transcript: transcript)
        }
        .background(Color.brandSecondary.ignoresSafeArea())
        .onTapGesture {
            inputFocused = false
        }
    }

    // ── Messages list ────────────────────────────────────────────────

    private func messagesList(for transcript: TranscriptEntity) -> some View {
        let sorted = transcript.messages.sorted { $0.date < $1.date }
        let groups = Self.groupedByDay(sorted)

        return ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 0) {
                    ForEach(groups, id: \.day) { group in
                        Section {
                            ForEach(Array(group.messages.enumerated()), id: \.element.id) { idx, msg in
                                ChatBubble(
                                    text: msg.content,
                                    time: Self.timeFormatter.string(from: msg.date),
                                    isReceived: msg.isReceived,
                                    showTail: Self.showTail(at: idx, in: group.messages),
                                    onPlay: { tts.speak(msg.content) }
                                )
                                .id(msg.id)
                            }
                        } header: {
                            Text(Self.dayHeader(group.day))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 8)
                        }
                    }
                }
                .padding(.vertical, 4)
            }
            .onChange(of: sorted.count) { _ in
                if let lastID = sorted.last?.id {
                    withAnimation(.easeOut(duration: 0.2)) {
                        proxy.scrollTo(lastID, anchor: .bottom)
                    }
                }
            }
            .onAppear {
                if let lastID = sorted.last?.id {
                    proxy.scrollTo(lastID, anchor: .bottom)
                }
            }
        }
    }

    private var recordingStrip: some View {
        Text(stt.transcript)
            .frame(maxWidth: .infinity, minHeight: 60, alignment: .leading)
            .padding(12)
            .background(Color.white)
            .overlay(Rectangle().fill(Color.brandLightGray.opacity(0.4)).frame(height: 1), alignment: .top)
    }

    // ── Input row ────────────────────────────────────────────────────

    private func inputBar(transcript: TranscriptEntity) -> some View {
        HStack(alignment: .center, spacing: 10) {
            if stt.isRecording {
                Spacer(minLength: 0)
                Text("Listening — tap to stop")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                micStopButton(transcript: transcript)
                Spacer(minLength: 0)
            } else {
                TextField("Type your message here", text: $inputText, axis: .vertical)
                    .textFieldStyle(.plain)
                    .focused($inputFocused)
                    .lineLimit(1...4)
                if inputText.trimmingCharacters(in: .whitespaces).isEmpty {
                    micStartButton()
                } else {
                    sendButton(transcript: transcript)
                }
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(Color.white)
        .overlay(Rectangle().fill(Color.brandLightGray.opacity(0.4)).frame(height: 1), alignment: .top)
    }

    private func micStartButton() -> some View {
        Button {
            Task { await startListening() }
        } label: {
            roundedIcon("mic.fill", size: 18)
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Start voice dictation")
    }

    private func micStopButton(transcript: TranscriptEntity) -> some View {
        Button {
            commitDictation(into: transcript)
        } label: {
            roundedIcon("stop.fill", size: 18, large: true)
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Stop dictation")
    }

    private func sendButton(transcript: TranscriptEntity) -> some View {
        Button {
            sendTyped(into: transcript)
        } label: {
            roundedIcon("paperplane.fill", size: 16)
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Send message")
    }

    private func roundedIcon(_ name: String, size: CGFloat, large: Bool = false) -> some View {
        let dim: CGFloat = large ? 48 : 36
        return ZStack {
            Circle().fill(Color.brandMain)
            Image(systemName: name)
                .font(.system(size: size, weight: .semibold))
                .foregroundStyle(.white)
        }
        .frame(width: dim, height: dim)
    }

    // ── Actions ──────────────────────────────────────────────────────

    private func sendTyped(into transcript: TranscriptEntity) {
        let trimmed = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        addMessage(content: trimmed, isReceived: false, transcript: transcript)
        tts.speak(trimmed)
        inputText = ""
        inputFocused = false
    }

    private func startListening() async {
        do {
            try await stt.start()
        } catch {
            stt.resetTranscript()
            // The published lastError on stt picks up the description and
            // surfaces the alert.
        }
    }

    private func commitDictation(into transcript: TranscriptEntity) {
        let final = stt.stop().trimmingCharacters(in: .whitespacesAndNewlines)
        guard !final.isEmpty else { return }
        addMessage(content: final, isReceived: true, transcript: transcript)
    }

    private func addMessage(content: String, isReceived: Bool, transcript: TranscriptEntity) {
        let msg = MessageEntity(content: content, isReceived: isReceived, transcript: transcript)
        modelContext.insert(msg)
        try? modelContext.save()
    }

    private func createTranscript(named name: String) {
        let t = TranscriptEntity(name: name)
        modelContext.insert(t)
        try? modelContext.save()
        app.selectedTranscriptID = t.id
    }

    // ── Helpers ──────────────────────────────────────────────────────

    private var currentTranscript: TranscriptEntity? {
        if let id = app.selectedTranscriptID {
            if let match = transcripts.first(where: { $0.id == id }) { return match }
        }
        return transcripts.first
    }

    private var navTitle: String {
        currentTranscript?.name ?? "Chat"
    }

    private static let timeFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "h:mm a"
        return f
    }()

    private struct DayGroup {
        let day: Date
        let messages: [MessageEntity]
    }

    private static func groupedByDay(_ messages: [MessageEntity]) -> [DayGroup] {
        let calendar = Calendar.current
        let dict = Dictionary(grouping: messages) { calendar.startOfDay(for: $0.date) }
        return dict
            .sorted { $0.key < $1.key }
            .map { DayGroup(day: $0.key, messages: $0.value) }
    }

    private static func dayHeader(_ day: Date) -> String {
        let calendar = Calendar.current
        if calendar.isDateInToday(day) { return "Today" }
        if calendar.isDateInYesterday(day) { return "Yesterday" }
        let f = DateFormatter()
        f.dateStyle = .medium
        return f.string(from: day)
    }

    private static func showTail(at index: Int, in run: [MessageEntity]) -> Bool {
        // Show the tail when the next message has a different sender, or
        // when this is the last message in the day's run.
        guard index < run.count - 1 else { return true }
        return run[index + 1].isReceived != run[index].isReceived
    }
}
