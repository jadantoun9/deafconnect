// TranscriptsView.swift
//
// Lists every saved chat from SwiftData, sorted newest-first. Tap a row to
// jump to the Chat tab with that transcript opened. Swipe a row to rename
// or delete. Mirrors deafconnect's lib/screens/transcripts.screen.dart at
// SwiftUI fidelity.
import SwiftData
import SwiftUI

struct TranscriptsView: View {
    @EnvironmentObject private var app: AppState
    @Environment(\.modelContext) private var modelContext

    @Query(sort: \TranscriptEntity.dateCreated, order: .reverse)
    private var transcripts: [TranscriptEntity]

    @State private var renameTarget: TranscriptEntity?
    @State private var deleteTarget: TranscriptEntity?

    var body: some View {
        NavigationStack {
            content
                .navigationTitle("Transcripts")
                .navigationBarTitleDisplayMode(.large)
                .sheet(item: $renameTarget) { target in
                    TranscriptDialog(initialName: target.name) { newName in
                        rename(target, to: newName)
                    }
                }
                .alert(
                    "Delete this conversation?",
                    isPresented: deleteAlertBinding,
                    presenting: deleteTarget
                ) { target in
                    Button("Delete", role: .destructive) { delete(target) }
                    Button("Cancel", role: .cancel) { deleteTarget = nil }
                } message: { target in
                    Text("\"\(target.name)\" and its \(target.messages.count) message\(target.messages.count == 1 ? "" : "s") will be removed. This can't be undone.")
                }
        }
    }

    @ViewBuilder
    private var content: some View {
        if transcripts.isEmpty {
            emptyState
        } else {
            list
        }
    }

    private var list: some View {
        List {
            ForEach(transcripts) { transcript in
                Button {
                    open(transcript)
                } label: {
                    row(for: transcript)
                }
                .buttonStyle(.plain)
                .swipeActions(edge: .trailing, allowsFullSwipe: false) {
                    Button(role: .destructive) {
                        deleteTarget = transcript
                    } label: {
                        Label("Delete", systemImage: "trash")
                    }
                    Button {
                        renameTarget = transcript
                    } label: {
                        Label("Rename", systemImage: "pencil")
                    }
                    .tint(Color.brandMain)
                }
            }
        }
        .listStyle(.plain)
    }

    private func row(for transcript: TranscriptEntity) -> some View {
        let preview = lastMessagePreview(transcript)
        return HStack(spacing: 12) {
            Circle()
                .fill(Color.brandMain.opacity(0.12))
                .frame(width: 44, height: 44)
                .overlay {
                    Image(systemName: "bubble.left.fill")
                        .foregroundStyle(Color.brandMain)
                }

            VStack(alignment: .leading, spacing: 3) {
                HStack {
                    Text(transcript.name.isEmpty ? "Untitled" : transcript.name)
                        .font(.body.weight(.semibold))
                        .foregroundStyle(.black)
                        .lineLimit(1)
                    Spacer(minLength: 8)
                    Text(Self.relativeDate(transcript.dateCreated))
                        .font(.caption)
                        .foregroundStyle(Color.brandLightGray)
                }
                Text(preview)
                    .font(.footnote)
                    .foregroundStyle(Color.brandLightGray)
                    .lineLimit(1)
            }

            Image(systemName: "chevron.right")
                .font(.footnote)
                .foregroundStyle(Color.brandLightGray)
        }
        .padding(.vertical, 6)
        .contentShape(Rectangle())
    }

    private var emptyState: some View {
        VStack(spacing: 14) {
            Spacer()
            Image(systemName: "list.bullet.rectangle")
                .font(.system(size: 56))
                .foregroundStyle(Color.brandMain.opacity(0.4))
            Text("No conversations yet")
                .font(.headline)
                .foregroundStyle(.black)
            Text("Start one in the Chat tab — it will show up here automatically.")
                .font(.footnote)
                .foregroundStyle(Color.brandLightGray)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)
            Button {
                app.activeTab = 0
            } label: {
                Label("Go to Chat", systemImage: "bubble.left.and.bubble.right.fill")
                    .padding(.horizontal, 16)
                    .padding(.vertical, 10)
            }
            .buttonStyle(.borderedProminent)
            .tint(Color.brandMain)
            .padding(.top, 8)
            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // ── Actions ──────────────────────────────────────────────────────

    private func open(_ transcript: TranscriptEntity) {
        app.selectedTranscriptID = transcript.id
        app.activeTab = 0
    }

    private func rename(_ transcript: TranscriptEntity, to newName: String) {
        let trimmed = newName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        transcript.name = trimmed
        try? modelContext.save()
    }

    private func delete(_ transcript: TranscriptEntity) {
        // If the user just deleted the currently-open chat, clear the
        // selection so Chat tab doesn't try to read a dead reference.
        if app.selectedTranscriptID == transcript.id {
            app.selectedTranscriptID = nil
        }
        modelContext.delete(transcript)
        try? modelContext.save()
        deleteTarget = nil
    }

    /// Two-way binding the alert needs. Clears the target on dismiss so the
    /// alert closes cleanly when the user taps Cancel or outside it.
    private var deleteAlertBinding: Binding<Bool> {
        Binding(
            get: { deleteTarget != nil },
            set: { if !$0 { deleteTarget = nil } }
        )
    }

    // ── Helpers ──────────────────────────────────────────────────────

    private func lastMessagePreview(_ transcript: TranscriptEntity) -> String {
        guard let last = transcript.messages.max(by: { $0.date < $1.date }) else {
            return "No messages yet"
        }
        let prefix = last.isReceived ? "" : "You: "
        return "\(prefix)\(last.content)"
    }

    private static func relativeDate(_ date: Date) -> String {
        let calendar = Calendar.current
        if calendar.isDateInToday(date) {
            let f = DateFormatter()
            f.dateFormat = "h:mm a"
            return f.string(from: date)
        }
        if calendar.isDateInYesterday(date) { return "Yesterday" }
        let days = calendar.dateComponents([.day], from: date, to: .now).day ?? 0
        if days < 7 {
            let f = DateFormatter()
            f.dateFormat = "EEEE"
            return f.string(from: date)
        }
        let f = DateFormatter()
        f.dateStyle = .short
        return f.string(from: date)
    }
}
