// TranscriptDialog.swift
//
// Small sheet to create or rename a chat transcript. Used both from the
// Chat tab's "+" toolbar and from Phase 6's Transcripts list (rename).
// Mirrors deafconnect's lib/widgets/dialogs/transcript_dialog.dart.
import SwiftData
import SwiftUI

struct TranscriptDialog: View {
    @Environment(\.dismiss) private var dismiss

    /// Pre-filled name when editing an existing transcript. Empty for "create".
    let initialName: String
    /// Callback runs with the user-supplied name. Caller decides whether to
    /// insert (create flow) or update (rename flow). Empty input dismisses.
    let onSubmit: (String) -> Void

    @State private var text: String = ""
    @FocusState private var focused: Bool

    init(initialName: String = "", onSubmit: @escaping (String) -> Void) {
        self.initialName = initialName
        self.onSubmit = onSubmit
        _text = State(initialValue: initialName)
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 18) {
                Text(initialName.isEmpty ? "New conversation" : "Rename conversation")
                    .font(.headline)
                    .padding(.top, 8)

                TextField("Conversation name", text: $text)
                    .textFieldStyle(.roundedBorder)
                    .focused($focused)
                    .submitLabel(.done)
                    .onSubmit(commit)

                HStack(spacing: 12) {
                    Button("Cancel", role: .cancel) { dismiss() }
                        .frame(maxWidth: .infinity)
                        .buttonStyle(.bordered)
                    Button(initialName.isEmpty ? "Create" : "Save", action: commit)
                        .frame(maxWidth: .infinity)
                        .buttonStyle(.borderedProminent)
                        .tint(Color.brandMain)
                        .disabled(text.trimmingCharacters(in: .whitespaces).isEmpty)
                }
                Spacer()
            }
            .padding()
            .presentationDetents([.height(220)])
            .onAppear { focused = true }
        }
    }

    private func commit() {
        let trimmed = text.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return }
        onSubmit(trimmed)
        dismiss()
    }
}
