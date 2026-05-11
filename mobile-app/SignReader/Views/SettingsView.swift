// SettingsView.swift
//
// Tunables exposed to the user. All settings persist via AppState's
// UserDefaults plumbing.
import SwiftUI

public struct SettingsView: View {
    @EnvironmentObject private var app: AppState

    public init() {}

    public var body: some View {
        NavigationStack {
            Form {
                Section("Model") {
                    Picker("Words model", selection: $app.wordsModelVariant) {
                        ForEach(WordsModelVariant.allCases) { v in
                            Text(v.displayName).tag(v)
                        }
                    }
                }

                Section("Camera") {
                    Stepper("Frame rate: \(app.fps) fps", value: $app.fps, in: 10...60, step: 5)
                }

                Section("About") {
                    HStack { Text("Version"); Spacer(); Text(appVersion).foregroundStyle(.secondary) }
                    HStack { Text("Build");   Spacer(); Text(appBuild).foregroundStyle(.secondary) }
                }
            }
            .navigationTitle("Settings")
        }
    }

    private var appVersion: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "—"
    }
    private var appBuild: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "—"
    }
}
