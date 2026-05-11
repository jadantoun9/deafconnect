// SettingsLauncher.swift
//
// Drop-in modifier each NavigationStack-rooted tab applies to surface the
// gear icon → Settings sheet. SwiftUI tab views don't have a single global
// toolbar, so per-tab is the cleanest path. The sheet itself is light and
// the SettingsView resets correctly on each presentation.
import SwiftUI

private struct WithSettingsLauncher: ViewModifier {
    @State private var showingSettings = false

    func body(content: Content) -> some View {
        content
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        showingSettings = true
                    } label: {
                        Image(systemName: "gearshape.fill")
                            .foregroundStyle(Color.brandMain)
                    }
                    .accessibilityLabel("Settings")
                }
            }
            .sheet(isPresented: $showingSettings) {
                SettingsView()
            }
    }
}

extension View {
    /// Adds a top-trailing gear button that presents `SettingsView` as a
    /// sheet. Apply to the inner content of each tab's NavigationStack.
    func withSettingsLauncher() -> some View { modifier(WithSettingsLauncher()) }
}
