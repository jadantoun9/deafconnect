// SwiftUI app entry point. Phase 0.8 brings back the three production tabs
// (Live / Practice / Settings) sharing a single AppState. SanityCheckView is
// reachable from Settings -> Debug for the Phase 0.3 gate.
//
// Camera lifecycle is driven by scenePhase here, not per-view onAppear /
// onDisappear. Live and Practice both consume the same shared camera, so
// tearing it down on tab switches caused 1-2s of lag and a race that left
// the session stuck in `.unconfigured` ("Camera idle") on the way back.
//
// Persistence: SwiftData container is attached at the WindowGroup level so
// every view in the deafconnect-port tab tree can `@Query` the entities.
// Schema is registered through `PersistenceSchema` (Persistence/Models.swift).
import SwiftData
import SwiftUI

@main
struct SignReaderApp: App {
    @StateObject private var appState = AppState()
    @Environment(\.scenePhase) private var scenePhase

    /// One container, shared across the app. Crashes loudly if construction
    /// fails — there's no graceful degradation for "the database is broken"
    /// in a chat app.
    private let modelContainer: ModelContainer = {
        let schema = Schema([
            TranscriptEntity.self,
            MessageEntity.self,
            ShortcutEntity.self,
            AvatarMessageEntity.self,
        ])
        let config = ModelConfiguration(schema: schema, isStoredInMemoryOnly: false)
        do {
            return try ModelContainer(for: schema, configurations: [config])
        } catch {
            fatalError("SwiftData ModelContainer init failed: \(error)")
        }
    }()

    init() {
        CrashLogger.install()
    }

    var body: some Scene {
        WindowGroup {
            SplashView()
                .environmentObject(appState)
                .environmentObject(appState.camera)
                // Force light mode app-wide. The deafconnect port wasn't
                // designed against both appearances, so dynamic system
                // colours were rendering chat bubbles white-on-white in
                // dark mode. Pinning the scheme keeps things consistent
                // until the theme is properly two-tone.
                .preferredColorScheme(.light)
                .onChange(of: scenePhase) { phase in
                    switch phase {
                    case .active:
                        appState.startCamera()
                    case .background:
                        appState.stopCamera()
                    case .inactive:
                        // Brief transitions (control center, notifications).
                        // Keep the camera running so we don't pay for a
                        // reconfigure when the user comes right back.
                        break
                    @unknown default:
                        break
                    }
                }
        }
        .modelContainer(modelContainer)
    }
}

struct RootView: View {
    @EnvironmentObject private var app: AppState

    var body: some View {
        TabView(selection: $app.activeTab) {
            ChatView()
                .tabItem {
                    Label("Chat", systemImage: "bubble.left.and.bubble.right.fill")
                }
                .tag(0)
            TranscriptsView()
                .tabItem {
                    Label("Transcripts", systemImage: "list.bullet.rectangle")
                }
                .tag(1)
            TextToSignView()
                .tabItem {
                    Label("Text to Sign", systemImage: "figure.wave.circle.fill")
                }
                .tag(2)
            SignToTextView()
                .tabItem {
                    Label("Sign to Text", systemImage: "camera.viewfinder")
                }
                .tag(3)
        }
        .tint(Color.brandMain)
    }
}
