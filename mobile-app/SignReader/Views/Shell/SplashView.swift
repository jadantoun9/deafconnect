// SplashView.swift
//
// Brief launch splash mirroring deafconnect's
// lib/screens/splash_screen.screen.dart. Shows the logo on the brand
// background, fades into the root tab view after ~1.5 s. The wait isn't a
// loading bar — it's a beat so the first frame the user sees isn't a hard
// cut to a tab bar.
import SwiftUI

struct SplashView: View {
    @State private var hasFinished = false
    /// Tunable: shorten in dev builds if it gets in the way.
    var duration: TimeInterval = 1.5

    var body: some View {
        Group {
            if hasFinished {
                RootView()
                    .transition(.opacity)
            } else {
                splashBody
                    .transition(.opacity)
            }
        }
        .animation(.easeInOut(duration: 0.35), value: hasFinished)
    }

    private var splashBody: some View {
        ZStack {
            // Brand-blue background to match the logo, replacing the
            // earlier light-grey scaffold colour.
            Color.brandMain.ignoresSafeArea()
            VStack(spacing: 18) {
                Image("AppLogo")
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: 180, maxHeight: 180)
                Text("DeafConnect")
                    .font(.system(size: 26, weight: .semibold, design: .rounded))
                    .foregroundStyle(.white)
            }
        }
        .task {
            try? await Task.sleep(for: .seconds(duration))
            hasFinished = true
        }
    }
}
