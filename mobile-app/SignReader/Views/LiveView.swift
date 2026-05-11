// LiveView.swift
//
// The home tab. Shows the live camera, the most recent prediction
// (label + confidence bar), a scrolling history log, and inline warnings
// for low-light / thermal / memory-pressure conditions.
import SwiftUI

public struct LiveView: View {
    @EnvironmentObject private var app: AppState

    public init() {}

    public var body: some View {
        NavigationStack {
            VStack(spacing: 12) {
                EnvironmentBanner()
                    .environmentObject(app.camera)

                preview

                predictionPanel

                historyPanel

                Spacer()
            }
            .padding()
            .navigationTitle("Live")
            // Camera lifecycle is driven by scenePhase in SignReaderApp,
            // not per-view. Tab switching no longer reconfigures the
            // session, which was the source of the "Camera idle" hang.
        }
    }

    @ViewBuilder
    private var preview: some View {
        ZStack {
            // Always attach the preview layer to the session. Once the
            // session starts running on its background queue the layer
            // shows frames immediately — independent of the @Published
            // status update which can lag a few hundred ms. Toggling the
            // preview on `status == .running` used to leave the view stuck
            // on the placeholder.
            CameraPreview(session: app.camera.session)
                .aspectRatio(3.0 / 4.0, contentMode: .fit)
                .cornerRadius(12)
            if app.camera.status != .running {
                placeholder
            }
            if app.engine.isInferring {
                Image(systemName: "waveform")
                    .imageScale(.large)
                    .padding(8)
                    .background(.ultraThinMaterial, in: Circle())
                    .padding()
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
            }
        }
    }

    @ViewBuilder
    private var placeholder: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 12).fill(.gray.opacity(0.2))
            VStack(spacing: 8) {
                Image(systemName: cameraIcon).font(.largeTitle)
                Text(cameraMessage).font(.footnote).multilineTextAlignment(.center)
            }
            .padding()
        }
        .aspectRatio(3.0 / 4.0, contentMode: .fit)
    }

    private var cameraIcon: String {
        switch app.camera.status {
        case .denied: return "camera.metering.center.weighted"
        case .unavailable: return "exclamationmark.triangle.fill"
        default: return "camera.fill"
        }
    }
    private var cameraMessage: String {
        switch app.camera.status {
        case .unconfigured: return "Camera idle."
        case .authorized:   return "Configuring camera…"
        case .running:      return ""
        case .denied:       return "Camera access denied. Enable it in Settings."
        case .unavailable(let r): return "Camera unavailable: \(r)"
        }
    }

    @ViewBuilder
    private var predictionPanel: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Text("Prediction").font(.headline)
                Spacer()
                if let p = app.engine.prediction {
                    Text(String(format: "%.0f ms", p.latencyMs))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            HStack(alignment: .firstTextBaseline) {
                Text(app.engine.prediction?.label ?? "—")
                    .font(.title.bold())
                Spacer()
                if let p = app.engine.prediction {
                    Text(String(format: "%.0f%%", p.confidence * 100))
                        .font(.title3)
                        .foregroundStyle(.secondary)
                }
            }
            ProgressView(value: Double(app.engine.prediction?.confidence ?? 0))
                .progressViewStyle(.linear)
        }
        .padding()
        .background(.secondary.opacity(0.1))
        .cornerRadius(12)
    }

    @ViewBuilder
    private var historyPanel: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Recent").font(.headline)
                Spacer()
                Button("Clear") { app.engine.clearHistory() }
                    .buttonStyle(.borderless)
                    .disabled(app.engine.history.isEmpty)
            }
            if app.engine.history.isEmpty {
                Text("No predictions yet.")
                    .foregroundStyle(.secondary)
                    .font(.footnote)
            } else {
                ScrollView(.vertical) {
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(app.engine.history) { item in
                            HStack {
                                Text(item.label)
                                Spacer()
                                Text(String(format: "%.0f%%", item.confidence * 100))
                                    .foregroundStyle(.secondary)
                            }
                            .font(.subheadline.monospacedDigit())
                        }
                    }
                }
                .frame(maxHeight: 160)
            }
        }
        .padding()
        .background(.secondary.opacity(0.05))
        .cornerRadius(12)
    }
}

private struct EnvironmentBanner: View {
    @EnvironmentObject private var camera: CameraManager

    var body: some View {
        VStack(spacing: 4) {
            if camera.thermalState == .serious || camera.thermalState == .critical {
                bannerLine(systemImage: "thermometer.high",
                           text: "Device is hot — performance may drop.",
                           color: .orange)
            }
            if camera.memoryPressure == .warning {
                bannerLine(systemImage: "memorychip",
                           text: "Low memory — model accuracy may degrade.",
                           color: .yellow)
            } else if camera.memoryPressure == .critical {
                bannerLine(systemImage: "memorychip",
                           text: "Critical memory pressure.",
                           color: .red)
            }
            if camera.isLowLight {
                bannerLine(systemImage: "sun.min",
                           text: "Low light — try moving to a brighter area.",
                           color: .blue)
            }
            if let err = camera.lastError {
                bannerLine(systemImage: "exclamationmark.triangle.fill",
                           text: err,
                           color: .red)
            }
        }
    }

    private func bannerLine(systemImage: String, text: String, color: Color) -> some View {
        Label(text, systemImage: systemImage)
            .font(.footnote)
            .foregroundStyle(color)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.vertical, 6)
            .padding(.horizontal, 10)
            .background(color.opacity(0.12))
            .cornerRadius(8)
    }
}
