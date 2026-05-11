// AppState.swift
//
// Top-level @StateObject for the app. Owns:
//   - the CameraManager (one global instance across views)
//   - the PredictionEngine
//   - the user's tunable preferences (model kind, FPS, threshold, debug)
//
// Why one app-level state object instead of per-view state:
//   the camera and engine are expensive — recreating them every time the
//   user taps between Live and Practice tabs would burn battery and
//   re-trigger the camera permission prompt on the first open.
//
// Model loading: each track will register a constructor by adding to
// ModelLoader.load(kind:). Until Tracks 1/2/3 are implemented, the loader
// returns a PlaceholderModel so the app still builds and the shell is
// exercisable.
//
// Re-publishing nested ObservableObjects:
//   `camera` and `engine` are themselves @Published-ful ObservableObjects.
//   Stored as `let` on AppState, their changes do NOT propagate up — views
//   that read `app.camera.status` would render once and then go stale.
//   We forward each child's objectWillChange into AppState's so views with
//   `@EnvironmentObject var app: AppState` re-render on any nested change.
import Combine
import Foundation
import SwiftUI

@MainActor
public final class AppState: ObservableObject {
    @Published public var selectedKind: ModelKind {
        didSet {
            guard oldValue != selectedKind else { return }
            switchModel(to: selectedKind)
        }
    }
    @Published public var fps: Int {
        didSet {
            UserDefaults.standard.set(fps, forKey: Keys.fps)
            if camera.status == .running {
                camera.stop()
                camera.start(fps: fps) { [weak self] frame in self?.engine.feed(frame) }
            }
        }
    }
    @Published public var confidenceThreshold: Float {
        didSet {
            UserDefaults.standard.set(Double(confidenceThreshold), forKey: Keys.threshold)
            engine.setConfidenceThreshold(confidenceThreshold)
        }
    }
    @Published public var debugMode: Bool {
        didSet { UserDefaults.standard.set(debugMode, forKey: Keys.debug) }
    }
    /// Currently-selected TabView tab. Held in AppState so any tab can
    /// programmatically jump to another (e.g. Transcripts list → Chat).
    /// 0 = Chat, 1 = Transcripts, 2 = Text to Sign, 3 = Sign to Text.
    @Published public var activeTab: Int = 0
    /// Currently-open chat transcript. Stored as a string because
    /// UserDefaults can't hold UUID directly. Use `selectedTranscriptID`
    /// (UUID-typed) for actual lookups.
    @Published public var selectedTranscriptIDString: String? {
        didSet { UserDefaults.standard.set(selectedTranscriptIDString, forKey: Keys.transcriptID) }
    }
    public var selectedTranscriptID: UUID? {
        get { selectedTranscriptIDString.flatMap(UUID.init(uuidString:)) }
        set { selectedTranscriptIDString = newValue?.uuidString }
    }
    /// Avatar choice for Text-to-Sign. Mirrors deafconnect's `isFemale`
    /// flag but typed as an enum-like string for clarity.
    @Published public var avatarKind: AvatarKind {
        didSet { UserDefaults.standard.set(avatarKind.rawValue, forKey: Keys.avatar) }
    }
    /// Background-image asset name for Text-to-Sign. Empty string = no
    /// background (transparent / system default).
    @Published public var textToSignBackground: String {
        didSet { UserDefaults.standard.set(textToSignBackground, forKey: Keys.background) }
    }
    /// Which underlying classifier powers the "Words" option in the
    /// Sign-to-Text picker. User-facing UI always says "Words" — this
    /// setting only decides which of the three word-level models
    /// (landmark v1, landmark v2, video Transformer) actually runs.
    @Published public var wordsModelVariant: WordsModelVariant {
        didSet { UserDefaults.standard.set(wordsModelVariant.rawValue, forKey: Keys.wordsVariant) }
    }

    public let camera: CameraManager
    public let engine: PredictionEngine

    private var nestedSubscriptions: Set<AnyCancellable> = []

    public init() {
        let savedKind = UserDefaults.standard.string(forKey: Keys.kind).flatMap(ModelKind.init(rawValue:)) ?? .landmark
        let savedFPS = max(10, UserDefaults.standard.integer(forKey: Keys.fps).nonZeroOr(20))
        let savedThreshold = Float(UserDefaults.standard.double(forKey: Keys.threshold).nonZeroOr(0.6))
        self.selectedKind = savedKind
        self.fps = savedFPS
        self.confidenceThreshold = savedThreshold
        self.debugMode = UserDefaults.standard.bool(forKey: Keys.debug)
        self.selectedTranscriptIDString = UserDefaults.standard.string(forKey: Keys.transcriptID)
        self.avatarKind = UserDefaults.standard.string(forKey: Keys.avatar)
            .flatMap(AvatarKind.init(rawValue:)) ?? .girl
        self.textToSignBackground = UserDefaults.standard.string(forKey: Keys.background) ?? ""
        self.wordsModelVariant = UserDefaults.standard.string(forKey: Keys.wordsVariant)
            .flatMap(WordsModelVariant.init(rawValue:)) ?? .landmarkV2

        let camera = CameraManager()
        let model = ModelLoader.load(kind: savedKind)
        let engine = PredictionEngine(model: model, confidenceThreshold: savedThreshold)
        self.camera = camera
        self.engine = engine

        // Forward nested child ObservableObject changes so views observing
        // AppState re-render when camera.status / engine.prediction change.
        camera.objectWillChange
            .sink { [weak self] _ in self?.objectWillChange.send() }
            .store(in: &nestedSubscriptions)
        engine.objectWillChange
            .sink { [weak self] _ in self?.objectWillChange.send() }
            .store(in: &nestedSubscriptions)
    }

    public func startCamera() {
        camera.start(fps: fps) { [weak self] frame in
            self?.engine.feed(frame)
        }
    }

    public func stopCamera() { camera.stop() }

    private func switchModel(to kind: ModelKind) {
        UserDefaults.standard.set(kind.rawValue, forKey: Keys.kind)
        let next = ModelLoader.load(kind: kind)
        engine.setModel(next)
    }

    private enum Keys {
        static let kind = "signreader.modelKind"
        static let fps = "signreader.fps"
        static let threshold = "signreader.confidenceThreshold"
        static let debug = "signreader.debugMode"
        static let transcriptID = "signreader.selectedTranscriptID"
        static let avatar = "signreader.avatarKind"
        static let background = "signreader.textToSignBackground"
        static let wordsVariant = "signreader.wordsModelVariant"
    }
}

/// The three underlying classifiers that can sit behind the
/// user-facing "Words" picker in Sign-to-Text. Selected from Settings.
public enum WordsModelVariant: String, CaseIterable, Identifiable {
    case landmarkV1
    case landmarkV2
    case video

    public var id: String { rawValue }
    public var displayName: String {
        switch self {
        case .landmarkV1: return "Landmark v1"
        case .landmarkV2: return "Landmark v2"
        case .video:      return "Video"
        }
    }
}

/// Boy / girl rigged avatar in the Text-to-Sign tab.
public enum AvatarKind: String, CaseIterable, Identifiable {
    case boy
    case girl

    public var id: String { rawValue }
    /// Bundle resource name (without extension) for the .glb file.
    public var glbName: String { rawValue }
    /// Asset-catalog image used in the avatar picker thumbnail.
    public var thumbnailAsset: String {
        switch self {
        case .boy:  return "AvatarBoyThumb"
        case .girl: return "AvatarGirlThumb"
        }
    }
    public var displayName: String {
        switch self {
        case .boy:  return "Boy"
        case .girl: return "Girl"
        }
    }
}

private extension Int {
    func nonZeroOr(_ fallback: Int) -> Int { self == 0 ? fallback : self }
}
private extension Double {
    func nonZeroOr(_ fallback: Double) -> Double { self == 0 ? fallback : self }
}
