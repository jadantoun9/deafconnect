// SpeechRecognizer.swift
//
// Live streaming speech-to-text via the Speech framework, replacing
// deafconnect's `speech_to_text` Flutter plugin. The Flutter version
// emits partial recognition results as the user speaks; we mirror that
// by publishing `transcript` continuously while recording.
//
// Permission flow: SFSpeechRecognizer.requestAuthorization() asks once
// per install. AVAudioSession.requestRecordPermission() is the mic-side
// permission; both must be granted before recording. Both prompts pull
// the strings from Info.plist (NSSpeechRecognitionUsageDescription and
// NSMicrophoneUsageDescription, set in ios-app/project.yml).
import AVFoundation
import Combine
import Foundation
import Speech

@MainActor
public final class SpeechRecognizer: ObservableObject {

    public enum RecognizerError: Error, LocalizedError {
        case notAuthorized
        case noRecognizer
        case audioSession(Error)
        case engine(Error)
        case requestFailed(Error)

        public var errorDescription: String? {
            switch self {
            case .notAuthorized:        return "Speech recognition isn't authorized. Enable it in Settings → SignReader."
            case .noRecognizer:         return "Speech recognition isn't available for the current locale."
            case .audioSession(let e):  return "Audio session failed: \(e.localizedDescription)"
            case .engine(let e):        return "Audio engine failed: \(e.localizedDescription)"
            case .requestFailed(let e): return "Recognition failed: \(e.localizedDescription)"
            }
        }
    }

    @Published public private(set) var transcript: String = ""
    @Published public private(set) var isRecording: Bool = false
    @Published public private(set) var lastError: String?

    private let recognizer: SFSpeechRecognizer?
    private let audioEngine = AVAudioEngine()
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    /// Set when the user calls `stop()` deliberately. Any error fired by
    /// the recognition task after this point is the cancellation echo
    /// (e.g. kAFAssistantErrorDomain 1110 "no speech detected") and must
    /// not bubble up as a user-facing alert.
    private var stoppedByUser: Bool = false

    public init(locale: Locale = Locale(identifier: "en-US")) {
        recognizer = SFSpeechRecognizer(locale: locale)
    }

    /// Begin recording. Idempotent: calling while already recording is a
    /// no-op. Throws on permission / hardware failures so callers can
    /// surface a sensible message.
    public func start() async throws {
        guard !isRecording else { return }
        guard let recognizer else { throw RecognizerError.noRecognizer }
        guard recognizer.isAvailable else { throw RecognizerError.noRecognizer }

        try await ensureAuthorized()

        // Reset published state up front so the UI clears between sessions.
        transcript = ""
        lastError = nil
        stoppedByUser = false

        // Configure audio session for record. .measurement minimises system
        // processing (echo cancellation, etc.) which gives cleaner input
        // for the recognizer.
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.record, mode: .measurement, options: [.duckOthers])
            try session.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            throw RecognizerError.audioSession(error)
        }

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        if recognizer.supportsOnDeviceRecognition {
            request.requiresOnDeviceRecognition = true
        }
        self.request = request

        let inputNode = audioEngine.inputNode
        let format = inputNode.outputFormat(forBus: 0)
        // Tap buffers and feed them to the recognizer.
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak request] buffer, _ in
            request?.append(buffer)
        }

        audioEngine.prepare()
        do {
            try audioEngine.start()
        } catch {
            cleanup()
            throw RecognizerError.engine(error)
        }

        task = recognizer.recognitionTask(with: request) { [weak self] result, error in
            guard let self else { return }
            Task { @MainActor in
                if let result {
                    self.transcript = result.bestTranscription.formattedString
                    if result.isFinal {
                        self.cleanup()
                    }
                }
                if let error {
                    // Swallow the cancellation echo when the user just
                    // tapped stop — Speech reports a final error in that
                    // case ("no speech detected" / "request was cancelled")
                    // even though the transcription succeeded.
                    if !self.stoppedByUser {
                        self.lastError = error.localizedDescription
                    }
                    self.cleanup()
                }
            }
        }

        isRecording = true
    }

    /// Stop recording and return the final transcript captured so far.
    @discardableResult
    public func stop() -> String {
        stoppedByUser = true
        let finalText = transcript
        cleanup()
        return finalText
    }

    public func resetTranscript() {
        transcript = ""
    }

    // ── Internals ────────────────────────────────────────────────────

    private func cleanup() {
        if audioEngine.isRunning {
            audioEngine.stop()
            audioEngine.inputNode.removeTap(onBus: 0)
        }
        request?.endAudio()
        request = nil
        task?.cancel()
        task = nil
        isRecording = false

        // Yield the audio session so TextToSpeech (.playback) can take it
        // back without fighting.
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    private func ensureAuthorized() async throws {
        // Speech recognition authorization
        let speechStatus = await Self.requestSpeechAuth()
        guard speechStatus == .authorized else { throw RecognizerError.notAuthorized }

        // Microphone authorization (separate prompt)
        let micGranted = await Self.requestMicAuth()
        guard micGranted else { throw RecognizerError.notAuthorized }
    }

    private static func requestSpeechAuth() async -> SFSpeechRecognizerAuthorizationStatus {
        await withCheckedContinuation { cont in
            SFSpeechRecognizer.requestAuthorization { status in cont.resume(returning: status) }
        }
    }

    private static func requestMicAuth() async -> Bool {
        await withCheckedContinuation { cont in
            AVAudioApplication.requestRecordPermission { granted in cont.resume(returning: granted) }
        }
    }
}
