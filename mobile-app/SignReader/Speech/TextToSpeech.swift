// TextToSpeech.swift
//
// Thin wrapper over AVSpeechSynthesizer for the deafconnect Chat tab. The
// Flutter version uses `flutter_tts`, which on iOS configures the playback
// audio category with allowBluetooth + allowBluetoothA2DP + mixWithOthers
// + defaultToSpeaker. We mirror that here so external Bluetooth speakers /
// AirPods / car audio behave the same way.
import AVFoundation
import Combine
import Foundation

@MainActor
public final class TextToSpeech: ObservableObject {
    @Published public private(set) var isSpeaking: Bool = false

    private let synthesizer = AVSpeechSynthesizer()
    private let coordinator = SynthesizerCoordinator()

    public init() {
        // Wire delegate without making `self` the delegate (which would force
        // the class to be NSObjectProtocol-conforming and pull UIKit into the
        // initializer). The coordinator publishes start/stop events back into
        // our @Published flag via a closure.
        coordinator.onSpeakingChanged = { [weak self] speaking in
            Task { @MainActor in self?.isSpeaking = speaking }
        }
        synthesizer.delegate = coordinator
        configureAudioSession()
    }

    /// Speak `text` with the given language voice. Cuts off the previous
    /// utterance if one is in flight — matches deafconnect's
    /// `flutterTts.speak()` behaviour.
    public func speak(_ text: String, language: String = "en-US", rate: Float = AVSpeechUtteranceDefaultSpeechRate, pitch: Float = 1.0) {
        guard !text.isEmpty else { return }
        // Re-arm the audio session every speak. SpeechRecognizer flips the
        // shared session to .record (and deactivates it on stop), which
        // leaves the synthesizer with no active playback session — silent
        // failure on the next utterance otherwise.
        configureAudioSession()
        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
        }
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: language) ?? AVSpeechSynthesisVoice(language: "en-US")
        utterance.rate = rate
        utterance.pitchMultiplier = pitch
        synthesizer.speak(utterance)
    }

    public func stop() {
        synthesizer.stopSpeaking(at: .immediate)
    }

    /// Configure once at construction. The .playback category is what
    /// deafconnect requested via `setIosAudioCategory(playback, ...)`.
    /// .duckOthers is preferred over .mixWithOthers in modern iOS — it
    /// lowers other audio rather than fighting it.
    private func configureAudioSession() {
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers, .allowBluetoothA2DP, .defaultToSpeaker])
            try session.setActive(true, options: [])
        } catch {
            // Audio session can fail on simulator / when STT is mid-record.
            // Don't crash; the synthesizer falls back to whatever the system
            // gives it. Logged via NSLog to keep this module UIKit-free.
            NSLog("TextToSpeech: audio session config failed: \(error.localizedDescription)")
        }
    }
}

private final class SynthesizerCoordinator: NSObject, AVSpeechSynthesizerDelegate {
    var onSpeakingChanged: ((Bool) -> Void)?

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        onSpeakingChanged?(true)
    }
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        onSpeakingChanged?(false)
    }
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        onSpeakingChanged?(false)
    }
}
