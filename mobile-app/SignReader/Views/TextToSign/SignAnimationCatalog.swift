// SignAnimationCatalog.swift
//
// Map of animation name → expected playback duration. Ported verbatim from
// deafconnect's lib/utils/animation_durations.dart so the Swift translator
// produces the same word-by-word timing as the Flutter version.
//
// Keys must match the animation names baked into the avatar GLB files. The
// translator capitalises the first letter of each word ("hello" → "Hello")
// before lookup so user input doesn't have to worry about case. If a word
// isn't found, the translator falls back to per-letter fingerspelling.
import Foundation

enum SignAnimationCatalog {
    /// Per-animation duration in seconds. Falls back to `defaultDuration`
    /// when a key is missing.
    static let durations: [String: TimeInterval] = [
        // Fingerspelling (A-Z)
        "A": 1.8, "B": 2.3, "C": 2.3, "D": 1.8, "E": 2.1,
        "F": 2.0, "G": 2.5, "H": 2.5, "I": 1.8, "J": 2.2,
        "K": 1.2, "L": 2.1, "M": 2.0, "N": 2.0, "O": 2.0,
        "P": 2.0, "Q": 2.3, "R": 1.8, "S": 1.8, "T": 2.1,
        "U": 2.0, "V": 2.0, "W": 1.6, "X": 1.9, "Y": 2.3,
        "Z": 2.0,
        // Word-level animations the Flutter version ships with the avatar.
        "Hello": 1.6, "How": 1.6, "Meet": 2.1, "Me": 2.0,
        "Name": 1.0, "Nice": 1.4, "No": 1.2, "Please": 1.7,
        "Want": 1.6, "Where": 1.5, "Yes": 1.45, "You": 1.4,
    ]

    /// Used when `durations[name]` is nil. Two seconds is what deafconnect's
    /// `?? const Duration(seconds: 2)` pattern resolves to.
    static let defaultDuration: TimeInterval = 2.0

    /// "hello" → "Hello", " hi" → "Hi". Mirrors deafconnect's
    /// `capitalizeFirstLetter` in lib/utils/utils.dart.
    static func capitalizeFirstLetter(_ word: String) -> String {
        guard let first = word.first else { return word }
        return first.uppercased() + word.dropFirst()
    }

    /// Look up the duration for an animation by its raw name (already cased).
    static func duration(for name: String) -> TimeInterval {
        durations[name] ?? defaultDuration
    }
}
