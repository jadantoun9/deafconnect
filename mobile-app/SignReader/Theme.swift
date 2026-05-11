// Theme.swift
//
// App-wide colour palette, ported from deafconnect's lib/utils/colors.dart so
// the SwiftUI port reads visually identical to the Flutter original. One
// source of truth — every view should reference these constants instead of
// hard-coding hex literals.
import SwiftUI

extension Color {
    /// #2469E8 — primary brand colour (chat bubbles, send button, active tab tint).
    static let brandMain = Color(red: 0x24 / 255, green: 0x69 / 255, blue: 0xE8 / 255)
    /// #E8E8E8 — soft grey background used on chat scaffold and the text-to-sign tray.
    static let brandSecondary = Color(red: 0xE8 / 255, green: 0xE8 / 255, blue: 0xE8 / 255)
    /// #939393 — placeholder text and disabled state.
    static let brandLightGray = Color(red: 0x93 / 255, green: 0x93 / 255, blue: 0x93 / 255)
    /// #CECECE — fallback avatar background when no image is set.
    static let brandAvatarBg = Color(red: 0xCE / 255, green: 0xCE / 255, blue: 0xCE / 255)
}
