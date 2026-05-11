// Models.swift
//
// SwiftData @Model entities ported from deafconnect's
// lib/models/{message,transcript,shortcut,avatar_message}.model.dart.
//
// Why a single file: the four models are small (under ~30 lines each) and
// related; splitting buys nothing, and SwiftData's relationship inference
// works fine when they live in the same module.
//
// Why SwiftData rather than Core Data: SwiftUI-native, no .xcdatamodeld file
// to maintain alongside Swift sources, and the project deployment target is
// iOS 17 (per the deafconnect-port plan).
import Foundation
import SwiftData

/// One conversation thread. Holds an ordered list of `Message`s.
/// Mirrors deafconnect's `Transcript` model.
@Model
public final class TranscriptEntity {
    @Attribute(.unique) public var id: UUID
    public var name: String
    public var dateCreated: Date
    /// Cascade so deleting a transcript drops its messages.
    @Relationship(deleteRule: .cascade, inverse: \MessageEntity.transcript)
    public var messages: [MessageEntity] = []

    public init(id: UUID = UUID(), name: String, dateCreated: Date = .now) {
        self.id = id
        self.name = name
        self.dateCreated = dateCreated
    }
}

/// One chat bubble. `isReceived = true` means the other party signed/spoke,
/// `false` means the app user typed/dictated. Mirrors `Message` model.
@Model
public final class MessageEntity {
    @Attribute(.unique) public var id: UUID
    public var content: String
    public var isReceived: Bool
    public var date: Date
    public var transcript: TranscriptEntity?

    public init(
        id: UUID = UUID(),
        content: String,
        isReceived: Bool,
        date: Date = .now,
        transcript: TranscriptEntity? = nil
    ) {
        self.id = id
        self.content = content
        self.isReceived = isReceived
        self.date = date
        self.transcript = transcript
    }
}

/// Saved phrase shown in the chat shortcut grid for one-tap sending.
/// Mirrors `Shortcut` model.
@Model
public final class ShortcutEntity {
    @Attribute(.unique) public var id: UUID
    public var name: String
    public var dateCreated: Date

    public init(id: UUID = UUID(), name: String, dateCreated: Date = .now) {
        self.id = id
        self.name = name
        self.dateCreated = dateCreated
    }
}

/// One past text-to-sign translation, surfaced in the avatar history sheet.
/// Mirrors `AvatarMessage` model.
@Model
public final class AvatarMessageEntity {
    @Attribute(.unique) public var id: UUID
    public var text: String
    public var date: Date

    public init(id: UUID = UUID(), text: String, date: Date = .now) {
        self.id = id
        self.text = text
        self.date = date
    }
}

/// All entity types the app's `ModelContainer` should know about. Centralised
/// so AppState can construct the container in one call.
public enum PersistenceSchema {
    public static let entities: [any PersistentModel.Type] = [
        TranscriptEntity.self,
        MessageEntity.self,
        ShortcutEntity.self,
        AvatarMessageEntity.self,
    ]
}
