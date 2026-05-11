// BuildInfo.swift
//
// Surfaces the app's build timestamp at runtime. The app binary's
// modification date is the build time — read once at launch so a small
// chip in the top-right corner of the app shows the owner exactly which
// build is installed. Helps confirm fresh installs vs stale ones during
// rapid Phase 0/Phase 1 iteration.
import Foundation

public enum BuildInfo {
    public static let buildString: String = {
        guard let url = Bundle.main.executableURL,
              let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
              let date = attrs[.modificationDate] as? Date else {
            return "build ?"
        }
        let formatter = DateFormatter()
        formatter.dateFormat = "MMM d HH:mm:ss"
        return "build \(formatter.string(from: date))"
    }()
}
