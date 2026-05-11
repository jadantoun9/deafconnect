// CrashLogger.swift
//
// File-based crash logger. Wires into NSSetUncaughtExceptionHandler and the
// six common Unix signals (SIGABRT, SIGSEGV, SIGBUS, SIGILL, SIGFPE,
// SIGPIPE) so an unrecoverable crash leaves a forensic trail in the app's
// Documents directory before the process exits.
//
// What it can NOT catch:
//   - Watchdog kills from iOS (you'll see a missing tail on the previous
//     log). Investigate via the Xcode Devices window and console.
//   - Pure Swift fatalError / preconditionFailure that bypass NSException —
//     these terminate via abort(), which raises SIGABRT, which we catch.
//
// Log location:
//   Documents/SignReader/crashlogs/<YYYY-MM-DD-HHMMSS>.log
//
// Each entry has: timestamp, kind, signal/exception name, call stack.
import Foundation

public enum CrashLogger {

    private static let queue = DispatchQueue(label: "signreader.crashlogger", qos: .utility)
    private static let formatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd-HHmmss"
        f.locale = Locale(identifier: "en_US_POSIX")
        return f
    }()

    /// Call once from the SwiftUI App initializer.
    public static func install() {
        NSSetUncaughtExceptionHandler { exception in
            CrashLogger.write(kind: "uncaught", name: exception.name.rawValue, stack: exception.callStackSymbols)
        }
        for sig in [SIGABRT, SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGPIPE] {
            signal(sig) { signo in
                let name = String(cString: strsignal(signo))
                let trace = Thread.callStackSymbols
                CrashLogger.write(kind: "signal", name: name, stack: trace)
                // Re-raise so iOS still gets the chance to record a normal
                // crash report; we don't want to swallow the exit.
                signal(signo, SIG_DFL)
                raise(signo)
            }
        }
    }

    /// Write a non-crash diagnostic note (useful for "model failed to load"
    /// type events that the user would want to share with the developer).
    public static func note(_ message: String) {
        write(kind: "note", name: message, stack: [])
    }

    /// Returns the directory containing all crash logs, creating it if
    /// missing.
    public static func logDirectory() -> URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("SignReader/crashlogs", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    public static func recentLogs() -> [URL] {
        let dir = logDirectory()
        let urls = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.creationDateKey])) ?? []
        return urls.sorted { (a, b) in
            (try? a.resourceValues(forKeys: [.creationDateKey]).creationDate ?? .distantPast)
                ?? .distantPast
                > (try? b.resourceValues(forKeys: [.creationDateKey]).creationDate ?? .distantPast)
                ?? .distantPast
        }
    }

    private static func write(kind: String, name: String, stack: [String]) {
        let now = Date()
        let stamp = formatter.string(from: now)
        let header = "[\(stamp)] kind=\(kind) name=\(name)"
        let lines = ([header] + stack.map { "  \($0)" }).joined(separator: "\n") + "\n\n"
        // Synchronous on the crashlogger queue: the signal handler path may
        // be called late in the process lifecycle and we want the write to
        // complete before the default action terminates the process.
        queue.sync {
            let dir = logDirectory()
            let url = dir.appendingPathComponent("\(stamp).log")
            if let data = lines.data(using: .utf8) {
                if FileManager.default.fileExists(atPath: url.path),
                   let handle = try? FileHandle(forWritingTo: url) {
                    handle.seekToEndOfFile()
                    handle.write(data)
                    try? handle.close()
                } else {
                    try? data.write(to: url)
                }
            }
        }
    }
}
