// Avatar3DView.swift
//
// SwiftUI host for the rigged glTF avatar (boy.glb / girl.glb), powered by
// GLTFKit2's SceneKit bridge. Replaces deafconnect's Flutter3DViewer.
//
// Why a UIViewRepresentable rather than a SwiftUI SceneView: SceneView
// (iOS 17+) doesn't expose a way to imperatively call SCNAnimationPlayer
// methods on inner nodes mid-frame, which is what we need for the
// "play this animation, await its duration, play the next" pipeline.
// Wrapping SCNView ourselves keeps that control.
//
// The View exposes a small command-pump API via `@Binding var commands`:
// the parent appends `.play("Hello")` / `.idle` / `.stop` items, and the
// coordinator drains them in `updateUIView`. This pattern is more robust
// than direct method calls because UIViewRepresentable can rebuild the
// representable struct at any time without rebuilding the underlying
// UIView.
import GLTFKit2
import SceneKit
import SwiftUI

/// Drive the avatar from SwiftUI by appending commands to a binding. The
/// view's coordinator drains the queue on every `updateUIView` pass.
enum Avatar3DCommand: Equatable {
    case play(animationName: String)
    case idle
    case stop
}

struct Avatar3DView: UIViewRepresentable {
    /// Asset name of the .glb file (without extension) — "boy" or "girl".
    let glbName: String
    @Binding var commands: [Avatar3DCommand]

    func makeUIView(context: Context) -> SCNView {
        let view = SCNView(frame: .zero)
        view.backgroundColor = .clear
        view.autoenablesDefaultLighting = true
        // Built-in pan-to-orbit + pinch-to-zoom. The user expects to be
        // able to spin the avatar by dragging on it.
        view.allowsCameraControl = true
        view.antialiasingMode = .multisampling2X
        view.preferredFramesPerSecond = 60
        load(into: view, glbName: glbName, coordinator: context.coordinator)
        return view
    }

    func updateUIView(_ uiView: SCNView, context: Context) {
        // If the avatar resource changed, swap the loaded scene.
        if context.coordinator.currentGLB != glbName {
            load(into: uiView, glbName: glbName, coordinator: context.coordinator)
        }
        // Drain the command queue. Commands are immutable values; clearing
        // the binding tells the parent we're done with them.
        if !commands.isEmpty {
            for cmd in commands {
                context.coordinator.run(cmd)
            }
            DispatchQueue.main.async { commands = [] }
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    // MARK: - Loading

    /// Load the .glb via GLTFKit2 into an SCNScene attached to the SCNView.
    /// Captures the avatar's animation map onto the coordinator so we can
    /// drive playback by name.
    private func load(into view: SCNView, glbName: String, coordinator: Coordinator) {
        guard let url = Bundle.main.url(forResource: glbName, withExtension: "glb")
                ?? Bundle.main.url(forResource: glbName, withExtension: "gltf")
                ?? Bundle.main.url(forResource: "Avatars/\(glbName)", withExtension: "glb")
        else {
            NSLog("Avatar3DView: avatar glb '\(glbName)' not found in bundle.")
            return
        }

        // GLTFAsset.load(with:options:handler:) is the asynchronous loader.
        // For our small (~11 MB) avatars synchronous load on the main
        // thread is fine but the API is still async — wrap it.
        GLTFAsset.load(with: url, options: [:]) { progress, status, asset, error, _ in
            guard status == .complete, let asset else {
                if let error { NSLog("Avatar3DView: GLTFAsset.load failed: \(error.localizedDescription)") }
                return
            }
            DispatchQueue.main.async {
                let source = GLTFSCNSceneSource(asset: asset)
                guard let scene = source.defaultScene else {
                    NSLog("Avatar3DView: no default scene in \(glbName)")
                    return
                }
                view.scene = scene
                framUpperBody(in: view, scene: scene)
                coordinator.attach(source: source, scene: scene, glbName: glbName)
            }
        }
    }

    /// Add a custom camera positioned to frame the avatar's head + arms,
    /// then pin it as the SCNView's pointOfView. Crops at the SceneKit
    /// rendering level so the SwiftUI layout (background + bottom bar +
    /// keyboard) is never affected.
    ///
    /// With allowsCameraControl=true, this becomes the *initial* camera
    /// pose; the user can still drag to orbit from there.
    private func framUpperBody(in view: SCNView, scene: SCNScene) {
        let (bbMin, bbMax) = scene.rootNode.boundingBox
        let height = max(0.1, bbMax.y - bbMin.y)
        let width  = max(0.1, bbMax.x - bbMin.x)
        let depth  = max(0.1, bbMax.z - bbMin.z)

        // Aim ~70% up the body height (between chest and head). Hands
        // animate forward of the chest, so this puts them near the
        // viewport centre during sign animations.
        let centerX = (bbMin.x + bbMax.x) / 2
        let centerZ = (bbMin.z + bbMax.z) / 2
        let targetY = bbMin.y + height * 0.70
        let target  = SCNVector3(centerX, targetY, centerZ)

        // Distance is the dominant trade-off: too close looks zoomed,
        // too far shows the legs again. 0.9 × height with a 38° FOV
        // gives roughly head-to-waist framing on Mixamo-proportioned
        // humans (which both boy.glb and girl.glb are).
        let distance = max(height * 0.9, max(width, depth) * 1.6)

        let camera = SCNCamera()
        camera.fieldOfView = 38
        camera.zNear = 0.05
        camera.zFar = 200

        let cameraNode = SCNNode()
        cameraNode.camera = camera
        cameraNode.position = SCNVector3(centerX, targetY, centerZ + distance)
        cameraNode.look(at: target)

        scene.rootNode.addChildNode(cameraNode)
        view.pointOfView = cameraNode

        NSLog("Avatar3DView: bbox=(\(bbMin.x),\(bbMin.y),\(bbMin.z))→(\(bbMax.x),\(bbMax.y),\(bbMax.z)) " +
              "camera@(\(centerX),\(targetY),\(centerZ + distance)) target=(\(centerX),\(targetY),\(centerZ))")
    }

    // MARK: - Coordinator

    /// Owns the animation map for the loaded avatar. SwiftUI re-creates the
    /// `Avatar3DView` struct on every body re-render but the coordinator
    /// survives, so per-animation `SCNAnimationPlayer` references stay live.
    final class Coordinator {
        var currentGLB: String?
        private var scene: SCNScene?
        /// One SCNAnimationPlayer per named animation. GLTFKit2's
        /// GLTFSCNSceneSource hands these out via its `animations` array
        /// with every per-joint target already wired into the player's
        /// scnAnimation. We just hold a reference and call play()/stop().
        private var playersByName: [String: SCNAnimationPlayer] = [:]
        private var currentlyPlayingName: String?

        func attach(source: GLTFSCNSceneSource, scene: SCNScene, glbName: String) {
            self.scene = scene
            self.currentGLB = glbName

            // Pull the SceneKit-bridged animations directly off the source.
            // Each GLTFSCNAnimation has .name (String) and .animationPlayer
            // (SCNAnimationPlayer). The player's scnAnimation already
            // targets the right per-joint nodes — playing the player
            // animates the whole rig.
            playersByName.removeAll()
            for anim in source.animations {
                let player = anim.animationPlayer
                player.stop()
                // Attach to the scene root so the player has somewhere to
                // tick from. The animation's per-node targets fire
                // regardless of which node holds the player; rootNode is
                // the canonical owner.
                scene.rootNode.addAnimationPlayer(player, forKey: anim.name)
                // Last writer wins if duplicate names exist (the GLB
                // contains "A" plus a few "Armature|A|default" aliases —
                // we want the clean short name).
                playersByName[anim.name] = player
            }

            let names = Array(playersByName.keys).sorted()
            NSLog("Avatar3DView: loaded '\(glbName)' with \(names.count) animations: \(names)")
        }

        /// Resolve a logical name (e.g. "Hello") against the avatar's
        /// actual animation names. Tries exact match first, then common
        /// renaming schemes, then a case-insensitive fallback.
        private func resolveName(_ name: String) -> String? {
            if playersByName[name] != nil { return name }
            let lower = name.lowercased()
            let upper = name.uppercased()
            for variant in [lower, upper, "Armature|\(name)|default"] {
                if playersByName[variant] != nil { return variant }
            }
            return playersByName.keys.first { $0.lowercased() == lower }
        }

        func run(_ command: Avatar3DCommand) {
            switch command {
            case .play(let name):
                playOne(name: name)
            case .idle:
                playOne(name: "Idle")
            case .stop:
                stopAll()
            }
        }

        private func playOne(name: String) {
            // Stop the previous animation cleanly before starting the new
            // one, otherwise the rig interpolates between both.
            if let prev = currentlyPlayingName, let prevPlayer = playersByName[prev] {
                prevPlayer.stop()
            }
            guard let resolved = resolveName(name), let player = playersByName[resolved] else {
                NSLog("Avatar3DView: animation '\(name)' not found. Available: \(Array(playersByName.keys).sorted())")
                return
            }
            player.play()
            currentlyPlayingName = resolved
        }

        private func stopAll() {
            for (_, player) in playersByName {
                player.stop()
            }
            currentlyPlayingName = nil
        }

        /// Owner can introspect which animations the loaded avatar exposes
        /// (used by TextToSignView's word-vs-letter fallback logic).
        func availableAnimations() -> Set<String> {
            Set(playersByName.keys)
        }
    }
}
