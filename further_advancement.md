# Telekinesis: Tony Stark Level Upgrade Plan

> Zero-latency, jitterless hand-gesture PC control → multimodal spatial computing platform.

---

## TIER 1: New Gesture Controls

Expand the current 6-class vocabulary. Each new gesture adds a dimension of control.

| Gesture | Action | How |
|---|---|---|
| **Thumb up** | Confirm / Enter / Select | Thumb extended, all others closed |
| **Thumb down** | Cancel / Back / Escape | Thumb extended downward, others closed |
| **Pistol (thumb + index)** | Right-click | Thumb and index at 90°, other fingers closed |
| **Peace sign (two fingers V)** | Multi-desktop switch | Horizontal swipe = virtual desktop switch via `xdotool` |
| **OK sign (thumb + index circle)** | Toggle zoom mode on/off | Pinch distance tracking until "OK" sign detected |
| **Shaka (thumb + pinky)** | Volume up/down | Tilt hand left/right while holding gesture |
| **Closed fist + rotate** | Window resize | Wrist rotation angle = resize proportion |
| **Open palm push forward** | "Throw" window to other monitor | Palm moves toward camera (z-axis increase) |

---

## TIER 2: Multimodal Fusion (gesture + voice + gaze)

This is the real Tony Stark differentiator. Multiple input channels working in concert.

### Gaze Tracking

- **Implementation:** MediaPipe Face Mesh (468 face landmarks) → iris landmarks 468/473 for gaze direction
- **Use case:** Look where you want to click, then pinch — eliminates cursor movement entirely
- **Why:** This is exactly how Apple Vision Pro works. Eyes aim, hands act.

### Voice Commands

- **Implementation:** Whisper.cpp (local, fast) or Vosk for offline STT
- **Use case:** Say "open terminal", "close window", "switch workspace" while gesturing
- **Why:** Hands occupied? Use voice. Noisy environment? Use gestures. Both fail? Use gaze + head pose.

### Head Pose Estimation

- **Implementation:** SolvePnP with face mesh + camera intrinsics
- **Use case:** Head tilt = scroll, head turn = switch virtual desktop, nod = confirm
- **Why:** Eyes and hands tired? Head becomes the controller.

### Blink Detection

- **Implementation:** EAR (eye aspect ratio) from face mesh landmarks
- **Use case:** Double-blink = double-click, long blink = drag mode
- **Why:** Accessibility. Hands-free clicking for users with motor limitations.

### The Killer Combo

```
Gaze aims → Gesture clicks → Voice confirms
         ↕                  ↕
    Head pose scrolls    Blink selects
```

---

## TIER 3: Visual Feedback / HUD Overlay

Currently `ui/overlay.py` is empty. Fill it with a Tony Stark-worthy HUD.

| Feature | Description |
|---|---|
| **Holographic cursor trail** | Afterimage of cursor path with fading alpha — shows movement history |
| **Gesture recognition indicator** | Detected gesture label + confidence arc rendered in corner |
| **Active gesture zone visualization** | Blue translucent overlay showing dead zones / active zones |
| **Pinch feedback** | When pinch detected, draw contracting circle at fingertip |
| **HUD status bar** | FPS, current gesture, active mode, emergency stop status |
| **Visual sound wave** | If voice commands added, show waveform when listening |
| **Transition animations** | When gesture changes, brief morphing animation between states |
| **Mode indicator** | Show current interaction mode (cursor / drag / zoom / scroll) with icon |
| **Hand skeleton overlay** | Wireframe hand with glowing joints (already partially exists in debug) |

### Implementation Options

- **OpenCV drawing** — lightweight, no extra deps, fast
- **PyGame overlay window** — transparency support, good for complex shapes
- **PyQt with frameless transparent window** — best for production-quality HUD
- **OpenGL overlay** — if going full holographic

---

## TIER 4: Advanced Interaction Modes

### Gesture Chords

Combine gestures simultaneously across both hands:

| Left Hand | Right Hand | Result |
|---|---|---|
| Fist | Point | Grab and resize window |
| Open palm | Pinch | Screenshot region |
| Two fingers | Point | Multi-cursor (drag between monitors) |
| Shaka | Point | Volume control mode |

### Gesture Sequences

Detect temporal patterns — not just single-frame gestures:

- **Pinch → pull → release** = drag and drop
- **Point → swipe → point** = move and place
- **Fist → open → fist** = toggle maximize/restore
- **Two pinches in sequence** = double-click with distance

### Adaptive Sensitivity

- Track user fatigue (slower movements over time) and auto-adjust thresholds
- Morning session: tighter thresholds. Late night: looser, more forgiving.
- Learn per-user calibration on first run.

### Gesture Profiles

Save/load gesture mappings per application:

```json
{
  "vscode": {
    "point": "cursor",
    "pinch": "click",
    "fist": "terminal_toggle",
    "two_fingers": "scroll",
    "shaka": "debugger_toggle"
  },
  "browser": {
    "point": "cursor",
    "pinch": "click",
    "fist": "bookmark",
    "two_fingers": "scroll",
    "shaka": "zoom"
  }
}
```

### Precision Mode

- Hold pinky to enter fine-control mode (smaller cursor movements, higher accuracy)
- Useful for design tools, terminal selection, pixel-level work

### Gesture Lock-On

- Once a gesture is confirmed, lock the cursor to current target
- Prevents drift during sustained interactions
- Auto-release on gesture change or timeout

### Spatial Memory

- Remember last N screen positions
- Gesture-based "snap back" to previously visited locations
- Two-finger-tap = return to last position

---

## TIER 5: System Intelligence

### Context-Aware Gestures

- Detect active window via `xdotool` / `wmctrl`
- Remap gestures automatically based on what app is focused
- VSCode: fist = toggle terminal. Browser: fist = bookmark. Video player: fist = play/pause.

### Gesture Learning from Usage

- Log gesture confidence + success rate to local database
- Auto-retrain MLP on high-confidence, user-confirmed data
- Continual learning loop: collect → label → train → improve

### Predictive Cursor

- Track common cursor trajectories (e.g., always move to "Open" button after pinch)
- Predict and pre-move cursor to likely target
- Reduce physical movement needed

### Anomaly Detection

- If hand is too fast / erratic, suppress all actions
- Reduces false positives from accidental gestures
- Learned threshold per-user over time

### Multi-Monitor Awareness

- Track which monitor cursor is on
- Adjust mapping boundaries per-monitor
- Gesture to "throw" cursor between monitors

### Session Recording

- Record gesture sequences for playback / debugging / sharing
- Export as video with gesture overlay
- Useful for demos and presentations

---

## TIER 6: Architecture Upgrades

### ONNX Runtime

- Replace PyTorch inference with ONNX Runtime
- 2-4x faster inference on CPU
- Smaller binary, faster startup
- Drop-in replacement: export `.pt` → `.onnx` → load with `onnxruntime`

### Temporal Model (TCN/LSTM)

- Replace frame-by-frame MLP with a temporal model using last 5-10 frames
- Catches dynamic gestures (wave, swipe, rotation) that current MLP misses
- Architecture options:
  - **TCN** (Temporal Convolutional Network) — fast, parallelizable
  - **LSTM** — good for sequential patterns
  - **Small Transformer** — if gesture vocabulary grows large

### Two-Hand Tracking

- MediaPipe wrapper already can detect 2 hands — use it
- Left hand = modifier key (shift, ctrl, alt)
- Right hand = action (cursor, click, drag)
- Both hands together = compound gestures (zoom, rotate, resize)

### WebSocket Remote Control

- Expose gesture state over WebSocket server
- Other apps on the network can consume gesture data
- Enables: phone as gesture display, distributed control, multi-device setups

### Plugin System

- Register custom gesture handlers as plugins
- Drop a Python file in `plugins/` → auto-discovered and loaded
- Priority-based routing (like the JARVIS project architecture)
- Community-contributed gestures and actions

### Config Hot-Reload

- YAML/JSON config for all thresholds, gestures, mappings
- Change any parameter without restarting the process
- Watch config file for changes, apply on-the-fly

---

## Recommended Build Order (highest impact first)

| Priority | Feature | Tier | Impact |
|---|---|---|---|
| 1 | **Gaze tracking + pinch-to-click** | 2 | Vision Pro territory — eyes aim, hands act |
| 2 | **Visual HUD overlay** | 3 | Makes it look and feel Tony Stark |
| 3 | **Voice commands** | 2 | Multimodal fusion — true JARVIS |
| 4 | **ONNX Runtime swap** | 6 | Free performance, no feature change |
| 5 | **Temporal model** | 6 | Enables dynamic gesture recognition |
| 6 | **New gestures** | 1 | Expand vocabulary |
| 7 | **Gesture profiles + context awareness** | 4/5 | Smart, per-app control |
| 8 | **Two-hand tracking** | 6 | Compound interactions |
| 9 | **Plugin system** | 6 | Extensibility |
| 10 | **Voice + gesture fusion** | 2 | Full multimodal |

---

## Current State Summary

| Component | Status |
|---|---|
| Hand tracking | MediaPipe, 21 landmarks, CPU |
| Gesture classification | MLP (63→128→64→32→6), 6 classes |
| Smoothing | One-Euro adaptive filter |
| State machine | 6-state hysteresis FSM |
| OS control | evdev (primary), ydotool, xdotool |
| Two-hand support | Zoom only |
| Voice | None |
| Gaze | None |
| HUD/overlay | Empty placeholder |
| Config | Hardcoded |

## Target State Summary

| Component | Target |
|---|---|
| Hand tracking | MediaPipe + Face Mesh + Iris |
| Gesture classification | TCN/LSTM temporal model, 15+ classes |
| Smoothing | One-Euro + predictive smoothing |
| State machine | Context-aware, per-app profiles |
| OS control | evdev + voice + gaze |
| Two-hand support | Full compound gestures |
| Voice | Whisper.cpp / Vosk offline STT |
| Gaze | Iris tracking → cursor aim |
| HUD/overlay | Full transparent HUD with animations |
| Config | YAML hot-reload, per-app profiles |
