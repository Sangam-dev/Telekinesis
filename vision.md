# Vision: The Computer-Vision Side of Telekinesis

This document covers **everything** on the vision / image-processing side of the
project: OpenCV camera capture & image handling, the MediaPipe Hands module,
the 21 hand landmarks and how they're read, the feature extraction (the metrics
we pull out of landmarks), and the data-collection workflow that produced
`data/raw/gesture.csv`.

> The vision subsystem is the *front door* of the whole pipeline: it converts
> raw camera photons into structured 63-number feature vectors that the ML
> classifier then labels as gestures, which the control engine then acts on.
> This doc stops at the classifier boundary — the ML is in `course.md`, the
> control logic in `trace.md`.

---

## Table of Contents

1. [The Vision Pipeline, Top to Bottom](#1-the-vision-pipeline)
2. [OpenCV — Camera Capture & Image Handling](#2-opencv)
3. [Color Spaces & Conversion (BGR → RGB)](#3-color-spaces)
4. [The MediaPipe Hands Module](#4-mediapipe-hands)
5. [The 21 Hand Landmarks](#5-the-21-hand-landmarks)
6. [The HandTracker Wrapper](#6-the-handtracker-wrapper)
7. [Landmark → Numpy Representation](#7-landmarks-as-numpy)
8. [Feature Extraction — the Metrics We Keep](#8-feature-extraction)
9. [Why Normalization Matters (Visual Intuition)](#9-why-normalization-matters)
10. [Data Collection — How We Built the Dataset](#10-data-collection)
11. [The Collected Dataset](#11-the-dataset)
12. [Key Rate / Latency Concepts](#12-key-rate--latency)
13. [Glossary of Vision Terms](#13-glossary)

---

## 1. The Vision Pipeline

```
cv2.VideoCapture(0)   (V4L2, MJPG, 640x480, 30fps, buffersize 1)
   │  cap.read() → BGR frame  (numpy HxWx3 uint8)
   ▼
cv2.flip(frame, 1)     ← mirror horizontally (natural interaction)
   ▼
cv2.cvtColor(BGR → RGB)  ← MediaPipe requires RGB
   ▼
mediapipe Hands.process()
   │  → 21 landmarks per hand: (x, y, z), x/y normalized [0..1], z relative depth
   ▼
HandTracker.process() → np.float32 (21, 3) per hand
   ▼
features/extractor.extract_feature() → normalized 63-vector
   │  (translation-cancel + scale-normalize)
   ▼
   → ML classifier (model.py) → gesture label
```

The **vision subsystem ends right after feature extraction**. Everything from
there is ML (`course.md`) and control (`trace.md`).

---

## 2. OpenCV

OpenCV (`cv2`) provides two very different jobs here: **capture** and
**rendering**.

### Capturing frames (`cv2.VideoCapture`)

In `main.py`, the camera is opened with deliberately tuned, latency-focused
settings:

```python
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)      # explicit V4L2 backend
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # kernel queues at most 1 frame
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
```

Each setting has a *why*:

| Setting | Why |
|---------|-----|
| `cap=0`, `CAP_V4L2` | camera index 0; explicitly ask for the Linux **Video4Linux2** backend (not V4L / GStreamer defaults) → more predictable latency. |
| `BUFFERSIZE=1` | the kernel's V4L2 ring buffer holds at most **1 queued frame**. This prevents the camera from buffering old frames ahead of us — we always read the freshest. Less buffering → less latency. |
| `640×480` | small enough that MediaPipe (lightweight model) runs at 30+ fps on CPU; large enough to track the hand reliably. |
| `FPS=30` | standard webcam rate; the frame-rate of the whole loop. |
| `FOURCC=MJPG` | ask the camera to **compress to MJPEG on-chip**. A compressed frame is far smaller than raw YUV, so it transfers over USB faster → lower latency into `cap.read()`. |

### Two capture modes in the project

- **`main.py`** opens the camera with all the low-latency tunables, and reads it
  on a **dedicated `_CameraReader` thread** so inference never blocks on USB
  I/O and we always get the most recent frame.
- **`collect_data.py` / `test_live.py` / `hand_tracker.py --main`** use the
  simple `cap.read()` in the main loop — fine for offline collection / sanity
  checks where per-frame latency matters less.

### Rendering frames

```python
cv2.imshow("window name", frame)   # display the frame
cv2.waitKey(1) & 0xFF               # pump the GUI + catch key presses
cv2.putText(frame, text, (x,y), FONT, scale, color, thickness)   # overlay text
cv2.destroyAllWindows()             # close windows on exit (paired with cap.release())
```

- `imshow` + `waitKey` are the OpenCV GUI loop. `waitKey(1)` returns the pressed
  key (bitwise-AND with `0xFF` to get a single ASCII/scan byte).
- `ESC` (27) is the emergency stop / quit key; `q` also quits; digit keys select
  labels in the collector.
- `putText` overlays live status (label, confidence, FPS, counts) — this is how
  the debug windows show you what the system "sees."

### Mirroring with `cv2.flip`

```python
frame = cv2.flip(frame, 1)   # flip horizontally (left↔right)
```

Webcams face you, so the image is a "mirror." Flipping horizontally makes
raising your **right** hand move the cursor as your **right** hand — natural
interaction. This flip is applied **before** MediaPipe, so all downstream
landmark coordinates are in the mirrored (user-friendly) space.

---

## 3. Color Spaces

OpenCV reads images as **BGR** (blue, green, red channel order) — this is a
historical OpenCV convention. MediaPipe, like most CV libraries, expects **RGB**.

```python
frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
```

This channel re-ordering is just data reshuffling (same byte content), not a
real color transform — but it is **mandatory**; feeding BGR into MediaPipe would
make the hand model see wrong colors and track incorrectly.

(Note: `cv2.flip` operates on BGR; the BGR→RGB conversion happens inside
`HandTracker.process`, after the frame has been flipped.)

---

## 4. The MediaPipe Hands Module

MediaPipe (`mediapipe==0.10.14`, pinned in `pyproject.toml`) provides a
pre-trained **hand landmark model**. It does NOT come from this project — it's a
Google-published deep model. We consume it via:

```python
import mediapipe as mp
self.mp_hands = mp.solutions.hands
self.hands = self.mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=max_hands,
    min_detection_confidence=detection_conf,    # 0.6
    min_tracking_confidence=tracking_conf,      # 0.6
    model_complexity=0,
)
```

### Parameter meanings

| Param | Value | Meaning |
|-------|-------|---------|
| `static_image_mode` | `False` | **Video mode.** MediaPipe tracks the hand between frames (cheaper than re-detecting every frame). If `True`, it re-detects each frame like independent images (slow, only for stills). |
| `max_num_hands` | configurable | max hands to detect. Runs use 2 (main) or 1 (collector). |
| `min_detection_confidence` | `0.6` | minimum confidence to accept an initial hand *detection*. |
| `min_tracking_confidence` | `0.6` | minimum confidence to keep *tracking* (not re-detect) the hand across frames. |
| `model_complexity` | `0` | `0` = lightweight/fast hand model; `1` = more accurate but ~2× slower on CPU. **For real-time 30fps control, `0` is the right choice** — the speed/accuracy tradeoff favors speed here. |

### Detection vs Tracking (two modes baked in)

- **Detection**: finding a hand in the full frame (expensive).
- **Tracking**: following an already-found hand to the next frame (cheap).
- With `static_image_mode=False`, MediaPipe interleaves these: track when
  possible, re-detect only when tracking confidence drops below
  `min_tracking_confidence`. This is why `hand_tracker.py` achieves high FPS.

### Drawing

```python
self.mp_draw = mp.solutions.drawing_utils
...
self.mp_draw.draw_landmarks(frame_bgr, hand_lms, self.mp_hands.HAND_CONNECTIONS)
```

`draw_landmarks` draws the landmark dots and their connecting lines
(`HAND_CONNECTIONS` = the known 21-point skeleton topology) for the debug
overlay. It draws directly onto the frame in-place.

---

## 5. The 21 Hand Landmarks

MediaPipe returns **21 landmarks** per hand, each with `(x, y, z)`. The indices
are a fixed, documented convention:

```
 8 ── 9 ──10 ──11 ──12        index(8)  middle(9,10) ring(11,12)
 │    │                       pinky(17,18)
 7    6    5    13 ──16        thumb has 4 (1,2,3,4)
 │    │    │    │
 4 ── 3 ── 2    14 ──15        wrist = 0
      │    │    │
      ...palm(13..17)...
 0 (wrist)
```

The complete mapping (abridged to the indices this project uses):

| Index | Point | Used for |
|-------|-------|----------|
| **0** | **Wrist** | translation normalization anchor (`extractor.py`); window-drag reference & scroll position (`geometry.py`). The *most stable* landmark. |
| **4** | **Thumb tip** | pinch midpoint (`engine.py` thumb side). |
| **8** | **Index fingertip** | cursor position (POINT); pinch midpoint (index side); swipe velocity. |
| **9** | **Middle-finger MCP** (base) | scale-reference distance anchor (`extractor.py`); palm center. |

Each landmark is `(x, y, z)`:
- `x` ∈ [0,1] — normalized horizontal position within the frame.
- `y` ∈ [0,1] — normalized vertical position within the frame (0 = top).
- `z` — **relative depth** between landmarks, not absolute distance. It's a
  scale-normalized pseudo-depth: larger magnitude = further from camera.

The two landmark indices that anchor *feature extraction* are:
```python
WRIST_IDX = 0
MIDDLE_MCP_IDX = 9
```

---

## 6. The HandTracker Wrapper

`vision/hand_tracker.py` wraps MediaPipe so the rest of the project never
touches MediaPipe's object API directly. It exposes three methods:

```python
class HandTracker:
    def __init__(self, max_hands=2, detection_conf=0.6, tracking_conf=0.6):
        # configures mp.solutions.hands + drawing utils

    def process(self, frame_bgr):
        """BGR frame → (results, list-of-(21,3) numpy arrays)."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        hands_landmarks = []
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark],
                               dtype=np.float32)          # (21, 3)
                hands_landmarks.append(pts)
        return results, hands_landmarks

    def draw(self, frame_bgr, results):
        # overlay landmarks + connections onto the frame, in-place
        return frame_bgr
```

It returns:
- **`results`** — the raw MediaPipe result object (needed for `draw`, which uses
  `.multi_hand_landmarks`).
- **`hands_landmarks`** — a Python list of clean `np.float32 (21, 3)` arrays,
  one per detected hand. This is the *clean, ML-friendly* interface the rest of
  the code consumes.

Why this wrapper matters: it isolates the MediaPipe API behind a minimal
interface, converts to a standard numpy shape once, and provides drawing — so
`main.py`, `collect_data.py`, and `test_live.py` all share identical vision
logic.

---

## 7. Landmarks as Numpy

```python
pts = np.array(
    [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark],
    dtype=np.float32,
)   # shape (21, 3)
```

This is the pivotal transformation: **21 MediaPipe objects → one `(21, 3)`
float32 matrix**.

- `dtype=np.float32` matches PyTorch's default weight dtype.
- Row index = landmark number (0..20).
- Column 0 = x, column 1 = y (both normalized `[0,1]`), column 2 = z (relative
  depth).

From here on, ALL downstream code operates on numpy: slicing rows (e.g.,
`lm[8]` = index fingertip, `lm[0]` = wrist), vector math (subtraction, norms),
and flattening for the ML feature vector.

---

## 8. Feature Extraction — the Metrics We Keep

`features/extractor.py` is where raw landmarks become the **63-dimensional
feature vector** the classifier trains on.

```python
import numpy as np

WRIST_IDX = 0
MIDDLE_MCP_IDX = 9

def extract_feature(landmarks):
    assert landmarks.shape == (21, 3)

    wrist = landmarks[WRIST_IDX]
    middle_mcp = landmarks[MIDDLE_MCP_IDX]

    centered = landmarks - wrist            # translate: wrist → origin
    scale = np.linalg.norm(middle_mcp - wrist)
    scale_ref = max(scale, 1e-6)            # guard against divide-by-zero
    normalized = centered / scale_ref       # scale: hand → unit size

    return normalized.flatten().astype(np.float32)   # (63,)
```

### Two metrics/operations define the whole representation

**1. Translation cancellation (`centered = landmarks - wrist`)**
Every landmark is re-expressed *relative to the wrist*. Now the wrist is the
origin, and all other points are offsets from it. This removes the hand's
**absolute position** from the features — the same gesture looks the same no
matter where in the frame the hand is.

**2. Scale normalization (`normalized = centered / scale_ref`)**
`scale_ref` is the Euclidean distance from wrist to the middle-finger MCP
(`np.linalg.norm(middle_mcp - wrist)`), floored at `1e-6` to avoid dividing by
zero. Dividing every offset by this distance makes the whole hand a "unit
length." This removes the hand's **size** (distance from camera) from the
features — a big hand and a small hand make the same shape.

### Why wrist→middle-MCP as the scale reference?
- The **wrist** is the most jitter-resistant landmark.
- The **middle-finger MCP** is near the hand's geometric center and moves
  consistently regardless of which fingers are extended. So wrist→MCP distance
  is a reliable, gesture-independent proxy for hand size.

### The output: a 63-vector
`(21, 3)` → `(21×3,)` = **63 features**, ordered `x0,y0,z0, x1,y1,z1, ...` —
exactly the CSV column layout (`x0,y0,z0` ... `x20,y20,z20`).

### What we deliberately throw away
- Absolute position (canceled).
- Absolute size (normalized).
- Raw depth sign/magnitude (only relative offsets remain).
The 63 numbers encode **only the hand's shape** — which is precisely what
distinguishes `point` from `fist` from `pinch`, independent of where/how big
the hand is.

---

## 9. Why Normalization Matters

Without normalization, the same "point" gesture would produce wildly different
feature vectors depending on:
- position in frame (left vs right, top vs bottom), and
- distance from camera (close = large, far = small).

The classifier would then learn to recognize *locations/sizes* instead of
*shapes* — and would fail when you move your hand around the screen while
pointing (which you constantly do to move the cursor!).

Normalization makes the feature space clean: **shape is the only signal**.
This is what lets a tiny MLP classify accurately on a small dataset.

---

## 10. Data Collection — How We Built the Dataset

`collect_data.py` turns live camera frames into labeled CSV rows.

### The interaction controls (UI)

```python
LABELS = {
    ord("1"): "neutral",    ord("2"): "point",
    ord("3"): "pinch",      ord("4"): "open_palm",
    ord("5"): "fist",       ord("6"): "two_fingers",
}
```

| Key | Action |
|-----|--------|
| `1`–`6` | set the current gesture **label** (the class being recorded) |
| `SPACE` | record **one** sample now (only if a hand is detected) |
| `r` | toggle **continuous recording** (auto-record every ~0.1s) |
| `ESC` | quit |

### The loop, step by step

```python
csv_file = open(OUTPUT_CSV, "a", newline="")     # append mode
writer = csv.writer(csv_file)
if not file_exists:
    writer.writerow(["label"] + [f"{axis}{i}" for i in range(21) for axis in ["x","y","z"]])

cap = cv2.VideoCapture(0)                         # collect via webcam too
tracker = HandTracker(max_hands=1)                # one hand per sample
```

1. **Read a frame**, mirror it (`cv2.flip`), run `HandTracker.process`.
2. **Read the key.** Digit keys change `current_label`; `r` toggles continuous;
   `ESC` exits.
3. **Decide whether to record this frame:**
   ```python
   have_hand = len(hands_landmarks) > 0
   should_record = False
   if key == 32 and have_hand:        # spacebar → one-shot
       should_record = True
   elif continuous_recording and have_hand:
       if now - last_continuous_write >= CONTINUOUS_INTERVAL:   # 0.1s
           should_record = True
           last_continuous_write = now
   ```
4. **If recording** (and a hand exists):
   ```python
   feature_vector = extract_feature(hands_landmarks[0])
   writer.writerow([current_label] + feature_vector.tolist())
   counts[current_label] += 1
   ```
   One row = the current label + the 63 feature values.
5. **Draw the UI** (label, REC status, per-class sample counts) and display.

### Key data-collection details

- **Append mode (`"a"`)** — collecting again *adds* to an existing CSV rather
  than overwriting; the header is written only if the file is brand new.
- **Header format** matches feature order exactly:
  `label,x0,y0,z0,x1,y1,z1,...,x20,y20,z20`.
- **`max_hands=1`** — exactly one hand per row keeps each sample's meaning
  unambiguous (no "which hand" ambiguity).
- **Continuous interval 0.1s** — when `r` is on, at most 10 samples/sec, so a
  held pose yields many slightly-varying samples (jitter = useful augmentation).
- **Only records when a hand is present** — you can't record empty frames.
- **Live sample counts** are shown so the collector knows when ~350 samples per
  class have been gathered (the README target).
- **Intended workflow:** set the label with a digit key, hold the pose, hammer
  `SPACE` (or toggle `r`) to capture many examples, `space` to switch gesture.
  Vary hand position/orientation so the normalizer's invariance is exercised.

---

## 11. The Collected Dataset

The result, `data/raw/gesture.csv`, currently contains ~5,137 data rows:

```
  934  fist
  578  neutral
  696  open_palm
  829  pinch
  848  point
 1252  two_fingers
```

**Imbalance note:** `two_fingers` (1,252) has ~2× the samples of `neutral`
(578). This is exactly why the ML side uses *stratified splits* (see
`course.md`) — to keep this imbalance consistent across train/val/test and why
accuracy alone is insufficient (precision/recall/F1 and the confusion matrix
matter). Collecting more `neutral` samples would be the natural next step for a
more balanced model.

Each row: `label` + 63 normalized floats. This CSV is the single source of
truth for `ml/dataset.py::load_splits`, which reads it, maps labels to integer
class indices, and builds the train/val/test tensors.

---

## 12. Key Rate / Latency Concepts

- **30 fps target**: the camera runs at 30 fps; MediaPipe `model_complexity=0`
  is chosen so each frame's inference fits comfortably in that budget on CPU.
- **One frame in → one feature out**: every `cap.read()` feeds straight through
  detection → landmarks → features. There's no batching; latency per frame ≈
  capture + MediaPipe + feature math.
- **Video vs image mode**: `static_image_mode=False` enables *tracking* between
  frames, which is far cheaper than re-detection — this is what makes 30 fps
  feasible.
- **Decoupled capture (main.py)**: the `_CameraReader` thread continuously
  reads the freshest frame, so even if inference is slower than 30fps, the
  system always processes the *latest* hand position (lowest visual lag).

---

## 13. Glossary of Vision Terms

| Term | Meaning |
|------|---------|
| **Frame** | a single image from the camera; a numpy `H×W×3` uint8 array. |
| **BGR / RGB** | OpenCV reads BGR; MediaPipe needs RGB; `cvtColor` converts. |
| **`cv2.flip(frame, 1)`** | horizontal mirror for natural (right-hand) interaction. |
| **V4L2** | Linux Video4Linux2 camera backend. |
| **MJPEG / FOURCC** | camera-side JPEG compression → smaller, faster USB transfer. |
| **Buffersize** | kernel frame queue length; `1` = read the freshest frame. |
| **MediaPipe Hands** | Google's pre-trained hand-landmark deep model (not trained here). |
| **Detection** | finding a hand in the whole frame (expensive). |
| **Tracking** | following an already-found hand frame-to-frame (cheap). |
| **Landmark** | a hand keypoint with `(x, y, z)`; 21 per hand. |
| **`x`/`y` normalized** | `[0,1]` position within the frame. |
| **`z` relative depth** | scale-normalized pseudo-depth between landmarks. |
| **Translation invariance** | canceling absolute position by subtracting the wrist. |
| **Scale invariance** | canceling hand size by dividing by wrist→middle-MCP distance. |
| **Feature vector** | the 63 normalized values fed to the classifier. |
| **`extract_feature`** | function that maps a `(21,3)` landmark array → `(63,)`. |
| **CSV row** | `label` + 63 features; one collected hand sample. |
| **Continuous recording** | auto-capturing ~10 samples/sec while a pose is held. |
| **Stratified split** | preserving class proportions in train/val/test (see ML course). |
| **`cv2.imshow` / `waitKey`** | OpenCV GUI loop for display + key handling. |
| **`cv2.putText`** | draw on-screen text overlays for debug/status. |
| **`cv2.VideoCapture.release`** | close the camera (paired with `destroyAllWindows`). |

---

## Summary

The vision subsystem converts raw camera data into clean, invariant features:

1. **OpenCV** captures frames (latency-tuned: V4L2, MJPG, buffersize 1) and
   renders the debug GUI, with `cv2.flip` mirroring for natural interaction.
2. **MediaPipe Hands** (a pre-trained deep model) finds hands and returns **21
   landmarks** `(x, y, z)` each, in **video/tracking mode** with the lightweight
   `model_complexity=0` for speed.
3. **`HandTracker`** wraps MediaPipe into a clean `(21, 3)` numpy interface.
4. **`extract_feature`** cancels translation and scale to yield a **63-vector**
   that encodes *only hand shape*.
5. **`collect_data.py`** records thousands of these vectors (with labels) into
   `data/raw/gesture.csv`, which seeds the ML classifier.

The vision work is deliberately "thin" because MediaPipe does the heavy lifting;
the engineering value is in the *latency tuning, the wrapper abstraction, and
the position/scale-invariant feature representation*.
