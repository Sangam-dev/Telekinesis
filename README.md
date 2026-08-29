# AI Telekinesis — Neural Spatial Computer Control

Real-time hand-gesture control of a real (Linux/Wayland) computer, using
a custom-trained MLP for static gesture classification and deterministic
geometry for motion-based interactions (cursor, zoom, swipe, scroll).

See `AI_Telekinesis_2_Day_Full_Plan.md` for the full build plan, architecture
rationale, and hour-by-hour schedule.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Wayland input control daemon — must be running before main.py works
sudo apt install ydotool
sudo ydotoold &

# Sanity check ydotool BEFORE doing anything else:
ydotool mousemove 500 500
ydotool click 0xC0
```

## Build order

```bash
# 1. Sanity-check webcam + hand tracking
python vision/hand_tracker.py

# 2. Collect training data (~350 samples x 6 classes)
python collect_data.py

# 3. Train the MLP (+ RandomForest/SVM baselines for comparison)
python -m ml.train

# 4. Validate the classifier live, no OS control yet
python test_live.py

# 5. Run the full pipeline with real OS control
python main.py
```

## Gesture map

| Gesture | Action |
|---|---|
| POINT | Move cursor |
| PINCH | Click / drag (when held + moved) |
| TWO_FINGER | Scroll |
| Two hands apart/together | Zoom in/out |
| Fast lateral swipe | Next/previous slide or page |

## Emergency stop

Press **ESC** in the video window at any time — this immediately disables
all OS control calls. Restart `main.py` to resume.

## Project structure

```
ai-telekinesis/
├── data/raw/              # collected CSV dataset
├── models/                # saved gesture_mlp.pt, confusion matrix, training curve
├── vision/hand_tracker.py # OpenCV + MediaPipe wrapper
├── features/extractor.py  # landmarks -> normalized 63-feature vector
├── ml/                    # model, dataset, training (MLP + RF/SVM baselines)
├── control/os_control.py  # ydotool wrapper for real OS input
├── interaction/           # state machine, geometry, interaction engine
├── collect_data.py
├── test_live.py
└── main.py
```
