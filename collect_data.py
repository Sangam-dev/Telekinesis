"""
Data collection script for hand gesture recognition.

controls:

1= Neutral
2= Point
3= Pinch
4= open  palm
5= fist
6= tw0 fingers

space = next gesture
r= toggle continuous recording
esc = quit
"""

import cv2
import csv

import os
import time

from vision.hand_tracker import HandTracker
from features.extractor import extract_feature

LABELS = {
    ord("1"): "neutral",
    ord("2"): "point",
    ord("3"): "pinch",
    ord("4"): "open_palm",
    ord("5"): "fist",
    ord("6"): "two_fingers",
}

OUTPUT_CSV = "data/raw/gesture.csv"
CONTINUOUS_INTERVAL = 0.1

def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    file_exists = os.path.isfile(OUTPUT_CSV)

    csv_file = open(OUTPUT_CSV, "a", newline="")
    writer = csv.writer(csv_file)

    if not file_exists:
        writer.writerow(["label"] + [f"{axis}{i}" for i in range(21) for axis in ["x", "y", "z"]])

    cap = cv2.VideoCapture(0)
    tracker = HandTracker(max_hands=1)

    current_label = "neutral"
    continuous_recording = False
    last_continuous_write = 0.0
    counts = {label: 0 for label in LABELS.values()}

    print("Starting data collection. Press keys 1-6 to label gestures, space to move to next gesture, 'r' to toggle continuous recording, and 'esc' to quit.")
    print("Controls: 1=Neutral, 2=Point, 3=Pinch, 4=Open Palm, 5=Fist, 6=Two Fingers")

    while True:
        ok, frame = cap.read()

        if not ok:
            print("Failed to read from webcam. Check camera index / permissions.")
            break

        frame = cv2.flip(frame, 1)  # mirror for natural interaction
        results, hands_landmarks = tracker.process(frame)
        frame = tracker.draw(frame, results)

        key = cv2.waitKey(1) & 0xFF

        if key in LABELS:
            current_label = LABELS[key]
            print(f"Current label set to: {current_label}")
        elif key == ord("r"):
            continuous_recording = not continuous_recording
            print(f"Continuous recording {'enabled' if continuous_recording else 'disabled'}.")
        elif key == 27:  # ESC
            print("Exiting data collection.")
            break

        have_hand = len(hands_landmarks) > 0
        should_record_this_frame = False

        if key == 32 and have_hand:
            should_record_this_frame = True
        elif continuous_recording and have_hand:
            now = time.time()
            if now - last_continuous_write >= CONTINUOUS_INTERVAL:
                should_record_this_frame = True
                last_continuous_write = now

        if should_record_this_frame:
            feature_vector = extract_feature(hands_landmarks[0])
            writer.writerow([current_label] + feature_vector.tolist())
            counts[current_label] += 1
            print(f"Recorded {counts[current_label]} samples for label '{current_label}'.")

        status = f"Label: {current_label}  | REC: {'ON' if continuous_recording else 'off'}  | Hand: {'YES' if have_hand else 'no'}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        counts_str = " | ".join(f"{k}:{v}" for k, v in counts.items())
        cv2.putText(frame, counts_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, "1-6=label SPACE=record R=continuous ESC=quit",
                    (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Data Collection", frame)

    cap.release()

    cv2.destroyAllWindows()
    csv_file.close()

    print("\n final counts:")
    for label, count in counts.items():
        print(f"{label}: {count}")

    print(f"Data saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

        