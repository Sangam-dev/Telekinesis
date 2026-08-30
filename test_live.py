"""
test_live.py

End of Day 1 checkpoint: webcam -> landmarks -> features -> trained MLP
-> predicted gesture + confidence, displayed live. NO OS control here —
this validates the classifier in isolation before wiring up real control.

Run: python test_live.py
"""

import cv2
import torch
import torch.nn.functional as F

from vision.hand_tracker import HandTracker
from features.extractor import extract_feature
from ml.model import load_model, GESTURE_CLASSES


def main():
    model = load_model("models/gesture_mlp.pt")
    cap = cv2.VideoCapture(0)
    tracker = HandTracker(max_hands=2)

    print("Live inference test. Press ESC to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        results, hands_lms = tracker.process(frame)
        frame = tracker.draw(frame, results)

        if hands_lms:
            feats = extract_feature(hands_lms[0])
            x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                conf, pred_idx = torch.max(probs, dim=1)
            label = GESTURE_CLASSES[pred_idx.item()]
            conf_val = conf.item()

            color = (0, 255, 0) if conf_val >= 0.85 else (0, 165, 255)
            cv2.putText(frame, f"{label} ({conf_val:.2f})", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        else:
            cv2.putText(frame, "No hand detected", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow("Live Gesture Test (ESC to quit)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
