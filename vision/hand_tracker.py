import cv2
import mediapipe as mp
import numpy as np
import time


class HandTracker:
    def __init__(self, max_hands=2, detection_conf=0.5, tracking_conf=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process(self, frame_bgr):
        """
        Takes a BGR frame (from cv2.VideoCapture), returns:
            results: raw MediaPipe result object (has .multi_hand_landmarks)
            hands_landmarks: list of numpy arrays, each shape (21, 3) -> (x, y, z)
                              x, y are normalized [0,1] relative to frame; z is relative depth.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        hands_landmarks = []
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                pts = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark],
                    dtype=np.float32,
                )  # shape (21, 3)
                hands_landmarks.append(pts)

        return results, hands_landmarks

    def draw(self, frame_bgr, results):
        """Draws landmark overlay on the frame (for debugging / collection UI)."""
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame_bgr, hand_lms, self.mp_hands.HAND_CONNECTIONS
                )
        return frame_bgr


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()

    prev_time = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read from webcam. Check camera index / permissions.")
            break

        frame = cv2.flip(frame, 1)  # mirror for natural interaction
        results, hands_lms = tracker.process(frame)
        frame = tracker.draw(frame, results)

        # FPS overlay
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}  Hands: {len(hands_lms)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Hand Tracker Sanity Check (press ESC to quit)", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
