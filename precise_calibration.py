import sys
import cv2
import numpy as np
import mediapipe as mp

from interaction.geometry import CursorMapper
from config.cursor_calibration import ACTIVE_ZONE, CURSOR_OFFSET_X, CURSOR_OFFSET_Y


def get_screen_resolution():
    try:
        import subprocess
        out = subprocess.check_output(["xrandr"]).decode()
        for line in out.splitlines():
            if " connected" in line and "x" in line:
                for token in line.split():
                    if "x" in token and token[0].isdigit():
                        w, h = token.split("+")[0].split("x")
                        return int(w), int(h)
    except Exception:
        pass
    return 1920, 1080


def main():
    screen_w, screen_h = get_screen_resolution()
    print(f"Screen resolution: {screen_w}x{screen_h}")
    
    # Open camera
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # MediaPipe setup
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        model_complexity=0,
    )
    
    # Cursor mapper
    mapper = CursorMapper(screen_w, screen_h, active_zone=ACTIVE_ZONE)
    
    print("\n=== PRECISE CALIBRATION ===\n")
    print("Instructions:")
    print("1. Point your INDEX FINGER at the CENTER of your screen")
    print("2. Hold it perfectly still")
    print("3. Watch the coordinates below")
    print("4. Measure the offset between 'Hand Position' and 'Calculated Cursor'")
    print("\nPress 'q' to quit\n")
    
    measurements = []
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Detect hand
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            hand_lms = results.multi_hand_landmarks[0]
            
            # Get index fingertip (landmark 8)
            index_tip = hand_lms.landmark[8]
            raw_x, raw_y = index_tip.x, index_tip.y
            
            # Calculate cursor position using mapper
            lms_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)
            cursor_x, cursor_y = mapper.update(lms_array)
            
            # Apply calibration offset
            final_x = cursor_x + CURSOR_OFFSET_X
            final_y = cursor_y + CURSOR_OFFSET_Y
            
            # Clamp to screen
            final_x = max(0, min(screen_w - 1, final_x))
            final_y = max(0, min(screen_h - 1, final_y))
            
            # Draw on frame
            cv2.circle(frame, (int(raw_x * w), int(raw_y * h)), 5, (0, 255, 0), -1)  # Green = raw hand
            cv2.putText(frame, f"Index: ({raw_x:.2f}, {raw_y:.2f})", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Raw Camera Pixels: ({int(raw_x*w)}, {int(raw_y*h)})", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Cursor Position: ({cursor_x}, {cursor_y})", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            cv2.putText(frame, f"With Offset: ({final_x}, {final_y})", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Calculate offset from center
            center_x, center_y = screen_w // 2, screen_h // 2
            offset_from_center = (final_x - center_x, final_y - center_y)
            cv2.putText(frame, f"Offset from center: {offset_from_center}", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            measurements.append({
                'raw': (raw_x, raw_y),
                'cursor': (cursor_x, cursor_y),
                'final': (final_x, final_y)
            })
        
        # Draw center crosshair on preview
        cv2.line(frame, (w//2 - 20, h//2), (w//2 + 20, h//2), (255, 255, 255), 1)
        cv2.line(frame, (w//2, h//2 - 20), (w//2, h//2 + 20), (255, 255, 255), 1)
        
        cv2.imshow("Precise Calibration", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            measurements = []
            print("Measurements cleared")
    
    cap.release()
    cv2.destroyAllWindows()
    
    if measurements:
        print("\n=== ANALYSIS ===\n")
        avg_cursor = np.array([m['cursor'] for m in measurements]).mean(axis=0)
        avg_final = np.array([m['final'] for m in measurements]).mean(axis=0)
        
        print(f"Average cursor position (before offset): {avg_cursor}")
        print(f"Average final position (with offset): {avg_final}")
        print(f"Current offset: X={CURSOR_OFFSET_X}, Y={CURSOR_OFFSET_Y}")
        print("\nNow check where your orb actually appears on the main screen.")
        print("If it's still offset, measure the gap and adjust CURSOR_OFFSET_X/Y")


if __name__ == "__main__":
    main()
