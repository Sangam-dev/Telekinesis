import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math
import time

pyautogui.FAILSAFE = False  # don't abort when cursor hits a corner

# -------------- Settings --------------

wCam, hCam = 640, 480      # webcam resolution
frameR = 20                # border around frame (0 = full frame)
smoothening = 7            # higher = smoother, slower

CLICK_DIST = 35            # distance threshold for click (index–middle pinch)

# Scroll tuning
SCROLL_DEADZONE = 2        # minimum vertical movement to trigger scroll
SCROLL_SCALE = 15          # higher = faster scroll

# Double-click timing (seconds between two pinches)
DOUBLE_CLICK_THRESHOLD = 0.35

# Debounce frames (gesture must be stable this many frames)
SCROLL_ACTIVATE_FRAMES = 3
DRAG_ACTIVATE_FRAMES = 3

# --------------------------------------

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mpDraw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()

prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

click_down = False
dragging = False
mode_text = ""

prev_scroll_y = None
last_click_time = 0.0      # for double-click detection

# debounce counters
scroll_frames = 0
drag_frames = 0


def fingers_up(lm_list):
    """
    Return [thumb, index, middle, ring, pinky] (1 = up, 0 = down)
    lm_list: list of (id, x, y)
    """
    fingers = []
    tip_ids = [4, 8, 12, 16, 20]

    # Thumb: compare x (sideways)
    if lm_list[tip_ids[0]][1] > lm_list[tip_ids[0] - 1][1]:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers: compare y (tip vs joint below)
    for i in range(1, 5):
        if lm_list[tip_ids[i]][2] < lm_list[tip_ids[i] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    lm_list = []
    mode_text = ""
    scroll_mode = False

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * wCam), int(lm.y * hCam)
                lm_list.append((id, cx, cy))
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    if lm_list:
        # Key landmarks
        x_index, y_index = lm_list[8][1], lm_list[8][2]      # index tip
        x_middle, y_middle = lm_list[12][1], lm_list[12][2]  # middle tip

        # Show usable area
        cv2.rectangle(
            img,
            (frameR, frameR),
            (wCam - frameR, hCam - frameR),
            (255, 0, 255),
            2,
        )

        fingers = fingers_up(lm_list)
        thumb_f, index_f, middle_f, ring_f, pinky_f = fingers
        num_up = sum(fingers)

        # ---------- Gesture patterns ----------
        # Scroll: 3 central fingers up (index, middle, ring), thumb & pinky down
        scroll_pattern = (
            thumb_f == 0 and
            index_f == 1 and
            middle_f == 1 and
            ring_f == 1 and
            pinky_f == 0
        )

        # Fist: 0 up
        hand_closed = (num_up == 0)
        hand_open = (num_up >= 3)

        # ---------- Debounce counters ----------
        # Scroll debounce
        if scroll_pattern:
            scroll_frames += 1
        else:
            scroll_frames = 0
        scroll_mode = scroll_frames >= SCROLL_ACTIVATE_FRAMES

        # Drag debounce
        if hand_closed:
            drag_frames += 1
        else:
            drag_frames = 0

        # -------------------- CURSOR ALWAYS MOVES WITH INDEX --------------------
        x3 = np.interp(x_index, (frameR, wCam - frameR), (0, screen_w))
        y3 = np.interp(y_index, (frameR, hCam - frameR), (0, screen_h))

        curr_x = prev_x + (x3 - prev_x) / smoothening
        curr_y = prev_y + (y3 - prev_y) / smoothening

        pyautogui.moveTo(curr_x, curr_y)
        prev_x, prev_y = curr_x, curr_y

        cv2.circle(img, (x_index, y_index), 10, (255, 0, 255), cv2.FILLED)
        mode_text = "MOVE"

        # -------------------- SCROLL MODE --------------------
        if scroll_mode:
            if prev_scroll_y is None:
                prev_scroll_y = y_index

            dy = y_index - prev_scroll_y  # positive if hand moves down

            if abs(dy) > SCROLL_DEADZONE:
                scroll_amount = int(-dy * SCROLL_SCALE)  # up hand = scroll up
                pyautogui.scroll(scroll_amount)
                prev_scroll_y = y_index

            mode_text = "SCROLL"
        else:
            prev_scroll_y = None

        # -------------------- CLICK + DOUBLE-CLICK (index + middle pinch) --------------------
        # Only when NOT in scroll mode and not dragging
        if (not scroll_mode and
                index_f == 1 and middle_f == 1 and not dragging):
            dist_im = math.hypot(x_middle - x_index, y_middle - y_index)

            cv2.line(img, (x_index, y_index), (x_middle, y_middle),
                     (255, 0, 255), 2)
            cv2.circle(img, (x_index, y_index), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x_middle, y_middle), 10, (255, 0, 255), cv2.FILLED)

            if dist_im < CLICK_DIST:
                cx, cy = (x_index + x_middle) // 2, (y_index + y_middle) // 2
                cv2.circle(img, (cx, cy), 12, (0, 255, 0), cv2.FILLED)

                if not click_down:
                    now = time.time()
                    if now - last_click_time <= DOUBLE_CLICK_THRESHOLD:
                        mode_text = "DOUBLE CLICK"
                    else:
                        mode_text = "CLICK"

                    pyautogui.click()  # OS detects 2 fast clicks as double-click
                    last_click_time = now
                    click_down = True
            else:
                click_down = False
        else:
            click_down = False

        # -------------------- DRAG (FIST = 0 fingers up) --------------------
        if drag_frames >= DRAG_ACTIVATE_FRAMES and not dragging:
            pyautogui.mouseDown()
            dragging = True

        if hand_open and dragging:
            pyautogui.mouseUp()
            dragging = False

        if dragging:
            mode_text = "DRAG"

    else:
        # No hand visible -> ensure drag stopped, reset state
        if dragging:
            pyautogui.mouseUp()
            dragging = False
        click_down = False
        prev_scroll_y = None
        scroll_frames = 0
        drag_frames = 0

    # -------------------- On-screen UX help --------------------
    cv2.putText(img, f"Mode: {mode_text}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    instructions = [
        "Index: Move cursor",
        "Pinch index+middle: Click / Double-click",
        "Fist (0 fingers): Drag (open hand to drop)",
        "3 fingers (index-middle-ring): Scroll up/down",
        "Press 'q' to quit"
    ]
    y0 = 70
    for i, text in enumerate(instructions):
        cv2.putText(img, text, (10, y0 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("AI Virtual Mouse", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()