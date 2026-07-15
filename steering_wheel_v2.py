"""
Virtual Steering Wheel v3 — Full Gaming HUD Edition
------------------------------------------------------
Based on the idea from jayesh-cmd/virtual-steering-wheel, extended with:

1. RIGHT hand only  -> steering (tilt right hand left/right, like a wheel).
   Hand held flat/level = CENTER (forward), tilt right = RIGHT,
   tilt left = LEFT. Works correctly whether your palm or the back of
   your hand faces the camera.
2. LEFT  hand fist (mutthi) -> accelerator (UP arrow held)
3. LEFT  hand 1 finger (index only) -> brake/reverse (DOWN arrow held)
4. Steering wheel graphic anchored to your actual right hand (wrist),
   sized to your hand, rotates with your tilt.
5. Full-window cyberpunk racing dashboard: corner HUD brackets, scanlines,
   vignette, animated glowing title, speedometer dial, gear indicator,
   side status panels.
6. Live sensitivity tuning: press +/- to widen/narrow the steering dead
   zone without editing the code.
7. Snapshot: press C to save the current HUD frame as a PNG.
8. Session timer shown in the status panel.

Install:
    pip install mediapipe opencv-python pynput numpy

Run:
    python steering_wheel_v2.py
Press Q to quit, S to toggle hand swap.
"""

import math
import time
import cv2
import numpy as np
import mediapipe as mp
from pynput.keyboard import Controller, Key

# ----------------------------- CONFIG ---------------------------------- #
CAMERA_INDEX = 0
FLIP_CAMERA = True
DEAD_ZONE_DEG = 10
GRACE_FRAMES = 8
MIN_DET_CONF = 0.7
MIN_TRACK_CONF = 0.5

STEERING_HAND = "Right"
GAS_BRAKE_HAND = "Left"
SWAP_HANDS = False

WHEEL_MIN_RADIUS = 40
WHEEL_MAX_RADIUS = 110
WHEEL_SCALE = 1.8            # hand-span multiplier (lower = smaller wheel)

SENSITIVITY_STEP = 2         # degrees changed per +/- key press
STEER_SMOOTHING = 0.4        # 0 = no smoothing, closer to 1 = smoother/laggier

# Neon theme colours (BGR)
COL_CYAN = (255, 220, 40)
COL_MAGENTA = (200, 40, 255)
COL_GREEN = (90, 255, 120)
COL_RED = (60, 60, 255)
COL_AMBER = (0, 190, 255)
COL_YELLOW = (0, 230, 255)
COL_TEXT = (235, 235, 235)
COL_DIM = (110, 110, 110)

MAX_SPEED = 220          # purely cosmetic speedometer top value
SPEED_ACCEL_RATE = 90    # units/sec while accelerating
SPEED_BRAKE_RATE = 160   # units/sec while braking
SPEED_DECAY_RATE = 55    # units/sec natural decay when neutral

# ----------------------------- SETUP ------------------------------------ #
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=MIN_DET_CONF,
    min_tracking_confidence=MIN_TRACK_CONF,
)

keyboard = Controller()

key_state = {Key.left: False, Key.right: False, Key.up: False, Key.down: False}


def set_key(key, should_be_down):
    if should_be_down and not key_state[key]:
        keyboard.press(key)
        key_state[key] = True
    elif not should_be_down and key_state[key]:
        keyboard.release(key)
        key_state[key] = False


def release_all():
    for k in list(key_state.keys()):
        set_key(k, False)


# ----------------------------- LANDMARK HELPERS -------------------------- #
LM = mp_hands.HandLandmark


def on_screen_label(handedness_label: str) -> str:
    if SWAP_HANDS:
        return "Left" if handedness_label == "Right" else "Right"
    return handedness_label


def steering_tilt_angle(landmarks) -> float:
    """Angle of the line across the knuckles (index MCP -> pinky MCP), folded
    so that a level/flat hand always reads as ~0 degrees.

    atan2(dy, dx) on this line gives an angle near 0 deg when the hand is
    flat with the palm facing the camera, but near +-180 deg when the hand
    is flat with the *back* of the hand facing the camera (which is the
    natural way people hold a "steering wheel" grip). That +-180 baseline
    was being read as a real tilt, so the wheel looked like it defaulted to
    RIGHT even when the hand was straight. Since the knuckle line has no
    inherent direction, we fold anything past +-90 deg back by 180 deg so
    both hand orientations agree on what "flat" (0 deg, forward) means.
    """
    idx_mcp = landmarks[LM.INDEX_FINGER_MCP]
    pinky_mcp = landmarks[LM.PINKY_MCP]
    dx = pinky_mcp.x - idx_mcp.x
    dy = pinky_mcp.y - idx_mcp.y
    angle = math.degrees(math.atan2(dy, dx))

    if angle > 90:
        angle -= 180
    elif angle <= -90:
        angle += 180

    return angle


def finger_is_extended(landmarks, tip_idx, pip_idx) -> bool:
    return landmarks[tip_idx].y < landmarks[pip_idx].y


def is_fist(landmarks) -> bool:
    fingers = [
        (LM.INDEX_FINGER_TIP, LM.INDEX_FINGER_PIP),
        (LM.MIDDLE_FINGER_TIP, LM.MIDDLE_FINGER_PIP),
        (LM.RING_FINGER_TIP, LM.RING_FINGER_PIP),
        (LM.PINKY_TIP, LM.PINKY_PIP),
    ]
    return all(not finger_is_extended(landmarks, tip, pip) for tip, pip in fingers)


def is_one_finger(landmarks) -> bool:
    index_up = finger_is_extended(landmarks, LM.INDEX_FINGER_TIP, LM.INDEX_FINGER_PIP)
    middle_down = not finger_is_extended(landmarks, LM.MIDDLE_FINGER_TIP, LM.MIDDLE_FINGER_PIP)
    ring_down = not finger_is_extended(landmarks, LM.RING_FINGER_TIP, LM.RING_FINGER_PIP)
    pinky_down = not finger_is_extended(landmarks, LM.PINKY_TIP, LM.PINKY_PIP)
    return index_up and middle_down and ring_down and pinky_down


def hand_span_px(landmarks, w, h) -> float:
    idx_mcp = landmarks[LM.INDEX_FINGER_MCP]
    pinky_mcp = landmarks[LM.PINKY_MCP]
    dx = (pinky_mcp.x - idx_mcp.x) * w
    dy = (pinky_mcp.y - idx_mcp.y) * h
    return math.hypot(dx, dy)


# ----------------------------- DRAWING: WHEEL ON HAND --------------------- #
def draw_wheel_on_hand(frame, landmarks, angle_deg, w, h, glow_color):
    wrist = landmarks[LM.WRIST]
    cx, cy = int(wrist.x * w), int(wrist.y * h)
    span = hand_span_px(landmarks, w, h)
    radius = int(np.clip(span * WHEEL_SCALE, WHEEL_MIN_RADIUS, WHEEL_MAX_RADIUS))

    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), radius + 10, glow_color, thickness=2)
    cv2.circle(overlay, (cx, cy), radius, (50, 50, 50), thickness=10)
    cv2.circle(overlay, (cx, cy), radius, glow_color, thickness=3)

    display_angle = max(-60, min(60, angle_deg))
    rad = math.radians(display_angle)

    for spoke_offset in (0, 120, 240):
        a = rad + math.radians(spoke_offset - 90)
        x2 = int(cx + radius * math.cos(a))
        y2 = int(cy + radius * math.sin(a))
        cv2.line(overlay, (cx, cy), (x2, y2), glow_color, thickness=7)
        cv2.line(overlay, (cx, cy), (x2, y2), (255, 255, 255), thickness=2)

    for deg in range(0, 360, 30):
        a = math.radians(deg) + rad
        gx = int(cx + radius * math.cos(a))
        gy = int(cy + radius * math.sin(a))
        cv2.circle(overlay, (gx, gy), 4, (255, 255, 255), thickness=-1)

    cv2.circle(overlay, (cx, cy), 16, glow_color, thickness=-1)
    cv2.circle(overlay, (cx, cy), 16, (255, 255, 255), thickness=2)

    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)


def draw_fist_burst(frame, landmarks, w, h, color):
    wrist = landmarks[LM.WRIST]
    cx, cy = int(wrist.x * w), int(wrist.y * h)
    overlay = frame.copy()
    for deg in range(0, 360, 45):
        a = math.radians(deg)
        x1 = int(cx + 55 * math.cos(a))
        y1 = int(cy + 55 * math.sin(a))
        x2 = int(cx + 80 * math.cos(a))
        y2 = int(cy + 80 * math.sin(a))
        cv2.line(overlay, (x1, y1), (x2, y2), color, thickness=4)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)


# ----------------------------- FULL-WINDOW GAMING HUD ---------------------- #
def apply_vignette(frame, strength=0.55):
    h, w = frame.shape[:2]
    kx = cv2.getGaussianKernel(w, w * 0.6)
    ky = cv2.getGaussianKernel(h, h * 0.6)
    mask = ky @ kx.T
    mask = mask / mask.max()
    mask = (mask * (1 - strength) + strength)
    out = frame.astype(np.float32)
    for c in range(3):
        out[:, :, c] *= mask
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_scanlines(frame, alpha=0.08, gap=4):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    for y in range(0, h, gap):
        cv2.line(overlay, (0, y), (w, y), (0, 0, 0), 1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_corner_brackets(frame, color, length=45, thick=4, margin=14):
    h, w = frame.shape[:2]
    pts = [(margin, margin), (w - margin, margin), (margin, h - margin), (w - margin, h - margin)]
    dirs = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
    for (x, y), (dx, dy) in zip(pts, dirs):
        cv2.line(frame, (x, y), (x + dx * length, y), color, thick)
        cv2.line(frame, (x, y), (x, y + dy * length), color, thick)


def glow_text(frame, text, org, font_scale, color, thickness=2, glow=6):
    overlay = frame.copy()
    cv2.putText(overlay, text, org, cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness + glow, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness, cv2.LINE_AA)


def rounded_panel(frame, x, y, w_, h_, color=(20, 12, 8), alpha=0.55, border=None):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w_, y + h_), color, thickness=-1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    if border:
        cv2.rectangle(frame, (x, y), (x + w_, y + h_), border, thickness=2)


def status_pill(frame, x, y, label, value, color, active):
    dot_color = color if active else (80, 80, 80)
    cv2.circle(frame, (x, y), 7, dot_color, thickness=-1)
    cv2.circle(frame, (x, y), 7, (255, 255, 255), thickness=1)
    cv2.putText(frame, label, (x + 16, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_TEXT, 1, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(frame, value, (x + 210 - tw, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, dot_color, 2, cv2.LINE_AA)


def draw_speedometer(frame, cx, cy, radius, speed, max_speed, color):
    """Big semicircular speedometer dial, bottom-center of the window."""
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, 180, 360, (55, 55, 55), 14)
    frac = min(1.0, speed / max_speed)
    end_angle = 180 + int(180 * frac)
    dial_color = color if frac < 0.85 else COL_RED
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, 180, end_angle, dial_color, 14)

    # tick marks
    for i in range(0, 11):
        a = math.radians(180 + i * 18)
        x1 = int(cx + (radius - 20) * math.cos(a))
        y1 = int(cy + (radius - 20) * math.sin(a))
        x2 = int(cx + (radius - 5) * math.cos(a))
        y2 = int(cy + (radius - 5) * math.sin(a))
        cv2.line(frame, (x1, y1), (x2, y2), (150, 150, 150), 2)

    # needle
    needle_angle = math.radians(180 + 180 * frac)
    nx = int(cx + (radius - 25) * math.cos(needle_angle))
    ny = int(cy + (radius - 25) * math.sin(needle_angle))
    cv2.line(frame, (cx, cy), (nx, ny), COL_YELLOW, 3)
    cv2.circle(frame, (cx, cy), 8, COL_YELLOW, -1)

    glow_text(frame, f"{int(speed)}", (cx - 38, cy - 18), 1.1, (255, 255, 255), 2, glow=4)
    cv2.putText(frame, "KM/H", (cx - 26, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_DIM, 1, cv2.LINE_AA)


def draw_gear_indicator(frame, x, y, gear, color):
    rounded_panel(frame, x - 35, y - 35, 70, 70, color=(15, 15, 15), alpha=0.6, border=color)
    (tw, th), _ = cv2.getTextSize(gear, cv2.FONT_HERSHEY_DUPLEX, 1.1, 3)
    cv2.putText(frame, gear, (x - tw // 2, y + th // 2),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 3, cv2.LINE_AA)


def draw_dashboard(frame, steer_state, gas_state, angle, swap_on, fps, speed, t,
                    dead_zone, elapsed):
    h, w = frame.shape[:2]

    frame[:] = apply_vignette(frame, strength=0.75)
    draw_scanlines(frame, alpha=0.06)

    pulse = 0.5 + 0.5 * math.sin(t * 3)
    title_color = tuple(int(c * (0.6 + 0.4 * pulse)) for c in COL_CYAN)
    draw_corner_brackets(frame, title_color)

    # top title bar
    rounded_panel(frame, w // 2 - 190, 8, 380, 40, color=(10, 10, 10), alpha=0.5, border=title_color)
    glow_text(frame, "VIRTUAL RACING HUD", (w // 2 - 172, 36), 0.75, title_color, 2, glow=4)

    # left status panel
    rounded_panel(frame, 14, 60, 260, 156, color=(10, 10, 10), alpha=0.55, border=COL_CYAN)
    cv2.putText(frame, "STATUS", (28, 82), cv2.FONT_HERSHEY_DUPLEX, 0.55, COL_CYAN, 1, cv2.LINE_AA)

    steer_color = COL_GREEN if steer_state == "CENTER" else COL_AMBER if steer_state != "NO HAND" else COL_RED
    gas_color = COL_GREEN if gas_state.startswith("ACCEL") else COL_RED if gas_state.startswith("BRAKE") else COL_DIM

    status_pill(frame, 34, 110, "STEER", steer_state, steer_color, steer_state != "NO HAND")
    status_pill(frame, 34, 138, "PEDAL", gas_state, gas_color, gas_state != "NO HAND")
    status_pill(frame, 34, 166, "SWAP-S", "ON" if swap_on else "OFF", COL_MAGENTA, swap_on)

    mins, secs = divmod(int(elapsed), 60)
    cv2.putText(frame, f"SESSION {mins:02d}:{secs:02d}", (34, 194),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_DIM, 1, cv2.LINE_AA)

    # top-right decorative steering angle gauge
    gauge_cx, gauge_cy = w - 80, 100
    cv2.ellipse(frame, (gauge_cx, gauge_cy), (55, 55), 0, 135, 405, (60, 60, 60), 10)
    frac = min(1.0, abs(angle) / 60.0)
    end_angle = 135 + int(270 * frac)
    cv2.ellipse(frame, (gauge_cx, gauge_cy), (55, 55), 0, 135, end_angle, COL_CYAN, 10)
    cv2.putText(frame, f"{int(abs(angle))}", (gauge_cx - 20, gauge_cy + 8),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, COL_TEXT, 2, cv2.LINE_AA)
    cv2.putText(frame, "DEG", (gauge_cx - 16, gauge_cy + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_DIM, 1, cv2.LINE_AA)

    # bottom-center speedometer
    draw_speedometer(frame, w // 2, h - 15, 95, speed, MAX_SPEED, COL_CYAN)

    # gear indicator next to speedometer
    gear = "D" if gas_state.startswith("ACCEL") else "R" if gas_state.startswith("BRAKE") else "N"
    gear_color = COL_GREEN if gear == "D" else COL_RED if gear == "R" else COL_DIM
    draw_gear_indicator(frame, w // 2 + 150, h - 60, gear, gear_color)

    # bottom-left FPS + hints
    cv2.putText(frame, f"FPS {fps:.0f}  |  DEADZONE {dead_zone}deg", (14, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_DIM, 1, cv2.LINE_AA)
    cv2.putText(frame, "Q QUIT  S SWAP  C SNAPSHOT  +/- SENSITIVITY", (14, h - 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_DIM, 1, cv2.LINE_AA)


# ----------------------------- MAIN LOOP --------------------------------- #
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_ANY)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    missing_steer_frames = 0
    missing_gas_frames = 0
    steer_state_text = "CENTER"
    gas_state_text = "NEUTRAL"
    current_angle = 0.0
    smoothed_angle = 0.0
    speed = 0.0
    global SWAP_HANDS, DEAD_ZONE_DEG

    prev_t = time.time()
    fps = 0.0
    start_t = time.time()
    snapshot_flash_until = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if FLIP_CAMERA:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            now = time.time()
            dt = now - prev_t
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, dt))
            prev_t = now

            found_steer = False
            found_gas = False

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    raw_label = handedness.classification[0].label
                    label = on_screen_label(raw_label)
                    lm = hand_landmarks.landmark

                    if label == STEERING_HAND:
                        found_steer = True
                        missing_steer_frames = 0
                        raw_angle = steering_tilt_angle(lm)
                        smoothed_angle = (STEER_SMOOTHING * smoothed_angle
                                          + (1 - STEER_SMOOTHING) * raw_angle)
                        angle = smoothed_angle
                        current_angle = angle

                        if angle > DEAD_ZONE_DEG:
                            set_key(Key.right, True)
                            set_key(Key.left, False)
                            steer_state_text = "RIGHT"
                        elif angle < -DEAD_ZONE_DEG:
                            set_key(Key.left, True)
                            set_key(Key.right, False)
                            steer_state_text = "LEFT"
                        else:
                            set_key(Key.left, False)
                            set_key(Key.right, False)
                            steer_state_text = "CENTER"

                        wheel_color = COL_GREEN if steer_state_text == "CENTER" else COL_AMBER
                        draw_wheel_on_hand(frame, lm, angle, w, h, wheel_color)

                    elif label == GAS_BRAKE_HAND:
                        found_gas = True
                        missing_gas_frames = 0

                        if is_fist(lm):
                            set_key(Key.up, True)
                            set_key(Key.down, False)
                            gas_state_text = "ACCEL (fist)"
                            draw_fist_burst(frame, lm, w, h, COL_GREEN)
                        elif is_one_finger(lm):
                            set_key(Key.down, True)
                            set_key(Key.up, False)
                            gas_state_text = "BRAKE (1 finger)"
                            draw_fist_burst(frame, lm, w, h, COL_RED)
                        else:
                            set_key(Key.up, False)
                            set_key(Key.down, False)
                            gas_state_text = "NEUTRAL"

                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style(),
                        )

            if not found_steer:
                missing_steer_frames += 1
                if missing_steer_frames > GRACE_FRAMES:
                    set_key(Key.left, False)
                    set_key(Key.right, False)
                    steer_state_text = "NO HAND"
                    current_angle = 0.0

            if not found_gas:
                missing_gas_frames += 1
                if missing_gas_frames > GRACE_FRAMES:
                    set_key(Key.up, False)
                    set_key(Key.down, False)
                    gas_state_text = "NO HAND"

            # cosmetic speed simulation for the speedometer dial
            if gas_state_text.startswith("ACCEL"):
                speed = min(MAX_SPEED, speed + SPEED_ACCEL_RATE * dt)
            elif gas_state_text.startswith("BRAKE"):
                speed = max(0.0, speed - SPEED_BRAKE_RATE * dt)
            else:
                speed = max(0.0, speed - SPEED_DECAY_RATE * dt)

            draw_dashboard(frame, steer_state_text, gas_state_text, current_angle,
                            SWAP_HANDS, fps, speed, now - start_t,
                            DEAD_ZONE_DEG, now - start_t)

            # snapshot flash feedback
            if now < snapshot_flash_until:
                flash = frame.copy()
                flash[:] = (255, 255, 255)
                cv2.addWeighted(flash, 0.25, frame, 0.75, 0, frame)

            cv2.imshow("Virtual Steering Wheel v3 - Gaming HUD", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                SWAP_HANDS = not SWAP_HANDS
                release_all()
            elif key == ord("c"):
                fname = time.strftime("steering_snapshot_%Y%m%d_%H%M%S.png")
                cv2.imwrite(fname, frame)
                snapshot_flash_until = now + 0.15
                print(f"[SNAPSHOT] saved {fname}")
            elif key in (ord("+"), ord("=")):
                DEAD_ZONE_DEG = min(45, DEAD_ZONE_DEG + SENSITIVITY_STEP)
            elif key == ord("-"):
                DEAD_ZONE_DEG = max(0, DEAD_ZONE_DEG - SENSITIVITY_STEP)

    finally:
        release_all()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
