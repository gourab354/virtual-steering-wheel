"""

 FRUIT SLICE  —  Fruit-Ninja style game.

 HOW TO PLAY
   1. Step out of frame, press B to capture the background.
   2. Fruits launch from the bottom, arc up, fall back down under gravity.
   3. Touch a fruit with your INDEX FINGERTIP to slice it (+1 score).
   4. Let 5 fruits fall off the bottom -> you disintegrate -> Game Over.
   5. The longer you survive, the faster & higher the fruits fly.

 CONTROLS
   B : capture background (only during the pre-game capture screen)
   R : restart after Game Over
   Q / ESC : quit

 Requirements:
   pip install opencv-python mediapipe numpy pygame

 MODEL FILES (auto-downloaded next to this script on first run):
   hand_landmarker.task       (~7 MB)   -> fingertip tracking
   selfie_segmenter.tflite    (~250 KB) -> person mask for the dissolve

"""

import os
import time
import random
import math
import urllib.request
import wave
import struct
import threading

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision

try:
    import pygame
    HAVE_PYGAME = True
except ImportError:
    HAVE_PYGAME = False


# MODEL FILES

_HERE      = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL = os.path.join(_HERE, "hand_landmarker.task")
SEG_MODEL  = os.path.join(_HERE, "selfie_segmenter.tflite")

HAND_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task")
SEG_URL  = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
            "selfie_segmenter/float16/latest/selfie_segmenter.tflite")


def ensure_model(path, url):
    if os.path.exists(path):
        return
    print(f"Downloading {os.path.basename(path)} ...")
    try:
        urllib.request.urlretrieve(url, path)
        print("  done.")
    except Exception as e:
        raise SystemExit(
            f"\nCould not download {os.path.basename(path)}:\n  {e}\n"
            f"Please download it manually from:\n  {url}\n"
            f"and place it next to this script."
        )


# TUNABLES

CAM_INDEX = 0
FRAME_W   = 640
FRAME_H   = 480

MAX_MISSES = 5

# --- difficulty ramp ---
SPAWN_INTERVAL_START = 1.1
SPAWN_INTERVAL_MIN   = 0.35
LAUNCH_SPEED_START   = 16.0
LAUNCH_SPEED_MAX     = 23.0
RAMP_TIME            = 40.0

GRAVITY       = 0.45
FRUIT_RADIUS  = 30

FRUIT_COLORS = [
    ("Apple",      (60,  60, 220)),
    ("Orange",     (30, 140, 250)),
    ("Watermelon", (70, 180,  60)),
    ("Grape",      (180,  60, 160)),
    ("Lemon",      (40, 220, 230)),
]

_FONT = cv2.FONT_HERSHEY_SIMPLEX



# GAME-OVER SOUND  —  synthesized once, cached to _fahh_synth.wav
#
# NOTE: this is a synthesized approximation of the "FAHHHH" meme scream
# built from scratch with a source-filter voice model (no copyrighted
# audio is embedded or downloaded). If you have the real clip, just drop
# a file named fahh.wav / fahh.mp3 / fahh.ogg next to this script and
# it will be used automatically instead of the synthesized version.

FAHH_BASENAME  = "fahh"              # drop your own real clip named fahh.wav/.mp3/.ogg to override
FAHH_EXTS      = (".wav", ".mp3", ".ogg")
FAHH_SYNTH_WAV = os.path.join(_HERE, "_fahh_synth.wav")   # fallback synth cache


def find_user_fahh_clip():
    """Look for a real user-supplied fahh.(wav|mp3|ogg) next to the script."""
    for ext in FAHH_EXTS:
        p = os.path.join(_HERE, FAHH_BASENAME + ext)
        if os.path.exists(p):
            return p
    return None


def _biquad_bandpass(f0, bw, fs):
    """RBJ bandpass biquad coefficients (constant 0dB peak gain)."""
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) * np.sinh(np.log(2) / 2 * (bw / f0) * w0 / np.sin(w0))
    b0, b1, b2 = alpha, 0.0, -alpha
    a0, a1, a2 = 1 + alpha, -2 * np.cos(w0), 1 - alpha
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a


def _lfilter(b, a, x):
    """Minimal direct-form-II biquad filter (avoids a scipy dependency)."""
    y = np.zeros_like(x)
    x1 = x2 = y1 = y2 = 0.0
    for i in range(len(x)):
        xi = x[i]
        yi = b[0] * xi + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
        x2, x1 = x1, xi
        y2, y1 = y1, yi
        y[i] = yi
    return y


def synthesize_fahh_wav(path, sample_rate=44100):
    """Vocal-formant style descending 'faaahhh' scream: a buzzy glottal
    pulse train shaped by vowel formants, plus a breathy noise onset."""
    duration = 1.0
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    rng = np.random.default_rng(42)

    # pitch contour: quick rise then a long dramatic fall
    rise_n = int(0.08 * n)
    fall_n = n - rise_n
    pitch = np.concatenate([
        np.linspace(180, 340, rise_n),
        np.linspace(340, 70, fall_n) * (1 + 0.02 * np.sin(np.linspace(0, 30, fall_n))),
    ])

    # glottal buzz via summed harmonics of a saw-like pulse train
    phase = 2 * np.pi * np.cumsum(pitch) / sample_rate
    buzz = np.zeros(n)
    for k in range(1, 9):
        buzz += np.sin(k * phase) / k
    buzz /= np.max(np.abs(buzz)) + 1e-9

    # vowel formants (open "ah") for a vocal-tract-ish tone
    formant_sig = 0.7 * _lfilter(*_biquad_bandpass(750, 120, sample_rate), buzz) \
                + 0.5 * _lfilter(*_biquad_bandpass(1200, 160, sample_rate), buzz)
    formant_sig = formant_sig / (np.max(np.abs(formant_sig)) + 1e-9)

    # breathy noise, strongest at the sharp intake onset
    noise = rng.uniform(-1, 1, n)
    breath_env = np.exp(-t / 0.12) * 0.8 + 0.08
    breath = noise * breath_env

    # amplitude envelope: fast attack, sustained belt, long tail-off
    env = np.piecewise(
        t,
        [t < 0.05, (t >= 0.05) & (t < 0.55), t >= 0.55],
        [lambda x: x / 0.05, lambda x: 1.0, lambda x: np.exp(-(x - 0.55) / 0.35)]
    )

    sig = (formant_sig * 0.85 + breath * 0.35) * env
    sig = sig / (np.max(np.abs(sig)) + 1e-9)

    pcm = (sig * 32767 * 0.9).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<%dh" % len(pcm), *pcm))


_fahh_sound_obj = None


def init_audio():
    global _fahh_sound_obj
    if not HAVE_PYGAME:
        print("pygame not installed -- running without sound "
              "(pip install pygame to enable the game-over SFX).")
        return
    try:
        pygame.mixer.init()
        user_clip = find_user_fahh_clip()
        if user_clip:
            # a real user-supplied clip is present -- use it as-is
            sound_path = user_clip
        else:
            if not os.path.exists(FAHH_SYNTH_WAV):
                print("Synthesizing fahh sound effect ...")
                synthesize_fahh_wav(FAHH_SYNTH_WAV)
                print("  done.")
            sound_path = FAHH_SYNTH_WAV
        _fahh_sound_obj = pygame.mixer.Sound(sound_path)
    except Exception as e:
        print(f"Audio init failed ({e}); continuing without sound.")
        _fahh_sound_obj = None


def play_dissolve_sound():
    if _fahh_sound_obj is None:
        return

    def _worker():
        try:
            _fahh_sound_obj.play()
        except Exception:
            pass

    threading.Thread(target=_worker, daemon=True).start()



# TEXT TARGET POINTS  —  "GAME OVER" rendered to pixel coordinates that the
# embers will be steered toward during the reform phase.

def build_text_target_points(w, h, text="GAME OVER", max_points=6000):
    canvas = np.zeros((h, w), dtype=np.uint8)
    scale = w / 420.0
    thickness = max(2, int(6 * scale))
    (tw, th), _ = cv2.getTextSize(text, _FONT, scale * 1.6, thickness)
    x = (w - tw) // 2
    y = (h + th) // 2
    cv2.putText(canvas, text, (x, y), _FONT, scale * 1.6, 255, thickness, cv2.LINE_AA)

    ys, xs = np.where(canvas > 0)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[idx]
    return pts



# DISSOLVE + REFORM EFFECT
#
# Phase machine per particle set:
#   SWEEP    -> body dissolves left-to-right, embers blow outward (as before)
#   SCATTER  -> a short beat of continued outward drift (lets it breathe)
#   GATHER   -> every ember steers toward an assigned "GAME OVER" pixel
#   HOLD     -> letters sit formed, gently glowing, then done

MAX_PARTICLES   = 30000
SWEEP_SPEED     = 0.028
EDGE_GLOW       = 0.06
BUOYANCY        = 0.35
GRAVITY_EMBER   = 0.06
TURBULENCE      = 0.45
OUTWARD_SPREAD  = 1.2
DAMPING         = 0.97
DUST_INTENSITY  = 1.7
EMBER_BGR       = np.array([30, 120, 255], dtype=np.float32)

SCATTER_FRAMES  = 25
GATHER_FRAMES   = 55
HOLD_FRAMES     = 90
GATHER_EASE     = 0.10

_SEG_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def mask_from_segmentation(result, w, h):
    m = None
    if result.confidence_masks:
        conf = result.confidence_masks[0].numpy_view()
        m = (conf > 0.5).astype(np.uint8)
    elif result.category_mask is not None:
        cat = result.category_mask.numpy_view()
        m = (cat > 0).astype(np.uint8)
    if m is None:
        return np.zeros((h, w), dtype=bool)
    if m.shape[0] != h or m.shape[1] != w:
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  _SEG_KERNEL)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _SEG_KERNEL)
    return m.astype(bool)


class DisintegrationReform:
    """Freeze the current frame + person mask, dissolve into embers, then
    steer every ember to assemble into the GAME OVER text."""

    def __init__(self, frozen_frame, mask, background, noise, target_pts):
        self.frame = frozen_frame
        self.mask  = mask
        self.bg    = background
        self.noise = noise
        self.phase = "SWEEP"
        self.sweep_progress = 0.0
        self.phase_frame = 0
        self.done = False

        h, w = mask.shape
        ys, xs = np.where(mask)
        if len(xs) == 0:
            self.done = True
            self.n = 0
            return

        if len(xs) > MAX_PARTICLES:
            idx = np.random.choice(len(xs), MAX_PARTICLES, replace=False)
            ys, xs = ys[idx], xs[idx]
        self.n = len(xs)

        self.base_col = self.frame[ys, xs].astype(np.float32)

        n_at = self.noise[ys, xs]
        sweep = xs / float(w)
        self.birth = np.clip(0.55 * n_at + 0.45 * sweep, 0.0, 1.0)

        self.x = xs.astype(np.float32)
        self.y = ys.astype(np.float32)

        cx, cy = xs.mean(), ys.mean()
        dx, dy = self.x - cx, self.y - cy
        norm = np.sqrt(dx * dx + dy * dy) + 1e-6
        self.vx = (dx / norm) * OUTWARD_SPREAD + np.random.randn(self.n) * 0.3
        self.vy = (dy / norm) * OUTWARD_SPREAD * 0.4 \
                  - BUOYANCY * 2.0 + np.random.randn(self.n) * 0.3

        # assign each ember a target text pixel (cycle through if fewer
        # target points than embers, so letters still look dense)
        if len(target_pts) == 0:
            target_pts = np.array([[w / 2, h / 2]], dtype=np.float32)
        reps = int(np.ceil(self.n / len(target_pts)))
        tiled = np.tile(target_pts, (reps, 1))[: self.n]
        np.random.shuffle(tiled)
        self.target_x = tiled[:, 0]
        self.target_y = tiled[:, 1]

    def step(self):
        if self.done or self.n == 0:
            self.done = True
            return

        if self.phase == "SWEEP":
            self.sweep_progress += SWEEP_SPEED
            born = self.birth <= self.sweep_progress
            if born.any():
                self.vy[born] -= BUOYANCY
                self.vy[born] += GRAVITY_EMBER
                self.vx[born] += (np.random.rand(born.sum()) - 0.5) * TURBULENCE
                self.vy[born] += (np.random.rand(born.sum()) - 0.5) * TURBULENCE
                self.vx[born] *= DAMPING
                self.vy[born] *= DAMPING
                self.x[born] += self.vx[born]
                self.y[born] += self.vy[born]
            if self.sweep_progress > 1.05:
                self.phase = "SCATTER"
                self.phase_frame = 0

        elif self.phase == "SCATTER":
            self.vy += GRAVITY_EMBER
            self.vx += (np.random.rand(self.n) - 0.5) * TURBULENCE * 0.6
            self.vy += (np.random.rand(self.n) - 0.5) * TURBULENCE * 0.6
            self.vx *= DAMPING
            self.vy *= DAMPING
            self.x += self.vx
            self.y += self.vy
            self.phase_frame += 1
            if self.phase_frame > SCATTER_FRAMES:
                self.phase = "GATHER"
                self.phase_frame = 0

        elif self.phase == "GATHER":
            self.vx *= 0.85
            self.vy *= 0.85
            self.x += (self.target_x - self.x) * GATHER_EASE + self.vx * 0.15
            self.y += (self.target_y - self.y) * GATHER_EASE + self.vy * 0.15
            self.phase_frame += 1
            if self.phase_frame > GATHER_FRAMES:
                self.phase = "HOLD"
                self.phase_frame = 0
                self.x = self.target_x.copy()
                self.y = self.target_y.copy()

        elif self.phase == "HOLD":
            self.phase_frame += 1
            if self.phase_frame > HOLD_FRAMES:
                self.done = True

    def render(self):
        out = self.bg.copy()
        h, w = self.mask.shape

        if self.phase == "SWEEP":
            solid = self.mask & (self.noise > self.sweep_progress)
            out[solid] = self.frame[solid]
            edge = self.mask & (self.noise > self.sweep_progress) \
                             & (self.noise < self.sweep_progress + EDGE_GLOW)
            if edge.any():
                glow = out[edge].astype(np.float32)
                glow = glow * 0.4 + EMBER_BGR * 0.9
                out[edge] = np.clip(glow, 0, 255).astype(np.uint8)
            active = self.birth <= self.sweep_progress
        else:
            active = np.ones(self.n, dtype=bool)

        if self.n > 0 and active.any():
            ax = self.x[active]
            ay = self.y[active]
            base = self.base_col[active]

            inb = (ax >= 0) & (ax < w) & (ay >= 0) & (ay < h)
            ax, ay = ax[inb].astype(np.int32), ay[inb].astype(np.int32)
            base = base[inb]

            if self.phase in ("GATHER", "HOLD"):
                blend = 0.85 if self.phase == "HOLD" else \
                        min(1.0, self.phase_frame / GATHER_FRAMES)
                pulse = 1.0
                if self.phase == "HOLD":
                    pulse = 0.85 + 0.15 * math.sin(self.phase_frame * 0.25)
                col = (base * (1 - blend) + EMBER_BGR * blend) * pulse
            else:
                col = base * 0.9 + EMBER_BGR * 0.3

            dust = np.zeros((h, w, 3), dtype=np.float32)
            np.add.at(dust, (ay, ax), col)
            blur = 1.4 if self.phase != "HOLD" else 0.8
            dust = cv2.GaussianBlur(dust, (0, 0), blur) * DUST_INTENSITY
            out = np.clip(out.astype(np.float32) + dust, 0, 255).astype(np.uint8)

        return out



# FRUIT

class Fruit:
    __slots__ = ("x", "y", "vx", "vy", "radius", "name", "color",
                 "sliced", "rot", "rot_speed", "spawn_t")

    def __init__(self, w, h, launch_speed):
        self.x = random.uniform(w * 0.2, w * 0.8)
        self.y = h + FRUIT_RADIUS
        self.vx = random.uniform(-1.5, 1.5)
        self.vy = -launch_speed * random.uniform(0.85, 1.05)
        self.radius = FRUIT_RADIUS
        self.name, self.color = random.choice(FRUIT_COLORS)
        self.sliced = False
        self.rot = 0.0
        self.rot_speed = random.uniform(-6, 6)
        self.spawn_t = time.time()

    def step(self):
        self.vy += GRAVITY
        self.x += self.vx
        self.y += self.vy
        self.rot += self.rot_speed

    def offscreen_bottom(self, h):
        return self.y - self.radius > h

    def contains(self, px, py):
        return math.hypot(self.x - px, self.y - py) < self.radius + 6

    def draw(self, img):
        c = (int(self.x), int(self.y))
        cv2.circle(img, c, self.radius, self.color, -1, cv2.LINE_AA)
        cv2.circle(img, c, self.radius, (255, 255, 255), 2, cv2.LINE_AA)
        leaf_x = int(self.x + self.radius * 0.3 * math.cos(math.radians(self.rot)))
        leaf_y = int(self.y - self.radius * 0.9)
        cv2.ellipse(img, (leaf_x, leaf_y), (7, 4), self.rot, 0, 360,
                   (60, 160, 60), -1, cv2.LINE_AA)


class SliceFX:
    __slots__ = ("x", "y", "color", "radius", "life", "max_life",
                 "ang", "pieces_v")

    def __init__(self, fruit, slice_angle):
        self.x, self.y = fruit.x, fruit.y
        self.color = fruit.color
        self.radius = fruit.radius
        self.ang = slice_angle
        self.max_life = 18
        self.life = self.max_life
        perp = slice_angle + math.pi / 2
        self.pieces_v = [
            (math.cos(perp) * 3.0,  math.sin(perp) * 3.0 - 2.0),
            (-math.cos(perp) * 3.0, -math.sin(perp) * 3.0 - 2.0),
        ]

    def step(self):
        self.life -= 1

    def done(self):
        return self.life <= 0

    def draw(self, img):
        t = 1.0 - (self.life / self.max_life)
        for i, (dx, dy) in enumerate(self.pieces_v):
            ox = int(self.x + dx * (self.max_life - self.life))
            oy = int(self.y + dy * (self.max_life - self.life)
                      + 0.4 * (self.max_life - self.life) ** 2 * 0.05)
            start = 90 if i == 0 else 270
            cv2.ellipse(img, (ox, oy), (self.radius, self.radius),
                       math.degrees(self.ang), start, start + 180,
                       self.color, -1, cv2.LINE_AA)
        alpha = max(0.0, 1.0 - t * 1.3)
        if alpha > 0:
            L = self.radius + 14
            p1 = (int(self.x - L * math.cos(self.ang)), int(self.y - L * math.sin(self.ang)))
            p2 = (int(self.x + L * math.cos(self.ang)), int(self.y + L * math.sin(self.ang)))
            overlay = img.copy()
            cv2.line(overlay, p1, p2, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)



# HUD

def draw_banner(img, text, at_top=True, scale=0.6, thick=1):
    (tw, th), base = cv2.getTextSize(text, _FONT, scale, thick)
    h, w = img.shape[:2]
    strip = th + base + 18
    if at_top:
        y0, y1, ty = 0, strip, 9 + th
    else:
        y0, y1, ty = h - strip, h, h - 9 - base
    overlay = img.copy()
    cv2.rectangle(overlay, (0, y0), (w, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
    cv2.putText(img, text, (12, ty), _FONT, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
    cv2.putText(img, text, (12, ty), _FONT, scale, (255, 255, 255), thick, cv2.LINE_AA)


def draw_hearts(img, misses_left, total=MAX_MISSES):
    h, w = img.shape[:2]
    r = 10
    for i in range(total):
        cx = w - 20 - i * (2 * r + 8)
        cy = 30
        alive = i < misses_left
        col = (60, 60, 255) if alive else (70, 70, 70)
        cv2.circle(img, (cx, cy), r, col, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), r, (255, 255, 255), 1, cv2.LINE_AA)


def draw_trail(img, pts):
    n = len(pts)
    for i in range(1, n):
        alpha = i / n
        thickness = max(1, int(6 * alpha))
        cv2.line(img, pts[i - 1], pts[i], (255, 255, 255), thickness, cv2.LINE_AA)



# MAIN

def main():
    ensure_model(HAND_MODEL, HAND_URL)
    ensure_model(SEG_MODEL, SEG_URL)
    init_audio()

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        print("ERROR: could not open webcam. Try a different CAM_INDEX.")
        return

    BaseOptions   = mp.tasks.BaseOptions
    VisionRunning = mp_vision.RunningMode

    hand_options = mp_vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=VisionRunning.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)

    seg_options = mp_vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=SEG_MODEL),
        running_mode=VisionRunning.VIDEO,
        output_confidence_masks=True,
        output_category_mask=True,
    )
    segmenter = mp_vision.ImageSegmenter.create_from_options(seg_options)

    dissolve_noise = np.random.rand(FRAME_H, FRAME_W).astype(np.float32)
    game_over_targets = build_text_target_points(FRAME_W, FRAME_H, "GAME OVER")

    def new_game():
        return {
            "state": "CAPTURE_BG",     # CAPTURE_BG -> PLAYING -> DYING -> GAME_OVER
            "background": None,
            "fruits": [],
            "fx": [],
            "misses": 0,
            "score": 0,
            "start_t": None,
            "last_spawn": None,
            "trails": {},
            "effect": None,
            "final_render": None,
        }

    g = new_game()

    frame_count = 0
    prev_t = time.time()
    fps = 0.0

    print("FRUIT SLICE — step out and press B to capture background.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=np.ascontiguousarray(rgb))
        ts = frame_count * 33
        frame_count += 1

        display = frame.copy()

        # ---- CAPTURE_BG: wait for the player to step out and hit B ----
        if g["state"] == "CAPTURE_BG":
            draw_banner(display, "Step OUT of frame, then press [B] to capture background",
                       at_top=True)
            draw_banner(display, "[Q] quit", at_top=False)

        # ---- PLAYING ----
        elif g["state"] == "PLAYING":
            hand_result = landmarker.detect_for_video(mp_image, ts)
            fingertips = []
            if hand_result.hand_landmarks:
                for lm in hand_result.hand_landmarks:
                    tip = lm[8]
                    fingertips.append((int(tip.x * FRAME_W), int(tip.y * FRAME_H)))

            elapsed = time.time() - g["start_t"]
            ramp = min(elapsed / RAMP_TIME, 1.0)
            spawn_interval = SPAWN_INTERVAL_START + \
                (SPAWN_INTERVAL_MIN - SPAWN_INTERVAL_START) * ramp
            launch_speed = LAUNCH_SPEED_START + \
                (LAUNCH_SPEED_MAX - LAUNCH_SPEED_START) * ramp

            now = time.time()
            if now - g["last_spawn"] > spawn_interval:
                g["last_spawn"] = now
                g["fruits"].append(Fruit(FRAME_W, FRAME_H, launch_speed))
                if ramp > 0.4 and random.random() < 0.25:
                    g["fruits"].append(Fruit(FRAME_W, FRAME_H, launch_speed))

            seen_idx = set()
            for i, tip in enumerate(fingertips):
                seen_idx.add(i)
                tr = g["trails"].setdefault(i, [])
                tr.append(tip)
                if len(tr) > 10:
                    tr.pop(0)
            for i in list(g["trails"].keys()):
                if i not in seen_idx:
                    g["trails"][i] = []

            for i, tr in g["trails"].items():
                if not tr:
                    continue
                tip = tr[-1]
                p0 = tr[-2] if len(tr) >= 2 else tip
                for f in g["fruits"]:
                    if f.sliced:
                        continue
                    if f.contains(tip[0], tip[1]):
                        f.sliced = True
                        g["score"] += 1
                        ang = math.atan2(tip[1] - p0[1], tip[0] - p0[0]) \
                            if tip != p0 else 0.0
                        g["fx"].append(SliceFX(f, ang))

            still_flying = []
            for f in g["fruits"]:
                if f.sliced:
                    continue
                f.step()
                if f.offscreen_bottom(FRAME_H):
                    g["misses"] += 1
                    continue
                still_flying.append(f)
            g["fruits"] = still_flying

            g["fx"] = [fx for fx in g["fx"] if not fx.done()]
            for fx in g["fx"]:
                fx.step()

            for f in g["fruits"]:
                f.draw(display)
            for fx in g["fx"]:
                fx.draw(display)
            for tr in g["trails"].values():
                draw_trail(display, tr)
            for tip in fingertips:
                cv2.circle(display, tip, 6, (255, 255, 255), -1, cv2.LINE_AA)

            draw_banner(display, f"SCORE: {g['score']}", at_top=True)
            draw_hearts(display, MAX_MISSES - g["misses"])
            draw_banner(display, "[R] restart   [Q] quit", at_top=False)

            if g["misses"] >= MAX_MISSES:
                play_dissolve_sound()
                seg_result = segmenter.segment_for_video(mp_image, ts)
                mask = mask_from_segmentation(seg_result, FRAME_W, FRAME_H)
                g["effect"] = DisintegrationReform(
                    frame.copy(), mask.copy(), g["background"],
                    dissolve_noise, game_over_targets
                )
                g["state"] = "DYING"

        # ---- DYING: sweep -> scatter -> gather into "GAME OVER" text ----
        elif g["state"] == "DYING":
            g["effect"].step()
            display = g["effect"].render()
            draw_banner(display, f"SCORE: {g['score']}", at_top=True)
            if g["effect"].done:
                g["final_render"] = display.copy()
                g["state"] = "GAME_OVER"

        # ---- GAME_OVER: hold the assembled text, wait for restart ----
        elif g["state"] == "GAME_OVER":
            display = g["final_render"].copy()
            draw_banner(display, f"Fruits sliced: {g['score']}   [R] restart   [Q] quit",
                       at_top=False)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-6))
        prev_t = now

        cv2.imshow("Fruit Slice", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('b') and g["state"] == "CAPTURE_BG":
            g["background"] = frame.copy()
            g["state"] = "PLAYING"
            g["start_t"] = time.time()
            g["last_spawn"] = time.time()
            print("Background captured. Go!")
        elif key == ord('r'):
            g = new_game()
            print("Restarted — step out and press [B] to capture background.")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    segmenter.close()


if __name__ == "__main__":
    main()
