"""
app.py — Bimanual 2D Spatial Operating Interface
=================================================
Pure OpenCV clinical dashboard. MediaPipe Tasks API (ARM64 v0.10+).
Hardware: Auto-hunting camera init across [1, 0, 2] to bypass Continuity Camera drops.
Window: cv2.WINDOW_AUTOSIZE on "Dashboard" eliminating distortion.
Zero AI Slop. Zero 3D/PyVista dependencies. Zero multithreading.

State machine:
  IDLE   — Tracks hands, dual-palm open held simultaneously for 2.0s unlocks ACTIVE.
  ACTIVE — Bimanual 2D Kinematics:
           1 Hand  : Index tip (LM 8) delta → Translate (pan).
           2 Hands : Euclidean distance between index tips → Scale (zoom).
                     math.atan2 between index tips → 2D Rotation.
                     Midpoint delta between index tips → Translate anchor.
           All channels smoothed with Exponential Moving Average (alpha = 0.15).
           Rendered via cv2.warpAffine and cv2.getRotationMatrix2D.

Spatial sorting:
  Sort detected hands strictly by wrist X-coordinate (LM 0) ascending:
  Index 0 = Left hand, Index 1 = Right hand (MediaPipe handedness labels ignored).
"""

from __future__ import annotations

import math
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
EMA_ALPHA           = 0.15
ACTIVATION_TIME_SEC = 2.0

CANVAS_W = 1400
CANVAS_H = 700
PANEL_W  = CANVAS_W // 2  # 700 px each

LM_WRIST      = 0
LM_INDEX_TIP  = 8
LM_FINGERTIPS = [8, 12, 16, 20]
LM_PIPS       = [6, 10, 14, 18]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

BASE_DIR   = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

FONT    = cv2.FONT_HERSHEY_SIMPLEX
C_CYAN  = (0, 255, 220)
C_LIME  = (0, 255, 100)
C_AMBER = (0, 190, 255)
C_DIM   = (90, 90, 90)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers & Signal Processing
# ─────────────────────────────────────────────────────────────────────────────
def _ema(prev: float, new: float) -> float:
    return EMA_ALPHA * new + (1.0 - EMA_ALPHA) * prev


def _is_open_palm(hand: list) -> bool:
    """True if all four fingertips are vertically higher (smaller Y) than their PIP joints."""
    for tip_i, pip_i in zip(LM_FINGERTIPS, LM_PIPS):
        if hand[tip_i].y >= hand[pip_i].y:
            return False
    return True


def _sort_hands(hands: list) -> list:
    """Sort hands strictly by wrist X-coordinate ascending → 0=Left, 1=Right."""
    return sorted(hands, key=lambda h: h[LM_WRIST].x)


def _download_model() -> None:
    if MODEL_PATH.exists():
        return
    print(f"[INFO] Downloading hand landmarker → {MODEL_PATH}")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("[INFO] Download complete.")


def _build_detector() -> HandLandmarker:
    _download_model()
    opts = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        num_hands=2,
        min_hand_detection_confidence=0.60,
        min_hand_presence_confidence=0.60,
        min_tracking_confidence=0.50,
    )
    return HandLandmarker.create_from_options(opts)


def _load_scan() -> np.ndarray:
    for name in ["stark_skull.jpg", "mock_scan.png"]:
        p = BASE_DIR / name
        img = cv2.imread(str(p))
        if img is not None:
            print(f"[IMG] Loaded {name} {img.shape[:2]}.")
            return img
    raise FileNotFoundError("[IMG] Neither stark_skull.jpg nor mock_scan.png found.")


def letterbox(img: np.ndarray, w: int, h: int) -> tuple[np.ndarray, float, int, int]:
    """Fit img into a w×h black canvas, preserving aspect ratio without distortion."""
    ih, iw = img.shape[:2]
    r = min(w / iw, h / ih)
    nw, nh = int(iw * r), int(ih * r)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    y0 = (h - nh) // 2
    x0 = (w - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, x0, y0


def _apply_transform(base: np.ndarray, tx: float, ty: float, scale: float, angle_deg: float) -> np.ndarray:
    h, w = base.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), -angle_deg, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(base, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def _draw_hand(panel: np.ndarray, hand: list, fw: int, fh: int,
               r: float, x0: int, y0: int, color: tuple, tag: str) -> tuple[int, int]:
    pts = [(int(lm.x * fw * r + x0), int(lm.y * fh * r + y0)) for lm in hand]
    for a, b in HAND_CONNECTIONS:
        cv2.line(panel, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(panel, pt, 4, color, -1, cv2.LINE_AA)
    wx, wy = pts[LM_WRIST]
    cv2.putText(panel, tag, (wx - 16, wy - 14), FONT, 0.50, color, 1, cv2.LINE_AA)
    return pts[LM_INDEX_TIP]


# ─────────────────────────────────────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:  # noqa: C901
    # ── 1. Robust Camera Auto-Hunting ─────────────────────────────────────────
    cap = None
    for i in [1, 0, 2]:
        temp_cap = cv2.VideoCapture(i)
        ret, frame = temp_cap.read()
        if ret and frame is not None:
            cap = temp_cap
            print(f"[CAM] Successfully locked actual webcam at index {i}")
            break
        temp_cap.release()
    if cap is None:
        raise RuntimeError("No active camera found.")

    detector = _build_detector()
    raw_scan = _load_scan()
    base_scan, _, _, _ = letterbox(raw_scan, PANEL_W, CANVAS_H)

    # ── State Machine ─────────────────────────────────────────────────────────
    active        = False
    left_palm_t:  float | None = None
    right_palm_t: float | None = None
    flash_end:    float | None = None
    flash_msg     = ""

    # ── EMA Spatial Transformation Channels (alpha = 0.15) ────────────────────
    s_tx    = 0.0
    s_ty    = 0.0
    s_scale = 1.0
    s_angle = 0.0  # degrees

    # ── Frame-to-frame Delta Anchors ──────────────────────────────────────────
    prev_ix:    float | None = None
    prev_iy:    float | None = None
    prev_dist:  float | None = None
    prev_angle: float | None = None
    prev_mid_x: float | None = None
    prev_mid_y: float | None = None

    WIN = "Dashboard"
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

    while True:
        # ── Bulletproof Frame Read ────────────────────────────────────────────
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Frame dropped. Retrying...")
            continue

        frame  = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        now    = time.time()

        # ── MediaPipe Hand Landmark Detection ─────────────────────────────────
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_img)
        raw    = result.hand_landmarks
        n      = len(raw)
        hands  = _sort_hands(raw) if n > 0 else []

        # ── Letterbox Camera Panel (aspect ratio strictly preserved) ──────────
        cam_panel, r_cam, ox, oy = letterbox(frame, PANEL_W, CANVAS_H)

        # ── State: IDLE ───────────────────────────────────────────────────────
        if not active:
            l_open = n >= 1 and _is_open_palm(hands[0])
            r_open = n >= 2 and _is_open_palm(hands[1])

            if l_open and r_open:
                if left_palm_t is None:
                    left_palm_t = now
            else:
                left_palm_t = None

            held_time = (now - left_palm_t) if left_palm_t else 0.0

            if held_time >= ACTIVATION_TIME_SEC:
                active       = True
                left_palm_t  = None
                flash_msg    = "INTERFACE UNLOCKED"
                flash_end    = now + 1.8
                prev_ix      = None
                prev_iy      = None
                prev_dist    = None
                prev_angle   = None
                prev_mid_x   = None
                prev_mid_y   = None

            # Render hand skeletons in IDLE
            for i, hand in enumerate(hands):
                is_open = (i == 0 and l_open) or (i == 1 and r_open)
                col     = C_LIME if is_open else C_DIM
                _draw_hand(cam_panel, hand, fw, fh, r_cam, ox, oy, col, "L" if i == 0 else "R")

            cv2.putText(cam_panel, "STATE: IDLE", (16, 36), FONT, 0.70, C_CYAN, 2, cv2.LINE_AA)
            cv2.putText(cam_panel, f"HOLD BOTH PALMS: {held_time:.1f}s / {ACTIVATION_TIME_SEC:.1f}s",
                        (16, 70), FONT, 0.55, C_CYAN, 1, cv2.LINE_AA)
            mode = "IDLE"

        # ── State: ACTIVE ─────────────────────────────────────────────────────
        else:
            if n == 0:
                mode       = "HOLD"
                prev_ix    = None
                prev_iy    = None
                prev_dist  = None
                prev_angle = None
                prev_mid_x = None
                prev_mid_y = None

            elif n == 1:
                mode = "TRANSLATE (1 HAND)"
                hand = hands[0]

                # Denormalized pixel coordinates in camera frame
                ix = hand[LM_INDEX_TIP].x * fw
                iy = hand[LM_INDEX_TIP].y * fh

                if prev_ix is not None:
                    s_tx = _ema(s_tx, s_tx + (ix - prev_ix))
                    s_ty = _ema(s_ty, s_ty + (iy - prev_iy))

                prev_ix    = ix
                prev_iy    = iy
                prev_dist  = None
                prev_angle = None
                prev_mid_x = None
                prev_mid_y = None

                _draw_hand(cam_panel, hand, fw, fh, r_cam, ox, oy, C_CYAN, "L")

            else:
                mode  = "BIMANUAL (2 HANDS)"
                lhand = hands[0]
                rhand = hands[1]

                # Denormalized pixel coordinates
                lx, ly = lhand[LM_INDEX_TIP].x * fw, lhand[LM_INDEX_TIP].y * fh
                rx, ry = rhand[LM_INDEX_TIP].x * fw, rhand[LM_INDEX_TIP].y * fh

                # Scale: Euclidean distance between index fingertips
                dist = math.hypot(rx - lx, ry - ly)

                # Rotation: Angle between index fingertips
                raw_angle_deg = math.degrees(math.atan2(ry - ly, rx - lx))

                # Translation anchor: Midpoint between index fingertips
                mid_x = (lx + rx) * 0.5
                mid_y = (ly + ry) * 0.5

                if prev_dist is not None:
                    # Zoom / Scale
                    if prev_dist > 1.0:
                        ratio   = dist / prev_dist
                        s_scale = _ema(s_scale, max(0.05, min(s_scale * ratio, 20.0)))

                    # 2D Rotation (wrap-around safe to prevent flips)
                    d_angle = raw_angle_deg - prev_angle  # type: ignore[operator]
                    d_angle = (d_angle + 180.0) % 360.0 - 180.0
                    s_angle = _ema(s_angle, s_angle + d_angle)

                    # Midpoint translation
                    s_tx = _ema(s_tx, s_tx + (mid_x - prev_mid_x))  # type: ignore[operator]
                    s_ty = _ema(s_ty, s_ty + (mid_y - prev_mid_y))  # type: ignore[operator]

                prev_dist  = dist
                prev_angle = raw_angle_deg
                prev_mid_x = mid_x
                prev_mid_y = mid_y
                prev_ix    = None
                prev_iy    = None

                lip = _draw_hand(cam_panel, lhand, fw, fh, r_cam, ox, oy, C_LIME,  "L")
                rip = _draw_hand(cam_panel, rhand, fw, fh, r_cam, ox, oy, C_AMBER, "R")
                cv2.line(cam_panel, lip, rip, (60, 60, 60), 1, cv2.LINE_AA)
                cv2.circle(cam_panel, ((lip[0] + rip[0]) // 2, (lip[1] + rip[1]) // 2), 5, (180, 180, 180), -1)

            # Telemetry display
            cv2.putText(cam_panel, "STATE: ACTIVE", (16, 36), FONT, 0.70, C_LIME, 2, cv2.LINE_AA)
            rows = [
                f"MODE  : {mode}",
                f"TX    : {s_tx:+.1f} px",
                f"TY    : {s_ty:+.1f} px",
                f"SCALE : {s_scale:.3f}x",
                f"ANGLE : {s_angle:+.1f} deg",
                f"HANDS : {n}",
            ]
            for idx_row, row_text in enumerate(rows):
                cv2.putText(cam_panel, row_text, (16, 68 + idx_row * 22), FONT, 0.50, C_LIME, 1, cv2.LINE_AA)

        # ── Overlay Flash Messages ────────────────────────────────────────────
        if flash_end and now < flash_end:
            cv2.putText(cam_panel, flash_msg, (PANEL_W // 2 - 170, CANVAS_H // 2),
                        FONT, 0.85, (50, 255, 100), 2, cv2.LINE_AA)

        # ── Render Scan via cv2.warpAffine ────────────────────────────────────
        scan_panel = _apply_transform(base_scan, s_tx, s_ty, s_scale, s_angle)

        # ── Pure Black Composite Canvas ───────────────────────────────────────
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        canvas[:, :PANEL_W]          = cam_panel
        canvas[:, PANEL_W:PANEL_W*2] = scan_panel
        cv2.line(canvas, (PANEL_W, 0), (PANEL_W, CANVAS_H), (35, 35, 35), 1)

        cv2.putText(canvas, "CAMERA FEED", (16, CANVAS_H - 12), FONT, 0.40, C_DIM, 1, cv2.LINE_AA)
        cv2.putText(canvas, "SCAN VIEWER", (PANEL_W + 16, CANVAS_H - 12), FONT, 0.40, C_DIM, 1, cv2.LINE_AA)

        cv2.imshow(WIN, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        elif key in (ord("r"), ord("R")):
            s_tx, s_ty, s_scale, s_angle = 0.0, 0.0, 1.0, 0.0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
