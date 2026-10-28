"""
clinical_viewer.py — Bimanual Spatial Computing Interface
──────────────────────────────────────────────────────────
Engine    : MediaPipe HandLandmarker IMAGE mode (mediapipe 0.10.35)
Renderer  : OpenCV native — cv2.imshow — zero web frameworks

State Machine  (EMA α = 0.15 on all spatial variables):
  IDLE      (0 hands) — transforms hold; no drift
  PAN       (1 hand)  — LM8 frame-delta  → accumulate tx / ty translation
  TRANSFORM (2 hands) — LM8 × LM8 Euclidean distance → relative scale
                        LM8 × LM8 arctangent angle    → relative rotation
                        LM8 × LM8 midpoint delta      → translation anchor

Controls: Q / ESC = quit  |  R = reset all transforms
"""

import math
import sys
import urllib.request
from enum import IntEnum
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ─────────────────────────────────────────────────────────────────────────────
# Model bootstrap
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = Path("hand_landmarker.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

def ensure_model() -> None:
    if not MODEL_PATH.exists():
        print("[INFO] Downloading hand_landmarker.task (~8 MB) …", flush=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[INFO] Model saved → {MODEL_PATH}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────
CANVAS_W = 1400
CANVAS_H = 700
PANEL_W  = CANVAS_W // 2    # 700 — camera panel AND scan panel are each 700 × 700

# ─────────────────────────────────────────────────────────────────────────────
# Kinematics
# ─────────────────────────────────────────────────────────────────────────────
EMA_ALPHA    = 0.15          # low-pass filter coefficient
LM_INDEX_TIP = 8             # MediaPipe landmark: index finger tip

# ─────────────────────────────────────────────────────────────────────────────
# State machine
# ─────────────────────────────────────────────────────────────────────────────
class State(IntEnum):
    IDLE      = 0
    PAN       = 1
    TRANSFORM = 2

STATE_LABEL = {State.IDLE: "IDLE", State.PAN: "PAN", State.TRANSFORM: "TRANSFORM"}

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette (BGR)
# ─────────────────────────────────────────────────────────────────────────────
C_BG       = (  4,   4,   4)
C_DIVIDER  = ( 32,  32,  32)
C_TEXT     = (210, 210, 210)
C_LABEL    = (105, 135, 165)
C_LM8_H0   = (  0, 200, 255)   # cyan  — hand 0 index tip
C_LM8_H1   = ( 60, 230,  60)   # green — hand 1 index tip
C_AXIS     = (  0, 160, 255)   # transform axis line
C_MID      = (  0, 210, 255)   # midpoint marker
C_PAN_TIP  = (110,  70, 255)   # violet — 1-hand index tip
C_SKELETON = ( 38,  38,  38)
C_JOINT    = ( 62,  62,  62)
C_ACTIVE   = ( 35, 195,  70)
C_IDLE_CLR = ( 80,  80,  80)

FONT = cv2.FONT_HERSHEY_DUPLEX

# ─────────────────────────────────────────────────────────────────────────────
# Hand skeleton (21-landmark topology)
# ─────────────────────────────────────────────────────────────────────────────
SKELETON = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


# ─────────────────────────────────────────────────────────────────────────────
# Signal processing
# ─────────────────────────────────────────────────────────────────────────────
def smooth(cur: float, tgt: float, a: float) -> float:
    """Scalar EMA: converges cur → tgt at rate a."""
    return cur + a * (tgt - cur)


def smooth_angle(cur: float, tgt: float, a: float) -> float:
    """EMA for angles — resolves ±180° wrap-around discontinuity."""
    delta = ((tgt - cur + 180.0) % 360.0) - 180.0
    return cur + a * delta


# ─────────────────────────────────────────────────────────────────────────────
# Image utilities
# ─────────────────────────────────────────────────────────────────────────────
def letterbox(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Fit img into a w×h black canvas, preserving aspect ratio."""
    ih, iw = img.shape[:2]
    r       = min(w / iw, h / ih)
    nw, nh  = int(iw * r), int(ih * r)
    out     = np.zeros((h, w, 3), dtype=np.uint8)
    y0, x0  = (h - nh) // 2, (w - nw) // 2
    out[y0:y0+nh, x0:x0+nw] = cv2.resize(img, (nw, nh),
                                           interpolation=cv2.INTER_LINEAR)
    return out


def apply_transform(base: np.ndarray,
                    scale: float, angle: float,
                    tx: float,    ty: float,
                    w: int,       h: int) -> np.ndarray:
    """
    Affine transform: rotate + scale around panel centre, then translate.
    Returns a w × h BGR image.
    """
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(base, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0))


# ─────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ─────────────────────────────────────────────────────────────────────────────
def lm_to_panel(lm_list, cam_w: int, cam_h: int,
                disp_w: int, disp_h: int,
                offs_x: int, offs_y: int,
                panel_w: int, panel_h: int) -> list:
    """
    Convert MediaPipe normalised landmarks → pixel coords in the letterboxed
    camera panel.  Returns list of (px, py) tuples.
    """
    return [
        (
            int(lm.x * disp_w + offs_x),
            int(lm.y * disp_h + offs_y),
        )
        for lm in lm_list
    ]


def draw_hand(panel: np.ndarray, pts: list, lm8_color: tuple) -> tuple:
    """
    Draw dim skeleton + highlighted LM8 on panel.
    Returns the LM8 pixel position (px, py).
    """
    for a, b in SKELETON:
        cv2.line(panel, pts[a], pts[b], C_SKELETON, 1, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(panel, pt, 2, C_JOINT, -1, cv2.LINE_AA)

    ip = pts[LM_INDEX_TIP]
    cv2.circle(panel, ip, 10, lm8_color, -1, cv2.LINE_AA)
    cv2.circle(panel, ip, 13, lm8_color,  1, cv2.LINE_AA)
    cv2.putText(panel, "LM8", (ip[0] + 14, ip[1] - 6),
                FONT, 0.32, lm8_color, 1, cv2.LINE_AA)
    return ip


def draw_transform_axis(panel: np.ndarray,
                        ip0: tuple, ip1: tuple) -> None:
    """Draw axis line + midpoint marker between the two LM8 positions."""
    mid = ((ip0[0] + ip1[0]) // 2, (ip0[1] + ip1[1]) // 2)
    cv2.line(panel, ip0, ip1, C_AXIS, 1, cv2.LINE_AA)
    cv2.circle(panel, mid, 6, C_MID, -1, cv2.LINE_AA)
    cv2.circle(panel, mid, 9, C_MID,  1, cv2.LINE_AA)


def draw_hud(canvas: np.ndarray,
             state: State,
             tx: float, ty: float,
             scale: float, angle: float,
             dist_px: float) -> None:
    """
    Kinematic telemetry chip — bottom-left of camera panel.
    Only real-time values computed from hand landmarks are displayed.
    """
    bx, by = 14, CANVAS_H - 196
    row_h  = 28
    chip_w = 370

    state_color = C_ACTIVE if state != State.IDLE else C_IDLE_CLR
    dist_str    = f"{dist_px:6.1f} px" if dist_px > 0 else "      — px"

    rows = [
        ("STATE",        STATE_LABEL[state],   state_color),
        ("SCALE",        f"{scale:6.3f} x",     C_TEXT),
        ("ROTATION",     f"{angle:+7.1f} DEG",  C_TEXT),
        ("PAN X",        f"{tx:+8.1f} px",       C_TEXT),
        ("PAN Y",        f"{ty:+8.1f} px",       C_TEXT),
        ("DIST LM8-LM8", dist_str,               C_TEXT),
    ]

    chip_h = len(rows) * row_h + 18
    cv2.rectangle(canvas,
                  (bx - 8,  by - 22),
                  (bx + chip_w, by + chip_h - 22),
                  (6, 6, 6), -1)
    cv2.rectangle(canvas,
                  (bx - 8,  by - 22),
                  (bx + chip_w, by + chip_h - 22),
                  C_DIVIDER, 1)

    for i, (label, value, color) in enumerate(rows):
        y = by + i * row_h
        cv2.putText(canvas, label,       (bx,       y), FONT, 0.37, C_LABEL, 1, cv2.LINE_AA)
        cv2.putText(canvas, value,       (bx + 200, y), FONT, 0.42, color,   1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:

    ensure_model()

    # ── Scan image ─────────────────────────────────────────────────────────────
    scan_path = Path("mock_scan.png")
    if not scan_path.exists():
        sys.exit(f"[ABORT] {scan_path} not found — place it next to clinical_viewer.py")
    raw_scan = cv2.imread(str(scan_path))
    if raw_scan is None:
        sys.exit(f"[ABORT] cv2.imread failed on {scan_path}")
    base_scan = letterbox(raw_scan, PANEL_W, CANVAS_H)   # pre-letterboxed once

    # ── Camera ─────────────────────────────────────────────────────────────────
    cap = None
    for idx in range(3):
        _c = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if _c.isOpened():
            cap = _c
            print(f"[INFO] Camera opened (index {idx})", flush=True)
            break
        _c.release()
    if cap is None:
        print(
            "[ABORT] No camera found.\n"
            "  Fix: System Settings > Privacy & Security > Camera\n"
            "       Toggle ON Terminal.app / iTerm2",
            file=sys.stderr,
        )
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # ── MediaPipe HandLandmarker — IMAGE mode, 2 hands ────────────────────────
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,                              # bimanual
        min_hand_detection_confidence=0.70,
        min_hand_presence_confidence=0.55,
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)

    # ── Spatial state (EMA-smoothed) ──────────────────────────────────────────
    ema_tx    = 0.0    # translation X (px)
    ema_ty    = 0.0    # translation Y (px)
    ema_scale = 1.0    # zoom factor
    ema_angle = 0.0    # rotation (degrees)

    # Raw / target values that EMA converges toward
    target_tx    = 0.0
    target_ty    = 0.0
    target_scale = 1.0
    target_angle = 0.0

    # State machine
    state = State.IDLE

    # PAN state — previous LM8 position in normalised coords
    prev_lm8_norm: tuple | None = None

    # TRANSFORM state — reference values captured on entry
    ref_dist:      float | None = None
    ref_raw_angle: float | None = None
    ref_mid:       tuple | None = None   # (px, py) in panel pixels
    ref_ema_scale  = 1.0
    ref_ema_angle  = 0.0
    ref_ema_tx     = 0.0
    ref_ema_ty     = 0.0

    # Telemetry
    dist_display = 0.0

    # ── Window ─────────────────────────────────────────────────────────────────
    cv2.namedWindow("CLINICAL VIEWER", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CLINICAL VIEWER", CANVAS_W, CANVAS_H)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame  = cv2.flip(frame, 1)
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_img)

            n_hands  = len(result.hand_landmarks)

            # ── Letterbox geometry for landmark projection ────────────────────
            cam_h, cam_w = frame.shape[:2]
            r       = min(PANEL_W / cam_w, CANVAS_H / cam_h)
            disp_w  = int(cam_w * r)
            disp_h  = int(cam_h * r)
            offs_x  = (PANEL_W  - disp_w) // 2
            offs_y  = (CANVAS_H - disp_h) // 2

            # ── State machine ─────────────────────────────────────────────────
            new_state = (State.IDLE      if n_hands == 0 else
                         State.PAN       if n_hands == 1 else
                         State.TRANSFORM)

            # ── PAN (1 hand) ──────────────────────────────────────────────────
            if new_state == State.PAN:
                lm   = result.hand_landmarks[0]
                lm8x = lm[LM_INDEX_TIP].x
                lm8y = lm[LM_INDEX_TIP].y

                if state == State.PAN and prev_lm8_norm is not None:
                    # Frame-delta panning — accumulate into targets
                    target_tx += (lm8x - prev_lm8_norm[0]) * PANEL_W
                    target_ty += (lm8y - prev_lm8_norm[1]) * CANVAS_H

                prev_lm8_norm = (lm8x, lm8y)

                # Scale and rotation held at their current EMA (no change)
                target_scale = ema_scale
                target_angle = ema_angle
                dist_display = 0.0

            # ── TRANSFORM (2 hands) ───────────────────────────────────────────
            elif new_state == State.TRANSFORM:
                lm0  = result.hand_landmarks[0]
                lm1  = result.hand_landmarks[1]

                # LM8 positions in panel pixel space
                x0 = lm0[LM_INDEX_TIP].x * disp_w + offs_x
                y0 = lm0[LM_INDEX_TIP].y * disp_h + offs_y
                x1 = lm1[LM_INDEX_TIP].x * disp_w + offs_x
                y1 = lm1[LM_INDEX_TIP].y * disp_h + offs_y

                dx         = x1 - x0
                dy         = y1 - y0
                curr_dist  = math.hypot(dx, dy)
                curr_angle = math.degrees(math.atan2(dy, dx))
                curr_mid   = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

                dist_display = curr_dist

                # Capture reference on first TRANSFORM frame
                if state != State.TRANSFORM:
                    ref_dist      = max(curr_dist, 1.0)
                    ref_raw_angle = curr_angle
                    ref_mid       = curr_mid
                    ref_ema_scale = ema_scale
                    ref_ema_angle = ema_angle
                    ref_ema_tx    = ema_tx
                    ref_ema_ty    = ema_ty

                # Relative scale from entry distance
                target_scale = ref_ema_scale * (curr_dist / ref_dist)

                # Relative rotation from entry angle (wrap-safe)
                angle_delta  = ((curr_angle - ref_raw_angle + 180) % 360) - 180
                target_angle = ref_ema_angle + angle_delta

                # Translation locked to midpoint displacement from entry
                target_tx = ref_ema_tx + (curr_mid[0] - ref_mid[0])
                target_ty = ref_ema_ty + (curr_mid[1] - ref_mid[1])

                prev_lm8_norm = None

            # ── IDLE (0 hands) ────────────────────────────────────────────────
            else:
                prev_lm8_norm = None
                dist_display  = 0.0
                # Hold targets at current EMA — no drift
                target_tx    = ema_tx
                target_ty    = ema_ty
                target_scale = ema_scale
                target_angle = ema_angle

            state = new_state

            # ── EMA low-pass filter (α = 0.15) ────────────────────────────────
            ema_tx    = smooth      (ema_tx,    target_tx,    EMA_ALPHA)
            ema_ty    = smooth      (ema_ty,    target_ty,    EMA_ALPHA)
            ema_scale = smooth      (ema_scale, target_scale, EMA_ALPHA)
            ema_angle = smooth_angle(ema_angle, target_angle, EMA_ALPHA)

            # ── Compose canvas ────────────────────────────────────────────────
            canvas    = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
            canvas[:] = C_BG

            # Left panel — letterboxed camera + landmark overlay
            cam_panel = letterbox(frame, PANEL_W, CANVAS_H)

            if n_hands >= 1:
                pts0 = lm_to_panel(result.hand_landmarks[0],
                                   cam_w, cam_h, disp_w, disp_h,
                                   offs_x, offs_y, PANEL_W, CANVAS_H)

                if state == State.TRANSFORM and n_hands == 2:
                    pts1 = lm_to_panel(result.hand_landmarks[1],
                                       cam_w, cam_h, disp_w, disp_h,
                                       offs_x, offs_y, PANEL_W, CANVAS_H)
                    ip0 = draw_hand(cam_panel, pts0, C_LM8_H0)
                    ip1 = draw_hand(cam_panel, pts1, C_LM8_H1)
                    draw_transform_axis(cam_panel, ip0, ip1)
                else:
                    draw_hand(cam_panel, pts0, C_PAN_TIP)

            canvas[:, :PANEL_W] = cam_panel

            # Right panel — scan with affine transform applied
            canvas[:, PANEL_W:] = apply_transform(
                base_scan, ema_scale, ema_angle, ema_tx, ema_ty,
                PANEL_W, CANVAS_H,
            )

            # Vertical divider
            cv2.line(canvas, (PANEL_W, 0), (PANEL_W, CANVAS_H), C_DIVIDER, 1)

            # Kinematic telemetry HUD
            draw_hud(canvas, state, ema_tx, ema_ty, ema_scale, ema_angle, dist_display)

            cv2.imshow("CLINICAL VIEWER", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):        # Q or ESC — quit
                break
            elif key == ord('r'):            # R — reset all transforms
                ema_tx = ema_ty = target_tx = target_ty = 0.0
                ema_scale = target_scale = 1.0
                ema_angle = target_angle = 0.0
                ref_dist = ref_raw_angle = ref_mid = None
                prev_lm8_norm = None

    except KeyboardInterrupt:
        pass
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
