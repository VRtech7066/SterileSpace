"""
sterile_3d.py — Bimanual Gesture-Controlled Volumetric Medical Viewer
======================================================================
NOTE ON IMPORTS:
  This environment runs MediaPipe 0.10.35 (ARM64 native wheel).
  This build ships ONLY the Tasks API.
  'mediapipe.python.solutions' and 'mediapipe.solutions' do not exist;
  importing them raises ModuleNotFoundError.
  The Tasks API (mediapipe.tasks.python.vision.HandLandmarker) is the
  correct and only entry-point on this platform.

Threading model:
  CVThread (daemon) — cv2.VideoCapture + MediaPipe Tasks, EMA smoothing,
                      2-second dual-palm wake-word, spatial X-sorting.
                      Headless: zero cv2.imshow / cv2.waitKey calls.
                      Pushes telemetry dict to a thread-safe queue.Queue.
  Main thread       — PyVista Plotter (Stanford Bunny placeholder),
                      polls telemetry queue every frame, updates camera.

Activation gesture:
  SCAN_LEFT → SCAN_RIGHT → IDLE (dual-palm 2 s) → ACTIVE

Gesture map (ACTIVE):
  1 hand  → wrist→MCP yaw + index-tip pan
  2 hands → midpoint pan + wrist→MCP pitch + bimanual wrist-distance zoom
"""

from __future__ import annotations

import math
import queue
import sys
import threading
import time
import urllib.request
from enum import IntEnum
from pathlib import Path

import cv2
import numpy as np

# ── MediaPipe (Tasks API — only API available on ARM64 MediaPipe ≥ 0.10) ────
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

# ── PyVista ───────────────────────────────────────────────────────────────────
try:
    import pyvista as pv
    from pyvista import examples as pv_examples
    PYVISTA_OK = True
except ImportError:
    PYVISTA_OK = False
    print("[FATAL] pyvista not installed.  Run: pip install pyvista")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
EMA_ALPHA           = 0.05    # aggressive smoothing — eliminates tremor
CALIB_DURATION_SEC  = 2.0
ACTIVATION_TIME_SEC = 2.0

ROTATION_SENSITIVITY = 0.20
PAN_SENSITIVITY      = 0.40
ZOOM_STEP_WORLD      = 15.0
PINCH_SENSITIVITY    = 0.04

FRAME_W = 960
FRAME_H = 540

# MediaPipe 21-point hand landmark indices
LM_WRIST       = 0
LM_INDEX_TIP   = 8
LM_MIDDLE_MCP  = 9
LM_FINGERTIPS  = [8, 12, 16, 20]
LM_PIPS        = [6, 10, 14, 18]

# Skeleton connectivity for raw OpenCV draw (no mp.solutions.drawing_utils)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

BASE_DIR   = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# ─────────────────────────────────────────────────────────────────────────────
# State machine
# ─────────────────────────────────────────────────────────────────────────────
class State(IntEnum):
    SCAN_LEFT  = 0
    SCAN_RIGHT = 1
    IDLE       = 2
    ACTIVE     = 3

STATE_LABEL = {
    State.SCAN_LEFT:  "CALIBRATION: RAISE LEFT HAND",
    State.SCAN_RIGHT: "CALIBRATION: RAISE RIGHT HAND",
    State.IDLE:       "IDLE  —  DUAL PALM 2s TO ACTIVATE",
    State.ACTIVE:     "ACTIVE",
}

# ─────────────────────────────────────────────────────────────────────────────
# Thread-safe queues
# ─────────────────────────────────────────────────────────────────────────────
telemetry_q: queue.Queue = queue.Queue(maxsize=1)
frame_q:     queue.Queue = queue.Queue(maxsize=1)   # annotated frames for imshow

# ─────────────────────────────────────────────────────────────────────────────
# Pure-math utilities
# ─────────────────────────────────────────────────────────────────────────────
def _ema(prev: float, new: float) -> float:
    return EMA_ALPHA * new + (1.0 - EMA_ALPHA) * prev


def _fxy(lm, w: int, h: int) -> tuple[float, float]:
    return lm.x * w, lm.y * h


def _px(lm, w: int, h: int) -> tuple[int, int]:
    return int(lm.x * w), int(lm.y * h)


def _is_open_palm(hand: list, w: int, h: int) -> bool:
    """All four fingertips strictly above (lower Y than) their PIP joints."""
    for tip_i, pip_i in zip(LM_FINGERTIPS, LM_PIPS):
        if hand[tip_i].y * h >= hand[pip_i].y * h:
            return False
    return True


def _wrist_mcp_angle(hand: list, w: int, h: int) -> float:
    wx, wy = _fxy(hand[LM_WRIST], w, h)
    mx, my = _fxy(hand[LM_MIDDLE_MCP], w, h)
    return math.degrees(math.atan2(my - wy, mx - wx))


def _download_model() -> None:
    if MODEL_PATH.exists():
        return
    print(f"[INFO] Downloading hand landmarker → {MODEL_PATH}")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("[INFO] Download complete.")


# ─────────────────────────────────────────────────────────────────────────────
# CV Thread  (daemon — headless, no GUI calls)
# ─────────────────────────────────────────────────────────────────────────────
class CVThread(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True, name="CVThread")

        # EMA-smoothed output channels
        self.s_azimuth   = 0.0
        self.s_elevation = 0.0
        self.s_zoom      = 1.0
        self.s_focal_x   = 0.0
        self.s_focal_y   = 0.0

        # Delta anchors
        self._prev_pan_x: float | None = None
        self._prev_pan_y: float | None = None
        self._prev_dist:  float | None = None

        # State machine
        self.state           = State.SCAN_LEFT
        self.calib_start:    float | None = None
        self.flash_end:      float | None = None
        self.flash_msg:      str = ""
        self.left_palm_t:    float | None = None
        self.right_palm_t:   float | None = None

        # Lifecycle
        self.running: bool = True

    # ── Camera init ───────────────────────────────────────────────────────────
    def _open_camera(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(1)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
            ok, _ = cap.read()
            if ok:
                print("[CV] Locked camera index 1.")
                return cap
            cap.release()
        # Fallback hunt
        for idx in [0, 2]:
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
            for _ in range(5):
                ok, _ = cap.read()
                if ok:
                    print(f"[CV] Locked camera index {idx}.")
                    return cap
                time.sleep(0.05)
            cap.release()
        raise RuntimeError("[CV] No readable camera found.")

    # ── MediaPipe detector ────────────────────────────────────────────────────
    def _build_detector(self) -> HandLandmarker:
        _download_model()
        opts = HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        return HandLandmarker.create_from_options(opts)

    # ── Spatial sort: smaller wrist-X → Left (index 0) ───────────────────────
    def _sort(self, hands: list, w: int) -> list:
        return sorted(hands, key=lambda h: h[LM_WRIST].x * w)

    # ── OpenCV skeleton (no mp.solutions.drawing_utils) ───────────────────────
    def _draw_hand(self, canvas: np.ndarray, hand: list,
                   w: int, h: int, color: tuple) -> None:
        pts = [_px(lm, w, h) for lm in hand]
        for a, b in HAND_CONNECTIONS:
            cv2.line(canvas, pts[a], pts[b], color, 2, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(canvas, pt, 4, color, -1, cv2.LINE_AA)

    # ── HUD overlay ───────────────────────────────────────────────────────────
    def _hud(self, canvas: np.ndarray, hands: list,
             w: int, h: int, now: float) -> None:
        colors = [(0, 255, 120), (0, 180, 255)]
        for i, hand in enumerate(hands):
            self._draw_hand(canvas, hand, w, h, colors[i % 2])

        active = self.state == State.ACTIVE
        s_col  = (0, 255, 200) if active else (0, 200, 255)
        cv2.putText(canvas, STATE_LABEL[self.state],
                    (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, s_col, 2, cv2.LINE_AA)

        if self.flash_end and now < self.flash_end:
            cv2.putText(canvas, self.flash_msg,
                        (w // 2 - 200, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 255, 100), 3, cv2.LINE_AA)

        if self.state in (State.SCAN_LEFT, State.SCAN_RIGHT) and self.calib_start:
            elapsed = now - self.calib_start
            pct     = min(elapsed / CALIB_DURATION_SEC, 1.0)
            bx      = w // 2 - 200
            cv2.rectangle(canvas, (bx, h - 60),
                          (bx + int(400 * pct), h - 40), (0, 210, 120), -1)
            cv2.rectangle(canvas, (bx, h - 60), (bx + 400, h - 40), (70, 70, 70), 2)
            cv2.putText(canvas, f"{elapsed:.1f} / {CALIB_DURATION_SEC:.1f}s",
                        (w // 2 - 50, h - 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (160, 160, 160), 1)

        if self.state == State.IDLE:
            lt = (now - self.left_palm_t)  if self.left_palm_t  else 0.0
            rt = (now - self.right_palm_t) if self.right_palm_t else 0.0
            cv2.putText(canvas, f"L-PALM: {lt:.1f}s   |   R-PALM: {rt:.1f}s",
                        (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                        (0, 200, 255), 2, cv2.LINE_AA)

        if self.state == State.ACTIVE:
            for i, ln in enumerate([
                f"AZI   {self.s_azimuth:+.1f}",
                f"ELEV  {self.s_elevation:+.1f}",
                f"ZOOM  {self.s_zoom:.3f}x",
                f"FX    {self.s_focal_x:+.1f}",
                f"FY    {self.s_focal_y:+.1f}",
            ]):
                cv2.putText(canvas, ln, (16, 64 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 255, 160), 1)

    # ── Telemetry push ────────────────────────────────────────────────────────
    def _push(self) -> None:
        pkg = {
            "azimuth":   self.s_azimuth,
            "elevation": self.s_elevation,
            "zoom":      self.s_zoom,
            "focal_x":   self.s_focal_x,
            "focal_y":   self.s_focal_y,
            "active":    self.state == State.ACTIVE,
        }
        try:
            telemetry_q.get_nowait()
        except queue.Empty:
            pass
        telemetry_q.put_nowait(pkg)

    def _flash(self, msg: str, now: float, dur: float = 1.3) -> None:
        self.flash_msg = msg
        self.flash_end = now + dur

    def _reset(self) -> None:
        self._prev_pan_x = None
        self._prev_pan_y = None
        self._prev_dist  = None

    # ── Main run loop (headless — no cv2.imshow / cv2.waitKey) ───────────────
    def run(self) -> None:  # noqa: C901
        cap      = self._open_camera()
        detector = self._build_detector()
        fails    = 0
        MAX_FAIL = 5

        while self.running:
            ok, frame = cap.read()
            if not ok:
                fails += 1
                if fails >= MAX_FAIL:
                    print(f"[CV] {MAX_FAIL} consecutive failures — stopping.")
                    self.running = False
                    break
                time.sleep(0.02)
                continue
            fails = 0

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            now   = time.time()

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_img)
            raw    = result.hand_landmarks
            n      = len(raw)
            hands  = self._sort(raw, w) if n > 0 else []
            canvas = frame.copy()

            # ── State machine ─────────────────────────────────────────────
            if self.state == State.SCAN_LEFT:
                if n == 1:
                    if self.calib_start is None:
                        self.calib_start = now
                    elif now - self.calib_start >= CALIB_DURATION_SEC:
                        self._flash("LEFT HAND VERIFIED", now)
                        self.calib_start = None
                        self.state = State.SCAN_RIGHT
                else:
                    self.calib_start = None

            elif self.state == State.SCAN_RIGHT:
                if n == 1:
                    if self.calib_start is None:
                        self.calib_start = now
                    elif now - self.calib_start >= CALIB_DURATION_SEC:
                        self._flash("RIGHT HAND VERIFIED", now)
                        self.calib_start = None
                        self.state = State.IDLE
                else:
                    self.calib_start = None

            elif self.state == State.IDLE:
                l_open = n >= 1 and _is_open_palm(hands[0], w, h)
                r_open = n >= 2 and _is_open_palm(hands[1], w, h)

                if l_open:
                    if self.left_palm_t is None:
                        self.left_palm_t = now
                else:
                    self.left_palm_t = None

                if r_open:
                    if self.right_palm_t is None:
                        self.right_palm_t = now
                else:
                    self.right_palm_t = None

                lt = (now - self.left_palm_t)  if self.left_palm_t  else 0.0
                rt = (now - self.right_palm_t) if self.right_palm_t else 0.0

                if lt >= ACTIVATION_TIME_SEC and rt >= ACTIVATION_TIME_SEC:
                    self.left_palm_t  = None
                    self.right_palm_t = None
                    self.state        = State.ACTIVE
                    self._flash("INTERFACE UNLOCKED", now, 1.8)
                    self._reset()

            elif self.state == State.ACTIVE:
                if n == 0:
                    self._reset()

                elif n == 1:
                    hand = hands[0]
                    ix, iy = _fxy(hand[LM_INDEX_TIP], w, h)
                    if self._prev_pan_x is not None:
                        dx = (ix - self._prev_pan_x) * PAN_SENSITIVITY
                        dy = (iy - self._prev_pan_y) * PAN_SENSITIVITY  # type: ignore[operator]
                        self.s_focal_x = _ema(self.s_focal_x, self.s_focal_x + dx)
                        self.s_focal_y = _ema(self.s_focal_y, self.s_focal_y + dy)
                    self._prev_pan_x  = ix
                    self._prev_pan_y  = iy
                    self._prev_dist   = None
                    self.s_azimuth    = _ema(self.s_azimuth,
                                            _wrist_mcp_angle(hand, w, h)
                                            * ROTATION_SENSITIVITY)

                else:  # 2 hands
                    h0, h1 = hands[0], hands[1]
                    ix0, iy0 = _fxy(h0[LM_INDEX_TIP], w, h)
                    ix1, iy1 = _fxy(h1[LM_INDEX_TIP], w, h)
                    mid_x = (ix0 + ix1) * 0.5
                    mid_y = (iy0 + iy1) * 0.5

                    if self._prev_pan_x is not None:
                        dx = (mid_x - self._prev_pan_x) * PAN_SENSITIVITY
                        dy = (mid_y - self._prev_pan_y) * PAN_SENSITIVITY  # type: ignore[operator]
                        self.s_focal_x = _ema(self.s_focal_x, self.s_focal_x + dx)
                        self.s_focal_y = _ema(self.s_focal_y, self.s_focal_y + dy)
                    self._prev_pan_x = mid_x
                    self._prev_pan_y = mid_y

                    # Bimanual wrist-distance zoom
                    wx0, wy0 = _fxy(h0[LM_WRIST], w, h)
                    wx1, wy1 = _fxy(h1[LM_WRIST], w, h)
                    dist = math.hypot(wx1 - wx0, wy1 - wy0)
                    if self._prev_dist and self._prev_dist > 1.0:
                        delta = (dist - self._prev_dist) * PINCH_SENSITIVITY
                        self.s_zoom = _ema(self.s_zoom,
                                          max(0.05, min(self.s_zoom + delta, 20.0)))
                    self._prev_dist = dist

                    # Right hand wrist→MCP → elevation
                    self.s_elevation = _ema(
                        self.s_elevation,
                        _wrist_mcp_angle(h1, w, h) * ROTATION_SENSITIVITY
                    )

                self._push()

            # ── Annotate frame + push to frame_q (for main-thread imshow) ──
            self._hud(canvas, hands, w, h, now)
            try:
                frame_q.get_nowait()
            except queue.Empty:
                pass
            frame_q.put_nowait(canvas)

        cap.release()
        print("[CV] Thread exited.")


# ─────────────────────────────────────────────────────────────────────────────
# 3D Mesh — Stanford Bunny (default placeholder)
# ─────────────────────────────────────────────────────────────────────────────
def _build_mesh() -> "pv.PolyData":
    try:
        mesh = pv_examples.download_bunny_coarse()
        print("[3D] Stanford Bunny loaded.")
    except Exception:
        mesh = pv.Sphere(radius=1.0, theta_resolution=64, phi_resolution=64)
        print("[3D] Sphere fallback loaded.")
    return mesh


# ─────────────────────────────────────────────────────────────────────────────
# Main thread
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    if not PYVISTA_OK:
        sys.exit("[FATAL] pip install pyvista")

    # ── Start CV daemon ───────────────────────────────────────────────────────
    cv_thread = CVThread()
    cv_thread.start()
    print("[MAIN] CVThread started.")

    # ── PyVista scene ─────────────────────────────────────────────────────────
    pv.set_plot_theme("dark")
    plotter = pv.Plotter(
        title="SterileSpace — 3D Spatial Viewer",
        window_size=(960, 700),
        off_screen=False,
    )
    mesh = _build_mesh()
    plotter.add_mesh(
        mesh,
        color="#ddc8a8",
        specular=0.9,
        specular_power=50,
        smooth_shading=True,
    )
    plotter.add_light(pv.Light(position=(4,  4, 8),  intensity=0.9))
    plotter.add_light(pv.Light(position=(-3, -2, 4), intensity=0.4, color="#a8d0ff"))
    plotter.set_background("#060810")
    plotter.camera_position = "xy"
    plotter.reset_camera()

    # Capture base camera vectors (used for absolute-angle resets every frame)
    base_pos   = plotter.camera.position
    base_focal = plotter.camera.focal_point
    base_up    = plotter.camera.up

    # Session-accumulated camera angles
    acc_azimuth   = 0.0
    acc_elevation = 0.0
    prev_s_azi    = 0.0
    prev_s_elev   = 0.0

    # OpenCV window on main thread (macOS Cocoa safe)
    WIN = "SterileSpace — Hand Tracking"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 640, 360)

    print("[MAIN] Render loop running. Close the PyVista window or press Q to quit.")
    plotter.show(interactive_update=True, auto_close=False)

    try:
        while not plotter.iren.interactor.GetDone():

            # 1. Poll telemetry
            tele = None
            try:
                tele = telemetry_q.get_nowait()
            except queue.Empty:
                pass

            # 2. Apply camera kinematics
            if tele and tele["active"]:
                cam = plotter.camera

                # Diff EMA signals → per-frame delta → accumulate total
                d_azi  = tele["azimuth"]   - prev_s_azi
                d_elev = tele["elevation"] - prev_s_elev
                prev_s_azi  = tele["azimuth"]
                prev_s_elev = tele["elevation"]

                acc_azimuth   += d_azi
                acc_elevation  = max(-89.0, min(89.0, acc_elevation + d_elev))

                # Reset to base every frame → apply absolute accumulated angles
                cam.position    = base_pos
                cam.focal_point = base_focal
                cam.up          = base_up
                cam.azimuth     = acc_azimuth
                cam.elevation   = acc_elevation

                # Focal-point pan (screen Y inverted)
                cam.focal_point = (
                    base_focal[0] + tele["focal_x"],
                    base_focal[1] - tele["focal_y"],
                    base_focal[2],
                )

                # Zoom
                cam.zoom(max(0.05, tele["zoom"]))

            else:
                # Not active — zero delta refs to prevent jump on re-activation
                prev_s_azi  = 0.0
                prev_s_elev = 0.0

            # 3. Tick PyVista
            plotter.update()

            # 4. Display annotated CV frame (main thread — macOS safe)
            try:
                cv_frame = frame_q.get_nowait()
                cv2.imshow(WIN, cv_frame)
            except queue.Empty:
                pass

            # 5. Key pump
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

            # 6. Exit if CVThread died
            if not cv_thread.running and not cv_thread.is_alive():
                print("[MAIN] CVThread stopped — shutting down.")
                break

    except KeyboardInterrupt:
        pass
    finally:
        cv_thread.running = False
        print("[MAIN] Shutdown.")
        plotter.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
