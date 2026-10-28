# Integration Plan: WonderSnap → SterileSpace

**Status:** PROPOSAL. No existing application code has been modified. Awaiting approval.
**Source repo:** https://github.com/AkbarSheikh-debug/wondersnap (MIT licence)
**Target workspace:** `/Users/veerraghuvanshi/Desktop/Veer/SterileSpace`

---

## 1. Key finding: the two projects are different stacks

| | **WonderSnap** | **SterileSpace (yours)** |
|---|---|---|
| Language | JavaScript (ES modules), Node >= 18 | Python 3 (conda env `sterile-space`) |
| Runtime | Browser (Chrome/Edge) + tiny Node static server | Native desktop process |
| Hand tracking | `@mediapipe/tasks-vision` (WASM, GPU delegate), VIDEO mode | `mediapipe` Python Tasks API, IMAGE mode |
| Model file | `models/hand_landmarker.task` (7.8 MB) | `hand_landmarker.task` (7.8 MB). **Same file size** |
| Rendering | Hand-written WebGL2 particle engine (transform feedback, up to 250k particles) | `app.py`: OpenCV 2D affine. `sterile_3d.py`: PyVista/VTK mesh |
| Content | 33 procedural point-cloud models (organs, engines, wonders) | One scan image (`stark_skull.jpg`) / one mesh |
| Gestures | Snap, fist, open, point, pinch, peace, twist, two-hand zoom | Dual-palm 2 s wake-word, 1-hand pan, 2-hand scale/rotate |
| Threading | Single render loop (`poll()` per frame) | `app.py` single loop; `sterile_3d.py` `CVThread` + PyVista main thread |
| Tests | Playwright (7 spec files), CI workflow | None |

**Consequence:** the WonderSnap code cannot be dropped in. The JS, WebGL2 and Node server don't run in your Python processes. What *can* be reused is its **logic and algorithms**, ported to Python, and its **data** (model definitions, part labels). That shapes the whole plan.

> [!IMPORTANT]
> Both projects use the same MediaPipe `HandLandmarker` model file. **No new ML model is needed.** WonderSnap's "models" are procedural geometry definitions (`src/models/*.js`), not neural networks.

---

## 2. WonderSnap architecture (what I read)

```
index.html / styles.css / server.mjs        UI shell + zero-dependency static server
src/app.js (52 KB)                          Main loop, UI, labels, quiz, cut-away, etc.
src/hands.js                                Camera + HandLandmarker wrapper, poll() per frame, mirrors x
src/logic/gestures.js                       classify(), openness(), pinched(), PoseDebouncer, SnapDetector
src/logic/controller.js                     Landmarks -> pose/roll/openness/pinch/two-hand state
src/logic/state.js                          WonderState: IDLE -> SPHERE -> FORMED -> DISSOLVE
src/logic/simRef.js, synth.js               CPU reference physics, audio
src/gl/renderer.js, shaders.js              WebGL2 particle renderer
src/lib/{sampling,shapes,vec,rng}.js        Point-cloud sampling from primitives, seeded RNG
src/models/*.js                             anatomy, anatomy2, body, biology, engines, machines, vehicles, wonders; catalog.js
src/features.js                             Voice commands (Web Speech), TTS, recording, command parser
tests/*.spec.js                             Playwright tests (logic, models, gpu, camera, features)
```

### Design points worth borrowing
1. **Pure-logic separation.** `gestures.js` / `controller.js` / `state.js` take plain landmark arrays and have no I/O, so they are unit-testable with synthetic hands.
2. **Rotation-invariant pose classifier.** A finger counts as extended when `dist(tip, wrist) > dist(pip, wrist) * 1.15`. Your current `_is_open_palm` compares tip.y < pip.y, which breaks if the hand is tilted or rotated.
3. **Continuous `openness()` (0 = fist, 1 = open)** from fingertip/MCP ratios. Useful as an analogue control, such as an exploded-view slider.
4. **`PoseDebouncer`.** A pose must persist for N frames (4, or 6 for peace) before it becomes the stable pose. This addresses the flicker your current code handles only with sorting and EMA.
5. **Hand twist from wrist→middle-MCP**, `atan2(dx*aspect, -dy)`, which is unaffected by opening or closing the fingers. Your `sterile_3d.py` already has a similar `_wrist_mcp_angle`.
6. **Primary-hand tracking by nearest previous wrist.** This is better than raw wrist-X sorting when hands cross, though your X-sort is a deliberate requirement (see Risk R2).
7. **Pinch + two-hand zoom** with debounce (3 frames) and a rising-edge latch.
8. **State machine as a small class with an event log**, instead of booleans spread through `main()`.
9. **Gesture-sensitivity profiles** (`STANDARD` / `FORGIVING`): frozen threshold dicts selectable by name.

---

## 3. Compatibility analysis (your workspace)

| File | Role | Touch it? |
|---|---|---|
| `app.py` (383 lines) | 2D OpenCV viewer; probes camera [1,0,2]; WINDOW_AUTOSIZE; EMA 0.15; dual-palm 2 s unlock | **No.** Stays the stable baseline |
| `sterile_3d.py` (593 lines) | PyVista 3D viewer, `CVThread` + queue, `State(IntEnum)` | **No** (optionally a later, separate opt-in phase) |
| `clinical_viewer.py` (476 lines) | Earlier 2D viewer | **No** |
| `requirements.txt` | `mediapipe`, `opencv-python`, `numpy` | Additive only, behind an extras file (see 5.4) |
| `run.sh` | Launcher: `app.py` default, `3d` → `sterile_3d.py` | Additive only: new subcommand, existing behaviour unchanged |

---

## 4. Integration strategy: additive, opt-in, zero edits to existing files

**Principle:** add new files beside the existing ones. Nothing existing imports the new code unless you later opt in. If anything goes wrong, you delete the new files and the project is exactly as it is today.

### Proposed new layout
```
SterileSpace/
├── app.py                      (unchanged)
├── sterile_3d.py               (unchanged)
├── clinical_viewer.py          (unchanged)
├── integration-plan.md         (this file)
└── wondersnap_port/            NEW (package)
    ├── __init__.py
    ├── gestures.py             Port of gestures.js: classify, openness, pinched, PoseDebouncer, SnapDetector
    ├── controller.py           Port of controller.js (landmarks -> pose / roll / openness / pinch / 2-hand)
    ├── state.py                Port of state.js (generic state machine, enum + event log)
    ├── catalog.py              Model registry (see 4.3)
    ├── models/                 Optional ported geometry (see 4.3)
    └── tests/                  pytest, synthetic-landmark tests (mirrors tests/logic.spec.js)
```
A new entry point, `app_wonder.py`, would be the only file wiring the pieces together. It is separate from `app.py`, so your working 2D viewer is never at risk.

### 4.1 Phase 1: Gesture logic port (low risk, highest value)
- Port `gestures.js` → `gestures.py` **using the same numeric thresholds** (`extendedFingerRatio 1.15`, `curledFingerRatio 1.25`, openness ranges 1.05–1.75, etc.) so behaviour matches upstream.
- Port `PoseDebouncer`, `SnapDetector`, `pinched`, `openness`, `rollOf`.
- Landmarks in your code are MediaPipe objects with `.x/.y`; the port takes a thin adapter `lm_to_xy(hand) -> [(x, y), ...]`. Upstream mirrors x (`1 - x`); your frames are already `cv2.flip`-ed, so the adapter must **not** double-mirror.
- Remember upstream landmarks are normalised x and y separately; pass the frame aspect ratio (`fw / fh`) into the roll calculation as upstream does.
- Deliverable: pure functions plus pytest tests with synthetic hands. No camera, no GUI.

### 4.2 Phase 2: State machine port
- `state.py`: `WonderState` as a Python class with `IntEnum` states, matching the `State(IntEnum)` style already in `sterile_3d.py`.
- Map your existing wake-word flow onto it (IDLE → ACTIVE) so a future viewer reuses one machine rather than ad-hoc booleans.
- Do not replace the dual-palm 2 s unlock. It would be kept as the gate in front of the new machine.

### 4.3 Phase 3: Models and content (choose one, needs your decision)
WonderSnap's models are **procedural point clouds with named, explodable parts** (Heart, Brain, Skull, Skeleton, Lungs, Kidney, Eye, Ear, Tooth, plus engines, vehicles, wonders).

| Option | Description | Effort | Fit for a "sterile/clinical" tool |
|---|---|---|---|
| **A. Anatomy-only subset** | Port only `anatomy.js`, `anatomy2.js`, `body.js` (about 48 KB of JS) to Python point-cloud generators and render in PyVista as `PolyData` points | Medium-high | **Best.** Medical content, reuses your PyVista path |
| **B. Full catalog** | Port all 33 models (about 150 KB JS) | High | Engines/wonders are off-topic for a clinical tool |
| **C. Labels/metadata only** | Port only part names and descriptions, to annotate your existing skull image or mesh | Low | Good lightweight win |
| **D. Skip models** | Gesture/state logic only | None | Minimal, fully safe |

**Recommendation: Phase 1 + 2, then Option C or A.** Skip B.

Rendering note: WonderSnap renders 200–250k particles with GPU transform feedback. PyVista/VTK has no equivalent. Expect roughly 20–50k points at interactive rates, so the particle-physics effects (swirl, burst, morph) would be simplified or dropped. The geometry carries over; the visual style will not match.

### 4.4 Phase 4: Optional features (each independent, each opt-in)
| WonderSnap feature | Python equivalent | Notes |
|---|---|---|
| Voice commands (Web Speech) | `speech_recognition` or `vosk` | New dependency. Voice is hands-free, which is relevant for a sterile setting, but it needs a mic and offline-model decisions |
| Read-aloud (TTS) | `pyttsx3` / macOS `say` | `say` needs no dependency on this Mac |
| Session recording | `cv2.VideoWriter` | Already available in OpenCV |
| Cut-away cross-section | `pv.Plotter.add_mesh_clip_plane` | Natural fit for a PyVista mesh |
| Quiz mode | Pure Python state | Probably out of scope for a clinical tool |
| Snap gesture | `SnapDetector` port | Needs about 30 fps and a good view of the hand; it may be unreliable at your webcam's frame rate |

### 4.5 Dependencies
Do **not** edit `requirements.txt` in place. Add a separate `requirements-wonder.txt`:
```
-r requirements.txt
pytest>=8.0           # Phase 1-2 tests
# Phase 3 (only if Option A/B): pyvista already used by sterile_3d.py
# Phase 4 (only if approved): vosk / SpeechRecognition, pyttsx3
```
Everything in Phases 1–2 needs **no new runtime dependency** beyond `numpy` (already present).

**Not adopted:** the Node server, `@mediapipe/tasks-vision`, Playwright, `package.json`. They apply to the browser build only. A browser build of SterileSpace would be a different project and is not recommended unless you want a web version.

### 4.6 Launcher
Add one new subcommand to `run.sh`, for example `bash run.sh wonder`. The existing `3d` and default branches are not changed. This is the only edit to an existing file, and it is optional and additive. Say so at approval time if you'd rather leave `run.sh` alone and use a direct `python app_wonder.py`.

---

## 5. Risks and mitigations

| # | Risk | Mitigation |
|---|---|---|
| R1 | **Threshold drift.** Python landmarks and the JS port can differ slightly (aspect handling, mirroring) | Port constants verbatim; replicate upstream `logic.spec.js` cases as pytest; verify with recorded landmark fixtures |
| R2 | **Hand-assignment conflict.** Upstream picks the primary hand by nearest previous wrist; your requirement is strict wrist-X sort for Left/Right | Keep your X-sort for L/R assignment. Use upstream's nearest-wrist logic *only* to choose which hand drives pose gestures. Document the split |
| R3 | **Double mirroring** of x | Adapter takes already-flipped coordinates; unit-test it |
| R4 | **macOS threading.** GUI calls must stay on the main thread (this is what broke `sterile_3d.py` earlier) | New code is pure logic; `app_wonder.py` stays single-threaded or follows the `CVThread` headless + queue pattern |
| R5 | **Camera flakiness** (Continuity Camera hijack) | Reuse the exact `[1, 0, 2]` probe-read loop from `app.py`; no hard-coded index |
| R6 | **Performance.** Per-frame classification is cheap, but point-cloud models are not | Cap particle count; benchmark before Phase 3 |
| R7 | **Scope creep.** The upstream is a consumer "wow" demo, while yours is a utilitarian clinical interface (your "Zero AI Slop" rule) | Default to Phases 1–2 and Option C/A; skip the quiz, wonders and engines |
| R8 | **Licence.** MIT requires keeping the copyright notice | Include upstream `LICENSE` text and attribution in `wondersnap_port/` |
| R9 | **Regression of existing behaviour** | No existing file is edited (except the optional `run.sh` line); verify by `diff` and by running `app.py` unchanged |

---

## 6. Verification plan
1. `git`-style check: confirm `app.py`, `sterile_3d.py`, `clinical_viewer.py` are byte-identical before and after (checksums recorded first).
2. `pytest wondersnap_port/tests` passes with synthetic hands (fist, open, peace, point, pinch, tilted palm).
3. Smoke-run `python app.py` and `bash run.sh 3d` unchanged.
4. Manual check of `app_wonder.py` with the camera: debounced poses are stable and there is no L/R flicker.

---

## 7. Decisions needed from you

1. **Scope:** Phases 1–2 only (gesture and state logic), or also Phase 3 (models) and Phase 4 (voice, TTS, recording, cut-away)?
2. **Models (if Phase 3):** Option A (anatomy subset), B (full), C (labels only), or D (skip)?
3. **Host app:** a new standalone `app_wonder.py` (recommended, zero risk), or later wire the new gesture logic into `sterile_3d.py`?
4. **`run.sh`:** may I add a `wonder` subcommand, or leave it untouched?
5. **Dependencies:** OK to add `requirements-wonder.txt` (pytest, and optionally voice/TTS packages)?

**No files other than this document have been created or changed. Reply with your choices and I'll start.**
