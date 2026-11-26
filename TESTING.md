# HamsterMirror: teammate testing guide

About 20 minutes start to finish. You need a laptop with a webcam and Python 3.11 or newer.
Works on Windows, macOS and Linux.

---

## 1. Get the code

Unzip `hamstermirror.zip` somewhere easy, like `Downloads`.

Open a terminal **inside the project folder**, the one that contains `app.py`:

```bash
cd path/to/hamstermirror
```

In VS Code, open that same folder with File > Open Folder. If you open the folder
above it, VS Code shows fake red import errors (the code still runs).

---

## 2. Install (5 to 10 minutes, one time)

```bash
python -m venv .venv
```

On macOS use `python3 -m venv .venv` (macOS has no `python` command until the venv is active).

Activate it:

| OS | Command |
|---|---|
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
| Windows (cmd) | `.venv\Scripts\activate.bat` |
| macOS / Linux | `source .venv/bin/activate` |

You should see `(.venv)` at the start of the prompt. Then:

```bash
pip install -r requirements.txt
```

**NVIDIA GPU on Windows or Linux?** The plain install is CPU only and will feel laggy.
Swap in the CUDA build:

```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available())"
```

It should print `True`. Apple Silicon Macs use the GPU automatically, so skip this.

---

## 3. Check the install (2 minutes)

```bash
python -m pytest -q tests
```

✅ Expect `42 passed`. If anything fails, stop and send the output to the group chat.

---

## 4. Download models and build the meme index (2 minutes, one time)

```bash
python scripts/download_models.py
python scripts/build_index.py
```

✅ The first prints five "ok" lines (face, hand, face landmarker, CLIP, ResNet18).
✅ The second ends with `indexed 26 memes`.

---

## 5. Calibrate your resting face (1 minute)

```bash
python scripts/enroll.py --neutral-only
```

A webcam window opens. Sit how you normally would for the demo, relax your face, look
at the camera and press **space**. Hold still for 2 seconds. The window closes and the
terminal prints `neutral saved`.

Redo this whenever the lighting or your seat changes a lot.

---

## 6. Run the demo

```bash
python app.py
```

✅ The terminal should print a line like:

```
[app] detector=mediapipe hands=mediapipe methods=['expr', 'expr+clip', 'clip', ...] memes=26 (6 gesture-tagged)
```

If you see `detector=haar`, `hand tracking off` or `expression matcher unavailable`,
step 4 did not finish. Run it again.

The window shows your webcam on the left and the matched meme on the right. The bar at
the bottom shows the method, FPS, latency and the top 3 memes with scores.

### Keys

| Key | Does |
|---|---|
| `m` | Switch method: expr → expr+clip → clip → baseline (if trained) |
| `s` | Smoothing on/off (off makes it flicker, that is expected) |
| `g` | Hand gestures on/off |
| `t` | CLIP text weight (only matters in clip modes) |
| `space` | Save a screenshot to `results/` |
| `q` | Quit |

---

## 7. Test checklist

Use the default `expr` method. Hold each face for about 1 second. Mark ✅ if the
expected meme shows up (or is in the top 3 on the bottom bar), ❌ if not, and write
down what showed instead.

### Faces

| # | Make this face | Expected meme | Result | What showed instead |
|---|---|---|---|---|
| 1 | Relaxed, blank | tiny blank stare / calm content / deadpan | | |
| 2 | Normal closed-mouth smile | sweet content smile / calm content | | |
| 3 | Huge smile, lots of teeth, eyes wide | forced toothy grin | | |
| 4 | Huge smile, teeth, eyes squinting | trollface grin / laughing hamster | | |
| 5 | Smirk (one side of mouth up), brows down | smug grin | | |
| 6 | Smug closed smile, narrowed eyes | smug grin | | |
| 7 | One eyebrow raised + half smile | smug grin | | |
| 8 | Frown, brows down, squint | deeply unimpressed | | |
| 9 | Jaw dropped, eyes wide, brows up | jaw drop shock | | |
| 10 | Duck face / pout | duckface pout | | |
| 11 | Angry yell, brows down, teeth showing | rage powering up | | |
| 12 | Eyes wide, mouth closed | stunned stare | | |
| 13 | Look sideways with squinted eyes | suspicious side eye | | |
| 14 | Eyes closed, slight smile | blushing flowers / exasperated sigh | | |

### Hand gestures (keep your face in view)

| # | Gesture | Expected meme | Result | Notes |
|---|---|---|---|---|
| 15 | Heart with both hands under your chin | heart hands | | |
| 16 | Palm on your forehead, eyes closed | exasperated sigh | | |
| 17 | Both palms on cheeks, mouth open (Home Alone) | screaming horror | | |
| 18 | Peace sign beside your face, playful wink | peace sign hamster | | |
| 19 | Thumbs up beside your face, pleased smile | thumbs-up hamster | | |
| 20 | Point one finger up beside your face | glasses pointing hamster | | |
| 21 | Stick your tongue out playfully | tongue-out hamster | | |

Gesture-tagged memes should not appear when their gesture is absent or gesture
tracking is off, even if your face matches.

The detected gesture name appears in green at the top left of the webcam panel.
Press `g` to confirm the meme changes back when gestures are off.

### Stability and speed

| # | Check | Result |
|---|---|---|
| 22 | Hold a still face 5 seconds. Meme stays put? | |
| 23 | Press `s` (smoothing off). Does it flicker more? (expected yes) | |
| 24 | FPS on the bottom bar | |
| 25 | Latency (ms) on the bottom bar | |
| 26 | Press `m` to `clip`. Is it worse than `expr`? (expected yes) | |

---

## 8. If a face keeps missing: enroll yourself (5 minutes)

```bash
python scripts/enroll.py
```

Each meme appears next to your webcam. Copy the face and press **space**. **s** skips,
**b** goes back, **q** saves and quits. The bottom line shows what the matcher reads from
your face live, like `smile 0.62, brow_down 0.40`. Then rerun `python app.py` and redo
the ❌ rows in the checklist.

Enrollment is saved per laptop in `checkpoints/expr_calib.npz`. Do not commit it.

---

## 9. Send back

Post in the group chat:

1. Your checklist table (copy the markdown or a photo).
2. Your laptop, GPU or no GPU, and the FPS / latency numbers.
3. Lighting (bright, normal, dim) and whether you wear glasses.
4. Whether you did full enrollment, and which rows changed after it.
5. One or two screenshots from `results/` of good and bad matches.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Camera does not open | `python app.py --camera 1`. On macOS allow camera access for Terminal (or VS Code) in System Settings > Privacy & Security > Camera, then restart the terminal |
| Mac shows your iPhone camera | Continuity Camera grabbed index 0. Use `python app.py --camera 1` |
| Very laggy (under 10 FPS) | NVIDIA: do the CUDA step in section 2. Otherwise close other apps, or try `python app.py --no-gestures` |
| `pip install` fails on mediapipe | Make the venv with a different Python version (3.11 or 3.12 are safest) |
| `expression matcher unavailable` | Rerun `python scripts/download_models.py` |
| One meme shows for everything | Redo step 5, then step 8 |
| PowerShell will not run Activate.ps1 | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use cmd |
| Red import errors in VS Code | Open the folder that contains `app.py`, not the one above it |
