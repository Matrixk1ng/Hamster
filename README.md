# HamsterMirror

Real-time facial expression to hamster-meme retrieval (CS4391 project).
Webcam on the left, the hamster meme that best matches your face on the right.

```
webcam frame
   │
   ▼
face detection + square crop       (MediaPipe BlazeFace, Haar fallback)
   │
   ├──► Expression matcher (default): MediaPipe Face Landmarker → 52 blendshapes
   │        → 17 features (smile, smile asymmetry, teeth, jaw open, brow down, brow raise,
   │        squint, pucker, sneer, ...) → nearest meme profile (memes/expressions.csv,
   │        or your own enrolled profiles)
   │
   ├──► hand tracking (MediaPipe, 21 points per hand) → rule-based gesture
   │        (heart hands, hand on forehead, hands on cheeks, steepled fingers, thumbs up, ...)
   │
   ├──► Baseline: ResNet18 on FER2013 → 7 emotions → meme tagged with that emotion
   │
   └──► Proposed: CLIP ViT-B/32 embedding of the face (+ hands when near the face)
              (+ optional projection head)
              cosine similarity vs. meme embeddings
              (meme = blend of CLIP image embedding and CLIP text embedding of its description)
   │
   ▼
gesture bonus: memes tagged with the detected gesture get a score boost
   │
   ▼
temporal smoothing: EMA over the vector + hysteresis (switch only if a new meme
wins by a margin for N frames in a row)
   │
   ▼
meme on screen
```

**Teammates testing the demo: follow [TESTING.md](TESTING.md).**

## Repo map

| Path | What it does |
|---|---|
| `app.py` | Live demo. Keys: `q` quit, `m` switch method, `s` smoothing on/off, `g` gestures on/off, `t` cycle CLIP text weight, `f` mirror, `space` snapshot |
| `hamstermirror/face.py` | Face detection backends + padded square crop |
| `hamstermirror/baseline.py` | ResNet18 expression classifier and emotion-to-meme matcher |
| `hamstermirror/clip_retrieval.py` | CLIP encoder, meme index, CLIP matcher |
| `hamstermirror/projection.py` | Optional residual projection head trained on (face, meme) pairs |
| `hamstermirror/expression.py` | Blendshape features, expression matcher, hybrid expr+CLIP matcher |
| `memes/expressions.csv` | Hand-written target expression for each meme |
| `scripts/enroll.py` | Calibrate your neutral face and record your own version of each meme |
| `hamstermirror/hands.py` | MediaPipe hand landmark tracking + skeleton drawing |
| `hamstermirror/gestures.py` | Rule-based gesture classifier, gesture smoothing, face+hands crop |
| `hamstermirror/smoothing.py` | EMA + hysteresis |
| `hamstermirror/pipeline.py` | Frame to meme, shared by the app and every eval |
| `memes/` | Meme images + `memes.csv` (file, emotion, description) |
| `scripts/download_models.py` | Fetch BlazeFace, Face Landmarker, Hand Landmarker, CLIP and ResNet18 weights once |
| `scripts/build_index.py` | Encode the meme gallery with CLIP |
| `scripts/train_baseline.py` | Fine-tune the baseline on FER2013 |
| `scripts/record_clips.py` | Record consenting participants making prompted expressions |
| `scripts/annotate.py` | Crop faces from clips and label emotion + acceptable memes |
| `scripts/train_projection.py` | Train the projection head on your annotations |
| `eval/eval_expression.py` | Accuracy + macro-F1 on FER2013 and your clips |
| `eval/eval_retrieval.py` | Top-1/top-3 hit rate, text-weight and smoothing ablations, failure analysis |
| `eval/eval_realtime.py` | FPS, latency, flicker rate |
| `eval/user_study.py` | Blind A/B Likert study and its analysis |
| `tests/` | Unit tests + an end-to-end smoke test (no webcam, no downloads) |

---

## Part 1. Get the demo running (about 10 minutes)

Use Python 3.10 to 3.12 (MediaPipe wheels lag behind new Python releases).

```bash
cd hamstermirror
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

**1. Check the install works** (no webcam or downloads needed):

```bash
python -m pytest -q tests
python tests/e2e_smoke.py
```

Expect `42 passed` and `E2E SMOKE TEST PASSED`. The smoke test trains, indexes,
evaluates and runs the app on synthetic data with random CLIP weights, so its
numbers mean nothing. It only proves every script runs.

**2. Download the real weights** (about 350 MB, one time):

```bash
python scripts/download_models.py
```

**3. Build the meme index:**

```bash
python scripts/build_index.py
```

**4. Run the live demo:**

```bash
python app.py
```

A window opens with your webcam and the matched meme. With only the one seeded
hamster it will always show that hamster, which is expected. Add more memes
(Part 2) to see retrieval actually choose. If the camera does not open, try
`python app.py --camera 1`. On macOS, allow camera access for your terminal.

Things to try in the window:
- `t` cycles the CLIP text weight (0 = pure image matching, 1 = pure description matching). Watch how the choice changes.
- `s` turns smoothing off. The meme should start flickering on a still face.
- `m` switches to the baseline once you have trained it (Part 3).

---

## Part 2. Build the meme gallery (100 to 200 memes)

Drop images in `memes/` and add a row per image to `memes/memes.csv`:

```csv
file,emotion,description,gesture
hamster_laughing.png,happy,"like it is laughing hysterically with its mouth wide open and eyes squeezed shut",
hamster_smug.png,happy,"smug and self-satisfied with a half smile",
hamster_peace.png,happy,"winking and holding up a peace sign",peace
```

- `emotion` must be one of `angry, disgust, fear, happy, sad, surprise, neutral`. The baseline can only use this tag.
- `description` is free text and is what lets CLIP tell "smug" from "laughing". It is inserted into `a photo of a face that looks {description}`, so write it as how the face looks, not what the meme is about.
- `gesture` is optional, but required for any image that depends on a hand pose. A tagged meme is excluded unless that gesture is detected.
- Re-run `python scripts/build_index.py` after every gallery change. It prints how many memes retrieve their own description, which is a quick check that descriptions are distinct.

The app also rebuilds a stale index in memory, so forgetting is not fatal, just slower to start.

---

## Part 2a. The expression matcher (default method)

CLIP alone compares a human photo with a cartoon drawing as a whole. It does not
measure mouth corners or eyebrows, so a few memes become "hubs" that win for almost
any face. The expression matcher measures your face directly:

1. MediaPipe Face Landmarker outputs 52 blendshape scores per frame.
2. These are folded into 17 features, including **smile asymmetry** (smirk vs smile),
   **brow asymmetry** (one raised eyebrow), **teeth** (upper lip up + lower lip down),
   **brow down**, **squint**, **jaw open** and **pucker**.
3. Each meme has a target profile in `memes/expressions.csv` (0 = absent, 1 = strong).
   The closest profile wins.

**Calibrate first (1 minute, strongly recommended):**

```bash
python scripts/enroll.py --neutral-only
```

This records your resting face and subtracts it, so a neutral face reads as neutral
and not as "deeply unimpressed".

**Enroll your own faces (5 minutes, most accurate):**

```bash
python scripts/enroll.py
```

Each meme appears next to your webcam. Copy the face and press space. It records about
a second and stores your version as that meme's profile. `s` skips, `b` goes back, `q`
saves and quits. The bottom line shows your live top features ("you now = smile 0.62,
brow_down 0.40 ..."), which helps you see what the matcher sees. Enrollment is per
person (`checkpoints/expr_calib.npz`), so re-run it for each demo user or just do
`--neutral-only` for them.

In the app, `m` cycles `expr` → `expr+clip` (expression plus a bit of CLIP) → `clip` →
`baseline`, so you can show the difference live. When you add a new meme, add a row
to `expressions.csv` too, or enroll it. Memes without a profile are skipped by `expr`.

---

## Part 2b. Hand gestures

The app tracks your hands as well as your face. A gesture changes the meme in two ways:

1. **Gesture requirement.** Memes can have a `gesture` in `memes.csv`. When you make
   that gesture (held for about 2 frames), the meme becomes eligible and gets a score
   boost. If the gesture is absent, not detected, or gesture tracking is off, that meme
   is excluded; its face alone cannot make it win.
2. **Wider crop for CLIP.** When a hand is near your face, CLIP sees a crop of face plus
   hands (thin orange box in the app), so untagged memes with hands can still match.

Gestures the classifier knows:

| Gesture | How to make it |
|---|---|
| `heart_hands` | Both hands, index tips touching on top, thumb tips touching below |
| `steepled_fingers` | Both hands, fingertips touching (scheming villain) |
| `hands_on_cheeks` | One palm on each side of your face (Home Alone scream) |
| `hand_on_forehead` | Palm on your forehead |
| `hand_over_face` | Palm over your mouth or nose |
| `thumbs_up` / `thumbs_down` | Fist with thumb up or down |
| `peace` | Index and middle finger up |
| `pointing` | Only the index finger up |
| `open_palm` | All fingers spread |
| `fist` | Closed fist |

The detected gesture shows in green in the top left of the webcam panel, and `g`
turns gestures off. Gesture-tagged memes remain excluded while gestures are off, so
they cannot match on face alone. If gestures fire too easily or not enough, change the
boost with `python app.py --gesture-bonus 0.2` (default 0.1 for CLIP). To add a new
gesture meme, put the gesture name in the last column of its row.

The gesture rules are geometry on the 21 hand landmarks, measured relative to your
face width, so they work at any distance from the camera. They live in
`hamstermirror/gestures.py` if you want to tweak thresholds.

---

## Part 3. Train the baseline on FER2013

Download FER2013 from Kaggle in either format:
- the original challenge CSV `fer2013.csv` (columns `emotion,pixels,Usage`), or
- the image-folder version (`train/<class>/*.png`, `test/<class>/*.png`).

```bash
python scripts/train_baseline.py --fer data/fer2013.csv --epochs 15
# or
python scripts/train_baseline.py --fer data/fer2013 --epochs 15
```

On a laptop GPU or Apple Silicon this takes roughly 10 to 30 minutes. On CPU use
`--limit 5000 --epochs 3` for a quick version. A ResNet18 fine-tuned like this
usually lands around 65 to 70% test accuracy on FER2013. The best checkpoint goes
to `checkpoints/baseline.pt` and `app.py` picks it up automatically (`m` key).

AffectNet: export it to the same folder layout with the 7 shared classes (drop
"contempt") and pass that folder to `--fer`.

---

## Part 4. Collect and label your own data

**Record** (one run per participant per lighting/pose condition):

```bash
python scripts/record_clips.py --participant p01 --lighting bright --pose frontal --glasses no --skin-tone monk-5 --spontaneous 3
python scripts/record_clips.py --participant p01 --lighting dim --pose turned
```

It asks to confirm consent first, then prompts 12 expressions (the 7 FER classes
plus smug, betrayed, panicked, laughing hysterically, deadpan stare) and 7
gestures (heart hands, hand on forehead, hands on cheeks, steepled fingers,
hand over face, thumbs up, peace sign; skip them with `--no-gestures`) with a
countdown and 3 s of recording each. Skin tone is optional and self-reported.
Clips and metadata go to `data/clips/` and `data/clips.csv`. Recordings are in
`.gitignore` on purpose, keep them off GitHub.

**Annotate:**

```bash
python scripts/annotate.py --annotator obinna
```

For each clip it shows the face crops and a numbered sheet of all memes (also
saved as `results/meme_sheet.png`). Type the emotion number, then every
acceptable meme number with the best one first, like `14,3`. Labels apply to all
sampled frames of that clip. Have each teammate annotate with their own
`--annotator` name if you want inter-annotator agreement.

**Optional projection head:**

```bash
python scripts/train_projection.py
```

Restart `app.py` and `clip+proj` appears in the `m` cycle.

---

## Part 5. Run the evaluations from the proposal

All results are written as markdown tables to `results/`, ready to paste into the report.

The retrieval eval also reports `expr`, `expr+clip` and `expr+gestures`. The user study
compares the baseline against `expr` by default (`--proposed clip` to use CLIP instead).

| Proposal item | Command | Output |
|---|---|---|
| Expression accuracy (acc, macro-F1) on FER2013 and own clips | `python eval/eval_expression.py --fer data/fer2013.csv` | `results/expression.md` |
| Retrieval top-1/top-3, baseline vs CLIP, smoothing ablation, failure analysis | `python eval/eval_retrieval.py --proj-cv` | `results/retrieval.md`, `retrieval_per_frame.csv` |
| Real-time FPS, latency, flicker rate | `python eval/eval_realtime.py` (uses your neutral/deadpan clips) or `--webcam 10` | `results/realtime.md` |
| User study, 10 to 20 participants, blind 1 to 5 ratings | `python eval/user_study.py --participant u01` per person, then `--analyze` | `results/user_study.csv`, `user_study.md` |

Notes on what each number means:
- **Gestures**: `annotate.py` also saves a face+hands crop and the detected gesture for each frame (`data/frame_meta.csv`). The retrieval eval then adds `clip+gestures` and `baseline+gestures` rows, plus a breakdown by whether the correct meme is a gesture meme. That comparison is the evidence that gestures help.
- **Retrieval** evaluates on saved face crops, so detector misses do not count against the matcher. It also sweeps the CLIP text weight (0, 0.3, 0.6, 1.0), which is a free ablation for the report.
- **`--proj-cv`** trains the projection head leave-one-participant-out, so `clip+proj` is never scored on a person it trained on. Do not report numbers from a head trained on all participants.
- **Smoothing** is applied within each clip in frame order. "Switch rate" is how often the shown meme changed between consecutive sampled frames.
- **Flicker rate** in `eval_realtime.py` is meme switches per second on clips where the face is held still. Lower is better, and it should drop sharply with smoothing on.
- **Failure analysis** groups top-1 by lighting, pose, glasses, skin tone and posed vs spontaneous, using the metadata you entered while recording.
- **User study**: participant makes a face and presses space. The baseline and CLIP picks appear as A and B in random order and they rate each 1 to 5. `--analyze` gives means, win rate and a Wilcoxon signed-rank test once you have at least 5 participants.

---

## Tuning knobs

| Knob | Where | Default | Effect |
|---|---|---|---|
| Text weight | `--text-weight` / `t` key | 0.6 | Higher leans on meme descriptions, which bridges the human vs cartoon domain gap |
| EMA alpha | `HamsterMirror(alpha=)` | 0.3 | Lower is smoother but slower to react |
| Hysteresis margin | `HamsterMirror(margin=)` | 0.01 CLIP, 0.05 baseline | How much a new meme must win by |
| Hysteresis frames | `HamsterMirror(min_frames=)` | 3 | Consecutive frames a challenger must win |
| CLIP model | `--model` in `build_index.py`, `config.CLIP_MODEL` | ViT-B-32 / openai | Bigger models are more accurate and slower |
| Detector | `--detector` | auto | `mediapipe`, `mp_legacy`, `haar` |

## Troubleshooting

- **`expression matcher unavailable` printed at startup**: run `scripts/download_models.py` to get `face_landmarker.task`. The app falls back to CLIP.
- **One meme keeps winning**: run `scripts/enroll.py --neutral-only`. If it is still sticky, enroll that meme and its neighbors so their profiles are your actual faces.
- **`hand tracking off` printed at startup**: run `scripts/download_models.py` to get `hand_landmarker.task`.
- **`detector=haar` printed at startup**: the BlazeFace model is missing. Run `scripts/download_models.py`. Haar works but misses turned heads.
- **"Meme index is stale"**: you changed `memes.csv`. Run `scripts/build_index.py`.
- **Slow on CPU**: CLIP is about 90 ms per frame on a laptop CPU (around 8 FPS). A GPU or Apple Silicon (`mps`, picked automatically) is much faster. The baseline runs about 2x faster on CPU.
- **MediaPipe will not install**: you are probably on Python 3.13. Make the venv with 3.11 or 3.12.
