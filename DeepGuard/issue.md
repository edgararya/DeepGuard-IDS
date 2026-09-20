# Phase 1 – Setup FastAPI Backend & Core AI Engine

Plan for issue #1 (`edgararya/DeepGuard-IDS`).

## 1. Problem statement

Issue #1 asks for a FastAPI backend that loads a 4-model deepfake-detection ensemble
(Meso4_DF, Meso4_F2F, MesoInception_DF, XceptionNet) and exposes REST endpoints for
image and video analysis, with a strict file-validation order and explicit error codes.

**Current state:** the backend already exists and is ~95% compliant. `backend/app/`
contains `main.py`, `detector.py`, `face_utils.py`, `classifiers.py`, `schemas.py`,
`__init__.py`; `backend/weights/` holds all 6 `.h5` files; plus `requirements.txt`,
`Dockerfile`, `docker-compose.yml`, `run.py`, `run_docker.bat`.

Verified against the issue, the following **deviate from the spec** and are the actual
work of this plan:

| # | Spec requirement | Current behaviour |
|---|---|---|
| 1 | Validation order = **Size → MIME/Extension → Decode → Face** | Extension is checked **first**, size second |
| 2 | `INVALID_FILE_TYPE` must fire on **incorrect MIME type** ("a `.txt` renamed to `.jpg`") | Only the **file extension** is checked — a `.txt` renamed `.jpg` falls through to `CORRUPT_FILE` |
| 3 | `warnings` array communicates partial-model state | Error responses always hard-code `warnings: []`, so partial-model context is lost |
| 4 | CORS: allow all origins `["*"]` | `allow_credentials=True` combined with wildcard origin is invalid per the CORS spec (browsers reject it) |

Plus minor hygiene: dead variable `tempvideo_name`, and `schemas.py` is dead code
(imported but never used as a `response_model`).

Everything else already matches the spec and will be **left alone**:
`face_utils.detect_and_crop_face` (CLAHE 2.0 / (8,8), largest face, 35 % margin, `None,None` on failure),
`detector.run_inference` (256² & 299² at `/255.0`, weights 1/1/2/3 with dynamic denominator,
`fake_prob` 1 d.p., 44 % verdict threshold, `null` for unloaded models),
`detector.get_model_status`, GPU memory growth, `TF_CPP_MIN_LOG_LEVEL=3`,
per-model try/except loading, all 6 weight files, all endpoints and response field names.

## 2. Approach

Two parts:

1. **Deliver this plan** as `issue.md` — commit it and push it to `DeepGuard-IDS`, then
   post it as a comment on issue #1.
2. **Implement the four spec fixes** in `backend/app/main.py` and prove them end-to-end
   against the already-built `backend` Docker container.

### Decision log

- **`requirements.txt`** — keep the current, more portable pins
  (`tensorflow>=2.10.0,<2.16.0`, `opencv-python-headless`). Do **not** revert to the
  issue's literal `tensorflow==2.10.0` / `opencv-python`.
- **`issue.md` location** — written at the workspace root, then mirrored into
  `MesoNet-master\DeepGuard\issue.md` and committed/pushed from the `MesoNet-master`
  repo (which carries the `deepguard-ids` remote) so the nested `DeepGuard/` layout
  is preserved.
- **Git scope** — commit **only `issue.md`**. No code and no weights go into git.
- **Issue delivery** — post the plan as a **comment** on issue #1; the original issue
  body is left untouched.
- **Verification** — use the existing Docker container named `backend`;
  rebuild/restart it after the edits and exercise every endpoint with real media from
  `MesoNet-master\dfdc_sample\`.
- **Not in scope** (issue explicitly defers): async/background video processing,
  `/ws/live-scan`, `/history`, job queueing, Electron frontend.

### Git context

- The workspace is **not** a git repo.
- `MesoNet-master` **is** a git repo, with:
  - `origin` → `https://github.com/edgararya/MesoNet-IDS.git` (renamed → redirects)
  - `deepguard-ids` → `https://github.com/edgararya/DeepGuard-IDS.git` ← **target**
- Local HEAD and `deepguard-ids/main` are both at `7c21c85`, so committing only
  `issue.md` adds exactly one commit and pushes nothing else.
- `MesoNet-master\DeepGuard\` is currently untracked and is a byte-identical copy of
  the workspace (verified by MD5 diff across every non-`__pycache__` file).
- ⚠️ The remote `weights/` holds only the 4 small MesoNet `.h5` files (repo size 963 KB).
  `xception_finetuned.h5` is 148 MB, which exceeds GitHub's 100 MB per-file push limit —
  which is why **no weights are committed**. `DeepGuard/` must never be `git add`-ed
  wholesale.

## 3. Todos

### T0 — Write `issue.md` with this plan
Create `DeepGuard/issue.md` containing the full plan (problem statement,
spec-deviation table, approach, decisions, todos, verification matrix, notes).

### T1 — Commit and push `issue.md` to `DeepGuard-IDS`
1. Copy the workspace `issue.md` to `MesoNet-master\DeepGuard\issue.md`.
2. In `MesoNet-master`: `git add DeepGuard/issue.md` (**exact path only** — never
   `git add DeepGuard/` or `-A`, to avoid staging the 250 MB of weights).
3. `git status` to confirm the staged set is exactly one file.
4. Commit with a message describing the Phase 1 plan.
5. `git push deepguard-ids HEAD:main`, then verify the file is present in the remote repo.
6. Confirm `gh auth status` before pushing; if unauthenticated, stop and report rather
   than attempting credential workarounds.

### T2 — Post the plan as a comment on issue #1
`gh issue comment 1 --repo edgararya/DeepGuard-IDS --body-file <path to issue.md>`,
then confirm the comment exists via the API.

### T3 — Reorder the `/analyze/image` validation pipeline
File: `backend/app/main.py`

New order, exactly as the issue's *File Validation Flow*:

1. **Size check** — fast-fail on `UploadFile.size` when populated, then `await file.read()`
   and re-check `len(contents)`. `> 10 MB` → `FILE_TOO_LARGE`.
2. **Extension check** — `.jpg/.jpeg/.png/.webp/.bmp` whitelist → else `INVALID_FILE_TYPE`.
3. **MIME check** — declared `content_type` must be an allowed image MIME (or
   `application/octet-stream`), **and** the real bytes must sniff as a known image
   format via Pillow (`Image.open`). A `.txt` renamed `.jpg` → `INVALID_FILE_TYPE`.
   Empty body → `CORRUPT_FILE`.
4. **Decode check** — `cv2.imdecode` → `None` → `CORRUPT_FILE`.
5. **Model availability** → `ALL_MODELS_FAILED` (HTTP 503).
6. **Face detection** → `NO_FACE_DETECTED`.
7. Inference → 200 with `warnings`.

### T4 — Reorder the `/analyze/video` validation pipeline
File: `backend/app/main.py`

1. **Size check** → `> 100 MB` → `FILE_TOO_LARGE`.
2. **Extension check** — `.mp4/.avi/.mov/.mkv/.webm` → else `INVALID_FILE_TYPE`.
3. **MIME check** — declared `content_type` must be video-ish (or
   `application/octet-stream`), **and** magic bytes must match a known container
   (`ftyp` ISO-BMFF for mp4/mov, `RIFF....AVI ` for avi, `\x1aE\xdf\xa3` EBML for mkv/webm)
   → else `INVALID_FILE_TYPE`.
4. **Decode check** — write temp file, `cv2.VideoCapture(...).isOpened()` false → `CORRUPT_FILE`.
5. **Model availability** → `ALL_MODELS_FAILED` (HTTP 503).
6. Frame scan (unchanged: sample every 10th frame, `detect_and_crop_face` → `run_inference`).
7. No face in any sampled frame → `NO_FACE_DETECTED_IN_VIDEO`.
8. Success 200 with all contract fields (unchanged).

### T5 — Surface model warnings in error responses
File: `backend/app/main.py`

Resolve `get_model_status()` once per request and reuse it for both the
`ALL_MODELS_FAILED` gate and the `warnings` field, so error payloads report
partial-model state instead of a hard-coded `[]`. Still `[]` when all 4 models load.

### T6 — Fix CORS wildcard + credentials
File: `backend/app/main.py`

Set `allow_credentials=False` while keeping `allow_origins=["*"]` (the issue only
requires all origins; wildcard + credentials is rejected by browsers).

### T7 — Hygiene
File: `backend/app/main.py`

- Remove the dead `tempvideo_name` variable.
- Wire `schemas.py` in as `response_model` for the two analyze endpoints (it is
  currently imported but unused), and drop now-unneeded imports. `JSONResponse` error
  paths bypass the response model, so the error contract is unaffected.

### T8 — Verify end-to-end against the Docker container
Rebuild/restart the `backend` container, then run:

| # | Request | Expected |
|---|---|---|
| 1 | `GET /health` | 200, `status: ok`, all `models_loaded` true, `warnings: []` |
| 2 | `POST /analyze/image` — `dfdc_sample\image\1.jpg` | 200, `has_face: true`, `fake_prob`, `verdict`, 4 `scores`, `face_box` |
| 3 | `POST /analyze/image` — generated no-face PNG | 400 `NO_FACE_DETECTED` |
| 4 | `POST /analyze/image` — 11 MB jpg | 400 `FILE_TOO_LARGE` |
| 5 | `POST /analyze/image` — `notes.txt` renamed `.jpg` | 400 `INVALID_FILE_TYPE` |
| 6 | `POST /analyze/image` — `.exe` | 400 `INVALID_FILE_TYPE` |
| 7 | `POST /analyze/image` — valid JPEG header + garbage body | 400 `CORRUPT_FILE` |
| 8 | `POST /analyze/video` — `dfdc_sample\deepfake\3.mp4` | 200, frame counts + verdict |
| 9 | `POST /analyze/video` — `notes.txt` | 400 `INVALID_FILE_TYPE` |
| 10 | `POST /analyze/video` — corrupt `.mp4` | 400 `CORRUPT_FILE` |
| 11 | Hide one `.h5` weight, restart, `GET /health` + `POST /analyze/image` | `status: partial`, non-empty `warnings`, **200 OK** with adjusted weighting → then restore the weight and re-verify |

Also assert the JSON key set of every response matches the contract exactly
(`success, message, processing_time_ms, warnings, result, error`), and confirm
`fake_prob` has 1 decimal place and the verdict threshold is 44 %.

### T9 — Clean up temp test artifacts
Delete generated fixtures (oversize file, renamed `.txt`, no-face PNG, corrupt files)
created during verification.

## 4. Dependencies

```
T0 ─> T1 ─> T2                       (plan delivery)
T3 ─┐
T4 ─┼─> T5 ─> T6 ─> T7 ─> T8 ─> T9   (implementation + verification)
```

T3 and T4 are independent edits to the same file, applied together; T5–T7 build on them;
T8 verifies everything; T9 cleans up. The two chains are independent of each other.

## 5. Notes & considerations

- **`UploadFile.size`** may be `None` on older Starlette. Guard with
  `getattr(file, "size", None)` and always re-check after `read()`.
- **Pillow vs OpenCV roles** are deliberately split: Pillow answers *"is this really an
  image?"* (→ `INVALID_FILE_TYPE`), OpenCV answers *"can this image be decoded?"*
  (→ `CORRUPT_FILE`). This is what makes the spec's `.txt`-renamed-`.jpg` example land
  on the right error code.
- **`content_type` is untrusted** — it is only a secondary signal. The magic-byte /
  Pillow sniff is the authoritative check, so a client lying about the MIME type still
  gets the correct error.
- **`application/octet-stream` is tolerated** so Postman/Electron uploads that do not
  set a MIME type are not falsely rejected.
- **XceptionNet loading** (`xception_finetuned.h5`, 155 MB) matches `live_scanner.py`
  exactly — `keras.applications.Xception(weights=None)` + `load_weights`. Not changed.
  `weights/xception_deepfake_k2.h5` is an unused legacy artifact and is left in place
  (the issue says to copy *all* `.h5` files).
- **Video sampling** (`SAMPLE_INTERVAL = 10`) is unchanged; the issue marks synchronous
  video processing as an accepted limitation for this phase.
- The `warnings` field will now be populated on error responses when models are
  partially loaded. The issue's error example shows `[]`, which is still produced
  whenever all 4 models load successfully — this only adds information in the degraded case.
- **`git add` discipline** — stage `DeepGuard/issue.md` by explicit path only. A bare
  `git add DeepGuard/` or `git add -A` would stage 250 MB of weights and the push
  would be rejected by GitHub's 100 MB limit.
- The local `MesoNet-master` tree has pre-existing uncommitted modifications to
  `.gitignore` and `live_scanner.py`. These are **not** part of this work and will not
  be staged or committed.
