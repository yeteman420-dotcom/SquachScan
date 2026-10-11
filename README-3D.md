# ModToes 3D Foot Reconstruction — Prototype v0.1

`reconstruct3d.py` turns a **10-second phone video** (or a set of overlapping
photos) into a **3D foot mesh** (OBJ + STL) for 3D printing ModToes prototypes.

The 2D scanner (`app.py` / `pipeline.py`) is untouched — this is a separate
prototype living in the same venv.

## How it works

```
video.mp4 (or photo dir)
  → 1. extract ~40 frames
  → 2. SfM: pycolmap (SIFT → exhaustive match → incremental mapping)
  → 3. foot silhouettes: the 2D scanner's U²-Net segmenter, per frame
     (multi-threshold + centre-crop fallback for varied framing)
  → 4. visual-hull voxel carving: a voxel survives only if it projects
     inside the foot silhouette in ≥75% of views where visible
  → 5. marching cubes → mesh cleanup (largest component)
  → 6. scale to mm: US quarter (24.26 mm) auto-detect, or
     --foot-length-mm for a tape measurement (recommended)
  → 7. export OBJ + STL
```

Why visual hull instead of COLMAP dense (PatchMatch)?
Full COLMAP must be compiled from source (Ceres, etc.) — infeasible on this
VM and on Render. `pycolmap` ships prebuilt wheels (sparse SfM only), and the
visual hull reuses the already-working neural foot segmenter. Trade-off: the
hull can't recover concavities (e.g. the arch underside reads flatter than
reality). Good enough for a first printable prototype, not a medical scan.

## What was installed (into the scanner venv)

| package | version | why |
|---|---|---|
| pycolmap | 4.2.1 | prebuilt COLMAP bindings: SIFT, matching, incremental SfM |
| trimesh | 5.1.1 | mesh cleanup + OBJ/STL export |
| scikit-image | 0.26.0 | marching cubes |
| scipy | 1.18.1 | Gaussian smoothing of the occupancy grid |

No compilation, no system packages, no sudo needed.

## Usage

```bash
cd ~/workspace/squachacers/scanner-v2
./venv/bin/python reconstruct3d.py --video foot.mp4 --out foot_mesh
./venv/bin/python reconstruct3d.py --video foot.mp4 --out foot_mesh \
    --frames 40 --voxels 200 --foot-length-mm 265
./venv/bin/python reconstruct3d.py --images ./frames/ --out foot_mesh
./venv/bin/python reconstruct3d.py --selftest   # synthetic validation
```

Outputs: `foot_mesh.obj`, `foot_mesh.stl` (slicer-ready), plus a console
summary with the scale note and timing.

Key flags:
- `--frames N` — frames sampled from the video (default 40)
- `--voxels N` — voxel resolution along the longest bbox side (default 200;
  higher = finer mesh, more RAM/time)
- `--foot-length-mm` — tape-measured foot length; overrides quarter detection
  (recommended: faster and more reliable than finding a coin in blurry frames)
- `--workdir` — keep scratch (frames, COLMAP db) for inspection

## Resource requirements

Measured on the dev VM (2 vCPU, ~8 GB RAM):

| stage | 40 frames @1280px | notes |
|---|---|---|
| frame extract | ~10 s | cv2, trivial |
| SfM (SIFT+match+map) | 3–8 min | single-threaded-ish; low-texture skin is the risk |
| segmentation (40× U²-Net) | 1–2 min | ONNX CPU, ~2 s/frame |
| voxel carve (200³) | 1–2 min | numpy vectorised; ~50 MB RAM |
| marching cubes + export | ~10 s | |
| **peak RAM** | **~1.2–1.5 GB** | SfM feature matching dominates |

## Can this run on Render?

**Not on the free tier** (512 MB RAM, shared CPU) — SfM alone needs >1 GB,
and a 40-frame reconstruction takes minutes, far beyond a request timeout.

What production would need:
- A worker box with **≥2 GB RAM** (Render Starter is 1 GB — not enough;
  Standard 2 GB is the realistic minimum, 4 GB comfortable).
- **Async job model**: upload video → queue → background worker (5–15 min) →
  notify + download link. Cannot run inside the web request.
- Options: Render Background Worker ($$), or a cheap Hetzner/DO box, or
  run reconstructions on this VM via an internal job queue for now.

## Capture instructions (for Yetey)

The video UX he asked for — no "take 30 photos with 60% overlap":

1. **Put your foot on a textured floor** (the rug is GOOD — plain white
   floors give SfM nothing to match; keep the rug visible around the foot).
2. **Bright, even light.** No harsh shadows, no flash.
3. **Start recording video** (~10 seconds, 1080p or 4K).
4. **Slowly walk the phone around your foot** — one smooth 360° orbit,
   phone tilted ~30–45° down, foot centred in frame the whole time.
   Go slow. No sudden moves.
5. **(Optional but recommended)** put a quarter flat next to your foot
   before recording, or just tape-measure your foot length after.
6. Upload the video.

What makes it fail:
- Motion blur (move slower)
- Foot leaves the frame
- Featureless background (plain floor + plain wall = no match points)
- Foot moves between frames (keep it planted)

## Known limitations (prototype)

- Visual hull can't carve concavities: arch underside will look filled-in.
- Toes may fuse if they touch in most views — spread toes slightly.
- Scale via quarter assumes coin ≈ foot depth; tape measurement is better.
- Skin is low-texture: SfM relies on the background (rug) for matches.
  If registration fails (<8 images), the script errors out with guidance.
- Single foot per video; no multi-object support.
- Phone portrait videos: cv2 may not apply rotation metadata — record in
  landscape or verify frame orientation if SfM looks wrong.
- Not yet wired into the web app (no upload endpoint, no job queue).

## Validation

- `reconstruct3d.py --selftest`: synthetic orbit (24 views) of a textured
  blob → SfM → carve → mesh. **PASS** on 2026-10-11: 15/15 registered,
  736 sparse points, 107 foot points isolated, watertight mesh with
  elongation 1.96.
- Full CLI on synthetic frames (`--images`, U²-Net masks, `--foot-length-mm
  260`): 20 frames → 15 registered → 20/20 masks → OBJ + STL exported,
  longest PCA axis scaled to exactly 260.0 mm, mesh volume ~1.2 L
  (plausible foot volume). 29.8 s total on 2 vCPU.
- Frame extraction: verified on dummy video (12 frames, correct downscale).
- Quarter detector: heuristic — works when the coin is clear and near the
  foot; falls back to UNSCALED + note when not found. `--foot-length-mm`
  (tape measure) is the recommended scaling path.
- Robust segmenter: verified on a wide-framed real foot photo where the
  plain 0.5-threshold mask failed (centre-crop fallback recovered the foot
  at 8.9% frame area).
- 2D scanner untouched: `pipeline.py` / `app.py` import and parse cleanly
  after the new package installs (pycolmap, trimesh, scikit-image, scipy).
