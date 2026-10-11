# SquachScan v3.0.0 — Deploy Notes

## What changed (v2 → v3)

**The segmentation engine was replaced.** v2 used classical computer vision
(Otsu thresholding + GrabCut + largest contour). It worked on plain
backgrounds but failed completely on carpet — textured backgrounds break
thresholding. v3 uses a neural salient-object segmentation model:

- **U²-Net (`u2netp`, 4.7 MB) via ONNX Runtime (CPU-only).** Runs at 320×320,
  mask upscaled + morphologically refined, largest connected component →
  foot contour. Inference ~0.4–0.8s on a weak CPU; full scan ~0.5–1.5s.
- **Why U²-Net, not BiRefNet:** BiRefNet (the SOTA pick) needs 109–930 MB of
  weights and GBs of RAM. The Render free tier has 512 MB RAM — BiRefNet
  cannot load there, and its CPU inference would blow the scan time budget.
  U²-Net fits the deployment target with huge headroom and still achieves
  0.97–0.99 IoU on synthetic carpet tests.
- **Classical pipeline kept as last resort:** if the model weights are
  missing/corrupt or return no sane mask, the v2 Otsu → GrabCut chain runs
  automatically. The API never breaks because of a model problem.

**Hardened reference detection:**
- Quarter: HoughCircles + edge-support verification (rejects texture ghosts)
  + hill-climbing (center, radius) refinement + contour-circularity fallback
  detector for when texture defeats the Hough accumulator.
- Credit card: Canny quads with convex-hull recovery, rectangularity checks
  (area ratio + corner angles), foot-overlap rejection; plus a
  uniformity-based fallback (credit cards are flat manufactured objects with
  far lower local color variance than carpet) that re-runs edge detection in
  a tight ROI for high-contrast patterned rugs.
- Absurd detections are rejected, never calibrated from. Unscaled fallback
  is unchanged and honest.

**Unchanged:** API contract, frontend (templates/, static/), privacy
architecture (in-memory only, zero photo disk writes, EXIF/GPS stripped,
consent gate, 10/hr/IP rate limit, security headers, no analytics/cookies),
CAD output with all ESTIMATED labels, calibration constants, 8 MB limit,
/privacy and /terms pages.

## Deploy steps

1. Push this directory to the `yeteman420-dotcom/SquachScan` repo
   (or your scanner repo). **Do NOT commit `models/`** — it's gitignored.
2. Render auto-deploys from the repo. The `buildCommand` is now:
   `pip install -r requirements.txt && python scripts/fetch_model.py`
3. **First build takes several extra minutes**: it downloads the 4.4 MB
   model weights and verifies SHA-256. Subsequent builds re-verify the
   cached file (fast path).
4. Verify: `GET /healthz` → `{"ok": true, "version": "3.0.0"}`.
5. No persistent disk needed. Do NOT attach one for photos.

## Local dev

```bash
./venv/bin/python -m pip install -r requirements.txt
python scripts/fetch_model.py   # downloads + verifies weights once
./venv/bin/python app.py        # http://127.0.0.1:5000
```

## Honest limitations

- **QA was synthetic only.** 10 procedurally generated test images (foot
  silhouette on beige carpet, gray berber, patterned rug, white, wood +
  quarter/card/none). No real-foot testing has been done. Real-world
  accuracy on actual feet, lighting, and phone cameras is UNVERIFIED.
- Segmentation IoU on synthetic tests: 0.97–0.99. Calibration error on
  synthetic tests: 0.4–4.8% (all under 5%).
- Single-photo measurements remain ESTIMATES (perspective, lens distortion).
  Every drawing is labeled "ESTIMATED — VERIFY WITH PHYSICAL MEASUREMENT."
- Not certified, not medical-grade. No precision claims beyond the method.

## v3.1.0 — real-world framing fix (2026-10-11)
- Bug: v3.0 rejected any foot touching the frame edge. Real phone photos
  almost always have the ankle/leg running out of frame -> NO_FOOT dead end.
- Fix: contour may touch exactly one frame edge. New framing analysis:
  - Heel cut off by frame -> HEEL_CLIPPED error with specific guidance
    ("pull back so your whole foot — toes to heel — is inside the frame").
  - Toes cut off -> TOES_CLIPPED with guidance.
  - Heel complete but ankle out of frame -> contour truncated at the ankle
    dip so leg can't pollute the length measurement.
- Verified on a real user photo (heel-clipped -> correct guidance) and
  synthetic ankle-out case (truncated at ankle, not frame edge).

