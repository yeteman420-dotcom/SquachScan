# ModToes Foot Scanner v3 — Backend

A privacy-first Flask backend that turns a top-down foot photo into a CAD
technical drawing (SVG + PNG) with estimated length/width measurements.

**What it is:** `app.py` (Flask API) + `pipeline.py` (neural measurement
pipeline) + `cad.py` (CAD renderer, SVG + Pillow PNG). No photo is ever
stored — processing happens entirely in memory and the image is gone the
moment the drawing is made.

**v3 segmentation engine:** U²-Net (`u2netp`, 4.7 MB) salient-object
segmentation via ONNX Runtime (CPU-only). The model runs at 320×320, the
mask is upscaled and morphologically refined, and the largest connected
component becomes the foot contour. This replaces v2's Otsu/GrabCut
thresholding, which failed on carpet and textured backgrounds. The
classical pipeline is kept as a last-resort fallback if the model is
unavailable. Model weights download at build time
(`scripts/fetch_model.py`) and are never committed to git. BiRefNet was
evaluated and rejected: its 109–930 MB weights cannot load in the Render
free tier's 512 MB RAM. Synthetic QA: IoU 0.97–0.99 on carpet textures,
calibration error under 5%, full scan ~0.5–1.5s on CPU. No real-foot
testing yet — see DEPLOY_NOTES.md.

## API

### POST /api/consent
```json
{ "consent": true }
```
→ `200 {"ok": true, "consent_id": "<uuid4>"}`.
Appends one line to `consent_log/consent.jsonl`:
`{"ts": "<iso>", "consent_id": "<uuid>", "ip_hash": "<sha256(ip+salt)>}"`.
Nothing else is stored. No photo, no name.
`{"consent": <anything but true>}` → `400 {"ok": false, "error": "...", "code": "NO_CONSENT"}`.

### POST /api/scan
Multipart form: `photo` (file, max 8 MB), `foot_side` (`left`|`right`),
`reference` (`quarter`|`card`|`none`), `consent_id`.

Guards run in this order:
1. Valid `consent_id` → else `400 NO_CONSENT`
2. Rate limit: 10 scans/hour/IP, sliding window, hand-rolled in memory → else `429 RATE_LIMITED`
3. Valid image (EXIF-stripped, decodable) → else `400 BAD_IMAGE`
4. Foot found in frame → else `400 NO_FOOT`

Success → `200`:
```json
{
  "ok": true,
  "drawing_id": "<uuid4>",
  "svg": "<svg …>",
  "png_base64": "…",
  "measurements": {
    "length_mm": 248.1, "ball_width_mm": 98.2, "heel_width_mm": 61.4,
    "arch_width_mm": 72.0, "length_in": 9.77,
    "calibrated": true, "reference": "quarter", "px_per_mm": 4.1217
  },
  "warnings": ["ESTIMATED — verify with physical measurement"]
}
```
When unscaled (`reference: "none"` or reference not detected), all `*_mm` /
`length_in` are `null`, `calibrated` is `false`, and `px_per_mm` is `null` —
the drawing is shape-only and **no dimensions are rendered**. The backend
never guesses a scale.

Errors are always `{"ok": false, "error": "plain message", "code": "…"}`.

### GET /healthz → `{"ok": true, "version": "3.0.0"}`
### GET / → scanner UI · GET /privacy, /terms → policy pages

## How it works

1. **EXIF strip** (`pipeline.strip_exif`): Pillow re-encodes the upload to
   RGB JPEG with zero metadata; verified — no EXIF survives.
2. **Decode + downscale** to max 1200 px (scale factor tracked).
3. **Segmentation**: U²-Net neural saliency mask → upscale → morphological
   refine → largest connected component → foot contour (sanity-checked:
   area 5–60% of image, minAreaRect aspect 1.8–4.2, not touching the frame).
   Last-resort fallbacks: Otsu (both polarities), then GrabCut. Raises
   `NoFootFound` otherwise.
4. **Reference detection**: quarter via HoughCircles with edge-support
   verification + hill-climbing refinement + contour-circularity fallback
   (diameter must be 6–16% of foot length; `px_per_mm = d_px / 24.26`);
   credit card via `approxPolyDP` quadrilaterals on convex hulls with aspect
   1.586 ± 0.12, rectangularity checks, foot-overlap rejection, plus a
   uniformity-based fallback for patterned rugs
   (`px_per_mm = long_side_px / 85.60`). Absurd detections are rejected,
   never calibrated from.
5. **Landmarks**: PCA long axis; length = max heel→toe extent; widths along
   the minor axis at stations (ball = max width in forefoot third, heel =
   width at rear 12–18%, arch = min width in middle third). Heel/toe
   disambiguation: the narrower end is the heel.
6. **CAD** (`cad.FootDrawing`): rotates the foot heel-down/toe-up, draws the
   outline (navy #1a2b4a), chain-thin centerline, dimension lines with
   arrowheads (calibrated only), title block, and the red
   `ESTIMATED — VERIFY WITH PHYSICAL MEASUREMENT. Not medical advice.` footer.
   `to_svg()` → standalone SVG; `to_png()` → ~150 DPI PNG via Pillow only.

## Model weights

`models/u2netp.onnx` (4.4 MB) is **not** in git. It downloads at build time:
`python scripts/fetch_model.py` verifies size + SHA-256 and fails loudly on
a bad download. Render runs it via `buildCommand` (see `render.yaml`).
For local dev, run the script once manually.

## Security & privacy architecture

- **Photo in memory only**: upload bytes → `BytesIO` → numpy. No code path
  writes a photo to disk. (If that ever changes: `tempfile` + `os.remove`
  in `finally`, with a comment.)
- **EXIF stripped before processing**, verified programmatically.
- **Only persistent state**: `consent_log/consent.jsonl` — one line per
  consent (`ts`, `consent_id`, `ip_hash`). No photos, names, emails,
  measurements, or drawings are stored.
- **Logs carry only** `drawing_id`, `calibrated`, and timing. Never image
  bytes, filenames, dimensions, or measurements.
- **No cookies, no analytics, no external requests** from the server.
- **Security headers on every response**: strict CSP
  (`default-src 'self'; img-src 'self' data: blob:; script-src 'self';
  style-src 'self' 'unsafe-inline'; object-src 'none'; base-uri 'self'`),
  `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`,
  `Referrer-Policy: no-referrer`.
- **Secrets**: none required. Optional `CONSENT_SALT` env var salts the
  one-way IP hashes; unset = random per-process salt.
- **No precision claims beyond the method** — no "certified", no medical
  claims anywhere in code or drawings.

## Deploy

### Render (backend)
1. Push this directory to a repo (or use Render's directory support).
2. In Render: **New → Web Service** → point at the repo.
   Or apply `render.yaml` via Infrastructure-as-Code: it sets
   `buildCommand: pip install -r requirements.txt` and
   `startCommand: gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 120`,
   with `healthCheckPath: /healthz` and a generated `CONSENT_SALT`.
3. Note: **Render disks are ephemeral** — `consent_log/consent.jsonl` does
   not survive restarts/deploys. Users simply re-consent. Do NOT attach a
   persistent disk for photos; there must be nowhere for them to land.

### Cloudflare DNS + HTTPS
1. In Cloudflare, add a `CNAME` for your hostname (e.g. `scan.squachacers.com`)
   → your Render service URL (`<name>.onrender.com`), proxied (orange cloud).
2. SSL/TLS mode: **Full (strict)**.
3. **Enable HSTS**: Cloudflare dashboard → SSL/TLS → Edge Certificates →
   turn on **HTTP Strict Transport Security (HSTS)** (include subdomains +
   preload as desired). The app itself does not emit HSTS (it can't know the
   canonical host); Cloudflare terminates TLS, so HSTS belongs there.

### Local dev
```bash
./venv/bin/python -m pip install -r requirements.txt   # already installed
./venv/bin/python app.py                                # http://127.0.0.1:5000
# or: ./venv/bin/gunicorn app:app --bind 127.0.0.1:5000
```

## Files
- `app.py` — routes, security headers, rate limiting, consent log, errors
- `pipeline.py` — neural pipeline (EXIF strip → U²-Net segment → reference → landmarks)
- `cad.py` — CAD renderer (SVG string + PNG bytes, Pillow only)
- `models/` — U²-Net weights (downloaded at build, gitignored, never committed)
- `scripts/fetch_model.py` — verified weights downloader (build + local dev)
- `templates/` — UI + privacy/terms pages (frontend owned by sibling)
- `static/` — frontend assets (sibling)
- `consent_log/` — append-only consent records (only persistent state)
- `render.yaml`, `requirements.txt`, `DEPLOY_NOTES.md`
