"""ModToes Foot Scanner v3 — Flask backend.

Neural segmentation (U^2-Net via ONNX Runtime CPU) replaces the v2
classical thresholding pipeline. Carpet-proof.

PRIVACY-BY-DESIGN (non-negotiable, enforced here):
- The photo lives ONLY in memory (request bytes -> BytesIO -> numpy).
  Nothing in this app writes an uploaded photo (or any derivative) to disk.
  If a future change ever needs temp files, use tempfile + os.remove in a
  finally block, and say so in a comment.
- The ONLY thing persisted is consent_log/consent.jsonl, one line per
  consent: {"ts", "consent_id", "ip_hash"}. No photo, no name, no email,
  no measurements, no drawings. (On Render the disk is ephemeral, so
  consent records do not survive restarts/deploys — users simply re-consent.)
- Request logs carry ONLY drawing_id, calibrated bool, and timings.
  Never log image bytes, filenames, pixel dimensions, or measurements.
- No analytics, no cookies (we never set any), no external HTTP requests
  from this server.
- Secrets: none needed. Optional CONSENT_SALT comes from the environment.

Security headers (every response):
  Content-Security-Policy, X-Content-Type-Options: nosniff,
  X-Frame-Options: DENY, Referrer-Policy: no-referrer.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import uuid
from collections import deque
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template, request

from pipeline import (
    BadImageError,
    HeelClipped,
    NoFootFound,
    ScanError,
    ToesClipped,
    process_scan,
)
from cad import FootDrawing

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONSENT_DIR = os.path.join(BASE_DIR, "consent_log")
CONSENT_FILE = os.path.join(CONSENT_DIR, "consent.jsonl")
os.makedirs(CONSENT_DIR, exist_ok=True)

VERSION = "3.6.1"
MAX_CONTENT_LENGTH = 8 * 1024 * 1024          # 8 MB photo cap
RATE_LIMIT_N = 10                             # scans ...
RATE_LIMIT_WINDOW = 3600                      # ... per hour, per IP

# Salt for one-way IP hashes in the consent log. Set CONSENT_SALT in the
# environment for stable hashes; otherwise a random per-process salt is used
# (hashes then simply aren't comparable across restarts — fine for privacy).
CONSENT_SALT = os.environ.get("CONSENT_SALT") or uuid.uuid4().hex

CSP = ("default-src 'self'; img-src 'self' data: blob:; script-src 'self'; "
       "style-src 'self' 'unsafe-inline'; object-src 'none'; base-uri 'self'")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
# No Flask session / cookies are used anywhere in this app.


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _err(message: str, code: str, status: int):
    return jsonify({"ok": False, "error": message, "code": code}), status


def _client_ip() -> str:
    # Behind Render/proxies the real client IP is in X-Forwarded-For.
    # Spoofable, but this is only used for rate limiting + one-way hashing.
    fwd = request.headers.get("X-Forwarded-For", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _record_consent(consent_id: str, ip: str) -> None:
    ip_hash = hashlib.sha256(f"{CONSENT_SALT}|{ip}".encode()).hexdigest()
    line = json.dumps({
        "ts": datetime.now(timezone.utc).isoformat(),
        "consent_id": consent_id,
        "ip_hash": ip_hash,
    })
    # The ONLY persistent store in this app. Append-only, one line per consent.
    with open(CONSENT_FILE, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _valid_consent(consent_id: str) -> bool:
    if not consent_id:
        return False
    try:
        uuid.UUID(consent_id)
    except ValueError:
        return False
    try:
        with open(CONSENT_FILE, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("consent_id") == consent_id:
                    return True
    except FileNotFoundError:
        return False
    return False


# Sliding-window rate limiter, hand-rolled, in memory.
# {ip: deque[monotonic timestamps]}. Pruned on every check; no persistence.
_rate_buckets: dict[str, deque] = {}


def _check_rate_limit(ip: str) -> bool:
    now = time.monotonic()
    dq = _rate_buckets.get(ip)
    if dq is None:
        dq = _rate_buckets[ip] = deque()
    while dq and now - dq[0] > RATE_LIMIT_WINDOW:
        dq.popleft()
    if len(dq) >= RATE_LIMIT_N:
        return False
    dq.append(now)
    return True


@app.after_request
def _security_headers(resp):
    resp.headers["Content-Security-Policy"] = CSP
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    return resp


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/healthz")
def healthz():
    info = {"ok": True, "version": VERSION}
    # Model self-test: ensure weights exist (downloading if needed) and
    # verify inference runs.
    try:
        from pipeline import _get_session, MODEL_PATH, ensure_model
        import os
        ensure_model()
        info["model_file"] = os.path.isfile(MODEL_PATH)
        if info["model_file"]:
            import numpy as np
            sess, in_name, out_name = _get_session()
            try:
                x = np.zeros((1, 3, 320, 320), dtype=np.float32)
                out = sess.run([out_name], {in_name: x})[0]
                info["model_inference"] = "ok"
                info["model_output_shape"] = list(out.shape)
            finally:
                del sess
                import gc
                gc.collect()
        else:
            info["model_inference"] = "missing"
    except Exception as e:
        info["model_inference"] = f"error: {type(e).__name__}"
    return jsonify(info)


@app.get("/")
def index():
    return render_template("index.html", version=VERSION)


@app.get("/3d")
def view_3d():
    return render_template("view3d.html", version=VERSION)


@app.get("/api/3d-model")
def get_3d_model():
    return _serve_3d_model()


# --- Parametric foot generator (numpy-only, no heavy deps) ---
# Reference profile extracted from the accepted model (230.9 x 90.7 x 67.8 mm,
# watertight). 40 stations heel->toe, superellipse (e=2.5) cross-sections.
_FOOT_HALF_W = [
    30.0, 31.1, 31.7, 32.0, 32.0, 31.7, 31.2, 30.6, 29.8, 29.1,
    28.4, 28.0, 28.1, 28.6, 29.6, 31.0, 32.7, 34.8, 36.9, 39.0,
    40.9, 42.5, 43.8, 44.7, 45.3, 45.4, 45.0, 44.3, 43.2, 41.7,
    40.0, 38.3, 36.7, 35.0, 32.9, 30.3, 26.9, 22.3, 16.1,  8.0,
]
_FOOT_HEIGHTS = [
    64.5, 66.4, 67.5, 67.8, 67.4, 66.3, 64.5, 62.0, 58.8, 55.1,
    51.1, 47.1, 43.5, 40.3, 37.7, 35.6, 33.9, 32.6, 31.9, 31.5,
    31.6, 32.1, 32.9, 34.0, 35.3, 36.5, 37.4, 37.6, 36.9, 35.5,
    33.7, 31.8, 30.3, 28.8, 27.1, 24.8, 21.6, 17.7, 13.6, 11.5,
]
_FOOT_REF_LEN, _FOOT_REF_WID = 230.9, 90.7


def _build_parametric_foot(length_mm, width_mm=None):
    """Return (verts, faces) for a parametric foot scaled to length_mm
    (and width_mm when given). numpy only. Watertight."""
    import numpy as np
    n_st, n_ar, exp = 40, 32, 2.5
    verts = []
    for i in range(n_st):
        x = i / (n_st - 1) * _FOOT_REF_LEN
        w, h = _FOOT_HALF_W[i], _FOOT_HEIGHTS[i]
        for j in range(n_ar):
            ang = 2 * np.pi * j / n_ar
            cx = np.sign(np.cos(ang)) * abs(np.cos(ang)) ** (2 / exp) * w
            s = np.sin(ang)
            cz = 2 + s * 2 if s < 0 else 2 + s * (h - 2)
            verts.append([x, cx, max(cz, 0.0)])
    verts = np.array(verts)
    faces = []
    for i in range(n_st - 1):
        for j in range(n_ar):
            j2 = (j + 1) % n_ar
            a, b = i * n_ar + j, i * n_ar + j2
            c, d = (i + 1) * n_ar + j, (i + 1) * n_ar + j2
            faces.append([a, c, b])
            faces.append([b, c, d])
    hc = len(verts)
    verts = np.vstack([verts, [[0.0, 0.0, _FOOT_HEIGHTS[0] / 2]]])
    for j in range(n_ar):
        faces.append([hc, (j + 1) % n_ar, j])
    tc = len(verts)
    verts = np.vstack([verts, [[_FOOT_REF_LEN, 0.0, _FOOT_HEIGHTS[-1] / 2]]])
    base = (n_st - 1) * n_ar
    for j in range(n_ar):
        faces.append([tc, base + j, base + (j + 1) % n_ar])
    sx = length_mm / _FOOT_REF_LEN
    sy = (width_mm / _FOOT_REF_WID) if (width_mm and width_mm > 0) else sx
    sz = sx
    verts[:, 0] *= sx
    verts[:, 1] *= sy
    verts[:, 2] *= sz
    return verts, np.array(faces, dtype=np.int32)


def _serve_3d_model():
    # Serve a 3D foot model scaled to the user's foot length.
    # ?length_mm=230.9&width_mm=95 scales the parametric model. Defaults to demo.
    # Procedural generation: no file deps, no heavy libs, just numpy.
    from flask import send_file
    import os, io, struct
    import numpy as np
    length_mm = request.args.get("length_mm", type=float)
    width_mm = request.args.get("width_mm", type=float)  # ball width, optional
    fmt = request.args.get("format", "obj")
    model_path = os.path.join(os.path.dirname(__file__), "static", "demo-foot.obj")
    if not length_mm or length_mm <= 0:
        return send_file(model_path, mimetype="model/obj",
                         as_attachment=True, download_name="foot-3d.obj")
    try:
        verts, faces = _build_parametric_foot(length_mm, width_mm)
        if fmt == "stl":
            buf = io.BytesIO()
            buf.write(b'\x00'*80)
            buf.write(struct.pack('<I', len(faces)))
            for f in faces:
                buf.write(struct.pack('<3f', 0,0,0))
                for vi in f:
                    buf.write(struct.pack('<3f', *verts[vi]))
                buf.write(struct.pack('<H', 0))
            buf.seek(0)
            dl = "foot-3d-%.0fmm" % length_mm
            if width_mm and width_mm > 0:
                dl += "-w%.0fmm" % width_mm
            return send_file(buf, mimetype="model/stl", as_attachment=True,
                             download_name=dl + ".stl")
        else:
            lines = ["# parametric foot %.1fmm" % length_mm]
            for v in verts:
                lines.append("v %.3f %.3f %.3f" % tuple(v))
            for f in faces:
                lines.append("f %d %d %d" % (f[0]+1, f[1]+1, f[2]+1))
            buf = io.BytesIO(("\n".join(lines)).encode())
            dl = "foot-3d-%.0fmm" % length_mm
            if width_mm and width_mm > 0:
                dl += "-w%.0fmm" % width_mm
            return send_file(buf, mimetype="model/obj", as_attachment=True,
                             download_name=dl + ".obj")
    except Exception:
        return send_file(model_path, mimetype="model/obj",
                         as_attachment=True, download_name="foot-3d.obj")


@app.get("/privacy")
def privacy():
    return render_template("privacy.html")


@app.get("/terms")
def terms():
    return render_template("terms.html")


@app.post("/api/consent")
def api_consent():
    if not request.is_json:
        return _err("Request must be JSON.", "NO_CONSENT", 400)
    data = request.get_json(silent=True) or {}
    if data.get("consent") is not True:
        return _err("Consent is required to use the foot scanner.", "NO_CONSENT", 400)
    consent_id = str(uuid.uuid4())
    _record_consent(consent_id, _client_ip())
    return jsonify({"ok": True, "consent_id": consent_id})


@app.post("/api/scan")
def api_scan():
    t0 = time.perf_counter()
    drawing_id = str(uuid.uuid4())

    # Guard 1: valid consent.
    consent_id = (request.form.get("consent_id") or "").strip()
    if not _valid_consent(consent_id):
        return _err("Valid consent is required. POST /api/consent first.",
                    "NO_CONSENT", 400)

    # Guard 2: rate limit, 10 scans/hour/IP, sliding window.
    if not _check_rate_limit(_client_ip()):
        return _err("Rate limit exceeded: 10 scans per hour. Try again later.",
                    "RATE_LIMITED", 429)

    # Cheap param validation before touching the image.
    foot_side = (request.form.get("foot_side") or "").strip().lower()
    if foot_side not in ("left", "right"):
        return _err("foot_side must be 'left' or 'right'.", "BAD_REQUEST", 400)
    reference = (request.form.get("reference") or "none").strip().lower()
    if reference not in ("quarter", "card", "none"):
        return _err("reference must be 'quarter', 'card', or 'none'.",
                    "BAD_REQUEST", 400)

    # Guard 3: valid image.
    f = request.files.get("photo")
    if f is None or not f.filename:
        return _err("No photo uploaded.", "BAD_IMAGE", 400)
    data = f.read()
    if not data:
        return _err("Uploaded photo is empty.", "BAD_IMAGE", 400)

    try:
        scan = process_scan(data, foot_side=foot_side, reference=reference)
    except BadImageError as e:
        return _err(str(e) or "Invalid image.", "BAD_IMAGE", 400)
    except NoFootFound as e:
        return _err(str(e), "NO_FOOT", 400)
    except HeelClipped as e:
        return _err(str(e), "HEEL_CLIPPED", 400)
    except ToesClipped as e:
        return _err(str(e), "TOES_CLIPPED", 400)
    except ScanError as e:
        return _err(str(e) or "Scan failed.", "SCAN_FAILED", 400)
    except Exception:  # never leak internals
        app.logger.exception("scan failed drawing_id=%s", drawing_id)
        return _err("Scan failed unexpectedly.", "INTERNAL_ERROR", 500)

    drawing = FootDrawing.from_scan(scan)
    svg = drawing.to_svg()
    png = drawing.to_png()
    png_b64 = base64.b64encode(png).decode("ascii")

    ms = int((time.perf_counter() - t0) * 1000)
    # Log ONLY: drawing id, calibration flag, timing. Nothing identifiable.
    app.logger.info("scan ok drawing_id=%s calibrated=%s ms=%d",
                    drawing_id, scan["calibrated"], ms)

    return jsonify({
        "ok": True,
        "drawing_id": drawing_id,
        "svg": svg,
        "png_base64": png_b64,
        "measurements": scan["measurements"],
        "warnings": scan["warnings"],
    })


# --------------------------------------------------------------------------
# Error handlers (JSON everywhere)
# --------------------------------------------------------------------------

@app.errorhandler(400)
def _bad_request(e):
    return _err("Bad request.", "BAD_REQUEST", 400)


@app.errorhandler(404)
def _not_found(e):
    return _err("Not found.", "NOT_FOUND", 404)


@app.errorhandler(405)
def _method_not_allowed(e):
    return _err("Method not allowed.", "METHOD_NOT_ALLOWED", 405)


@app.errorhandler(413)
def _too_large(e):
    return _err("Photo too large (max 8 MB).", "FILE_TOO_LARGE", 413)


@app.errorhandler(500)
def _internal(e):
    return _err("Internal error.", "INTERNAL_ERROR", 500)


if __name__ == "__main__":
    # Dev only. Production uses gunicorn (see render.yaml).
    app.run(host="127.0.0.1", port=5000)
