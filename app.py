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

VERSION = "3.3.0"
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
    # Serve a 3D foot model scaled to the user's foot length.
    # ?length_mm=230.9 scales the parametric model. Defaults to demo.
    from flask import send_file
    import os, io
    length_mm = request.args.get("length_mm", type=float)
    model_path = os.path.join(os.path.dirname(__file__), "static", "demo-foot.obj")
    if not os.path.exists(model_path):
        return {"error": "3D model not available yet"}, 404
    if not length_mm or length_mm <= 0:
        return send_file(model_path, mimetype="model/obj",
                         as_attachment=True, download_name="foot-3d.obj")
    # Scale the parametric STL (clean, watertight) to the user's length
    try:
        import trimesh
        stl_path = os.path.join(os.path.dirname(__file__), "static", "parametric-foot.stl")
        if not os.path.exists(stl_path):
            return send_file(model_path, mimetype="model/obj",
                             as_attachment=True, download_name="foot-3d.obj")
        mesh = trimesh.load(stl_path, force='mesh')
        # Parametric model is 230.9mm long on X; scale uniformly
        base_len = 230.9
        mesh.vertices *= (length_mm / base_len)
        buf = io.BytesIO()
        mesh.export(buf, file_type='stl')
        buf.seek(0)
        return send_file(buf, mimetype="model/stl",
                         as_attachment=True, download_name="foot-3d.stl")
    except Exception as e:
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
