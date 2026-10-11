#!/usr/bin/env python3
"""Fetch the U^2-Net segmentation weights at build/deploy time.

The model file is intentionally NOT committed to git (it is ~4.4 MB of
binary weights). Render runs this via the buildCommand in render.yaml;
run it manually for local dev:

    python scripts/fetch_model.py

Verifies size + SHA-256 after download. Exits non-zero on any failure so
a broken download fails the build loudly instead of shipping a scanner
with no segmentation engine.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.environ.get("SCAN_MODEL_DIR", os.path.join(BASE_DIR, "models"))
MODEL_PATH = os.path.join(MODEL_DIR, "u2netp.onnx")
MODEL_URL = ("https://github.com/danielgatis/rembg/releases/download/"
             "v0.0.0/u2netp.onnx")
MODEL_SHA256 = "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8"
MODEL_SIZE = 4574861


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.isfile(MODEL_PATH):
        if os.path.getsize(MODEL_PATH) == MODEL_SIZE and \
                _sha256(MODEL_PATH) == MODEL_SHA256:
            print(f"model already present and verified: {MODEL_PATH}")
            return 0
        print("existing model file failed verification; re-downloading...")
        os.remove(MODEL_PATH)

    print(f"downloading {MODEL_URL} ...")
    tmp = MODEL_PATH + ".part"
    try:
        req = urllib.request.Request(
            MODEL_URL, headers={"User-Agent": "squachscan-fetch/3.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, \
                open(tmp, "wb") as fh:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
    except Exception as e:
        print(f"download failed: {e}", file=sys.stderr)
        if os.path.exists(tmp):
            os.remove(tmp)
        return 1

    size = os.path.getsize(tmp)
    digest = _sha256(tmp)
    if size != MODEL_SIZE or digest != MODEL_SHA256:
        print(f"verification FAILED: size={size} (want {MODEL_SIZE}), "
              f"sha256={digest}", file=sys.stderr)
        os.remove(tmp)
        return 1

    os.replace(tmp, MODEL_PATH)
    print(f"verified OK: {MODEL_PATH} ({size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
