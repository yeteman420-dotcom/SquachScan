"""ModToes Foot Scanner v2 — OpenCV measurement pipeline.

PRIVACY-BY-DESIGN (non-negotiable):
- The photo lives ONLY in memory: raw upload bytes -> Pillow -> numpy array.
  This module NEVER writes the image (or any derivative) to disk.
- EXIF metadata (including GPS) is stripped BEFORE any processing or
  measurement. The re-encoded JPEG carries zero metadata; we verify that.
- Callers must not log image bytes, filenames, or pixel dimensions. The
  `debug` dict returned here contains only geometric/technical notes.

What this does:
  1. strip_exif()      - Pillow re-encode to RGB JPEG, zero metadata, verified.
  2. decode + downscale to max dim 1200px (scale factor tracked).
  3. segment_foot()    - Otsu (both polarities) -> close -> largest sane
                         contour; GrabCut fallback; raises NoFootFound.
  4. reference detect  - quarter via HoughCircles (24.26 mm), credit card
                         via approxPolyDP quadrilaterals (85.60 x 53.98 mm).
                         If the requested reference is not found we return
                         calibrated=False and NEVER guess a scale.
  5. landmarks         - PCA long axis; length = max extent heel->toe;
                         widths at stations along the minor axis.
  6. process_scan()    - returns drawing-ready geometry + measurements.

This module makes NO precision claims beyond the method and NO medical
claims. Output drawings are always labeled ESTIMATED.
"""

from __future__ import annotations

import io
import math

import cv2
import numpy as np
from PIL import Image, ImageOps

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

QUARTER_DIAMETER_MM = 24.26          # US quarter, NIST spec diameter
CARD_LONG_MM = 85.60                 # ISO/IEC 7810 ID-1 credit card
CARD_SHORT_MM = 53.98
CARD_ASPECT = CARD_LONG_MM / CARD_SHORT_MM  # ~= 1.586
CARD_ASPECT_TOL = 0.12

MM_PER_INCH = 25.4

MAX_WORK_DIM = 1200                 # downscale so longest side <= this px

# Quarter diameter is expected to be 6-16% of the foot length in px
# (24.26 mm vs a ~150-400 mm foot photograph).
QUARTER_FRAC_MIN = 0.06
QUARTER_FRAC_MAX = 0.16

ESTIMATED_WARNING = "ESTIMATED \u2014 verify with physical measurement"


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------

class ScanError(Exception):
    """Base class for scan failures (user-facing message in str())."""


class BadImageError(ScanError):
    """The upload is not a usable image."""


class NoFootFound(ScanError):
    """Segmentation could not find a plausible foot contour."""


# --------------------------------------------------------------------------
# 1. EXIF strip
# --------------------------------------------------------------------------

def strip_exif(image_bytes: bytes) -> bytes:
    """Re-encode to RGB JPEG with zero metadata; verify nothing survives.

    Applies the EXIF orientation first (so the image is upright), then drops
    ALL metadata by re-encoding without any exif/icc/profile data.
    Raises BadImageError if the result still carries EXIF.
    """
    if not image_bytes:
        raise BadImageError("Empty file.")
    try:
        with Image.open(io.BytesIO(image_bytes)) as src:
            src.load()
            img = ImageOps.exif_transpose(src)
    except BadImageError:
        raise
    except Exception:
        raise BadImageError("File is not a readable image.")

    if img.mode in ("RGBA", "LA", "PA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    else:
        img = img.convert("RGB")

    buf = io.BytesIO()
    # NOTE: no exif= / icc_profile= kwargs -> zero metadata written.
    img.save(buf, format="JPEG", quality=92)
    clean = buf.getvalue()

    # Verify: no EXIF tag directory and no raw "Exif\\0\\0" APP1 marker.
    with Image.open(io.BytesIO(clean)) as check:
        if len(check.getexif()) != 0:
            raise BadImageError("Could not strip image metadata.")
    if b"Exif\x00\x00" in clean:
        raise BadImageError("Could not strip image metadata.")
    return clean


def _decode_bgr(jpeg_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise BadImageError("Could not decode image.")
    return bgr


# --------------------------------------------------------------------------
# 3. Segmentation
# --------------------------------------------------------------------------

def _contour_passes_sanity(contour, img_area: float, img_w: int, img_h: int) -> bool:
    area = cv2.contourArea(contour)
    if not (0.05 * img_area <= area <= 0.60 * img_area):
        return False
    # Reject anything touching the frame (that is the background, not the foot).
    x, y, w, h = cv2.boundingRect(contour)
    m = 0.02 * max(img_w, img_h)
    if x < m or y < m or x + w > img_w - m or y + h > img_h - m:
        return False
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    if min(rw, rh) < 1e-6:
        return False
    aspect = max(rw, rh) / min(rw, rh)
    # A foot is elongated: not a square-ish blob, not a thin strip.
    return 1.8 <= aspect <= 4.2


def _grabcut_fallback(bgr: np.ndarray, img_area: float):
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w * 0.10), int(h * 0.10), int(w * 0.80), int(h * 0.80))
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best = None
    for c in contours:
        if _contour_passes_sanity(c, img_area, w, h):
            if best is None or cv2.contourArea(c) > cv2.contourArea(best):
                best = c
    return best


def segment_foot(bgr: np.ndarray):
    """Return (contour, method_note). Raises NoFootFound."""
    h, w = bgr.shape[:2]
    img_area = float(h * w)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    best = None
    best_area = 0.0
    for polarity in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        _, th = cv2.threshold(blur, 0, 255, polarity + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
        # CHAIN_APPROX_NONE: width stations sample contour points in narrow
        # bands along the foot axis; SIMPLE would drop all intermediate
        # points on straight edges (e.g. a rectangular midfoot -> zero
        # points in the band -> width 0). approxPolyDP still simplifies
        # the stored outline later.
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            if _contour_passes_sanity(c, img_area, w, h):
                a = cv2.contourArea(c)
                if a > best_area:
                    best, best_area = c, a
    if best is not None:
        return best, "otsu"

    fb = _grabcut_fallback(bgr, img_area)
    if fb is not None:
        return fb, "grabcut"
    raise NoFootFound(
        "No foot detected. Photograph the foot on a plain, contrasting "
        "background with the whole foot visible."
    )


# --------------------------------------------------------------------------
# 5. Landmarks (PCA long axis + width stations)
# --------------------------------------------------------------------------

def _width_profile(pts_centered, major, minor):
    u = pts_centered @ major
    v = pts_centered @ minor
    u_min = float(u.min())
    length = float(u.max() - u_min)

    def width_at(t: float, half: float = 0.03) -> float:
        for h2 in (half, 0.06, 0.10, 0.15):
            lo = u_min + (t - h2) * length
            hi = u_min + (t + h2) * length
            sel = v[(u >= lo) & (u <= hi)]
            if sel.size >= 6:
                p1, p99 = np.percentile(sel, [1, 99])
                return float(p99 - p1)
        return 0.0

    def center_at(t: float):
        lo = u_min + (t - 0.05) * length
        hi = u_min + (t + 0.05) * length
        sel = v[(u >= lo) & (u <= hi)]
        c = float(np.median(sel)) if sel.size else 0.0
        return c

    return u, v, u_min, length, width_at, center_at


def compute_landmarks(contour) -> dict:
    """PCA-based foot geometry in working-image px.

    Returns lengths/widths in px plus centerline station points.
    Heel/toe disambiguation: the narrower end of the long axis is the heel.
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    mean = pts.mean(axis=0)
    centered = pts - mean
    vals, vecs = np.linalg.eigh(np.cov(centered.T))
    order = np.argsort(vals)
    major = vecs[:, order[1]] / np.linalg.norm(vecs[:, order[1]])
    minor = vecs[:, order[0]] / np.linalg.norm(vecs[:, order[0]])

    u, v, u_min, length, width_at, center_at = _width_profile(centered, major, minor)

    # Orient: heel = narrower end.
    if width_at(0.925, half=0.075) < width_at(0.075, half=0.075):
        major = -major
        minor = -minor
        u, v, u_min, length, width_at, center_at = _width_profile(centered, major, minor)

    # Stations along the axis, t=0 at heel, t=1 at toe.
    ball_ts = np.arange(0.60, 0.96, 0.05)
    ball_t = float(ball_ts[np.argmax([width_at(t) for t in ball_ts])])
    arch_ts = np.arange(0.35, 0.66, 0.05)
    arch_t = float(arch_ts[np.argmin([width_at(t) for t in arch_ts])])
    heel_t = 0.15

    def axis_point(t: float):
        return mean + major * (u_min + t * length) + minor * center_at(t)

    heel_pt = mean + major * u_min
    toe_pt = mean + major * (u_min + length)

    return {
        "length_px": length,
        "ball_px": width_at(ball_t),
        "heel_px": width_at(heel_t),
        "arch_px": width_at(arch_t),
        "ball_t": ball_t,
        "arch_t": arch_t,
        "heel_t": heel_t,
        "heel_pt": heel_pt,          # axis extreme at heel end
        "toe_pt": toe_pt,           # axis extreme at toe end
        "stations": {               # centerline points at each width station
            "ball": axis_point(ball_t),
            "heel": axis_point(heel_t),
            "arch": axis_point(arch_t),
        },
    }


# --------------------------------------------------------------------------
# 4. Reference detection
# --------------------------------------------------------------------------

def _detect_quarter(gray: np.ndarray, foot_contour, foot_length_px: float):
    """HoughCircles; pick the in-range circle. Returns diameter px or None."""
    blur = cv2.medianBlur(gray, 5)
    min_r = int(foot_length_px * QUARTER_FRAC_MIN / 2)
    max_r = int(foot_length_px * QUARTER_FRAC_MAX / 2)
    if max_r <= min_r or min_r < 3:
        return None
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=max(20, int(foot_length_px * 0.05)),
        param1=100, param2=25,
        minRadius=min_r, maxRadius=max_r,
    )
    if circles is None:
        return None
    best = None
    for x, y, r in circles[0]:
        d = float(2 * r)
        if not (QUARTER_FRAC_MIN * foot_length_px <= d <= QUARTER_FRAC_MAX * foot_length_px):
            continue
        # Reject circles sitting on the foot itself.
        if cv2.pointPolygonTest(foot_contour, (float(x), float(y)), False) >= 0:
            continue
        if best is None or r > best[2]:
            best = (x, y, r)
    return float(2 * best[2]) if best else None


def _detect_card(gray: np.ndarray, foot_contour, foot_area: float, img_area: float):
    """Quadrilateral with credit-card aspect ratio. Returns long-side px or None."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        if peri < 40:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        area = cv2.contourArea(approx)
        if not (0.002 * img_area <= area <= 0.35 * img_area):
            continue
        if area > 0.35 * foot_area:
            continue  # a card is far smaller than a foot
        (_, _), (rw, rh), _ = cv2.minAreaRect(approx)
        if min(rw, rh) < 1e-6:
            continue
        aspect = max(rw, rh) / min(rw, rh)
        if abs(aspect - CARD_ASPECT) > CARD_ASPECT_TOL:
            continue
        cx, cy = float(approx[:, 0, 0].mean()), float(approx[:, 0, 1].mean())
        if cv2.pointPolygonTest(foot_contour, (cx, cy), False) >= 0:
            continue
        long_side = float(max(rw, rh))
        if best is None or long_side > best:
            best = long_side
    return best


# --------------------------------------------------------------------------
# 6. Full scan
# --------------------------------------------------------------------------

def process_scan(image_bytes: bytes, foot_side: str = "left", reference: str = "none") -> dict:
    """Run the whole pipeline. Returns drawing-ready geometry + measurements.

    The returned `outline` / `centerline` / `stations` are in drawing units:
    millimeters when calibrated, otherwise arbitrary units (= original px).
    `outline_px` is always the simplified contour in original-image px.
    All `*_mm` measurements are None when unscaled.
    """
    if foot_side not in ("left", "right"):
        raise ScanError("foot_side must be 'left' or 'right'.")
    if reference not in ("quarter", "card", "none"):
        raise ScanError("reference must be 'quarter', 'card', or 'none'.")

    # 1+2. Strip EXIF (verified), decode, downscale.
    clean = strip_exif(image_bytes)
    bgr = _decode_bgr(clean)
    h0, w0 = bgr.shape[:2]
    scale = min(1.0, MAX_WORK_DIM / max(h0, w0))
    inv = 1.0 / scale
    work = bgr if scale >= 1.0 else cv2.resize(
        bgr, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)

    # 3. Segment.
    contour, seg_method = segment_foot(work)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

    # 5. Landmarks (working px).
    lm = compute_landmarks(contour)
    foot_area = float(cv2.contourArea(contour))
    img_area = float(work.shape[0] * work.shape[1])
    (_, _), (frw, frh), _ = cv2.minAreaRect(contour)
    aspect = float(max(frw, frh) / max(min(frw, frh), 1e-6))

    # 4. Reference -> px_per_mm (in ORIGINAL image px).
    px_per_mm = None
    ref_note = "none requested"
    if reference == "quarter":
        d_work = _detect_quarter(gray, contour, lm["length_px"])
        if d_work:
            px_per_mm = (d_work * inv) / QUARTER_DIAMETER_MM
            ref_note = f"quarter d={d_work * inv:.1f}px"
        else:
            ref_note = "quarter requested but not detected"
    elif reference == "card":
        long_work = _detect_card(gray, contour, foot_area, img_area)
        if long_work:
            px_per_mm = (long_work * inv) / CARD_LONG_MM
            ref_note = f"card long_side={long_work * inv:.1f}px"
        else:
            ref_note = "card requested but not detected"

    calibrated = px_per_mm is not None and px_per_mm > 0
    to_units = (lambda px: px * inv / px_per_mm) if calibrated else (lambda px: px * inv)
    units = "mm" if calibrated else "arb"

    # Simplified outline: approxPolyDP in drawing units + original px.
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.005 * peri, True).reshape(-1, 2)
    outline = [[to_units(float(x)), to_units(float(y))] for x, y in approx]
    outline_px = [[round(float(x) * inv, 1), round(float(y) * inv, 1)] for x, y in approx]

    def conv(pt):
        return [to_units(float(pt[0])), to_units(float(pt[1]))]

    centerline = {"heel": conv(lm["heel_pt"]), "toe": conv(lm["toe_pt"])}
    stations = {k: conv(v) for k, v in lm["stations"].items()}

    if calibrated:
        length_mm = lm["length_px"] * inv / px_per_mm
        ball_mm = lm["ball_px"] * inv / px_per_mm
        heel_mm = lm["heel_px"] * inv / px_per_mm
        arch_mm = lm["arch_px"] * inv / px_per_mm
        measurements = {
            "length_mm": round(length_mm, 1),
            "ball_width_mm": round(ball_mm, 1),
            "heel_width_mm": round(heel_mm, 1),
            "arch_width_mm": round(arch_mm, 1),
            "length_in": round(length_mm / MM_PER_INCH, 2),
            "calibrated": True,
            "reference": reference,
            "px_per_mm": round(px_per_mm, 4),
        }
    else:
        measurements = {
            "length_mm": None,
            "ball_width_mm": None,
            "heel_width_mm": None,
            "arch_width_mm": None,
            "length_in": None,
            "calibrated": False,
            "reference": reference,
            "px_per_mm": None,
        }

    warnings = [ESTIMATED_WARNING]
    if not calibrated:
        if reference == "none":
            warnings.append("UNSCALED \u2014 no reference used; shape only, no dimensions.")
        else:
            warnings.append(
                f"UNSCALED \u2014 reference '{reference}' not detected; "
                "shape only, no dimensions.")

    debug = {
        "segmentation": seg_method,
        "foot_area_frac": round(foot_area / img_area, 4),
        "min_area_rect_aspect": round(aspect, 2),
        "contour_points": len(outline),
        "reference_note": ref_note,
        "units": units,
    }

    return {
        "outline": outline,
        "outline_px": outline_px,
        "centerline": centerline,
        "stations": stations,
        "units": units,
        "measurements": measurements,
        "calibrated": calibrated,
        "px_per_mm": px_per_mm,
        "foot_side": foot_side,
        "reference": reference,
        "warnings": warnings,
        "debug": debug,
    }
