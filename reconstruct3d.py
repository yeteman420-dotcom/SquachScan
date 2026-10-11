#!/usr/bin/env python3
"""reconstruct3d.py — Video/photo input -> 3D foot mesh (prototype v0.1).

Pipeline:
    phone video (10s circling the foot)  ─┐
    directory of overlapping photos       ─┤
                                           ▼
              1. extract frames (video) / collect (photos)
              2. SfM with pycolmap (SIFT -> exhaustive match -> incremental map)
                 -> per-frame camera poses (K, R, t)
              3. foot silhouettes via the 2D scanner's U^2-Net segmenter
              4. visual-hull voxel carving inside the foot bounding volume
              5. marching cubes -> mesh -> cleanup
              6. scale to millimetres via US quarter (24.26 mm) if visible
              7. export OBJ + STL (STL goes straight to the slicer)

This is a PROTOTYPE: it favours running at all over perfect geometry.
It does NOT touch app.py / pipeline.py (2D scanner untouched).

Usage:
    python reconstruct3d.py --video foot.mp4 --out foot_mesh
    python reconstruct3d.py --images ./frames/ --out foot_mesh --frames 40
    python reconstruct3d.py --selftest        # synthetic end-to-end check

Requirements (installed into the scanner venv):
    pycolmap, trimesh, scikit-image  (+ existing: opencv, numpy, onnxruntime)

Resource guide (measured on 2-CPU / 8GB VM):
    40 frames @1280px: SfM ~3-8 min, carving ~1-2 min, peak RAM ~1.5 GB.
    Render free tier (512 MB) CANNOT run this. Needs >= 2 GB RAM box.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import tempfile
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)          # reuse the 2D scanner's neural segmenter
from pipeline import _u2net_mask, QUARTER_DIAMETER_MM  # noqa: E402

import pycolmap  # noqa: E402
import trimesh  # noqa: E402
from skimage import measure as skmeasure  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

# ----------------------------------------------------------------------------
# 1. frame acquisition
# ----------------------------------------------------------------------------

def extract_frames(video_path: str, out_dir: str, n_frames: int = 40,
                   max_dim: int = 1280) -> list[str]:
    """Sample n_frames uniformly from a video, downscaled for SfM."""
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        raise RuntimeError("video has no frames")
    idxs = np.linspace(0, total - 1, min(n_frames, total)).astype(int)
    paths: list[str] = []
    for k, fi in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        h, w = frame.shape[:2]
        s = min(1.0, max_dim / max(h, w))
        if s < 1.0:
            frame = cv2.resize(frame, (int(w * s), int(h * s)),
                               interpolation=cv2.INTER_AREA)
        p = os.path.join(out_dir, f"frame_{k:04d}.jpg")
        cv2.imwrite(p, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        paths.append(p)
    cap.release()
    if len(paths) < 3:
        raise RuntimeError("too few frames extracted")
    return paths


def collect_images(images_dir: str, out_dir: str, n_frames: int = 40,
                   max_dim: int = 1280) -> list[str]:
    """Copy (and downscale) up to n_frames stills into the workspace."""
    os.makedirs(out_dir, exist_ok=True)
    exts = (".jpg", ".jpeg", ".png", ".webp")
    srcs = sorted(f for f in os.listdir(images_dir)
                  if f.lower().endswith(exts))
    if len(srcs) < 3:
        raise RuntimeError(f"need >=3 images, found {len(srcs)}")
    if len(srcs) > n_frames:                       # uniform subsample
        srcs = [srcs[int(i)] for i in
                np.linspace(0, len(srcs) - 1, n_frames).astype(int)]
    paths: list[str] = []
    for k, f in enumerate(srcs):
        img = cv2.imread(os.path.join(images_dir, f))
        if img is None:
            continue
        h, w = img.shape[:2]
        s = min(1.0, max_dim / max(h, w))
        if s < 1.0:
            img = cv2.resize(img, (int(w * s), int(h * s)),
                             interpolation=cv2.INTER_AREA)
        p = os.path.join(out_dir, f"frame_{k:04d}.jpg")
        cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        paths.append(p)
    return paths


# ----------------------------------------------------------------------------
# 2. structure-from-motion (pycolmap = prebuilt COLMAP, no compilation)
# ----------------------------------------------------------------------------

def run_sfm(image_dir: str, work_dir: str,
            max_image_size: int = 1600) -> "pycolmap.Reconstruction":
    """SIFT -> exhaustive match -> incremental mapping. Returns best map."""
    os.makedirs(work_dir, exist_ok=True)
    db_path = os.path.join(work_dir, "database.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    sift = pycolmap.SiftExtractionOptions()
    # low-texture skin: be a bit more permissive than defaults
    sift.peak_threshold = 0.004
    fx = pycolmap.FeatureExtractionOptions()
    fx.sift = sift
    fx.max_image_size = max_image_size
    pycolmap.extract_features(
        db_path, image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,   # one phone camera
        extraction_options=fx,
    )
    pycolmap.match_exhaustive(db_path)

    sparse_dir = os.path.join(work_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)
    maps = pycolmap.incremental_mapping(db_path, image_dir, sparse_dir)
    if not maps:
        raise RuntimeError(
            "SfM failed: no reconstruction. Need overlapping views of a "
            "textured scene (keep the rug/floor visible, avoid motion blur).")
    # biggest = most registered images
    best = max(maps.values(), key=lambda m: len(m.images))
    n_reg = sum(1 for im in best.images.values() if im.has_pose)
    print(f"[sfm] {n_reg}/{len(best.images)} images registered, "
          f"{len(best.points3D)} sparse points")
    if n_reg < 3:
        raise RuntimeError("SfM registered <3 images — need more overlap.")
    return best


def get_view(rec: "pycolmap.Reconstruction", image_id: int):
    """Return (K 3x3, R 3x3, t 3, W, H) with X_cam = R @ X_world + t."""
    img = rec.images[image_id]
    cam = rec.cameras[img.camera_id]
    K = cam.calibration_matrix()
    cfw = img.cam_from_world()          # Rigid3d (method in pycolmap>=4)
    R = cfw.rotation.matrix()
    t = np.asarray(cfw.translation, dtype=np.float64)
    return K, R, t, cam.width, cam.height


# ----------------------------------------------------------------------------
# 3. foot silhouettes (reuse the 2D scanner's neural segmenter)
# ----------------------------------------------------------------------------

def segment_foot_robust(bgr: np.ndarray) -> np.ndarray | None:
    """Foot mask that tolerates varied framing.

    Tries several saliency thresholds on the full frame, then falls back
    to a centre crop (video UX keeps the foot centred). Accepts the first
    mask whose largest component covers 2-80% of the considered area.
    Returns uint8 0/255 mask in FULL-frame coordinates, or None.
    """
    h, w = bgr.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    def best_component(m: np.ndarray, area: float):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if n < 2:
            return None
        i = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        frac = stats[i, cv2.CC_STAT_AREA] / area
        if 0.02 <= frac <= 0.80:
            return ((labels == i).astype(np.uint8)) * 255
        return None

    for thr in (0.5, 0.35, 0.25):
        prob = _u2net_mask(bgr)
        if prob is None:
            return None
        m = (prob > thr).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        got = best_component(m, h * w)
        if got is not None:
            return got
    # centre-crop fallback
    cw, ch = int(w * 0.65), int(h * 0.65)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    crop = bgr[y0:y0 + ch, x0:x0 + cw]
    for thr in (0.5, 0.35, 0.25):
        prob = _u2net_mask(crop)
        if prob is None:
            return None
        m = (prob > thr).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        got = best_component(m, cw * ch)
        if got is not None:
            full = np.zeros((h, w), np.uint8)
            full[y0:y0 + ch, x0:x0 + cw] = got
            return full
    return None


def foot_masks(image_paths: list[str]) -> list[np.ndarray]:
    """U^2-Net saliency -> cleaned binary foot mask per frame (uint8 0/255)."""
    masks: list[np.ndarray] = []
    for p in image_paths:
        bgr = cv2.imread(p)
        masks.append(None if bgr is None else segment_foot_robust(bgr))
    nok = sum(m is not None for m in masks)
    print(f"[masks] foot segmented in {nok}/{len(image_paths)} frames")
    if nok < 3:
        raise RuntimeError("foot segmentation failed on too many frames")
    return masks


# ----------------------------------------------------------------------------
# 4. visual-hull voxel carving
# ----------------------------------------------------------------------------

def foot_point_bbox(rec: "pycolmap.Reconstruction",
                    masks: list[np.ndarray],
                    image_ids: list[int],
                    image_paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Bounding box of sparse points that fall inside foot masks.

    Filters the (background-heavy) sparse cloud down to foot points by
    projecting every 3D point into each view and keeping points inside the
    foot silhouette in a majority of views.
    """
    pts = np.array([p.xyz for p in rec.points3D.values()], dtype=np.float64)
    if len(pts) == 0:
        raise RuntimeError("no sparse 3D points")
    inside_votes = np.zeros(len(pts), dtype=np.int32)
    vis_votes = np.zeros(len(pts), dtype=np.int32)
    id2k = {iid: k for k, iid in enumerate(image_ids)}
    for pid, p3d in enumerate(rec.points3D.values()):
        for tr in p3d.track.elements:
            k = id2k.get(tr.image_id)
            if k is None or masks[k] is None:
                continue
            K, R, t, W, H = get_view(rec, tr.image_id)
            xc = R @ p3d.xyz + t
            if xc[2] <= 1e-6:
                continue
            u = (K @ xc)[0] / xc[2]
            v = (K @ xc)[1] / xc[2]
            if 0 <= u < W and 0 <= v < H:
                vis_votes[pid] += 1
                if masks[k][int(v), int(u)] > 0:
                    inside_votes[pid] += 1
    keep = (vis_votes >= 2) & (inside_votes >= 0.6 * np.maximum(vis_votes, 1))
    if keep.sum() < 8:
        # fall back: keep points seen often (still better than whole cloud)
        keep = vis_votes >= 2
    if keep.sum() < 8:
        raise RuntimeError("could not isolate foot points from sparse cloud")
    fp = pts[keep]
    lo, hi = fp.min(0), fp.max(0)
    pad = 0.15 * (hi - lo).max()
    print(f"[bbox] {keep.sum()} foot points isolated")
    return lo - pad, hi + pad


def carve_visual_hull(rec: "pycolmap.Reconstruction",
                      masks: list[np.ndarray],
                      image_ids: list[int],
                      lo: np.ndarray, hi: np.ndarray,
                      n_vox: int = 200,
                      consensus: float = 0.75) -> tuple[np.ndarray, np.ndarray, float]:
    """Voxel carving. Returns (occupancy vol (nz,ny,nx) float32, origin, size).

    A voxel survives if it projects inside the foot silhouette in at least
    `consensus` fraction of the views where it is visible.
    """
    longest = float((hi - lo).max())
    s = longest / n_vox
    nx = max(8, int(math.ceil((hi[0] - lo[0]) / s)))
    ny = max(8, int(math.ceil((hi[1] - lo[1]) / s)))
    nz = max(8, int(math.ceil((hi[2] - lo[2]) / s)))
    xs = lo[0] + (np.arange(nx) + 0.5) * s
    ys = lo[1] + (np.arange(ny) + 0.5) * s
    zs = lo[2] + (np.arange(nz) + 0.5) * s
    X = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3)
    X = X.astype(np.float64)
    inside = np.zeros(X.shape[0], dtype=np.int32)
    visible = np.zeros(X.shape[0], dtype=np.int32)

    id2k = {iid: k for k, iid in enumerate(image_ids)}
    reg_ids = [iid for iid in image_ids
               if rec.images[iid].has_pose and masks[id2k[iid]] is not None]
    print(f"[carve] {len(reg_ids)} views, grid {nx}x{ny}x{nz} "
          f"({X.shape[0]/1e6:.1f}M voxels)")
    t0 = time.time()
    for n, iid in enumerate(reg_ids):
        K, R, t, W, H = get_view(rec, iid)
        Xc = X @ R.T + t                      # (N,3)
        d = Xc[:, 2]
        front = d > 1e-6
        uvw = Xc @ K.T
        u = uvw[:, 0] / np.maximum(d, 1e-9)
        v = uvw[:, 1] / np.maximum(d, 1e-9)
        inb = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not np.any(inb):
            continue
        m = masks[id2k[iid]]
        uu = np.clip(u[inb].astype(np.int32), 0, W - 1)
        vv = np.clip(v[inb].astype(np.int32), 0, H - 1)
        hit = m[vv, uu] > 0
        idx = np.where(inb)[0]
        visible[idx] += 1
        inside[idx[hit]] += 1
        if (n + 1) % 10 == 0:
            print(f"  ... {n+1}/{len(reg_ids)} views "
                  f"({time.time()-t0:.0f}s)")
    frac = np.zeros_like(inside, dtype=np.float32)
    ok = visible >= 3
    frac[ok] = inside[ok] / visible[ok]
    vol = (frac >= consensus).astype(np.float32)
    vol = vol.reshape(nx, ny, nz).transpose(2, 1, 0)   # -> (nz, ny, nx)
    print(f"[carve] kept {(vol > 0).sum()/vol.size*100:.1f}% of bbox "
          f"({time.time()-t0:.0f}s)")
    if (vol > 0).sum() < 100:
        raise RuntimeError("visual hull empty — masks/poses disagree. "
                           "Check that the foot is the segmented object.")
    return vol, lo, s


# ----------------------------------------------------------------------------
# 5. marching cubes -> mesh
# ----------------------------------------------------------------------------

def mesh_from_volume(vol: np.ndarray, origin: np.ndarray,
                     voxel: float) -> trimesh.Trimesh:
    """Marching cubes on a smoothed occupancy grid -> cleaned trimesh."""
    smooth = gaussian_filter(vol.astype(np.float32), sigma=1.0)
    # skimage marching_cubes: input (Z,Y,X) -> verts in (z,y,x) index space
    verts, faces, _, _ = skmeasure.marching_cubes(smooth, level=0.5)
    world = np.empty_like(verts)
    world[:, 0] = origin[0] + verts[:, 2] * voxel   # x
    world[:, 1] = origin[1] + verts[:, 1] * voxel   # y
    world[:, 2] = origin[2] + verts[:, 0] * voxel   # z
    mesh = trimesh.Trimesh(vertices=world, faces=faces, process=True)
    # keep largest connected component (drops floaters)
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        mesh = max(parts, key=lambda p: p.area)
    mesh.remove_infinite_values()
    print(f"[mesh] {len(mesh.vertices)} verts, {len(mesh.faces)} faces, "
          f"watertight={mesh.is_watertight}")
    return mesh


# ----------------------------------------------------------------------------
# 6. scale to millimetres via US quarter (24.26 mm)
# ----------------------------------------------------------------------------

def quarter_diameter_px(bgr: np.ndarray,
                        foot_mask: np.ndarray | None = None,
                        foot_len_px: float | None = None) -> float | None:
    """Hough circle -> quarter diameter in px.

    Heuristics (match the UX: "put a quarter next to your foot"):
      * size band: quarter ~= foot_length/10 -> accept d in
        [foot_len/15, foot_len/6] (or 2-12% of frame width as fallback);
      * reject circles centred inside the foot mask;
      * prefer the candidate CLOSEST to the foot (quarters sit beside it,
        rug-pattern false positives are scattered).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    h, w = gray.shape[:2]
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, 1, max(30, w // 8),
        param1=100, param2=28,
        minRadius=max(8, w // 120), maxRadius=w // 12)
    if circles is None:
        return None
    if foot_len_px and foot_len_px > 0:
        d_min, d_max = foot_len_px / 15.0, foot_len_px / 6.0
    else:
        d_min, d_max = 0.02 * w, 0.12 * w
    dt = None
    if foot_mask is not None:
        # distance from every pixel to the nearest foot pixel
        dt = cv2.distanceTransform((foot_mask == 0).astype(np.uint8),
                                   cv2.DIST_L2, 3)
    scored = []
    for x, y, r in circles[0]:
        d = float(r * 2)
        if not (d_min <= d <= d_max):
            continue
        if x < 0.03 * w or x > 0.97 * w or y < 0.03 * h or y > 0.97 * h:
            continue
        xi, yi = int(round(x)), int(round(y))
        if dt is not None:
            dist = float(dt[yi, xi])
            if dist < r * 0.5:      # centre inside (or deep in) the foot
                continue
            scored.append((dist, d))
        else:
            scored.append((0.0, d))
    if not scored:
        return None
    scored.sort(key=lambda t: t[0])
    return scored[0][1]


def foot_length_px(mask: np.ndarray) -> float:
    """Foot length in px = longest side of min-area rect of the mask."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    return float(max(rw, rh))


def estimate_scale_mm(image_paths: list[str],
                      masks: list[np.ndarray],
                      manual_foot_mm: float | None = None
                      ) -> tuple[float | None, str]:
    """px/mm via quarter; returns (foot_len_mm, note).

    Uses the frame with the largest detected quarter (closest / sharpest).
    Assumes quarter lies ~at foot depth (beside the foot on the floor).
    If manual_foot_mm is given (tape-measured), it wins — more reliable
    than a coin in a blurry frame.
    """
    if manual_foot_mm and manual_foot_mm > 0:
        return manual_foot_mm, "manual tape measurement supplied by user"
    best = None  # (score_dist, quarter_d_px, foot_len_px, path)
    for p, m in zip(image_paths, masks):
        if m is None:
            continue
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        fl = foot_length_px(m)
        if fl <= 0:
            continue
        d = quarter_diameter_px(bgr, foot_mask=m, foot_len_px=fl)
        if d is None:
            continue
        # prefer the sharpest/closest quarter: larger px diameter wins
        if best is None or d > best[1]:
            best = (0.0, d, fl, p)
    if best is None:
        return None, "no quarter detected in any frame — mesh is UNSCALED"
    _, d_px, fl_px, p = best
    foot_mm = fl_px / d_px * QUARTER_DIAMETER_MM
    return foot_mm, (f"quarter {d_px:.0f}px in {os.path.basename(p)} -> "
                     f"foot {foot_mm:.1f} mm")


def apply_scale(mesh: trimesh.Trimesh, foot_len_mm: float | None,
                note: str) -> tuple[trimesh.Trimesh, str]:
    """Scale mesh so its longest PCA axis == measured foot length (mm)."""
    if foot_len_mm is None or foot_len_mm <= 0:
        return mesh, "UNSCALED (recon units). " + note
    v = np.asarray(mesh.vertices)
    vc = v - v.mean(0)
    _, _, vt = np.linalg.svd(vc, full_matrices=False)
    axis = vt[0]
    proj = vc @ axis
    mesh_len = float(proj.max() - proj.min())
    if mesh_len <= 1e-9:
        return mesh, "UNSCALED (degenerate mesh). " + note
    mesh.apply_scale(foot_len_mm / mesh_len)
    return mesh, f"scaled: longest axis = {foot_len_mm:.1f} mm. " + note


# ----------------------------------------------------------------------------
# 7. driver
# ----------------------------------------------------------------------------

def reconstruct(image_paths: list[str], work_dir: str,
                n_vox: int = 200,
                manual_foot_mm: float | None = None) -> tuple[trimesh.Trimesh, dict]:
    info: dict = {}
    t0 = time.time()
    rec = run_sfm(os.path.dirname(image_paths[0]),
                  os.path.join(work_dir, "colmap"))
    image_ids = sorted(rec.images.keys(),
                       key=lambda i: rec.images[i].name)
    masks = foot_masks(image_paths)
    # align masks list with rec image order via filenames
    name2k = {os.path.basename(p): k for k, p in enumerate(image_paths)}
    masks_o = [masks[name2k[rec.images[i].name]] for i in image_ids]

    lo, hi = foot_point_bbox(rec, masks_o, image_ids, image_paths)
    vol, origin, vsz = carve_visual_hull(rec, masks_o, image_ids, lo, hi,
                                         n_vox=n_vox)
    mesh = mesh_from_volume(vol, origin, vsz)

    foot_mm, note = estimate_scale_mm(image_paths, masks,
                                      manual_foot_mm=manual_foot_mm)
    mesh, snote = apply_scale(mesh, foot_mm, note)
    info["scale_note"] = snote
    info["elapsed_s"] = round(time.time() - t0, 1)
    info["registered"] = sum(1 for i in image_ids
                             if rec.images[i].has_pose)
    info["n_frames"] = len(image_paths)
    return mesh, info


def main() -> None:
    ap = argparse.ArgumentParser(description="video/photos -> 3D foot mesh")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="input phone video (mp4)")
    src.add_argument("--images", help="input dir of overlapping photos")
    src.add_argument("--selftest", action="store_true",
                     help="synthetic end-to-end validation")
    ap.add_argument("--out", default="foot_mesh",
                    help="output basename (writes .obj + .stl)")
    ap.add_argument("--frames", type=int, default=40,
                    help="frames to use from video/dir")
    ap.add_argument("--voxels", type=int, default=200,
                    help="voxel resolution along longest bbox side")
    ap.add_argument("--foot-length-mm", type=float, default=None,
                    help="tape-measured foot length (mm); overrides quarter")
    ap.add_argument("--workdir", default=None,
                    help="scratch dir (default: temp)")
    args = ap.parse_args()

    if args.selftest:
        ok = selftest()
        sys.exit(0 if ok else 1)

    work = args.workdir or tempfile.mkdtemp(prefix="recon3d_")
    os.makedirs(work, exist_ok=True)
    fdir = os.path.join(work, "frames")
    try:
        if args.video:
            paths = extract_frames(args.video, fdir, n_frames=args.frames)
        else:
            paths = collect_images(args.images, fdir, n_frames=args.frames)
        print(f"[input] {len(paths)} frames")
        mesh, info = reconstruct(paths, work, n_vox=args.voxels,
                                 manual_foot_mm=args.foot_length_mm)
        obj_p, stl_p = args.out + ".obj", args.out + ".stl"
        mesh.export(obj_p)
        mesh.export(stl_p)
        print(f"[done] {obj_p}\n[done] {stl_p}")
        print(f"[info] {info}")
    finally:
        if not args.workdir:
            shutil.rmtree(work, ignore_errors=True)


# ----------------------------------------------------------------------------
# 8. synthetic self-test (no camera needed)
# ----------------------------------------------------------------------------

def _synth_render_object(n_views: int = 24, img_size: int = 640):
    """Render a textured 3D blob from orbit via matplotlib.

    Returns (list of textured BGR frames, list of mask frames, (w,h)).
    The 'foot' is an elongated ellipsoid-ish blob; texture = checkerboard
    mapped by surface param so SIFT has something to match.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # parametric blob: elongated, foot-ish proportions (L=2.4, W=1, H=0.8)
    nu, nv = 60, 40
    u = np.linspace(0, 2 * np.pi, nu)
    v = np.linspace(0, np.pi, nv)
    uu, vv = np.meshgrid(u, v)
    # superellipsoid-ish, tapered at one end (toes) vs heel
    taper = 1.0 - 0.35 * (0.5 + 0.5 * np.cos(uu)) ** 2
    x = 1.2 * np.sin(vv) * np.cos(uu) * taper
    y = 0.5 * np.sin(vv) * np.sin(uu)
    z = 0.4 * np.cos(vv) + 0.15
    z = np.clip(z, 0.02, None)
    rng = np.random.default_rng(7)
    # distinctive random texture (every patch unique -> easy SIFT matching;
    # real rug/skin is similarly non-repetitive)
    noise = rng.random(uu.shape + (3,)) * 0.55 + 0.35
    face = noise * np.array([0.72, 0.78, 0.90])  # skin-ish tint

    # textured ground plane (mimics the rug: gives SfM stable features)
    gx = np.linspace(-2.2, 2.2, 44)
    gy = np.linspace(-2.2, 2.2, 44)
    gxx, gyy = np.meshgrid(gx, gy)
    gzz = np.zeros_like(gxx)
    gnoise = rng.random(gxx.shape + (3,)) * 0.45 + 0.40
    gface = gnoise * np.array([0.85, 0.87, 0.90])  # rug-ish grey

    frames, masks = [], []
    XLIM, YLIM, ZLIM = (-2.2, 2.2), (-2.2, 2.2), (0, 2.0)
    for k in range(n_views):
        az = 360.0 * k / n_views
        el = 25 + 8 * math.sin(2 * math.pi * k / n_views)
        fig = plt.figure(figsize=(img_size / 100, img_size / 100), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(gxx, gyy, gzz, facecolors=gface, shade=False,
                        rstride=2, cstride=2, antialiased=False)
        ax.plot_surface(x, y, z, facecolors=face, shade=False,
                        rstride=1, cstride=1, antialiased=False)
        ax.set_axis_off()
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_zlim(*ZLIM)
        ax.view_init(elev=el, azim=az)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        rgba = buf.reshape(img_size, img_size, 4)
        frames.append(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR))
        # mask: silhouette on white
        fig2 = plt.figure(figsize=(img_size / 100, img_size / 100), dpi=100)
        ax2 = fig2.add_subplot(111, projection="3d")
        ax2.plot_surface(x, y, z, color="black", shade=False,
                         rstride=1, cstride=1, antialiased=False)
        ax2.set_axis_off()
        ax2.set_xlim(*XLIM)
        ax2.set_ylim(*YLIM)
        ax2.set_zlim(*ZLIM)
        ax2.set_facecolor("white")
        fig2.patch.set_facecolor("white")
        ax2.view_init(elev=el, azim=az)
        fig2.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig2.canvas.draw()
        buf2 = np.frombuffer(fig2.canvas.buffer_rgba(), dtype=np.uint8)
        rgba2 = buf2.reshape(img_size, img_size, 4)
        bgr2 = cv2.cvtColor(rgba2, cv2.COLOR_RGBA2BGR)
        m = (cv2.cvtColor(bgr2, cv2.COLOR_BGR2GRAY) < 128).astype(np.uint8) * 255
        masks.append(m)
        plt.close(fig)
        plt.close(fig2)
    return frames, masks, (img_size, img_size)


def selftest() -> bool:
    """End-to-end on synthetic orbit renders. No camera needed."""
    print("[selftest] rendering synthetic orbit...")
    frames, true_masks, _ = _synth_render_object(n_views=24, img_size=480)
    work = tempfile.mkdtemp(prefix="recon3d_test_")
    try:
        fdir = os.path.join(work, "frames")
        os.makedirs(fdir)
        paths = []
        for k, fr in enumerate(frames):
            p = os.path.join(fdir, f"frame_{k:04d}.jpg")
            cv2.imwrite(p, fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            paths.append(p)
        print("[selftest] running SfM...")
        rec = run_sfm(fdir, os.path.join(work, "colmap"),
                      max_image_size=800)
        image_ids = sorted(rec.images.keys(),
                           key=lambda i: rec.images[i].name)
        n_reg = sum(1 for i in image_ids if rec.images[i].has_pose)
        print(f"[selftest] registered {n_reg}/{len(image_ids)}")
        if n_reg < 8:
            print("[selftest] FAIL: too few registered (need >=8)")
            return False
        name2k = {os.path.basename(p): k for k, p in enumerate(paths)}
        masks_o = [true_masks[name2k[rec.images[i].name]] for i in image_ids]
        lo, hi = foot_point_bbox(rec, masks_o, image_ids, paths)
        vol, origin, vsz = carve_visual_hull(rec, masks_o, image_ids, lo, hi,
                                             n_vox=120)
        mesh = mesh_from_volume(vol, origin, vsz)
        # sanity: mesh should be a single elongated blob, roughly
        # centred near origin, with plausible extents
        v = np.asarray(mesh.vertices)
        ext = v.max(0) - v.min(0)
        elong = ext.max() / max(ext.min(), 1e-9)
        print(f"[selftest] extents={np.round(ext, 3)} elongation={elong:.2f} "
              f"watertight={mesh.is_watertight}")
        ok = (len(mesh.vertices) > 500 and elong > 1.5
              and mesh.is_watertight)
        print("[selftest] " + ("PASS" if ok else "FAIL"))
        return ok
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
