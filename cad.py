"""ModToes Foot Scanner v3 — CAD renderer.

Shared drawing model: one FootDrawing produces both a standalone SVG string
(to_svg) and PNG bytes rendered with Pillow only (to_png). No extra
dependencies (no cairosvg etc.).

PRIVACY-BY-DESIGN:
- This module only ever sees geometry (polylines, numbers). It never sees
  the photo. Nothing here touches disk.
- No precision claims beyond the method: no "certified", no medical claims.
  Every drawing carries the ESTIMATED footer warning.

Layout (drawing units; mm when calibrated, arbitrary units otherwise):
  - white background
  - foot outline polyline, dark navy #1a2b4a, 2px
  - centerline heel->toe, chain-thin dash pattern
  - dimension lines + arrowheads + labels (CALIBRATED ONLY)
  - title block: MODTOES FOOT SCAN, date UTC, LEFT/RIGHT, reference, scale note
  - footer: "ESTIMATED — VERIFY WITH PHYSICAL MEASUREMENT. Not medical advice."
    in red
"""

from __future__ import annotations

import datetime
import io
import math
import xml.sax.saxutils as sax

from PIL import Image, ImageDraw, ImageFont

NAVY = "#1a2b4a"
NAVY_RGB = (26, 43, 74)
DIM_GRAY = "#555555"
DIM_RGB = (85, 85, 85)
WARN_RED = "#a31621"
WARN_RGB = (163, 22, 33)
BLACK = "#111111"

SANS_STACK = "DejaVu Sans, Helvetica, Arial, sans-serif"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

FOOTER_WARNING = "ESTIMATED \u2014 VERIFY WITH PHYSICAL MEASUREMENT. Not medical advice."
REFERENCE_LABELS = {
    "quarter": "US quarter (24.26 mm)",
    "card": "credit card (85.60 \u00d7 53.98 mm)",
    "none": "none",
}


def _rot_up(points, heel, toe):
    """Rotate so heel->toe points straight up; return (pts, heel, toe)."""
    dx, dy = toe[0] - heel[0], toe[1] - heel[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return list(points), list(heel), list(toe)
    phi = -math.pi / 2 - math.atan2(dy, dx)
    c, s = math.cos(phi), math.sin(phi)

    def rot(p):
        x, y = p[0] - heel[0], p[1] - heel[1]
        return [x * c - y * s, x * s + y * c]

    return [rot(p) for p in points], rot(heel), rot(toe)


def _edges_at_y(outline, y_s, tol):
    xs = [p[0] for p in outline if abs(p[1] - y_s) <= tol]
    if len(xs) < 2:
        xs = [p[0] for p in outline]
    return min(xs), max(xs)


class FootDrawing:
    """Drawing model built from pipeline.process_scan() output."""

    def __init__(self, outline, centerline, stations, dims, foot_side,
                 reference, calibrated, units, ts_utc):
        # Rotate so the foot stands heel-down / toe-up.
        outline, heel, toe = _rot_up(outline, centerline["heel"], centerline["toe"])
        stations = {k: _rot_up([v], centerline["heel"], centerline["toe"])[0][0]
                    for k, v in stations.items()}

        xs = [p[0] for p in outline]
        ys = [p[1] for p in outline]
        self.x0, self.x1 = min(xs), max(xs)
        self.y0, self.y1 = min(ys), max(ys)  # y0 top (toe), y1 bottom (heel)
        self.w = self.x1 - self.x0
        self.h = self.y1 - self.y0

        span = max(self.w, self.h, 1e-9)
        self.margin = max(8.0, 0.06 * span)
        self.W = self.w + 2 * self.margin
        self.ox = self.margin - self.x0   # translate content into canvas
        self.oy = self.margin - self.y0

        self.outline = [[p[0] + self.ox, p[1] + self.oy] for p in outline]
        self.heel = [heel[0] + self.ox, heel[1] + self.oy]
        self.toe = [toe[0] + self.ox, toe[1] + self.oy]
        self.stations = {k: [v[0] + self.ox, v[1] + self.oy] for k, v in stations.items()}

        self.dims = dims or {}            # {"length": mm, "ball": mm, ...} or {}
        self.foot_side = foot_side
        self.reference = reference
        self.calibrated = calibrated
        self.units = units
        self.ts_utc = ts_utc
        self.arrow = max(1.6, 0.016 * span)

        if calibrated:
            ref = REFERENCE_LABELS.get(reference, reference)
            self.scale_note = f"Scale: 1 unit = 1 mm (from {ref})"
        else:
            self.scale_note = "UNSCALED \u2014 FOR SHAPE ONLY"

        # ---- typography sized to fit the canvas width (canvas can be narrow
        # ---- for a tall foot; never let text overflow) ----
        title_lines = self._title_lines()
        avail = self.W - 1.4 * self.margin
        longest = max([len(FOOTER_WARNING)] + [len(l) for l in title_lines])
        fs_default = max(3.0, 0.028 * span)
        # DejaVu Sans averages ~0.6 * fontsize per character.
        self.fs_body = min(fs_default, avail / (longest * 0.60))
        self.fs_title = self.fs_body * 1.35
        self.fs_foot = min(self.fs_body, avail / (len(FOOTER_WARNING) * 0.60))

        # ---- vertical layout (canvas coords; drawing occupies margin..margin+h)
        draw_bottom = self.margin + self.h
        self.step = max(7.0, 0.045 * self.h, 2.6 * self.fs_body)
        if calibrated:
            self.stack_top = draw_bottom + self.margin * 0.4
            stack_bottom = self.stack_top + 2 * self.step + self.fs_body
        else:
            self.stack_top = 0.0
            stack_bottom = draw_bottom
        line_h = self.fs_body * 1.55
        self.tb_h = 4 * line_h + self.fs_body * 0.9   # 4 title lines + padding
        self.tb_y = stack_bottom + self.margin * 0.35
        self.ft_h = self.fs_foot * 2.4
        self.H = self.tb_y + self.tb_h + self.ft_h

    # -- construction -----------------------------------------------------

    @classmethod
    def from_scan(cls, scan: dict, ts_utc: str | None = None):
        m = scan["measurements"]
        dims = {}
        if scan["calibrated"]:
            dims = {
                "length": m["length_mm"],
                "ball": m["ball_width_mm"],
                "heel": m["heel_width_mm"],
                "arch": m["arch_width_mm"],
            }
        ts = ts_utc or datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC")
        return cls(
            outline=scan["outline"],
            centerline=scan["centerline"],
            stations=scan["stations"],
            dims=dims,
            foot_side=scan["foot_side"].upper(),
            reference=scan["reference"],
            calibrated=scan["calibrated"],
            units=scan["units"],
            ts_utc=ts,
        )

    # -- shared helpers ---------------------------------------------------

    def _dim_specs(self):
        """Dimension line specs (calibrated only)."""
        if not self.calibrated:
            return []
        specs = []
        # Length: vertical dim line right of the foot.
        x_d = self.x1 + self.ox + self.margin * 0.45
        specs.append({
            "kind": "length", "label": f"{self.dims['length']:.1f} mm",
            "x1": x_d, "y1": self.toe[1], "x2": x_d, "y2": self.heel[1],
            "ext": [((self.toe[0] + self.heel[0]) / 2, self.toe[1], x_d + self.arrow),
                    ((self.toe[0] + self.heel[0]) / 2, self.heel[1], x_d + self.arrow)],
            "label_pos": (x_d + self.arrow * 1.6, (self.toe[1] + self.heel[1]) / 2),
            "rotate_label": True,
        })
        # Widths: stacked horizontal dim lines below the drawing.
        base_y = self.stack_top
        step = self.step
        tol = max(1.0, 0.02 * self.h)
        for i, (key, name) in enumerate(
                (("ball", "ball width"), ("heel", "heel width"), ("arch", "arch width"))):
            y_s = self.stations[key][1]
            x_l, x_r = _edges_at_y(self.outline, y_s, tol)
            y_d = base_y + i * step
            specs.append({
                "kind": "width", "label": f"{self.dims[key]:.1f} mm \u2014 {name}",
                "x1": x_l, "y1": y_d, "x2": x_r, "y2": y_d,
                "ext": [(x_l, y_s, y_d), (x_r, y_s, y_d)],
                "label_pos": ((x_l + x_r) / 2, y_d - self.arrow * 0.9),
                "rotate_label": False,
            })
        return specs

    def _title_lines(self):
        return [
            "MODTOES FOOT SCAN",
            f"Date (UTC): {self.ts_utc}",
            f"Foot: {self.foot_side}    Reference: "
            f"{REFERENCE_LABELS.get(self.reference, self.reference)}",
            self.scale_note,
        ]

    # -- SVG --------------------------------------------------------------

    def to_svg(self) -> str:
        e = sax.escape
        W, H = self.W, self.H
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.2f} {H:.2f}" '
            f'width="{W:.1f}mm" height="{H:.1f}mm" font-family="{SANS_STACK}">',
            f'<rect x="0" y="0" width="{W:.2f}" height="{H:.2f}" fill="white"/>',
        ]

        dim_specs = self._dim_specs()
        if dim_specs:
            # Arrowhead marker defs — only emitted when dimensions exist.
            parts.insert(1, "</defs>")
            parts.insert(1,
                f'<marker id="arr" markerWidth="{self.arrow * 2:.1f}" '
                f'markerHeight="{self.arrow * 2:.1f}" refX="{self.arrow * 1.7:.1f}" '
                f'refY="{self.arrow:.1f}" orient="auto" markerUnits="userSpaceOnUse">'
                f'<path d="M0,0 L{self.arrow * 2:.1f},{self.arrow:.1f} L0,{self.arrow * 2:.1f} z" '
                f'fill="{DIM_GRAY}"/></marker>')
            parts.insert(1, "<defs>")

        # Foot outline.
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in self.outline)
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{NAVY}" stroke-width="2" '
            f'stroke-linejoin="round"/>')

        # Centerline, chain-thin dash pattern.
        parts.append(
            f'<line x1="{self.heel[0]:.2f}" y1="{self.heel[1]:.2f}" '
            f'x2="{self.toe[0]:.2f}" y2="{self.toe[1]:.2f}" stroke="{NAVY}" '
            f'stroke-width="0.8" stroke-dasharray="12,3,3,3" opacity="0.65"/>')

        # Dimensions (calibrated only).
        fs = max(3.0, 0.028 * max(self.w, self.h))
        for d in dim_specs:
            for ex in d["ext"]:
                if d["kind"] == "length":
                    x0, y0, x1 = ex
                    parts.append(
                        f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y0:.2f}" '
                        f'stroke="{DIM_GRAY}" stroke-width="0.6"/>')
                else:
                    x0, y0, y1 = ex
                    parts.append(
                        f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x0:.2f}" y2="{y1:.2f}" '
                        f'stroke="{DIM_GRAY}" stroke-width="0.6"/>')
            parts.append(
                f'<line x1="{d["x1"]:.2f}" y1="{d["y1"]:.2f}" x2="{d["x2"]:.2f}" '
                f'y2="{d["y2"]:.2f}" stroke="{DIM_GRAY}" stroke-width="1" '
                f'marker-start="url(#arr)" marker-end="url(#arr)"/>')
            lx, ly = d["label_pos"]
            if d["rotate_label"]:
                parts.append(
                    f'<text x="{lx:.2f}" y="{ly:.2f}" font-size="{fs:.1f}" fill="{BLACK}" '
                    f'text-anchor="middle" transform="rotate(-90 {lx:.2f} {ly:.2f}">'
                    f'{e(d["label"])}</text>')
            else:
                parts.append(
                    f'<text x="{lx:.2f}" y="{ly:.2f}" font-size="{fs:.1f}" fill="{BLACK}" '
                    f'text-anchor="middle">{e(d["label"])}</text>')

        # Title block (full canvas width so text always fits).
        tb_x = self.margin * 0.5
        tb_w = self.W - self.margin
        parts.append(
            f'<rect x="{tb_x:.2f}" y="{self.tb_y:.2f}" '
            f'width="{tb_w:.2f}" height="{self.tb_h:.2f}" fill="none" '
            f'stroke="{NAVY}" stroke-width="1"/>')
        lines = self._title_lines()
        line_h = self.fs_body * 1.55
        for i, line in enumerate(lines):
            y = self.tb_y + line_h * (i + 0.85)
            weight = "bold" if i == 0 else "normal"
            size = self.fs_title if i == 0 else self.fs_body
            parts.append(
                f'<text x="{self.W / 2:.2f}" y="{y:.2f}" '
                f'font-size="{size:.2f}" font-weight="{weight}" fill="{NAVY}" '
                f'text-anchor="middle">{e(line)}</text>')

        # Footer warning.
        fy = self.tb_y + self.tb_h + self.ft_h * 0.62
        parts.append(
            f'<text x="{W / 2:.2f}" y="{fy:.2f}" font-size="{self.fs_foot:.2f}" '
            f'font-weight="bold" fill="{WARN_RED}" text-anchor="middle">'
            f'{e(FOOTER_WARNING)}</text>')

        parts.append("</svg>")
        return "\n".join(parts)

    # -- PNG (Pillow only) ------------------------------------------------

    def to_png(self, dpi: int = 150) -> bytes:
        if self.units == "mm":
            px_per_unit = dpi / 25.4
        else:
            px_per_unit = 1400.0 / max(self.W, self.H, 1e-9)
        S = px_per_unit
        PW, PH = max(1, int(self.W * S)), max(1, int(self.H * S))

        img = Image.new("RGB", (PW, PH), "white")
        dr = ImageDraw.Draw(img)
        try:
            f_title = ImageFont.truetype(FONT_BOLD_PATH, max(8, int(self.fs_title * S)))
            f_body = ImageFont.truetype(FONT_PATH, max(8, int(self.fs_body * S)))
            f_foot = ImageFont.truetype(FONT_PATH, max(8, int(self.fs_foot * S)))
            f_dim = ImageFont.truetype(FONT_PATH, int(max(8, 3.2 * S)))
        except OSError:
            f_title = f_body = f_foot = f_dim = ImageFont.load_default()

        def P(p):
            return (p[0] * S, p[1] * S)

        # Foot outline.
        dr.line([P(p) for p in self.outline] + [P(self.outline[0])],
                fill=NAVY_RGB, width=max(2, int(0.35 * S)), joint="curve")

        # Centerline, chain-thin dash: long, gap, short, gap.
        x0, y0 = P(self.heel)
        x1, y1 = P(self.toe)
        segs = _dash_segments((x0, y0), (x1, y1),
                              [10 * S, 3 * S, 2.5 * S, 3 * S])
        for a, b in segs:
            dr.line([a, b], fill=NAVY_RGB, width=max(1, int(0.14 * S)))

        # Dimensions.
        for d in self._dim_specs():
            for ex in d["ext"]:
                if d["kind"] == "length":
                    xa, ya, xb = ex
                    dr.line([P((xa, ya)), P((xb, ya))], fill=DIM_RGB, width=1)
                else:
                    xa, ya, yb = ex
                    dr.line([P((xa, ya)), P((xa, yb))], fill=DIM_RGB, width=1)
            a, b = P((d["x1"], d["y1"])), P((d["x2"], d["y2"]))
            dr.line([a, b], fill=DIM_RGB, width=2)
            _arrowhead(dr, a, b, self.arrow * S, DIM_RGB)
            _arrowhead(dr, b, a, self.arrow * S, DIM_RGB)
            lx, ly = d["label_pos"][0] * S, d["label_pos"][1] * S
            if d["rotate_label"]:
                timg = Image.new("RGBA", (int(30 * S), int(8 * S)), (255, 255, 255, 0))
                td = ImageDraw.Draw(timg)
                td.text((4, 2), d["label"], font=f_dim, fill=(17, 17, 17, 255))
                timg = timg.rotate(90, expand=True)
                img.paste(timg, (int(lx - timg.width / 2), int(ly - timg.height / 2)), timg)
            else:
                dr.text((lx, ly), d["label"], font=f_dim, fill=(17, 17, 17),
                        anchor="ms")

        # Title block (full canvas width).
        tb_y = self.tb_y * S
        tb_x = self.margin * 0.5 * S
        dr.rectangle([tb_x, tb_y, (self.W - self.margin * 0.5) * S,
                      tb_y + self.tb_h * S],
                     outline=NAVY_RGB, width=2)
        line_h = self.fs_body * 1.55 * S
        for i, line in enumerate(self._title_lines()):
            y = tb_y + line_h * (i + 0.85)
            dr.text((PW / 2, y), line,
                    font=f_title if i == 0 else f_body,
                    fill=NAVY_RGB, anchor="mm")

        # Footer warning.
        fy = tb_y + self.tb_h * S + self.ft_h * S * 0.55
        dr.text((PW / 2, fy), FOOTER_WARNING, font=f_foot,
                fill=WARN_RGB, anchor="mm")

        buf = io.BytesIO()
        img.save(buf, format="PNG")  # Pillow writes no EXIF to PNG by default
        return buf.getvalue()


def _dash_segments(a, b, pattern):
    """Split segment a->b into 'on' sub-segments following a dash pattern."""
    import math as _m
    dx, dy = b[0] - a[0], b[1] - a[1]
    total = _m.hypot(dx, dy)
    if total < 1e-9:
        return []
    ux, uy = dx / total, dy / total
    segs, pos, i, on = [], 0.0, 0, True
    while pos < total:
        ln = pattern[i % len(pattern)]
        nxt = min(total, pos + ln)
        if on and nxt > pos:
            segs.append(((a[0] + ux * pos, a[1] + uy * pos),
                         (a[0] + ux * nxt, a[1] + uy * nxt)))
        pos = nxt
        i += 1
        on = not on
    return segs


def _arrowhead(dr, tip, tail, size, color):
    import math as _m
    dx, dy = tip[0] - tail[0], tip[1] - tail[1]
    ang = _m.atan2(dy, dx)
    s = max(3.0, size)
    p1 = (tip[0] - s * _m.cos(ang - 0.42), tip[1] - s * _m.sin(ang - 0.42))
    p2 = (tip[0] - s * _m.cos(ang + 0.42), tip[1] - s * _m.sin(ang + 0.42))
    dr.polygon([tip, p1, p2], fill=color)
