#!/usr/bin/env python3
"""Generate the ICLR 2026 Figure 1 teaser (Lingua-SafetyBench view).

Scatter: x = multilingual comprehension (puzzle reconstruction),
         y = reachable jailbreak (oracle-verified ASR).
One marker per model, colored by family, area ~ parameter count,
translucent family hull behind each cluster, compact family->lever legend.

Output: paper/figures/fig_intro_teaser.svg
"""
import math
import os

# ---------------------------------------------------------------- data
# model, family, params(B), comprehension x, jailbreak y, lever
DATA = [
    ("Qwen-3B",      "Qwen",  3,  0.62, 0.65, "fiction"),
    ("Qwen-7B",      "Qwen",  7,  0.86, 0.75, "role separation"),
    ("Qwen-14B",     "Qwen",  14, 0.98, 0.90, "role separation"),
    ("Qwen-32B",     "Qwen",  32, 0.98, 0.90, "role separation"),
    ("Llama-3.2-3B", "Llama", 3,  0.61, 0.55, "multilingual"),
    ("Llama-3.1-8B", "Llama", 8,  0.77, 0.78, "multilingual"),
    ("Gemma-2B",     "Gemma", 2,  0.48, 0.78, "persona"),
    ("Gemma-9B",     "Gemma", 9,  0.94, 0.93, "persona"),
    ("Gemma-27B",    "Gemma", 27, 0.97, 0.98, "persona"),
]

FAM_COLOR = {"Qwen": "#1b6ca8", "Llama": "#ef8a3f", "Gemma": "#2e8b57"}
# a slightly deeper stroke tone per family for marker outlines / text
FAM_DARK = {"Qwen": "#12496f", "Llama": "#c25e18", "Gemma": "#1e5d3a"}
FAM_LEVER = {
    "Qwen": "role separation",
    "Llama": "multilingual / amount",
    "Gemma": "persona",
}

# ---------------------------------------------------------------- canvas
W, H = 1100, 520
FONT = "DejaVu Sans, Helvetica, Arial, sans-serif"

# plot rect
PX0, PY0 = 96, 80        # top-left of plot area
PW, PH = 648, 350        # plot width / height
PX1, PY1 = PX0 + PW, PY0 + PH

# axis data ranges (padded so big markers never clip the frame)
XMIN, XMAX = 0.40, 1.02
YMIN, YMAX = 0.48, 1.02


def sx(x):
    return PX0 + (x - XMIN) / (XMAX - XMIN) * PW


def sy(y):
    return PY1 - (y - YMIN) / (YMAX - YMIN) * PH


def radius(p):
    return 5.4 + math.sqrt(p) * 3.15


# ---------------------------------------------------------------- svg helpers
def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def txt(x, y, s, size, fill, anchor="start", weight="normal",
        style="normal", opacity=1.0, ls=None):
    extra = f' letter-spacing="{ls}"' if ls else ""
    op = f' opacity="{opacity}"' if opacity != 1.0 else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" '
            f'font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}" font-style="{style}"'
            f'{op}{extra} dominant-baseline="middle">{esc(s)}</text>')


def convex_hull(pts):
    pts = sorted(set(pts))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def expand(hull, pad):
    cx = sum(p[0] for p in hull) / len(hull)
    cy = sum(p[1] for p in hull) / len(hull)
    out = []
    for (x, y) in hull:
        dx, dy = x - cx, y - cy
        d = math.hypot(dx, dy) or 1.0
        out.append((x + dx / d * pad, y + dy / d * pad))
    return out


# ---------------------------------------------------------------- build
parts = []
parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" '
             f'height="{H}" viewBox="0 0 {W} {H}">')
parts.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

# ---- title ----
parts.append(txt(PX0 - 8, 22,
                 "Comprehension and reachable jailbreak vary across LLMs —",
                 15.5, "#0f172a", weight="bold"))
parts.append(txt(PX0 - 8, 43,
                 "yet the winning attack lever is consistent within each family",
                 15.5, "#0f172a", weight="bold"))
parts.append(txt(PX0 - 8, 63,
                 "Lingua-SafetyBench: one marker per model, area ∝ parameters; "
                 "shaded region = model family",
                 12, "#64748b"))

# ---- plot frame + grid ----
# faint diagonal "larger models" trend cue (very subtle, behind grid)
parts.append(f'<line x1="{sx(0.50):.1f}" y1="{sy(0.55):.1f}" '
             f'x2="{sx(0.99):.1f}" y2="{sy(0.99):.1f}" stroke="#cbd5e1" '
             f'stroke-width="10" stroke-linecap="round" opacity="0.20"/>')

parts.append(f'<rect x="{PX0}" y="{PY0}" width="{PW}" height="{PH}" '
             f'fill="#fbfcfd" stroke="#cbd5e1" stroke-width="1.4"/>')

# grid lines
for gx in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    xp = sx(gx)
    parts.append(f'<line x1="{xp:.1f}" y1="{PY0}" x2="{xp:.1f}" y2="{PY1}" '
                 f'stroke="#e6eaef" stroke-width="1"/>')
    parts.append(txt(xp, PY1 + 16, f"{gx:.1f}", 11.5, "#64748b", anchor="middle"))
for gy in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    yp = sy(gy)
    parts.append(f'<line x1="{PX0}" y1="{yp:.1f}" x2="{PX1}" y2="{yp:.1f}" '
                 f'stroke="#e6eaef" stroke-width="1"/>')
    parts.append(txt(PX0 - 9, yp, f"{gy:.1f}", 11.5, "#64748b", anchor="end"))

# ---- axis titles ----
parts.append(txt((PX0 + PX1) / 2, PY1 + 38,
                 "multilingual comprehension  (puzzle reconstruction →)",
                 13.5, "#334155", anchor="middle", weight="bold"))
parts.append(
    f'<g transform="translate({PX0 - 52},{(PY0 + PY1) / 2}) rotate(-90)">'
    + txt(0, 0, "reachable jailbreak  (oracle-verified ASR →)",
          13.5, "#334155", anchor="middle", weight="bold")
    + '</g>')

# ---- family hulls (behind markers) ----
byfam = {}
for row in DATA:
    byfam.setdefault(row[1], []).append(row)

for fam, rows in byfam.items():
    pts = [(sx(r[3]), sy(r[4])) for r in rows]
    col = FAM_COLOR[fam]
    uniq = sorted(set(pts))
    if len(uniq) >= 3:
        hull = expand(convex_hull(uniq), 34)
        d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in hull) + " Z"
        # translucent fill + thick round-joined stroke => soft rounded blob
        parts.append(f'<path d="{d}" fill="{col}" fill-opacity="0.10" '
                     f'stroke="{col}" stroke-width="30" '
                     f'stroke-opacity="0.10" stroke-linejoin="round"/>')
        parts.append(f'<path d="{d}" fill="none" stroke="{col}" '
                     f'stroke-width="1.3" stroke-opacity="0.45" '
                     f'stroke-linejoin="round" stroke-dasharray="2 4"/>')
    else:
        # 2-point family -> capsule via fat round-capped line
        (x1, y1), (x2, y2) = uniq
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" '
                     f'y2="{y2:.1f}" stroke="{col}" stroke-width="62" '
                     f'stroke-opacity="0.11" stroke-linecap="round"/>')
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" '
                     f'y2="{y2:.1f}" stroke="{col}" stroke-width="62" '
                     f'stroke-opacity="0.0"/>')

# family name watermark near each cluster
fam_label_pos = {
    "Gemma": (sx(0.545), sy(1.005)),
    "Qwen":  (sx(0.905), sy(0.665)),
    "Llama": (sx(0.545), sy(0.545)),
}
for fam, (lx, ly) in fam_label_pos.items():
    parts.append(txt(lx, ly, fam, 15, FAM_COLOR[fam], anchor="middle",
                     weight="bold", opacity=0.85))

# ---- markers ----
# explicit label placement to avoid overlap: (dx, dy, anchor)
LBL = {
    "Qwen-3B":      (0, 1),      # below
    "Qwen-7B":      (1, 0),      # right
    "Llama-3.2-3B": (0, 1),
    "Llama-3.1-8B": (-1, 0),     # left
    "Gemma-2B":     (0, -1),     # above
    "Gemma-9B":     (-1, 0),     # left
    "Gemma-27B":    (-1, 0),     # left (avoid legend panel)
}


def draw_marker(row, front=True):
    name, fam, p, x, y, lever = row
    cx, cy = sx(x), sy(y)
    r = radius(p)
    col = FAM_COLOR[fam]
    dk = FAM_DARK[fam]
    parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" '
                 f'fill="{col}" fill-opacity="0.88" stroke="#ffffff" '
                 f'stroke-width="1.8"/>')
    parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" '
                 f'fill="none" stroke="{dk}" stroke-width="1.2" '
                 f'stroke-opacity="0.65"/>')


# draw the coincident Qwen 14B/32B pair as concentric markers
coincident = [r for r in DATA if (r[3], r[4]) == (0.98, 0.90)]
others = [r for r in DATA if (r[3], r[4]) != (0.98, 0.90)]

for row in others:
    draw_marker(row)

# concentric: larger (32B) behind, smaller (14B) in front
coincident.sort(key=lambda r: -r[2])
for row in coincident:
    draw_marker(row)

# ---- per-model labels ----
for row in others:
    name, fam, p, x, y, lever = row
    cx, cy = sx(x), sy(y)
    r = radius(p)
    dk = FAM_DARK[fam]
    dx, dy = LBL[name]
    gap = 6
    if dx > 0:
        tx, anchor = cx + r + gap, "start"
    elif dx < 0:
        tx, anchor = cx - r - gap, "end"
    else:
        tx, anchor = cx, "middle"
    if dy > 0:
        ty0 = cy + r + gap + 8
    elif dy < 0:
        ty0 = cy - r - gap - 12
    else:
        ty0 = cy - 5
    parts.append(txt(tx, ty0, name, 11.5, "#1f2937", anchor=anchor,
                     weight="bold"))
    parts.append(txt(tx, ty0 + 13, lever, 10.5, dk, anchor=anchor,
                     style="italic"))

# label for the concentric Qwen pair (below, centered - open area)
cx, cy = sx(0.98), sy(0.90)
rmax = radius(32)
ty = cy + rmax + 13
parts.append(txt(cx - 2, ty, "Qwen-14B / -32B", 11.5, "#1f2937",
                 anchor="middle", weight="bold"))
parts.append(txt(cx - 2, ty + 13, "role separation", 10.5, FAM_DARK["Qwen"],
                 anchor="middle", style="italic"))

# ============================================================ legend panel
LX, LY, LW = 786, PY0, 300
parts.append(f'<rect x="{LX}" y="{LY}" width="{LW}" height="{PH}" rx="10" '
             f'fill="#f8fafc" stroke="#e2e8f0" stroke-width="1.2"/>')

cy = LY + 30
parts.append(txt(LX + 18, cy, "Consistent lever per family",
                 13.5, "#0f172a", weight="bold"))
cy += 30
lever_rows = [("Gemma", "persona"),
              ("Qwen", "role separation"),
              ("Llama", "multilingual / amount")]
for fam, lever in lever_rows:
    parts.append(f'<circle cx="{LX + 28}" cy="{cy:.1f}" r="8" '
                 f'fill="{FAM_COLOR[fam]}" stroke="#ffffff" stroke-width="1.6"/>')
    parts.append(txt(LX + 46, cy - 1, fam, 12.5, "#1f2937", weight="bold"))
    parts.append(txt(LX + 46 + 62, cy - 1, "→  " + lever, 12.5,
                     FAM_DARK[fam]))
    cy += 30

# divider
cy += 4
parts.append(f'<line x1="{LX + 18}" y1="{cy:.1f}" x2="{LX + LW - 18}" '
             f'y2="{cy:.1f}" stroke="#e2e8f0" stroke-width="1.2"/>')
cy += 26

# marker-size legend
parts.append(txt(LX + 18, cy, "Marker area ∝ parameters", 12.5,
                 "#334155", weight="bold"))
cy += 30
size_ref = [(3, "3B"), (14, "14B"), (32, "32B")]
xoff = LX + 34
for p, lab in size_ref:
    r = radius(p)
    parts.append(f'<circle cx="{xoff:.1f}" cy="{cy:.1f}" r="{r:.1f}" '
                 f'fill="#94a3b8" fill-opacity="0.85" stroke="#ffffff" '
                 f'stroke-width="1.5"/>')
    parts.append(txt(xoff, cy + r + 12, lab, 10.5, "#64748b", anchor="middle"))
    xoff += r + 46

# divider
cy += 44
parts.append(f'<line x1="{LX + 18}" y1="{cy:.1f}" x2="{LX + LW - 18}" '
             f'y2="{cy:.1f}" stroke="#e2e8f0" stroke-width="1.2"/>')
cy += 22

# takeaway
parts.append(txt(LX + 18, cy, "⇒  Attack must be", 12.5, "#0f172a",
                 weight="bold"))
cy += 18
parts.append(txt(LX + 18, cy, "     per-target &", 12.5, "#0f172a",
                 weight="bold"))
cy += 18
parts.append(txt(LX + 18, cy, "     comprehension-aware.", 12.5, "#b91c1c",
                 weight="bold"))

parts.append('</svg>')

svg = "".join(parts)
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper", "figures", "fig_intro_teaser.svg")
with open(out, "w") as f:
    f.write(svg)
print("wrote", out, len(svg), "bytes")
