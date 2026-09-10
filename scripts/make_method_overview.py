#!/usr/bin/env python3
"""Illustrated method overview: a harmful request is split into parallel-language fragments and
interleaved into a multilingual jigsaw; a harmless plaintext probe measures how well a blind target
reconstructs such puzzles as they get harder (more languages, more disorder); the search picks the
hardest puzzle the target still solves (plus persona/fiction levers); the target then reconstructs the
request and answers it, safety bypassed. Saves paper/figures/fig_method_overview.{pdf,png}.
Uses only mathtext-safe strings (no LaTeX macros) so the rendered PNG/PDF is faithful."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch, FancyBboxPatch

OUT = Path("paper/figures"); OUT.mkdir(parents=True, exist_ok=True)
LANG = {"EN": "#1b6ca8", "ES": "#2e8b57", "FR": "#8c6d1f", "ZH": "#b23b3b",
        "SW": "#7a5195", "FI": "#ef8a3f", "RU": "#3aa0a0"}


def jigsaw(cx, cy, s, right=1, top=1, left=-1, bottom=-1, tab_r=0.16):
    """Path of a jigsaw piece centred at (cx,cy), side s; edge code +1 tab-out, -1 blank-in, 0 flat."""
    h = s / 2.0; r = tab_r * s
    corners = [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]

    def edge(p0, p1, code):
        (x0, y0), (x1, y1) = p0, p1
        dx, dy = x1 - x0, y1 - y0; L = np.hypot(dx, dy); ux, uy = dx / L, dy / L
        nx, ny = -uy, ux
        pts = [(x0, y0)]
        if code == 0:
            return pts
        a, b = -0.36 * L, 0.36 * L
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        pts.append((mx + a * ux, my + a * uy))
        for t in np.linspace(0, 1, 20):
            along = a + (b - a) * t
            bulge = code * r * np.sin(np.pi * t)
            pts.append((mx + along * ux + bulge * nx, my + along * uy + bulge * ny))
        return pts
    verts = []
    codes = [bottom, right, top, left]
    for i in range(4):
        verts += edge(corners[i], corners[(i + 1) % 4], codes[i])
    verts.append(corners[0])
    return MPath(verts, closed=True)


def piece(ax, cx, cy, s, lang, label="", edges=(1, 1, -1, -1), fs=6.5):
    p = jigsaw(cx, cy, s, right=edges[0], top=edges[1], left=edges[2], bottom=edges[3])
    ax.add_patch(PathPatch(p, facecolor=LANG[lang], edgecolor="white", lw=1.3, alpha=0.93, zorder=2))
    ax.text(cx, cy, label or lang, ha="center", va="center", color="white", fontsize=fs,
            fontweight="bold", zorder=3)


fig, ax = plt.subplots(figsize=(13, 6.0))
ax.set_xlim(0, 13); ax.set_ylim(0, 6.0); ax.axis("off")

# ---------------- (a) request -> parallel fragments ----------------
ax.text(1.7, 5.82, "(a) split & interleave", ha="center", fontsize=10.5, fontweight="bold", color="#333")
ax.add_patch(FancyBboxPatch((0.25, 5.05), 2.9, 0.5, boxstyle="round,pad=0.04", fc="#eef2f5", ec="#888", lw=1.2))
ax.text(1.7, 5.30, "harmful request $q$ (English)", ha="center", va="center", fontsize=8.6, color="#333")
ax.annotate("", xy=(1.7, 4.72), xytext=(1.7, 5.02), arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.5))
for i, lg in enumerate(["EN", "ES", "FR", "ZH"]):
    ax.add_patch(FancyBboxPatch((0.5 + i * 0.62, 4.28), 0.5, 0.34, boxstyle="round,pad=0.02", fc=LANG[lg], ec="white", lw=1))
    ax.text(0.75 + i * 0.62, 4.45, lg, ha="center", va="center", color="white", fontsize=7, fontweight="bold")
ax.text(1.7, 4.06, "parallel translations (no new MT)", ha="center", fontsize=7, color="#555")

# ---------------- (b) the jigsaw, made harder ----------------
ax.text(8.2, 5.82, "(b) the multilingual jigsaw, made harder", ha="center", fontsize=10.5, fontweight="bold", color="#333")
rng = np.random.default_rng(3)
levels = [("$n{=}2$ (ordered)", ["EN", "ES"], 0.0, 4.55),
          ("$n{=}6$", ["EN", "ES", "FR", "ZH", "SW", "FI"], 0.25, 7.7),
          ("$n{=}10$ + disorder", ["EN", "ES", "FR", "ZH", "SW", "FI", "RU", "EN", "ZH", "ES"], 1.0, 10.9)]
for title, langs, shuffle, xc in levels:
    k = len(langs); cols = 2 if k <= 2 else 3
    s = 0.5
    for j, lg in enumerate(langs):
        gx = xc - (cols - 1) * s / 2 + (j % cols) * s
        gy = 5.15 - (j // cols) * s
        jit = shuffle * 0.11
        e = tuple(int(x) for x in rng.choice([1, -1], 4)) if shuffle > 0.5 else (1, 1, -1, -1)
        piece(ax, gx + rng.uniform(-jit, jit), gy + rng.uniform(-jit, jit), s, lg, label=f"f{j+1}", edges=e)
    ax.text(xc, 3.42, title, ha="center", fontsize=8.6, color="#333")
ax.annotate("", xy=(11.6, 3.15), xytext=(3.9, 3.15), arrowprops=dict(arrowstyle="-|>", color="#b23b3b", lw=2.4))
ax.text(7.75, 2.92, "more languages  $\\cdot$  more disorder  $\\cdot$  $+$persona / fiction   $\\Rightarrow$   "
        "harder to read AND harder for a safety filter", ha="center", fontsize=8.2, color="#b23b3b")

# ---------------- (c) probe curve (drawn as inset) ----------------
axc = fig.add_axes([0.055, 0.10, 0.30, 0.32])
n = np.array([2, 3, 4, 6, 8, 10])
strong = np.array([1.0, 0.99, 0.98, 0.95, 0.90, 0.86])
weak = np.array([0.95, 0.9, 0.82, 0.62, 0.40, 0.25])
axc.plot(n, strong, "-o", color="#1b6ca8", ms=4, label="strong-comprehension target")
axc.plot(n, weak, "-s", color="#ef8a3f", ms=4, label="weaker target")
axc.axhline(0.8, color="#999", ls=":", lw=1)
axc.scatter([10], [0.86], s=140, marker="*", color="#1b6ca8", zorder=5)
axc.scatter([6], [0.62], s=140, marker="*", color="#ef8a3f", zorder=5)
axc.annotate("$n^\\star$", (10, 0.86), (8.7, 0.93), color="#1b6ca8", fontsize=10)
axc.set_xlabel("puzzle difficulty $n$", fontsize=8)
axc.set_ylabel("reconstruction\n(harmless probe)", fontsize=8)
axc.set_title("(c) find the comprehension frontier", fontsize=9.2, color="#333")
axc.tick_params(labelsize=7); axc.set_ylim(0.15, 1.06); axc.legend(fontsize=6.3, loc="lower left")

# ---------------- middle-bottom message ----------------
ax.text(6.55, 1.15, "the optimization signal that chooses the puzzle\ncomes from benign plaintext (c) --- "
        "no harmful query\nis spent learning which attack to use",
        ha="center", va="center", fontsize=8.4, color="#333",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fbfbe8", ec="#cccc88", lw=0.9))

# ---------------- (d) outcome ----------------
ax.text(11.0, 2.55, "(d) solve the puzzle, safety bypassed", ha="center", fontsize=10.5, fontweight="bold", color="#333")
ax.add_patch(FancyBboxPatch((9.1, 0.5), 3.75, 1.7, boxstyle="round,pad=0.06,rounding_size=0.08", fc="#fff8f0", ec="#ef8a3f", lw=1.7))
ax.add_patch(FancyBboxPatch((9.32, 1.42), 3.3, 0.5, boxstyle="round,pad=0.03", fc="#eef4fa", ec="#1b6ca8", lw=1.2))
ax.text(10.97, 1.67, "[RECONSTRUCTED] $=q$   (recovered)", ha="center", va="center", fontsize=8, color="#1b6ca8")
ax.add_patch(FancyBboxPatch((9.32, 0.74), 3.3, 0.5, boxstyle="round,pad=0.03", fc="#fdecec", ec="#b23b3b", lw=1.2))
ax.text(10.97, 0.99, "[ANSWER] $= \\langle$unsafe$\\rangle$", ha="center", va="center", fontsize=8, color="#b23b3b")
ax.text(10.97, 0.34, "absorbed in reconstruction, the model complies", ha="center", fontsize=7, color="#555")
ax.annotate("", xy=(9.05, 1.5), xytext=(10.6, 2.45), arrowprops=dict(arrowstyle="-|>", color="#2e8b57", lw=1.9))
ax.text(9.5, 2.28, "select $n^\\star$ + levers", fontsize=7.8, color="#2e8b57", ha="left")

fig.savefig(OUT / "fig_method_overview.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_method_overview.png", dpi=200, bbox_inches="tight")
print("wrote", OUT / "fig_method_overview.pdf")
