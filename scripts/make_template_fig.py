"""Draw the architecture template figure.

Three bands stacked on one column axis: the partial-product array, the Dadda
reduction strip where the bulk of the genome lives (60 of the 64 genes at
8x8), and the final adder.  The point the figure has to make is that the three
regions -- truncated, approximate, exact -- are the same columns all the way
down, so the bands share an x axis and the region shading runs behind all of
them.

Redrawn for the single-column journal format: the previous version inherited
absolute 4.4-5.6pt annotations chosen for a 3.4in two-column figure, which
became unreadably small once the canvas grew, and the reduction cells were
thin enough to read as stacked rules rather than as one cell per slot.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import figstyle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "paper", "figs")
os.makedirs(FIGS, exist_ok=True)

N = 8
T, A = 5, 4
NCOL = 2 * N - 1
STAGES = 4                      # Dadda stages for N = 8
TARGETS = [6, 4, 3, 2]

BASE = figstyle.FONT            # 8.0 in the ACM format, 7.0 in the IEEE one
F_HEAD = BASE * 0.95            # region headers
F_SIDE = BASE * 0.90            # left-hand band labels
F_ANN = BASE * 0.85             # in-box annotations
F_TICK = BASE * 0.72            # column indices, stage labels

plt.rcParams.update(figstyle.rc())
fig, ax = plt.subplots(figsize=figstyle.size("template"))

TRUNC, APPROX, EXACT = "#f6c6c0", "#ffe4ad", "#d4e6f8"
INK_T, INK_A, INK_E = "#7b241c", "#7d6608", "#1a5276"

# ---- vertical geometry ---------------------------------------------------
ARR_TOP = N - 0.4               # top row of partial products
SY0, SDY = -1.55, 0.52          # first reduction stage, stage pitch
STRIP_BOT = SY0 - (STAGES - 1) * SDY - 0.30
CY = STRIP_BOT - 0.46           # column-index row
AY = CY - 1.35                  # final-adder strip

BAND_TOP = N + 0.35
BAND_BOT = STRIP_BOT - 0.12

# region shading behind every band
for x0, wid, col in ((NCOL - T - 0.5, T, TRUNC),
                     (NCOL - T - A - 0.5, A, APPROX),
                     (-0.5, NCOL - T - A, EXACT)):
    ax.add_patch(Rectangle((x0, BAND_BOT), wid, BAND_TOP - BAND_BOT,
                           facecolor=col, edgecolor="none", zorder=0))

# ---- partial-product array ----------------------------------------------
for i in range(N):
    for j in range(N):
        c = i + j
        x, y = NCOL - 1 - c, N - 1 - i
        if c < T:
            ax.plot(x, y, marker="x", ms=3.4, mew=0.8, color="#b03a2e",
                    zorder=2)
        else:
            ax.plot(x, y, marker="o", ms=3.0, color="#222222", zorder=2)

# ---- reduction strip: one cell per (stage, column) slot ------------------
w, h = 0.70, 0.34
for s in range(STAGES):
    y = SY0 - s * SDY
    for c in range(T, NCOL):
        x = NCOL - 1 - c
        approx = c < T + A
        ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h,
                               facecolor="#e8a33d" if approx else "#ffffff",
                               edgecolor="#7d6608" if approx else "#8899a6",
                               lw=0.55, zorder=2))
    ax.text(-0.95, y, f"$s_{s}$", ha="right", va="center", fontsize=F_TICK,
            color="#555555")
    ax.text(NCOL - 0.05, y, f"$\\rightarrow\\,{TARGETS[s]}$", ha="left",
            va="center", fontsize=F_TICK, color="#8a8a8a")

# ---- band labels down the left ------------------------------------------
for y, txt in (((ARR_TOP + 0.4) / 2 + 0.6, "partial\nproducts"),
               (SY0 - (STAGES - 1) * SDY / 2, "Dadda\nreduction"),
               (AY + 0.45, "final\nadder")):
    ax.text(-2.35, y, txt, ha="center", va="center", fontsize=F_SIDE,
            color="#333333", linespacing=1.15)

# ---- region headers ------------------------------------------------------
for xc, txt, col in ((NCOL - T / 2 - 0.5, r"$T$ truncated", INK_T),
                     (NCOL - T - A / 2 - 0.5, r"$A$ approximate", INK_A),
                     ((NCOL - T - A) / 2 - 0.5, "exact", INK_E)):
    ax.text(xc, BAND_TOP + 0.12, txt, ha="center", va="bottom",
            fontsize=F_HEAD, color=col)

ax.text(NCOL - T - A / 2 - 0.5, SY0 + 0.44, "one style gene per slot",
        ha="center", va="bottom", fontsize=F_ANN, color=INK_A)

# ---- column indices ------------------------------------------------------
for c in range(NCOL):
    ax.text(NCOL - 1 - c, CY, str(c), ha="center", va="top", fontsize=F_TICK,
            color="#444444")
ax.text(-0.95, CY, "column", ha="right", va="top", fontsize=F_TICK,
        color="#888888")

# ---- final adder ---------------------------------------------------------
def box(x0, wid, edge, label, ink):
    ax.add_patch(FancyBboxPatch((x0, AY), wid, 0.9,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor="#ffffff", edgecolor=edge, lw=0.7,
                                zorder=2))
    ax.text(x0 + wid / 2, AY + 0.45, label, ha="center", va="center",
            fontsize=F_ANN, color=ink, zorder=3)


box(NCOL - T - 0.45, T - 0.1, INK_T, "dropped", INK_T)
box(NCOL - T - 2.45, 2.0, INK_A, r"OR ($k$)", INK_A)
box(-0.45, NCOL - T - 2.1, INK_E, "exact carry-propagate adder", INK_E)
ax.text(NCOL - 0.05, AY + 0.45, r"$+\;c\,\overline{E}_T$", ha="left",
        va="center", fontsize=F_ANN, color=INK_T)

# ---- legend in the empty upper-left of the array -------------------------
ax.plot([], [], marker="o", ms=3.0, color="#222222", ls="none",
        label="partial product")
ax.plot([], [], marker="x", ms=3.4, mew=0.8, color="#b03a2e", ls="none",
        label="truncated before compression")
# the array is a parallelogram, so the free space is the top-left corner;
# the lower-left is full of partial products
ax.legend(fontsize=F_TICK, loc="upper left", frameon=False,
          bbox_to_anchor=(0.02, 0.995), handletextpad=0.3, borderpad=0.1,
          labelspacing=0.35)

ax.set_xlim(-3.3, NCOL + 1.3)
ax.set_ylim(AY - 0.30, BAND_TOP + 0.75)
ax.axis("off")
fig.tight_layout(pad=0.05)
fig.savefig(os.path.join(FIGS, "template.pdf"))
print("wrote", os.path.join(FIGS, "template.pdf"))
