"""Emit Figure 1 as an editable draw.io diagram.

`make_template_fig.py` draws the template figure for the paper; this writes the
same geometry as `paper/figs/template.drawio` so it can be opened and edited in
draw.io (or the VS Code extension).  Every constant below is read from the
matplotlib generator's own values, so the two stay in step as long as the
template parameters do.

The output is a *starting point*.  Once the file has been edited by hand,
re-running this script overwrites those edits -- copy it aside first.

Data coordinates match the matplotlib figure exactly.  They are mapped to
draw.io pixels with separate x and y scales taken from the published canvas,
because matplotlib stretches the data to fill it; an equal-scale copy comes out
noticeably taller than the figure in the paper.
"""
from __future__ import annotations

import os
from xml.sax.saxutils import escape

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "paper", "figs")
OUT = os.path.join(FIGS, "template.drawio")

# ---- the template, identical to make_template_fig.py ---------------------
N = 8
T, A = 5, 4
NCOL = 2 * N - 1
STAGES = 4
TARGETS = [6, 4, 3, 2]

ARR_TOP = N - 0.4
SY0, SDY = -1.55, 0.52
STRIP_BOT = SY0 - (STAGES - 1) * SDY - 0.30
CY = STRIP_BOT - 0.46
AY = CY - 1.35
BAND_TOP = N + 0.35
BAND_BOT = STRIP_BOT - 0.12

TRUNC, APPROX, EXACT = "#f6c6c0", "#ffe4ad", "#d4e6f8"
INK_T, INK_A, INK_E = "#7b241c", "#7d6608", "#1a5276"
CELL_FILL, CELL_EDGE = "#e8a33d", "#7d6608"
PP_INK, CUT_INK = "#222222", "#b03a2e"

# ---- data space -> draw.io pixels ----------------------------------------
# matplotlib stretches the data to fill the canvas, so the published figure is
# not drawn at equal x and y scale.  Read the canvas from figstyle and match
# its proportions, or the draw.io copy comes out noticeably taller.
import figstyle                                                  # noqa: E402

XMIN, XMAX = -3.3, NCOL + 1.3
YMIN, YMAX = AY - 0.30, BAND_TOP + 0.75
FIG_W, FIG_H = figstyle.size("template")

SCALE_Y = 40.0
SCALE_X = SCALE_Y * (FIG_W / (XMAX - XMIN)) / (FIG_H / (YMAX - YMIN))
PAD = 20.0


def px(x: float) -> float:
    return round((x - XMIN) * SCALE_X + PAD, 2)


def py(y: float) -> float:
    return round((YMAX - y) * SCALE_Y + PAD, 2)


cells: list[str] = []


def add(style: str, x: float, y: float, w: float, h: float,
        value: str = "") -> None:
    """One mxCell, positioned by its top-left corner in pixels."""
    cells.append(
        f'        <mxCell id="c{len(cells) + 2}" value="{escape(value)}" '
        f'style="{style}" vertex="1" parent="1">\n'
        f'          <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" '
        f'as="geometry" />\n'
        f'        </mxCell>')


def rect(x0: float, y0: float, w: float, h: float, fill: str,
         stroke: str = "none", rounded: int = 0, lw: float = 1.0) -> None:
    """Rectangle given in data space by its bottom-left corner."""
    stroke_s = "none" if stroke == "none" else stroke
    add(f"rounded={rounded};whiteSpace=wrap;html=1;fillColor={fill};"
        f"strokeColor={stroke_s};strokeWidth={lw};"
        + ("arcSize=18;" if rounded else ""),
        px(x0), py(y0 + h), round(w * SCALE_X, 2), round(h * SCALE_Y, 2))


def label(x: float, y: float, text: str, size: int = 11,
          colour: str = "#333333", ha: str = "center", va: str = "middle",
          bold: bool = False, w: float = 150.0, h: float = 22.0) -> None:
    """Text anchored the way matplotlib's ha/va anchor it."""
    left = {"center": x - w / 2, "left": x, "right": x - w}[ha]
    # matplotlib va="bottom" puts the text *above* the anchor and
    # va="top" puts it below; the box has to follow, not mirror it.
    top = {"middle": y - h / 2, "bottom": y - h, "top": y}[va]
    align = {"center": "center", "left": "left", "right": "right"}[ha]
    valign = {"middle": "middle", "bottom": "bottom", "top": "top"}[va]
    add(f"text;html=1;strokeColor=none;fillColor=none;align={align};"
        f"verticalAlign={valign};whiteSpace=wrap;fontSize={size};"
        f"fontColor={colour};" + ("fontStyle=1;" if bold else ""),
        round(left, 2), round(top, 2), w, h, text)


def label_at(xd: float, yd: float, text: str, **kw) -> None:
    """label() taking data-space coordinates."""
    w = kw.pop("w", 150.0)
    h = kw.pop("h", 22.0)
    ha = kw.get("ha", "center")
    va = kw.get("va", "middle")
    x = {"center": px(xd), "left": px(xd), "right": px(xd)}[ha]
    y = {"middle": py(yd), "bottom": py(yd), "top": py(yd)}[va]
    label(x, y, text, w=w, h=h, **kw)


def marker(xd: float, yd: float, kind: str) -> None:
    if kind == "pp":
        d = 8.0
        add(f"ellipse;html=1;fillColor={PP_INK};strokeColor=none;",
            round(px(xd) - d / 2, 2), round(py(yd) - d / 2, 2), d, d)
    else:
        d = 16.0
        add(f"text;html=1;strokeColor=none;fillColor=none;align=center;"
            f"verticalAlign=middle;fontSize=13;fontColor={CUT_INK};",
            round(px(xd) - d / 2, 2), round(py(yd) - d / 2, 2), d, d, "✕")


# ---- 1. region shading, behind everything --------------------------------
for x0, wid, col, name in ((NCOL - T - 0.5, T, TRUNC, "truncated columns"),
                           (NCOL - T - A - 0.5, A, APPROX, "approximate window"),
                           (-0.5, NCOL - T - A, EXACT, "exact columns")):
    rect(x0, BAND_BOT, wid, BAND_TOP - BAND_BOT, col)

# ---- 2. partial-product array --------------------------------------------
for i in range(N):
    for j in range(N):
        c = i + j
        marker(NCOL - 1 - c, N - 1 - i, "cut" if c < T else "pp")

# ---- 3. Dadda reduction strip, one cell per (stage, column) slot ----------
w, h = 0.70, 0.34
for s in range(STAGES):
    y = SY0 - s * SDY
    for c in range(T, NCOL):
        approx = c < T + A
        rect(NCOL - 1 - c - w / 2, y - h / 2, w, h,
             CELL_FILL if approx else "#ffffff",
             CELL_EDGE if approx else "#8899a6", lw=0.8)
    label_at(-0.95, y, f"s<sub>{s}</sub>", size=9, colour="#555555",
             ha="right", w=40.0, h=16.0)
    label_at(NCOL - 0.05, y, f"&#8594; {TARGETS[s]}", size=9,
             colour="#8a8a8a", ha="left", w=48.0, h=16.0)

# ---- 4. band labels down the left ----------------------------------------
for y, txt in (((ARR_TOP + 0.4) / 2 + 0.6, "partial<br>products"),
               (SY0 - (STAGES - 1) * SDY / 2, "Dadda<br>reduction"),
               (AY + 0.45, "final<br>adder")):
    label_at(-2.35, y, txt, size=12, colour="#333333", w=110.0, h=40.0)

# ---- 5. region headers ----------------------------------------------------
for xc, txt, col in ((NCOL - T / 2 - 0.5, "<i>T</i> truncated", INK_T),
                     (NCOL - T - A / 2 - 0.5, "<i>A</i> approximate", INK_A),
                     ((NCOL - T - A) / 2 - 0.5, "exact", INK_E)):
    label_at(xc, BAND_TOP + 0.12, txt, size=13, colour=col, va="bottom",
             w=180.0, h=22.0)

label_at(NCOL - T - A / 2 - 0.5, SY0 + 0.44, "one style gene per slot",
         size=11, colour=INK_A, va="bottom", w=200.0, h=20.0)

# ---- 6. column indices ----------------------------------------------------
for c in range(NCOL):
    label_at(NCOL - 1 - c, CY, str(c), size=9, colour="#444444", va="top",
             w=30.0, h=16.0)
label_at(-0.95, CY, "column", size=9, colour="#888888", ha="right", va="top",
         w=60.0, h=16.0)

# ---- 7. final adder -------------------------------------------------------
for x0, wid, ink, txt in ((NCOL - T - 0.45, T - 0.1, INK_T, "dropped"),
                          (NCOL - T - 2.45, 2.0, INK_A, "OR (<i>k</i>)"),
                          (-0.45, NCOL - T - 2.1, INK_E,
                           "exact carry-propagate adder")):
    rect(x0, AY, wid, 0.9, "#ffffff", ink, rounded=1, lw=1.0)
    label_at(x0 + wid / 2, AY + 0.45, txt, size=11, colour=ink,
             w=round(wid * SCALE_X, 2), h=20.0)

label_at(NCOL - 0.05, AY + 0.45, "+ <i>c</i> E&#772;<sub>T</sub>", size=11,
         colour=INK_T, ha="left", w=80.0, h=20.0)

# ---- 8. legend, in the free upper-left corner of the array ----------------
LEG_X, LEG_Y = -3.0, 8.55        # top-left corner, as in the figure
marker(LEG_X, LEG_Y, "pp")
label_at(LEG_X + 0.28, LEG_Y, "partial product", size=9, colour="#333333",
         ha="left", w=120.0, h=16.0)
marker(LEG_X, LEG_Y - 0.58, "cut")
label_at(LEG_X + 0.28, LEG_Y - 0.58, "truncated before compression", size=9,
         colour="#333333", ha="left", w=180.0, h=16.0)

# ---- write ----------------------------------------------------------------
W = round((XMAX - XMIN) * SCALE_X + 2 * PAD, 2)
H = round((YMAX - YMIN) * SCALE_Y + 2 * PAD, 2)

xml = (
    '<mxfile host="app.diagrams.net" type="device">\n'
    '  <diagram name="Approx-LG template" id="approxlg-template">\n'
    f'    <mxGraphModel dx="{int(W)}" dy="{int(H)}" grid="1" gridSize="10" '
    'guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" '
    f'pageScale="1" pageWidth="{int(W)}" pageHeight="{int(H)}" math="0" '
    'shadow="0">\n'
    "      <root>\n"
    '        <mxCell id="0" />\n'
    '        <mxCell id="1" parent="0" />\n'
    + "\n".join(cells) + "\n"
    "      </root>\n"
    "    </mxGraphModel>\n"
    "  </diagram>\n"
    "</mxfile>\n"
)

os.makedirs(FIGS, exist_ok=True)
with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
    fh.write(xml)
print(f"wrote {OUT}  ({len(cells)} shapes, {int(W)}x{int(H)} px)")
