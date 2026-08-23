"""One place for figure geometry, so the plots match the paper's format.

The paper started in the IEEE conference format, two columns of 3.4in, and is
now submitted to ACM TODAES, which uses ``acmsmall``: a single column
395.8pt = 5.48in wide.  A plot drawn for one format and dropped into the other
is scaled by LaTeX, and the scaling lands on the tick labels: a 7pt font in a
7.0in figure becomes 5.5pt when squeezed into 5.48in, which is below what ACM
accepts and below what anyone can read.

Sizing therefore lives here rather than in each plotting script, and the
figures are drawn at their final printed size so that no scaling happens at
all.  Set ``APPROXLG_FMT=ieee`` to get the old two-column geometry back.
"""
from __future__ import annotations

import os

FMT = os.environ.get("APPROXLG_FMT", "acm").lower()

if FMT == "ieee":
    FULL = 7.0          # in, two-column span
    HALF = 3.4          # in, one column
    FONT = 7.0
else:                    # acmsmall / TODAES
    FULL = 5.4          # in, just inside the 5.48in text block
    HALF = 5.4          # single column: a "half" figure is the same width
    FONT = 8.0


def rc(font: float | None = None) -> dict:
    """Matplotlib settings shared by every figure in the paper."""
    return {
        "font.size": FONT if font is None else font,
        "font.family": "serif",
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.1,
        "legend.frameon": False,
        "figure.dpi": 300,
    }


def size(kind: str) -> tuple:
    """Figure size in inches for the named plot."""
    ar = {                      # (width, height) per format
        "hv_both": (FULL, 1.78 if FMT == "ieee" else 2.35),
        "sota_grid": (FULL, 3.42 if FMT == "ieee" else 4.05),
        "sota_single": (HALF, 2.3 if FMT == "ieee" else 3.35),
        "template": (HALF, 1.88 if FMT == "ieee" else 3.05),
        "styles": (HALF, 1.62 if FMT == "ieee" else 2.15),
    }[kind]
    return ar
