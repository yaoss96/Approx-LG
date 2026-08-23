"""Render the image workloads for four iso-accuracy design pairs.

The applications table gives PSNR and SSIM, which say how far an output is from
the exact one but not what the damage looks like.  These two figures show it.
Each column is an iso-accuracy pair -- an Approx-LG design above, the published
design of matching NMED below -- so a reader can see that at equal accuracy the
two are visually indistinguishable while the table records that ours costs
30-50% less energy.

One figure per workload: blending two images, and a 5x5 Gaussian smoothing.
Both are computed through exactly the product tables the accuracy and power
numbers came from, so nothing here is a separate model of the circuit.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "paper", "figs")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import figstyle                                              # noqa: E402
from approxlg.arch import Space                              # noqa: E402
from approxlg.simulate import Evaluator                      # noqa: E402
from run_apps import (blend, build_lut, build_lut_ref,       # noqa: E402
                      load_images, smooth)

# top row: ours; bottom row: the published design of matching NMED
OURS = ["ALG-2", "ALG-3", "ALG-4", "ALG-5"]
THEIRS = ["mul8u_125K", "mul8u_QJD", "mul8u_NGR", "mul8u_DM1"]


def psnr(ref, x):
    mse = float(((ref.astype(np.float64) - x) ** 2).mean())
    return 99.0 if mse <= 0 else 10 * np.log10(255.0 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="camera")
    # A busy second scene competes with the artefacts it is supposed to reveal;
    # "coins" is worse still, a grid of near-identical discs that reads as a
    # texture rather than a photograph.  "moon" is nearly flat, so what the
    # eye picks up in the blend is the multiplier and not the picture.
    ap.add_argument("--second", default="moon")
    args = ap.parse_args()

    designs = {d["name"]: d for d in
               json.load(open(os.path.join(RESULTS, "app_designs.json")))}
    refs = {d["name"]: d for d in
            json.load(open(os.path.join(RESULTS, "app_refs.json")))}
    # Panel labels come from the same file Table 5 is built from, so the two
    # cannot drift.  Those PSNRs average the six standard images; the panel
    # shows one of them, which is why a label need not match what a reader
    # would measure off the printed image.
    quality = {d["name"]: d for d in
               json.load(open(os.path.join(RESULTS, "apps.json")))}
    sp = Space(8)
    ev = Evaluator(sp)

    luts = {}
    for nm in OURS:
        luts[nm] = build_lut(sp, ev, designs[nm]["g"])
    for nm in THEIRS:
        luts[nm] = build_lut_ref(refs[nm]["json"], refs[nm]["top"], ev.stim)
    lut_exact = build_lut(sp, ev, np.zeros(sp.length, dtype=np.int32))

    ims = load_images()
    x, y = ims[args.image], ims[args.second]

    for task, fn, fname in (
            ("blend", lambda l: blend(l, x, y), "apps_blend.pdf"),
            ("smooth", lambda l: smooth(l, x), "apps_smooth.pdf")):
        ref = fn(lut_exact)
        plt.rcParams.update(figstyle.rc())
        w = figstyle.FULL
        fig, grid = plt.subplots(2, 4, figsize=(w, w / 2 + 0.62))
        for row, names in enumerate((OURS, THEIRS)):
            for col, nm in enumerate(names):
                out = fn(luts[nm])
                ax = grid[row][col]
                ax.imshow(out, cmap="gray", vmin=0, vmax=255,
                          interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                # matplotlib titles are not typeset by LaTeX here, so no
                # thin-space escapes
                q = quality[nm][f"{task}_psnr"]
                sub = "lossless" if q >= 99.0 - 1e-9 else f"{q:.1f} dB"
                ax.set_title(f"{nm.replace('_', ' ')}\n"
                             f"{sub},  {quality[nm]['PDP']:.1f} fJ",
                             fontsize=figstyle.FONT * 0.8, pad=2.5,
                             linespacing=1.25)
                for s in ax.spines.values():
                    s.set_linewidth(0.4)
                    s.set_color("#c62828" if row == 0 else "#455a64")
        fig.tight_layout(pad=0.28, h_pad=1.1, w_pad=0.35)
        out_path = os.path.join(FIGS, fname)
        fig.savefig(out_path)
        plt.close(fig)
        print("wrote", out_path)


if __name__ == "__main__":
    main()
