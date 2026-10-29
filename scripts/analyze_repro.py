"""Reproduction check of the compressor-tree reference methods.

Every design of Esposito et al., Wang et al. and OPACT carries error figures
that do not depend on the technology, so a reproduction can be checked
against the published numbers exactly rather than by trend.  The published
values below are copied from the papers (Esposito et al., Table XI; Wang et
al., Table I; OPACT, Table I).  Esposito et al. report Monte-Carlo estimates
with a 1% relative error at 99% confidence; the 8x8 figures of Wang et al.
and OPACT are exhaustive.

Reproduced values come from ``results/refgen_n{8,16}.jsonl`` (the designs as
they enter the comparison: exhaustive at 8x8, the campaigns' 2^22-vector
stimulus at 16x16).  For OPACT, which publishes one selected design per
width, the closest design of our OPACT family in MED is reported.
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
TABS = os.path.join(ROOT, "paper", "tcad", "tabs")

# (method label, design, width) -> (published MED, published ER or None)
PUBLISHED = [
    ("Esposito", "1StepFull", 8, 22.5, 0.22),
    ("Esposito", "2StepsFull", 8, 352.0, 0.4897),
    ("Esposito", "1StepTrunc", 8, 210.0, 0.97),
    ("Esposito", "2StepsTrunc", 8, 510.0, 1.0),
    ("Wang", "AM8-5", 8, 283.0, 0.5011),
    ("OPACT", "OPACT1", 8, 14.6, 0.230),
    ("OPACT", "OPACT2", 8, 283.0, 0.501),
    ("Esposito", "1StepFull", 16, 2.60e4, 0.54),
    ("Esposito", "2StepsFull", 16, 2.60e6, 0.91),
    ("Esposito", "1StepTrunc", 16, 1.30e5, 1.0),
    ("Esposito", "2StepsTrunc", 16, 2.70e6, 1.0),
    ("Wang", "AM16-10", 16, 1.29e6, 0.890),
    ("OPACT", "OPACT1", 16, 2.34e4, 0.895),
    ("OPACT", "OPACT2", 16, 1.07e6, 0.970),
]


def load(n):
    return [json.loads(l) for l in open(os.path.join(RESULTS,
                                                     f"refgen_n{n}.jsonl"))
            if l.strip()]


def fmt(x):
    if x >= 1e4:
        m, e = f"{x:.2e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"
    return f"{x:.1f}" if x < 100 else f"{x:.0f}"


def main():
    os.makedirs(TABS, exist_ok=True)
    data = {8: load(8), 16: load(16)}
    rows, out = [], []
    for lib, name, n, med, er in PUBLISHED:
        recs = data[n]
        if lib == "OPACT":
            cand = [r for r in recs if r["lib"] == "OPACT"]
            r = min(cand, key=lambda r: abs(r["MED"] - med))
        else:
            want = name if lib == "Wang" else f"{name}{n}"
            r = next(r for r in recs if r["name"] == want)
        dev = 100 * (r["MED"] / med - 1)
        rows.append((lib, name, n, med, er, r["MED"], r["ER"], dev))
        out.append(dict(lib=lib, name=name, n=n, pub_med=med, pub_er=er,
                        med=r["MED"], er=r["ER"], dev=dev, ref=r["name"]))
        print(f"{lib:9s} {name:12s} {n:2d}  MED pub {med:10.4g} ours "
              f"{r['MED']:10.4g} ({dev:+5.1f}%)  ER pub {er} ours {r['ER']:.3f}"
              f"   [{r['name']}]")
    with open(os.path.join(TABS, "repro.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    with open(os.path.join(TABS, "repro.tex"), "w") as fh:
        for lib, name, n, med, er, rmed, rer, dev in rows:
            lab = {"Esposito": r"\cite{esposito2018}", "Wang": r"\cite{wang2020}",
                   "OPACT": r"\cite{opact}"}[lib]
            fh.write(f"{name}~{lab} & ${n}$ & {fmt(med)} & {fmt(rmed)} & "
                     f"${dev:+.1f}$ & {100 * er:.0f} & {100 * rer:.0f} \\\\\n")
    devs = [abs(r[-1]) for r in rows]
    print(f"max |deviation| {max(devs):.1f}%")


if __name__ == "__main__":
    main()
