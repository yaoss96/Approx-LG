"""Render the exact prompts of every guidance operator for the supplement.

The prompts are produced by calling the operators themselves with a client
that records the request instead of sending it, on an archive taken from a
real 8x8 run, so the supplement shows byte for byte what the model receives.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from approxlg import llm as L                                # noqa: E402
from approxlg.arch import Space                              # noqa: E402

OUT = os.path.join(ROOT, "paper", "tcad", "tabs", "prompts.tex")


class Recorder:
    sham = False

    def __init__(self):
        self.calls = []
        self.stats = L.LLMStats()

    def chat(self, system, user, max_tokens=2048, temperature=None):
        self.calls.append((system, user, max_tokens, temperature))
        return None


def archive_from_run():
    """Non-dominated set of seed 0 of the 8x8 guided campaign."""
    for line in open(os.path.join(ROOT, "results", "search_n8L.jsonl")):
        d = json.loads(line)
        if d["algo"] == "approxlg" and d["seed"] == 0:
            return [{"g": np.array(p["g"], dtype=np.int32), "NMED": p["NMED"],
                     "err": p["NMED"], "area": p["area"], "delay": p["delay"]}
                    for p in d["pareto"]]
    raise RuntimeError("no run")


def esc(txt):
    return txt.replace("\t", "    ")


def main():
    sp = Space(8)
    arch = archive_from_run()
    rec = Recorder()
    L.warm_start(rec, sp, 16)
    L.propose_candidates(rec, sp, arch, 6,
                         "Budget: 160/400 evaluations used; generation 5.")
    L.guide_mutation(rec, sp, arch[:4], 4)
    L.elicit_prior(rec, sp, arch, 4)
    L.refine_space(rec, sp, arch, 5, 240)
    L.llm_only_batch(rec, sp, arch, 32, 160, 400)
    names = ["O1 warm start", "O2 Pareto-gap injection",
             "O3 local search (guided child)", "O3 proposal distribution",
             "O4 space refinement", "LLM-only baseline"]
    temps = {0: "0.8", 1: "0.8", 2: "0.8", 3: "0.4", 4: "0.3", 5: "0.8"}
    with open(OUT, "w", encoding="utf-8") as fh:
        sysmsg = rec.calls[0][0]
        fh.write("\\subsection*{System message (all operators)}\n")
        fh.write("\\begin{lstlisting}\n" + esc(sysmsg) + "\n\\end{lstlisting}\n")
        # the template description is shared; print it once
        shared = rec.calls[0][1].split("\nNo design has been characterised")[0]
        fh.write("\\subsection*{Template description (head of every user "
                 "message)}\n")
        fh.write("\\begin{lstlisting}\n" + esc(shared) + "\n\\end{lstlisting}\n")
        for i, (name, (s, u, mt, t)) in enumerate(zip(names, rec.calls)):
            tail = u[len(shared):] if u.startswith(shared) else u
            fh.write(f"\\subsection*{{{name}}}\n")
            fh.write(f"Decoding: temperature {temps[i]}, at most {mt} output "
                     f"tokens, JSON output mode, extended reasoning off.\n")
            fh.write("\\begin{lstlisting}\n" + esc(tail.strip("\n")) +
                     "\n\\end{lstlisting}\n")
    print("wrote", OUT, sum(len(c[1]) for c in rec.calls), "chars")


if __name__ == "__main__":
    main()
