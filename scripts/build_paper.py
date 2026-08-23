"""Assemble and compile the paper.

The generated table bodies are spliced into the source rather than pulled in
with \\input: on this TeX installation an \\input inside a tabular breaks the
alignment, and inlining also makes the submitted .tex self-contained.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
OUT = "main_build"


def splice(text: str, depth: int = 0) -> str:
    def rep(m):
        name = m.group(1)
        p = os.path.join(PAPER, name + ("" if name.endswith(".tex") else ".tex"))
        if not os.path.exists(p):
            print(f"  !! missing fragment {name}")
            return "% missing " + name
        body = open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")
        body = "\n".join(l for l in body.split("\n")
                         if not l.lstrip().startswith("%"))
        return body.rstrip("\n")
    return re.sub(r"\\input\{(tabs/[^}]+)\}", rep, text)


def main():
    src = open(os.path.join(PAPER, "main.tex"), encoding="utf-8").read()
    src = splice(src)
    out = os.path.join(PAPER, OUT + ".tex")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(src)
    env = dict(os.environ)
    for cmd in ([["pdflatex", "-interaction=nonstopmode", OUT]],
                [["bibtex", OUT]],
                [["pdflatex", "-interaction=nonstopmode", OUT]],
                [["pdflatex", "-interaction=nonstopmode", OUT]]):
        for c in cmd:
            subprocess.run(c, cwd=PAPER, env=env, capture_output=True,
                           text=True, timeout=900)
    log = os.path.join(PAPER, OUT + ".log")
    errs, warns = [], []
    if os.path.exists(log):
        for line in open(log, errors="replace"):
            if line.startswith("! "):
                errs.append(line.strip())
            elif "Warning" in line and "Font" not in line:
                warns.append(line.strip())
    pdf = os.path.join(PAPER, OUT + ".pdf")
    pages = 0
    if os.path.exists(pdf):
        r = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if line.startswith("Pages:"):
                pages = int(line.split()[1])
    print(f"pages={pages} errors={len(errs)} warnings={len(warns)}")
    for e in errs[:8]:
        print("  ERROR", e)
    for w in warns[:12]:
        print("  warn ", w)
    return 0 if (pages and not errs) else 1


if __name__ == "__main__":
    sys.exit(main())
