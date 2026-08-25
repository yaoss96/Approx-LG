"""Build the source-document zip the journal asks to be uploaded unpacked.

What goes in is the *spliced* manuscript, paper/main_build.tex, not paper/main.tex:
build_paper.py inlines every \\input{tabs/...} into it, which both dodges the
\\input-inside-tabular breakage of our TeX installation and leaves a source tree
with no fragment files to lose.  Everything the spliced file still needs comes
with it -- numbers.tex, refs.bib, the .bbl, and only the figures actually
included.

The staging tree is then compiled from scratch, on its own, before it is
zipped: a submission that only builds next to the repository is not a
submission.

    python scripts/make_submission.py            # build, verify, zip
    python scripts/make_submission.py --keep     # also leave the staging tree
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
OUT = os.path.join(PAPER, "submission")
STAGE = os.path.join(OUT, "source")
ZIP = os.path.join(OUT, "approx-lg-source.zip")

# Strings that would break double-anonymous review.  Account handles, mail
# addresses and machine paths are leaks wherever they appear; author names are
# leaks everywhere *except* inside the bibliography, where ACM requires a
# self-citation to stay unanonymised and be cited in the third person.
FORBIDDEN = ("razer", "yaoss", "outlook",
             "desktop\\approx-lg", "desktop/approx-lg")
FORBIDDEN_OUTSIDE_BIB = ("shangshang", "zuoning")
BIB_FILES = ("refs.bib", "main.bbl")


DROP = {".aux", ".log", ".out", ".blg", ".pdf", ".synctex.gz", ".fls",
        ".fdb_latexmk"}


def packaged(stage):
    """Relative paths of the files that go into the zip, build products out."""
    out = []
    for dirpath, _, names in os.walk(stage):
        for n in sorted(names):
            rel = os.path.relpath(os.path.join(dirpath, n), stage)
            in_figs = rel.startswith("figs" + os.sep)
            if os.path.splitext(n)[1] in DROP and not in_figs:
                continue
            out.append(rel)
    return out


def sh(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=900)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true",
                    help="leave the staging tree beside the zip")
    args = ap.parse_args()

    src = os.path.join(PAPER, "main_build.tex")
    if not os.path.exists(src):
        print("!! paper/main_build.tex missing; run scripts/build_paper.py first")
        return 1

    if os.path.exists(STAGE):
        shutil.rmtree(STAGE)
    os.makedirs(os.path.join(STAGE, "figs"))

    # ---- the manuscript, under a name whose .bbl matches -----------------
    tex = open(src, encoding="utf-8").read()
    open(os.path.join(STAGE, "main.tex"), "w", encoding="utf-8",
         newline="\n").write(tex)

    # ---- everything it still refers to -----------------------------------
    for name in ("numbers.tex", "refs.bib"):
        shutil.copy2(os.path.join(PAPER, name), os.path.join(STAGE, name))
    bbl = os.path.join(PAPER, "main_build.bbl")
    if os.path.exists(bbl):
        shutil.copy2(bbl, os.path.join(STAGE, "main.bbl"))

    figs = sorted(set(re.findall(r"\\includegraphics\[[^]]*\]\{figs/([^}]+)\}",
                                 tex)))
    for f in figs:
        shutil.copy2(os.path.join(PAPER, "figs", f + ".pdf"),
                     os.path.join(STAGE, "figs", f + ".pdf"))
    print(f"staged: main.tex, numbers.tex, refs.bib, main.bbl, "
          f"{len(figs)} figures ({', '.join(figs)})")

    # ---- does it build on its own? ---------------------------------------
    print("compiling the staging tree by itself ...")
    for cmd in (["pdflatex", "-interaction=nonstopmode", "main"],
                ["bibtex", "main"],
                ["pdflatex", "-interaction=nonstopmode", "main"],
                ["pdflatex", "-interaction=nonstopmode", "main"]):
        sh(cmd, STAGE)
    log = os.path.join(STAGE, "main.log")
    errs = [l.strip() for l in open(log, errors="replace") if l.startswith("! ")]
    pdf = os.path.join(STAGE, "main.pdf")
    pages = 0
    if os.path.exists(pdf):
        r = sh(["pdfinfo", pdf], STAGE)
        for line in r.stdout.splitlines():
            if line.startswith("Pages:"):
                pages = int(line.split()[1])
    print(f"  standalone build: pages={pages} errors={len(errs)}")
    for e in errs[:5]:
        print("   ", e)
    if not pages or errs:
        print("!! the package does not build on its own; not zipping")
        return 1

    # ---- anonymity, over exactly what will be zipped ---------------------
    bad = []
    for rel in packaged(STAGE):
        blob = open(os.path.join(STAGE, rel), "rb").read().lower()
        terms = list(FORBIDDEN)
        if os.path.basename(rel) not in BIB_FILES:
            terms += list(FORBIDDEN_OUTSIDE_BIB)
        for f in terms:
            if f.encode() in blob or f in rel.lower():
                bad.append((rel, f))
    if bad:
        print("!! identifying strings found, not zipping:")
        for rel, f in bad:
            print(f"    {rel}: {f}")
        return 1
    print("  anonymity scan: clean")

    # ---- zip only the sources, not the build products --------------------
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in packaged(STAGE):
            z.write(os.path.join(STAGE, rel), rel)
        listed = z.namelist()

    mb = os.path.getsize(ZIP) / 2**20
    print(f"\nwrote {ZIP}  ({mb:.2f} MB, {len(listed)} files)")
    for n in listed:
        print("   ", n)
    if mb > 100:
        print("!! over the 100 MB limit; split the supplement into its own zip")
        return 1
    if not args.keep:
        shutil.rmtree(STAGE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
