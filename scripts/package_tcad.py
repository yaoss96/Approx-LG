"""Bundle the TCAD manuscript and supplement for submission.

Collects every file the two documents read -- section sources, macro files,
table bodies, figures, bibliography -- plus both PDFs into
paper/tcad/dist/approxlg_tcad_<date>.zip, after checking that no placeholder
is left (\\pend or an input of pending.tex) and that both PDFs are newer than
their sources.

    python scripts/package_tcad.py
"""
import datetime
import os
import re
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper", "tcad")


def read(name):
    return open(os.path.join(PAPER, name), encoding="utf-8").read()


def main():
    files = {"main.tex", "supplement.tex", "refs.bib", "main.bbl",
             "numbers.tex", "numbers_tcad.tex", "main.pdf", "supplement.pdf"}
    srcs = ["main.tex", "supplement.tex"]
    for m in re.finditer(r"\\input\{(sec_[a-z]+)\}", read("main.tex")):
        files.add(m.group(1) + ".tex")
        srcs.append(m.group(1) + ".tex")
    problems = []
    for s in srcs:
        t = read(s)
        files.update(re.findall(r"\\tabinput\{([^}]+)\}", t))
        files.update(f + ".tex" for f in re.findall(r"\\input\{(tabs/[^}]+)\}", t))
        files.update(f if os.path.splitext(f)[1] in (".pdf", ".png", ".jpg")
                     else f + ".pdf"
                     for f in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", t))
        if "\\input{pending}" in t:
            problems.append(f"{s} still inputs pending.tex")
        for m in re.finditer(r"\\pend\{", t):
            if "newcommand{\\pend}" not in t[max(0, m.start() - 20):m.start() + 6]:
                problems.append(f"{s} still uses \\pend")
                break
    missing = [f for f in sorted(files) if not os.path.exists(os.path.join(PAPER, f))]
    if missing:
        problems.append(f"missing files: {missing}")
    newest = max(os.path.getmtime(os.path.join(PAPER, f)) for f in files
                 if not f.endswith(".pdf") and os.path.exists(os.path.join(PAPER, f)))
    for pdf in ("main.pdf", "supplement.pdf"):
        if os.path.getmtime(os.path.join(PAPER, pdf)) < newest:
            problems.append(f"{pdf} is older than its sources; rebuild")
    if problems:
        print("not packaged:\n  " + "\n  ".join(problems))
        sys.exit(1)
    os.makedirs(os.path.join(PAPER, "dist"), exist_ok=True)
    out = os.path.join(PAPER, "dist",
                       f"approxlg_tcad_{datetime.date.today():%Y%m%d}.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):
            z.write(os.path.join(PAPER, f), f)
    print(f"wrote {out} ({len(files)} files)")


if __name__ == "__main__":
    main()
