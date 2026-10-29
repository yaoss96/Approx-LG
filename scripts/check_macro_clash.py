"""Macros defined both in the old numbers.tex (\\newcommand, loaded first) and
in numbers_tcad.tex (\\providecommand, silently ignored when the name exists).
Any name listed with differing values prints the OLD value in the paper."""
import os
import re

PAPER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "paper", "tcad")


def defs(path, kind):
    out = {}
    pat = re.compile(r"\\" + kind + r"\{\\([A-Za-z]+)\}\{(.*)\}\s*$")
    for line in open(path, encoding="utf-8"):
        m = pat.match(line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


old = defs(os.path.join(PAPER, "numbers.tex"), "newcommand")
new = defs(os.path.join(PAPER, "numbers_tcad.tex"), "providecommand")
clash = sorted(set(old) & set(new))
for k in clash:
    flag = "SAME" if old[k] == new[k] else "DIFF"
    print(f"{flag} {k}: old={old[k][:40]!r} new={new[k][:40]!r}")
print(f"{len(clash)} shared names, "
      f"{sum(old[k] != new[k] for k in clash)} with different values")
