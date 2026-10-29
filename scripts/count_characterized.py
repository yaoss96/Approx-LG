"""Distinct characterized designs over every campaign, one per template and
genome (the caches of later campaigns were seeded from earlier ones, so the
same record appears in several files), and the number of records scanned.
Writes paper/tcad/tabs/charcount.json for analyze_tcad.py.

    python scripts/count_characterized.py
"""
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def template(tag):
    # the extended alphabet decodes base-alphabet genomes to the same
    # circuits, so n8mrex shares the 8x8 template's identities
    if tag.startswith("n8"):
        return "8"
    for pre in ("n16u", "n16s", "n32u"):
        if tag.startswith(pre):
            return pre[1:]
    return tag


def main():
    seen, records = {}, 0
    for p in sorted(glob.glob(os.path.join(RESULTS, "designs_*.jsonl"))):
        tag = os.path.basename(p)[len("designs_"):-len(".jsonl")]
        s = seen.setdefault(template(tag), set())
        for line in open(p):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            records += 1
            s.add(tuple(d["g"]))
    out = {k: len(v) for k, v in seen.items()}
    out["total"] = sum(out.values())
    out["records"] = records
    with open(os.path.join(ROOT, "paper", "tcad", "tabs", "charcount.json"),
              "w") as fh:
        json.dump(out, fh, indent=1)
    print(out)


if __name__ == "__main__":
    main()
