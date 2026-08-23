"""How far the elicited proposal distribution is from the fixed one.

The prior control (scripts/analyze_priors.py) asks whether the two priors
perform differently.  This asks the prior question: whether they are even
different advice.  Every logged answer to the prior prompt is parsed back into
ranked style lists and compared, offset by offset, with the prior that
Sec.~"Does the guidance need a language model?" writes down by rule.

Answers come from the campaign log, which accumulates across every run that
used the tag, so the count is of *answers*, not of runs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from approxlg.cells import SLOT_MODES, SLOT_MODE_BY_NAME     # noqa: E402
from approxlg.llm import OFFSET_KEYS                         # noqa: E402
from approxlg.priors import expert_prior                     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
TABS = os.path.join(ROOT, "paper", "tabs")

MARKER = "give a *proposal distribution* for the evolutionary operator"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", nargs="+", default=["llm_n8L.jsonl"])
    ap.add_argument("--out", default="prior_agreement.json")
    args = ap.parse_args()

    ep = expert_prior().by_offset
    heads = defaultdict(Counter)
    n_answer = n_list = n_head = n_top3 = 0
    for name in args.log:
        p = os.path.join(RESULTS, name)
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if MARKER not in d.get("user", ""):
                continue
            try:
                pri = json.loads(d["resp"]).get("prior")
            except Exception:
                continue
            if not isinstance(pri, dict):
                continue
            n_answer += 1
            for k in OFFSET_KEYS:
                v = pri.get(k) or pri.get(k.rstrip("+"))
                if not isinstance(v, list):
                    continue
                idx = [SLOT_MODE_BY_NAME[str(x).strip().upper()] for x in v
                       if str(x).strip().upper() in SLOT_MODE_BY_NAME]
                if not idx:
                    continue
                n_list += 1
                heads[k][SLOT_MODES[idx[0]].name] += 1
                n_head += idx[0] == ep[k][0]
                n_top3 += idx[0] in ep[k][:3]

    if not n_list:
        print("no prior answers found in", args.log)
        return
    far = OFFSET_KEYS[-1]
    tot_far = sum(heads[far].values())
    out = {
        "answers": n_answer,
        "lists": n_list,
        "head_match": 100.0 * n_head / n_list,
        "top3_match": 100.0 * n_top3 / n_list,
        "exact_far": 100.0 * heads[far]["EXACT"] / max(tot_far, 1),
        "heads": {k: dict(v.most_common(5)) for k, v in heads.items()},
        "expert": {k: [SLOT_MODES[i].name for i in ep[k]] for k in ep},
    }
    with open(os.path.join(TABS, args.out), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"{n_answer} prior answers, {n_list} offset lists")
    print(f"head matches the fixed prior: {out['head_match']:.0f}%  "
          f"within its top three: {out['top3_match']:.0f}%")
    print(f"exact style first at offset {far}: {out['exact_far']:.0f}% "
          f"of answers")
    for k in OFFSET_KEYS:
        s = sum(heads[k].values()) or 1
        top = ", ".join(f"{a} {100*c/s:.0f}%" for a, c in heads[k].most_common(3))
        print(f"  d={k:<2s} fixed={SLOT_MODES[ep[k][0]].name:<6s} "
              f"model: {top}")


if __name__ == "__main__":
    main()
