from __future__ import annotations
from collections import Counter


def mine_confusions(rows, route_fn):
    out = []
    for r in rows:
        if r.get("type") not in ("single", "ambiguous"):
            continue
        gold = r.get("expected_functions", [])
        ranked = route_fn(r["utterance"])
        if not ranked or ranked[0] in gold:
            continue
        if any(g in ranked for g in gold):   # gold present but ranked below the distractor
            out.append({"gold": gold[0], "distractor": ranked[0], "utterance": r["utterance"]})
    return out


def summarize(confusions):
    c = Counter((x["gold"], x["distractor"]) for x in confusions)
    return c.most_common()
