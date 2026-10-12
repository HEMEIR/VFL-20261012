"""Score reassign_sym with the OFFICIAL formula, using A_clean from O's snapshot clean accuracy
(clean has no attack, so O's extra defence leaves clean accuracy essentially unchanged) and
A_attack recovered from the earlier score_breakdown run.

    <python> scripts/quick_score.py reassign_sym o
"""

import glob
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS = [f"dataset_0{i}" for i in range(1, 6)]
LFBA = ("lfba_p05", "lfba_p10", "lfba_p20")
BACKDOOR = ("villain", "pmc", "amc", "composite")  # same tuple score_breakdown uses for perf

# A_clean from O snapshot clean accuracy.
A_CLEAN = {
    "dataset_01": 0.743333,
    "dataset_02": 0.970063,
    "dataset_03": 0.975588,
    "dataset_04": 0.852375,
    "dataset_05": 0.989320,
}

A_ATTACK = {
    ("dataset_01", "amc"): 0.6633,
    ("dataset_01", "composite"): 0.5525,
    ("dataset_01", "pmc"): 0.6708,
    ("dataset_02", "amc"): 0.8897,
    ("dataset_02", "composite"): 0.9453,
    ("dataset_02", "pmc"): 0.9419,
    ("dataset_03", "amc"): 0.6338,
    ("dataset_03", "composite"): 0.6447,
    ("dataset_03", "pmc"): 0.6609,
    ("dataset_04", "amc"): 0.7076,
    ("dataset_04", "composite"): 0.6951,
    ("dataset_04", "pmc"): 0.7179,
    ("dataset_05", "amc"): 0.6194,
    ("dataset_05", "composite"): 0.8214,
    ("dataset_05", "pmc"): 0.8330,
}


def minimum(d):
    return 1.0 / 6.0 if d == "dataset_05" else 0.5


def clip(v):
    return max(0.0, min(1.0, v))


def load(variant):
    out = {}
    for p in glob.glob(str(PROJECT_ROOT / "outputs" / "three_variants" / variant / "*" / "*" / "metrics.json")):
        parts = Path(p).parts
        v = json.load(open(p, encoding="utf-8"))["validationResults"]
        out[(parts[-3], parts[-2])] = {
            "acc": v["mainTask"]["accuracy"],
            "asr": (v.get("backdoor") or {}).get("attackSuccessRate"),
            "lra": (v.get("labelInference") or {}).get("labelRecoveryAccuracy"),
        }
    return out


def score(variant):
    rec = load(variant)

    # 1) performance: 25 entries = LFBA family-averaged (1) + villain/pmc/amc/composite (4) x5
    utils = []
    for d in DATASETS:
        a = minimum(d)
        span = A_CLEAN[d] - a
        family = [clip((rec[(d, s)]["acc"] - a) / span) for s in LFBA if (d, s) in rec]
        if family:
            utils.append(sum(family) / len(family))
        for s in BACKDOOR:
            if (d, s) in rec:
                utils.append(clip((rec[(d, s)]["acc"] - a) / span))
    perf = 20 * sum(utils) / len(utils)

    # 2) privacy: 15 entries
    qps = []
    for d in DATASETS:
        for s in ("pmc", "amc", "composite"):
            r = rec.get((d, s))
            base = A_ATTACK.get((d, s))
            if not r or base is None or not isinstance(r["lra"], float):
                continue
            a = minimum(d)
            span = base - a
            p = clip((base - r["lra"]) / span) if span > 0 else 0.0
            g = clip((r["acc"] - a) / (0.9 * A_CLEAN[d] - a))
            qps.append(g * p)
    priv = 20 * sum(qps) / len(qps)

    # 3) robustness: all scenarios with an ASR (villain + lfba*3 + composite) x5 = 25
    qrs = []
    for key in sorted(rec):
        r = rec[key]
        if not isinstance(r["asr"], float):
            continue
        d = key[0]
        a = minimum(d)
        g = clip((r["acc"] - a) / (0.9 * A_CLEAN[d] - a))
        rr = 1.0 - r["asr"]
        qrs.append(2.0 * g * rr / (g + rr) if (g + rr) > 0 else 0.0)
    sec = 20 * sum(qrs) / len(qrs)

    print("  %-14s perf=%.6f priv=%.6f sec=%.6f  total=%.6f  (n=%d/%d/%d)"
          % (variant, perf, priv, sec, perf + priv + sec, len(utils), len(qps), len(qrs)))


if __name__ == "__main__":
    import sys
    for v in sys.argv[1:] or ["reassign_sym", "o"]:
        score(v)
