"""Decompose reassign_sym's privacy term per scenario to find where points are lost.

    <python> scripts/privacy_loss_reassign_sym.py
"""

import glob
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS = [f"dataset_0{i}" for i in range(1, 6)]

A_CLEAN = {
    "dataset_01": 0.743333,
    "dataset_02": 0.970063,
    "dataset_03": 0.975588,
    "dataset_04": 0.852375,
    "dataset_05": 0.989320,
}
A_ATTACK = {
    ("dataset_01", "amc"): 0.6633, ("dataset_01", "composite"): 0.5525, ("dataset_01", "pmc"): 0.6708,
    ("dataset_02", "amc"): 0.8897, ("dataset_02", "composite"): 0.9453, ("dataset_02", "pmc"): 0.9419,
    ("dataset_03", "amc"): 0.6338, ("dataset_03", "composite"): 0.6447, ("dataset_03", "pmc"): 0.6609,
    ("dataset_04", "amc"): 0.7076, ("dataset_04", "composite"): 0.6951, ("dataset_04", "pmc"): 0.7179,
    ("dataset_05", "amc"): 0.6194, ("dataset_05", "composite"): 0.8214, ("dataset_05", "pmc"): 0.8330,
}


def minimum(d):
    return 1.0 / 6.0 if d == "dataset_05" else 0.5


def clip(v):
    return max(0.0, min(1.0, v))


def main():
    rec = {}
    for p in glob.glob(str(PROJECT_ROOT / "outputs" / "three_variants" / "reassign_sym" / "*" / "*" / "metrics.json")):
        parts = Path(p).parts
        v = json.load(open(p, encoding="utf-8"))["validationResults"]
        rec[(parts[-3], parts[-2])] = {
            "acc": v["mainTask"]["accuracy"],
            "lra": (v.get("labelInference") or {}).get("labelRecoveryAccuracy"),
        }

    rows = []
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
            q = g * p
            rows.append((q, d, s, base, r["lra"], p, g))

    rows.sort()
    print("  %-22s %9s %9s %8s %8s %8s  %s" % ("场景", "A_attack", "A_rec", "P", "G_p", "Q_p", "丢分原因"))
    print("  " + "-" * 88)
    for q, d, s, base, rec_, p, g in rows:
        reason = ""
        if p <= 1e-9:
            reason = "P=0 防御后恢复率≥基线"
        elif p < 0.5:
            reason = "P低 改善不足"
        elif g < 0.99:
            reason = "G_p低 精度拖累"
        print("  %-22s %9.4f %9.4f %8.4f %8.4f %8.4f  %s"
              % ("%s/%s" % (d, s), base, rec_, p, g, q, reason))

    total = sum(r[0] for r in rows)
    print()
    print("  mean(Q_p) = %.6f  ->  S_priv = %.6f" % (total / len(rows), 20 * total / len(rows)))
    print()
    print("  === 丢分最多的 5 个场景 ===")
    for q, d, s, base, rec_, p, g in rows[:5]:
        headroom = base - minimum(d)
        lost = g * (1.0 - p)
        print("    %-22s 余量=%.4f P=%.4f Q_p=%.4f 未得分=%.4f" % ("%s/%s" % (d, s), headroom, p, q, lost))


if __name__ == "__main__":
    main()
