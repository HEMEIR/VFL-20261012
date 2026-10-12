"""Explain the privacy score from first principles, scenario by scenario.

Checks two things the headline figures hide:

  1. How many of the 15 label-inference scenarios actually reach chance, and how much each
     one contributes to the 20-point privacy term.
  2. What the privacy term WOULD be if every scenario hit A_min exactly - i.e. the best
     attainable value - so the gap can be attributed either to the defence or to the scoring.

    <python> scripts/privacy_explain.py
"""

import glob
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS = [f"dataset_0{i}" for i in range(1, 6)]
LFBA = ("lfba_p05", "lfba_p10", "lfba_p20")
BACKDOOR = ("villain", "pmc", "amc", "composite")
LABEL_SCENARIOS = ("pmc", "amc", "composite")


def minimumAccuracy(dataset):
    return 1.0 / 6.0 if dataset == "dataset_05" else 0.5


def clip(value):
    return max(0.0, min(1.0, value))


def load(root):
    out = {}
    for path in glob.glob(str(root / "*" / "*" / "metrics.json")):
        parts = Path(path).parts
        v = json.load(open(path, encoding="utf-8"))["validationResults"]
        out[(parts[-3], parts[-2])] = {
            "acc": float(v["mainTask"]["accuracy"]),
            "lra": (v.get("labelInference") or {}).get("labelRecoveryAccuracy"),
            "asr": (v.get("backdoor") or {}).get("attackSuccessRate"),
        }
    return out


def main():
    baseline = load(PROJECT_ROOT / "outputs" / "scenarios")
    variant = load(PROJECT_ROOT / "outputs" / "three_variants" / "o")
    clean = {d: baseline[(d, "clean")]["acc"] for d in DATASETS}

    print("=" * 104)
    print("隐私项逐场景分解（候选 O）")
    print("=" * 104)
    print("  %-22s %9s %9s %8s %9s %9s %8s"
          % ("场景", "A_attack", "A_rec", "P", "G_p", "Q_p", "说明"))
    print("  " + "-" * 100)
    total = 0.0
    perfect = 0.0
    rows = []
    for dataset in DATASETS:
        a = minimumAccuracy(dataset)
        for scenario in LABEL_SCENARIOS:
            key = (dataset, scenario)
            row = variant.get(key)
            base = baseline.get(key)
            if not row or not base or not isinstance(base.get("lra"), float):
                continue
            if not isinstance(row.get("lra"), float):
                continue
            span = base["lra"] - a
            p = clip((base["lra"] - row["lra"]) / span) if span > 0 else 0.0
            g = clip((row["acc"] - a) / (0.9 * clean[dataset] - a))
            q = g * p
            # Best attainable: the defence drives A_rec all the way to A_min -> P = 1.
            qBest = g * 1.0
            total += q
            perfect += qBest
            note = ""
            if p <= 1e-9:
                note = "P=0 攻击本就弱"
            elif p > 0.95:
                note = "P≈1 已到随机"
            rows.append((key, base["lra"], row["lra"], p, g, q, qBest, note))

    for key, base, rec, p, g, q, qBest, note in rows:
        print("  %-22s %9.4f %9.4f %8.4f %9.4f %9.4f  %s"
              % ("%s/%s" % key, base, rec, p, g, q, note))

    meanQ = total / len(rows)
    meanBest = perfect / len(rows)
    print()
    print("  实际 mean(Q_p)  = %.6f   -> 隐私项 = %.6f / 20" % (meanQ, 20 * meanQ))
    print("  理论上限 mean    = %.6f   -> 隐私项 = %.6f / 20" % (meanBest, 20 * meanBest))
    print("  差距             = %.6f" % (20 * meanBest - 20 * meanQ))
    print()

    atChance = sum(1 for r in rows if r[3] > 0.95)
    atZero = sum(1 for r in rows if r[3] <= 1e-9)
    print("  15 个场景中：")
    print("    已把恢复率压到接近随机 (P>0.95) 的: %d 个" % atChance)
    print("    P = 0（攻击本就弱，无从改善）的    : %d 个" % atZero)
    print("    中间状态                           : %d 个" % (len(rows) - atChance - atZero))
    print()
    print("  === 为什么 P=0 的场景无法改善 ===")
    for key, base, rec, p, g, q, qBest, note in rows:
        if p <= 1e-9:
            print("    %-22s 基线恢复率 %.4f，防御后 %.4f（未低于基线，P=0）"
                  % ("%s/%s" % key, base, rec))
    print()
    print("  这些场景的基线恢复率本身只比随机高一点，防御没有可改善的空间；")
    print("  即使把 A_rec 压到 A_min，也只是把 P 从 0 提到 1 —— 那是上限，不是现状。")


if __name__ == "__main__":
    main()
