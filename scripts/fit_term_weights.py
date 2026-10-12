"""Fit a per-term calibration from the local formula to the platform score.

The single-variable fit (platform ~ total formula) has now been contradicted once in a
way that points at a specific mechanism: candidate O gained +1.011 in the formula over H,
of which +1.418 came from the ROBUSTNESS term, and the platform gave +1.679 - far more
than the +0.61 the total-only model predicted.

Hypothesis: the three terms do not convert to platform points at the same rate.  This
fits and cross-validates a three-predictor model against every measured submission:

    platform = a * performance + b * privacy + c * robustness + d

Measured submissions (platform score is the organiser's number):

    precision_v3  37.9599
    k             40.6421
    i             41.07465
    h             41.6341
    o             43.313455

    python scripts/fit_term_weights.py
"""

import glob
import json
import itertools
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN = {
    "dataset_01": 0.7308,
    "dataset_02": 0.9699,
    "dataset_03": 0.9783,
    "dataset_04": 0.8552,
    "dataset_05": 0.9874,
}
PLATFORM = {
    "precision_v3": 37.9599,
    "k": 40.6421,
    "i": 41.07465,
    "h": 41.6341,
    "o": 43.313455,
}


def minimum(dataset):
    return 1.0 / 6.0 if dataset == "dataset_05" else 0.5


def loadBaseline():
    out = {}
    for path in glob.glob(str(PROJECT_ROOT / "outputs" / "scenarios" / "*" / "*"
                              / "metrics.json")):
        parts = Path(path).parts
        v = json.load(open(path, encoding="utf-8"))["validationResults"]
        out[(parts[-3], parts[-2])] = {
            "acc": v["mainTask"]["accuracy"],
            "lra": (v.get("labelInference") or {}).get("labelRecoveryAccuracy"),
        }
    return out


def loadSnapshot(variant):
    out = {}
    for path in glob.glob(str(PROJECT_ROOT / "outputs" / "three_variants" / variant
                              / "*" / "*" / "metrics.json")):
        parts = Path(path).parts
        v = json.load(open(path, encoding="utf-8"))["validationResults"]
        out[(parts[-3], parts[-2])] = {
            "acc": v["mainTask"]["accuracy"],
            "asr": (v.get("backdoor") or {}).get("attackSuccessRate"),
            "lra": (v.get("labelInference") or {}).get("labelRecoveryAccuracy"),
        }
    return out


def cluster(value):
    return max(0.0, min(1.0, value))


def terms(records, baseline):
    clean = {d: baseline[(d, "clean")]["acc"] for d in CLEAN if (d, "clean") in baseline}
    utilities = []
    for dataset in CLEAN:
        amin = minimum(dataset)
        span = clean[dataset] - amin
        family = [cluster((records[(dataset, s)]["acc"] - amin) / span)
                  for s in ("lfba_p05", "lfba_p10", "lfba_p20") if (dataset, s) in records]
        if family:
            utilities.append(sum(family) / len(family))
        for s in ("villain", "pmc", "amc", "composite"):
            if (dataset, s) in records:
                utilities.append(cluster((records[(dataset, s)]["acc"] - amin) / span))
    performance = 20.0 * sum(utilities) / len(utilities)

    priv = []
    for key, row in records.items():
        if not isinstance(row.get("lra"), float) or key not in baseline:
            continue
        base = baseline[key].get("lra")
        if not isinstance(base, float):
            continue
        dataset = key[0]
        amin = minimum(dataset)
        span = base - amin
        attack = cluster((base - row["lra"]) / span) if span > 0 else 0.0
        guard = cluster((row["acc"] - amin) / (0.9 * clean[dataset] - amin))
        priv.append(guard * attack)
    privacy = 20.0 * sum(priv) / len(priv)

    sec = []
    for key, row in records.items():
        if not isinstance(row.get("asr"), float):
            continue
        dataset = key[0]
        amin = minimum(dataset)
        guard = cluster((row["acc"] - amin) / (0.9 * clean[dataset] - amin))
        resistance = 1.0 - row["asr"]
        total = guard + resistance
        sec.append((2.0 * guard * resistance / total) if total > 0 else 0.0)
    robustness = 20.0 * sum(sec) / len(sec)
    return performance, privacy, robustness


def main():
    baseline = loadBaseline()
    rows = []
    for variant, platform in PLATFORM.items():
        records = loadSnapshot(variant)
        if not records:
            continue
        p, pr, s = terms(records, baseline)
        rows.append((variant, p, pr, s, platform))
        print("%-14s 性能 %7.3f  隐私 %7.3f  鲁棒 %7.3f  ->  平台 %9.5f"
              % (variant, p, pr, s, platform))

    print()
    X = np.array([[r[1], r[2], r[3], 1.0] for r in rows])
    y = np.array([r[4] for r in rows])
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    print("三项拟合:  平台 = %.4f*性能 %+.4f*隐私 %+.4f*鲁棒 %+.4f" % tuple(coeffs))
    print("  （总公式模型的斜率是 0.7196；若三项同权，这里应三项都接近该值）")
    print()

    print("留一交叉验证（每次留一个点拟合，预测它）：")
    totalErr = 0.0
    for i, row in enumerate(rows):
        mask = [j for j in range(len(rows)) if j != i]
        c, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
        pred = float(X[i] @ c)
        totalErr += abs(pred - row[4])
        print("  留出 %-14s 实测 %9.5f  预测 %9.3f  误差 %+7.3f"
              % (row[0], row[4], pred, pred - row[4]))
    print("  留一平均绝对误差 = %.3f" % (totalErr / len(rows)))

    print()
    print("对照：单变量总公式模型的拟合绝对误差均值 0.450，")
    print("      4 点模型对 O 的预测误差 1.393。")


if __name__ == "__main__":
    main()
