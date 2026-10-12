"""Compute the official competition score from local per-scenario metrics.

Implements the published scoring rules exactly:

  1. Model performance (20 pts)
       U   = clip((A - A_min) / (A_clean - A_min))
       S_perf = 20 * mean(U)
     Clean scenarios are EXCLUDED from the mean; they only supply A_clean.  The three
     LFBA poison rates are averaged within the LFBA attack family first, so LFBA carries
     one entry per dataset rather than three.

  2. Privacy (20 pts)
       P   = clip((A_attack - A_recovered) / (A_attack - A_min))
       G_p = clip((A - A_min) / (0.9 * A_clean - A_min))
       Q_p = G_p * P
       S_priv = 20 * mean(Q_p)

  3. Security / robustness (20 pts)
       G_r = clip((A - A_min) / (0.9 * A_clean - A_min))
       R   = 1 - ASR
       Q_r = 2 G_r R / (G_r + R)
       S_sec = 20 * mean(Q_r)

A_min is 0.5 for two-class tasks (dataset_01..04) and 1/6 for the six-class task
(dataset_05).
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS = [f"dataset_0{i}" for i in range(1, 6)]
LFBA = ["lfba_p05", "lfba_p10", "lfba_p20"]

# robust_vfl clean-scenario accuracy supplied the reference, measured earlier.
CLEAN_ACCURACY = {
    "dataset_01": 0.7308,
    "dataset_02": 0.9699,
    "dataset_03": 0.9783,
    "dataset_04": 0.8552,
    "dataset_05": 0.9874,
}
# robust_vfl mean label-recovery accuracy on the label-inference scenarios, i.e. the
# undefended attack strength A_attack.
ATTACK_LRA = 0.7332


def minimumAccuracy(dataset):
    return 1.0 / 6.0 if dataset == "dataset_05" else 0.5


def clip(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def loadTree(root, layout):
    pattern = {
        "current": os.path.join(root, "*", "*", "current", "metrics.json"),
        "flat": os.path.join(root, "*", "*", "metrics.json"),
    }[layout]
    records = {}
    for path in glob.glob(pattern):
        parts = Path(path).parts
        dataset, scenario = parts[-4], parts[-3] if layout == "current" else parts[-3], parts[-2]
        if layout == "flat":
            dataset, scenario = parts[-3], parts[-2]
        try:
            validation = json.load(open(path, encoding="utf-8"))["validationResults"]
        except Exception:
            continue
        records[(dataset, scenario)] = {
            "acc": validation["mainTask"]["accuracy"],
            "asr": (validation.get("backdoor") or {}).get("attackSuccessRate"),
            "lra": (validation.get("labelInference") or {}).get("labelRecoveryAccuracy"),
        }
    return records


def loadSnapshot(variant):
    root = PROJECT_ROOT / "outputs" / "three_variants" / variant
    records = {}
    for path in glob.glob(str(root / "*" / "*" / "metrics.json")):
        parts = Path(path).parts
        validation = json.load(open(path, encoding="utf-8"))["validationResults"]
        records[(parts[-3], parts[-2])] = {
            "acc": validation["mainTask"]["accuracy"],
            "asr": (validation.get("backdoor") or {}).get("attackSuccessRate"),
            "lra": (validation.get("labelInference") or {}).get("labelRecoveryAccuracy"),
        }
    return records


def score(records):
    # ---- 1. performance: LFBA averaged within the family first, clean excluded
    utilities = []
    for dataset in DATASETS:
        amin = minimumAccuracy(dataset)
        span = CLEAN_ACCURACY[dataset] - amin
        entries = []
        for scenario in LFBA:
            row = records.get((dataset, scenario))
            if row and isinstance(row["acc"], float):
                entries.append(clip((row["acc"] - amin) / span))
        if entries:
            utilities.append(sum(entries) / len(entries))        # LFBA family -> one entry
        for scenario in ("villain", "pmc", "amc", "composite"):
            row = records.get((dataset, scenario))
            if row and isinstance(row["acc"], float):
                utilities.append(clip((row["acc"] - amin) / span))  # each its own entry
    performance = 20.0 * (sum(utilities) / len(utilities)) if utilities else float("nan")

    # ---- 2. privacy
    privacies = []
    for (dataset, scenario), row in records.items():
        if not isinstance(row.get("lra"), float):
            continue
        amin = minimumAccuracy(dataset)
        spanAttack = ATTACK_LRA - amin
        attack = clip((ATTACK_LRA - row["lra"]) / spanAttack) if spanAttack > 0 else 0.0
        denominator = 0.9 * CLEAN_ACCURACY[dataset] - amin
        guard = clip((row["acc"] - amin) / denominator) if denominator > 0 else 0.0
        privacies.append(guard * attack)
    privacy = 20.0 * (sum(privacies) / len(privacies)) if privacies else float("nan")

    # ---- 3. security
    securities = []
    for (dataset, scenario), row in records.items():
        if not isinstance(row.get("asr"), float):
            continue
        amin = minimumAccuracy(dataset)
        denominator = 0.9 * CLEAN_ACCURACY[dataset] - amin
        guard = clip((row["acc"] - amin) / denominator) if denominator > 0 else 0.0
        resistance = 1.0 - row["asr"]
        total = guard + resistance
        securities.append((2.0 * guard * resistance / total) if total > 0 else 0.0)
    security = 20.0 * (sum(securities) / len(securities)) if securities else float("nan")

    return performance, privacy, security, len(utilities), len(privacies), len(securities)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+",
                        default=["precision_v3", "jitter_villain", "nojitter",
                                 "norm_excess", "villain_gate"])
    arguments = parser.parse_args()

    print(f"{'变体':16} {'性能':>7} {'隐私':>7} {'鲁棒':>7} {'总分':>7}   {'n(性能/隐私/鲁棒)':>16}")
    print("-" * 74)
    for variant in arguments.variants:
        records = loadSnapshot(variant)
        if not records:
            print(f"{variant:16} (无快照)")
            continue
        performance, privacy, security, nu, np_, ns = score(records)
        total = performance + privacy + security
        print(f"{variant:16} {performance:7.3f} {privacy:7.3f} {security:7.3f} "
              f"{total:7.3f}   {nu:5d}/{np_:3d}/{ns}")
    print()
    print("已知平台分：precision_v3 = 37.9599, jitter_villain = 38.1100")


if __name__ == "__main__":
    main()
