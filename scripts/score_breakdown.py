"""Reproduce the three official score components for one variant, from raw inputs.

Prints every intermediate so the arithmetic can be audited rather than trusted.

    python scripts/score_breakdown.py --variant o

Note on A_clean: the published rules define the performance term against the OFFICIAL
undefended clean baseline for each dataset.  This script reads that value from the
harness's own undefended run (outputs/scenarios/<dataset>/clean/metrics.json) rather
than from a hard-coded table, and prints it so the value being used is explicit.  An
earlier hard-coded table was wrong by up to 0.022 on dataset_02, which shifts the
guard factors G_p and G_r and therefore all three components.
"""

import argparse
import glob
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS = [f"dataset_0{i}" for i in range(1, 6)]
LFBA = ("lfba_p05", "lfba_p10", "lfba_p20")
BACKDOOR_SCENARIOS = ("villain", "pmc", "amc", "composite")


def minimumAccuracy(dataset):
    return 1.0 / 6.0 if dataset == "dataset_05" else 0.5


def clip(value):
    return max(0.0, min(1.0, value))


def loadBaseline():
    out = {}
    for path in glob.glob(str(PROJECT_ROOT / "outputs" / "scenarios" / "*" / "*"
                              / "metrics.json")):
        parts = Path(path).parts
        v = json.load(open(path, encoding="utf-8"))["validationResults"]
        out[(parts[-3], parts[-2])] = {
            "acc": v["mainTask"]["accuracy"],
            "asr": (v.get("backdoor") or {}).get("attackSuccessRate"),
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="o")
    arguments = parser.parse_args()

    baseline = loadBaseline()
    records = loadSnapshot(arguments.variant)
    if not records:
        print("no snapshot for", arguments.variant)
        return

    clean = {d: baseline[(d, "clean")]["acc"] for d in DATASETS if (d, "clean") in baseline}

    print("=" * 78)
    print("输入常数（A_clean = harness 无防御基线 clean 场景精度）")
    print("=" * 78)
    print("  %-11s %12s %12s %10s" % ("数据集", "A_clean", "0.9*A_clean", "A_min"))
    for d in DATASETS:
        print("  %-11s %12.6f %12.6f %10.4f"
              % (d, clean[d], 0.9 * clean[d], minimumAccuracy(d)))

    print()
    print("=" * 78)
    print("1) 性能项  S_perf = 20 * mean(U),  U = clip((A - A_min)/(A_clean - A_min))")
    print("=" * 78)
    utilities = []
    for d in DATASETS:
        a = minimumAccuracy(d)
        span = clean[d] - a
        family = []
        for s in LFBA:
            if (d, s) in records:
                family.append(clip((records[(d, s)]["acc"] - a) / span))
        if family:
            familyMean = sum(family) / len(family)
            utilities.append(familyMean)
            print("  %-11s LFBA 族内均值 %.6f   (三档: %s)"
                  % (d, familyMean, ", ".join("%.6f" % v for v in family)))
        for s in BACKDOOR_SCENARIOS:
            if (d, s) in records:
                value = clip((records[(d, s)]["acc"] - a) / span)
                utilities.append(value)
                print("  %-11s %-10s U = %.6f" % (d, s, value))
    meanU = sum(utilities) / len(utilities)
    print("  条目数 %d    mean(U) = %.6f    S_perf = %.6f"
          % (len(utilities), meanU, 20 * meanU))

    print()
    print("=" * 78)
    print("2) 隐私项  S_priv = 20 * mean(Q_p)")
    print("     P   = clip((A_attack - A_recovered)/(A_attack - A_min))   A_attack 逐场景")
    print("     G_p = clip((A - A_min)/(0.9*A_clean - A_min))")
    print("     Q_p = G_p * P")
    print("=" * 78)
    print("  %-24s %9s %9s %9s %9s %9s" % ("场景", "A_attack", "A_rec", "P", "G_p", "Q_p"))
    privacies = []
    for key in sorted(records):
        row = records[key]
        if not isinstance(row.get("lra"), float) or key not in baseline:
            continue
        base = baseline[key].get("lra")
        if not isinstance(base, float):
            continue
        d = key[0]
        a = minimumAccuracy(d)
        span = base - a
        attack = clip((base - row["lra"]) / span) if span > 0 else 0.0
        guard = clip((row["acc"] - a) / (0.9 * clean[d] - a))
        value = guard * attack
        privacies.append(value)
        print("  %-24s %9.4f %9.4f %9.4f %9.4f %9.4f"
              % ("%s/%s" % key, base, row["lra"], attack, guard, value))
    meanQp = sum(privacies) / len(privacies)
    print("  场景数 %d    mean(Q_p) = %.6f    S_priv = %.6f"
          % (len(privacies), meanQp, 20 * meanQp))

    print()
    print("=" * 78)
    print("3) 鲁棒项  S_sec = 20 * mean(Q_r)")
    print("     G_r = clip((A - A_min)/(0.9*A_clean - A_min))")
    print("     R   = 1 - ASR")
    print("     Q_r = 2*G_r*R/(G_r + R)")
    print("=" * 78)
    securities = []
    for key in sorted(records):
        row = records[key]
        if not isinstance(row.get("asr"), float):
            continue
        d = key[0]
        a = minimumAccuracy(d)
        guard = clip((row["acc"] - a) / (0.9 * clean[d] - a))
        resistance = 1.0 - row["asr"]
        total = guard + resistance
        value = (2.0 * guard * resistance / total) if total > 0 else 0.0
        securities.append(value)
    meanQr = sum(securities) / len(securities)
    print("  场景数 %d    mean(Q_r) = %.6f    S_sec = %.6f"
          % (len(securities), meanQr, 20 * meanQr))

    print()
    print("=" * 78)
    print("合计:  %.6f + %.6f + %.6f = %.6f"
          % (20 * meanU, 20 * meanQp, 20 * meanQr,
             20 * meanU + 20 * meanQp + 20 * meanQr))
    print("=" * 78)


if __name__ == "__main__":
    main()
