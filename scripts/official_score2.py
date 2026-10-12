"""Score competition variants by the official formula, with per-scenario baselines.

The published rules define the privacy and robustness terms against the UNDEFENDED
baseline for the SAME scenario:

    P   = clip((A_attack - A_recovered) / (A_attack - A_min))
    G_p = clip((A - A_min) / (0.9 * A_clean - A_min))

``A_attack`` is therefore a per-scenario quantity, not the single global mean.  Using a
global mean wrongly drives ``P`` to zero on every scenario whose undefended recovery
rate sits below that mean, which silently understates the privacy score.  This script
reads the real per-scenario baseline that the repository already stores under
``outputs/scenarios/<dataset>/<scenario>/metrics.json``.

    python scripts/official_score2.py --variants h j k
"""

import argparse
import glob
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS = [f"dataset_0{i}" for i in range(1, 6)]
LFBA = ["lfba_p05", "lfba_p10", "lfba_p20"]


def minimumAccuracy(dataset):
    return 1.0 / 6.0 if dataset == "dataset_05" else 0.5


def clip(value):
    return max(0.0, min(1.0, value))


def loadBaseline():
    """Undefended (robust_vfl) per-scenario accuracy and label recovery."""
    baseline = {}
    for path in glob.glob(str(PROJECT_ROOT / "outputs" / "scenarios" / "*" / "*"
                              / "metrics.json")):
        parts = Path(path).parts
        validation = json.load(open(path, encoding="utf-8"))["validationResults"]
        baseline[(parts[-3], parts[-2])] = {
            "acc": validation["mainTask"]["accuracy"],
            "lra": (validation.get("labelInference") or {}).get("labelRecoveryAccuracy"),
        }
    return baseline


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


def score(records, baseline):
    clean = {d: baseline[(d, "clean")]["acc"] for d in DATASETS if (d, "clean") in baseline}

    utilities = []
    for dataset in DATASETS:
        amin = minimumAccuracy(dataset)
        span = clean[dataset] - amin
        family = []
        for scenario in LFBA:
            row = records.get((dataset, scenario))
            if row and isinstance(row["acc"], float):
                family.append(clip((row["acc"] - amin) / span))
        if family:
            utilities.append(sum(family) / len(family))
        for scenario in ("villain", "pmc", "amc", "composite"):
            row = records.get((dataset, scenario))
            if row and isinstance(row["acc"], float):
                utilities.append(clip((row["acc"] - amin) / span))
    performance = 20.0 * sum(utilities) / len(utilities) if utilities else float("nan")

    privacy = []
    for (dataset, scenario), row in records.items():
        if not isinstance(row.get("lra"), float):
            continue
        base = baseline.get((dataset, scenario))
        if base is None or not isinstance(base.get("lra"), float):
            continue
        amin = minimumAccuracy(dataset)
        attackSpan = base["lra"] - amin
        attack = clip((base["lra"] - row["lra"]) / attackSpan) if attackSpan > 0 else 0.0
        guard = clip((row["acc"] - amin) / (0.9 * clean[dataset] - amin))
        privacy.append(guard * attack)
    privacyScore = 20.0 * sum(privacy) / len(privacy) if privacy else float("nan")

    security = []
    for (dataset, scenario), row in records.items():
        if not isinstance(row.get("asr"), float):
            continue
        amin = minimumAccuracy(dataset)
        guard = clip((row["acc"] - amin) / (0.9 * clean[dataset] - amin))
        resistance = 1.0 - row["asr"]
        total = guard + resistance
        security.append((2.0 * guard * resistance / total) if total > 0 else 0.0)
    securityScore = 20.0 * sum(security) / len(security) if security else float("nan")

    return performance, privacyScore, securityScore, len(privacy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+",
                        default=["precision_v3", "i", "h", "j", "k"])
    arguments = parser.parse_args()
    baseline = loadBaseline()
    print("per-scenario baseline loaded: %d scenarios" % len(baseline))
    print()
    print(f"{'变体':14} {'性能':>8} {'隐私':>8} {'鲁棒':>8} {'总分':>8}   {'n(隐私)':>8}")
    print("-" * 62)
    results = {}
    for variant in arguments.variants:
        records = loadSnapshot(variant)
        if not records:
            print(f"{variant:14} (无快照)")
            continue
        performance, privacyScore, securityScore, count = score(records, baseline)
        total = performance + privacyScore + securityScore
        results[variant] = total
        print(f"{variant:14} {performance:8.3f} {privacyScore:8.3f} {securityScore:8.3f} "
              f"{total:8.3f}   {count:8d}")
    print()
    if results:
        for variant, total in sorted(results.items(), key=lambda kv: -kv[1]):
            print(f"  {variant:14} {total:8.3f}")
    print()
    print("已验证的平台分：precision_v3=37.9599, h=41.6341, k=40.6421")
    print()
    print("该口径的实测准确度（三次平台提交）：")
    print("  precision_v3  公式 36.858  平台 37.9599  偏差 +1.102")
    print("  h             公式 42.409  平台 41.6341  偏差 -0.775")
    print("  k             公式 41.928  平台 40.6421  偏差 -1.286")
    print("  平均偏差 -0.320，绝对误差均值 1.054")
    print()
    print("对照：早期版本用 A_attack 的全局均值（而非逐场景值）时，")
    print("  绝对误差均值 1.452，且把 h 与 k 的排序判反。")
    print("  故本脚本的逐场景口径是应当使用的口径。")
    print()
    print("注意：两次以'加噪'为核心的改动（k 的 FIDP 选择式应用）在平台上损失")
    print("  0.99 分，说明该机制的本地收益未能迁移到隐藏测试集。")


if __name__ == "__main__":
    main()
