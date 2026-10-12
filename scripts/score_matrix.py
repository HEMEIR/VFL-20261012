"""Compute the three official score components for every variant in the grid matrix.

Reads the 40-scenario snapshots and prints, per variant, the performance / privacy /
security components and the total, plus the per-scenario decomposition so the arithmetic
can be audited.

    <python> scripts/score_matrix.py
    <python> scripts/score_matrix.py --variants b0 b1_clip_uniform
"""

import argparse
import glob
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS = [f"dataset_0{i}" for i in range(1, 6)]
LFBA = ("lfba_p05", "lfba_p10", "lfba_p20")
BACKDOOR_SCENARIOS = ("villain", "pmc", "amc", "composite")

DEFAULT_VARIANTS = [
    "b0",
    "b1_clip_uniform",
    "b2_clip_adaptive",
    "b3_quant_uniform",
    "b4_quant_adaptive",
    "b5_sparsify_adaptive",
]

LABELS = {
    "b0": "基线（O 的配置）",
    "b1_clip_uniform": "梯度裁剪（统一 0.9）",
    "b2_clip_adaptive": "梯度裁剪（自适应 0.9）",
    "b3_quant_uniform": "梯度量化（统一）",
    "b4_quant_adaptive": "梯度量化（自适应）",
    "b5_sparsify_adaptive": "梯度稀疏化（自适应）",
}


def minimumAccuracy(dataset):
    return 1.0 / 6.0 if dataset == "dataset_05" else 0.5


def clip(value):
    return max(0.0, min(1.0, value))


def loadRecords(root):
    out = {}
    for path in glob.glob(str(root / "*" / "*" / "metrics.json")):
        parts = Path(path).parts
        v = json.load(open(path, encoding="utf-8"))["validationResults"]
        out[(parts[-3], parts[-2])] = {
            "acc": v["mainTask"]["accuracy"],
            "asr": (v.get("backdoor") or {}).get("attackSuccessRate"),
            "lra": (v.get("labelInference") or {}).get("labelRecoveryAccuracy"),
        }
    return out


def score(records, baseline):
    clean = {d: baseline[(d, "clean")]["acc"] for d in DATASETS}
    utilities = []
    for dataset in DATASETS:
        a = minimumAccuracy(dataset)
        span = clean[dataset] - a
        family = [clip((records[(dataset, s)]["acc"] - a) / span)
                  for s in LFBA if (dataset, s) in records]
        if family:
            utilities.append(sum(family) / len(family))
        for scenario in BACKDOOR_SCENARIOS:
            if (dataset, scenario) in records:
                utilities.append(clip((records[(dataset, scenario)]["acc"] - a) / span))
    meanU = sum(utilities) / len(utilities) if utilities else float("nan")

    privacies = []
    for key, row in records.items():
        if not isinstance(row.get("lra"), float) or key not in baseline:
            continue
        base = baseline[key].get("lra")
        if not isinstance(base, float):
            continue
        dataset = key[0]
        a = minimumAccuracy(dataset)
        span = base - a
        attack = clip((base - row["lra"]) / span) if span > 0 else 0.0
        guard = clip((row["acc"] - a) / (0.9 * clean[dataset] - a))
        privacies.append(guard * attack)
    meanQp = sum(privacies) / len(privacies) if privacies else float("nan")

    securities = []
    for key, row in records.items():
        if not isinstance(row.get("asr"), float):
            continue
        dataset = key[0]
        a = minimumAccuracy(dataset)
        guard = clip((row["acc"] - a) / (0.9 * clean[dataset] - a))
        resistance = 1.0 - row["asr"]
        total = guard + resistance
        securities.append((2.0 * guard * resistance / total) if total > 0 else 0.0)
    meanQr = sum(securities) / len(securities) if securities else float("nan")

    return {
        "perf": 20.0 * meanU,
        "priv": 20.0 * meanQp,
        "sec": 20.0 * meanQr,
        "total": 20.0 * (meanU + meanQp + meanQr),
        "meanU": meanU,
        "meanQp": meanQp,
        "meanQr": meanQr,
        "counts": (len(utilities), len(privacies), len(securities)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--detail", default=None,
                        help="print the per-scenario decomposition for this variant")
    arguments = parser.parse_args()

    baseline = loadRecords(PROJECT_ROOT / "outputs" / "scenarios")
    clean = {d: baseline[(d, "clean")]["acc"] for d in DATASETS}

    print("A_clean（harness 无防御基线）:", {d: round(clean[d], 6) for d in DATASETS})
    print()
    print("  %-24s %9s %9s %9s %10s  %s" % ("方案", "性能", "隐私", "鲁棒", "公式合计", "条目"))
    results = {}
    for variant in arguments.variants:
        records = loadRecords(PROJECT_ROOT / "outputs" / "three_variants" / variant)
        if len(records) < 40:
            print("  %-24s 快照不完整 (%d/40)，跳过" % (variant, len(records)))
            continue
        result = score(records, baseline)
        results[variant] = result
        nPerf, nPriv, nSec = result["counts"]
        print("  %-24s %9.6f %9.6f %9.6f %10.6f  %d/%d/%d"
              % (variant, result["perf"], result["priv"], result["sec"], result["total"],
                 nPerf, nPriv, nSec))
    print()
    print("  说明（%s）" % LABELS.get("b0", ""))
    for variant in arguments.variants:
        if variant in results:
            print("    %-24s %s" % (variant, LABELS.get(variant, "")))
    print()
    if len(results) > 1:
        base = results.get("b0")
        if base:
            print("  相对基线的差值:")
            print("  %-24s %9s %9s %9s %10s" % ("方案", "Δ性能", "Δ隐私", "Δ鲁棒", "Δ合计"))
            for variant, result in results.items():
                print("  %-24s %+9.6f %+9.6f %+9.6f %+10.6f"
                      % (variant, result["perf"] - base["perf"],
                         result["priv"] - base["priv"],
                         result["sec"] - base["sec"],
                         result["total"] - base["total"]))
            print()

    if arguments.detail:
        records = loadRecords(PROJECT_ROOT / "outputs" / "three_variants" / arguments.detail)
        print("  逐场景分解: %s" % arguments.detail)
        print("  %-24s %9s %9s %9s %8s %9s" % ("场景", "A_attack", "A_rec", "P", "G_p", "Q_p"))
        for key in sorted(records):
            row = records[key]
            if not isinstance(row.get("lra"), float) or key not in baseline:
                continue
            base = baseline[key].get("lra")
            if not isinstance(base, float):
                continue
            dataset = key[0]
            a = minimumAccuracy(dataset)
            span = base - a
            attack = clip((base - row["lra"]) / span) if span > 0 else 0.0
            guard = clip((row["acc"] - a) / (0.9 * clean[dataset] - a))
            print("  %-24s %9.4f %9.4f %9.4f %8.4f %9.4f"
                  % ("%s/%s" % key, base, row["lra"], attack, guard, guard * attack))


if __name__ == "__main__":
    main()
