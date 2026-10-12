"""Rank the privacy scenarios by headroom, to locate where the real loss is.

The privacy term is 20 * mean(Q_p) over 15 scenarios. A scenario can only contribute if its
undefended recovery rate is meaningfully above chance, so headroom = A_attack - A_min. A
scenario with large headroom and small P is unexploited opportunity; a scenario with small
headroom cannot contribute whatever the defence does.

    <python> scripts/privacy_headroom_rank.py
"""

import glob
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS = [f"dataset_0{i}" for i in range(1, 6)]
SCENARIOS = ("pmc", "amc", "composite")


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
        }
    return out


def main():
    baseline = load(PROJECT_ROOT / "outputs" / "scenarios")
    variant = load(PROJECT_ROOT / "outputs" / "three_variants" / "o")
    clean = {d: baseline[(d, "clean")]["acc"] for d in DATASETS}

    rows = []
    for dataset in DATASETS:
        a = minimumAccuracy(dataset)
        for scenario in SCENARIOS:
            key = (dataset, scenario)
            base = baseline.get(key)
            row = variant.get(key)
            if not base or not row:
                continue
            if not isinstance(base.get("lra"), float) or not isinstance(row.get("lra"), float):
                continue
            headroom = base["lra"] - a
            span = headroom
            p = clip((base["lra"] - row["lra"]) / span) if span > 0 else 0.0
            g = clip((row["acc"] - a) / (0.9 * clean[dataset] - a))
            # Lost contribution = what a perfect P would add over the current P.
            lost = g * (1.0 - p)
            rows.append((lost, key, base["lra"], a, headroom, row["lra"], p, g))

    rows.sort(reverse=True)
    print("  %-22s %9s %9s %9s %9s %8s %9s" %
          ("场景", "A_attack", "A_min", "余量", "A_rec", "P", "未得分"))
    print("  " + "-" * 86)
    totalLost = 0.0
    for lost, key, base, a, headroom, rec, p, g in rows:
        totalLost += lost
        print("  %-22s %9.4f %9.4f %9.4f %9.4f %8.4f %9.4f"
              % ("%s/%s" % key, base, a, headroom, rec, p, lost))

    print()
    print("  Σ 未得分 = %.4f  -> 隐私项可提升上限 = %.4f 分"
          % (totalLost, 20 * totalLost / len(rows)))
    print()
    print("  === 按余量分档 ===")
    for low, high, label in [(0.0, 0.15, "低余量 (<0.15)"),
                             (0.15, 0.35, "中余量 (0.15-0.35)"),
                             (0.35, 1.0, "高余量 (>=0.35)")]:
        group = [r for r in rows if low <= r[4] < high]
        if not group:
            continue
        lost = sum(r[0] for r in group)
        meanP = sum(r[6] for r in group) / len(group)
        print("    %-20s %2d 个场景  平均 P=%.4f  Σ未得分=%.4f  可提升 %.4f 分"
              % (label, len(group), meanP, lost, 20 * lost / len(rows)))

    print()
    print("  === 优先改进目标（未得分最高）===")
    for lost, key, base, a, headroom, rec, p, g in rows[:5]:
        print("    %-22s 余量 %.4f 却只拿到 P=%.4f  ->  未得分 %.4f"
              % ("%s/%s" % key, headroom, p, lost))


if __name__ == "__main__":
    main()
