"""Quantify what each direction-B/C variant would add to the privacy term.

Only the scenarios the pilot covered are recomputed; the rest keep candidate O's values, so
the delta is a LOWER bound on the full-set effect (a change that helps here could also help
elsewhere, but nothing is assumed).

    <python> scripts/eval_pe_candidates.py
"""

import glob
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRID = PROJECT_ROOT / "outputs" / "grid" / "PE"
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


def qp(base, row, clean, dataset):
    a = minimumAccuracy(dataset)
    span = base["lra"] - a
    p = clip((base["lra"] - row["lra"]) / span) if span > 0 else 0.0
    g = clip((row["acc"] - a) / (0.9 * clean[dataset] - a))
    return g * p, p, g


def main():
    baseline = load(PROJECT_ROOT / "outputs" / "scenarios")
    candidateO = load(PROJECT_ROOT / "outputs" / "three_variants" / "o")
    clean = {d: baseline[(d, "clean")]["acc"] for d in DATASETS}

    covered = {
        "dataset_05_composite": ("dataset_05", "composite"),
        "dataset_02_composite": ("dataset_02", "composite"),
        "dataset_04_pmc": ("dataset_04", "pmc"),
        "dataset_03_composite": ("dataset_03", "composite"),
    }

    variants = sorted(p.name for p in GRID.iterdir() if p.is_dir())
    print("  %-22s %-22s %9s %9s %9s %10s"
          % ("变体", "场景", "Q_p 基线", "Q_p 变体", "ΔQ_p", "P/Q_p 说明"))
    print("  " + "-" * 92)

    summary = {}
    for variant in variants:
        totalDelta = 0.0
        coveredCount = 0
        for scenarioFile, key in covered.items():
            path = GRID / variant / scenarioFile / "metrics.json"
            if not path.is_file():
                continue
            v = json.load(open(path, encoding="utf-8"))["validationResults"]
            row = {"acc": float(v["mainTask"]["accuracy"]),
                   "lra": (v.get("labelInference") or {}).get("labelRecoveryAccuracy")}
            if not isinstance(row["lra"], float):
                continue
            base = baseline.get(key)
            old = candidateO.get(key)
            if not base or not old or not isinstance(base.get("lra"), float):
                continue
            if not isinstance(old.get("lra"), float):
                continue
            newQ, newP, newG = qp(base, row, clean, key[0])
            oldQ, oldP, oldG = qp(base, old, clean, key[0])
            delta = newQ - oldQ
            totalDelta += delta
            coveredCount += 1
            note = ""
            if newP > oldP + 1e-6:
                note = "P 提升"
            elif newP < oldP - 1e-6:
                note = "P 下降"
            print("  %-22s %-22s %9.4f %9.4f %+9.4f  %s"
                  % (variant, scenarioFile, oldQ, newQ, delta, note))
        summary[variant] = (totalDelta, coveredCount)
        print()

    print("  === 仅按覆盖场景计算的隐私项增量（全 15 场景平均）===")
    print("  %-22s %14s %14s" % ("变体", "Σ覆盖ΔQ_p", "隐私项增量"))
    for variant, (delta, count) in sorted(summary.items(), key=lambda kv: -kv[1][0]):
        print("  %-22s %14.4f %+14.4f  (覆盖 %d 场景)"
              % (variant, delta, 20.0 * delta / 15.0, count))
    print()
    print("  注：分母固定为 15（评分要求），未覆盖场景按候选 O 的原值计入，")
    print("      因此这是增量下界；未覆盖场景若也受益，实际增量更大。")


if __name__ == "__main__":
    main()
