"""Render the full 40-scenario comparison of the two official runs as markdown."""

import sys

sys.path.insert(0, "scripts")
from compare_defenses import loadTree  # noqa: E402

ORDER = ["clean", "lfba_p05", "lfba_p10", "lfba_p20", "villain", "pmc", "amc", "composite"]
DATASETS = [f"dataset_0{i}" for i in range(1, 6)]


def f(value):
    return f"{value:.4f}" if isinstance(value, (int, float)) else "–"


def main():
    base = loadTree("outputs/scenarios", "flat")
    new = loadTree("outputs/scenarios", "current")
    print("| 数据集 | 场景 | 自带 acc | 新 acc | Δacc | 自带 ASR | 新 ASR | 自带 LRA | 新 LRA |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    accDeltas, asrDeltas, lraDeltas = [], [], []
    for dataset in DATASETS:
        for scenario in ORDER:
            b = base.get((dataset, scenario))
            c = new.get((dataset, scenario))
            if not b or not c:
                continue
            dacc = (
                f"{c['acc'] - b['acc']:+.4f}"
                if isinstance(c["acc"], float) and isinstance(b["acc"], float)
                else "–"
            )
            if isinstance(c["acc"], float) and isinstance(b["acc"], float):
                accDeltas.append(c["acc"] - b["acc"])
            if isinstance(c["asr"], float) and isinstance(b["asr"], float):
                asrDeltas.append(c["asr"] - b["asr"])
            if isinstance(c["lra"], float) and isinstance(b["lra"], float):
                lraDeltas.append(c["lra"] - b["lra"])
            print(
                f"| {dataset} | {scenario} | {f(b['acc'])} | {f(c['acc'])} | {dacc} | "
                f"{f(b['asr'])} | {f(c['asr'])} | {f(b['lra'])} | {f(c['lra'])} |"
            )
    print()
    mean = lambda values: sum(values) / len(values) if values else None
    print(f"mean accuracy delta   : {mean(accDeltas):+.4f}  (n={len(accDeltas)})")
    print(f"mean ASR delta        : {mean(asrDeltas):+.4f}  (n={len(asrDeltas)}, negative is better)")
    print(f"mean LRA delta        : {mean(lraDeltas):+.4f}  (n={len(lraDeltas)}, negative is better)")
    improved = sum(1 for d in accDeltas if d > 0)
    print(f"accuracy improved in {improved}/{len(accDeltas)} scenarios")


if __name__ == "__main__":
    main()
