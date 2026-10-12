"""Break the official robustness term down per scenario for several variants.

The robustness term is a harmonic mean per scenario,

    G_r = clip((A - A_min) / (0.9 * A_clean - A_min))
    R   = 1 - ASR
    Q_r = 2 * G_r * R / (G_r + R)

and simple per-scenario thinking about it proved unreliable here, so this prints the
actual contribution of every scenario that carries a backdoor, for each variant, and
where the differences come from.

    python scripts/robustness_breakdown.py --variants h m n
"""

import argparse
import glob
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN = {
    "dataset_01": 0.7308,
    "dataset_02": 0.9699,
    "dataset_03": 0.9783,
    "dataset_04": 0.8552,
    "dataset_05": 0.9874,
}


def minimum(dataset):
    return 1.0 / 6.0 if dataset == "dataset_05" else 0.5


def load(variant):
    records = {}
    for path in glob.glob(str(PROJECT_ROOT / "outputs" / "three_variants" / variant
                              / "*" / "*" / "metrics.json")):
        parts = Path(path).parts
        validation = json.load(open(path, encoding="utf-8"))["validationResults"]
        records[(parts[-3], parts[-2])] = {
            "acc": validation["mainTask"]["accuracy"],
            "asr": (validation.get("backdoor") or {}).get("attackSuccessRate"),
        }
    return records


def contribution(row, dataset):
    amin = minimum(dataset)
    guard = max(0.0, min(1.0, (row["acc"] - amin) / (0.9 * CLEAN[dataset] - amin)))
    resistance = 1.0 - row["asr"]
    total = guard + resistance
    return (2.0 * guard * resistance / total) if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=["h", "m", "n"])
    arguments = parser.parse_args()

    tables = {v: load(v) for v in arguments.variants}
    keys = sorted(k for k in set().union(*[set(t) for t in tables.values()])
                  if any(t.get(k, {}).get("asr") is not None for t in tables.values()))

    header = "%-24s" % "scenario"
    for v in arguments.variants:
        header += " %10s" % v
    print(header + " %10s %10s" % ("d(1st-last)", "sum"))
    print("-" * len(header))
    totals = {v: 0.0 for v in arguments.variants}
    for key in keys:
        row = "%-24s" % ("%s/%s" % key)
        values = []
        for v in arguments.variants:
            rec = tables[v].get(key)
            q = contribution(rec, key[0]) if rec and rec["asr"] is not None else None
            values.append(q)
        for q in values:
            row += " %10s" % ("%.4f" % q if q is not None else "-")
        for v, q in zip(arguments.variants, values):
            if q is not None:
                totals[v] += q
        if all(q is not None for q in values):
            row += " %+10.4f %10.4f" % (values[-1] - values[0], sum(values))
        print(row)

    print()
    print("鲁棒项 = 20 * 合计(Q_r) / 场景数（场景数 = 25：lfba 合并为 1 + villain/pmc/amc/composite）")
    print("本表按 lfba 三个强度各自列出，故这里的合计只用于比较，不是官方口径的分子。")


if __name__ == "__main__":
    main()
