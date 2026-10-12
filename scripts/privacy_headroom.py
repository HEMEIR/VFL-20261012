"""Break down the privacy component per scenario, with the local headroom of each.

The privacy term per scenario is
    Q_p = G_p * clip((A_attack - A_recovered) / (A_attack - A_min))
so the DENOMINATOR ``A_attack - A_min`` is the local headroom: a scenario whose undefended
recovery rate is already near chance cannot contribute no matter what a defence does, and a
scenario with a large denominator has room.

That makes "how much of a candidate's privacy gain sits in high-headroom scenarios" a
quantity computable from LOCAL data only - unlike the gain itself, whose transfer to the
hidden test is what failed for candidate p and cc_ramp100.

    <python> scripts/privacy_headroom.py --variants o cc_ramp100
"""

import argparse
import glob
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS = [f"dataset_0{i}" for i in range(1, 6)]


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
            "acc": float(v["mainTask"]["accuracy"]),
            "lra": (v.get("labelInference") or {}).get("labelRecoveryAccuracy"),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=["o", "cc_ramp100"])
    arguments = parser.parse_args()

    baseline = loadRecords(PROJECT_ROOT / "outputs" / "scenarios")
    clean = {d: baseline[(d, "clean")]["acc"] for d in DATASETS}

    print("%-22s %9s %9s %8s %9s %9s %9s"
          % ("场景", "A_attack", "A_min", "余量", "P(o)", "P(变体)", "ΔP"))
    print("-" * 82)
    rows = []
    for dataset in DATASETS:
        for scenario in ("pmc", "amc", "composite"):
            key = (dataset, scenario)
            if key not in baseline:
                continue
            base = baseline[key].get("lra")
            if not isinstance(base, float):
                continue
            a = minimumAccuracy(dataset)
            headroom = base - a
            values = {}
            for variant in arguments.variants:
                records = loadRecords(PROJECT_ROOT / "outputs" / "three_variants" / variant)
                row = records.get(key)
                if row is None or not isinstance(row.get("lra"), float):
                    values[variant] = None
                    continue
                values[variant] = clip((base - row["lra"]) / headroom) if headroom > 0 else 0.0
            rows.append((key, base, a, headroom, values))

    for key, base, a, headroom, values in rows:
        first = arguments.variants[0]
        p0 = values[first]
        rest = " ".join(
            "%9s" % ("%.4f" % values[v] if values.get(v) is not None else "-")
            for v in arguments.variants[1:]
        )
        delta = ""
        if len(arguments.variants) > 1:
            last = arguments.variants[-1]
            if values.get(last) is not None and p0 is not None:
                delta = "%+9.4f" % (values[last] - p0)
        print("%-22s %9.4f %9.4f %8.4f %9s %s %s"
              % ("%s/%s" % key, base, a, headroom,
                 "%.4f" % p0 if p0 is not None else "-", rest, delta))

    print()
    print("说明：余量 = A_attack - A_min，即该场景本地可供改善的最大幅度。")
    print("      余量小的场景（攻击本就接近随机）对隐私项几乎无法贡献。")
    for variant in arguments.variants:
        total = 0.0
        high = 0.0
        for key, base, a, headroom, values in rows:
            v = values.get(variant)
            if v is None:
                continue
            total += v
            if headroom >= 0.15:
                high += v
        print("  %-12s ΣQ_p = %7.4f   其中位于高余量场景(>=0.15)的部分 = %7.4f (%.1f%%)"
              % (variant, total, high, 100.0 * high / total if total > 0 else 0.0))


if __name__ == "__main__":
    main()
