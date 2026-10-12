"""Compare the packaged defense variants scenario by scenario.

Reads the per-variant snapshots written by ``scripts/build_variants.py``
(``<snapshotRoot>/<variant>/<dataset>/<scenario>/metrics.json``), plus the shipped
baseline under ``outputs/scenarios`` (legacy flat layout), and prints:

* an aggregate table of the three scored dimensions per variant;
* the per-scenario breakdown, so the trade-off is visible where it is actually
  paid.

Reminder: the ``attackSuccessRate`` recorded in ``metrics.json`` is computed on the
**validation** split, while the leaderboard scores ASR on the **hidden test** split
with private labels.  The ASR column here is therefore a local proxy that cannot
be reproduced for the scored quantity - see ``docs/defense_report.md`` section 10.

    python scripts/compare_variants.py
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_defenses import loadTree  # noqa: E402

SCENARIO_ORDER = [
    "clean",
    "lfba_p05",
    "lfba_p10",
    "lfba_p20",
    "villain",
    "pmc",
    "amc",
    "composite",
]
DATASETS = [f"dataset_0{index}" for index in range(1, 6)]


def loadSnapshot(root):
    records = {}
    pattern = os.path.join(root, "*", "*", "metrics.json")
    for path in glob.glob(pattern):
        parts = path.replace("\\", "/").split("/")
        if len(parts) < 3:
            continue
        dataset, scenario = parts[-3], parts[-2]
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            continue
        validation = data.get("validationResults")
        if not validation:
            continue
        records[(dataset, scenario)] = {
            "acc": validation.get("mainTask", {}).get("accuracy"),
            "asr": (validation.get("backdoor") or {}).get("attackSuccessRate"),
            "lra": (validation.get("labelInference") or {}).get(
                "labelRecoveryAccuracy"
            ),
        }
    return records


def mean(values):
    usable = [v for v in values if isinstance(v, (int, float))]
    return sum(usable) / len(usable) if usable else None


def fmt(value):
    return f"{value:.4f}" if isinstance(value, (int, float)) else "-"


def summarize(label, records):
    accuracies = [v["acc"] for v in records.values()]
    asrs = [v["asr"] for v in records.values()]
    lras = [v["lra"] for v in records.values()]
    return {
        "label": label,
        "scenarios": len(records),
        "meanAcc": mean(accuracies),
        "nAcc": len([a for a in accuracies if isinstance(a, (int, float))]),
        "meanAsr": mean(asrs),
        "nAsr": len([a for a in asrs if isinstance(a, (int, float))]),
        "meanLra": mean(lras),
        "nLra": len([a for a in lras if isinstance(a, (int, float))]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", dest="snapshotRoot", default="outputs/variants")
    parser.add_argument("--baseline", default=os.path.join("outputs", "scenarios"))
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()

    baseline = loadTree(arguments.baseline, "flat")
    variants = {}
    for entry in sorted(os.listdir(arguments.snapshotRoot)):
        directory = os.path.join(arguments.snapshotRoot, entry)
        if not os.path.isdir(directory):
            continue
        records = loadSnapshot(directory)
        if records:
            variants[entry] = records

    if not variants:
        print(f"no variant snapshots under {arguments.snapshotRoot}")
        return

    summaries = [summarize("baseline (robust_vfl)", baseline)]
    summaries += [summarize(name, records) for name, records in variants.items()]

    header = (
        f"{'variant':24} {'scenarios':>9} {'meanAcc':>9} {'meanASR':>9} "
        f"{'meanLRA':>9}"
    )
    print(header)
    print("-" * len(header))
    for item in summaries:
        print(
            f"{item['label']:24} {item['scenarios']:>9} "
            f"{fmt(item['meanAcc']):>9} {fmt(item['meanAsr']):>9} "
            f"{fmt(item['meanLra']):>9}"
        )
    print()
    print("meanASR is the validation-split proxy; the scored ASR uses the hidden test split.")
    print()

    names = list(variants)
    for dataset in DATASETS:
        print(f"=== {dataset} ===")
        header = f"{'scenario':12} {'base_acc':>9}"
        for name in names:
            header += f" {name[:10] + '_acc':>13} {'asr':>8}"
        print(header)
        for scenario in SCENARIO_ORDER:
            key = (dataset, scenario)
            base = baseline.get(key)
            if base is None and all(key not in variants[n] for n in names):
                continue
            line = f"{scenario:12} {fmt(base['acc'] if base else None):>9}"
            for name in names:
                record = variants[name].get(key)
                line += (
                    f" {fmt(record['acc'] if record else None):>13} "
                    f"{fmt(record['asr'] if record else None):>8}"
                )
            print(line)
        print()

    if arguments.json:
        with open(arguments.json, "w", encoding="utf-8") as handle:
            json.dump({"summaries": summaries}, handle, ensure_ascii=False, indent=2)
        print(f"# wrote {arguments.json}")


if __name__ == "__main__":
    main()
