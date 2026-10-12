"""Report a per-scenario accuracy / ASR / LRA comparison between variants.

Reads variant snapshots written by ``scripts/build_variants.py``
(``<snapshotRoot>/<variant>/<dataset>/<scenario>/metrics.json``) and the shipped
baseline under ``outputs/scenarios`` (legacy flat layout).

    python scripts/variant_details.py --variants precision balanced
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_defenses import loadTree  # noqa: E402

ORDER = ["clean", "lfba_p05", "lfba_p10", "lfba_p20", "villain", "pmc", "amc", "composite"]
DATASETS = [f"dataset_0{index}" for index in range(1, 6)]


def loadSnapshot(root, variant):
    records = {}
    for path in glob.glob(os.path.join(root, variant, "*", "*", "metrics.json")):
        parts = path.replace("\\", "/").split("/")
        dataset, scenario = parts[-3], parts[-2]
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            continue
        validation = data.get("validationResults") or {}
        records[(dataset, scenario)] = {
            "acc": validation.get("mainTask", {}).get("accuracy"),
            "asr": (validation.get("backdoor") or {}).get("attackSuccessRate"),
            "lra": (validation.get("labelInference") or {}).get(
                "labelRecoveryAccuracy"
            ),
        }
    return records


def f(value):
    return f"{value:.4f}" if isinstance(value, (int, float)) else "-"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", dest="snapshotRoot", default="outputs/variants")
    parser.add_argument("--baseline", default=os.path.join("outputs", "scenarios"))
    parser.add_argument("--variants", nargs="+", default=["precision", "balanced", "robust"])
    arguments = parser.parse_args()

    baseline = loadTree(arguments.baseline, "flat")
    trees = {}
    for variant in arguments.variants:
        records = loadSnapshot(arguments.snapshotRoot, variant)
        if records:
            trees[variant] = records
            print(f"{variant}: {len(records)} scenarios")

    for variant, records in trees.items():
        print()
        print(f"=== {variant} : accuracy deltas vs baseline (negative = worse) ===")
        regressions = []
        for dataset in DATASETS:
            line = f"{dataset:11}"
            for scenario in ORDER:
                key = (dataset, scenario)
                base = baseline.get(key)
                current = records.get(key)
                if not base or not current:
                    line += f" {scenario[:6]:>8}"
                    continue
                if isinstance(base["acc"], float) and isinstance(current["acc"], float):
                    delta = current["acc"] - base["acc"]
                    line += f" {delta:>+8.4f}"
                    if delta < -0.01:
                        regressions.append((delta, key))
                else:
                    line += f" {'-':>8}"
            print(line)
        print(f"{'':11}" + "".join(f" {s[:6]:>8}" for s in ORDER))
        print()
        print(f"--- {variant}: accuracy regressions worse than -0.01 ---")
        for delta, key in sorted(regressions):
            base = baseline[key]
            current = records[key]
            asrText = ""
            if isinstance(base.get("asr"), float) and isinstance(current.get("asr"), float):
                asrText = f"  asr {base['asr']:.4f} -> {current['asr']:.4f}"
            print(
                f"  {key[0]:11} {key[1]:10} acc {base['acc']:.4f} -> "
                f"{current['acc']:.4f} ({delta:+.4f}){asrText}"
            )


if __name__ == "__main__":
    main()
