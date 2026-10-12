"""Recover the previous-generation (rotated) scenario results and compare them.

``outputs/scenarios/<dataset>/<scenario>/previous`` holds the generation that was
current before the latest run.  After a re-run this is the only place the earlier
defense's metrics survive, so this script reads it to quantify how much each
scenario contributes to a metric difference against the shipped baseline.

    python scripts/compare_previous.py
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_defenses import loadTree  # noqa: E402


def loadRotated(root, generation):
    records = {}
    pattern = os.path.join(root, "*", "*", generation, "metrics.json")
    for path in glob.glob(pattern):
        parts = path.replace("\\", "/").split("/")
        if len(parts) < 5:
            continue
        dataset, scenario = parts[-4], parts[-3]
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.path.join("outputs", "scenarios"))
    parser.add_argument("--generation", default="previous")
    arguments = parser.parse_args()

    baseline = loadTree(arguments.root, "flat")
    rotated = loadRotated(arguments.root, arguments.generation)
    if not rotated:
        print(f"no {arguments.generation} results found under {arguments.root}")
        return

    accDeltas = []
    asrPairs = []
    lraDeltas = []
    for key in sorted(rotated):
        base = baseline.get(key)
        rolled = rotated[key]
        if base is None:
            continue
        if isinstance(base["acc"], float) and isinstance(rolled["acc"], float):
            accDeltas.append(rolled["acc"] - base["acc"])
        if isinstance(base["asr"], float) and isinstance(rolled["asr"], float):
            asrPairs.append((rolled["asr"] - base["asr"], key, base["asr"], rolled["asr"]))
        if isinstance(base["lra"], float) and isinstance(rolled["lra"], float):
            lraDeltas.append(rolled["lra"] - base["lra"])

    mean = lambda values: sum(values) / len(values) if values else float("nan")
    print(f"scenarios compared            : {len(rotated)}")
    print(f"mean accuracy delta           : {mean(accDeltas):+.4f} (n={len(accDeltas)})")
    print(f"mean ASR delta                : {mean([p[0] for p in asrPairs]):+.4f} "
          f"(n={len(asrPairs)}, negative is better)")
    print(f"mean label-recovery delta     : {mean(lraDeltas):+.4f} "
          f"(n={len(lraDeltas)}, negative is better)")
    if asrPairs:
        asrPairs.sort(key=lambda item: -item[0])
        print()
        print("per-scenario ASR delta, worst first (negative = better than baseline):")
        for delta, key, baseValue, rolledValue in asrPairs:
            print(
                f"  {key[0]:11} {key[1]:10} {delta:+.4f}  "
                f"(baseline {baseValue:.4f} -> {rolledValue:.4f})"
            )
        total = sum(item[0] for item in asrPairs)
        worst = asrPairs[0][0]
        if len(asrPairs) > 1:
            print()
            print(
                f"excluding the single worst scenario, mean ASR delta becomes "
                f"{((total - worst) / (len(asrPairs) - 1)):+.4f}"
            )


if __name__ == "__main__":
    main()
