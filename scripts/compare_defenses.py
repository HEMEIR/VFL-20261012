"""Compare the shipped baseline defense against a candidate across all 40 scenarios.

Reads the official scenario results (``outputs/scenarios``) as the reference and
one or more candidate result trees, then prints a per-scenario delta table and an
aggregate summary of the three scored dimensions:

* performance - main-task accuracy (higher is better)
* robustness  - backdoor attack success rate (lower is better)
* privacy     - label-recovery accuracy (lower is better)

    python scripts/compare_defenses.py --candidate outputs/final/purified_full
"""

import argparse
import glob
import json
import os

REFERENCE_ROOT = os.path.join("outputs", "scenarios")

# The shipped baseline keeps its official results under outputs/scenarios/.
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


def resolveKey(path):
    """Return ``(dataset, scenario)`` for a metrics file, or ``None``."""
    parts = path.replace("\\", "/").split("/")
    # Official layout: <root>/<dataset>/<scenario>/current/metrics.json - the
    # scenario directory sits two levels above the file.
    if (
        len(parts) >= 4
        and parts[-2] in {"current", "previous"}
        and parts[-4].startswith("dataset_")
    ):
        return parts[-4], parts[-3]
    # Development-sweep layout: <root>/<dataset>_<scenario>/metrics.json, where
    # the scenario suffix is everything after the dataset number.
    if len(parts) >= 2 and parts[-2].startswith("dataset_"):
        tokens = parts[-2].split("_")
        if len(tokens) < 3:
            return None
        return "_".join(tokens[:2]), "_".join(tokens[2:])
    # Legacy flat official layout: <root>/<dataset>/<scenario>/metrics.json.
    if len(parts) >= 3 and parts[-3].startswith("dataset_"):
        return parts[-3], parts[-2]
    return None


def loadTree(root, layout="all"):
    """Map ``dataset/scenario`` to the metrics of the newest matching run.

    ``layout='current'`` reads only the rotated run results under ``current/``,
    which is what a fresh official re-run produces; ``layout='flat'`` reads only
    the legacy flat files that ship with the repository, which is the shipped
    baseline; ``layout='all'`` accepts either and prefers a rotated result.  The
    distinction matters because a re-run writes into the same tree as the
    baseline it is being compared against.
    """
    records = {}
    for path in glob.glob(os.path.join(root, "**", "metrics.json"), recursive=True):
        key = resolveKey(path)
        if key is None:
            continue
        parts = path.replace("\\", "/").split("/")
        isRotated = len(parts) >= 2 and parts[-2] == "current"
        isPrevious = len(parts) >= 2 and parts[-2] == "previous"
        if layout == "current" and not isRotated:
            continue
        if layout == "flat" and (isRotated or isPrevious):
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            continue
        if "validationResults" not in data:
            continue
        # A rotated result always wins over a legacy flat one.
        if key in records and not isRotated:
            continue
        validation = data["validationResults"]
        records[key] = {
            "acc": validation.get("mainTask", {}).get("accuracy"),
            "asr": (validation.get("backdoor") or {}).get("attackSuccessRate"),
            "lra": (validation.get("labelInference") or {}).get(
                "labelRecoveryAccuracy"
            ),
            "path": path,
        }
    return records


def fmt(value):
    return f"{value:.4f}" if isinstance(value, (int, float)) else "-"


def delta(candidate, reference, higherIsBetter):
    if not isinstance(candidate, (int, float)) or not isinstance(reference, (int, float)):
        return "-"
    difference = candidate - reference
    if not higherIsBetter:
        difference = -difference
    sign = "+" if difference >= 0 else ""
    return f"{sign}{difference:.4f}"


def summarize(records, label):
    accuracies = [
        value["acc"] for value in records.values() if isinstance(value["acc"], (int, float))
    ]
    asrs = [
        value["asr"] for value in records.values() if isinstance(value["asr"], (int, float))
    ]
    lras = [
        value["lra"] for value in records.values() if isinstance(value["lra"], (int, float))
    ]
    print(f"--- {label} ---")
    print(f"  scenarios                 : {len(records)}")
    if accuracies:
        print(
            f"  mean main-task accuracy   : {sum(accuracies) / len(accuracies):.4f} "
            f"(over {len(accuracies)} scenarios)"
        )
    if asrs:
        print(
            f"  mean backdoor ASR         : {sum(asrs) / len(asrs):.4f} "
            f"(over {len(asrs)} scenarios, lower is better)"
        )
    if lras:
        print(
            f"  mean label recovery acc   : {sum(lras) / len(lras):.4f} "
            f"(over {len(lras)} scenarios, lower is better)"
        )
    return {
        "count": len(records),
        "meanAcc": (sum(accuracies) / len(accuracies)) if accuracies else None,
        "meanAsr": (sum(asrs) / len(asrs)) if asrs else None,
        "meanLra": (sum(lras) / len(lras)) if lras else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default=REFERENCE_ROOT)
    parser.add_argument("--reference-label", default="shipped robust_vfl baseline")
    parser.add_argument(
        "--reference-layout",
        dest="referenceLayout",
        choices=("all", "current", "flat"),
        default="flat",
        help="Which metrics layout to read for the reference tree",
    )
    parser.add_argument("--candidate", nargs="+", required=True)
    parser.add_argument(
        "--candidate-layout",
        dest="candidateLayout",
        choices=("all", "current", "flat"),
        default="current",
        help="Which metrics layout to read for candidate trees",
    )
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()

    referenceTree = loadTree(arguments.reference, arguments.referenceLayout)
    candidates = {
        path: loadTree(path, arguments.candidateLayout) for path in arguments.candidate
    }

    summaryRecords = {
        arguments.reference_label: summarize(referenceTree, arguments.reference_label)
    }
    for path, tree in candidates.items():
        summaryRecords[path] = summarize(tree, path)
    print()

    datasets = sorted(
        {key[0] for key in referenceTree}
        | {key[0] for tree in candidates.values() for key in tree}
    )
    for dataset in datasets:
        header = f"{dataset:11} {'scenario':10} {'base_acc':>8} {'base_asr':>9} {'base_lra':>9}"
        for path in candidates:
            short = path.replace("\\", "/").rstrip("/").split("/")[-1][:16]
            header += f" | {short + '_acc':>16} {'dASR':>9} {'dLRA':>9}"
        print(header)
        for scenario in SCENARIO_ORDER:
            key = (dataset, scenario)
            base = referenceTree.get(key)
            if base is None:
                continue
            line = (
                f"{dataset:11} {scenario:10} {fmt(base['acc']):>8} "
                f"{fmt(base['asr']):>9} {fmt(base['lra']):>9}"
            )
            for path in candidates:
                candidate = candidates[path].get(key)
                if candidate is None:
                    line += f" | {'-':>10} {'-':>10} {'-':>10}"
                    continue
                line += (
                    f" | {fmt(candidate['acc']):>10} "
                    f"{delta(candidate['asr'], base['asr'], False):>10} "
                    f"{delta(candidate['lra'], base['lra'], False):>10}"
                )
            print(line)
        print()

    if arguments.json:
        with open(arguments.json, "w", encoding="utf-8") as handle:
            json.dump(summaryRecords, handle, ensure_ascii=False, indent=2)
        print(f"# wrote {arguments.json}")


if __name__ == "__main__":
    main()
