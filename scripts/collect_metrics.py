"""Collect scenario metrics from outputs/scenarios into a compact table.

Usage:
    python scripts/collect_metrics.py [--root outputs/scenarios] [--json out.json]
"""

import argparse
import glob
import json
import os


def collect(root):
    rows = []
    pattern = os.path.join(root, "*", "*", "metrics.json")
    for path in sorted(glob.glob(pattern)):
        parts = path.replace("\\", "/").split("/")
        dataset = parts[-3]
        scenario = parts[-2]
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as error:  # pragma: no cover - diagnostics only
            rows.append({"dataset": dataset, "scenario": scenario, "error": str(error)})
            continue
        validation = data.get("validationResults", {})
        mainTask = validation.get("mainTask", {})
        backdoor = validation.get("backdoor", {})
        labelInference = validation.get("labelInference", {})
        rows.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "valAcc": mainTask.get("accuracy"),
                "asr": backdoor.get("attackSuccessRate"),
                "triggeredAcc": backdoor.get("triggeredAccuracy"),
                "lra": labelInference.get("labelRecoveryAccuracy"),
                "lraAuc": labelInference.get("labelRecoveryAuc"),
                "bestEpoch": data.get("training", {}).get("bestEpoch"),
                "attack": (backdoor.get("attack") or labelInference.get("attack")),
                "path": path,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/scenarios")
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()
    rows = collect(arguments.root)
    header = f"{'dataset':12} {'scenario':10} {'valAcc':>8} {'ASR':>8} {'trigAcc':>8} {'LRA':>8} {'lraAuc':>8} {'ep':>4}"
    print(header)
    print("-" * len(header))

    def fmt(value):
        return f"{value:.4f}" if isinstance(value, (int, float)) else "-"

    for row in rows:
        if "error" in row:
            print(f"{row['dataset']:12} {row['scenario']:10} ERROR {row['error']}")
            continue
        print(
            f"{row['dataset']:12} {row['scenario']:10} {fmt(row['valAcc']):>8} "
            f"{fmt(row['asr']):>8} {fmt(row['triggeredAcc']):>8} {fmt(row['lra']):>8} "
            f"{fmt(row['lraAuc']):>8} {str(row['bestEpoch']):>4}"
        )
    print(f"total scenarios found: {len(rows)}")
    if arguments.json:
        with open(arguments.json, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2, ensure_ascii=False)
        print(f"wrote {arguments.json}")


if __name__ == "__main__":
    main()
