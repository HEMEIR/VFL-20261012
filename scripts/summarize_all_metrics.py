"""Summarize every metrics JSON found in the repository into one table."""

import argparse
import glob
import json
import os


def loadRow(path):
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return None
    if not isinstance(data, dict) or "validationResults" not in data:
        return None
    validation = data["validationResults"]
    mainTask = validation.get("mainTask", {})
    backdoor = validation.get("backdoor", {})
    labelInference = validation.get("labelInference", {})
    return {
        "file": os.path.relpath(path).replace("\\", "/"),
        "dataset": data.get("dataset"),
        "backdoor": data.get("scenario", {}).get("backdoor"),
        "labelInference": data.get("scenario", {}).get("labelInference"),
        "valAcc": mainTask.get("accuracy"),
        "asr": backdoor.get("attackSuccessRate"),
        "triggeredBa": backdoor.get("triggeredAccuracy"),
        "lra": labelInference.get("labelRecoveryAccuracy"),
        "lraAuc": labelInference.get("labelRecoveryAuc"),
        "bestEpoch": data.get("training", {}).get("bestEpoch"),
    }


def fmt(value):
    return f"{value:.4f}" if isinstance(value, (int, float)) else "-"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", default=["outputs"])
    parser.add_argument("--filter", default=None)
    parser.add_argument("--sort", default="file")
    arguments = parser.parse_args()

    paths = []
    for root in arguments.roots:
        paths.extend(glob.glob(os.path.join(root, "**", "*.json"), recursive=True))
    rows = [row for row in (loadRow(path) for path in sorted(paths)) if row]
    if arguments.filter:
        needle = arguments.filter.lower()
        rows = [row for row in rows if needle in row["file"].lower()]
    rows.sort(key=lambda row: row[arguments.sort] if row[arguments.sort] is not None else "")

    header = (
        f"{'file':58} {'dataset':11} {'bd':8} {'li':5} {'valAcc':>8} "
        f"{'ASR':>8} {'trigBa':>8} {'LRA':>8} {'lraAuc':>8} {'ep':>4}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['file'][-58:]:58} {str(row['dataset']):11} "
            f"{str(row['backdoor']):8} {str(row['labelInference']):5} "
            f"{fmt(row['valAcc']):>8} {fmt(row['asr']):>8} {fmt(row['triggeredBa']):>8} "
            f"{fmt(row['lra']):>8} {fmt(row['lraAuc']):>8} {str(row['bestEpoch']):>4}"
        )
    print(f"rows: {len(rows)}")


if __name__ == "__main__":
    main()
