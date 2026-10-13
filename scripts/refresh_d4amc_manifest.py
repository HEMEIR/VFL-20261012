"""Refresh d04/amc run_manifest prediction hashes after swapping in the new predictions.

    <python> scripts/refresh_d4amc_manifest.py
"""

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def main():
    cache = PROJECT_ROOT / "outputs" / "cache_reassign_sym" / "dataset_04" / "amc"
    manifestPath = cache / "run_manifest.json"
    manifest = json.loads(manifestPath.read_text(encoding="utf-8"))
    for name in ("main_task_predictions.csv", "label_inference_predictions.csv"):
        path = cache / name
        manifest["predictionFiles"][name]["sha256"] = sha256(path)
        import csv
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        manifest["predictionFiles"][name]["rowCount"] = len(rows) - 1
    manifestPath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[done] refreshed d04/amc run_manifest prediction hashes")


if __name__ == "__main__":
    main()
