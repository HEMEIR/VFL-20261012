"""Incrementally replace d03/amc and d01/composite predictions in the cache, refresh manifests,
then repackage.  Only these two scenarios changed (code-level special cases, config hash
unchanged), so the rest of the cache stays valid.

    <python> scripts/apply_fix_and_package.py
"""

import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE = PROJECT_ROOT / "outputs" / "cache_reassign_sym"
PROBE = PROJECT_ROOT / "outputs" / "probe_fix"

SCENARIOS = [
    ("dataset_03", "amc", ["main_task_predictions.csv", "label_inference_predictions.csv"]),
    ("dataset_01", "composite", ["main_task_predictions.csv", "label_inference_predictions.csv", "backdoor_predictions.csv"]),
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def row_count(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return len(list(csv.reader(handle))) - 1


def main():
    for dataset, scenario, files in SCENARIOS:
        cacheDir = CACHE / dataset / scenario
        probeDir = PROBE / f"{dataset}_{scenario}"
        for name in files:
            src = probeDir / name
            dst = cacheDir / name
            if not src.is_file():
                print(f"[skip] {src} missing")
                continue
            shutil.copy2(src, dst)
        manifestPath = cacheDir / "run_manifest.json"
        manifest = json.loads(manifestPath.read_text(encoding="utf-8"))
        for name in files:
            manifest["predictionFiles"][name]["sha256"] = sha256(cacheDir / name)
            manifest["predictionFiles"][name]["rowCount"] = row_count(cacheDir / name)
        manifestPath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[done] replaced {dataset}/{scenario} predictions + manifest")

    print("[package] repackaging")
    result = subprocess.run(
        [
            sys.executable, str(PROJECT_ROOT / "scripts" / "build_submission.py"),
            "--package-only",
            "--defense-config", "configs/defense_purified_reassign_sym.yaml",
            "--results-dir", str(CACHE),
            "--team-id", "team001",
            "--model-version", "reassign_sym",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    print(result.stdout[-2000:])
    if result.returncode != 0:
        print(result.stderr[-2000:])


if __name__ == "__main__":
    main()
