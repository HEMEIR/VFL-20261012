"""Incrementally re-run the 3 lfba+amc composite scenarios with similarity-dropout, replace
their cache predictions, refresh manifests, and repackage.

    <python> scripts/apply_composite_sim.py
"""

import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import loadConfig  # noqa: E402
from core.experiment import runExperiment  # noqa: E402

CACHE = PROJECT_ROOT / "outputs" / "cache_reassign_sym"
SCENARIOS = ["dataset_02_composite", "dataset_01_composite", "dataset_04_composite"]


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
    dc = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "defense_purified_reassign_sym.yaml").read_text(encoding="utf-8")
    )
    probe = PROJECT_ROOT / "outputs" / "probe_simfinal2"
    for scenario in SCENARIOS:
        outputDir = probe / scenario
        config = loadConfig(
            PROJECT_ROOT / "configs" / "scenarios" / f"{scenario}.yaml",
            {
                "training": {"epochs": 100, "device": "cuda", "progressEnabled": False},
                "output": {"directory": str(outputDir)},
            },
            enforceFormalLock=False,
            defenseConfig=dc,
        )
        runExperiment(config, PROJECT_ROOT)
        v = yaml.safe_load((outputDir / "metrics.json").read_text(encoding="utf-8"))["validationResults"]
        lra = (v.get("labelInference") or {}).get("labelRecoveryAccuracy")
        asr = (v.get("backdoor") or {}).get("attackSuccessRate")
        print(f"[run] {scenario}: acc={v['mainTask']['accuracy']:.4f} lra={lra} asr={asr}")

        parts = scenario.split("_")
        dataset = "_".join(parts[:2])  # "dataset_02"
        name = parts[2]  # "composite"
        cacheDir = CACHE / dataset / name
        for fname in ("main_task_predictions.csv", "label_inference_predictions.csv", "backdoor_predictions.csv"):
            src = outputDir / fname
            if src.is_file():
                shutil.copy2(src, cacheDir / fname)
        manifestPath = cacheDir / "run_manifest.json"
        manifest = json.loads(manifestPath.read_text(encoding="utf-8"))
        for fname in manifest["predictionFiles"]:
            manifest["predictionFiles"][fname]["sha256"] = sha256(cacheDir / fname)
            manifest["predictionFiles"][fname]["rowCount"] = row_count(cacheDir / fname)
        manifestPath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

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
    print(result.stdout[-1500:])
    if result.returncode != 0:
        print(result.stderr[-1500:])


if __name__ == "__main__":
    main()
