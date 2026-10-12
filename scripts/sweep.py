"""Run several scenarios sequentially and print a compact comparison table.

Used for defense development sweeps: every run reads one formal scenario YAML
(with the protocol lock disabled) plus one defense YAML, and the resulting
metrics are collected into a single table and JSON record.

Example
-------
    python scripts/sweep.py --defense configs/defense_identity.yaml \
        --scenarios dataset_03_clean dataset_03_villain dataset_05_clean \
        --tag identity --epochs 100
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import loadConfig, loadDefenseConfig, resolveDefenseConfigPath  # noqa: E402
from core.experiment import runExperiment  # noqa: E402


def extractRow(scenarioName, summary, elapsed):
    validation = summary.get("validationResults", {})
    mainTask = validation.get("mainTask", {})
    backdoor = validation.get("backdoor", {})
    labelInference = validation.get("labelInference", {})
    return {
        "scenario": scenarioName,
        "dataset": summary.get("dataset"),
        "backdoor": summary.get("scenario", {}).get("backdoor"),
        "labelInference": summary.get("scenario", {}).get("labelInference"),
        "valAcc": mainTask.get("accuracy"),
        "asr": backdoor.get("attackSuccessRate"),
        "triggeredBa": backdoor.get("triggeredAccuracy"),
        "lra": labelInference.get("labelRecoveryAccuracy"),
        "lraAuc": labelInference.get("labelRecoveryAuc"),
        "bestEpoch": summary.get("training", {}).get("bestEpoch"),
        "elapsed": round(elapsed, 1),
    }


def fmt(value):
    return f"{value:.4f}" if isinstance(value, (int, float)) else "-"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", required=True)
    parser.add_argument("--defense", default="configs/defense.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tag", default="run")
    parser.add_argument("--out-root", dest="outRoot", default="outputs/sweeps")
    parser.add_argument("--record", default=None, help="JSON file to append rows to")
    parser.add_argument("--keep-checkpoint", dest="keepCheckpoint", action="store_true")
    arguments = parser.parse_args()

    defensePath = resolveDefenseConfigPath(PROJECT_ROOT, arguments.defense)
    defenseConfig = loadDefenseConfig(defensePath)
    print(f"# tag={arguments.tag} defense={defensePath.name}")
    print(f"# defenseConfig={json.dumps(defenseConfig['defense'], ensure_ascii=False)}")

    rows = []
    header = (
        f"{'scenario':26} {'valAcc':>8} {'ASR':>8} {'trigBa':>8} "
        f"{'LRA':>8} {'lraAuc':>8} {'ep':>4} {'sec':>7}"
    )
    print(header)
    print("-" * len(header))
    for scenarioName in arguments.scenarios:
        scenarioPath = PROJECT_ROOT / "configs" / "scenarios" / f"{scenarioName}.yaml"
        if not scenarioPath.is_file():
            print(f"{scenarioName:26} MISSING CONFIG")
            continue
        overrides = {"training": {"progressEnabled": False, "device": arguments.device}}
        if arguments.epochs is not None:
            overrides["training"]["epochs"] = int(arguments.epochs)
        overrides["output"] = {
            "directory": f"{arguments.outRoot}/{arguments.tag}/{scenarioName}"
        }
        config = loadConfig(
            scenarioPath,
            overrides,
            enforceFormalLock=False,
            defenseConfig=defenseConfig,
        )
        startTime = time.time()
        try:
            summary = runExperiment(config, PROJECT_ROOT)
        except Exception:
            traceback.print_exc()
            print(f"{scenarioName:26} FAILED")
            continue
        elapsed = time.time() - startTime
        row = extractRow(scenarioName, summary, elapsed)
        rows.append(row)
        print(
            f"{scenarioName:26} {fmt(row['valAcc']):>8} {fmt(row['asr']):>8} "
            f"{fmt(row['triggeredBa']):>8} {fmt(row['lra']):>8} "
            f"{fmt(row['lraAuc']):>8} {str(row['bestEpoch']):>4} {elapsed:>7.0f}",
            flush=True,
        )

    if arguments.record:
        recordPath = PROJECT_ROOT / arguments.record
        recordPath.parent.mkdir(parents=True, exist_ok=True)
        if recordPath.is_file():
            with recordPath.open(encoding="utf-8") as handle:
                existing = json.load(handle)
        else:
            existing = []
        existing.append(
            {
                "tag": arguments.tag,
                "defense": str(defensePath.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "defenseConfig": defenseConfig["defense"],
                "rows": rows,
            }
        )
        with recordPath.open("w", encoding="utf-8") as handle:
            json.dump(existing, handle, ensure_ascii=False, indent=2)
        print(f"# appended record -> {arguments.record}")


if __name__ == "__main__":
    main()
