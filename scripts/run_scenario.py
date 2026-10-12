"""Fast single-scenario experiment runner for defense development.

The formal scenario YAMLs under ``configs/scenarios/`` are frozen by
``configs/formal_config_lock.json`` and must never be modified.  This script
loads them with the lock disabled, applies debug-only overrides (epoch count,
device, progress bars) and runs one scenario into a scratch output directory so
defense ideas can be iterated quickly.

Formal submission runs still use ``main.py`` / ``scripts/build_submission.py``
with the untouched scenario YAMLs.

Examples
--------
    python scripts/run_scenario.py --scenario dataset_03_clean --defense configs/defense.yaml
    python scripts/run_scenario.py --scenario dataset_03_villain --epochs 20 --out outputs/debug/x
    python scripts/run_scenario.py --scenario dataset_01_lfba_p20 --no-checkpoint
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import loadConfig, loadDefenseConfig, resolveDefenseConfigPath  # noqa: E402
from core.experiment import runExperiment  # noqa: E402


SUMMARY_FIELDS = ("valAcc", "asr", "triggeredBa", "lra", "lraAuc")


def buildSummaryLines(summary):
    validation = summary.get("validationResults", {})
    mainTask = validation.get("mainTask", {})
    backdoor = validation.get("backdoor", {})
    labelInference = validation.get("labelInference", {})
    lines = [
        f"dataset           : {summary.get('dataset')}",
        f"scenario          : {summary.get('scenario')}",
        f"device            : {summary.get('device')}",
        f"bestEpoch         : {summary.get('training', {}).get('bestEpoch')}",
        f"valAcc            : {mainTask.get('accuracy')}",
    ]
    if backdoor:
        lines.append(f"ASR               : {backdoor.get('attackSuccessRate')}")
        lines.append(f"triggeredBa       : {backdoor.get('triggeredAccuracy')}")
        lines.append(f"attackedSamples   : {backdoor.get('attackedSamples')}")
    if labelInference:
        lines.append(f"labelRecoveryAcc  : {labelInference.get('labelRecoveryAccuracy')}")
        lines.append(f"labelRecoveryAuc  : {labelInference.get('labelRecoveryAuc')}")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, help="e.g. dataset_03_villain")
    parser.add_argument(
        "--defense",
        default="configs/defense.yaml",
        help="defense YAML inside configs/ (default: configs/defense.yaml)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="debug epoch override")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", dest="batchSize", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default=None, help="scratch output directory")
    parser.add_argument(
        "--no-checkpoint",
        dest="writeCheckpoint",
        action="store_false",
        default=True,
        help="skip writing checkpoint.pt (faster, less disk)",
    )
    parser.add_argument("--tag", default=None, help="label printed with the result")
    arguments = parser.parse_args()

    scenarioPath = PROJECT_ROOT / "configs" / "scenarios" / f"{arguments.scenario}.yaml"
    if not scenarioPath.is_file():
        raise SystemExit(f"Scenario config not found: {scenarioPath}")

    defensePath = resolveDefenseConfigPath(PROJECT_ROOT, arguments.defense)
    defenseConfig = loadDefenseConfig(defensePath)

    overrides = {"training": {"progressEnabled": False}}
    if arguments.epochs is not None:
        overrides["training"]["epochs"] = int(arguments.epochs)
    if arguments.device is not None:
        overrides["training"]["device"] = arguments.device
    if arguments.batchSize is not None:
        overrides["training"]["batchSize"] = int(arguments.batchSize)
    if arguments.seed is not None:
        overrides["training"]["seed"] = int(arguments.seed)
    outputDirectory = arguments.out or f"outputs/debug/{arguments.scenario}"
    overrides["output"] = {"directory": outputDirectory}

    config = loadConfig(
        scenarioPath,
        overrides,
        enforceFormalLock=False,
        defenseConfig=defenseConfig,
    )

    if not arguments.writeCheckpoint:
        # ``runExperiment`` always writes a checkpoint; redirect it to a scratch
        # location instead of patching module state destructively.
        import core.experiment as experimentModule

        originalSaveCheckpoint = experimentModule.saveCheckpoint

        def skipCheckpoint(path, models, summary):
            return Path(path)

        experimentModule.saveCheckpoint = skipCheckpoint
    else:
        originalSaveCheckpoint = None

    startTime = time.time()
    try:
        summary = runExperiment(config, PROJECT_ROOT)
    finally:
        if originalSaveCheckpoint is not None:
            import core.experiment as experimentModule

            experimentModule.saveCheckpoint = originalSaveCheckpoint
    elapsed = time.time() - startTime

    print("=" * 68)
    if arguments.tag:
        print(f"tag               : {arguments.tag}")
    for line in buildSummaryLines(summary):
        print(line)
    print(f"elapsedSeconds    : {elapsed:.1f}")
    print(f"outputDirectory   : {Path(outputDirectory).as_posix()}")
    defenseEcho = json.dumps(config.get("defense"), ensure_ascii=False)
    print(f"defense           : {defenseEcho}")
    print("=" * 68)


if __name__ == "__main__":
    main()
