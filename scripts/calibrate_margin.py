"""Calibrate the prototype-filter similarity margin on clean vs backdoored data.

For each candidate margin the script runs one clean scenario and one LFBA
scenario (strongest poison rate) and reports clean accuracy, flag rate, and ASR.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core.experiment as experimentModule  # noqa: E402
from core.config import loadConfig, normalizeDefenseConfigDocument  # noqa: E402
from defense import buildPartyADefense  # noqa: E402


def runOne(scenarioName, defenseConfig, epochs, device, outRoot, tag):
    scenarioPath = PROJECT_ROOT / "configs" / "scenarios" / f"{scenarioName}.yaml"
    config = loadConfig(
        scenarioPath,
        {
            "training": {"epochs": epochs, "device": device, "progressEnabled": False},
            "output": {"directory": f"{outRoot}/{tag}/{scenarioName}"},
        },
        enforceFormalLock=False,
        defenseConfig=defenseConfig,
    )
    holder = {}

    def spy(defenseConfigArg):
        instance = buildPartyADefense(defenseConfigArg)
        holder["defense"] = instance
        return instance

    originalBuild = experimentModule.buildPartyADefense
    originalSave = experimentModule.saveCheckpoint
    experimentModule.buildPartyADefense = spy
    experimentModule.saveCheckpoint = lambda path, models, summary: Path(path)
    try:
        summary = experimentModule.runExperiment(config, PROJECT_ROOT)
    finally:
        experimentModule.buildPartyADefense = originalBuild
        experimentModule.saveCheckpoint = originalSave

    defense = holder.get("defense")
    flagged = None
    if defense is not None and defense.epochFlaggedFractionCount:
        flagged = defense.epochFlaggedFractionSum / defense.epochFlaggedFractionCount
    validation = summary["validationResults"]
    return {
        "valAcc": validation["mainTask"]["accuracy"],
        "asr": (validation.get("backdoor") or {}).get("attackSuccessRate"),
        "flagged": flagged,
        "bestEpoch": summary["training"]["bestEpoch"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--margins", nargs="+", type=float, default=[0.0, 0.3, 0.5, 0.7])
    parser.add_argument("--z", type=float, default=3.0)
    parser.add_argument("--clean", default="dataset_03_clean")
    parser.add_argument("--lfba", default="dataset_03_lfba_p20")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-root", dest="outRoot", default="outputs/calib")
    arguments = parser.parse_args()

    print(
        f"{'margin':>7} {'cleanAcc':>9} {'cleanFlag':>10} {'lfbaAcc':>9} "
        f"{'lfbaASR':>9} {'lfbaFlag':>9}"
    )
    print("-" * 60)
    for margin in arguments.margins:
        defenseConfig = normalizeDefenseConfigDocument(
            {
                "defense": {
                    "name": "purified_vfl",
                    "parameters": {
                        "trainingPrototypeFilterStrength": 1.0,
                        "trainingPrototypeFilterZ": arguments.z,
                        "prototypeFilterMargin": margin,
                        "prototypeMomentum": 0.9,
                        "trainingRecenterStrength": 0.0,
                        "inferenceRecenterStrength": 1.0,
                        "recenterRepairWeight": 1.0,
                        "gradientScale": 1.0,
                        "gradientClipNorm": 0.0,
                        "gradientClipRelative": 0.0,
                        "gradientNoise": 0.0,
                        "gradientClassCenteringStrength": 0.0,
                    },
                }
            }
        )
        tag = f"m{margin}"
        clean = runOne(
            arguments.clean, defenseConfig, arguments.epochs, arguments.device,
            arguments.outRoot, tag,
        )
        lfba = runOne(
            arguments.lfba, defenseConfig, arguments.epochs, arguments.device,
            arguments.outRoot, tag,
        )
        fmt = lambda value: f"{value:.4f}" if isinstance(value, float) else "-"
        print(
            f"{margin:>7.2f} {fmt(clean['valAcc']):>9} {fmt(clean['flagged']):>10} "
            f"{fmt(lfba['valAcc']):>9} {fmt(lfba['asr']):>9} {fmt(lfba['flagged']):>9}",
            flush=True,
        )


if __name__ == "__main__":
    main()
