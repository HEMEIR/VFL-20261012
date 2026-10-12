"""Report per-epoch defense diagnostics for one scenario.

The defense instance is injected into the experiment so its accumulated
diagnostics (flagged fraction, ...) can be read after the run.

    python scripts/diagnose_defense.py --scenario dataset_03_clean --epochs 8
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core.experiment as experimentModule  # noqa: E402
from core.config import loadConfig, loadDefenseConfig, resolveDefenseConfigPath  # noqa: E402
from defense import buildPartyADefense  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--defense", default="configs/defense_purified.yaml")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="outputs/debug/diag")
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args()

    scenarioPath = PROJECT_ROOT / "configs" / "scenarios" / f"{arguments.scenario}.yaml"
    defensePath = resolveDefenseConfigPath(PROJECT_ROOT, arguments.defense)
    defenseConfig = loadDefenseConfig(defensePath)
    config = loadConfig(
        scenarioPath,
        {
            "training": {
                "epochs": arguments.epochs,
                "device": arguments.device,
                "progressEnabled": False,
            },
            "output": {"directory": arguments.out},
        },
        enforceFormalLock=False,
        defenseConfig=defenseConfig,
    )

    holder = {}

    def spy(defenseConfigArg):
        instance = buildPartyADefense(defenseConfigArg)
        holder["defense"] = instance
        return instance

    original = experimentModule.buildPartyADefense
    experimentModule.buildPartyADefense = spy
    if arguments.quiet:
        originalSaveCheckpoint = experimentModule.saveCheckpoint
        experimentModule.saveCheckpoint = lambda path, models, summary: Path(path)
    try:
        summary = experimentModule.runExperiment(config, PROJECT_ROOT)
    finally:
        experimentModule.buildPartyADefense = original
        if arguments.quiet:
            experimentModule.saveCheckpoint = originalSaveCheckpoint

    defense = holder.get("defense")
    print("=" * 60)
    if defense is not None:
        count = defense.epochFlaggedFractionCount
        print(f"flaggedFractionCount      : {count}")
        if count:
            print(
                "mean flagged fraction     : "
                f"{defense.epochFlaggedFractionSum / count:.4f}"
            )
    print(f"valAcc                    : {summary['validationResults']['mainTask']['accuracy']}")
    backdoor = summary["validationResults"].get("backdoor")
    if backdoor:
        print(f"ASR                       : {backdoor['attackSuccessRate']}")
    print(f"bestEpoch                 : {summary['training']['bestEpoch']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
