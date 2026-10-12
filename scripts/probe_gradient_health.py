"""Verify the gradient reassignment can never hand a degenerate row to the attack.

Runs the real LFBA pipeline with the shipped v5 defense instrumented, and checks
after every ``cutGradientHook`` call that no per-sample L1 norm is zero - the exact
condition ``attacks/backdoor/lfba.py::buildPoisonSet`` rejects.  The first epoch is
the one that matters, because that is when the attacker collects the gradients it
ranks samples by.

    python scripts/probe_gradient_health.py --scenario dataset_04_lfba_p05
"""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core.experiment as experimentModule  # noqa: E402
from core.config import loadConfig, loadDefenseConfig, resolveDefenseConfigPath  # noqa: E402

STATS = {"calls": 0, "zeroRows": 0, "minL1": float("inf")}


def buildSpy(innerBuild):
    def spy(defenseConfig):
        instance = innerBuild(defenseConfig)
        if instance is None:
            return None
        original = instance.cutGradientHook

        def hook(gradientB, *args, **kwargs):
            result = original(gradientB, *args, **kwargs)
            flat = result.detach().reshape(result.shape[0], -1)
            l1 = flat.abs().sum(dim=1)
            STATS["calls"] += 1
            STATS["zeroRows"] += int((l1 == 0).sum().item())
            STATS["minL1"] = min(STATS["minL1"], float(l1.min().item()))
            return result

        instance.cutGradientHook = hook
        return instance

    return spy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="dataset_04_lfba_p05")
    parser.add_argument("--defense", default="configs/defense_purified_v5.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()

    scenarioPath = PROJECT_ROOT / "configs" / "scenarios" / f"{arguments.scenario}.yaml"
    defenseConfig = loadDefenseConfig(
        resolveDefenseConfigPath(PROJECT_ROOT, arguments.defense)
    )
    config = loadConfig(
        scenarioPath,
        {
            "training": {
                "epochs": arguments.epochs,
                "device": arguments.device,
                "progressEnabled": False,
            },
            "output": {"directory": f"outputs/probe_grad/{arguments.scenario}"},
        },
        enforceFormalLock=False,
        defenseConfig=defenseConfig,
    )

    innerBuild = experimentModule.buildPartyADefense
    innerSave = experimentModule.saveCheckpoint
    experimentModule.buildPartyADefense = buildSpy(innerBuild)
    experimentModule.saveCheckpoint = lambda path, models, summary: Path(path)
    try:
        summary = experimentModule.runExperiment(config, PROJECT_ROOT)
        failure = None
    except Exception as error:
        failure = error
    finally:
        experimentModule.buildPartyADefense = innerBuild
        experimentModule.saveCheckpoint = innerSave

    print(f"scenario           : {arguments.scenario}")
    print(f"failed             : {failure}")
    print(f"hook calls         : {STATS['calls']}")
    print(f"zero-L1 rows seen  : {STATS['zeroRows']}")
    print(f"minimum L1 norm    : {STATS['minL1']:.6e}")
    print(f"completed          : acc={summary['validationResults']['mainTask']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
