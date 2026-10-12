"""Measure whether the passive bottom model collapses during a scenario.

The defense's representation statistics are per-dimension variances taken across
the batch.  If the passive model outputs the same vector for every sample, that
variance goes to zero even though the embedding norm is large, and every
variance-scaled transform in the defense becomes a no-op.  This probe reports the
passive embedding's across-batch variance for the shipped defense and for the
identity baseline so the two can be compared on the same scenario.

    python scripts/probe_passive_variance.py --scenarios dataset_02_composite dataset_02_clean
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

MEASUREMENTS = []


def buildSpy(innerBuild, holder):
    """Wrap the factory so the constructed defense can record statistics."""
    def spy(defenseConfig):
        instance = innerBuild(defenseConfig)
        holder["defense"] = instance
        if instance is not None:
            original = instance._updateReference

            def instrumented(representation, _original=original, _instance=instance):
                result = _original(representation)
                if _instance.referenceScale is not None:
                    MEASUREMENTS.append(
                        (
                            float(_instance.referenceScale.mean().item()),
                            float(_instance.referenceScale.max().item()),
                            float(representation.detach().float().norm(dim=1).mean().item()),
                        )
                    )
                return result

            instance._updateReference = instrumented
        return instance

    return spy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", required=True)
    parser.add_argument("--defense", default="configs/defense_purified.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()

    innerBuild = experimentModule.buildPartyADefense
    innerSave = experimentModule.saveCheckpoint
    for scenarioName in arguments.scenarios:
        MEASUREMENTS.clear()
        scenarioPath = (
            PROJECT_ROOT / "configs" / "scenarios" / f"{scenarioName}.yaml"
        )
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
                "output": {"directory": f"outputs/probe_variance/{scenarioName}"},
            },
            enforceFormalLock=False,
            defenseConfig=defenseConfig,
        )
        holder = {}
        experimentModule.buildPartyADefense = buildSpy(innerBuild, holder)
        experimentModule.saveCheckpoint = lambda path, models, summary: Path(path)
        try:
            summary = experimentModule.runExperiment(config, PROJECT_ROOT)
        finally:
            experimentModule.buildPartyADefense = innerBuild
            experimentModule.saveCheckpoint = innerSave

        if not MEASUREMENTS:
            print(f"{scenarioName:28} no measurements")
            continue
        # Look at the last 10% of the training stream, where collapse would show.
        tail = MEASUREMENTS[max(0, len(MEASUREMENTS) * 9 // 10) :]
        early = MEASUREMENTS[: max(1, len(MEASUREMENTS) // 10)]
        meanScaleTail = sum(item[0] for item in tail) / len(tail)
        maxScaleTail = max(item[1] for item in tail)
        normTail = sum(item[2] for item in tail) / len(tail)
        meanScaleEarly = sum(item[0] for item in early) / len(early)
        validation = summary["validationResults"]
        backdoor = validation.get("backdoor") or {}
        privacy = validation.get("labelInference") or {}
        print(
            f"{scenarioName:28} acc={validation['mainTask']['accuracy']:.4f} "
            f"asr={backdoor.get('attackSuccessRate')} "
            f"lra={privacy.get('labelRecoveryAccuracy')} | "
            f"perDimStd early={meanScaleEarly:.3e} tail={meanScaleTail:.3e} "
            f"tailMax={maxScaleTail:.3e} embNorm={normTail:.3f}"
        )


if __name__ == "__main__":
    main()
