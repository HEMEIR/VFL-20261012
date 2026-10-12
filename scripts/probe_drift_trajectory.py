"""Find where variant A's training stream first goes non-finite.

Variant A (per-sample excess clipping) is finite per call - verified directly - yet the
run reaches NaN some way into training, and the non-finite values then ARRIVE at
``cutGradientHook``.  That means the transform is destabilising the optimisation rather
than producing NaN itself, so the useful measurement is the trajectory: per-epoch
magnitudes of the representation and of the gradient B receives, up to the first
non-finite value.

    python scripts/probe_drift_trajectory.py --scenario dataset_04_clean
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

STATE = {"epoch": 0, "batch": 0, "worstRep": 0.0, "worstGrad": 0.0, "firstBad": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="dataset_04_clean")
    parser.add_argument("--defense", default="configs/defense_purified_per_sample.yaml")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()

    config = loadConfig(
        PROJECT_ROOT / "configs" / "scenarios" / f"{arguments.scenario}.yaml",
        {
            "training": {
                "epochs": arguments.epochs,
                "device": arguments.device,
                "progressEnabled": False,
            },
            "output": {"directory": f"outputs/probe_drift/{arguments.scenario}"},
        },
        enforceFormalLock=False,
        defenseConfig=loadDefenseConfig(
            resolveDefenseConfigPath(PROJECT_ROOT, arguments.defense)
        ),
    )

    innerBuild = experimentModule.buildPartyADefense
    innerSave = experimentModule.saveCheckpoint

    def spy(defenseCfg):
        instance = innerBuild(defenseCfg)
        if instance is None:
            return None
        originalRep = instance.trainingRepresentationHook
        originalGrad = instance.cutGradientHook

        def wrappedRep(cutA, cutB, labels=None, sampleIds=None, epoch=0,
                       batchIndex=0, context=None, generator=None,
                       _o=originalRep, _i=instance):
            outA, outB = _o(cutA, cutB, labels=labels, sampleIds=sampleIds,
                            epoch=epoch, batchIndex=batchIndex, context=context,
                            generator=generator)
            magnitude = float(outB.detach().float().abs().max().item())
            STATE["worstRep"] = max(STATE["worstRep"], magnitude)
            if STATE["firstBad"] is None and not torch.isfinite(outB).all():
                STATE["firstBad"] = ("representation", epoch, batchIndex)
            if batchIndex == 0:
                print(f"  epoch {epoch:3d} rep|max|={STATE['worstRep']:.4g} "
                      f"excessFrac={getattr(_i, 'lastExcessFraction', float('nan')):.3f}",
                      flush=True)
            return outA, outB

        def wrappedGrad(gradientB, *args, _o=originalGrad, **kwargs):
            if STATE["firstBad"] is None and not torch.isfinite(gradientB).all():
                STATE["firstBad"] = ("gradient", None, None)
            out = _o(gradientB, *args, **kwargs)
            return out

        instance.trainingRepresentationHook = wrappedRep
        instance.cutGradientHook = wrappedGrad
        return instance

    experimentModule.buildPartyADefense = spy
    experimentModule.saveCheckpoint = lambda path, models, summary: Path(path)
    error = None
    try:
        experimentModule.runExperiment(config, PROJECT_ROOT)
    except Exception as problem:
        error = problem
    finally:
        experimentModule.buildPartyADefense = innerBuild
        experimentModule.saveCheckpoint = innerSave

    print("scenario      :", arguments.scenario)
    print("first non-finite at:", STATE["firstBad"])
    print("worst rep |max|:", f"{STATE['worstRep']:.6g}")
    print("error         :", repr(error))


if __name__ == "__main__":
    main()
