"""Report the recentring anomaly-ratio distribution a trained model actually produces.

The shipped detection threshold is a fixed multiple of the learned fluctuation band
(3.0).  Round 6 measured that this scale is dataset-dependent: on dataset_03 clean
batches stay below 2.8 while on dataset_01 they reach 17.3.  A fixed threshold is
therefore too tight for dataset_01 and the correction over-fires on ordinary
batches, which is a plausible cause of dataset_01's accuracy regressions.

This probe runs the real pipeline and records, from the FINAL training epoch only
(by which time the reference mean has converged), the distribution of
``deviationNorm / bandNorm`` per batch.  The upper quantiles of that distribution
are the scale a dataset-appropriate threshold would need.

    python scripts/probe_ratio_scale.py --scenarios dataset_01_lfba_p20 dataset_03_lfba_p20
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

RATIOS = []


def buildSpy(innerBuild):
    def spy(defenseConfig):
        instance = innerBuild(defenseConfig)
        if instance is None:
            return None
        original = instance._recenter

        def instrumented(representation, strength, phase="train", context=None,
                         _o=original, _i=instance):
            result = _o(representation, strength, phase, context=context)
            if phase == "train":
                RATIOS.append(float(_i.lastAnomalyRatio))
            return result

        instance._recenter = instrumented
        return instance

    return spy


def quantile(values, q):
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", required=True)
    parser.add_argument("--defense", default="configs/defense_purified.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tail-fraction", dest="tailFraction", type=float, default=1.0)
    arguments = parser.parse_args()

    innerBuild = experimentModule.buildPartyADefense
    innerSave = experimentModule.saveCheckpoint
    print(
        f"{'scenario':26} {'epochs':>7} {'median':>8} {'p75':>8} {'p90':>8} "
        f"{'p99':>8} {'max':>9}"
    )
    for scenarioName in arguments.scenarios:
        RATIOS.clear()
        scenarioPath = PROJECT_ROOT / "configs" / "scenarios" / f"{scenarioName}.yaml"
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
                "output": {"directory": f"outputs/probe_ratio/{scenarioName}"},
            },
            enforceFormalLock=False,
            defenseConfig=defenseConfig,
        )
        experimentModule.buildPartyADefense = buildSpy(innerBuild)
        experimentModule.saveCheckpoint = lambda path, models, summary: Path(path)
        error = None
        try:
            experimentModule.runExperiment(config, PROJECT_ROOT)
        except Exception as problem:  # still report whatever was collected
            error = problem
        finally:
            experimentModule.buildPartyADefense = innerBuild
            experimentModule.saveCheckpoint = innerSave

        # Only the tail is meaningful: early epochs still chase the batch mean.
        tail = RATIOS[int(len(RATIOS) * (1.0 - arguments.tailFraction)):]
        print(
            f"{scenarioName:26} {len(tail):>7} {quantile(tail, 0.5):>8.3f} "
            f"{quantile(tail, 0.75):>8.3f} {quantile(tail, 0.90):>8.3f} "
            f"{quantile(tail, 0.99):>8.3f} {max(tail) if tail else float('nan'):>9.3f}"
            + (f"   ERROR {type(error).__name__}" if error else "")
        )


if __name__ == "__main__":
    main()
