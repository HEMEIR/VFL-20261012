"""Trace per-sample cut-gradient statistics through the defense hooks."""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core.experiment as experimentModule  # noqa: E402
from core.config import loadConfig, normalizeDefenseConfigDocument  # noqa: E402
from defense import buildPartyADefense  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="dataset_03_lfba_p20")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--z", type=float, default=3.0)
    parser.add_argument("--clip-relative", dest="clipRelative", type=float, default=0.0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--centering", type=float, default=0.0)
    parser.add_argument("--out", default="outputs/debug/trace")
    arguments = parser.parse_args()

    stats = {"raw": [], "post": []}

    class TracingDefense:
        def __init__(self, inner):
            self.inner = inner

        def trainingRepresentationHook(self, *args, **kwargs):
            return self.inner.trainingRepresentationHook(*args, **kwargs)

        def inferenceRepresentationHook(self, *args, **kwargs):
            return self.inner.inferenceRepresentationHook(*args, **kwargs)

        def cutGradientHook(self, gradientB, **kwargs):
            flat = gradientB.detach().float().reshape(gradientB.shape[0], -1)
            rawNorm = flat.norm(dim=1)
            transformed = self.inner.cutGradientHook(gradientB, **kwargs)
            postNorm = transformed.detach().float().reshape(
                transformed.shape[0], -1
            ).norm(dim=1)
            epoch = kwargs.get("epoch")
            stats["raw"].append(
                (epoch, float(rawNorm.min()), float(rawNorm.median()), int((rawNorm == 0).sum()))
            )
            stats["post"].append(
                (epoch, float(postNorm.min()), float(postNorm.median()), int((postNorm == 0).sum()))
            )
            return transformed

    defenseConfig = normalizeDefenseConfigDocument(
        {
            "defense": {
                "name": "purified_vfl",
                "parameters": {
                    "trainingPrototypeFilterStrength": 1.0,
                    "trainingPrototypeFilterZ": arguments.z,
                    "prototypeFilterMargin": arguments.margin,
                    "prototypeMomentum": 0.9,
                    "trainingRecenterStrength": 0.0,
                    "inferenceRecenterStrength": 1.0,
                    "recenterRepairWeight": 1.0,
                    "gradientClipNorm": 0.0,
                    "gradientClipRelative": arguments.clipRelative,
                    "gradientNoise": arguments.noise,
                    "gradientClassCenteringStrength": arguments.centering,
                },
            }
        }
    )
    scenarioPath = PROJECT_ROOT / "configs" / "scenarios" / f"{arguments.scenario}.yaml"
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

    def spy(defenseConfigArg):
        inner = buildPartyADefense(defenseConfigArg)
        if inner is None:
            return None
        return TracingDefense(inner)

    originalBuild = experimentModule.buildPartyADefense
    originalSave = experimentModule.saveCheckpoint
    experimentModule.buildPartyADefense = spy
    experimentModule.saveCheckpoint = lambda path, models, summary: Path(path)
    try:
        experimentModule.runExperiment(config, PROJECT_ROOT)
    finally:
        experimentModule.buildPartyADefense = originalBuild
        experimentModule.saveCheckpoint = originalSave

    print("=" * 72)
    print(f"{'stage':6} {'epoch':>5} {'minNorm':>12} {'medianNorm':>12} {'zeroCount':>10}")
    for stage in ("raw", "post"):
        for epoch, minimum, median, zeros in stats[stage]:
            print(f"{stage:6} {str(epoch):>5} {minimum:>12.3e} {median:>12.3e} {zeros:>10}")
    print("=" * 72)


if __name__ == "__main__":
    main()
