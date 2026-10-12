"""Probe the actual shape of cutA / cutB inside the hooks, per dataset.

The per-dataset configuration needs a discriminator available at hook time. Feature WIDTH is
the obvious candidate, but only if the cut tensors inherit the dataset's feature count
rather than a fixed model-wide embedding dimension. This measures which it is instead of
assuming either way.

    <python> scripts/probe_cut_shapes.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from core.config import loadConfig  # noqa: E402
import core.experiment as experiment  # noqa: E402
from defense import buildPartyADefense  # noqa: E402

SCENARIOS = ["dataset_01_pmc", "dataset_02_pmc", "dataset_03_pmc",
             "dataset_04_pmc", "dataset_05_pmc"]

observed = {}


def probe(defense):
    """Wrap the three hooks so the first call reports its tensor shapes."""
    original = defense.trainingRepresentationHook
    originalCut = defense.cutGradientHook

    def wrappedTraining(cutA, cutB, *args, **kwargs):
        key = "training"
        if key not in observed:
            observed[key] = (tuple(cutA.shape), tuple(cutB.shape))
        return original(cutA, cutB, *args, **kwargs)

    def wrappedCut(gradientB, passiveCut, *args, **kwargs):
        if "cut" not in observed:
            observed["cut"] = (tuple(gradientB.shape), tuple(passiveCut.shape))
        return originalCut(gradientB, passiveCut, *args, **kwargs)

    defense.trainingRepresentationHook = wrappedTraining
    defense.cutGradientHook = wrappedCut
    return defense


def main():
    building = experiment.buildPartyADefense

    def spy(config):
        result = building(config)
        return probe(result) if result is not None else result

    experiment.buildPartyADefense = spy
    experiment.saveCheckpoint = lambda path, models, state: Path(path)

    for scenario in SCENARIOS:
        observed.clear()
        config = loadConfig(
            PROJECT_ROOT / "configs" / "scenarios" / f"{scenario}.yaml",
            {"training": {"epochs": 1, "device": "cuda", "progressEnabled": False},
             "output": {"directory": f"outputs/probe_cut/{scenario}"}},
            enforceFormalLock=False,
            defenseConfig={"defense": {"name": "purified_vfl", "parameters": {}}},
        )
        try:
            experiment.runExperiment(config, PROJECT_ROOT)
        except Exception as error:  # noqa: BLE001
            print("  %-16s 运行异常: %s: %s"
                  % (scenario, type(error).__name__, str(error)[:70]))
        training = observed.get("training")
        cut = observed.get("cut")
        print("  %-16s trainingHook cutA=%s cutB=%s   gradientHook gradB=%s passiveCut=%s"
              % (scenario,
                 training[0] if training else "-", training[1] if training else "-",
                 cut[0] if cut else "-", cut[1] if cut else "-"))


if __name__ == "__main__":
    main()
