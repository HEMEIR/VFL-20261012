"""Measure the geometry of the backdoor perturbation in the passive embedding.

The additive VILLAIN trigger lives in B's *embedding* space, but the LFBA
trigger lives in B's *input* space and is reshaped by B's non-linear bottom
model.  Both are only observable to A through the passive embedding, so this
probe asks the question that decides whether a single-direction removal can work:
after feeding genuinely triggered inputs through the passive bottom model, is the
resulting perturbation coherent across samples (one shared direction) or diffuse?
"""

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attacks.backdoor.triggers import (  # noqa: E402
    buildTriggeredDataset,
    selectTriggerDimensions,
)
from attacks.backdoor.villain import (  # noqa: E402
    buildVillainTrigger,
    collectKnownTargetEmbeddingStd,
    resolveKnownTargetIds,
    runVillainTraining,
)
from core.config import loadConfig, loadDefenseConfig  # noqa: E402
from core.experiment import resolveDevice  # noqa: E402
from core.interfaces import ModelBundle  # noqa: E402
from core.randomness import setRandomSeed  # noqa: E402
from core.trainer import VflTrainer  # noqa: E402
from datasets import loadDataset  # noqa: E402
from models import buildModels  # noqa: E402


def report(name, clean, triggered):
    difference = (triggered - clean).detach().float()
    centered = difference - difference.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = float((singular[0] ** 2) / (singular**2).sum().clamp_min(1e-12))
    unit = centered[0] / centered[0].norm().clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        centered, unit.reshape(1, -1).expand_as(centered), dim=1
    )
    print(f"--- {name} ---")
    print(
        f"  ||diff|| mean={float(difference.norm(dim=1).mean()):.6f} "
        f"std={float(difference.norm(dim=1).std()):.6f}"
    )
    print(
        f"  top-1 residual energy fraction = {energy:.4f} "
        f"(1.0 = perfectly coherent single direction)"
    )
    print(
        f"  |cos| with top direction: mean={float(cosine.abs().mean()):.4f} "
        f"min={float(cosine.abs().min()):.4f}"
    )
    print(f"  top-5 singular values = "
          + ", ".join(f"{float(v):.4f}" for v in singular[:5]))
    return energy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset_03")
    parser.add_argument(
        "--lfba-scenario",
        dest="lfbaScenario",
        default=None,
        help="scenario suffix for LFBA, e.g. dataset_03_lfba_p20",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()

    setRandomSeed(100)
    dataRoot = (PROJECT_ROOT / "data" / "datasets").resolve()
    results = {}
    for attackName in ("villain", "lfba"):
        if attackName == "lfba":
            scenarioName = arguments.lfbaScenario or f"{arguments.dataset}_lfba_p20"
        else:
            scenarioName = f"{arguments.dataset}_villain"
        scenarioPath = (
            PROJECT_ROOT / "configs" / "scenarios" / f"{scenarioName}.yaml"
        )
        serialized = loadConfig(
            scenarioPath,
            {"training": {"epochs": arguments.epochs, "progressEnabled": False}},
            enforceFormalLock=False,
            defenseConfig=loadDefenseConfig(
                PROJECT_ROOT / "configs" / "defense_identity.yaml"
            ),
        )
        seed = int(serialized["training"]["seed"])
        setRandomSeed(seed)
        bundle = loadDataset(serialized["dataset"], dataRoot)
        bottomA, bottomB, topModel = buildModels(bundle, serialized["model"])
        models = ModelBundle(bottomA, bottomB, topModel)
        device = resolveDevice(arguments.device)
        trainer = VflTrainer(models, serialized["training"], device)
        trainer.configureScenarioContext(attackName, "none")

        if attackName == "villain":
            runVillainTraining(trainer, bundle, serialized["backdoor"], seed)
            knownTargetIds = resolveKnownTargetIds(bundle.train, serialized["backdoor"])
            columnStd = collectKnownTargetEmbeddingStd(
                trainer, bundle.train, knownTargetIds
            )
            triggerVector, dimensions, delta = buildVillainTrigger(
                columnStd,
                int(serialized["backdoor"]["triggerDimensionCount"]),
                float(serialized["backdoor"]["beta"]),
            )
            triggerVector = triggerVector.to(device)
            models.bottomB.eval()
            with torch.no_grad():
                validationB = bundle.validation.partyB.to(device)
                clean = models.bottomB(validationB)
                triggered = models.bottomB(
                    (bundle.validation.partyB.to(device) + 0.0)
                ) + triggerVector.reshape(1, -1)
            print(
                f"[{arguments.dataset}/villain] trigger dims={dimensions} "
                f"delta={float(delta):.6f} ||trigger||={float(triggerVector.norm()):.6f}"
            )
            results["villain"] = report(
                f"{arguments.dataset} VILLAIN (embedding-space trigger)", clean, triggered
            )
        else:
            from attacks.backdoor.lfba import runLfbaTraining

            lfbaResult = runLfbaTraining(
                trainer, bundle, serialized["backdoor"], seed
            )
            triggerDimensions = lfbaResult.triggerDimensions
            triggerFeatureValue = float(
                serialized["backdoor"].get("triggerFeatureValue", 1.0)
            )
            triggeredValidation = buildTriggeredDataset(
                bundle.validation,
                str(serialized["backdoor"]["lfbaTriggerType"]),
                triggerDimensions,
                triggerFeatureValue,
            )
            models.bottomB.eval()
            with torch.no_grad():
                clean = models.bottomB(bundle.validation.partyB.to(device))
                triggered = models.bottomB(triggeredValidation.partyB.to(device))
            print(
                f"[{arguments.dataset}/lfba] trigger dims={triggerDimensions} "
                f"value={triggerFeatureValue}"
            )
            results["lfba"] = report(
                f"{arguments.dataset} LFBA (input-space trigger)", clean, triggered
            )

    print("=" * 68)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
