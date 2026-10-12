"""Compare the batch-mean deviation on clean vs triggered inference on one model.

Runs the real training pipeline once, then performs several inference passes over
the same frozen model: a clean pass, and triggered passes that reproduce the two
attacks' inference-time perturbation.  For each pass it reports how large the
batch-mean deviation is in units of the learned fluctuation band, and what the
accuracy would be with different recentering rules applied.

This is the measurement that decides whether a trigger is separable from ordinary
drift, which in turn decides whether a soft-thresholded recenter can cancel a
trigger without touching clean batches.

    python scripts/probe_recenter_tradeoff.py --scenario dataset_03_villain --epochs 30
"""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attacks.backdoor.triggers import (  # noqa: E402
    addTriggerToPartyB,
    buildTriggeredDataset,
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


def evaluateWithRule(trainer, dataset, rule, referenceMean, referenceMeanStd, batchSize, device):
    """Accuracy on ``dataset`` after applying ``rule`` to the batch mean deviation."""
    models = trainer.models
    models.bottomA.eval()
    models.bottomB.eval()
    models.topModel.eval()
    correct = 0
    total = 0
    ratios = []
    with torch.no_grad():
        for offset in range(0, len(dataset), batchSize):
            positions = list(range(offset, min(offset + batchSize, len(dataset))))
            inputA = dataset.partyA[positions].to(device)
            inputB = dataset.partyB[positions].to(device)
            embeddingB = models.bottomB(inputB)
            batchMean = embeddingB.mean(dim=0, keepdim=True)
            deviation = batchMean - referenceMean.reshape(1, -1)
            band = float(referenceMeanStd.norm().item())
            ratio = float(deviation.norm().item()) / band if band > 0 else 0.0
            ratios.append(ratio)
            corrected = rule(embeddingB, deviation, referenceMeanStd, ratio)
            logits = models.topModel(models.bottomA(inputA), corrected)
            predictions = logits.argmax(dim=1).cpu()
            labels = dataset.labels[positions]
            correct += int((predictions == labels).sum().item())
            total += len(positions)
    return correct / max(total, 1), ratios


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="dataset_03_villain")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--batch-size", dest="batchSize", type=int, default=64)
    arguments = parser.parse_args()

    scenarioPath = PROJECT_ROOT / "configs" / "scenarios" / f"{arguments.scenario}.yaml"
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
    dataRoot = (PROJECT_ROOT / "data" / "datasets").resolve()
    bundle = loadDataset(serialized["dataset"], dataRoot)
    bottomA, bottomB, topModel = buildModels(bundle, serialized["model"])
    models = ModelBundle(bottomA, bottomB, topModel)
    device = resolveDevice(arguments.device)
    trainer = VflTrainer(models, serialized["training"], device)
    trainer.configureScenarioContext("villain", "none")

    runVillainTraining(trainer, bundle, serialized["backdoor"], seed)

    # Recover the passive-embedding population mean and band from the training data.
    models.bottomB.eval()
    means = []
    with torch.no_grad():
        for offset in range(0, len(bundle.train), arguments.batchSize):
            positions = list(range(offset, min(offset + arguments.batchSize, len(bundle.train))))
            embeddings = models.bottomB(bundle.train.partyB[positions].to(device))
            means.append(embeddings.mean(dim=0).float())
    meanStack = torch.stack(means, dim=0)
    referenceMean = meanStack.mean(dim=0)
    referenceMeanStd = (meanStack - referenceMean).abs().mean(dim=0).clamp_min(1e-12)
    print(
        f"reference mean norm={float(referenceMean.norm()):.4f} "
        f"band norm={float(referenceMeanStd.norm()):.4f} "
        f"batches={meanStack.shape[0]}"
    )

    # Build the three inference views of the same labelled validation split.
    knownTargetIds = resolveKnownTargetIds(bundle.train, serialized["backdoor"])
    columnStd = collectKnownTargetEmbeddingStd(trainer, bundle.train, knownTargetIds)
    triggerVector, dimensions, delta = buildVillainTrigger(
        columnStd,
        int(serialized["backdoor"]["triggerDimensionCount"]),
        float(serialized["backdoor"]["beta"]),
    )
    triggerVector = triggerVector.to(device)

    lfbaTriggered = buildTriggeredDataset(
        bundle.validation,
        str(serialized["backdoor"]["lfbaTriggerType"]),
        serialized["backdoor"].get("triggerDimensionList") or [0, 1, 2, 4, 5],
        float(serialized["backdoor"].get("triggerFeatureValue", 1.0)),
    )

    identityRule = lambda embedding, deviation, band, ratio: embedding
    fullRule = lambda embedding, deviation, band, ratio: embedding - deviation
    softRule = (
        lambda embedding, deviation, band, ratio: embedding
        - torch.sign(deviation)
        * (deviation.abs() - arguments.threshold * band).clamp_min(0.0)
    )
    gateRule = (
        lambda embedding, deviation, band, ratio: embedding
        - deviation
        if ratio > arguments.threshold
        else embedding
    )

    views = {
        "clean validation": (bundle.validation, None),
        "LFBA-triggered validation": (lfbaTriggered, None),
    }
    # For the VILLAIN embedding trigger we add the fixed vector on the fly.
    for viewName, (dataset, transform) in views.items():
        line = f"{viewName:28}"
        for ruleName, rule in (
            ("none", identityRule),
            ("full", fullRule),
            ("soft", softRule),
            ("gate", gateRule),
        ):
            accuracy, ratios = evaluateWithRule(
                trainer,
                dataset,
                rule,
                referenceMean,
                referenceMeanStd,
                arguments.batchSize,
                device,
            )
            line += f" | {ruleName}={accuracy:.4f}"
        middle = sorted(ratios)[len(ratios) // 2]
        print(f"{line} | ratio median={middle:.3f} max={max(ratios):.3f}")


if __name__ == "__main__":
    main()
