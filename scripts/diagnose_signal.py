"""Diagnostics for the VFL challenge: how label-informative is each side, and
how much of each backdoor trigger survives a batch-common-mode removal.

These are cheap sklearn/data-level probes used to choose a defense direction
before spending GPU hours.

    python scripts/diagnose_signal.py
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import loadDataset  # noqa: E402


DATASETS = {
    "dataset_01": {"inputFormat": "imageHalf", "preprocessing": "none"},
    "dataset_02": {"inputFormat": "tabular", "preprocessing": "none"},
    "dataset_03": {"inputFormat": "tabular", "preprocessing": "none"},
    "dataset_04": {"inputFormat": "tabular", "preprocessing": "none"},
    "dataset_05": {
        "inputFormat": "tabular",
        "preprocessing": "none",
        "classCount": 6,
    },
}


def flatten(features):
    return features.reshape(features.shape[0], -1).numpy()


def linearProbe(trainFeatures, trainLabels, evalFeatures, evalLabels, classCount, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    trainScaled = scaler.fit_transform(trainFeatures)
    evalScaled = scaler.transform(evalFeatures)
    model = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
    model.fit(trainScaled, trainLabels)
    return float((model.predict(evalScaled) == evalLabels).mean())


def smallLabelProbe(
    allFeatures, allLabels, labeledCount, evalFeatures, evalLabels, classCount, seed=0
):
    """Probe fitted on only ``labeledCount`` labels, mimicking the PMC budget."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(seed)
    positions = []
    perClass = max(1, labeledCount // classCount)
    for label in range(classCount):
        candidates = np.flatnonzero(allLabels == label)
        chosen = rng.choice(candidates, size=min(perClass, len(candidates)), replace=False)
        positions.extend(chosen.tolist())
    positions = np.asarray(positions)
    scaler = StandardScaler()
    trainScaled = scaler.fit_transform(allFeatures[positions])
    evalScaled = scaler.transform(evalFeatures)
    model = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
    model.fit(trainScaled, allLabels[positions])
    return float((model.predict(evalScaled) == evalLabels).mean())


def main():
    dataRoot = PROJECT_ROOT / "data" / "datasets"
    for datasetName, options in DATASETS.items():
        config = {
            "name": datasetName,
            "inputFormat": options["inputFormat"],
            "preprocessing": options["preprocessing"],
        }
        if "classCount" in options:
            config["classCount"] = options["classCount"]
        bundle = loadDataset(config, dataRoot)
        classCount = bundle.classCount
        trainA = flatten(bundle.train.partyA)
        trainB = flatten(bundle.train.partyB)
        valA = flatten(bundle.validation.partyA)
        valB = flatten(bundle.validation.partyB)
        trainLabels = bundle.train.labels.numpy()
        valLabels = bundle.validation.labels.numpy()

        accA = linearProbe(trainA, trainLabels, valA, valLabels, classCount)
        accB = linearProbe(trainB, trainLabels, valB, valLabels, classCount)
        accAB = linearProbe(
            np.concatenate([trainA, trainB], axis=1),
            trainLabels,
            np.concatenate([valA, valB], axis=1),
            valLabels,
            classCount,
        )
        # A small labelled budget on B only: the attack's actual regime.
        fewB = smallLabelProbe(trainB, trainLabels, 128, valB, valLabels, classCount)
        print(
            f"{datasetName}: classCount={classCount} dimA={trainA.shape[1]} dimB={trainB.shape[1]} | "
            f"linearProbe A={accA:.4f} B={accB:.4f} A+B={accAB:.4f} | B-with-128-labels={fewB:.4f} "
            f"| chance={1.0/classCount:.4f}"
        )
        print(
            f"   partyB feature std: mean={trainB.std(axis=0).mean():.4f} "
            f"max={trainB.std(axis=0).max():.4f} nonzero-dims={int((trainB.std(axis=0) > 1e-9).sum())}"
        )


if __name__ == "__main__":
    main()
