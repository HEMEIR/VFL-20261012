"""Score any submission ZIP against the public validation labels.

The scored ASR uses the hidden test split, but the *main-task* and
*label-inference* metrics can be recomputed locally for any package by joining its
predictions with the public validation labels.  This makes it possible to compare
two packages exactly, even after the run metrics for one of them have been lost.

Label-inference accuracy is reported as the best accuracy over label permutations
(the attacker's head has no reason to use the true label order), which is the same
convention the training code uses.

    python scripts/score_package.py outputs/team001__precision_v2.zip outputs/team001__v6.zip
"""

import argparse
import csv
import io
import itertools
import json
import os
import sys
import zipfile
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "datasets")

ORDER = ["clean", "lfba_p05", "lfba_p10", "lfba_p20", "villain", "pmc", "amc", "composite"]
BACKDOOR = {"lfba_p05", "lfba_p10", "lfba_p20", "villain", "composite"}
LABEL_INFERENCE = {"pmc", "amc", "composite"}


def readCsvFromBytes(raw):
    text = raw.decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def findLabelColumn(fieldNames):
    for candidate in ("Label", "label", "target", "y", "class"):
        if candidate in fieldNames:
            return candidate
    for name in fieldNames:
        lowered = name.lower()
        if "label" in lowered or lowered in {"y", "target", "class"}:
            return name
    return None


def findPredictionColumn(fieldNames):
    for candidate in ("Prediction", "prediction", "predicted", "label", "y"):
        if candidate in fieldNames:
            return candidate
    return [c for c in fieldNames if c.upper() not in {"ID", "SAMPLE_ID"}][-1]


def idColumn(fieldNames):
    for candidate in ("ID", "id", "sample_id", "SampleId"):
        if candidate in fieldNames:
            return candidate
    return fieldNames[0]


def loadLabels(dataset, scenario, split):
    """Public validation labels, keyed by sample id, as strings."""
    path = os.path.join(DATA_ROOT, dataset, split, "party_A_validation.csv")
    if not os.path.isfile(path):
        return None
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        labelColumn = findLabelColumn(reader.fieldnames or [])
        keyColumn = idColumn(reader.fieldnames or [])
        if labelColumn is None:
            return None
        return {
            str(row[keyColumn]): str(row[labelColumn])
            for row in reader
            if row.get(keyColumn) is not None
        }


def permutationAccuracy(pairs):
    """Best accuracy over label permutations, as the attack head is unconstrained."""
    if not pairs:
        return None
    truths = {t for _, t in pairs}
    preds = {p for p, _ in pairs}
    if len(preds) > 8 or len(truths) > 8:
        return sum(1 for p, t in pairs if p == t) / len(pairs)
    best = 0.0
    predList = sorted(preds)
    for perm in itertools.permutations(sorted(truths), len(predList)):
        mapping = dict(zip(predList, perm))
        score = sum(1 for p, t in pairs if mapping[p] == t) / len(pairs)
        if score > best:
            best = score
    return best


def scorePackage(path):
    results = {}
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))
        for record in manifest["scenarios"]:
            dataset, scenario = record["dataset"], record["scenario"]
            labels = loadLabels(dataset, scenario, "validation")
            entry = {}
            for fileName, metric in (
                ("main_task_predictions.csv", "acc"),
                ("backdoor_predictions.csv", "asr"),
                ("label_inference_predictions.csv", "lra"),
            ):
                full = f"{dataset}/{scenario}/{fileName}"
                if full not in names or labels is None:
                    continue
                rows = readCsvFromBytes(archive.read(full))
                if not rows:
                    continue
                fieldNames = list(rows[0].keys())
                predictionColumn = findPredictionColumn(fieldNames)
                keyColumn = idColumn(fieldNames)
                pairs = []
                for row in rows:
                    truth = labels.get(str(row[keyColumn]).strip())
                    prediction = str(row[predictionColumn]).strip()
                    if truth is not None and prediction not in ("", "None"):
                        pairs.append((prediction, truth))
                if not pairs:
                    continue
                if metric == "lra":
                    entry[metric] = permutationAccuracy(pairs)
                else:
                    entry[metric] = sum(1 for p, t in pairs if p == t) / len(pairs)
            results[(dataset, scenario)] = entry
    return results


def summarize(name, results):
    def mean(key):
        values = [v[key] for v in results.values() if isinstance(v.get(key), float)]
        return (sum(values) / len(values), len(values)) if values else (None, 0)
    acc, na = mean("acc")
    asr, ns = mean("asr")
    lra, nl = mean("lra")
    print(
        f"{name:34} acc={acc:.4f} (n={na})  asr={asr:.4f} (n={ns})  "
        f"lra={lra:.4f} (n={nl})"
        if acc is not None
        else f"{name:34} no scorable predictions"
    )
    return {"acc": acc, "asr": asr, "lra": lra}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packages", nargs="+")
    arguments = parser.parse_args()
    summaries = {}
    for path in arguments.packages:
        name = os.path.basename(path)
        results = scorePackage(path)
        summaries[name] = summarize(name, results)
        summaries[name]["perScenario"] = {
            f"{d}/{s}": v for (d, s), v in sorted(results.items())
        }
    out = os.path.join(PROJECT_ROOT, "outputs", "final", "package_scores.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(summaries, handle, ensure_ascii=False, indent=2)
    print(f"# wrote {out}")


if __name__ == "__main__":
    main()
