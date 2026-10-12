import json
from pathlib import Path

import pandas as pd


def convertForJson(value):
    if isinstance(value, dict):
        return {key: convertForJson(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [convertForJson(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def writeRunResults(
    outputDirectory,
    summary,
    predictions,
    auditRecords,
    labelInferencePredictions=None,
    backdoorPredictions=None,
):
    outputPath = Path(outputDirectory)
    outputPath.mkdir(parents=True, exist_ok=True)
    with (outputPath / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(convertForJson(summary), handle, ensure_ascii=False, indent=2)
    with (outputPath / "attack_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(convertForJson(auditRecords), handle, ensure_ascii=False, indent=2)
    pd.DataFrame(
        {
            "ID": predictions["ids"],
            "Prediction": predictions["predictions"],
        }
    ).to_csv(outputPath / "main_task_predictions.csv", index=False)
    if backdoorPredictions is not None:
        pd.DataFrame(
            {
                "ID": backdoorPredictions["ids"],
                "Prediction": backdoorPredictions["predictions"],
            }
        ).to_csv(outputPath / "backdoor_predictions.csv", index=False)
    if labelInferencePredictions is not None:
        labelFrame = {
            "ID": labelInferencePredictions["ids"],
            "Prediction": labelInferencePredictions["predictions"],
        }
        optionalColumns = (
            ("stages", "Stage"),
            ("epochs", "Epoch"),
            ("batchIndices", "BatchIndex"),
            ("includedInScore", "IncludedInScore"),
        )
        for sourceName, columnName in optionalColumns:
            if sourceName in labelInferencePredictions:
                labelFrame[columnName] = labelInferencePredictions[sourceName]
        pd.DataFrame(labelFrame).to_csv(
            outputPath / "label_inference_predictions.csv",
            index=False,
        )
    return outputPath
