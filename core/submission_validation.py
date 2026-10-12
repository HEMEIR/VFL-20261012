from pathlib import Path

import pandas as pd

from .submission_schema import DATASET_CLASS_COUNTS


def buildIssue(
    severity,
    code,
    message,
    dataset=None,
    scenario=None,
    fileName=None,
):
    issue = {
        "severity": severity,
        "code": code,
        "message": message,
    }
    if dataset is not None:
        issue["dataset"] = dataset
    if scenario is not None:
        issue["scenario"] = scenario
    if fileName is not None:
        issue["file"] = fileName
    return issue


def validatePredictionFile(
    filePath,
    expectedIds,
    expectedColumns,
    datasetName,
    scenarioName,
    missingSeverity="error",
):
    path = Path(filePath)
    fileName = path.name
    if not path.is_file():
        return None, [
            buildIssue(
                missingSeverity,
                "missing_prediction_file",
                f"Required prediction file is missing: {path}",
                datasetName,
                scenarioName,
                fileName,
            )
        ]

    try:
        frame = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8",
        )
    except Exception as error:
        return None, [
            buildIssue(
                missingSeverity,
                "unreadable_prediction_file",
                f"Could not read {path}: {error}",
                datasetName,
                scenarioName,
                fileName,
            )
        ]

    issues = []
    if tuple(frame.columns) != tuple(expectedColumns):
        issues.append(
            buildIssue(
                missingSeverity,
                "invalid_columns",
                (
                    f"{path} columns must be {list(expectedColumns)}, "
                    f"got {frame.columns.tolist()}"
                ),
                datasetName,
                scenarioName,
                fileName,
            )
        )
        return None, issues

    ids = frame["ID"].astype(str).tolist()
    expectedIds = [str(sampleId) for sampleId in expectedIds]
    if frame["ID"].duplicated().any():
        issues.append(
            buildIssue(
                missingSeverity,
                "duplicate_ids",
                f"{path} contains duplicate IDs",
                datasetName,
                scenarioName,
                fileName,
            )
        )
    if ids != expectedIds:
        issues.append(
            buildIssue(
                missingSeverity,
                "id_order_mismatch",
                (
                    f"{path} must contain exactly {len(expectedIds)} expected "
                    "IDs in the published order"
                ),
                datasetName,
                scenarioName,
                fileName,
            )
        )

    predictionValues = frame["Prediction"].astype(str).str.strip()
    classCount = int(DATASET_CLASS_COUNTS[datasetName])
    allowedPredictions = tuple(str(label) for label in range(classCount))
    invalidPredictionMask = ~predictionValues.isin(allowedPredictions)
    if invalidPredictionMask.any():
        issues.append(
            buildIssue(
                missingSeverity,
                "invalid_predictions",
                (
                    f"{path} Prediction values must be integer labels from "
                    f"0 through {classCount - 1}"
                ),
                datasetName,
                scenarioName,
                fileName,
            )
        )

    if issues:
        return None, issues
    frame = frame.copy()
    frame["Prediction"] = predictionValues.astype(int)
    return frame, []
