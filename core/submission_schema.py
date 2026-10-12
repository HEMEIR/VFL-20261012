import re


DATASET_NAMES = (
    "dataset_01",
    "dataset_02",
    "dataset_03",
    "dataset_04",
    "dataset_05",
)
DATASET_CLASS_COUNTS = {
    "dataset_01": 2,
    "dataset_02": 2,
    "dataset_03": 2,
    "dataset_04": 2,
    "dataset_05": 6,
}

SCENARIO_DEFINITIONS = {
    "clean": {
        "backdoor": None,
        "labelInference": None,
    },
    "lfba_p05": {
        "backdoor": "lfba",
        "labelInference": None,
    },
    "lfba_p10": {
        "backdoor": "lfba",
        "labelInference": None,
    },
    "lfba_p20": {
        "backdoor": "lfba",
        "labelInference": None,
    },
    "villain": {
        "backdoor": "villain",
        "labelInference": None,
    },
    "pmc": {
        "backdoor": None,
        "labelInference": "pmc",
    },
    "amc": {
        "backdoor": None,
        "labelInference": "amc",
    },
    "composite": {
        "backdoor": "lfba",
        "labelInference": "pmc",
    },
}

COMPOSITE_SCENARIO_NAME = "composite"
COMPOSITE_ATTACKS_BY_DATASET = {
    "dataset_01": ("lfba", "amc"),
    "dataset_02": ("lfba", "amc"),
    "dataset_03": ("villain", "pmc"),
    "dataset_04": ("lfba", "amc"),
    "dataset_05": ("villain", "pmc"),
}


def getScenarioDefinition(datasetName, scenarioName):
    definition = dict(SCENARIO_DEFINITIONS[scenarioName])
    if scenarioName != COMPOSITE_SCENARIO_NAME:
        return definition
    backdoorName, labelInferenceName = COMPOSITE_ATTACKS_BY_DATASET[datasetName]
    definition.update(
        {
            "backdoor": backdoorName,
            "labelInference": labelInferenceName,
        }
    )
    return definition


MAIN_TASK_FILE = "main_task_predictions.csv"
BACKDOOR_FILE = "backdoor_predictions.csv"
LABEL_INFERENCE_FILE = "label_inference_predictions.csv"
SUBMISSION_MANIFEST_FILE = "manifest.json"
SUBMISSION_IDENTIFIER_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z"
)


def isValidSubmissionIdentifier(value):
    return (
        isinstance(value, str)
        and SUBMISSION_IDENTIFIER_PATTERN.fullmatch(value) is not None
    )


def getExpectedFiles(scenarioName):
    definition = SCENARIO_DEFINITIONS[scenarioName]
    expectedFiles = [MAIN_TASK_FILE]
    if definition["backdoor"] is not None:
        expectedFiles.append(BACKDOOR_FILE)
    if definition["labelInference"] is not None:
        expectedFiles.append(LABEL_INFERENCE_FILE)
    return tuple(expectedFiles)


def getLabelInferenceColumns(labelInferenceName):
    if labelInferenceName in ("pmc", "amc"):
        return ("ID", "Prediction")
    raise ValueError(f"Unsupported label inference attack: {labelInferenceName}")


def getExpectedRelativeFiles():
    relativeFiles = set()
    for datasetName in DATASET_NAMES:
        for scenarioName in SCENARIO_DEFINITIONS:
            for fileName in getExpectedFiles(scenarioName):
                relativeFiles.add(f"{datasetName}/{scenarioName}/{fileName}")
    return relativeFiles
