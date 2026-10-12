from copy import deepcopy
import hashlib
import math
from pathlib import Path

import yaml

from .formal_config import (
    canonicalJsonBytes,
    isFormalScenarioConfigPath,
    validateFormalConfigLock,
)


REQUIRED_SECTIONS = {"dataset", "model", "training", "backdoor", "labelInference", "output"}
SUPPORTED_DATASETS = {
    "dataset_01",
    "dataset_02",
    "dataset_03",
    "dataset_04",
    "dataset_05",
}
SUPPORTED_INPUT_FORMATS = {"imageHalf", "tabular"}
SUPPORTED_PREPROCESSING = {"none", "trainMinMax"}
SUPPORTED_BOTTOM_MODELS = {"resnet18", "mlp"}
SUPPORTED_LFBA_TRIGGERS = {"imageTriangle", "structuredDimensions"}
SUPPORTED_BACKDOORS = {"none", "lfba", "villain"}
SUPPORTED_LABEL_INFERENCE = {"none", "pmc", "amc"}
SUPPORTED_COMPLETION_BACKENDS = {
    "mixmatch",
    "logistic",
    "prototype",
    "knn",
    "mlp",
    "cv_ensemble",
}
SUPPORTED_COMPLETION_MEMBERS = {
    "mixmatch",
    "logistic",
    "prototype",
    "knn",
    "mlp",
}
SUPPORTED_COMPLETION_PREPROCESSING = {
    "raw",
    "standard",
    "l2",
    "standard_l2",
    "whiten",
}
DEFAULT_DEFENSE_CONFIG_RELATIVE_PATH = Path("configs") / "defense.yaml"
DEFENSE_DOCUMENT_FIELDS = {"defense"}
DEFENSE_FIELDS = {"name", "parameters"}


def validateDefenseParameterValue(value, path="defense.parameters"):
    """Require deterministic JSON-compatible values without limiting semantics."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            validateDefenseParameterValue(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} mapping keys must be strings")
            validateDefenseParameterValue(item, f"{path}.{key}")
        return
    raise ValueError(
        f"{path} must contain only YAML/JSON scalar, list, or mapping values"
    )


def normalizeDefenseConfigDocument(document):
    """Validate one participant-owned document and return a detached copy."""
    if not isinstance(document, dict):
        raise ValueError("Defense configuration root must be a mapping")
    rootFields = set(document)
    if rootFields != DEFENSE_DOCUMENT_FIELDS:
        raise ValueError(
            "Defense configuration root must contain only 'defense'; "
            f"got {sorted(str(field) for field in rootFields)}"
        )
    normalized = {"defense": deepcopy(document["defense"])}
    validateDefenseConfig(normalized)
    normalized["defense"].setdefault("parameters", {})
    return normalized


def loadDefenseConfig(path):
    """Load a defense-only YAML without allowing formal protocol overrides."""
    defensePath = Path(path).resolve()
    if not defensePath.is_file():
        raise FileNotFoundError(
            f"Defense configuration file does not exist: {defensePath}"
        )
    with defensePath.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    return normalizeDefenseConfigDocument(document)


def calculateDefenseConfigHash(defenseConfig):
    normalized = normalizeDefenseConfigDocument(defenseConfig)
    return hashlib.sha256(canonicalJsonBytes(normalized)).hexdigest()


def resolveDefenseConfigPath(projectRoot, pathValue=None):
    """Resolve a submitted defense YAML and keep it inside ``configs/``."""
    projectPath = Path(projectRoot).resolve()
    configuredPath = (
        DEFAULT_DEFENSE_CONFIG_RELATIVE_PATH
        if pathValue is None
        else Path(pathValue)
    )
    candidate = (
        configuredPath
        if configuredPath.is_absolute()
        else projectPath / configuredPath
    ).resolve()
    configsRoot = (projectPath / "configs").resolve()
    try:
        candidate.relative_to(configsRoot)
    except ValueError as error:
        raise ValueError(
            "Defense configuration must be located inside the repository's "
            "configs directory"
        ) from error
    if candidate.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("Defense configuration must use a .yaml or .yml suffix")
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Defense configuration file does not exist: {candidate}"
        )
    return candidate


def validateStructuredTriggerDimensions(backdoorConfig, triggerType, attackName):
    if triggerType != "structuredDimensions":
        return
    dimensionCount = int(backdoorConfig.get("triggerDimensions", 0))
    configuredDimensions = backdoorConfig.get("triggerDimensionIndices")
    if configuredDimensions is None:
        if dimensionCount <= 0:
            raise ValueError(
                f"{attackName} structuredDimensions requires positive triggerDimensions"
            )
        return
    if not isinstance(configuredDimensions, list) or not configuredDimensions:
        raise ValueError("triggerDimensionIndices must be a non-empty list")
    if any(
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension < 0
        for dimension in configuredDimensions
    ):
        raise ValueError("triggerDimensionIndices must contain non-negative integers")
    if len(set(configuredDimensions)) != len(configuredDimensions):
        raise ValueError("triggerDimensionIndices must not contain duplicates")
    if dimensionCount != len(configuredDimensions):
        raise ValueError(
            "triggerDimensions must match the length of triggerDimensionIndices"
        )


def mergeConfig(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = mergeConfig(result[key], value)
        else:
            result[key] = value
    return result


def readConfigTree(path, loadingPaths=None):
    """递归加载场景配置，并拒绝循环继承。

    `extends` 相对于当前 YAML 所在目录解析。场景文件只覆盖攻击开关和
    输出目录，数据集公共训练参数继续由对应正式配置统一维护。
    """
    configPath = Path(path).resolve()
    activePaths = set() if loadingPaths is None else set(loadingPaths)
    if configPath in activePaths:
        raise ValueError(f"Configuration inheritance cycle detected at: {configPath}")
    activePaths.add(configPath)
    with configPath.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    parentSetting = config.pop("extends", None)
    if parentSetting is None:
        return config
    if not isinstance(parentSetting, str) or not parentSetting.strip():
        raise ValueError("Configuration extends must be a non-empty path")
    parentPath = Path(parentSetting)
    if not parentPath.is_absolute():
        parentPath = configPath.parent / parentPath
    parentConfig = readConfigTree(parentPath, activePaths)
    return mergeConfig(parentConfig, config)


def loadConfig(
    path,
    overrides=None,
    enforceFormalLock=True,
    defenseConfig=None,
):
    configPath = Path(path).resolve()
    config = readConfigTree(configPath)
    missing = REQUIRED_SECTIONS.difference(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    config = mergeConfig(config, overrides or {})
    config["configPath"] = str(configPath)
    validateConfig(config)
    if enforceFormalLock and isFormalScenarioConfigPath(configPath):
        validateFormalConfigLock(configPath, config)
    if defenseConfig is not None:
        normalizedDefense = normalizeDefenseConfigDocument(defenseConfig)
        config["defense"] = normalizedDefense["defense"]
        validateDefenseConfig(config)
    return config


def validateConfig(config):
    validateDefenseConfig(config)
    datasetName = str(config["dataset"].get("name", "")).lower()
    if datasetName not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset in configuration: {datasetName}")
    classCount = int(config["dataset"].get("classCount", 2))
    if classCount < 2:
        raise ValueError("dataset.classCount must be at least 2")
    inputFormat = str(config["dataset"].get("inputFormat", ""))
    if inputFormat not in SUPPORTED_INPUT_FORMATS:
        raise ValueError(f"Unsupported dataset inputFormat: {inputFormat}")
    preprocessing = str(config["dataset"].get("preprocessing", ""))
    if preprocessing not in SUPPORTED_PREPROCESSING:
        raise ValueError(f"Unsupported dataset preprocessing: {preprocessing}")
    bottomType = str(config["model"].get("bottomType", ""))
    if bottomType not in SUPPORTED_BOTTOM_MODELS:
        raise ValueError(f"Unsupported model bottomType: {bottomType}")
    if (inputFormat == "imageHalf") != (bottomType == "resnet18"):
        raise ValueError("imageHalf inputFormat and resnet18 bottomType must be configured together")
    embeddingDim = int(config["model"].get("embeddingDim", 0))
    if embeddingDim != 64:
        raise ValueError("The competition protocol requires a 64-dimensional embedding per party")
    if bottomType == "mlp":
        hiddenDims = config["model"].get("bottomHiddenDims")
        if not isinstance(hiddenDims, list) or len(hiddenDims) != 3:
            raise ValueError("model.bottomHiddenDims must contain exactly three dimensions")
        if any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in hiddenDims
        ):
            raise ValueError("model.bottomHiddenDims must contain positive integers")
    for fieldName in ("epochs", "batchSize"):
        if int(config["training"].get(fieldName, 0)) <= 0:
            raise ValueError(f"training.{fieldName} must be positive")
    if float(config["training"].get("learningRate", 0.0)) <= 0:
        raise ValueError("training.learningRate must be positive")
    progressEnabled = config["training"].get("progressEnabled", False)
    if not isinstance(progressEnabled, bool):
        raise ValueError("training.progressEnabled must be a boolean")
    if float(config["training"].get("progressMinInterval", 0.5)) <= 0.0:
        raise ValueError("training.progressMinInterval must be positive")
    backdoorName = str(config["backdoor"].get("name", "none")).lower()
    if backdoorName not in SUPPORTED_BACKDOORS:
        raise ValueError(f"Unsupported backdoor attack: {backdoorName}")
    if backdoorName == "lfba":
        targetLabel = int(config["backdoor"].get("targetLabel", -1))
        if not 0 <= targetLabel < int(classCount):
            raise ValueError("LFBA targetLabel must be within the dataset classes")
        triggerType = str(config["backdoor"].get("lfbaTriggerType", ""))
        if triggerType not in SUPPORTED_LFBA_TRIGGERS:
            raise ValueError(f"Unsupported LFBA trigger type: {triggerType}")
        if (triggerType == "imageTriangle") != (inputFormat == "imageHalf"):
            raise ValueError("LFBA imageTriangle trigger requires imageHalf input")
        if config["backdoor"].get("anchorId") is None:
            raise ValueError("LFBA requires an explicit anchorId or zero-based anchor position")
        poisonRate = float(config["backdoor"].get("poisonRate", 0.0))
        switchRate = float(config["backdoor"].get("switchRate", 0.0))
        if not 0.0 < poisonRate < 1.0:
            raise ValueError("LFBA poisonRate must be between 0 and 1")
        if not 0.0 < switchRate <= 1.0:
            raise ValueError("LFBA switchRate must be between 0 and 1")
        if int(config["training"]["epochs"]) < 2:
            raise ValueError("LFBA needs at least two epochs: collect gradients, then inject")
        validateStructuredTriggerDimensions(
            config["backdoor"],
            triggerType,
            "LFBA",
        )
    if backdoorName == "villain":
        validateVillainConfig(config, classCount, embeddingDim)
    labelInferenceName = str(config["labelInference"].get("name", "none")).lower()
    if labelInferenceName not in SUPPORTED_LABEL_INFERENCE:
        raise ValueError(f"Unsupported label inference attack: {labelInferenceName}")
    if labelInferenceName in {"pmc", "amc"}:
        validatePmcConfig(config)
    if labelInferenceName == "amc":
        validateAmcConfig(config)


def validateDefenseConfig(config):
    """Validate the optional participant-owned defense configuration."""
    if "defense" not in config:
        return
    defenseConfig = config["defense"]
    if not isinstance(defenseConfig, dict):
        raise ValueError("defense must be a mapping")
    unexpectedFields = set(defenseConfig).difference(DEFENSE_FIELDS)
    if unexpectedFields:
        raise ValueError(
            "defense may contain only name and parameters; unexpected fields: "
            f"{sorted(str(field) for field in unexpectedFields)}"
        )
    defenseName = defenseConfig.get("name")
    if not isinstance(defenseName, str) or not defenseName.strip():
        raise ValueError("defense.name must be a non-empty string")
    parameters = defenseConfig.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("defense.parameters must be a mapping")
    validateDefenseParameterValue(parameters)


def validateVillainConfig(config, classCount, embeddingDim):
    backdoorConfig = config["backdoor"]
    targetLabel = int(backdoorConfig.get("targetLabel", -1))
    if not 0 <= targetLabel < int(classCount):
        raise ValueError("VILLAIN targetLabel must be within the dataset classes")
    knownTargetIds = backdoorConfig.get("knownTargetIds")
    if not isinstance(knownTargetIds, list) or not knownTargetIds:
        raise ValueError("VILLAIN requires a non-empty knownTargetIds list")
    normalizedIds = [str(sampleId) for sampleId in knownTargetIds]
    if len(set(normalizedIds)) != len(normalizedIds):
        raise ValueError("VILLAIN knownTargetIds must not contain duplicates")
    poisonRate = float(backdoorConfig.get("poisonRate", 0.0))
    if not 0.0 < poisonRate <= 1.0:
        raise ValueError("VILLAIN poisonRate must be in (0, 1]")
    triggerDimensionCount = int(
        backdoorConfig.get("triggerDimensionCount", 0)
    )
    if not 0 < triggerDimensionCount <= int(embeddingDim):
        raise ValueError(
            "VILLAIN triggerDimensionCount must be within model.embeddingDim"
        )
    if float(backdoorConfig.get("beta", 0.0)) <= 0.0:
        raise ValueError("VILLAIN beta must be positive")
    startEpoch = int(backdoorConfig.get("startEpoch", -1))
    if not 0 <= startEpoch < int(config["training"]["epochs"]):
        raise ValueError("VILLAIN startEpoch must be within the training epochs")
    testTriggerRate = float(backdoorConfig.get("testTriggerRate", 0.5))
    if not 0.0 < testTriggerRate <= 1.0:
        raise ValueError("VILLAIN testTriggerRate must be in (0, 1]")
    minimumAccuracy = float(
        backdoorConfig.get("minimumMainTaskAccuracy", 0.0)
    )
    if not 0.0 <= minimumAccuracy <= 1.0:
        raise ValueError(
            "VILLAIN minimumMainTaskAccuracy must be between 0 and 1"
        )
    triggerKeepRate = float(backdoorConfig.get("triggerKeepRate", 1.0))
    if not 0.0 < triggerKeepRate <= 1.0:
        raise ValueError("VILLAIN triggerKeepRate must be in (0, 1]")
    triggerShiftMin = float(backdoorConfig.get("triggerShiftMin", 1.0))
    triggerShiftMax = float(backdoorConfig.get("triggerShiftMax", 1.0))
    if triggerShiftMin <= 0.0:
        raise ValueError("VILLAIN triggerShiftMin must be positive")
    if triggerShiftMax < triggerShiftMin:
        raise ValueError(
            "VILLAIN triggerShiftMax must be greater than or equal to triggerShiftMin"
        )
    baseLearningRate = float(config["training"]["learningRate"])
    attackerPreAttackLearningRate = float(
        backdoorConfig.get("attackerPreAttackLearningRate", baseLearningRate)
    )
    attackerAttackLearningRate = float(
        backdoorConfig.get("attackerAttackLearningRate", baseLearningRate)
    )
    if attackerPreAttackLearningRate <= 0.0:
        raise ValueError(
            "VILLAIN attackerPreAttackLearningRate must be positive"
        )
    if attackerAttackLearningRate <= 0.0:
        raise ValueError("VILLAIN attackerAttackLearningRate must be positive")


def validatePmcConfig(config):
    labelInferenceConfig = config["labelInference"]
    auxiliaryLabelsPath = labelInferenceConfig.get("auxiliaryLabelsPath")
    if not isinstance(auxiliaryLabelsPath, str) or not auxiliaryLabelsPath.strip():
        raise ValueError("PMC requires an explicit auxiliaryLabelsPath")
    configuredPath = Path(auxiliaryLabelsPath)
    if not configuredPath.is_absolute():
        configuredPath = Path(config["configPath"]).resolve().parent / configuredPath
    if not configuredPath.is_file():
        raise ValueError(f"PMC auxiliary label file does not exist: {configuredPath.resolve()}")
    if int(labelInferenceConfig.get("evaluationBatchSize", 0)) <= 0:
        raise ValueError("labelInference.evaluationBatchSize must be positive")
    backend = str(
        labelInferenceConfig.get("completionBackend", "mixmatch")
    ).lower()
    if backend not in SUPPORTED_COMPLETION_BACKENDS:
        raise ValueError(f"Unsupported completion backend: {backend}")
    if backend == "mixmatch":
        validateMixMatchCompletionConfig(labelInferenceConfig, "labelInference")
        return
    if backend in SUPPORTED_COMPLETION_MEMBERS.difference({"mixmatch"}):
        validateClassicalCompletionConfig(
            backend,
            labelInferenceConfig.get(backend, {}),
            f"labelInference.{backend}",
        )
        return
    completionConfig = labelInferenceConfig.get("completion")
    if not isinstance(completionConfig, dict):
        raise ValueError(
            "labelInference.completion must be a mapping for cv_ensemble"
        )
    foldCount = int(completionConfig.get("folds", 0))
    if foldCount < 2:
        raise ValueError("labelInference.completion.folds must be at least 2")
    selectionMetric = str(
        completionConfig.get("selectionMetric", "accuracy")
    ).lower()
    if selectionMetric not in {"accuracy", "auc", "macro_recall"}:
        raise ValueError(
            "labelInference.completion.selectionMetric must be "
            "accuracy, auc or macro_recall"
        )
    members = completionConfig.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError(
            "labelInference.completion.members must be a non-empty list"
        )
    normalizedMembers = [str(member).lower() for member in members]
    if len(set(normalizedMembers)) != len(normalizedMembers):
        raise ValueError("labelInference.completion.members must be unique")
    unsupportedMembers = set(normalizedMembers).difference(
        SUPPORTED_COMPLETION_MEMBERS
    )
    if unsupportedMembers:
        raise ValueError(
            "Unsupported completion ensemble members: "
            f"{sorted(unsupportedMembers)}"
        )
    for member in normalizedMembers:
        if member == "mixmatch":
            effectiveConfig = mergeConfig(
                labelInferenceConfig,
                labelInferenceConfig.get("mixmatch", {}),
            )
            validateMixMatchCompletionConfig(
                effectiveConfig,
                "labelInference.mixmatch",
            )
        else:
            validateClassicalCompletionConfig(
                member,
                labelInferenceConfig.get(member, {}),
                f"labelInference.{member}",
            )


def validateMixMatchCompletionConfig(completionConfig, prefix):
    for fieldName in (
        "completionEpochs",
        "iterationsPerEpoch",
        "completionBatchSize",
    ):
        if int(completionConfig.get(fieldName, 0)) <= 0:
            raise ValueError(f"{prefix}.{fieldName} must be positive")
    if float(completionConfig.get("completionLearningRate", 0.0)) <= 0.0:
        raise ValueError(f"{prefix}.completionLearningRate must be positive")
    if float(completionConfig.get("temperature", 0.0)) <= 0.0:
        raise ValueError(f"{prefix}.temperature must be positive")
    if float(completionConfig.get("mixupAlpha", 0.0)) <= 0.0:
        raise ValueError(f"{prefix}.mixupAlpha must be positive")
    if float(completionConfig.get("lambdaU", -1.0)) < 0.0:
        raise ValueError(f"{prefix}.lambdaU must not be negative")
    emaDecay = float(completionConfig.get("emaDecay", -1.0))
    if not 0.0 <= emaDecay < 1.0:
        raise ValueError(f"{prefix}.emaDecay must be in [0, 1)")
    headLayers = int(completionConfig.get("headLayers", 0))
    if not 1 <= headLayers <= 5:
        raise ValueError(f"{prefix}.headLayers must be between 1 and 5")
    activation = str(completionConfig.get("activation", "")).lower()
    if activation not in {"relu", "sigmoid", "none"}:
        raise ValueError(f"{prefix}.activation must be relu, sigmoid or none")
    initialization = str(
        completionConfig.get("headInitialization", "ones")
    ).lower()
    if initialization not in {"ones", "kaiming"}:
        raise ValueError(f"{prefix}.headInitialization must be ones or kaiming")


def validateClassicalCompletionConfig(family, familyConfig, prefix):
    if not isinstance(familyConfig, dict):
        raise ValueError(f"{prefix} must be a mapping")
    preprocessing = str(familyConfig.get("preprocessing", "raw")).lower()
    if preprocessing not in SUPPORTED_COMPLETION_PREPROCESSING:
        raise ValueError(
            f"Unsupported completion preprocessing for {prefix}: {preprocessing}"
        )
    if preprocessing == "whiten" and int(
        familyConfig.get("whitenComponents", 32)
    ) <= 0:
        raise ValueError(f"{prefix}.whitenComponents must be positive")
    if family == "logistic":
        if float(familyConfig.get("c", 1.0)) <= 0.0:
            raise ValueError(f"{prefix}.c must be positive")
    elif family == "prototype":
        if float(familyConfig.get("temperature", 0.1)) <= 0.0:
            raise ValueError(f"{prefix}.temperature must be positive")
    elif family == "knn":
        if int(familyConfig.get("neighbors", 5)) <= 0:
            raise ValueError(f"{prefix}.neighbors must be positive")
        weights = str(familyConfig.get("weights", "uniform")).lower()
        if weights not in {"uniform", "distance"}:
            raise ValueError(f"{prefix}.weights must be uniform or distance")
    elif family == "mlp":
        hiddenDims = familyConfig.get("hiddenDims", [64])
        if (
            not isinstance(hiddenDims, list)
            or not hiddenDims
            or any(
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension <= 0
                for dimension in hiddenDims
            )
        ):
            raise ValueError(f"{prefix}.hiddenDims must contain positive integers")
        if float(familyConfig.get("alpha", 0.1)) < 0.0:
            raise ValueError(f"{prefix}.alpha must not be negative")
    if int(familyConfig.get("maxIterations", 2000)) <= 0:
        raise ValueError(f"{prefix}.maxIterations must be positive")


def validateAmcConfig(config):
    labelInferenceConfig = config["labelInference"]
    activeStartEpoch = int(labelInferenceConfig.get("activeStartEpoch", 1))
    if not 1 <= activeStartEpoch <= int(config["training"]["epochs"]):
        raise ValueError(
            "labelInference.activeStartEpoch must be between 1 and training.epochs"
        )
    momentum = float(labelInferenceConfig.get("maliciousMomentum", -1.0))
    if not 0.0 <= momentum < 1.0:
        raise ValueError("labelInference.maliciousMomentum must be in [0, 1)")
    dampening = float(labelInferenceConfig.get("maliciousDampening", -1.0))
    if not 0.0 <= dampening <= 1.0:
        raise ValueError("labelInference.maliciousDampening must be in [0, 1]")
    if float(labelInferenceConfig.get("gradientScaleGamma", -1.0)) < 0.0:
        raise ValueError("labelInference.gradientScaleGamma must not be negative")
    minimumScale = float(labelInferenceConfig.get("minimumGradientScale", 0.0))
    maximumScale = float(labelInferenceConfig.get("maximumGradientScale", 0.0))
    if minimumScale <= 0.0:
        raise ValueError("labelInference.minimumGradientScale must be positive")
    if maximumScale < minimumScale:
        raise ValueError(
            "labelInference.maximumGradientScale must not be smaller than minimumGradientScale"
        )
    if float(labelInferenceConfig.get("gradientScaleEpsilon", 0.0)) <= 0.0:
        raise ValueError("labelInference.gradientScaleEpsilon must be positive")
