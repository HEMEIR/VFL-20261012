import copy
import math
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from core.progress import writeProgress
from metrics.privacy import buildPrivacyMetrics
from metrics.utility import calculateAccuracy, calculateAuc

from .model_completion import (
    PartyBFeatureDataset,
    PassiveCompletionModel,
    PmcResult,
    buildCompletionSubmissionPredictions,
    loadAuxiliaryLabels,
    trainPmcEpoch,
)


SUPPORTED_CLASSICAL_BACKENDS = {"logistic", "prototype", "knn", "mlp"}


@dataclass
class CompletionBackendArtifact:
    backendName: str
    backendState: dict

    def state_dict(self):
        return {
            "completionBackend": self.backendName,
            "backendState": self.backendState,
        }


def makePreprocessor(name, config, trainEmbeddings):
    normalizedName = str(name).lower()
    if normalizedName == "raw":
        return FunctionTransformer(validate=False)
    if normalizedName == "standard":
        return StandardScaler()
    if normalizedName == "l2":
        return Normalizer(norm="l2")
    if normalizedName == "standard_l2":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("normalize", Normalizer(norm="l2")),
            ]
        )
    if normalizedName == "whiten":
        requestedComponents = int(config.get("whitenComponents", 32))
        componentCount = min(
            requestedComponents,
            int(trainEmbeddings.shape[1]),
            max(int(trainEmbeddings.shape[0]) - 1, 1),
        )
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "pca",
                    PCA(
                        n_components=componentCount,
                        whiten=True,
                        random_state=int(config.get("seed", 100)),
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported completion preprocessing: {name}")


def expandClassProbabilities(probabilities, classes, classCount):
    expanded = np.zeros((len(probabilities), int(classCount)), dtype=np.float64)
    expanded[:, np.asarray(classes, dtype=np.int64)] = np.asarray(
        probabilities,
        dtype=np.float64,
    )
    return expanded


def normalizeProbabilityMatrix(probabilities, classCount):
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim == 1:
        if int(classCount) != 2:
            raise ValueError("One-dimensional probabilities are valid only for binary data")
        values = np.stack([1.0 - values, values], axis=1)
    if values.ndim != 2 or values.shape[1] != int(classCount):
        raise ValueError(
            "Completion probabilities must have shape "
            f"[sampleCount, {int(classCount)}]"
        )
    values = np.clip(values, 1e-12, 1.0)
    return values / values.sum(axis=1, keepdims=True)


def fitClassicalPredictor(
    family,
    familyConfig,
    trainEmbeddings,
    trainLabels,
    classCount,
    seed,
):
    normalizedFamily = str(family).lower()
    config = dict(familyConfig or {})
    config["seed"] = int(seed)
    preprocessor = makePreprocessor(
        config.get("preprocessing", "raw"),
        config,
        trainEmbeddings,
    )
    transformedTrain = preprocessor.fit_transform(trainEmbeddings)
    labels = np.asarray(trainLabels, dtype=np.int64)
    state = {
        "family": normalizedFamily,
        "config": dict(familyConfig or {}),
        "preprocessor": preprocessor,
        "classCount": int(classCount),
    }
    if normalizedFamily == "logistic":
        estimator = LogisticRegression(
            C=float(config.get("c", 1.0)),
            max_iter=int(config.get("maxIterations", 5000)),
            solver="lbfgs",
            random_state=int(seed),
        )
        estimator.fit(transformedTrain, labels)
        state["estimator"] = estimator
        return state
    if normalizedFamily == "knn":
        estimator = KNeighborsClassifier(
            n_neighbors=int(config.get("neighbors", 5)),
            weights=str(config.get("weights", "uniform")).lower(),
            metric="euclidean",
        )
        estimator.fit(transformedTrain, labels)
        state["estimator"] = estimator
        return state
    if normalizedFamily == "mlp":
        estimator = MLPClassifier(
            hidden_layer_sizes=tuple(config.get("hiddenDims", [64])),
            activation="relu",
            solver="lbfgs",
            alpha=float(config.get("alpha", 0.1)),
            max_iter=int(config.get("maxIterations", 2000)),
            random_state=int(seed),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            estimator.fit(transformedTrain, labels)
        state["estimator"] = estimator
        return state
    if normalizedFamily == "prototype":
        normalizedTrain = transformedTrain / np.maximum(
            np.linalg.norm(transformedTrain, axis=1, keepdims=True),
            1e-12,
        )
        prototypes = []
        for label in range(int(classCount)):
            center = normalizedTrain[labels == label].mean(axis=0)
            center = center / max(float(np.linalg.norm(center)), 1e-12)
            prototypes.append(center)
        state["prototypes"] = np.stack(prototypes)
        state["temperature"] = float(config.get("temperature", 0.1))
        return state
    raise ValueError(f"Unsupported classical completion backend: {family}")


def predictClassicalPredictor(state, embeddings):
    transformed = state["preprocessor"].transform(embeddings)
    family = state["family"]
    if family in {"logistic", "knn", "mlp"}:
        estimator = state["estimator"]
        probabilities = estimator.predict_proba(transformed)
        return expandClassProbabilities(
            probabilities,
            estimator.classes_,
            state["classCount"],
        )
    if family == "prototype":
        normalized = transformed / np.maximum(
            np.linalg.norm(transformed, axis=1, keepdims=True),
            1e-12,
        )
        logits = normalized @ state["prototypes"].T
        logits = logits / state["temperature"]
        logits = logits - logits.max(axis=1, keepdims=True)
        exponentials = np.exp(logits)
        return exponentials / exponentials.sum(axis=1, keepdims=True)
    raise ValueError(f"Unsupported classical completion backend: {family}")


def extractEmbeddings(bottomModel, features, batchSize, device):
    bottomModel.eval()
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(features), int(batchSize)):
            embeddings.append(
                bottomModel(features[start : start + int(batchSize)].to(device)).cpu()
            )
    return torch.cat(embeddings).numpy().astype(np.float64)


def calculateDetailedMetrics(labels, probabilities):
    probabilityMatrix = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    predictions = probabilityMatrix.argmax(axis=1)
    matrix = confusion_matrix(
        labels,
        predictions,
        labels=list(range(probabilityMatrix.shape[1])),
    )
    recalls = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)
    return {
        "labelRecoveryAccuracy": float((predictions == labels).mean()),
        "labelRecoveryAuc": float(
            roc_auc_score(
                labels,
                probabilityMatrix,
                multi_class="ovr",
                average="macro",
            )
            if probabilityMatrix.shape[1] > 2
            else roc_auc_score(labels, probabilityMatrix[:, 1])
        ),
        "logLoss": float(
            log_loss(
                labels,
                probabilityMatrix,
                labels=list(range(probabilityMatrix.shape[1])),
            )
        ),
        "macroRecall": float(recalls.mean()),
        "minimumClassRecall": float(recalls.min()),
        "classRecalls": recalls.tolist(),
        "predictionCounts": np.bincount(
            predictions,
            minlength=probabilityMatrix.shape[1],
        ).tolist(),
    }


def rankMetrics(metricValues, selectionMetric):
    normalizedMetric = str(selectionMetric).lower()
    primaryKeys = {
        "accuracy": "labelRecoveryAccuracy",
        "auc": "labelRecoveryAuc",
        "macro_recall": "macroRecall",
    }
    primaryKey = primaryKeys[normalizedMetric]
    return (
        metricValues[primaryKey],
        metricValues["labelRecoveryAccuracy"],
        metricValues["macroRecall"],
        metricValues["labelRecoveryAuc"],
        metricValues["minimumClassRecall"],
        -metricValues["logLoss"],
    )


def serializeProbabilities(probabilities):
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape[1] == 2:
        return values[:, 1].tolist()
    return values.tolist()


def buildPredictionRecord(dataset, probabilities):
    values = np.asarray(probabilities, dtype=np.float64)
    return {
        "ids": [str(sampleId) for sampleId in dataset.ids.tolist()],
        "probabilities": serializeProbabilities(values),
        "predictions": values.argmax(axis=1).astype(np.int64).tolist(),
    }


def buildEvaluationRecord(dataset, probabilities):
    record = buildPredictionRecord(dataset, probabilities)
    labels = dataset.labels.cpu().tolist()
    record["labels"] = labels
    record["metrics"] = buildPrivacyMetrics(
        calculateAccuracy(labels, record["predictions"]),
        calculateAuc(labels, record["probabilities"]),
    )
    return record


def buildStratifiedFolds(labels, foldCount, seed):
    labels = np.asarray(labels, dtype=np.int64)
    classCounts = np.bincount(labels)
    if int(classCounts.min()) < int(foldCount):
        raise ValueError(
            "completion.folds must not exceed the auxiliary-label count "
            "of the smallest class"
        )
    splitter = StratifiedKFold(
        n_splits=int(foldCount),
        shuffle=True,
        random_state=int(seed),
    )
    return list(splitter.split(np.zeros(len(labels)), labels))


def runClassicalMemberCrossValidation(
    family,
    familyConfig,
    folds,
    auxiliaryEmbeddings,
    auxiliaryLabels,
    datasetEmbeddings,
    classCount,
    seed,
):
    oofProbabilities = np.zeros(
        (len(auxiliaryLabels), int(classCount)),
        dtype=np.float64,
    )
    datasetFoldProbabilities = {
        splitName: [] for splitName in datasetEmbeddings
    }
    foldStates = []
    for foldIndex, (trainIndices, holdoutIndices) in enumerate(folds):
        state = fitClassicalPredictor(
            family,
            familyConfig,
            auxiliaryEmbeddings[trainIndices],
            auxiliaryLabels[trainIndices],
            classCount,
            int(seed) + foldIndex,
        )
        oofProbabilities[holdoutIndices] = predictClassicalPredictor(
            state,
            auxiliaryEmbeddings[holdoutIndices],
        )
        for splitName, embeddings in datasetEmbeddings.items():
            datasetFoldProbabilities[splitName].append(
                predictClassicalPredictor(state, embeddings)
            )
        foldStates.append(state)
    return {
        "family": family,
        "config": dict(familyConfig or {}),
        "oofProbabilities": oofProbabilities,
        "oofMetrics": calculateDetailedMetrics(auxiliaryLabels, oofProbabilities),
        "datasetProbabilities": {
            splitName: np.mean(probabilities, axis=0)
            for splitName, probabilities in datasetFoldProbabilities.items()
        },
        "artifactState": {
            "family": family,
            "foldStates": foldStates,
        },
    }


def initializeCompletionHead(model, initialization):
    normalizedInitialization = str(initialization).lower()
    if normalizedInitialization == "ones":
        return
    if normalizedInitialization != "kaiming":
        raise ValueError(
            f"Unsupported completion head initialization: {initialization}"
        )
    for layer in model.inferenceHead.layers:
        nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


def evaluateRawCompletionProbabilities(model, features, batchSize, device):
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(features), int(batchSize)):
            logits = model(features[start : start + int(batchSize)].to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probabilities)


def buildFoldCompletionLoaders(
    trainDataset,
    auxiliaryLabels,
    trainIndices,
    batchSize,
    seed,
):
    trainPositions = [auxiliaryLabels.positions[index] for index in trainIndices]
    labeledDataset = TensorDataset(
        trainDataset.partyB[trainPositions],
        auxiliaryLabels.labels[trainIndices],
    )
    excludedPositions = set(int(position) for position in auxiliaryLabels.positions)
    unlabeledPositions = [
        position
        for position in range(len(trainDataset))
        if position not in excludedPositions
    ]
    labeledLoader = DataLoader(
        labeledDataset,
        batch_size=int(batchSize),
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    unlabeledLoader = DataLoader(
        PartyBFeatureDataset(trainDataset, unlabeledPositions),
        batch_size=int(batchSize),
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(int(seed) + 1),
    )
    return labeledLoader, unlabeledLoader


def mergeMixMatchConfig(labelInferenceConfig):
    merged = dict(labelInferenceConfig)
    merged.update(dict(labelInferenceConfig.get("mixmatch", {})))
    return merged


def runMixMatchMemberCrossValidation(
    bottomB,
    bundle,
    labelInferenceConfig,
    auxiliaryLabels,
    folds,
    classCount,
    attackSeed,
    selectionMetric,
    device,
    progressConfig,
    moduleName,
):
    mixMatchConfig = mergeMixMatchConfig(labelInferenceConfig)
    batchSize = int(mixMatchConfig["completionBatchSize"])
    evaluationBatchSize = int(mixMatchConfig.get("evaluationBatchSize", 512))
    oofProbabilities = np.zeros(
        (len(auxiliaryLabels.labels), int(classCount)),
        dtype=np.float64,
    )
    datasetFoldProbabilities = {
        "train": [],
        "validation": [],
        "test": [],
    }
    foldRecords = []
    foldStates = []
    auxiliaryLabelArray = auxiliaryLabels.labels.numpy()
    for foldIndex, (trainIndices, holdoutIndices) in enumerate(folds):
        foldSeed = int(attackSeed) + foldIndex
        torch.manual_seed(foldSeed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(foldSeed)
        labeledLoader, unlabeledLoader = buildFoldCompletionLoaders(
            bundle.train,
            auxiliaryLabels,
            trainIndices,
            batchSize,
            foldSeed,
        )
        model = PassiveCompletionModel(
            bottomB,
            int(mixMatchConfig.get("embeddingDim", 64)),
            int(classCount),
            int(mixMatchConfig.get("headLayers", 1)),
            str(mixMatchConfig.get("activation", "relu")),
            bool(mixMatchConfig.get("useBatchNorm", True)),
        ).to(device)
        initializeCompletionHead(
            model,
            mixMatchConfig.get("headInitialization", "ones"),
        )
        emaModel = copy.deepcopy(model).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(mixMatchConfig["completionLearningRate"]),
        )
        numpyGenerator = np.random.RandomState(foldSeed)
        torchGenerator = torch.Generator().manual_seed(foldSeed + 2)
        holdoutPositions = [
            auxiliaryLabels.positions[index] for index in holdoutIndices
        ]
        holdoutFeatures = bundle.train.partyB[holdoutPositions]
        holdoutLabels = auxiliaryLabelArray[holdoutIndices]
        bestRank = None
        bestEpoch = None
        bestState = None
        history = []
        for epoch in range(int(mixMatchConfig["completionEpochs"])):
            trainingMetrics = trainPmcEpoch(
                model,
                emaModel,
                optimizer,
                labeledLoader,
                unlabeledLoader,
                mixMatchConfig,
                epoch,
                device,
                numpyGenerator,
                torchGenerator,
                int(classCount),
                progressConfig,
                f"{moduleName}-CV{foldIndex + 1}",
            )
            holdoutProbabilities = evaluateRawCompletionProbabilities(
                emaModel,
                holdoutFeatures,
                evaluationBatchSize,
                device,
            )
            holdoutMetrics = calculateDetailedMetrics(
                holdoutLabels,
                holdoutProbabilities,
            )
            history.append(
                {
                    "epoch": epoch + 1,
                    "training": trainingMetrics,
                    "holdout": holdoutMetrics,
                }
            )
            currentRank = rankMetrics(holdoutMetrics, selectionMetric)
            if bestRank is None or currentRank > bestRank:
                bestRank = currentRank
                bestEpoch = epoch + 1
                bestState = {
                    key: value.detach().cpu().clone()
                    for key, value in emaModel.state_dict().items()
                }
        emaModel.load_state_dict(bestState)
        selectedHoldoutProbabilities = evaluateRawCompletionProbabilities(
            emaModel,
            holdoutFeatures,
            evaluationBatchSize,
            device,
        )
        oofProbabilities[holdoutIndices] = selectedHoldoutProbabilities
        for splitName, dataset in (
            ("train", bundle.train),
            ("validation", bundle.validation),
            ("test", bundle.test),
        ):
            datasetFoldProbabilities[splitName].append(
                evaluateRawCompletionProbabilities(
                    emaModel,
                    dataset.partyB,
                    evaluationBatchSize,
                    device,
                )
            )
        selectedMetrics = calculateDetailedMetrics(
            holdoutLabels,
            selectedHoldoutProbabilities,
        )
        foldRecords.append(
            {
                "fold": foldIndex + 1,
                "trainAuxiliaryCount": len(trainIndices),
                "holdoutAuxiliaryCount": len(holdoutIndices),
                "bestEpoch": bestEpoch,
                "holdoutMetrics": selectedMetrics,
                "history": history,
            }
        )
        foldStates.append(
            {
                "fold": foldIndex + 1,
                "bestEpoch": bestEpoch,
                "modelState": bestState,
            }
        )
        writeProgress(
            progressConfig,
            (
                f"[{moduleName}] CV fold {foldIndex + 1}/{len(folds)} finished | "
                f"best_epoch={bestEpoch} | "
                f"holdout_acc={selectedMetrics['labelRecoveryAccuracy']:.4f}"
            ),
        )
        del model, emaModel, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "family": "mixmatch",
        "config": {
            key: mixMatchConfig[key]
            for key in (
                "completionEpochs",
                "iterationsPerEpoch",
                "completionBatchSize",
                "completionLearningRate",
                "temperature",
                "mixupAlpha",
                "lambdaU",
                "emaDecay",
            )
        },
        "oofProbabilities": oofProbabilities,
        "oofMetrics": calculateDetailedMetrics(
            auxiliaryLabelArray,
            oofProbabilities,
        ),
        "datasetProbabilities": {
            splitName: np.mean(probabilities, axis=0)
            for splitName, probabilities in datasetFoldProbabilities.items()
        },
        "folds": foldRecords,
        "artifactState": {
            "family": "mixmatch",
            "modelConfig": {
                "embeddingDim": int(mixMatchConfig.get("embeddingDim", 64)),
                "classCount": int(classCount),
                "headLayers": int(mixMatchConfig.get("headLayers", 1)),
                "activation": str(mixMatchConfig.get("activation", "relu")),
                "useBatchNorm": bool(mixMatchConfig.get("useBatchNorm", True)),
                "headInitialization": str(
                    mixMatchConfig.get("headInitialization", "ones")
                ),
            },
            "foldStates": foldStates,
        },
    }


def fitSingleClassicalBackend(
    family,
    familyConfig,
    bottomB,
    bundle,
    auxiliaryLabels,
    evaluationBatchSize,
    classCount,
    attackSeed,
    device,
):
    datasetEmbeddings = {
        "train": extractEmbeddings(
            bottomB,
            bundle.train.partyB,
            evaluationBatchSize,
            device,
        ),
        "validation": extractEmbeddings(
            bottomB,
            bundle.validation.partyB,
            evaluationBatchSize,
            device,
        ),
        "test": extractEmbeddings(
            bottomB,
            bundle.test.partyB,
            evaluationBatchSize,
            device,
        ),
    }
    auxiliaryEmbeddings = datasetEmbeddings["train"][auxiliaryLabels.positions]
    state = fitClassicalPredictor(
        family,
        familyConfig,
        auxiliaryEmbeddings,
        auxiliaryLabels.labels.numpy(),
        classCount,
        attackSeed,
    )
    return {
        "datasetProbabilities": {
            splitName: predictClassicalPredictor(state, embeddings)
            for splitName, embeddings in datasetEmbeddings.items()
        },
        "artifactState": {
            "bottomModelState": {
                key: value.detach().cpu().clone()
                for key, value in bottomB.state_dict().items()
            },
            "predictorState": state,
        },
        "members": [
            {
                "family": family,
                "config": dict(familyConfig or {}),
            }
        ],
        "selectionPolicy": "fit once on every declared auxiliary label",
        "oofMetrics": None,
    }


def fitCrossValidatedEnsemble(
    bottomB,
    bundle,
    labelInferenceConfig,
    auxiliaryLabels,
    evaluationBatchSize,
    classCount,
    attackSeed,
    device,
    progressConfig,
    moduleName,
):
    completionConfig = dict(labelInferenceConfig.get("completion", {}))
    foldCount = int(completionConfig.get("folds", 5))
    selectionMetric = str(
        completionConfig.get("selectionMetric", "accuracy")
    ).lower()
    members = [
        str(member).lower() for member in completionConfig.get("members", [])
    ]
    auxiliaryLabelArray = auxiliaryLabels.labels.numpy()
    folds = buildStratifiedFolds(auxiliaryLabelArray, foldCount, attackSeed)
    memberResults = []
    classicalMembers = [
        member for member in members if member in SUPPORTED_CLASSICAL_BACKENDS
    ]
    datasetEmbeddings = None
    auxiliaryEmbeddings = None
    if classicalMembers:
        datasetEmbeddings = {
            "train": extractEmbeddings(
                bottomB,
                bundle.train.partyB,
                evaluationBatchSize,
                device,
            ),
            "validation": extractEmbeddings(
                bottomB,
                bundle.validation.partyB,
                evaluationBatchSize,
                device,
            ),
            "test": extractEmbeddings(
                bottomB,
                bundle.test.partyB,
                evaluationBatchSize,
                device,
            ),
        }
        auxiliaryEmbeddings = datasetEmbeddings["train"][auxiliaryLabels.positions]
    for member in members:
        writeProgress(
            progressConfig,
            f"[{moduleName}] completion member started | member={member}",
        )
        if member in SUPPORTED_CLASSICAL_BACKENDS:
            result = runClassicalMemberCrossValidation(
                member,
                labelInferenceConfig.get(member, {}),
                folds,
                auxiliaryEmbeddings,
                auxiliaryLabelArray,
                datasetEmbeddings,
                classCount,
                attackSeed,
            )
        elif member == "mixmatch":
            result = runMixMatchMemberCrossValidation(
                bottomB,
                bundle,
                labelInferenceConfig,
                auxiliaryLabels,
                folds,
                classCount,
                attackSeed,
                selectionMetric,
                device,
                progressConfig,
                moduleName,
            )
        else:
            raise ValueError(f"Unsupported completion ensemble member: {member}")
        memberResults.append(result)
        writeProgress(
            progressConfig,
            (
                f"[{moduleName}] completion member finished | member={member} | "
                f"oof_acc={result['oofMetrics']['labelRecoveryAccuracy']:.4f}"
            ),
        )
    ensembleProbabilities = {
        splitName: np.mean(
            [
                result["datasetProbabilities"][splitName]
                for result in memberResults
            ],
            axis=0,
        )
        for splitName in ("train", "validation", "test")
    }
    ensembleOofProbabilities = np.mean(
        [result["oofProbabilities"] for result in memberResults],
        axis=0,
    )
    memberAudit = []
    for result in memberResults:
        record = {
            "family": result["family"],
            "config": result["config"],
            "oofMetrics": result["oofMetrics"],
        }
        if "folds" in result:
            record["folds"] = result["folds"]
        memberAudit.append(record)
    return {
        "datasetProbabilities": ensembleProbabilities,
        "artifactState": {
            "bottomModelState": {
                key: value.detach().cpu().clone()
                for key, value in bottomB.state_dict().items()
            },
            "foldCount": foldCount,
            "members": [result["artifactState"] for result in memberResults],
        },
        "members": memberAudit,
        "selectionPolicy": (
            f"explicit member ensemble; {foldCount}-fold stratified out-of-fold "
            f"MixMatch checkpoint selection by {selectionMetric}"
        ),
        "oofMetrics": calculateDetailedMetrics(
            auxiliaryLabelArray,
            ensembleOofProbabilities,
        ),
    }


def runConfiguredCompletionBackend(
    bottomB,
    bundle,
    labelInferenceConfig,
    configPath,
    device,
    seed,
    progressConfig=None,
    moduleName="PMC",
):
    backendName = str(labelInferenceConfig.get("completionBackend", "mixmatch")).lower()
    if backendName == "mixmatch":
        raise ValueError("The legacy MixMatch backend must use runPassiveModelCompletion")
    attackSeed = int(labelInferenceConfig.get("attackSeed", 100))
    torch.manual_seed(attackSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(attackSeed)
    auxiliaryLabels = loadAuxiliaryLabels(
        bundle.train,
        labelInferenceConfig,
        configPath,
        bundle.classCount,
    )
    evaluationBatchSize = int(labelInferenceConfig.get("evaluationBatchSize", 512))
    writeProgress(
        progressConfig,
        f"[{moduleName}] configured completion started | backend={backendName}",
    )
    if backendName in SUPPORTED_CLASSICAL_BACKENDS:
        backendResult = fitSingleClassicalBackend(
            backendName,
            labelInferenceConfig.get(backendName, {}),
            bottomB,
            bundle,
            auxiliaryLabels,
            evaluationBatchSize,
            bundle.classCount,
            attackSeed,
            device,
        )
    elif backendName == "cv_ensemble":
        backendResult = fitCrossValidatedEnsemble(
            bottomB,
            bundle,
            labelInferenceConfig,
            auxiliaryLabels,
            evaluationBatchSize,
            bundle.classCount,
            attackSeed,
            device,
            progressConfig,
            moduleName,
        )
    else:
        raise ValueError(f"Unsupported completion backend: {backendName}")
    trainPredictions = buildEvaluationRecord(
        bundle.train,
        backendResult["datasetProbabilities"]["train"],
    )
    validationPredictions = buildEvaluationRecord(
        bundle.validation,
        backendResult["datasetProbabilities"]["validation"],
    )
    testPredictions = buildPredictionRecord(
        bundle.test,
        backendResult["datasetProbabilities"]["test"],
    )
    privacyMetrics = {
        "labelRecoveryAccuracy": validationPredictions["metrics"][
            "labelRecoveryAccuracy"
        ],
        "labelRecoveryAuc": validationPredictions["metrics"]["labelRecoveryAuc"],
        "trainLabelRecoveryAccuracy": trainPredictions["metrics"][
            "labelRecoveryAccuracy"
        ],
        "trainLabelRecoveryAuc": trainPredictions["metrics"]["labelRecoveryAuc"],
        "auxiliaryLabelCount": len(auxiliaryLabels.ids),
        "evaluationScope": "validation",
        "externalScoringRequired": True,
    }
    classCounts = {
        str(label): int((auxiliaryLabels.labels == label).sum().item())
        for label in range(bundle.classCount)
    }
    auditRecords = {
        "attack": "pmc",
        "attackerParty": "B, the passive party without labels",
        "threatModel": (
            "honest-but-curious during VFL training; post-training model completion"
        ),
        "completionBackend": backendName,
        "completionMembers": backendResult["members"],
        "completionOofMetrics": backendResult["oofMetrics"],
        "selectionPolicy": backendResult["selectionPolicy"],
        "auxiliaryLabelsPath": auxiliaryLabels.sourcePath,
        "auxiliaryLabelIds": auxiliaryLabels.ids,
        "auxiliaryClassCounts": classCounts,
        "classCount": int(bundle.classCount),
        "auxiliaryLabelAccess": (
            "labels are read only from the explicitly configured CSV"
        ),
        "unlabeledSampleCount": len(bundle.train) - len(auxiliaryLabels.ids),
        "readsPartyAFeatures": False,
        "readsTopModel": False,
        "mutatesVflBottomB": False,
        "hiddenTrainingLabelsUsedForOptimization": False,
        "hiddenLabelsUsedForMetricsOnly": True,
        "trainingLabelsUsedForLocalEvaluationOnly": True,
        "publicTestLabelsRead": False,
        "testPredictionCount": len(testPredictions["ids"]),
        "submissionScope": "testOnly",
        "trainingEvaluationMetrics": trainPredictions["metrics"],
        "trainingEvaluationSampleCount": len(trainPredictions["ids"]),
        "validationRecoveryMetrics": validationPredictions["metrics"],
        "algorithm": (
            "configurable frozen-embedding completion backend"
            if backendName != "cv_ensemble"
            else "stratified cross-validated completion ensemble"
        ),
        "trainingSeed": int(seed),
        "attackSeed": attackSeed,
        "evaluationBatchSize": evaluationBatchSize,
        "modelCompletionUsesBottomBCopy": True,
    }
    artifact = CompletionBackendArtifact(
        backendName=backendName,
        backendState=backendResult["artifactState"],
    )
    writeProgress(
        progressConfig,
        (
            f"[{moduleName}] configured completion finished | "
            f"backend={backendName} | "
            f"validation_acc={privacyMetrics['labelRecoveryAccuracy']:.4f}"
        ),
    )
    return PmcResult(
        privacyMetrics=privacyMetrics,
        auditRecords=auditRecords,
        predictions=buildCompletionSubmissionPredictions(testPredictions),
        completionModel=artifact,
    )
