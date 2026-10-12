import copy
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from core.progress import createProgressBar, writeProgress
from metrics.privacy import buildPrivacyMetrics
from metrics.utility import calculateAccuracy, calculateAuc


@dataclass
class AuxiliaryLabelSet:
    ids: list
    labels: torch.Tensor
    positions: list
    sourcePath: str


@dataclass
class PmcResult:
    privacyMetrics: dict
    auditRecords: dict
    predictions: dict
    completionModel: nn.Module


class PartyBFeatureDataset(Dataset):
    """只暴露恶意 B 方可见的局部特征，不返回训练集真实标签。"""

    def __init__(self, trainDataset, positions):
        self.trainDataset = trainDataset
        self.positions = list(positions)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, position):
        return self.trainDataset.partyB[self.positions[position]]


class CompletionHead(nn.Module):
    """每层在线性映射前执行 BN 和激活。"""

    def __init__(self, embeddingDim, classCount, layerCount, activationName, useBatchNorm):
        super().__init__()
        self.useBatchNorm = bool(useBatchNorm)
        self.activationName = str(activationName).lower()
        self.batchNorms = nn.ModuleList(
            [nn.BatchNorm1d(embeddingDim) for _ in range(layerCount)]
        )
        self.layers = nn.ModuleList(
            [nn.Linear(embeddingDim, embeddingDim) for _ in range(layerCount - 1)]
            + [nn.Linear(embeddingDim, classCount)]
        )
        for layer in self.layers:
            nn.init.ones_(layer.weight)

    def applyActivation(self, values):
        if self.activationName == "relu":
            return torch.relu(values)
        if self.activationName == "sigmoid":
            return torch.sigmoid(values)
        if self.activationName == "none":
            return values
        raise ValueError(f"Unsupported PMC activation: {self.activationName}")

    def forward(self, embedding):
        values = embedding
        for batchNorm, layer in zip(self.batchNorms, self.layers):
            if self.useBatchNorm:
                values = batchNorm(values)
            values = self.applyActivation(values)
            values = layer(values)
        return values


class PassiveCompletionModel(nn.Module):
    def __init__(
        self,
        bottomModel,
        embeddingDim,
        classCount,
        layerCount,
        activationName,
        useBatchNorm,
    ):
        super().__init__()
        # PMC 必须在副本上微调，不能改变仍用于 VFL 评测的 B 方底模。
        self.bottomModel = copy.deepcopy(bottomModel)
        self.inferenceHead = CompletionHead(
            embeddingDim,
            classCount,
            layerCount,
            activationName,
            useBatchNorm,
        )

    def forward(self, partyB):
        return self.inferenceHead(self.bottomModel(partyB))


def resolveAuxiliaryLabelsPath(labelInferenceConfig, configPath):
    configuredPath = Path(str(labelInferenceConfig["auxiliaryLabelsPath"]))
    if configuredPath.is_absolute():
        return configuredPath.resolve()
    return (Path(configPath).resolve().parent / configuredPath).resolve()


def loadAuxiliaryLabels(
    trainDataset,
    labelInferenceConfig,
    configPath,
    classCount=2,
):
    """只从显式 CSV 读取辅助标签；训练集 labels 不参与 PMC 拟合。"""
    labelsPath = resolveAuxiliaryLabelsPath(labelInferenceConfig, configPath)
    if not labelsPath.is_file():
        raise ValueError(f"PMC auxiliary label file does not exist: {labelsPath}")
    records = []
    with labelsPath.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["ID", "Label"]:
            raise ValueError("PMC auxiliary label CSV must contain exactly: ID,Label")
        for row in reader:
            sampleId = str(row["ID"]).strip()
            try:
                label = int(row["Label"])
            except (TypeError, ValueError) as error:
                raise ValueError("PMC auxiliary labels must be integers") from error
            records.append((sampleId, label))
    if not records:
        raise ValueError("PMC auxiliary label file must not be empty")
    ids = [sampleId for sampleId, _ in records]
    if len(set(ids)) != len(ids):
        raise ValueError("PMC auxiliary label IDs must not contain duplicates")
    missingIds = [sampleId for sampleId in ids if sampleId not in trainDataset.idToPosition]
    if missingIds:
        raise ValueError(f"PMC auxiliary label file contains {len(missingIds)} unknown IDs")
    labels = [label for _, label in records]
    expectedLabels = set(range(int(classCount)))
    if any(label not in expectedLabels for label in labels):
        raise ValueError(
            "PMC auxiliary labels must be integers in "
            f"0..{int(classCount) - 1}"
        )
    if set(labels) != expectedLabels:
        raise ValueError(
            "PMC auxiliary labels must contain every configured class: "
            f"expected {sorted(expectedLabels)}, got {sorted(set(labels))}"
        )
    positions = [trainDataset.idToPosition[sampleId] for sampleId in ids]
    return AuxiliaryLabelSet(
        ids=ids,
        labels=torch.tensor(labels, dtype=torch.long),
        positions=positions,
        sourcePath=str(labelsPath),
    )


def buildPmcLoaders(trainDataset, auxiliaryLabels, batchSize, attackSeed):
    if batchSize > len(auxiliaryLabels.ids):
        raise ValueError("PMC completionBatchSize must not exceed auxiliary label count")
    auxiliaryFeatures = trainDataset.partyB[auxiliaryLabels.positions]
    labeledDataset = TensorDataset(auxiliaryFeatures, auxiliaryLabels.labels)
    auxiliarySet = set(auxiliaryLabels.ids)
    unlabeledPositions = [
        position
        for position, sampleId in enumerate(trainDataset.ids.tolist())
        if str(sampleId) not in auxiliarySet
    ]
    if len(unlabeledPositions) < batchSize:
        raise ValueError("PMC requires at least one full unlabeled batch")
    labeledGenerator = torch.Generator().manual_seed(int(attackSeed))
    unlabeledGenerator = torch.Generator().manual_seed(int(attackSeed) + 1)
    labeledLoader = DataLoader(
        labeledDataset,
        batch_size=batchSize,
        shuffle=True,
        drop_last=True,
        generator=labeledGenerator,
    )
    unlabeledLoader = DataLoader(
        PartyBFeatureDataset(trainDataset, unlabeledPositions),
        batch_size=batchSize,
        shuffle=True,
        drop_last=True,
        generator=unlabeledGenerator,
    )
    return labeledLoader, unlabeledLoader, len(unlabeledPositions)


def getNextBatch(batchIterator, loader):
    try:
        return next(batchIterator), batchIterator
    except StopIteration:
        refreshedIterator = iter(loader)
        return next(refreshedIterator), refreshedIterator


def sharpenProbabilities(probabilities, temperature):
    sharpened = probabilities.pow(1.0 / float(temperature))
    return sharpened / sharpened.sum(dim=1, keepdim=True)


def calculateRampWeight(progress, epochCount, lambdaU):
    if epochCount <= 0:
        return float(lambdaU)
    return float(lambdaU) * float(np.clip(progress / epochCount, 0.0, 1.0))


def calculateInterleaveOffsets(batchSize, unlabeledBatchCount):
    groups = [batchSize // (unlabeledBatchCount + 1)] * (unlabeledBatchCount + 1)
    for offset in range(batchSize - sum(groups)):
        groups[-offset - 1] += 1
    offsets = [0]
    for groupSize in groups:
        offsets.append(offsets[-1] + groupSize)
    return offsets


def interleaveBatches(batches, batchSize):
    unlabeledBatchCount = len(batches) - 1
    offsets = calculateInterleaveOffsets(batchSize, unlabeledBatchCount)
    splitBatches = [
        [values[offsets[index] : offsets[index + 1]] for index in range(unlabeledBatchCount + 1)]
        for values in batches
    ]
    for index in range(1, unlabeledBatchCount + 1):
        splitBatches[0][index], splitBatches[index][index] = (
            splitBatches[index][index],
            splitBatches[0][index],
        )
    return [torch.cat(values, dim=0) for values in splitBatches]


def updateEmaModel(model, emaModel, emaDecay, learningRate):
    with torch.no_grad():
        for parameter, emaParameter in zip(model.parameters(), emaModel.parameters()):
            emaParameter.mul_(emaDecay).add_(parameter, alpha=1.0 - emaDecay)
            parameter.mul_(1.0 - 0.02 * learningRate)
        for buffer, emaBuffer in zip(model.buffers(), emaModel.buffers()):
            emaBuffer.copy_(buffer)


def trainPmcEpoch(
    model,
    emaModel,
    optimizer,
    labeledLoader,
    unlabeledLoader,
    config,
    epoch,
    device,
    numpyGenerator,
    torchGenerator,
    classCount=2,
    progressConfig=None,
    moduleName="PMC",
):
    model.train()
    labeledIterator = iter(labeledLoader)
    unlabeledIterator = iter(unlabeledLoader)
    iterations = int(config["iterationsPerEpoch"])
    batchSize = int(config["completionBatchSize"])
    totalLoss = 0.0
    totalLabeledLoss = 0.0
    totalUnlabeledLoss = 0.0
    totalWeight = 0.0
    progressBar = createProgressBar(
        range(iterations),
        progressConfig,
        (
            f"{moduleName} completion "
            f"{epoch + 1}/{int(config['completionEpochs'])}"
        ),
        "iter",
    )
    for iteration in progressBar:
        (labeledFeatures, declaredLabels), labeledIterator = getNextBatch(
            labeledIterator, labeledLoader
        )
        unlabeledFeatures, unlabeledIterator = getNextBatch(unlabeledIterator, unlabeledLoader)
        labeledFeatures = labeledFeatures.to(device)
        declaredLabels = declaredLabels.to(device)
        unlabeledFeatures = unlabeledFeatures.to(device)
        labeledTargets = torch.nn.functional.one_hot(
            declaredLabels,
            num_classes=int(classCount),
        ).float()

        # 未标注样本不读取真实标签，而是用当前补全模型产生并锐化伪标签。
        with torch.no_grad():
            guessedProbabilities = torch.softmax(model(unlabeledFeatures), dim=1)
            unlabeledTargets = sharpenProbabilities(
                guessedProbabilities,
                float(config["temperature"]),
            ).detach()

        allInputs = torch.cat([labeledFeatures, unlabeledFeatures], dim=0)
        allTargets = torch.cat([labeledTargets, unlabeledTargets], dim=0)
        # 直接在 B 方局部特征和软标签上做 MixUp，不增加数据增强视图。
        mixRatio = float(numpyGenerator.beta(config["mixupAlpha"], config["mixupAlpha"]))
        mixRatio = max(mixRatio, 1.0 - mixRatio)
        permutation = torch.randperm(allInputs.shape[0], generator=torchGenerator).to(device)
        mixedInputs = mixRatio * allInputs + (1.0 - mixRatio) * allInputs[permutation]
        mixedTargets = mixRatio * allTargets + (1.0 - mixRatio) * allTargets[permutation]

        # 交错有标注和无标注片段，稳定小批量 BN 统计。
        interleavedInputs = interleaveBatches(
            list(torch.split(mixedInputs, batchSize)),
            batchSize,
        )
        interleavedLogits = [model(values) for values in interleavedInputs]
        restoredLogits = interleaveBatches(interleavedLogits, batchSize)
        labeledLogits = restoredLogits[0]
        unlabeledLogits = torch.cat(restoredLogits[1:], dim=0)
        labeledLoss = -torch.mean(
            torch.sum(torch.log_softmax(labeledLogits, dim=1) * mixedTargets[:batchSize], dim=1)
        )
        unlabeledProbabilities = torch.softmax(unlabeledLogits, dim=1)
        unlabeledLoss = torch.mean(
            (unlabeledProbabilities - mixedTargets[batchSize:]).pow(2)
        )
        # 无监督 MSE 权重在整个补全训练期间线性升至 lambdaU。
        progress = epoch + iteration / max(iterations, 1)
        unlabeledWeight = calculateRampWeight(
            progress,
            int(config["completionEpochs"]),
            float(config["lambdaU"]),
        )
        loss = labeledLoss + unlabeledWeight * unlabeledLoss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        updateEmaModel(
            model,
            emaModel,
            float(config["emaDecay"]),
            float(config["completionLearningRate"]),
        )
        totalLoss += float(loss.detach().item())
        totalLabeledLoss += float(labeledLoss.detach().item())
        totalUnlabeledLoss += float(unlabeledLoss.detach().item())
        totalWeight += unlabeledWeight
        completedIterations = iteration + 1
        progressBar.set_postfix(
            loss=f"{totalLoss / completedIterations:.4f}",
            refresh=False,
        )
    divisor = max(iterations, 1)
    return {
        "loss": totalLoss / divisor,
        "labeledLoss": totalLabeledLoss / divisor,
        "unlabeledLoss": totalUnlabeledLoss / divisor,
        "unlabeledWeight": totalWeight / divisor,
    }


def predictCompletionModel(model, dataset, batchSize, device):
    """只使用 B 方特征恢复标签，不读取待恢复数据集的真实标签。"""
    model.eval()
    ids = []
    probabilities = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batchSize, shuffle=False):
            logits = model(batch["partyB"].to(device))
            batchProbabilities = torch.softmax(logits, dim=1)
            ids.extend(batch["id"])
            probabilities.extend(batchProbabilities.cpu().tolist())
    probabilityArray = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilityArray.argmax(axis=1).astype(np.int64)
    if probabilityArray.shape[1] == 2:
        serializedProbabilities = probabilityArray[:, 1].tolist()
    else:
        serializedProbabilities = probabilityArray.tolist()
    return {
        "ids": ids,
        "probabilities": serializedProbabilities,
        "predictions": predictions.tolist(),
    }


def evaluateCompletionModel(model, dataset, batchSize, device):
    """仅供带标签的公开训练/验证数据诊断，隐藏测试集不能调用。"""
    if not dataset.hasLabels():
        raise ValueError("Completion evaluation requires a labeled dataset")
    predictions = predictCompletionModel(model, dataset, batchSize, device)
    labels = dataset.labels.cpu().tolist()
    predictions["labels"] = labels
    predictions["metrics"] = buildPrivacyMetrics(
        calculateAccuracy(labels, predictions["predictions"]),
        calculateAuc(labels, predictions["probabilities"]),
    )
    return predictions


def evaluateAuxiliaryLabels(model, trainDataset, auxiliaryLabels, device):
    model.eval()
    with torch.no_grad():
        features = trainDataset.partyB[auxiliaryLabels.positions].to(device)
        probabilities = torch.softmax(model(features), dim=1).cpu().numpy()
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    declaredLabels = auxiliaryLabels.labels.numpy()
    return calculateAccuracy(declaredLabels, predictions)


def buildCompletionSubmissionPredictions(testPredictions):
    """正式模型补全提交仅包含隐藏测试集，不混入训练集评估样本。"""
    return {
        "ids": list(testPredictions["ids"]),
        "predictions": list(testPredictions["predictions"]),
        "probabilities": list(testPredictions["probabilities"]),
    }


def runPassiveModelCompletion(
    bottomB,
    bundle,
    labelInferenceConfig,
    configPath,
    device,
    seed,
    progressConfig=None,
    moduleName="PMC",
):
    """训练完成后执行 PMC；只使用 B 方特征和外部 CSV 中声明的辅助标签。"""
    completionBackend = str(
        labelInferenceConfig.get("completionBackend", "mixmatch")
    ).lower()
    if completionBackend != "mixmatch":
        # 延迟导入避免 completion_backends 复用本模块训练原语时形成初始化环。
        from .completion_backends import runConfiguredCompletionBackend

        return runConfiguredCompletionBackend(
            bottomB,
            bundle,
            labelInferenceConfig,
            configPath,
            device,
            seed,
            progressConfig,
            moduleName,
        )
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
    batchSize = int(labelInferenceConfig["completionBatchSize"])
    labeledLoader, unlabeledLoader, unlabeledCount = buildPmcLoaders(
        bundle.train,
        auxiliaryLabels,
        batchSize,
        attackSeed,
    )
    completionModel = PassiveCompletionModel(
        bottomB,
        int(labelInferenceConfig.get("embeddingDim", 64)),
        bundle.classCount,
        int(labelInferenceConfig.get("headLayers", 1)),
        str(labelInferenceConfig.get("activation", "relu")),
        bool(labelInferenceConfig.get("useBatchNorm", True)),
    ).to(device)
    emaModel = copy.deepcopy(completionModel).to(device)
    # 补全阶段独立使用 Adam，不复用 VFL 的优化器状态。
    optimizer = torch.optim.Adam(
        completionModel.parameters(),
        lr=float(labelInferenceConfig["completionLearningRate"]),
    )
    numpyGenerator = np.random.RandomState(attackSeed)
    torchGenerator = torch.Generator().manual_seed(attackSeed + 2)
    history = []
    bestAuxiliaryAccuracy = float("-inf")
    bestState = copy.deepcopy(emaModel.state_dict())
    completionEpochs = int(labelInferenceConfig["completionEpochs"])
    writeProgress(
        progressConfig,
        (
            f"[{moduleName}] model completion started | epochs={completionEpochs} | "
            f"iterations_per_epoch={int(labelInferenceConfig['iterationsPerEpoch'])}"
        ),
    )
    for epoch in range(completionEpochs):
        epochMetrics = trainPmcEpoch(
            completionModel,
            emaModel,
            optimizer,
            labeledLoader,
            unlabeledLoader,
            labelInferenceConfig,
            epoch,
            device,
            numpyGenerator,
            torchGenerator,
            bundle.classCount,
            progressConfig,
            moduleName,
        )
        auxiliaryAccuracy = evaluateAuxiliaryLabels(
            emaModel,
            bundle.train,
            auxiliaryLabels,
            device,
        )
        epochMetrics.update({"epoch": epoch, "auxiliaryAccuracy": auxiliaryAccuracy})
        history.append(epochMetrics)
        # 只用显式辅助标签选择模型，避免读取隐藏训练标签。
        if auxiliaryAccuracy > bestAuxiliaryAccuracy:
            bestAuxiliaryAccuracy = auxiliaryAccuracy
            bestState = copy.deepcopy(emaModel.state_dict())
    emaModel.load_state_dict(bestState)
    writeProgress(
        progressConfig,
        (
            f"[{moduleName}] model completion finished | "
            f"best_auxiliary_acc={bestAuxiliaryAccuracy:.4f}"
        ),
    )
    trainPredictions = evaluateCompletionModel(
        emaModel,
        bundle.train,
        int(labelInferenceConfig.get("evaluationBatchSize", 512)),
        device,
    )
    validationPredictions = evaluateCompletionModel(
        emaModel,
        bundle.validation,
        int(labelInferenceConfig.get("evaluationBatchSize", 512)),
        device,
    )
    testPredictions = predictCompletionModel(
        emaModel,
        bundle.test,
        int(labelInferenceConfig.get("evaluationBatchSize", 512)),
        device,
    )
    privacyMetrics = {
        "labelRecoveryAccuracy": validationPredictions["metrics"][
            "labelRecoveryAccuracy"
        ],
        "labelRecoveryAuc": validationPredictions["metrics"]["labelRecoveryAuc"],
        "trainLabelRecoveryAccuracy": trainPredictions["metrics"]["labelRecoveryAccuracy"],
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
        "threatModel": "honest-but-curious during VFL training; post-training model completion",
        "auxiliaryLabelsPath": auxiliaryLabels.sourcePath,
        "auxiliaryLabelIds": auxiliaryLabels.ids,
        "auxiliaryClassCounts": classCounts,
        "classCount": int(bundle.classCount),
        "auxiliaryLabelAccess": "labels are read only from the explicitly configured CSV",
        "unlabeledSampleCount": unlabeledCount,
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
        "completionBackend": "mixmatch",
        "algorithm": "customized MixMatch without data augmentation",
        "selectionPolicy": "best EMA model selected by declared auxiliary-label accuracy",
        "sourceSelectionAdaptation": "hidden training labels are not used for checkpoint selection",
        "optimizer": "Adam",
        "trainingSeed": int(seed),
        "attackSeed": attackSeed,
        "completionEpochs": int(labelInferenceConfig["completionEpochs"]),
        "iterationsPerEpoch": int(labelInferenceConfig["iterationsPerEpoch"]),
        "completionBatchSize": batchSize,
        "completionLearningRate": float(labelInferenceConfig["completionLearningRate"]),
        "temperature": float(labelInferenceConfig["temperature"]),
        "mixupAlpha": float(labelInferenceConfig["mixupAlpha"]),
        "lambdaU": float(labelInferenceConfig["lambdaU"]),
        "emaDecay": float(labelInferenceConfig["emaDecay"]),
        "headLayers": int(labelInferenceConfig.get("headLayers", 1)),
        "activation": str(labelInferenceConfig.get("activation", "relu")),
        "useBatchNorm": bool(labelInferenceConfig.get("useBatchNorm", True)),
        "epochs": history,
    }
    return PmcResult(
        privacyMetrics=privacyMetrics,
        auditRecords=auditRecords,
        predictions=buildCompletionSubmissionPredictions(testPredictions),
        completionModel=emaModel,
    )


def saveCompletionCheckpoint(path, completionModel, auditRecords):
    checkpointPath = Path(path)
    checkpointPath.parent.mkdir(parents=True, exist_ok=True)
    with checkpointPath.open("wb") as handle:
        torch.save(
            {
                "model": completionModel.state_dict(),
                "metadata": auditRecords,
            },
            handle,
        )
    return checkpointPath
