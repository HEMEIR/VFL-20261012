import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from core.checkpoint import captureModelState, restoreModelState
from core.interfaces import TrainingHistory
from core.progress import createProgressBar, writeProgress
from metrics.robustness import buildRobustnessMetrics
from .triggers import validateCleanBackdoorTestView
from metrics.utility import calculateAccuracy, calculateAuc


@dataclass
class VillainTrainingResult:
    history: TrainingHistory
    targetLabel: int
    triggerDimensions: list
    triggerVector: list
    backdoorPredictions: dict
    validationRobustness: dict
    auditRecords: dict


def resolveKnownTargetIds(trainDataset, backdoorConfig):
    """只解析配置显式提供的目标 ID，不扫描全训练标签筛选目标类。"""
    knownTargetIds = [
        str(sampleId) for sampleId in backdoorConfig.get("knownTargetIds", [])
    ]
    if not knownTargetIds:
        raise ValueError("VILLAIN requires configured knownTargetIds")
    if len(set(knownTargetIds)) != len(knownTargetIds):
        raise ValueError("VILLAIN knownTargetIds must not contain duplicates")
    missingIds = [
        sampleId
        for sampleId in knownTargetIds
        if sampleId not in trainDataset.idToPosition
    ]
    if missingIds:
        raise ValueError(f"VILLAIN known target IDs do not exist: {missingIds}")

    targetLabel = int(backdoorConfig["targetLabel"])
    configuredPositions = [
        trainDataset.idToPosition[sampleId] for sampleId in knownTargetIds
    ]
    configuredLabels = trainDataset.labels[configuredPositions]
    if not bool(torch.all(configuredLabels == targetLabel)):
        raise ValueError(
            "VILLAIN configured knownTargetIds must all have the configured target label"
        )
    return knownTargetIds


def selectPoisonIds(knownTargetIds, backdoorConfig, attackSeed):
    explicitPoisonIds = backdoorConfig.get("poisonIds")
    if explicitPoisonIds is not None:
        poisonIds = [str(sampleId) for sampleId in explicitPoisonIds]
        if not poisonIds:
            raise ValueError("VILLAIN poisonIds must not be empty")
        if len(set(poisonIds)) != len(poisonIds):
            raise ValueError("VILLAIN poisonIds must not contain duplicates")
        unknownIds = sorted(set(poisonIds).difference(knownTargetIds))
        if unknownIds:
            raise ValueError(
                "VILLAIN poisonIds must be a subset of knownTargetIds: "
                f"{unknownIds}"
            )
        return poisonIds

    poisonRate = float(backdoorConfig["poisonRate"])
    poisonCount = max(1, round(poisonRate * len(knownTargetIds)))
    poisonCount = min(poisonCount, len(knownTargetIds))
    generator = random.Random(int(attackSeed))
    return sorted(generator.sample(list(knownTargetIds), poisonCount))


def buildVillainTrigger(columnStd, triggerDimensionCount, beta):
    """按 FILTER 的 top-m 方差和 ++-- 符号规则构造 embedding trigger。"""
    stdValues = torch.as_tensor(columnStd, dtype=torch.float32).flatten()
    if stdValues.numel() == 0:
        raise ValueError("VILLAIN cannot build a trigger from an empty embedding")
    dimensionCount = int(triggerDimensionCount)
    if not 0 < dimensionCount <= int(stdValues.numel()):
        raise ValueError(
            "VILLAIN triggerDimensionCount must be within the embedding dimension"
        )
    selectedDimensions = torch.topk(
        stdValues,
        k=dimensionCount,
        largest=True,
        sorted=True,
    ).indices
    selectedDimensions = sorted(int(value) for value in selectedDimensions.tolist())
    delta = float(stdValues[selectedDimensions].mean().item())
    signs = torch.tensor(
        [1.0 if index % 4 < 2 else -1.0 for index in range(stdValues.numel())],
        dtype=torch.float32,
    )
    mask = torch.zeros_like(stdValues)
    mask[selectedDimensions] = 1.0
    trigger = mask * (float(beta) * delta * signs)
    return trigger, selectedDimensions, delta


def collectKnownTargetEmbeddingStd(trainer, trainDataset, knownTargetIds):
    positions = [trainDataset.idToPosition[sampleId] for sampleId in knownTargetIds]
    loader = DataLoader(
        Subset(trainDataset, positions),
        batch_size=int(trainer.config["batchSize"]),
        shuffle=False,
        num_workers=0,
    )
    previousMode = trainer.models.bottomB.training
    trainer.models.bottomB.train()
    batchStandardDeviations = []
    with torch.no_grad():
        for originalBatch in loader:
            inputB = originalBatch["partyB"].to(trainer.device)
            embeddings = trainer.models.bottomB(inputB)
            batchStandardDeviations.append(
                embeddings.detach().float().std(dim=0, unbiased=False).cpu()
            )
    trainer.models.bottomB.train(previousMode)
    if not batchStandardDeviations:
        raise ValueError("VILLAIN known target set produced no embeddings")
    return torch.stack(batchStandardDeviations, dim=0).mean(dim=0)


def applyVillainTrigger(embeddingB, triggerVector, rowMask=None):
    trigger = torch.as_tensor(
        triggerVector,
        dtype=embeddingB.dtype,
        device=embeddingB.device,
    )
    if trigger.ndim != 1 or trigger.numel() != embeddingB.shape[1]:
        raise ValueError("VILLAIN trigger dimension must match party-B embedding")
    if rowMask is None:
        return embeddingB + trigger.unsqueeze(0)
    mask = torch.as_tensor(rowMask, dtype=torch.bool, device=embeddingB.device)
    if mask.ndim != 1 or mask.numel() != embeddingB.shape[0]:
        raise ValueError("VILLAIN row mask must match the batch size")
    return embeddingB + mask.to(embeddingB.dtype).unsqueeze(1) * trigger.unsqueeze(0)


def buildTrainingTriggerVariants(
    triggerVector,
    rowCount,
    triggerKeepRate,
    triggerShiftMin,
    triggerShiftMax,
    generator,
    dtype,
    device,
):
    """为每个投毒样本生成仅在训练阶段使用的 Dropout/Shifting 触发器。"""
    baseTrigger = torch.as_tensor(triggerVector, dtype=torch.float32).flatten().cpu()
    sampleCount = int(rowCount)
    if sampleCount < 0:
        raise ValueError("VILLAIN trigger variant row count must not be negative")
    if sampleCount == 0:
        return torch.empty(
            (0, int(baseTrigger.numel())),
            dtype=dtype,
            device=device,
        )

    keepRate = float(triggerKeepRate)
    shiftMin = float(triggerShiftMin)
    shiftMax = float(triggerShiftMax)
    activeDimensions = baseTrigger.ne(0.0).unsqueeze(0)
    keepMask = torch.rand(
        (sampleCount, int(baseTrigger.numel())),
        generator=generator,
        dtype=torch.float32,
    ).lt(keepRate)
    keepMask = torch.logical_and(keepMask, activeDimensions)
    shifts = torch.empty((sampleCount, 1), dtype=torch.float32).uniform_(
        shiftMin,
        shiftMax,
        generator=generator,
    )
    variants = baseTrigger.unsqueeze(0) * keepMask.to(torch.float32) * shifts
    return variants.to(device=device, dtype=dtype)


def setOptimizerLearningRate(optimizer, learningRate):
    resolvedLearningRate = float(learningRate)
    for parameterGroup in optimizer.param_groups:
        parameterGroup["lr"] = resolvedLearningRate
    return resolvedLearningRate


def trainVillainBatch(
    trainer,
    batch,
    epoch,
    poisonIds,
    triggerVector,
    attackEnabled,
    augmentationConfig,
    augmentationGenerator,
    batchIndex=0,
):
    def injectVillainEmbedding(embeddingB):
        poisonMask = torch.tensor(
            [str(sampleId) in poisonIds for sampleId in batch["id"]],
            dtype=torch.bool,
            device=trainer.device,
        )
        poisonedRowCount = int(poisonMask.sum().item())
        if attackEnabled:
            triggerVariants = buildTrainingTriggerVariants(
                triggerVector,
                poisonedRowCount,
                augmentationConfig["triggerKeepRate"],
                augmentationConfig["triggerShiftMin"],
                augmentationConfig["triggerShiftMax"],
                augmentationGenerator,
                embeddingB.dtype,
                embeddingB.device,
            )
            if poisonedRowCount > 0:
                embeddingB = embeddingB.clone()
                embeddingB[poisonMask] = embeddingB[poisonMask] + triggerVariants
        return embeddingB, {"injectedSamples": poisonedRowCount}

    result = trainer.trainBatch(
        batch,
        epoch=epoch,
        batchIndex=batchIndex,
        trainingContext={
            "backdoor": "villain",
            "poisoningActive": bool(attackEnabled),
        },
        partyBEmbeddingTransform=injectVillainEmbedding,
    )
    return result.loss, int(result.metadata["injectedSamples"])


def trainVillainEpoch(
    trainer,
    trainDataset,
    epoch,
    poisonIds,
    triggerVector,
    startEpoch,
    augmentationConfig,
    augmentationGenerator,
):
    trainer.models.bottomA.train()
    trainer.models.bottomB.train()
    trainer.models.topModel.train()
    loader = trainer.buildTrainingLoader(trainDataset)
    totalLoss = 0.0
    totalSamples = 0
    injectedSamples = 0
    attackEnabled = triggerVector is not None and int(epoch) >= int(startEpoch)
    progressBar = createProgressBar(
        loader,
        trainer.config,
        f"VILLAIN Epoch {epoch + 1}/{int(trainer.config['epochs'])}",
        "batch",
    )
    poisonSet = set(poisonIds)
    for batchIndex, originalBatch in enumerate(progressBar):
        batch = trainer.moveBatchToDevice(originalBatch)
        loss, batchInjected = trainVillainBatch(
            trainer,
            batch,
            epoch,
            poisonSet,
            triggerVector,
            attackEnabled,
            augmentationConfig,
            augmentationGenerator,
            batchIndex=batchIndex,
        )
        batchSize = int(batch["label"].shape[0])
        totalLoss += loss * batchSize
        totalSamples += batchSize
        injectedSamples += batchInjected if attackEnabled else 0
        progressBar.set_postfix(
            loss=f"{totalLoss / max(totalSamples, 1):.4f}",
            refresh=False,
        )
    return totalLoss / max(totalSamples, 1), injectedSamples


def predictWithVillainTrigger(
    trainer,
    dataset,
    triggerVector,
    triggeredIds=None,
):
    triggeredIdSet = None if triggeredIds is None else set(str(value) for value in triggeredIds)

    def injectVillainEmbedding(embeddingB, sampleIds):
        if triggeredIdSet is None:
            return applyVillainTrigger(embeddingB, triggerVector)
        rowMask = [sampleId in triggeredIdSet for sampleId in sampleIds]
        return applyVillainTrigger(embeddingB, triggerVector, rowMask)

    return trainer.predict(
        dataset,
        partyBEmbeddingTransform=injectVillainEmbedding,
        inferenceContext={"backdoor": "villain", "triggered": True},
    )


def calculateVillainAttackSuccess(
    trainer,
    validationDataset,
    targetLabel,
    triggerVector,
):
    triggeredResults = predictWithVillainTrigger(
        trainer,
        validationDataset,
        triggerVector,
    )
    labels = np.asarray(validationDataset.labels.cpu().tolist())
    predictions = np.asarray(triggeredResults["predictions"])
    attackMask = labels != int(targetLabel)
    attackedSamples = int(attackMask.sum())
    if attackedSamples == 0:
        raise ValueError("VILLAIN ASR requires at least one non-target sample")
    asr = float((predictions[attackMask] == int(targetLabel)).mean())
    triggeredAccuracy = calculateAccuracy(labels, predictions)
    return buildRobustnessMetrics(asr, attackedSamples, triggeredAccuracy)


def runVillainTraining(trainer, bundle, backdoorConfig, seed):
    trainer.moveModelsToDevice()
    trainDataset = bundle.train
    knownTargetIds = resolveKnownTargetIds(trainDataset, backdoorConfig)
    attackSeed = int(backdoorConfig.get("attackSeed", seed))
    poisonIds = selectPoisonIds(knownTargetIds, backdoorConfig, attackSeed)
    targetLabel = int(backdoorConfig["targetLabel"])
    startEpoch = int(backdoorConfig.get("startEpoch", 0))
    minimumAccuracy = float(backdoorConfig.get("minimumMainTaskAccuracy", 0.0))
    baseLearningRate = float(trainer.config["learningRate"])
    augmentationConfig = {
        "triggerKeepRate": float(backdoorConfig.get("triggerKeepRate", 1.0)),
        "triggerShiftMin": float(backdoorConfig.get("triggerShiftMin", 1.0)),
        "triggerShiftMax": float(backdoorConfig.get("triggerShiftMax", 1.0)),
    }
    attackerPreAttackLearningRate = float(
        backdoorConfig.get("attackerPreAttackLearningRate", baseLearningRate)
    )
    attackerAttackLearningRate = float(
        backdoorConfig.get("attackerAttackLearningRate", baseLearningRate)
    )
    augmentationGenerator = torch.Generator(device="cpu").manual_seed(
        attackSeed + 2000003
    )

    history = TrainingHistory()
    bestState = captureModelState(trainer.models)
    bestTrigger = None
    bestDimensions = []
    bestDelta = None
    bestScore = float("-inf")
    triggerVector = None
    triggerDimensions = []
    triggerDelta = None
    patience = int(trainer.config.get("earlyStoppingPatience", 0))
    epochsWithoutImprovement = 0
    auditRecords = {
        "attack": "villain",
        "targetLabel": targetLabel,
        "targetLabelAccess": "configured knownTargetIds only; no full-label scan",
        "knownTargetIds": knownTargetIds,
        "poisonIds": poisonIds,
        "poisonRate": float(backdoorConfig["poisonRate"]),
        "poisonRateBasis": "configured knownTargetIds",
        "startEpoch": startEpoch,
        "triggerDimensionCount": int(backdoorConfig["triggerDimensionCount"]),
        "beta": float(backdoorConfig["beta"]),
        "signPattern": "++-- repeated across embedding coordinates",
        "triggerConstruction": "top-m mean batch embedding standard deviation",
        "embeddingParty": "B",
        "embeddingOperation": "additive",
        "backdoorDropout": {
            "trainingOnly": True,
            "triggerKeepRate": augmentationConfig["triggerKeepRate"],
            "triggerDropProbability": 1.0
            - augmentationConfig["triggerKeepRate"],
            "dropoutRatioInterpretation": (
                "0.75 is treated as retaining 75% of active trigger dimensions"
            ),
        },
        "triggerShifting": {
            "trainingOnly": True,
            "minimumScale": augmentationConfig["triggerShiftMin"],
            "maximumScale": augmentationConfig["triggerShiftMax"],
            "inferenceScale": 1.0,
        },
        "attackerLearningRateAdjustment": {
            "optimizer": "party-B bottom model only",
            "preAttackLearningRate": attackerPreAttackLearningRate,
            "attackLearningRate": attackerAttackLearningRate,
            "partyABottomLearningRate": baseLearningRate,
            "topModelLearningRate": baseLearningRate,
        },
        "labelModified": False,
        "partyAFeatureModified": False,
        "checkpointMinimumMainTaskAccuracy": minimumAccuracy,
        "checkpointSelection": (
            "highest (validation accuracy + ASR) / 2 among post-trigger checkpoints "
            "meeting minimumMainTaskAccuracy"
        ),
        "epochs": [],
    }

    totalEpochs = int(trainer.config["epochs"])
    for epoch in range(totalEpochs):
        epochStartTime = time.perf_counter()
        attackerLearningRate = setOptimizerLearningRate(
            trainer.optimizerB,
            attackerPreAttackLearningRate
            if epoch < startEpoch
            else attackerAttackLearningRate,
        )
        if epoch == startEpoch:
            columnStd = collectKnownTargetEmbeddingStd(
                trainer,
                trainDataset,
                knownTargetIds,
            )
            triggerVector, triggerDimensions, triggerDelta = buildVillainTrigger(
                columnStd,
                int(backdoorConfig["triggerDimensionCount"]),
                float(backdoorConfig["beta"]),
            )
            writeProgress(
                trainer.config,
                (
                    f"[VILLAIN] Epoch {epoch + 1}: embedding trigger built | "
                    f"dimensions={triggerDimensions} | delta={triggerDelta:.6f}"
                ),
            )

        trainingLoss, injectedSamples = trainVillainEpoch(
            trainer,
            trainDataset,
            epoch,
            poisonIds,
            triggerVector,
            startEpoch,
            augmentationConfig,
            augmentationGenerator,
        )
        validation = trainer.evaluate(
            bundle.validation,
            inferenceContext={"backdoor": "villain", "triggered": False},
        )
        if triggerVector is None:
            robustness = None
            selectionScore = float("-inf")
            checkpointEligible = False
        else:
            robustness = calculateVillainAttackSuccess(
                trainer,
                bundle.validation,
                targetLabel,
                triggerVector,
            )
            checkpointEligible = (
                float(validation["metrics"]["ba"]) >= minimumAccuracy
            )
            selectionScore = (
                float(validation["metrics"]["ba"]) + float(robustness["asr"])
            ) / 2.0

        epochRecord = {
            "epoch": epoch,
            "trainingLoss": trainingLoss,
            "validationBa": validation["metrics"]["ba"],
            "validationAuc": validation["metrics"]["auc"],
            "validationAsr": None if robustness is None else robustness["asr"],
            "selectionScore": None if robustness is None else selectionScore,
            "checkpointEligible": checkpointEligible,
            "injectedSamples": injectedSamples,
            "attackerLearningRate": attackerLearningRate,
        }
        history.epochs.append(epochRecord)
        auditRecords["epochs"].append(dict(epochRecord))
        if checkpointEligible and selectionScore > bestScore:
            bestScore = selectionScore
            history.bestEpoch = epoch
            history.bestValidationAccuracy = validation["metrics"]["ba"]
            history.bestValidationAuc = validation["metrics"]["auc"]
            history.bestSelectionScore = selectionScore
            bestState = captureModelState(trainer.models)
            bestTrigger = triggerVector.detach().cpu().clone()
            bestDimensions = list(triggerDimensions)
            bestDelta = triggerDelta
            epochsWithoutImprovement = 0
        elif triggerVector is not None:
            epochsWithoutImprovement += 1

        elapsedSeconds = time.perf_counter() - epochStartTime
        asrText = "pending" if robustness is None else f"{robustness['asr']:.4f}"
        writeProgress(
            trainer.config,
            (
                f"Epoch {epoch + 1}/{totalEpochs} | loss={trainingLoss:.4f} | "
                f"val_acc={validation['metrics']['ba']:.4f} | ASR={asrText} | "
                f"eligible={checkpointEligible} | time={elapsedSeconds:.1f}s"
            ),
        )
        if (
            patience > 0
            and triggerVector is not None
            and epochsWithoutImprovement >= patience
        ):
            break

    if history.bestEpoch < 0 or bestTrigger is None:
        raise RuntimeError(
            "VILLAIN did not produce a checkpoint meeting minimumMainTaskAccuracy"
        )
    restoreModelState(trainer.models, bestState)
    validationRobustness = calculateVillainAttackSuccess(
        trainer,
        bundle.validation,
        targetLabel,
        bestTrigger,
    )
    backdoorTest = bundle.backdoorTests.get("villain")
    if backdoorTest is None:
        backdoorTest = bundle.test
    else:
        validateCleanBackdoorTestView(backdoorTest, bundle.test, "villain")
    triggeredTestIds = [str(sampleId) for sampleId in backdoorTest.ids.tolist()]
    backdoorPredictions = predictWithVillainTrigger(
        trainer,
        backdoorTest,
        bestTrigger,
    )
    auditRecords.update(
        {
            "triggerDimensions": bestDimensions,
            "triggerVector": bestTrigger.tolist(),
            "triggerDelta": bestDelta,
            "testTriggerRate": 1.0,
            "triggeredTestIds": triggeredTestIds,
            "testEvaluationProtocol": (
                "complete attack-specific test view; every party-B embedding "
                "receives the fixed trigger and the organizer applies the private "
                "non-target mask"
            ),
        }
    )
    return VillainTrainingResult(
        history=history,
        targetLabel=targetLabel,
        triggerDimensions=bestDimensions,
        triggerVector=bestTrigger.tolist(),
        backdoorPredictions=backdoorPredictions,
        validationRobustness=validationRobustness,
        auditRecords=auditRecords,
    )
