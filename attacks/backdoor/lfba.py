import random
import time
from dataclasses import dataclass

import numpy as np
import torch

from core.checkpoint import captureModelState, restoreModelState
from core.interfaces import TrainingHistory
from core.progress import createProgressBar, writeProgress
from metrics.robustness import buildRobustnessMetrics

from .triggers import (
    addTriggerToPartyB,
    buildTriggeredDataset,
    selectTriggerDimensions,
    validateCleanBackdoorTestView,
)


@dataclass
class LfbaTrainingResult:
    history: TrainingHistory
    targetLabel: int
    triggerDimensions: list
    triggeredTestDataset: object
    validationRobustness: dict
    auditRecords: dict


def resolveAnchorId(anchorSetting, orderedIds):
    """同时支持字符串样本 ID 和零基位置索引。"""
    idList = [str(sampleId) for sampleId in orderedIds]
    directId = str(anchorSetting)
    if directId in idList:
        return directId
    try:
        anchorPosition = int(anchorSetting)
    except (TypeError, ValueError):
        anchorPosition = -1
    if 0 <= anchorPosition < len(idList):
        return idList[anchorPosition]
    raise ValueError(
        f"LFBA anchorId {anchorSetting!r} is neither a sample ID nor a valid zero-based position"
    )


def collectCutGradients(gradientTable, sampleIds, passiveGradients):
    """收集恶意被动方在当前 epoch 收到的 cut-layer 梯度。"""
    if len(sampleIds) != int(passiveGradients.shape[0]):
        raise ValueError("Sample IDs and passive gradients must have the same batch size")
    for position, sampleId in enumerate(sampleIds):
        normalizedId = str(sampleId)
        if normalizedId in gradientTable:
            raise ValueError(f"Duplicate gradient collected for sample {normalizedId}")
        gradientTable[normalizedId] = passiveGradients[position].detach().cpu().clone()


def buildPoisonSet(anchorId, orderedIds, gradientTable, poisonCount):
    """按内积/L1 范数相似度构造固定毒化集。"""
    if poisonCount <= 0 or poisonCount >= len(orderedIds):
        raise ValueError("LFBA poison count must be positive and smaller than the training set")
    missingIds = [str(sampleId) for sampleId in orderedIds if str(sampleId) not in gradientTable]
    if missingIds:
        raise ValueError(f"Missing cut gradients for {len(missingIds)} training samples")
    orderedGradients = torch.stack([gradientTable[str(sampleId)] for sampleId in orderedIds])
    anchorGradient = gradientTable[str(anchorId)]
    anchorNorm = torch.norm(anchorGradient, p=1)
    gradientNorms = torch.norm(orderedGradients, p=1, dim=1)
    if float(anchorNorm) == 0.0 or torch.any(gradientNorms == 0):
        raise ValueError("LFBA cannot calculate L1 similarity from zero gradients")
    similarities = (orderedGradients @ anchorGradient) / (gradientNorms * anchorNorm)
    selectedPositions = torch.topk(similarities, k=poisonCount, largest=True, sorted=True).indices
    return [str(orderedIds[int(position)]) for position in selectedPositions.tolist()]


def selectHardSamples(poisonIds, gradientTable, selectedCount):
    """从固定毒化集中选择梯度 L2 模长最大的目标样本。"""
    if selectedCount <= 0 or selectedCount > len(poisonIds):
        raise ValueError("LFBA selected sample count must be within the poison set")
    poisonGradients = torch.stack([gradientTable[str(sampleId)] for sampleId in poisonIds])
    gradientNorms = torch.norm(poisonGradients, p=2, dim=1)
    selectedPositions = torch.topk(
        gradientNorms,
        k=selectedCount,
        largest=True,
        sorted=True,
    ).indices
    selectedIds = [str(poisonIds[int(position)]) for position in selectedPositions.tolist()]
    selectedNorms = [float(gradientNorms[int(position)]) for position in selectedPositions.tolist()]
    return selectedIds, selectedNorms


def buildReplacementMapping(orderedIds, poisonIds, targetIds, randomGenerator):
    """从毒化集外抽取源样本，形成 source B → target B 的覆盖映射。"""
    poisonSet = set(str(sampleId) for sampleId in poisonIds)
    donorCandidates = [str(sampleId) for sampleId in orderedIds if str(sampleId) not in poisonSet]
    if len(donorCandidates) < len(targetIds):
        raise ValueError("Not enough samples outside the poison set for LFBA switching")
    donorIds = randomGenerator.sample(donorCandidates, len(targetIds))
    return [
        {"sourceId": sourceId, "targetId": str(targetId)}
        for sourceId, targetId in zip(donorIds, targetIds)
    ]


def injectPoisonFeatures(
    trainDataset,
    cleanPartyB,
    replacementMapping,
    triggerType,
    triggerDimensions,
    triggerFeatureValue,
):
    """恢复干净训练视图后，仅覆盖被选中目标样本的 B 方特征。"""
    trainDataset.partyB = cleanPartyB.clone()
    if not replacementMapping:
        return
    sourcePositions = [trainDataset.idToPosition[item["sourceId"]] for item in replacementMapping]
    targetPositions = [trainDataset.idToPosition[item["targetId"]] for item in replacementMapping]
    sourcePartyB = cleanPartyB[sourcePositions]
    triggeredSources = addTriggerToPartyB(
        sourcePartyB,
        triggerType,
        triggerDimensions,
        triggerFeatureValue,
    )
    trainDataset.partyB[targetPositions] = triggeredSources


def trainLfbaBatch(trainer, batch, epoch, batchIndex, poisonedIds):
    """执行一个 LFBA 训练 batch，并显式返回恶意 B 方收到的 cut 梯度。"""
    result = trainer.trainBatch(
        batch,
        epoch=epoch,
        batchIndex=batchIndex,
        trainingContext={
            "backdoor": "lfba",
            "poisoningActive": bool(poisonedIds),
        },
        capturePassiveCutGradient=True,
    )
    return result.loss, result.passiveCutGradient


def trainLfbaEpoch(trainer, trainDataset, epoch, poisonedIds):
    """训练一轮并按样本 ID 收集完整的 B 方梯度表。"""
    trainer.models.bottomA.train()
    trainer.models.bottomB.train()
    trainer.models.topModel.train()
    loader = trainer.buildTrainingLoader(trainDataset)
    gradientTable = {}
    totalLoss = 0.0
    totalSamples = 0
    totalEpochs = int(trainer.config["epochs"])
    progressBar = createProgressBar(
        loader,
        trainer.config,
        f"LFBA Epoch {epoch + 1}/{totalEpochs}",
        "batch",
    )
    for batchIndex, originalBatch in enumerate(progressBar):
        batch = trainer.moveBatchToDevice(originalBatch)
        loss, passiveGradients = trainLfbaBatch(
            trainer,
            batch,
            epoch,
            batchIndex,
            poisonedIds,
        )
        collectCutGradients(gradientTable, batch["id"], passiveGradients)
        batchSize = int(batch["label"].shape[0])
        totalLoss += float(loss) * batchSize
        totalSamples += batchSize
        progressBar.set_postfix(
            loss=f"{totalLoss / max(totalSamples, 1):.4f}",
            refresh=False,
        )
    return totalLoss / max(totalSamples, 1), gradientTable


def calculateAttackSuccess(trainer, triggeredDataset, targetLabel):
    triggeredResults = trainer.evaluate(
        triggeredDataset,
        inferenceContext={"backdoor": "lfba", "triggered": True},
    )
    labels = np.asarray(triggeredResults["labels"])
    predictions = np.asarray(triggeredResults["predictions"])
    attackMask = labels != int(targetLabel)
    attackedSamples = int(attackMask.sum())
    if attackedSamples == 0:
        raise ValueError("ASR requires at least one non-target sample")
    asr = float((predictions[attackMask] == int(targetLabel)).mean())
    return buildRobustnessMetrics(
        asr,
        attackedSamples,
        triggeredResults["metrics"]["ba"],
    )


def adjustLfbaLearningRate(trainer, epoch):
    learningRate = float(trainer.config["learningRate"]) * (0.1 ** ((epoch + 1) // 20))
    for optimizer in (trainer.optimizerA, trainer.optimizerB, trainer.optimizerTop):
        for parameterGroup in optimizer.param_groups:
            parameterGroup["lr"] = learningRate
    return learningRate


def isLfbaCheckpointEligible(epoch):
    """只允许已经执行特征投毒的 epoch 参与最佳模型选择。"""
    return int(epoch) >= 1


def runLfbaTraining(trainer, bundle, backdoorConfig, seed):
    """显式执行 LFBA，不通过通用攻击回调装配。"""
    trainer.moveModelsToDevice()
    trainDataset = bundle.train
    orderedIds = [str(sampleId) for sampleId in trainDataset.ids.tolist()]
    anchorId = resolveAnchorId(backdoorConfig["anchorId"], orderedIds)
    poisonCount = int(float(backdoorConfig["poisonRate"]) * len(trainDataset))
    selectedCount = int(poisonCount * float(backdoorConfig["switchRate"]))
    if selectedCount <= 0:
        raise ValueError("LFBA poisonRate and switchRate select zero samples")
    triggerDimensions = selectTriggerDimensions(
        str(backdoorConfig["lfbaTriggerType"]),
        trainDataset.partyB,
        int(backdoorConfig["triggerDimensions"]),
        seed,
        backdoorConfig.get("triggerDimensionIndices"),
    )
    triggerFeatureValue = float(backdoorConfig.get("triggerFeatureValue", 1.0))
    cleanPartyB = trainDataset.partyB.clone()
    triggeredValidation = buildTriggeredDataset(
        bundle.validation,
        str(backdoorConfig["lfbaTriggerType"]),
        triggerDimensions,
        triggerFeatureValue,
    )
    # 新数据包保留完整、干净、无标签的 LFBA 测试基准视图。正式配置
    # 在运行时向每一行统一注入触发器；合成测试可直接回退到 clean test。
    lfbaTestBase = bundle.backdoorTests.get("lfba", bundle.test)
    if "lfba" in bundle.backdoorTests:
        validateCleanBackdoorTestView(lfbaTestBase, bundle.test, "lfba")
    triggeredTest = buildTriggeredDataset(
        lfbaTestBase,
        str(backdoorConfig["lfbaTriggerType"]),
        triggerDimensions,
        triggerFeatureValue,
    )

    history = TrainingHistory()
    bestState = captureModelState(trainer.models)
    bestScore = float("-inf")
    patience = int(trainer.config.get("earlyStoppingPatience", 0))
    epochsWithoutImprovement = 0
    previousGradients = None
    poisonIds = None
    # 目标标签仅由组织方用于一致的验证/ASR 计算，不传给毒化集构造或特征注入函数。
    anchorPosition = trainDataset.idToPosition[anchorId]
    targetLabel = int(trainDataset.labels[anchorPosition])
    configuredTargetLabel = int(backdoorConfig["targetLabel"])
    if configuredTargetLabel != targetLabel:
        raise ValueError(
            f"LFBA targetLabel {configuredTargetLabel} does not match "
            f"anchor {anchorId} label {targetLabel}"
        )
    randomGenerator = random.Random(seed)
    auditRecords = {
        "attack": "lfba",
        "anchorSetting": backdoorConfig["anchorId"],
        "anchorId": anchorId,
        "targetLabel": targetLabel,
        "targetLabelAccess": "external evaluation only",
        "poisonRate": float(backdoorConfig["poisonRate"]),
        "switchRate": float(backdoorConfig["switchRate"]),
        "triggerType": str(backdoorConfig["lfbaTriggerType"]),
        "triggerDimensions": triggerDimensions,
        "triggerDimensionSpace": "party B local coordinates",
        "triggerFeatureValue": triggerFeatureValue,
        "similarity": "inner product divided by L1 norms",
        "poisonSetFixedAtEpoch": 1,
        "checkpointSelection": (
            "post-poisoning validation (BA + ASR) / 2; the gradient-collection "
            "epoch and test split are excluded"
        ),
        "testEvaluationProtocol": (
            "public LFBA base view: runtime inject the complete input-space "
            "trigger into every row; the organizer applies the private non-target mask"
            if "lfba" in bundle.backdoorTests
            else "synthetic compatibility fallback: runtime trigger every test sample"
        ),
        "labelModified": False,
        "partyAModified": False,
        "pairAppending": False,
        "switchDirection": "triggered source outside poison set -> target inside poison set",
        "poisonIds": [],
        "epochs": [],
    }

    totalEpochs = int(trainer.config["epochs"])
    writeProgress(
        trainer.config,
        "[LFBA] Epoch 1: collecting cut-layer gradients",
    )
    for epoch in range(totalEpochs):
        epochStartTime = time.perf_counter()
        replacementMapping = []
        selectedTargetIds = []
        selectedGradientNorms = []
        if epoch >= 1:
            if previousGradients is None:
                raise RuntimeError("LFBA requires gradients from the preceding epoch")
            if poisonIds is None:
                poisonIds = buildPoisonSet(anchorId, orderedIds, previousGradients, poisonCount)
                poisonPositions = [trainDataset.idToPosition[sampleId] for sampleId in poisonIds]
                poisonLabels = trainDataset.labels[poisonPositions]
                consistency = float((poisonLabels == targetLabel).float().mean().item())
                auditRecords["poisonIds"] = poisonIds
                auditRecords["poisonLabelConsistency"] = consistency
                writeProgress(
                    trainer.config,
                    (
                        f"[LFBA] Epoch {epoch + 1}: poison set established; "
                        f"poisoning enabled | poison_samples={len(poisonIds)}"
                    ),
                )
            selectedTargetIds, selectedGradientNorms = selectHardSamples(
                poisonIds,
                previousGradients,
                selectedCount,
            )
            replacementMapping = buildReplacementMapping(
                orderedIds,
                poisonIds,
                selectedTargetIds,
                randomGenerator,
            )
            injectPoisonFeatures(
                trainDataset,
                cleanPartyB,
                replacementMapping,
                str(backdoorConfig["lfbaTriggerType"]),
                triggerDimensions,
                triggerFeatureValue,
            )
        else:
            trainDataset.partyB = cleanPartyB.clone()

        trainingLoss, currentGradients = trainLfbaEpoch(
            trainer,
            trainDataset,
            epoch,
            selectedTargetIds,
        )
        if len(currentGradients) != len(trainDataset):
            raise RuntimeError("LFBA did not collect one passive gradient for every training sample")
        adjustLfbaLearningRate(trainer, epoch)
        validation = trainer.evaluate(
            bundle.validation,
            inferenceContext={"backdoor": "lfba", "triggered": False},
        )
        robustness = calculateAttackSuccess(
            trainer,
            triggeredValidation,
            targetLabel,
        )
        selectionScore = (validation["metrics"]["ba"] + robustness["asr"]) / 2.0
        history.epochs.append(
            {
                "epoch": epoch,
                "trainingLoss": trainingLoss,
                "validationBa": validation["metrics"]["ba"],
                "validationAuc": validation["metrics"]["auc"],
                "validationAsr": robustness["asr"],
                "selectionScore": selectionScore,
                "checkpointEligible": isLfbaCheckpointEligible(epoch),
            }
        )
        auditRecords["epochs"].append(
            {
                "epoch": epoch,
                "checkpointEligible": isLfbaCheckpointEligible(epoch),
                "gradientSamples": len(currentGradients),
                "selectedTargetIds": selectedTargetIds,
                "selectedGradientNorms": selectedGradientNorms,
                "replacementMapping": replacementMapping,
            }
        )
        if isLfbaCheckpointEligible(epoch):
            if selectionScore > bestScore:
                bestScore = selectionScore
                history.bestEpoch = epoch
                history.bestValidationAccuracy = validation["metrics"]["ba"]
                history.bestValidationAuc = validation["metrics"]["auc"]
                history.bestSelectionScore = selectionScore
                bestState = captureModelState(trainer.models)
                epochsWithoutImprovement = 0
            else:
                epochsWithoutImprovement += 1
        else:
            # Epoch 1 仅用于收集构造毒化集所需的梯度，不消耗早停耐心。
            epochsWithoutImprovement = 0
        previousGradients = currentGradients
        elapsedSeconds = time.perf_counter() - epochStartTime
        patienceText = (
            f"{epochsWithoutImprovement}/{patience}"
            if patience > 0
            else "off"
        )
        bestScoreText = (
            f"{history.bestSelectionScore:.4f}"
            if history.bestEpoch >= 1
            else "pending_poisoning"
        )
        writeProgress(
            trainer.config,
            (
                f"Epoch {epoch + 1}/{totalEpochs} | loss={trainingLoss:.4f} | "
                f"val_acc={validation['metrics']['ba']:.4f} | "
                f"ASR={robustness['asr']:.4f} | "
                f"selection_score={selectionScore:.4f} | "
                f"best_score={bestScoreText} | "
                f"patience={patienceText} | time={elapsedSeconds:.1f}s"
            ),
        )
        if patience > 0 and epochsWithoutImprovement >= patience:
            writeProgress(
                trainer.config,
                (
                    f"[EarlyStop][LFBA] stopped at epoch {epoch + 1}; "
                    f"best_epoch={history.bestEpoch + 1} | "
                    f"best_score={history.bestSelectionScore:.4f}"
                ),
            )
            break

    if history.bestEpoch < 1:
        raise RuntimeError("LFBA did not produce a post-poisoning checkpoint")
    restoreModelState(trainer.models, bestState)
    trainDataset.partyB = cleanPartyB
    validationRobustness = calculateAttackSuccess(
        trainer,
        triggeredValidation,
        targetLabel,
    )
    writeProgress(
        trainer.config,
        (
            f"[LFBA] completed | best_epoch={history.bestEpoch + 1} | "
            f"best_score={history.bestSelectionScore:.4f}"
        ),
    )
    return LfbaTrainingResult(
        history=history,
        targetLabel=targetLabel,
        triggerDimensions=triggerDimensions,
        triggeredTestDataset=triggeredTest,
        validationRobustness=validationRobustness,
        auditRecords=auditRecords,
    )
