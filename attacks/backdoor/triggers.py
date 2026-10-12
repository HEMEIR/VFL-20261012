import numpy as np
import torch

from datasets.dataset import TwoPartyDataset
from datasets.validator import validateCleanBackdoorTestView


def resolveStructuredTriggerDimensions(
    partyB,
    dimensionCount,
    seed,
    configuredDimensions=None,
):
    """优先使用配置中的显式局部坐标，否则按固定种子选择维度。"""
    featureCount = int(partyB.shape[1])
    if configuredDimensions is not None:
        if not isinstance(configuredDimensions, list) or not configuredDimensions:
            raise ValueError("triggerDimensionIndices must be a non-empty list")
        dimensions = []
        for dimension in configuredDimensions:
            if isinstance(dimension, bool) or not isinstance(dimension, int):
                raise ValueError("triggerDimensionIndices must contain integers")
            if dimension < 0 or dimension >= featureCount:
                raise ValueError(
                    f"Trigger dimension {dimension} is outside party B with "
                    f"{featureCount} features"
                )
            dimensions.append(int(dimension))
        if len(set(dimensions)) != len(dimensions):
            raise ValueError("triggerDimensionIndices must not contain duplicates")
        if dimensionCount not in (0, len(dimensions)):
            raise ValueError(
                "triggerDimensions must match the length of triggerDimensionIndices"
            )
        return sorted(dimensions)

    if dimensionCount <= 0 or dimensionCount > featureCount:
        raise ValueError(
            f"triggerDimensions must be between 1 and {featureCount}"
        )
    randomGenerator = np.random.RandomState(seed)
    dimensions = randomGenerator.choice(featureCount, dimensionCount, replace=False)
    return sorted(int(dimension) for dimension in dimensions.tolist())


def selectTriggerDimensions(
    triggerType,
    partyB,
    dimensionCount,
    seed,
    configuredDimensions=None,
):
    """按匿名配置和固定随机种子选择恶意被动方的局部触发维度。"""
    if triggerType == "imageTriangle":
        return []
    if triggerType != "structuredDimensions":
        raise ValueError(f"Unsupported LFBA trigger type: {triggerType}")
    # 未显式配置坐标时，使用固定的 NumPy RandomState 序列生成坐标。
    return resolveStructuredTriggerDimensions(
        partyB,
        dimensionCount,
        seed,
        configuredDimensions,
    )


def addTriggerToPartyB(partyB, triggerType, triggerDimensions, triggerFeatureValue):
    """在 B 方局部特征上加入 LFBA 触发器。"""
    triggeredPartyB = partyB.clone()
    if triggerType == "imageTriangle":
        if triggeredPartyB.ndim != 4 or tuple(triggeredPartyB.shape[1:]) != (3, 32, 16):
            raise ValueError("Image party B must have shape [N, 3, 32, 16]")
        triggeredPartyB[:, :, -3:, -3:] = 0.0
        triggeredPartyB[:, :, -3, -1] = triggerFeatureValue
        triggeredPartyB[:, :, -1, -3] = triggerFeatureValue
        triggeredPartyB[:, :, -2, -2] = triggerFeatureValue
        triggeredPartyB[:, :, -1, -1] = triggerFeatureValue
        return triggeredPartyB
    if triggerType != "structuredDimensions":
        raise ValueError(f"Unsupported LFBA trigger type: {triggerType}")
    if not triggerDimensions:
        raise ValueError("Structured LFBA trigger requires trigger dimensions")
    triggeredPartyB[:, triggerDimensions] = triggerFeatureValue
    return triggeredPartyB


def buildTriggeredDataset(dataset, triggerType, triggerDimensions, triggerFeatureValue):
    """复制数据集并只修改 B 方特征，标签和 A 方特征保持不变。"""
    triggeredPartyB = addTriggerToPartyB(
        dataset.partyB,
        triggerType,
        triggerDimensions,
        triggerFeatureValue,
    )
    return TwoPartyDataset(
        dataset.ids.copy(),
        dataset.partyA,
        triggeredPartyB,
        dataset.labels,
        dataset.datasetName,
    )
