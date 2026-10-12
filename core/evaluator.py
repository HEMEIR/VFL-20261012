import numpy as np
import torch
from torch.utils.data import DataLoader

from metrics.utility import calculateAccuracy, calculateAuc


def validateTransformedTensor(name, reference, transformed):
    if not torch.is_tensor(transformed):
        raise TypeError(f"{name} must return a torch.Tensor")
    if transformed.shape != reference.shape:
        raise ValueError(f"{name} changed tensor shape")
    if transformed.dtype != reference.dtype:
        raise ValueError(f"{name} changed tensor dtype")
    if transformed.device != reference.device:
        raise ValueError(f"{name} changed tensor device")
    if not bool(torch.isfinite(transformed).all().item()):
        raise FloatingPointError(f"{name} produced non-finite values")
    return transformed


def predictModels(
    models,
    dataset,
    batchSize,
    device,
    partyBEmbeddingTransform=None,
    partyARepresentationHook=None,
    inferenceContext=None,
):
    """通过统一推理入口返回预测；攻击变换先于 A 方防御 hook 执行。"""
    models.bottomA.eval()
    models.bottomB.eval()
    models.topModel.eval()
    context = {
        "phase": "inference",
        "backdoor": "none",
        "labelInference": "none",
        "triggered": False,
    }
    if inferenceContext is not None:
        context.update(dict(inferenceContext))
    ids = []
    probabilities = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batchSize, shuffle=False):
            inputA = batch["partyA"].to(device)
            inputB = batch["partyB"].to(device)
            embeddingA = models.bottomA(inputA)
            embeddingB = models.bottomB(inputB)
            sampleIds = tuple(str(sampleId) for sampleId in batch["id"])
            if partyBEmbeddingTransform is not None:
                transformedB = partyBEmbeddingTransform(embeddingB, sampleIds)
                embeddingB = validateTransformedTensor(
                    "partyBEmbeddingTransform",
                    embeddingB,
                    transformedB,
                )
            if partyARepresentationHook is not None:
                transformed = partyARepresentationHook(
                    embeddingA,
                    embeddingB,
                    sampleIds,
                    context,
                )
                if not isinstance(transformed, tuple) or len(transformed) != 2:
                    raise TypeError(
                        "partyARepresentationHook must return "
                        "(partyARepresentation, partyBRepresentation)"
                    )
                embeddingA = validateTransformedTensor(
                    "partyARepresentationHook party A output",
                    embeddingA,
                    transformed[0],
                )
                embeddingB = validateTransformedTensor(
                    "partyARepresentationHook party B output",
                    embeddingB,
                    transformed[1],
                )
            logits = models.topModel(embeddingA, embeddingB)
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


def evaluateModels(
    models,
    dataset,
    batchSize,
    device,
    partyBEmbeddingTransform=None,
    partyARepresentationHook=None,
    inferenceContext=None,
    predictionTransform=None,
):
    """评测只接受带标签数据；公开隐藏测试集必须改用 predictModels。"""
    if not dataset.hasLabels():
        raise ValueError("Evaluation requires a labeled dataset")
    predictions = predictModels(
        models,
        dataset,
        batchSize,
        device,
        partyBEmbeddingTransform=partyBEmbeddingTransform,
        partyARepresentationHook=partyARepresentationHook,
        inferenceContext=inferenceContext,
    )
    labels = dataset.labels.cpu().tolist()
    predictions["labels"] = labels
    # The inverse alias map MUST be applied BEFORE the metrics are computed.  Applying it in
    # the caller (VflTrainer.evaluate) was measured to be too late: this function had already
    # scored the ALIAS predictions, so dataset_02_pmc reported accuracy 0.130211 instead of
    # 0.87 and dataset_03_villain aborted on VILLAIN's minimum-accuracy checkpoint check.
    if predictionTransform is not None:
        predictions["predictions"] = list(
            predictionTransform(list(predictions["predictions"]), list(predictions["ids"]))
        )
    predictions["metrics"] = {
        "ba": calculateAccuracy(labels, predictions["predictions"]),
        "auc": calculateAuc(labels, predictions["probabilities"]),
    }
    return predictions
