import numpy as np
from sklearn.metrics import roc_auc_score


def calculateAccuracy(labels, predictions):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    return float((labels == predictions).mean())


def calculateAuc(labels, probabilities):
    probabilityArray = np.asarray(probabilities)
    try:
        if probabilityArray.ndim == 2 and probabilityArray.shape[1] > 2:
            return float(
                roc_auc_score(labels, probabilityArray, multi_class="ovr", average="macro")
            )
        return float(roc_auc_score(labels, probabilityArray))
    except ValueError:
        return None
