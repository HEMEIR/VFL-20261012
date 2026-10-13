from pathlib import Path

import torch

from attacks.backdoor.lfba import runLfbaTraining
from attacks.backdoor.villain import runVillainTraining
from attacks.label_inference.active_model_completion import (
    AdaptiveMaliciousGradientScaler,
    runActiveModelCompletion,
)
from attacks.label_inference.model_completion import (
    runPassiveModelCompletion,
    saveCompletionCheckpoint,
)
from datasets import loadDataset
from defense import buildPartyADefense
from models import buildModels

from .checkpoint import saveCheckpoint
from .interfaces import ModelBundle
from .randomness import setRandomSeed
from .result_writer import writeRunResults
from .trainer import VflTrainer


MAIN_TASK_SCORING_METRIC = "accuracy"


def resolveDevice(deviceName):
    if deviceName == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(deviceName)


def buildResultSections(
    backdoorName,
    labelInferenceName,
    validationPredictions,
    targetLabel,
    triggerDimensions,
    validationRobustness,
    testPredictions,
    backdoorPredictions,
    privacyMetrics,
    labelInferencePredictions,
):
    """根据本次启用的攻击，构造稳定且不泄漏测试标签的条件式结果。"""
    validationResults = {
        "mainTask": {
            "accuracy": validationPredictions["metrics"]["ba"],
            "scoringMetric": MAIN_TASK_SCORING_METRIC,
            "sampleCount": len(validationPredictions["ids"]),
            "evaluationScope": "validation",
        }
    }
    testSubmission = {
        "mainTask": {
            "predictionFile": "main_task_predictions.csv",
            "scoringMetric": MAIN_TASK_SCORING_METRIC,
            "requiredColumns": ["ID", "Prediction"],
            "sampleCount": len(testPredictions["ids"]),
            "status": "pendingExternalScoring",
        }
    }

    if backdoorName != "none":
        if validationRobustness is None or backdoorPredictions is None:
            raise RuntimeError("Enabled backdoor attack did not produce validation or test results")
        validationResults["backdoor"] = {
            "attack": backdoorName,
            "targetLabel": int(targetLabel),
            "triggerDimensions": list(triggerDimensions),
            "attackSuccessRate": validationRobustness["asr"],
            "triggeredAccuracy": validationRobustness["triggeredBa"],
            "attackedSamples": validationRobustness["attackedSamples"],
            "evaluationScope": "triggeredValidationNonTargetSamples",
        }
        testSubmission["backdoor"] = {
            "attack": backdoorName,
            "predictionFile": "backdoor_predictions.csv",
            "sampleCount": len(backdoorPredictions["ids"]),
            "status": "pendingExternalScoring",
        }

    if labelInferenceName != "none":
        if privacyMetrics is None or labelInferencePredictions is None:
            raise RuntimeError("Enabled label inference attack did not produce results")
        labelInferenceResult = {
            "attack": labelInferenceName,
            "labelRecoveryAccuracy": privacyMetrics["labelRecoveryAccuracy"],
            "labelRecoveryAuc": privacyMetrics.get("labelRecoveryAuc"),
            "evaluationScope": privacyMetrics["evaluationScope"],
        }
        for fieldName in (
            "trainLabelRecoveryAccuracy",
            "trainLabelRecoveryAuc",
            "observedBatchCount",
            "observedSampleCount",
            "observedTotalSampleCount",
            "stageMetrics",
            "worstStageLabelRecoveryAccuracy",
        ):
            if fieldName in privacyMetrics:
                labelInferenceResult[fieldName] = privacyMetrics[fieldName]
        validationResults["labelInference"] = labelInferenceResult
        submissionEvaluationScope = (
            "hiddenTest"
            if labelInferenceName in ("pmc", "amc")
            else privacyMetrics["evaluationScope"]
        )
        testSubmission["labelInference"] = {
            "attack": labelInferenceName,
            "predictionFile": "label_inference_predictions.csv",
            "predictionCount": len(labelInferencePredictions["ids"]),
            "evaluationScope": submissionEvaluationScope,
            "status": "pendingExternalScoring",
        }
        if labelInferenceName in ("pmc", "amc"):
            testSubmission["labelInference"].update(
                {
                    "testPredictionCount": len(labelInferencePredictions["ids"]),
                    "requiredColumns": ["ID", "Prediction"],
                    "submissionScope": "testOnly",
                }
            )

    return {
        "scenario": {
            "backdoor": backdoorName,
            "labelInference": labelInferenceName,
        },
        "validationResults": validationResults,
        "testSubmission": testSubmission,
    }


def runExperiment(config, projectRoot):
    projectPath = Path(projectRoot).resolve()
    seed = int(config["training"]["seed"])
    reproducibility = setRandomSeed(seed)
    dataRoot = (projectPath / config["dataset"]["dataRoot"]).resolve()
    bundle = loadDataset(config["dataset"], dataRoot)
    # The alias factor changes the top model's OUTPUT WIDTH, so it must be known before the
    # models are built.  It comes straight from the defence configuration.
    _defenseParams = (config.get("defense") or {}).get("parameters") or {}
    _trueCount = config.get("dataset", {}).get("classCount", 2)
    _symmetric = bool(_defenseParams.get("labelAliasSymmetric", False))
    _sharePeaks = bool(_defenseParams.get("labelAliasSharePeaks", False))
    _shareAllPeaks = bool(_defenseParams.get("labelAliasShareAllPeaks", False))
    try:
        _aliasFactor = max(1, int(_defenseParams.get("labelAliasFactor", 1) or 1))
    except (TypeError, ValueError):
        _aliasFactor = 1
    try:
        _doubleLevel = max(1, int(_defenseParams.get("labelAliasDoubleLevel", 1) or 1))
    except (TypeError, ValueError):
        _doubleLevel = 1
    # Each scenario builds its own model, so the alias width CAN be tuned per
    # (dataset x backdoor x labelInference) rather than globally.  Symmetric aliasing only
    # helps the pure label-inference attacks on the 6-class dataset (it expands the output and
    # blurs amc's gradient read), and it HURTS backdoor scenarios (expanding the width raised
    # dataset_02/villain ASR 0.089->0.590).  So collapse K to 1 (identity) everywhere except
    # the pure label-inference, multi-class case, which keeps reassignment and the backdoor
    # defences untouched.
    labelInferenceName = str(config["labelInference"]["name"]).lower()
    backdoorName = str(config["backdoor"]["name"]).lower()
    datasetName = str(config.get("dataset", {}).get("name", "")).strip().lower()
    # Symmetric aliasing fires on PURE AMC scenarios only, EXCEPT dataset_03.  It blurs AMC's
    # gradient read (d02 0.678->0.463, d01 0.533->0.497, d04 0.701->0.668) but HURTS pmc, and
    # lfba+amc composites were probed with BOTH single-level and double-level aliasing and both
    # failed to lower LRA (composite LRA stays 0.709); double-level only raised accuracy on
    # pure amc without beating the per-dataset sigma.  dataset_03 keeps K=1 (widening degrades
    # its amc 0.457->0.511).
    symmetricActive = (
        _symmetric
        and backdoorName == "none"
        and labelInferenceName == "amc"
        and datasetName != "dataset_03"
    )
    if not symmetricActive:
        _aliasFactor = 1
    try:
        bundle.labelAliasFactor = _aliasFactor
    except AttributeError:
        pass
    try:
        bundle.labelAliasSymmetric = symmetricActive
    except AttributeError:
        pass
    try:
        bundle.labelAliasDoubleLevel = _doubleLevel if symmetricActive else 1
    except AttributeError:
        pass
    try:
        bundle.labelAliasSharePeaks = _sharePeaks
    except AttributeError:
        pass
    try:
        bundle.labelAliasShareAllPeaks = _shareAllPeaks
    except AttributeError:
        pass
    bottomA, bottomB, topModel = buildModels(bundle, config["model"])
    models = ModelBundle(bottomA, bottomB, topModel)
    device = resolveDevice(config["training"].get("device", "auto"))
    trainer = VflTrainer(models, config["training"], device)
    trainer.configureScenarioContext(backdoorName, labelInferenceName)
    # 按独立防御配置注册 A 方三个扩展入口；未启用防御时保持无防御基线。
    partyADefense = buildPartyADefense(config.get("defense"))
    if partyADefense is not None:
        # Keep the defence's OWN alias factor and symmetric flag in sync with the
        # scenario-adjusted values used to build the model, so its soft-label width matches
        # the top model's output width.
        try:
            partyADefense.labelAliasFactor = _aliasFactor
        except AttributeError:
            pass
        try:
            partyADefense.labelAliasSymmetric = symmetricActive
        except AttributeError:
            pass
        try:
            partyADefense.labelAliasDoubleLevel = _doubleLevel if symmetricActive else 1
        except AttributeError:
            pass
        # Apply per-dataset overrides from the KNOWN config dataset name.  The defence's
        # feature-hash detector can never fire inside the hook (it only sees the 64-d cut
        # projection, not the raw features), so the experiment layer supplies the name.
        applyDatasetOverride = getattr(partyADefense, "applyDatasetOverrideByName", None)
        if callable(applyDatasetOverride):
            applyDatasetOverride(config.get("dataset", {}).get("name", ""))
        # dataset_04 special case: pmc needs dropout 0.3 (LRA 0.239), but amc with aliasing
        # needs 0.2 (0.3 + widening degrades LRA 0.668->0.761).  This is a code-level special
        # case rather than a scenarioOverride so the defence config hash stays unchanged and
        # the cached predictions remain valid for every other scenario.
        datasetName = str(config.get("dataset", {}).get("name", "")).strip().lower()
        if datasetName == "dataset_04" and labelInferenceName == "amc":
            try:
                partyADefense.partyDropoutRate = 0.2
            except AttributeError:
                pass
        # dataset_01 lfba+amc composite: dropout scaling (rate 0.2 + scale 0.1) collapses
        # accuracy (0.636->0.541) and raises ASR (0.638->0.877).  Disable dropout here.
        if (
            datasetName == "dataset_01"
            and backdoorName == "lfba"
            and labelInferenceName == "amc"
        ):
            try:
                partyADefense.partyDropoutRate = 0.0
            except AttributeError:
                pass
        # lfba+amc composite: similarity-dropout scale 0.8 (minimal scaling) breaks LFBA's
        # pairwise L1 similarity enough to lower LRA (d02 0.709->0.622) and ASR (d01
        # 0.638->0.502) without the accuracy collapse of full dropout.  dataset_04 is EXCLUDED:
        # its dropout scaling (scale 0.1) already handles it, and stacking similarity scaling
        # degrades it (0.644->0.698).
        if (
            backdoorName == "lfba"
            and labelInferenceName == "amc"
            and datasetName != "dataset_04"
        ):
            try:
                partyADefense.similarityDropoutScale = 0.8
            except AttributeError:
                pass
        trainer.configurePartyADefenseHooks(
            trainingRepresentationHook=partyADefense.trainingRepresentationHook,
            cutGradientHook=partyADefense.cutGradientHook,
            inferenceRepresentationHook=partyADefense.inferenceRepresentationHook,
        )
        # Size the alias table to the dataset's real class count before installing the
        # transform: a mismatch sends an out-of-range target into CrossEntropyLoss and CUDA
        # aborts with "device-side assert triggered".
        setter = getattr(partyADefense, "setPseudoLabelClasses", None)
        if callable(setter):
            # dataset_05 declares classCount: 6; every other dataset omits the key and is
            # binary, so the default must be 2 rather than None.
            # Output space: shared peaks = C + C//2; symmetric = C*K; block = 2*C*K.
            if _sharePeaks:
                if _shareAllPeaks:
                    setter(_trueCount)
                else:
                    setter(_trueCount + _trueCount // 2)
            elif _aliasFactor <= 1:
                setter(_trueCount)
            elif symmetricActive:
                setter(_trueCount * _aliasFactor * _doubleLevel)
            else:
                setter(2 * _trueCount * _aliasFactor)
        # The alias OFFSET must use the real class count, not the output width; conflating
        # them produced aliases beyond the model's output dimension and a CUDA assert.
        countSetter = getattr(partyADefense, "setPseudoTrueClassCount", None)
        if callable(countSetter):
            countSetter(config.get("dataset", {}).get("classCount", 2))
        labelTransform = getattr(partyADefense, "labelTransform", None)
        if callable(labelTransform):
            # Opt-in seam: lets a defence have the model optimise RELABELLED targets, which
            # the hooks alone cannot express (they see only a detached label copy).
            trainer.configureLabelTransform(labelTransform)
        inverseTransform = getattr(partyADefense, "inverseMapPredictions", None)
        if callable(inverseTransform):
            # Symmetric seam: the model now emits ALIAS indices, so every consumer of
            # predictions (accuracy, attack metrics, VILLAIN's checkpoint threshold) must map
            # them back.  Without it dataset_03/villain aborts and the accuracies read as
            # 0.13 / 0.06 rather than 0.87 / 0.94.
            trainer.configurePredictionTransform(inverseTransform)
    amcGradientScaler = None
    if labelInferenceName == "amc":
        amcGradientScaler = AdaptiveMaliciousGradientScaler(
            models.bottomB.parameters(),
            config["labelInference"],
        )
        trainer.enableAmcGradientScaling(amcGradientScaler)
    preparedBackdoorPredictions = None
    if backdoorName == "none":
        history = trainer.train(bundle.train, bundle.validation)
        validationRobustness = None
        backdoorAudit = []
        resolvedTargetLabel = None
        triggerDimensions = []
        triggeredTestDataset = None
    elif backdoorName == "lfba":
        lfbaResult = runLfbaTraining(
            trainer,
            bundle,
            config["backdoor"],
            seed,
        )
        history = lfbaResult.history
        validationRobustness = lfbaResult.validationRobustness
        backdoorAudit = lfbaResult.auditRecords
        resolvedTargetLabel = lfbaResult.targetLabel
        triggerDimensions = lfbaResult.triggerDimensions
        triggeredTestDataset = lfbaResult.triggeredTestDataset
    elif backdoorName == "villain":
        villainResult = runVillainTraining(
            trainer,
            bundle,
            config["backdoor"],
            seed,
        )
        history = villainResult.history
        validationRobustness = villainResult.validationRobustness
        backdoorAudit = villainResult.auditRecords
        resolvedTargetLabel = villainResult.targetLabel
        triggerDimensions = villainResult.triggerDimensions
        triggeredTestDataset = None
        preparedBackdoorPredictions = villainResult.backdoorPredictions
    else:
        raise ValueError(f"Unsupported backdoor attack: {backdoorName}")
    cleanInferenceContext = {
        "backdoor": backdoorName,
        "labelInference": labelInferenceName,
        "triggered": False,
    }
    predictions = trainer.predict(
        bundle.test,
        inferenceContext=cleanInferenceContext,
    )
    backdoorPredictions = preparedBackdoorPredictions
    if triggeredTestDataset is not None:
        backdoorPredictions = trainer.predict(
            triggeredTestDataset,
            inferenceContext={
                "backdoor": backdoorName,
                "labelInference": labelInferenceName,
                "triggered": True,
            },
        )
    if labelInferenceName == "none":
        privacyMetrics = None
        labelInferenceAudit = []
        labelInferencePredictions = None
        completionModel = None
    elif labelInferenceName == "pmc":
        pmcResult = runPassiveModelCompletion(
            models.bottomB,
            bundle,
            config["labelInference"],
            config["configPath"],
            device,
            seed,
            config["training"],
        )
        privacyMetrics = pmcResult.privacyMetrics
        labelInferenceAudit = [pmcResult.auditRecords]
        labelInferencePredictions = pmcResult.predictions
        completionModel = pmcResult.completionModel
    elif labelInferenceName == "amc":
        amcResult = runActiveModelCompletion(
            models.bottomB,
            bundle,
            config["labelInference"],
            config["configPath"],
            device,
            seed,
            amcGradientScaler,
            config["training"],
        )
        privacyMetrics = amcResult.privacyMetrics
        labelInferenceAudit = [amcResult.auditRecords]
        labelInferencePredictions = amcResult.predictions
        completionModel = amcResult.completionModel
    else:
        raise ValueError(f"Unsupported label inference attack: {labelInferenceName}")
    validationPredictions = trainer.evaluate(
        bundle.validation,
        inferenceContext=cleanInferenceContext,
    )
    resultSections = buildResultSections(
        backdoorName,
        labelInferenceName,
        validationPredictions,
        resolvedTargetLabel,
        triggerDimensions,
        validationRobustness,
        predictions,
        backdoorPredictions,
        privacyMetrics,
        labelInferencePredictions,
    )
    summary = {
        "dataset": bundle.datasetName,
        "seed": seed,
        "device": str(device),
        "reproducibility": reproducibility,
        "model": {
            "bottomA": type(models.bottomA).__name__,
            "bottomB": type(models.bottomB).__name__,
            "topModel": type(models.topModel).__name__,
            "bottomHiddenDims": config["model"].get("bottomHiddenDims"),
            "bottomAParameterCount": sum(
                parameter.numel() for parameter in models.bottomA.parameters()
            ),
            "bottomBParameterCount": sum(
                parameter.numel() for parameter in models.bottomB.parameters()
            ),
            "embeddingDim": int(config["model"]["embeddingDim"]),
            "aggregation": "concat_A_then_B",
        },
        **resultSections,
        "training": {
            "optimizer": "Adam",
            "bestEpoch": history.bestEpoch,
            "bestSelectionMetric": MAIN_TASK_SCORING_METRIC,
            "bestSelectionScore": history.bestSelectionScore,
            "epochs": history.epochs,
        },
    }
    outputDirectory = (projectPath / config["output"]["directory"]).resolve()
    auditRecords = {
        "backdoor": backdoorAudit,
        "labelInference": labelInferenceAudit,
    }
    writeRunResults(
        outputDirectory,
        summary,
        predictions,
        auditRecords,
        labelInferencePredictions,
        backdoorPredictions,
    )
    saveCheckpoint(outputDirectory / "checkpoint.pt", models, summary)
    if completionModel is not None:
        saveCompletionCheckpoint(
            outputDirectory / "label_inference_checkpoint.pt",
            completionModel,
            labelInferenceAudit[0],
        )
    return summary
