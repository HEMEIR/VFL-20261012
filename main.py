import argparse
import json
from pathlib import Path

from core.config import loadConfig, loadDefenseConfig, resolveDefenseConfigPath
from core.experiment import runExperiment


DATASET_ALIASES = {
    "dataset_01": "dataset_01",
    "dataset_02": "dataset_02",
    "dataset_03": "dataset_03",
    "dataset_04": "dataset_04",
    "dataset_05": "dataset_05",
}


def normalizeDatasetName(datasetName):
    normalizedName = str(datasetName).lower()
    if normalizedName not in DATASET_ALIASES:
        raise ValueError(f"Unsupported dataset: {datasetName}")
    return DATASET_ALIASES[normalizedName]


def buildParser():
    parser = argparse.ArgumentParser(
        description="Two-party VFL competition baseline",
        allow_abbrev=False,
    )
    parser.add_argument("--config", help="YAML configuration path")
    parser.add_argument(
        "--defense-config",
        dest="defenseConfig",
        default="configs/defense.yaml",
        help=(
            "Participant defense-only YAML inside configs/ "
            "(default: configs/defense.yaml)"
        ),
    )
    parser.add_argument(
        "--dataset",
        help="dataset_01, dataset_02, dataset_03, dataset_04 or dataset_05",
    )
    parser.add_argument("--epochs", "--epoch", dest="epochs", type=int)
    parser.add_argument("--batch-size", "--batch_size", dest="batchSize", type=int)
    parser.add_argument("--learning-rate", "--lr", dest="learningRate", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", help="auto, cpu, cuda or cuda:N")
    parser.add_argument("--gpu", type=int, help="Compatibility alias for --device cuda:N")
    parser.add_argument(
        "--no-progress",
        dest="progressEnabled",
        action="store_false",
        default=None,
        help="Disable tqdm progress bars and progress messages",
    )
    parser.add_argument("--output-directory", dest="outputDirectory")
    parser.add_argument(
        "--attack",
        type=str.lower,
        choices=["none", "lfba", "villain"],
        help="Backdoor robustness scenario: none, lfba or villain",
    )
    parser.add_argument("--anchor-id", "--anchor_idx", dest="anchorId")
    parser.add_argument("--poison-rate", "--poison_rate", dest="poisonRate", type=float)
    parser.add_argument("--switch-rate", "--select_rate", dest="switchRate", type=float)
    parser.add_argument(
        "--trigger-dimensions",
        "--poison_dimensions",
        dest="triggerDimensions",
        type=int,
    )
    parser.add_argument(
        "--trigger-feature-value",
        "--trigger_feature_clip",
        dest="triggerFeatureValue",
        type=float,
    )
    parser.add_argument("--target-label", "--target_label", dest="targetLabel", type=int)
    parser.add_argument(
        "--known-target-ids",
        "--known_target_ids",
        dest="knownTargetIds",
        nargs="+",
        help="Target-class sample IDs known to VILLAIN's passive party",
    )
    parser.add_argument("--attack-seed", "--attack_seed", dest="attackSeed", type=int)
    parser.add_argument(
        "--villain-start-epoch",
        dest="villainStartEpoch",
        type=int,
    )
    parser.add_argument(
        "--villain-trigger-dimension-count",
        dest="villainTriggerDimensionCount",
        type=int,
    )
    parser.add_argument("--villain-beta", dest="villainBeta", type=float)
    parser.add_argument(
        "--villain-test-trigger-rate",
        dest="villainTestTriggerRate",
        type=float,
    )
    parser.add_argument(
        "--minimum-main-task-accuracy",
        dest="minimumMainTaskAccuracy",
        type=float,
    )
    parser.add_argument(
        "--select-replace",
        "--select_replace",
        dest="selectReplace",
        action="store_true",
        help="Compatibility flag; standard LFBA always uses selective sample switching",
    )
    parser.add_argument(
        "--label-inference",
        dest="labelInference",
        type=str.lower,
        choices=["none", "pmc", "amc"],
        help="Label inference scenario: none, PMC or AMC",
    )
    parser.add_argument("--auxiliary-labels-path", dest="auxiliaryLabelsPath")
    parser.add_argument("--completion-epochs", dest="completionEpochs", type=int)
    parser.add_argument("--completion-iterations", dest="iterationsPerEpoch", type=int)
    parser.add_argument("--completion-batch-size", dest="completionBatchSize", type=int)
    parser.add_argument(
        "--completion-learning-rate",
        dest="completionLearningRate",
        type=float,
    )
    parser.add_argument("--malicious-momentum", dest="maliciousMomentum", type=float)
    parser.add_argument("--malicious-dampening", dest="maliciousDampening", type=float)
    parser.add_argument("--gradient-scale-gamma", dest="gradientScaleGamma", type=float)
    parser.add_argument("--minimum-gradient-scale", dest="minimumGradientScale", type=float)
    parser.add_argument("--maximum-gradient-scale", dest="maximumGradientScale", type=float)
    return parser


def selectConfigPath(arguments, projectRoot):
    if arguments.config:
        candidate = Path(arguments.config)
        return candidate if candidate.is_absolute() else projectRoot / candidate
    datasetName = normalizeDatasetName(arguments.dataset or "dataset_03")
    return projectRoot / "configs" / f"{datasetName}.yaml"


def selectDefenseConfigPath(arguments, projectRoot):
    return resolveDefenseConfigPath(projectRoot, arguments.defenseConfig)


def buildOverrides(arguments):
    overrides = {}
    if arguments.dataset:
        overrides.setdefault("dataset", {})["name"] = normalizeDatasetName(arguments.dataset)
    trainingOverrides = {
        "epochs": arguments.epochs,
        "batchSize": arguments.batchSize,
        "learningRate": arguments.learningRate,
        "seed": arguments.seed,
        "device": arguments.device,
        "progressEnabled": arguments.progressEnabled,
    }
    if arguments.gpu is not None:
        if arguments.device is not None:
            raise ValueError("Use either --device or --gpu, not both")
        trainingOverrides["device"] = f"cuda:{arguments.gpu}"
    selectedTraining = {key: value for key, value in trainingOverrides.items() if value is not None}
    if selectedTraining:
        overrides["training"] = selectedTraining
    if arguments.outputDirectory:
        overrides["output"] = {"directory": arguments.outputDirectory}
    backdoorOverrides = {
        "name": arguments.attack,
        "anchorId": arguments.anchorId,
        "poisonRate": arguments.poisonRate,
        "switchRate": arguments.switchRate,
        "triggerDimensions": arguments.triggerDimensions,
        "triggerFeatureValue": arguments.triggerFeatureValue,
        "targetLabel": arguments.targetLabel,
        "knownTargetIds": arguments.knownTargetIds,
        "attackSeed": arguments.attackSeed,
        "startEpoch": arguments.villainStartEpoch,
        "triggerDimensionCount": arguments.villainTriggerDimensionCount,
        "beta": arguments.villainBeta,
        "testTriggerRate": arguments.villainTestTriggerRate,
        "minimumMainTaskAccuracy": arguments.minimumMainTaskAccuracy,
    }
    selectedBackdoor = {key: value for key, value in backdoorOverrides.items() if value is not None}
    if selectedBackdoor:
        overrides["backdoor"] = selectedBackdoor
    labelInferenceOverrides = {
        "name": arguments.labelInference,
        "auxiliaryLabelsPath": arguments.auxiliaryLabelsPath,
        "completionEpochs": arguments.completionEpochs,
        "iterationsPerEpoch": arguments.iterationsPerEpoch,
        "completionBatchSize": arguments.completionBatchSize,
        "completionLearningRate": arguments.completionLearningRate,
        "maliciousMomentum": arguments.maliciousMomentum,
        "maliciousDampening": arguments.maliciousDampening,
        "gradientScaleGamma": arguments.gradientScaleGamma,
        "minimumGradientScale": arguments.minimumGradientScale,
        "maximumGradientScale": arguments.maximumGradientScale,
    }
    selectedLabelInference = {
        key: value for key, value in labelInferenceOverrides.items() if value is not None
    }
    if selectedLabelInference:
        overrides["labelInference"] = selectedLabelInference
    return overrides


def main():
    projectRoot = Path(__file__).resolve().parent
    arguments = buildParser().parse_args()
    configPath = selectConfigPath(arguments, projectRoot)
    defenseConfigPath = selectDefenseConfigPath(arguments, projectRoot)
    defenseConfig = loadDefenseConfig(defenseConfigPath)
    config = loadConfig(
        configPath,
        buildOverrides(arguments),
        defenseConfig=defenseConfig,
    )
    summary = runExperiment(config, projectRoot)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
