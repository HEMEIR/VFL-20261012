"""Grid evaluation of purified_vfl parameters on a small scenario set.

Each grid point is given as ``name:key=value,key=value`` and is evaluated on a
list of scenarios by running the real pipeline with the protocol lock disabled.

Example
-------
    python scripts/grid.py \
        --scenarios dataset_03_clean dataset_03_lfba_p20 dataset_03_pmc \
        --epochs 6 \
        --point "base:gradientClipRelative=0" \
        --point "gs0.2:gradientScale=0.2"
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import loadConfig, normalizeDefenseConfigDocument  # noqa: E402
from core.experiment import runExperiment  # noqa: E402


DEFAULTS = {
    # Mirrors the shipped defaults of configs/defense_purified.yaml so that the
    # "base" grid point is the real configuration and every other point is a
    # one-parameter perturbation of it.
    "trainingRecenterStrength": 0.5,
    "inferenceRecenterStrength": 1.0,
    "inferenceUnconditionalForAttacks": "",
    "permutationInvariant": 0.0,
    "quantizeStep": 0.0,
    "sharedDirectionStrength": 0.0,
    "sharedDirectionThreshold": 0.5,
    "perSampleExcessStrength": 0.0,
    "perSampleExcessFactor": 2.0,
    "normExcessStrength": 0.0,
    "normExcessFactor": 3.0,
    "secureSplitStrength": 0.0,
    "secureSplitAlpha": 1.0,
    "secureSplitLinkOnly": 0.0,
    "referenceMomentum": 0.99,
    "referenceWindow": 0.0,
    "recenterMode": "soft",
    "recenterDetectionThreshold": 3.0,
    "recenterQuantile": 0.0,
    "recenterQuantileMultiplier": 1.0,
    "recenterStableFraction": 0.25,
    "batchLocationStatistic": "mean",
    "gradientClipQuantile": 0.0,
    "gradientClipNorm": 0.0,
    "gradientNoise": 0.0,
    "gradientClassCenterStrength": 0.0,
    "gradientClassCenterLabelInferenceOnly": 1,
    "gradientNoiseStartFraction": 0.35,
    "gradientNoiseRampFraction": 0.25,
    "gradientNoiseLabelInferenceOnly": 0.0,
    "gradientNoisePureLabelInferenceOnly": 0.0,
    "embeddingImportanceNoise": 0.0,
    "embeddingImportanceMomentum": 0.9,
    "embeddingImportanceLabelInferenceOnly": 0.0,
    "embeddingImportanceAllowlist": "",
    "partyDropoutRate": 0.0,
    "partyDropoutLabelInferenceOnly": 1.0,
    "partyDropoutScale": 0.0,
    "gradientShuffleStrength": 0.0,
    "gradientShufflePoisonedOnly": 0.0,
    "gradientShuffleBackdoorOnly": 0.0,
    "gradientShufflePureLabelInferenceOnly": 0.0,
    "gradientShuffleAllowlist": "",
    "gradientShufflePoisoningActiveOnly": 0.0,
    "gradientShufflePoisoningBackdoors": "",
    "gradientShuffleAdaptiveEma": 0.9,
    "gradientShuffleWarmupProtect": 1.0,
    "gradientSparsifyAdaptive": 0.0,
    "gradientQuantizeStep": 0.0,
    "gradientQuantizeUniform": 0.0,
    "capacityLabelInferenceOnly": 1.0,
    "auxiliaryLossMaxEma": 0.9,
    "auxiliaryExcludeLabelInferenceOnly": 1.0,
    "auxiliarySampleIdList": "",
    "auxiliaryExcludeSide": "b",
    "auxiliaryOverfitStrength": 0.0,
    "auxiliaryExclude": 0.0,
    "auxiliaryLossMaxClip": 4.0,
    "auxiliaryLossMaxStrength": 0.0,
    "auxiliaryLabelsPath": "",
    "adversarialPerturbEma": 0.9,
    "adversarialPerturbGain": 1.0,
    "adversarialPerturbStrength": 0.0,
    "prototypeLabelInferenceOnly": 1.0,
    "prototypePullback": 0.5,
    "prototypeStrength": 0.0,
    "representationSparseRate": 0.5,
    "representationSparseStrength": 0.0,
    "lowRankRefreshBatches": 50,
    "lowRankDirections": 0,
    "lowRankStrength": 0.0,
    "gradientQuantizeAdaptiveStrength": 0.0,
    "gradientClipAdaptiveStrength": 0.0,
    "gradientSparsifyRate": 0.0,
    "gradientShuffleAdaptiveFloor": 0.0,
    "gradientShuffleAdaptiveGain": 1.0,
    "gradientShuffleAdaptiveStrength": 0.0,
    "gradientShuffleLabelInferenceOnly": 0.0,
    # These two MUST mirror configs/defense_purified.yaml.  They were previously
    # left at 0.0 while the shipped configuration uses 0.5 / true, so the "base"
    # grid point silently ran with the jitter mechanism DISABLED and every
    # comparison against it was measured against the wrong reference.  That bug
    # produced a spurious "dataset_02 clean collapses to 0.5110" reading and
    # invalidated the jitter conclusions drawn from earlier sweeps.
    "trainingJitter": 0.5,
    "jitterPoisonedOnly": 1.0,
    "jitterWithLabelInference": 0.0,
    "jitterAnomalyGate": 0.0,
    "jitterForAttacks": "",
    "passiveUpdateRecenterStrength": 0.0,
    "representationNormStrength": 0.0,
    "representationNormTolerance": 0.0,
    "datasetOverrides": "",
    "scenarioOverrides": "",
    "pseudoLabelStrength": 0.0,
    "pseudoLabelClasses": 0,
    "pseudoLabelSeed": 0,
    "labelAliasFactor": 1,
    "labelAliasOffset": "",
    "labelAliasSoft": 0,
    "labelAliasSymmetric": 0,
    "labelAliasSharePeaks": 0,
    "labelAliasSigma": 1.0,
    "labelAliasSmoothing": 0.0,
    "diagnostics": 0.0,
}


def parseValue(text):
    """Parse a grid value as a number when possible, otherwise keep it a string."""
    try:
        return float(text)
    except ValueError:
        return text


def parsePoint(text):
    name, _, body = text.partition(":")
    parameters = dict(DEFAULTS)
    if body.strip():
        for item in body.split(","):
            item = item.strip()
            if not item:
                continue
            key, _, value = item.partition("=")
            key = key.strip()
            if key not in DEFAULTS:
                raise SystemExit(f"Unknown parameter in grid point: {key}")
            parameters[key] = parseValue(value.strip())
    return name.strip(), parameters


def fmt(value):
    return f"{value:.4f}" if isinstance(value, (int, float)) else "-"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", required=True)
    parser.add_argument("--point", dest="points", action="append", required=True)
    parser.add_argument("--point-scenario", dest="pointScenarios", action="append",
                        default=[],
                        help="restrict a named point to one scenario, e.g. P=dataset_02_pmc")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-root", dest="outRoot", default="outputs/grid")
    parser.add_argument("--record", default=None)
    parser.add_argument("--tag", default="grid")
    arguments = parser.parse_args()

    points = [parsePoint(text) for text in arguments.points]
    pointScenarios = {}
    for text in arguments.pointScenarios:
        name, _, target = text.partition("=")
        if not name.strip() or not target.strip():
            raise SystemExit(f"bad --point-scenario {text!r}; expected NAME=SCENARIO")
        pointScenarios[name.strip()] = target.strip()
    records = []

    header = f"{'point':16}" + "".join(
        f" {name[:22]:>22}" for name in arguments.scenarios
    )
    print(header)
    print("-" * len(header))

    for name, parameters in points:
        defenseConfig = normalizeDefenseConfigDocument(
            {"defense": {"name": "purified_vfl", "parameters": parameters}}
        )
        cells = []
        row = {"point": name, "parameters": parameters, "results": {}}
        allowed = None
        if name in pointScenarios:
            allowed = pointScenarios[name]
        for scenarioName in arguments.scenarios:
            if allowed is not None and scenarioName != allowed:
                continue
            scenarioPath = (
                PROJECT_ROOT / "configs" / "scenarios" / f"{scenarioName}.yaml"
            )
            config = loadConfig(
                scenarioPath,
                {
                    "training": {
                        "epochs": arguments.epochs,
                        "device": arguments.device,
                        "progressEnabled": False,
                    },
                    "output": {
                        "directory": f"{arguments.outRoot}/{arguments.tag}/{name}/{scenarioName}"
                    },
                },
                enforceFormalLock=False,
                defenseConfig=defenseConfig,
            )
            startTime = time.time()
            try:
                summary = runExperiment(config, PROJECT_ROOT)
            except Exception:
                traceback.print_exc()
                cells.append(f"{'FAIL':>22}")
                row["results"][scenarioName] = {"error": True}
                continue
            validation = summary["validationResults"]
            mainTask = validation["mainTask"]["accuracy"]
            backdoor = (validation.get("backdoor") or {}).get("attackSuccessRate")
            privacy = (validation.get("labelInference") or {}).get(
                "labelRecoveryAccuracy"
            )
            row["results"][scenarioName] = {
                "valAcc": mainTask,
                "asr": backdoor,
                "lra": privacy,
                "elapsed": round(time.time() - startTime, 1),
                "bestEpoch": summary["training"]["bestEpoch"],
            }
            parts = [f"acc={fmt(mainTask)}"]
            if backdoor is not None:
                parts.append(f"asr={fmt(backdoor)}")
            if privacy is not None:
                parts.append(f"lra={fmt(privacy)}")
            cells.append(f"{' '.join(parts):>22}")
        records.append(row)
        print(f"{name:16}" + "".join(cells), flush=True)

    if arguments.record:
        recordPath = PROJECT_ROOT / arguments.record
        recordPath.parent.mkdir(parents=True, exist_ok=True)
        with recordPath.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=2)
        print(f"# wrote {arguments.record}")


if __name__ == "__main__":
    main()
