"""Measure the recenter anomaly statistic on a genuinely trained scenario.

``probe_recenter_tradeoff.py`` adds the trigger vector to an already-computed
embedding, which bypasses the passive model's own normalisation and therefore
cannot tell whether the trigger is visible in the real forward pass.  This probe
instead runs the actual pipeline with the shipped defense instrumented, so every
inference-hook call reports the statistics it actually saw.  Inference calls are
grouped by ``context["triggered"]``, which separates the clean pass from the
triggered pass on the same trained model.

    python scripts/probe_recenter_signal.py --scenarios dataset_02_villain dataset_03_villain
"""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core.experiment as experimentModule  # noqa: E402
from core.config import loadConfig, loadDefenseConfig, resolveDefenseConfigPath  # noqa: E402

# (triggered flag, ratio, deviationNorm, correctionNorm, strength)
SAMPLES = []


def buildSpy(innerBuild, holder):
    def spy(defenseConfig):
        instance = innerBuild(defenseConfig)
        holder["defense"] = instance
        if instance is None:
            return None
        original = instance._recenter

        def instrumented(
            representation, strength, phase="train", context=None,
            _o=original, _i=instance
        ):
            result = _o(representation, strength, phase, context=context)
            if (phase == "inference") and strength > 0.0:
                SAMPLES.append(
                    (
                        bool(getattr(_i, "_currentTriggered", False)),
                        float(_i.lastAnomalyRatio),
                        float(_i.lastDeviationNorm),
                        float(_i.lastCorrectionNorm),
                        float(strength),
                    )
                )
            return result

        instance._recenter = instrumented

        originalHook = instance.inferenceRepresentationHook

        def hook(embeddingA, embeddingB, sampleIds=None, context=None, generator=None, _o=originalHook, _i=instance):
            _i._currentTriggered = bool((context or {}).get("triggered", False))
            return _o(embeddingA, embeddingB, sampleIds=sampleIds, context=context, generator=generator)

        instance.inferenceRepresentationHook = hook
        return instance

    return spy


def summarize(label, values):
    if not values:
        print(f"  {label:10} (no calls)")
        return
    ordered = sorted(values)
    n = len(ordered)
    print(
        f"  {label:10} n={n:4d} min={ordered[0]:9.3f} p25={ordered[n // 4]:9.3f} "
        f"median={ordered[n // 2]:9.3f} p75={ordered[3 * n // 4]:9.3f} max={ordered[-1]:9.3f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", required=True)
    parser.add_argument("--defense", default="configs/defense_purified.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()

    innerBuild = experimentModule.buildPartyADefense
    innerSave = experimentModule.saveCheckpoint
    for scenarioName in arguments.scenarios:
        SAMPLES.clear()
        scenarioPath = PROJECT_ROOT / "configs" / "scenarios" / f"{scenarioName}.yaml"
        defenseConfig = loadDefenseConfig(
            resolveDefenseConfigPath(PROJECT_ROOT, arguments.defense)
        )
        config = loadConfig(
            scenarioPath,
            {
                "training": {
                    "epochs": arguments.epochs,
                    "device": arguments.device,
                    "progressEnabled": False,
                },
                "output": {"directory": f"outputs/probe_signal/{scenarioName}"},
            },
            enforceFormalLock=False,
            defenseConfig=defenseConfig,
        )
        holder = {}
        experimentModule.buildPartyADefense = buildSpy(innerBuild, holder)
        experimentModule.saveCheckpoint = lambda path, models, summary: Path(path)
        try:
            summary = experimentModule.runExperiment(config, PROJECT_ROOT)
        finally:
            experimentModule.buildPartyADefense = innerBuild
            experimentModule.saveCheckpoint = innerSave

        validation = summary["validationResults"]
        backdoor = validation.get("backdoor") or {}
        privacy = validation.get("labelInference") or {}
        print(
            f"{scenarioName} | acc={validation['mainTask']['accuracy']:.4f} "
            f"asr={backdoor.get('attackSuccessRate')} "
            f"lra={privacy.get('labelRecoveryAccuracy')}"
        )
        clean = [s[1] for s in SAMPLES if not s[0]]
        triggered = [s[1] for s in SAMPLES if s[0]]
        summarize("clean", clean)
        summarize("triggered", triggered)
        corrections = [s[3] for s in SAMPLES if s[0] and s[3] > 0.0]
        if corrections:
            print(f"  triggered batches with a non-zero correction: {len(corrections)}")
        else:
            print("  triggered batches with a non-zero correction: 0")


if __name__ == "__main__":
    main()
