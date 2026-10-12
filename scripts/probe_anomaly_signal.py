"""Measure the batch-mean anomaly signal that the recentering step keys on.

Runs the real pipeline twice per scenario - clean evaluation and triggered
evaluation - and reports, for each, how large the batch-mean deviation is in
units of the learned fluctuation band.  The separation between the two
distributions is what decides whether a soft-thresholded recentering can cancel
a trigger without touching clean batches.

    python scripts/probe_anomaly_signal.py --scenarios dataset_03_clean dataset_03_villain
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core.experiment as experimentModule  # noqa: E402
from core.config import loadConfig, normalizeDefenseConfigDocument  # noqa: E402
from defense import buildPartyADefense  # noqa: E402

# Collect the anomaly ratio for every inference hook call of a run.
RATIOS = {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.0)
    arguments = parser.parse_args()

    defenseConfig = normalizeDefenseConfigDocument(
        {
            "defense": {
                "name": "purified_vfl",
                "parameters": {
                    "trainingRecenterStrength": 0.5,
                    "inferenceRecenterStrength": 1.0,
                    "referenceMomentum": 0.99,
                    "recenterThreshold": arguments.threshold,
                    "gradientClipQuantile": 0.0,
                },
            }
        }
    )

    class Recorder:
        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def inferenceRepresentationHook(self, embeddingA, embeddingB, *a, **k):
            result = self.inner.inferenceRepresentationHook(
                embeddingA, embeddingB, *a, **k
            )
            RATIOS.setdefault("inference", []).append(self.inner.lastAnomalyRatio)
            return result

    for scenarioName in arguments.scenarios:
        RATIOS.clear()
        scenarioPath = PROJECT_ROOT / "configs" / "scenarios" / f"{scenarioName}.yaml"
        config = loadConfig(
            scenarioPath,
            {
                "training": {
                    "epochs": arguments.epochs,
                    "device": arguments.device,
                    "progressEnabled": False,
                },
                "output": {"directory": f"outputs/probe_anomaly/{scenarioName}"},
            },
            enforceFormalLock=False,
            defenseConfig=defenseConfig,
        )

        def spy(defenseConfigArg):
            inner = buildPartyADefense(defenseConfigArg)
            return None if inner is None else Recorder(inner)

        originalBuild = experimentModule.buildPartyADefense
        originalSave = experimentModule.saveCheckpoint
        experimentModule.buildPartyADefense = spy
        experimentModule.saveCheckpoint = lambda path, models, summary: Path(path)
        try:
            summary = experimentModule.runExperiment(config, PROJECT_ROOT)
        finally:
            experimentModule.buildPartyADefense = originalBuild
            experimentModule.saveCheckpoint = originalSave

        values = sorted(RATIOS.get("inference", []))
        validation = summary["validationResults"]
        backdoor = validation.get("backdoor") or {}
        privacy = validation.get("labelInference") or {}
        if values:
            count = len(values)
            median = values[count // 2]
            print(
                f"{scenarioName:26} acc={validation['mainTask']['accuracy']:.4f} "
                f"asr={backdoor.get('attackSuccessRate')} "
                f"lra={privacy.get('labelRecoveryAccuracy')} | "
                f"anomalyRatio min={values[0]:.3f} p25={values[count // 4]:.3f} "
                f"median={median:.3f} max={values[-1]:.3f}",
                flush=True,
            )
        else:
            print(f"{scenarioName:26} no inference calls recorded", flush=True)


if __name__ == "__main__":
    main()
