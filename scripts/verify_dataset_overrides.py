"""Verify datasetOverrides take effect in real training for the key scenarios.

    <python> scripts/verify_dataset_overrides.py
"""

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import loadConfig  # noqa: E402
from core.experiment import runExperiment  # noqa: E402

KEY_SCENARIOS = [
    "dataset_04_pmc",       # expect LRA ~0.239 (dropout 0.3)
    "dataset_04_amc",       # expect LRA ~0.701 (dropout 0.3)
    "dataset_05_composite",  # expect LRA ~0.683 (villain coverage + warmup 0)
    "dataset_05_villain",    # expect ASR ~0.018 (villain coverage, robustness fine)
    "dataset_03_composite",  # expect LRA ~0.671 (no villain coverage, no abort)
]


def main():
    defenseConfig = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "defense_purified_reassign_sym.yaml").read_text(encoding="utf-8")
    )
    for scenario in KEY_SCENARIOS:
        scenarioPath = PROJECT_ROOT / "configs" / "scenarios" / f"{scenario}.yaml"
        outputDir = PROJECT_ROOT / "outputs" / "probe_ds_override" / scenario
        config = loadConfig(
            scenarioPath,
            {
                "training": {"epochs": 100, "device": "cuda", "progressEnabled": False},
                "output": {"directory": str(outputDir)},
            },
            enforceFormalLock=False,
            defenseConfig=defenseConfig,
        )
        try:
            runExperiment(config, PROJECT_ROOT)
            metrics = yaml.safe_load((outputDir / "metrics.json").read_text(encoding="utf-8"))
            vr = metrics["validationResults"]
            acc = vr["mainTask"]["accuracy"]
            lra = (vr.get("labelInference") or {}).get("labelRecoveryAccuracy")
            asr = (vr.get("backdoor") or {}).get("attackSuccessRate")
            print(f"[done] {scenario}: acc={acc:.4f} lra={lra} asr={asr}")
        except Exception as error:  # noqa: BLE001
            print(f"[fail] {scenario}: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
