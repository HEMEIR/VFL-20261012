"""Test lfba+amc composite aliasing: does widening lower LRA without hurting ASR/acc.

    <python> scripts/test_composite_aliasing.py
"""

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import loadConfig  # noqa: E402
from core.experiment import runExperiment  # noqa: E402

SCENARIOS = [
    "dataset_02_composite",  # lfba+amc, sigma 0.1
    "dataset_04_composite",  # lfba+amc, sigma 1.0
    "dataset_01_composite",  # lfba+amc, sigma 0.5
]


def main():
    dc = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "defense_purified_reassign_sym.yaml").read_text(encoding="utf-8")
    )
    for scenario in SCENARIOS:
        outputDir = PROJECT_ROOT / "outputs" / "probe_composite_alias" / scenario
        config = loadConfig(
            PROJECT_ROOT / "configs" / "scenarios" / f"{scenario}.yaml",
            {
                "training": {"epochs": 100, "device": "cuda", "progressEnabled": False},
                "output": {"directory": str(outputDir)},
            },
            enforceFormalLock=False,
            defenseConfig=dc,
        )
        try:
            runExperiment(config, PROJECT_ROOT)
            v = yaml.safe_load((outputDir / "metrics.json").read_text(encoding="utf-8"))["validationResults"]
            acc = v["mainTask"]["accuracy"]
            lra = (v.get("labelInference") or {}).get("labelRecoveryAccuracy")
            asr = (v.get("backdoor") or {}).get("attackSuccessRate")
            print(f"[done] {scenario}: acc={acc:.4f} lra={lra} asr={asr}")
        except Exception as error:  # noqa: BLE001
            print(f"[fail] {scenario}: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
