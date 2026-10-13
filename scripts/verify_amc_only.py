"""Quick-verify amc-only aliasing fix: pmc keeps dropout, amc keeps aliasing.

    <python> scripts/verify_amc_only.py
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
    "dataset_04_pmc",   # expect LRA ~0.239 (dropout restored, no aliasing)
    "dataset_02_amc",   # expect LRA ~0.463 (sigma 0.1 aliasing)
    "dataset_05_pmc",   # expect LRA ~0.650 (no aliasing on pmc)
]


def main():
    defenseConfig = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "defense_purified_reassign_sym.yaml").read_text(encoding="utf-8")
    )
    for scenario in SCENARIOS:
        scenarioPath = PROJECT_ROOT / "configs" / "scenarios" / f"{scenario}.yaml"
        outputDir = PROJECT_ROOT / "outputs" / "probe_amc_only" / scenario
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
            v = yaml.safe_load((outputDir / "metrics.json").read_text(encoding="utf-8"))["validationResults"]
            acc = v["mainTask"]["accuracy"]
            lra = (v.get("labelInference") or {}).get("labelRecoveryAccuracy")
            print(f"[done] {scenario}: acc={acc:.4f} lra={lra}")
        except Exception as error:  # noqa: BLE001
            print(f"[fail] {scenario}: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
