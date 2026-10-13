"""Test specific scenarios with parameter overrides (dropout off for d01/composite, sigma sweep for d03/amc).

    <python> scripts/test_param_tuning.py
"""

import copy
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import loadConfig  # noqa: E402
from core.experiment import runExperiment  # noqa: E402

BASE_CONFIG = PROJECT_ROOT / "configs" / "defense_purified_reassign_sym.yaml"

# (label, scenario, {param overrides to apply to the defense parameters})
RUNS = [
    ("d01comp_base", "dataset_01_composite", {}),
    ("d01comp_drop0", "dataset_01_composite", {"partyDropoutRate": 0.0}),
    ("d01comp_dropScale0", "dataset_01_composite", {"partyDropoutScale": 0.0}),
    ("d03amc_sig01", "dataset_03_amc", {"labelAliasSigma": 0.1}),
    ("d03amc_sig03", "dataset_03_amc", {"labelAliasSigma": 0.3}),
    ("d03amc_sig05", "dataset_03_amc", {"labelAliasSigma": 0.5}),
]


def main():
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    for label, scenario, overrides in RUNS:
        dc = copy.deepcopy(base)
        for key, value in overrides.items():
            dc["defense"]["parameters"][key] = value
        outputDir = PROJECT_ROOT / "outputs" / "probe_tuning" / label
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
            print(f"[done] {label}: acc={acc:.4f} lra={lra} asr={asr}")
        except Exception as error:  # noqa: BLE001
            print(f"[fail] {label}: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
