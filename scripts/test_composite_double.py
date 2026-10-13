"""Test double-level aliasing on lfba+amc composites (the previously-failed scenarios).

    <python> scripts/test_composite_double.py
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

BASE = PROJECT_ROOT / "configs" / "defense_purified_reassign_sym.yaml"

RUNS = [
    ("d02comp_dbl", "dataset_02_composite", {"labelAliasFactor": 2, "labelAliasDoubleLevel": 2, "labelAliasSigma": 1.0}),
    ("d02comp_dbl_s05", "dataset_02_composite", {"labelAliasFactor": 2, "labelAliasDoubleLevel": 2, "labelAliasSigma": 0.5}),
    ("d04comp_dbl", "dataset_04_composite", {"labelAliasFactor": 2, "labelAliasDoubleLevel": 2, "labelAliasSigma": 1.0}),
    ("d01comp_dbl", "dataset_01_composite", {"labelAliasFactor": 2, "labelAliasDoubleLevel": 2, "labelAliasSigma": 0.5}),
]


def main():
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    for label, scenario, overrides in RUNS:
        dc = copy.deepcopy(base)
        for key, value in overrides.items():
            dc["defense"]["parameters"][key] = value
        outputDir = PROJECT_ROOT / "outputs" / "probe_dbl_composite" / label
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
