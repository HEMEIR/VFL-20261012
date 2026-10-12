"""Regenerate the undefended (robust_vfl) baseline flat layout under outputs/scenarios.

score_breakdown.py reads A_clean/A_attack from outputs/scenarios/<dataset>/<scenario>/metrics.json
(legacy flat layout).  That tree was accidentally deleted, so this regenerates it by running
the bundled robust_vfl baseline (configs/defense.yaml) across all 40 scenarios.

    <python> scripts/regenerate_baseline.py
"""

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import loadConfig  # noqa: E402
from core.experiment import runExperiment  # noqa: E402

DATASETS = [f"dataset_0{i}" for i in range(1, 6)]
SCENARIOS = ["clean", "lfba_p05", "lfba_p10", "lfba_p20", "villain", "pmc", "amc", "composite"]


def main():
    baseline = PROJECT_ROOT / "outputs" / "scenarios"
    # The shipped baseline is configs/defense.yaml (robust_vfl with its full parameters).
    defenseConfig = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "defense.yaml").read_text(encoding="utf-8")
    )
    for dataset in DATASETS:
        for scenario in SCENARIOS:
            scenarioPath = PROJECT_ROOT / "configs" / "scenarios" / f"{dataset}_{scenario}.yaml"
            if not scenarioPath.is_file():
                print(f"[skip] {dataset}/{scenario} (no config)")
                continue
            outputDir = baseline / dataset / scenario
            if (outputDir / "metrics.json").is_file():
                print(f"[exists] {dataset}/{scenario}")
                continue
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
                print(f"[done] {dataset}/{scenario}")
            except Exception as error:  # noqa: BLE001
                print(f"[fail] {dataset}/{scenario}: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
