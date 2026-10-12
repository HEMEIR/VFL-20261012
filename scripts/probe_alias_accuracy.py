"""Prove that aliasing ALL samples preserves the task, by inverting and re-scoring.

If the per-sample mapping is a bijection, the top model trained on aliases must score the
SAME accuracy once ``inverseMapPredictions`` is applied.  This compares:

  * baseline accuracy (no aliasing)
  * accuracy computed against ALIAS labels, with no inverse   -> expected near chance
  * accuracy computed against TRUE labels after the inverse   -> expected equal to baseline

The third number is the one that matters: it is the accuracy A actually reports, and it
should match the baseline exactly rather than merely "not collapse".

    <python> scripts/probe_alias_accuracy.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from core.config import loadConfig  # noqa: E402
from core.data import loadBundle  # noqa: E402
from core.evaluator import predictModels  # noqa: E402
import core.experiment as experiment  # noqa: E402
from defense import buildPartyADefense  # noqa: E402

SCENARIOS = ["dataset_02_pmc", "dataset_05_pmc"]


def trueLabels(path):
    import csv
    import io
    out = {}
    with io.open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            out[str(row["ID"])] = int(row["Label"])
    return out


def main():
    for scenarioName in SCENARIOS:
        config = loadConfig(
            PROJECT_ROOT / "configs" / "scenarios" / f"{scenarioName}.yaml",
            {"training": {"epochs": 3, "device": "cuda", "progressEnabled": False},
             "output": {"directory": f"outputs/probe_alias/{scenarioName}"}},
            enforceFormalLock=False,
            defenseConfig={"defense": {"name": "purified_vfl", "parameters": {}}},
        )
        datasetName = config["dataset"]["name"]
        labelPath = PROJECT_ROOT / "data" / "datasets" / datasetName / "validation" / "party_A_validation.csv"
        truth = trueLabels(labelPath)

        for tag, parameters in [
            ("无别名(基线)", {}),
            ("全部别名化", {"pseudoLabelStrength": 1.0,
                            "pseudoLabelClasses": 2 if "05" not in datasetName else 6,
                            "pseudoLabelSeed": 11}),
        ]:
            defense = buildPartyADefense({"name": "purified_vfl", "parameters": parameters})
            captured = {}

            def spy(cfg):
                return defense

            experiment.buildPartyADefense = spy
            experiment.saveCheckpoint = lambda path, models, state: Path(path)
            try:
                summary = experiment.runExperiment(config, PROJECT_ROOT)
            except Exception as error:  # noqa: BLE001
                print("  %-16s %-14s 运行异常: %s: %s"
                      % (scenarioName, tag, type(error).__name__, str(error)[:60]))
                continue

            predictions = summary.get("mainTaskPredictions") or {}
            ids = predictions.get("ids") or []
            values = predictions.get("predictions") or []
            if not ids:
                print("  %-16s %-14s 无预测输出" % (scenarioName, tag))
                continue

            rawCorrect = sum(1 for i, p in zip(ids, values)
                             if truth.get(str(i)) == int(p))
            rawAccuracy = rawCorrect / max(1, len(ids))

            if parameters:
                restored = defense.inverseMapPredictions(values, ids)
                invCorrect = sum(1 for i, p in zip(ids, restored)
                                 if truth.get(str(i)) == int(p))
                invAccuracy = invCorrect / max(1, len(ids))
                print("  %-16s %-14s 原始(对别名)=%.4f   逆映射后(对真实)=%.4f   记录映射 %d"
                      % (scenarioName, tag, rawAccuracy, invAccuracy,
                         len(defense.pseudoSampleMaps)))
            else:
                print("  %-16s %-14s 原始(对真实)=%.4f"
                      % (scenarioName, tag, rawAccuracy))


if __name__ == "__main__":
    main()
