"""Instrument the alias pipeline to find why predictions were not inverted.

Wraps the trainer's prediction transform and the defence's inverse map, then runs one short
scenario, reporting whether each link in the chain fires.

    <python> scripts/probe_alias_wiring.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import loadConfig  # noqa: E402
import core.experiment as experiment  # noqa: E402
import core.trainer as trainerModule  # noqa: E402

events = []

originalConfigure = trainerModule.VflTrainer.configurePredictionTransform


def patchedConfigure(self, transform=None):
    events.append(("configurePredictionTransform", transform is not None))
    return originalConfigure(self, transform)


originalApply = trainerModule.VflTrainer._applyPredictionTransform


def patchedApply(self, result):
    predictions = (result or {}).get("predictions") or []
    events.append(("apply", self.predictionTransform is not None, len(predictions)))
    return originalApply(self, result)


trainerModule.VflTrainer.configurePredictionTransform = patchedConfigure
trainerModule.VflTrainer._applyPredictionTransform = patchedApply

# Spy on the defence factory to inspect the object actually installed.
originalFactory = experiment.buildPartyADefense


def spyFactory(config):
    defense = originalFactory(config)
    if defense is not None:
        events.append(("built", getattr(defense, "pseudoLabelStrength", None),
                       getattr(defense, "pseudoTrueOf", "missing")))
        originalInverse = defense.inverseMapPredictions

        def wrappedInverse(predictions, sampleIds=None):
            before = list(predictions)[:4]
            out = originalInverse(predictions, sampleIds)
            events.append(("inverse", before, list(out)[:4]))
            return out

        defense.inverseMapPredictions = wrappedInverse
    return defense


experiment.buildPartyADefense = spyFactory
experiment.saveCheckpoint = lambda path, models, state: Path(path)

config = loadConfig(
    PROJECT_ROOT / "configs" / "scenarios" / "dataset_02_pmc.yaml",
    {"training": {"epochs": 2, "device": "cuda", "progressEnabled": False},
     "output": {"directory": "outputs/probe_wiring"}},
    enforceFormalLock=False,
    defenseConfig={"defense": {"name": "purified_vfl",
                               "parameters": {"pseudoLabelStrength": 1.0,
                                              "pseudoLabelSeed": 11}}},
)

try:
    experiment.runExperiment(config, PROJECT_ROOT)
except Exception as error:  # noqa: BLE001
    print("  run error: %s: %s" % (type(error).__name__, str(error)[:80]))

print("  === 事件序列 ===")
for event in events[:40]:
    print("   ", event)
print()
applies = [e for e in events if e[0] == "apply"]
inverses = [e for e in events if e[0] == "inverse"]
print("  _applyPredictionTransform 被调用次数:", len(applies))
print("  inverseMapPredictions 被调用次数   :", len(inverses))
configured = [e for e in events if e[0] == "configurePredictionTransform"]
print("  configurePredictionTransform 安装  :", configured)
print()
if applies and not any(e[1] for e in applies):
    print("  诊断: 应用点被执行，但 self.predictionTransform 为 None -> 未安装成功")
elif applies:
    print("  诊断: 应用点已执行且 transform 非空")
if not applies:
    print("  诊断: 应用点从未执行 -> predict/evaluate 未走到该分支")
