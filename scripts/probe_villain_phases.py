"""Does the reassignment fire during VILLAIN's benign warm-up, and does the public
``poisoningActive`` flag separate warm-up batches from attack batches?

The gate ``gradientShuffleLabelInferenceOnly`` deliberately forces
``shuffleStrength = 0`` whenever ``backdoor != "none"``, because enabling the
reassignment inside VILLAIN's pre-attack warm-up made VILLAIN fail to reach
``minimumMainTaskAccuracy`` and aborted the scenario.  That exclusion also blocks the
composite scenarios, which the per-scenario privacy gap now identifies as the largest
remaining opportunity (``dataset_02/composite`` recovery 0.9260 against a 0.9453
baseline, ``dataset_05/composite`` 0.8330 against 0.8214).

Before adding any finer gate, measure what the context actually reports.  ``poisoningActive``
is raised by both backdoor implementations for exactly the batches that contain an
injected sample, so if it is False throughout the warm-up and True only later, it can
separate the two phases without guessing.

    python scripts/probe_villain_phases.py --scenario dataset_03_composite
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core.experiment as experimentModule  # noqa: E402
from core.config import loadConfig, loadDefenseConfig, resolveDefenseConfigPath  # noqa: E402

STATE = {"rows": [], "seen": Counter()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="dataset_03_composite")
    parser.add_argument("--defense", default="configs/defense_purified.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()

    config = loadConfig(
        PROJECT_ROOT / "configs" / "scenarios" / f"{arguments.scenario}.yaml",
        {
            "training": {"epochs": arguments.epochs, "device": arguments.device,
                         "progressEnabled": False},
            "output": {"directory": f"outputs/probe_phases/{arguments.scenario}"},
        },
        enforceFormalLock=False,
        defenseConfig=loadDefenseConfig(
            resolveDefenseConfigPath(PROJECT_ROOT, arguments.defense)
        ),
    )

    innerBuild = experimentModule.buildPartyADefense
    innerSave = experimentModule.saveCheckpoint

    def spy(defenseCfg):
        instance = innerBuild(defenseCfg)
        if instance is None:
            return None
        original = instance.cutGradientHook

        def wrapped(gradientB, passiveCut=None, labels=None, sampleIds=None,
                    epoch=0, batchIndex=0, context=None, generator=None, *a, **k):
            scenarioContext = context or {}
            key = (
                bool(scenarioContext.get("poisoningActive", False)),
                str(scenarioContext.get("phase", "?")),
                str(scenarioContext.get("labelInference", "?"))[:4],
            )
            STATE["seen"][key] += 1
            STATE["rows"].append((int(epoch), int(batchIndex), key[0]))
            return original(gradientB, passiveCut, labels, sampleIds, epoch,
                            batchIndex, context, generator, *a, **k)

        instance.cutGradientHook = wrapped
        return instance

    experimentModule.buildPartyADefense = spy
    experimentModule.saveCheckpoint = lambda path, models, summary: Path(path)
    error = None
    try:
        experimentModule.runExperiment(config, PROJECT_ROOT)
    except Exception as problem:
        error = problem
    finally:
        experimentModule.buildPartyADefense = innerBuild
        experimentModule.saveCheckpoint = innerSave

    print("scenario:", arguments.scenario)
    print("hook calls:", len(STATE["rows"]))
    print("distinct (poisoningActive, phase, labelInference) combinations:")
    for key, count in STATE["seen"].most_common(12):
        print(f"   poisoningActive={key[0]!s:5} phase={key[1]:10} "
              f"labelInference={key[2]:5} calls={count}")
    firstActive = next((i for i, row in enumerate(STATE["rows"]) if row[2]), None)
    print("first poisoningActive call at index:", firstActive)
    if firstActive is not None:
        print("  that call is epoch/batch:",
              STATE["rows"][firstActive][0], STATE["rows"][firstActive][1])
    print("error:", repr(error))


if __name__ == "__main__":
    main()
