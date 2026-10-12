"""Measure when the reassignment actually fires, per scenario type and epoch.

The claim being checked: in composite scenarios the reassignment gate leaves the WARM-UP
phase (epoch 0) uncovered, which would explain why 4 of 5 composite scenarios score almost
no privacy despite carrying headroom (dataset_05/composite alone has 0.6547 of it).

Wraps the defence's cut-gradient hook and records, for each call, whether the shuffle
strength ended up non-zero - so the gate's real behaviour is measured rather than inferred
from reading it.

    <python> scripts/probe_shuffle_gate.py
"""

import collections
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import loadConfig  # noqa: E402
import core.experiment as experiment  # noqa: E402
from defense import PurifiedVflDefense  # noqa: E402

SCENARIOS = ["dataset_05_composite", "dataset_02_composite", "dataset_02_pmc"]

records = collections.defaultdict(list)
originalHook = PurifiedVflDefense.cutGradientHook


def wrappedHook(self, gradientB, passiveCut=None, labels=None, sampleIds=None,
                epoch=0, batchIndex=0, context=None, generator=None):
    before = self.gradientShuffleStrength
    result = originalHook(self, gradientB, passiveCut=passiveCut, labels=labels,
                          sampleIds=sampleIds, epoch=epoch, batchIndex=batchIndex,
                          context=context, generator=generator)
    changed = not (result is gradientB)
    if gradientB is not None and result is not None and result.shape == gradientB.shape:
        changed = bool((result != gradientB).any().item())
    scenario = getattr(self, "_probeScenario", "?")
    records[scenario].append((int(epoch), str((context or {}).get("backdoor")),
                              str((context or {}).get("labelInference")), changed))
    return result


PurifiedVflDefense.cutGradientHook = wrappedHook


def main():
    engine_build = experiment.buildPartyADefense

    def spy(config):
        defense = engine_build(config)
        if defense is not None:
            defense._probeScenario = currentScenario[0]
        return defense

    experiment.buildPartyADefense = spy
    experiment.saveCheckpoint = lambda path, models, state: Path(path)

    for scenario in SCENARIOS:
        currentScenario[0] = scenario
        config = loadConfig(
            PROJECT_ROOT / "configs" / "scenarios" / f"{scenario}.yaml",
            {"training": {"epochs": 3, "device": "cuda", "progressEnabled": False},
             "output": {"directory": f"outputs/probe_gate/{scenario}"}},
            enforceFormalLock=False,
            defenseConfig={"defense": {"name": "purified_vfl", "parameters": {
                "gradientShuffleStrength": 1.0,
                "gradientShuffleLabelInferenceOnly": True,
                "gradientShufflePoisoningActiveOnly": True,
                "gradientShufflePoisoningBackdoors": ["lfba"],
                "partyDropoutRate": 0.2,
            }}},
        )
        try:
            experiment.runExperiment(config, PROJECT_ROOT)
        except Exception as error:  # noqa: BLE001
            print("  %-24s run error: %s" % (scenario, type(error).__name__))

    print()
    for scenario in SCENARIOS:
        entries = records.get(scenario) or []
        if not entries:
            print("  %-24s 无记录" % scenario)
            continue
        byEpoch = collections.defaultdict(lambda: [0, 0])
        for epoch, _, _, changed in entries:
            byEpoch[epoch][0] += 1
            if changed:
                byEpoch[epoch][1] += 1
        print("  === %s ===" % scenario)
        for epoch in sorted(byEpoch):
            total, fired = byEpoch[epoch]
            print("    epoch %d: 调用 %5d  重指派生效 %5d  (%.1f%%)"
                  % (epoch, total, fired, 100.0 * fired / total))
        print()


currentScenario = ["?"]

if __name__ == "__main__":
    main()
