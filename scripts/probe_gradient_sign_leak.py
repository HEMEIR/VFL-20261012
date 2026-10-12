"""Does the per-sample cut gradient's sign reveal the label, in THIS harness?

The batch-inference line of work (Liu et al., "Batch Label Inference and Replacement
Attacks in Black-Boxed VFL", arXiv:2112.05409) rests on Lemma 1: for softmax + cross
entropy, the gradient w.r.t. the concatenated logits has the true class's coordinate
sign-flipped relative to the others, so knowing that gradient is equivalent to knowing
the label.

That leads to its CoAE defence (Eq. 15: y_bar = Enc(y; W_e), y_hat = Dec(y_bar; W_d)) which
confuses the label so the sign structure no longer points at the true class.

Before porting any of that, check the premise on our own data: is `sign(cutGradient)`
actually aligned with the label here?  If it is, the leak is real and CoAE is relevant;
the per-sample reassignment already scrambles exactly this channel, so the more useful
question is whether ANY per-sample signal survives it.

    python scripts/probe_gradient_sign_leak.py --scenario dataset_03_pmc
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core.experiment as experimentModule  # noqa: E402
from core.config import loadConfig, loadDefenseConfig, resolveDefenseConfigPath  # noqa: E402

STATE = {"calls": 0, "agree": 0, "total": 0, "perClass": Counter(), "samples": []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="dataset_03_pmc")
    parser.add_argument("--defense", default="configs/defense_purified_h.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()

    config = loadConfig(
        PROJECT_ROOT / "configs" / "scenarios" / f"{arguments.scenario}.yaml",
        {
            "training": {"epochs": arguments.epochs, "device": arguments.device,
                         "progressEnabled": False},
            "output": {"directory": f"outputs/probe_sign/{arguments.scenario}"},
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
            STATE["calls"] += 1
            # Inspect the gradient A would return BEFORE the defence rewrites it.
            if STATE["calls"] <= 40 and labels is not None and gradientB.dim() >= 2:
                flat = gradientB.detach().reshape(gradientB.shape[0], -1)
                target = labels.detach().reshape(-1)
                if target.shape[0] == flat.shape[0] and flat.shape[0] > 1:
                    # For a binary task: the mean sign of the row, compared with the label.
                    rowMean = flat.mean(dim=1)
                    for i in range(flat.shape[0]):
                        predicted = 1 if float(rowMean[i]) > 0 else 0
                        actual = int(target[i]) if float(target[i]) in (0.0, 1.0) else None
                        if actual is None:
                            continue
                        STATE["total"] += 1
                        if predicted == actual:
                            STATE["agree"] += 1
                        STATE["perClass"][actual] += 1
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

    print("scenario            :", arguments.scenario)
    print("defence             :", arguments.defense)
    print("hook calls          :", STATE["calls"])
    print("rows examined       :", STATE["total"])
    if STATE["total"]:
        rate = STATE["agree"] / STATE["total"]
        print("row-mean-sign agrees with label: %.4f  (chance = 0.5)" % rate)
        print("=> sign of the returned cut gradient %s the label"
              % ("REVEALS" if rate > 0.6 or rate < 0.4 else "does NOT reveal"))
    print("label distribution of examined rows:", dict(STATE["perClass"]))
    print("error               :", repr(error))


if __name__ == "__main__":
    main()
