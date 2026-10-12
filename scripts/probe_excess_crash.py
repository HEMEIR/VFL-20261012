"""Reproduce the dataset_04/clean non-finite crash and report the actual magnitudes.

The trainer rejects a hook output containing NaN/Inf via ``validateHookTensor``, which
aborts the scenario.  Rather than guessing which arithmetic overflows, this probe runs
the failing scenario with the transform instrumented and prints, at the first bad batch,
the magnitudes of every intermediate so the culprit is identified from data.

    python scripts/probe_excess_crash.py
"""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core.experiment as experimentModule  # noqa: E402
from core.config import loadConfig, loadDefenseConfig, resolveDefenseConfigPath  # noqa: E402

REPORTED = {"count": 0}


def describe(name, tensor):
    if not torch.is_tensor(tensor):
        return f"{name}=<{type(tensor).__name__}>"
    flat = tensor.detach().float().reshape(-1)
    finite = torch.isfinite(flat)
    bad = int((~finite).sum().item())
    return (
        f"{name}: shape={tuple(tensor.shape)} finite={bad == 0} bad={bad} "
        f"max={float(flat[finite].max().item()) if finite.any() else float('nan'):.4g} "
        f"absmax={float(flat[finite].abs().max().item()) if finite.any() else float('nan'):.4g}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="dataset_04_clean")
    parser.add_argument("--defense", default="configs/defense_purified_per_sample.yaml")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()

    scenarioPath = PROJECT_ROOT / "configs" / "scenarios" / f"{arguments.scenario}.yaml"
    defenseConfig = loadDefenseConfig(
        resolveDefenseConfigPath(PROJECT_ROOT, arguments.defense)
    )
    config = loadConfig(
        scenarioPath,
        {
            "training": {
                "epochs": arguments.epochs,
                "device": arguments.device,
                "progressEnabled": False,
            },
            "output": {"directory": f"outputs/probe_crash/{arguments.scenario}"},
        },
        enforceFormalLock=False,
        defenseConfig=defenseConfig,
    )

    innerBuild = experimentModule.buildPartyADefense
    innerSave = experimentModule.saveCheckpoint

    def spy(defenseCfg):
        instance = innerBuild(defenseCfg)
        if instance is None:
            return None
        originalRep = instance._clipPerSampleExcess
        originalGrad = instance.cutGradientHook

        def wrappedRep(representation, strength, factor, _o=originalRep):
            try:
                out = _o(representation, strength, factor)
            except Exception as problem:
                print(f"  REP RAISED {type(problem).__name__}: {problem}", flush=True)
                raise
            if REPORTED["count"] < 6 and not torch.isfinite(
                out.detach().float().reshape(-1)
            ).all():
                print("  REP NON-FINITE:", describe("out", out), flush=True)
            return out

        def wrappedGrad(gradientB, *args, **kwargs):
            incoming = torch.isfinite(
                gradientB.detach().float().reshape(-1)
            ).all().item()
            if not incoming and REPORTED["count"] < 12:
                print("  GRAD INCOMING NON-FINITE:", describe("grad", gradientB),
                      flush=True)
                REPORTED["count"] += 1
            out = originalGrad(gradientB, *args, **kwargs)
            outgoing = torch.isfinite(out.detach().float().reshape(-1)).all().item()
            if not outgoing and REPORTED["count"] < 12:
                print("  GRAD OUTPUT NON-FINITE:", describe("out", out), flush=True)
                REPORTED["count"] += 1
            return out

        instance._clipPerSampleExcess = wrappedRep
        instance.cutGradientHook = wrappedGrad
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
    print("error   :", repr(error))


if __name__ == "__main__":
    main()
