"""Run each auxiliary-sample variant ONLY on the scenario whose IDs it carries.

The previous launcher passed every scenario to every grid point, so a point carrying
dataset_02's auxiliary IDs also ran dataset_03 and dataset_05 - 25 cells per variant of
which only the 5 diagonal ones are meaningful, and each off-diagonal cell silently targeted
the wrong samples because the ``TRAIN_#####`` ID namespace is shared across datasets.

This launcher instead issues one grid invocation per variant whose points are
``<point>:<base parameters>`` and then maps each point to exactly ONE scenario through
``--point-scenario``, so the run cost is 5 cells per variant instead of 25.

    <python> scripts/run_auxiliary_pilots.py            # all variants
    <python> scripts/run_auxiliary_pilots.py V_sideA    # one variant
"""

import csv
import hashlib
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

BASE = (
    "gradientShuffleStrength=1.0,gradientShuffleLabelInferenceOnly=1,"
    "gradientShufflePoisoningActiveOnly=1,gradientShufflePoisoningBackdoors=lfba,"
    "jitterForAttacks=villain,normExcessStrength=1.0,normExcessFactor=2.5,"
    "partyDropoutRate=0.2"
)

SCENARIOS = [
    ("dataset_02_pmc", "dataset_02"),
    ("dataset_03_pmc", "dataset_03"),
    ("dataset_02_amc", "dataset_02"),
    ("dataset_05_pmc", "dataset_05"),
    ("dataset_03_villain", "dataset_03"),
]

VARIANTS = {
    "W_base": "",
    "W_sideA": "auxiliaryExclude=1,auxiliaryExcludeSide=a",
    "W_sideB": "auxiliaryExclude=1,auxiliaryExcludeSide=b",
    "W_sideAB": "auxiliaryExclude=1,auxiliaryExcludeSide=ab",
    "W_overfit3": "auxiliaryOverfitStrength=3.0",
    "W_overfit9": "auxiliaryOverfitStrength=9.0",
    "W_lossmax2": "auxiliaryLossMaxStrength=2.0",
}


def auxiliaryIds(dataset):
    path = PROJECT_ROOT / "configs" / "auxiliary_labels" / f"{dataset}_pmc.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [row["ID"] for row in csv.DictReader(handle) if row.get("ID")]


def run(tag, extra):
    if not extra:
        # A single baseline point covering all five scenarios at once.
        command = [PYTHON, "scripts/grid.py",
                   "--scenarios", *[n for n, _ in SCENARIOS],
                   "--epochs", "100",
                   "--point", f"T_base:{BASE}",
                   "--tag", tag, "--record", f"outputs/grid/{tag}.json"]
    else:
        # --scenarios is REQUIRED by grid.py even when --point-scenario narrows every point
        # to a single scenario; omitting it made all six intervention variants fail
        # immediately with "the following arguments are required: --scenarios".
        command = [PYTHON, "scripts/grid.py",
                   "--scenarios", *[n for n, _ in SCENARIOS],
                   "--epochs", "100"]
        for scenario, dataset in SCENARIOS:
            ids = ";".join(auxiliaryIds(dataset))
            body = f"{BASE},{extra},auxiliarySampleIdList={ids}"
            command += ["--point", f"T_{scenario}:{body}"]
            # Exactly one scenario per point: this is what keeps the IDs aligned.
            command += ["--point-scenario", f"T_{scenario}={scenario}"]
        command += ["--tag", tag, "--record", f"outputs/grid/{tag}.json"]
    print(f"[{tag}] started {time.strftime('%H:%M:%S')}")
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)


def main():
    defensePath = PROJECT_ROOT / "defense" / "purified_vfl.py"
    print("defense/purified_vfl.py")
    print("  sha256   : %s" % hashlib.sha256(defensePath.read_bytes()).hexdigest()[:16])
    print("  modified : %s"
          % time.strftime("%H:%M:%S", time.localtime(defensePath.stat().st_mtime)))
    print("  run start: %s" % time.strftime("%H:%M:%S"))
    print("  (a run that started BEFORE 'modified' used the previous code)")
    print()
    requested = sys.argv[1:] or list(VARIANTS)
    for tag in requested:
        if tag not in VARIANTS:
            print("unknown variant", tag)
            continue
        run(tag, VARIANTS[tag])


if __name__ == "__main__":
    main()
