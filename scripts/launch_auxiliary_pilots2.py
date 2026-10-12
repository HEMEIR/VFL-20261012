"""Launch the auxiliary-sample pilots with the CORRECT per-dataset IDs and verified code.

Two defects were found in the earlier attempt and are addressed here:

1. ONE fixed ``auxiliaryLabelsPath`` was passed to every scenario.  The harness reuses one
   defense configuration for all scenarios and the auxiliary CSVs share a ``TRAIN_#####``
   ID namespace across datasets, so the wrong file silently targeted wrong samples.  Fixed
   by reading each dataset's own CSV and passing the IDs explicitly.
2. The B-side exclusion branch did not check ``auxiliaryExcludeSide``, so ``side=a`` removed
   BOTH sides and a genuine A-side-only test was impossible.  Fixed in the defense.

A third, operational defect: the code was edited WHILE a run was in flight, and Python had
already loaded the module, so the run silently used the old code.  This launcher therefore
prints the defense file's modification time and its own start time, so the two can be
compared afterwards.

    <python> scripts/launch_auxiliary_pilots2.py
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

# tag -> extra parameters (the auxiliary ID list is appended per scenario)
VARIANTS = {
    "V_base": "",
    "V_sideA": "auxiliaryExclude=1,auxiliaryExcludeSide=a",
    "V_sideB": "auxiliaryExclude=1,auxiliaryExcludeSide=b",
    "V_sideAB": "auxiliaryExclude=1,auxiliaryExcludeSide=ab",
    "V_overfit3": "auxiliaryOverfitStrength=3.0",
    "V_overfit9": "auxiliaryOverfitStrength=9.0",
    "V_lossmax2": "auxiliaryLossMaxStrength=2.0",
}


def auxiliaryIds(dataset):
    path = PROJECT_ROOT / "configs" / "auxiliary_labels" / f"{dataset}_pmc.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [row["ID"] for row in csv.DictReader(handle) if row.get("ID")]


def run(tag, extra, outName):
    scenarioNames = [n for n, _ in SCENARIOS]
    points = []
    for scenario, dataset in SCENARIOS:
        ids = ";".join(auxiliaryIds(dataset))
        body = BASE
        if extra:
            body += "," + extra
        body += f",auxiliarySampleIdList={ids}"
        points.append((f"T_{scenario}", body))
    command = [PYTHON, "scripts/grid.py", "--scenarios", *scenarioNames, "--epochs", "100"]
    for name, body in points:
        command += ["--point", f"{name}:{body}"]
    # V_base is the shared reference and is produced once; every other variant is compared
    # against it.  It still needs the ID list only if the mechanism needs it, so it gets
    # none and stays a pure baseline.
    if not extra:
        command = [PYTHON, "scripts/grid.py", "--scenarios", *scenarioNames,
                   "--epochs", "100", "--point", f"T_base:{BASE}"]
    command += ["--tag", tag, "--record", f"outputs/grid/{outName}"]
    print(f"[{tag}] {len(command)} args; started at {time.strftime('%H:%M:%S')}")
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)


def main():
    defensePath = PROJECT_ROOT / "defense" / "purified_vfl.py"
    digest = hashlib.sha256(defensePath.read_bytes()).hexdigest()
    print("defense/purified_vfl.py")
    print("  modified: %s" % time.strftime("%H:%M:%S",
                                         time.localtime(defensePath.stat().st_mtime)))
    print("  sha256  : %s" % digest[:16])
    print("  (record this: a run that started BEFORE the modification time used old code)")
    print()
    requested = sys.argv[1:] or list(VARIANTS)
    for tag in requested:
        if tag not in VARIANTS:
            print("unknown variant", tag)
            continue
        run(tag, VARIANTS[tag], f"{tag}.json")


if __name__ == "__main__":
    main()
