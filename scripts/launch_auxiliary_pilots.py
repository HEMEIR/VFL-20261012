"""Launch the auxiliary-sample pilots with the CORRECT per-dataset auxiliary IDs.

The earlier pilot passed one fixed ``auxiliaryLabelsPath`` to every scenario, which is
wrong twice over: the harness reuses one defense configuration for all scenarios, and the
auxiliary CSVs share a ``TRAIN_#####`` ID namespace across datasets, so the wrong file
silently selected the wrong samples.  This launcher reads each dataset's own CSV and passes
the IDs explicitly for that dataset's scenarios.

    <python> scripts/launch_auxiliary_pilots.py
"""

import csv
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

BASE = (
    "gradientShuffleStrength=1.0,gradientShuffleLabelInferenceOnly=1,"
    "gradientShufflePoisoningActiveOnly=1,gradientShufflePoisoningBackdoors=lfba,"
    "jitterForAttacks=villain,normExcessStrength=1.0,normExcessFactor=2.5,"
    "partyDropoutRate=0.2"
)

# (scenario, dataset)
SCENARIOS = [
    ("dataset_02_pmc", "dataset_02"),
    ("dataset_03_pmc", "dataset_03"),
    ("dataset_02_amc", "dataset_02"),
    ("dataset_05_pmc", "dataset_05"),
    ("dataset_03_villain", "dataset_03"),
]


def auxiliaryIds(dataset):
    path = PROJECT_ROOT / "configs" / "auxiliary_labels" / f"{dataset}_pmc.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [row["ID"] for row in csv.DictReader(handle) if row.get("ID")]


def run(tag, extraTemplate, outName):
    scenarioNames = [n for n, _ in SCENARIOS]
    points = [("BASE", BASE)]
    for scenario, dataset in SCENARIOS:
        ids = ";".join(auxiliaryIds(dataset))
        extra = extraTemplate.format(ids=ids)
        points.append((f"T_{scenario}", f"{BASE},{extra}"))
    command = [PYTHON, "scripts/grid.py", "--scenarios", *scenarioNames,
               "--epochs", "100"]
    for name, body in points:
        command += ["--point", f"{name}:{body}"]
    command += ["--tag", tag, "--record", f"outputs/grid/{outName}"]
    print(f"[{tag}] launching with {len(points)} points over {len(scenarioNames)} scenarios")
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)


if __name__ == "__main__":
    # A-side first: it is the variant explicitly requested, and also the one whose
    # expected effect is most in question, since B keeps learning these samples either way.
    run("N1a", "auxiliaryExclude=1,auxiliaryExcludeSide=a,auxiliarySampleIdList={ids}",
        "N1a.json")
    run("N3overs", "auxiliaryOverfitStrength=3.0,auxiliarySampleIdList={ids}",
        "N3overs.json")
    run("N2ab", "auxiliaryExclude=1,auxiliaryExcludeSide=ab,auxiliarySampleIdList={ids}",
        "N2ab.json")
    run("M9b", "auxiliaryExclude=1,auxiliarySampleIdList={ids}", "M9b.json")
    run("M8b", "auxiliaryLossMaxStrength=2.0,auxiliarySampleIdList={ids}", "M8b.json")
