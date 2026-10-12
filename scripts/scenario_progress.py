"""Report how many official scenario runs have completed.

Reads the rotated results under ``outputs/scenarios/<dataset>/<scenario>/current``
and lists what is still missing.

    python scripts/scenario_progress.py
"""

import argparse
import glob
import os

DATASETS = [f"dataset_0{index}" for index in range(1, 6)]
SCENARIOS = [
    "clean",
    "lfba_p05",
    "lfba_p10",
    "lfba_p20",
    "villain",
    "pmc",
    "amc",
    "composite",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.path.join("outputs", "scenarios"))
    arguments = parser.parse_args()

    done = set()
    for path in glob.glob(
        os.path.join(arguments.root, "*", "*", "current", "metrics.json")
    ):
        parts = path.replace("\\", "/").split("/")
        done.add((parts[-4], parts[-3]))

    expected = {(dataset, scenario) for dataset in DATASETS for scenario in SCENARIOS}
    print(f"completed {len(done)}/{len(expected)}")
    for dataset in DATASETS:
        finished = sorted(s for d, s in done if d == dataset)
        missing = sorted(s for d, s in expected - done if d == dataset)
        print(f"  {dataset}: done={len(finished)}/8", end="")
        if finished:
            print(f" ({', '.join(finished)})", end="")
        if missing:
            print(f" | pending: {', '.join(missing)}", end="")
        print()


if __name__ == "__main__":
    main()
