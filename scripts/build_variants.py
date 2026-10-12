"""Run several defense variants over all 40 scenarios and package each one.

The official pipeline writes every scenario into the same rotating
``outputs/scenarios/<dataset>/<scenario>/{current,previous}`` tree and validates
its cache against the active defense-config hash, so variants cannot be produced
concurrently.  This driver therefore runs them one after another:

1. run all 40 scenarios with the variant's defense config;
2. snapshot every scenario's metrics into ``<snapshotRoot>/<variant>/``;
3. build the submission ZIP from that variant's cache;
4. move on to the next variant.

Re-running with the same arguments is safe: a variant whose snapshot and ZIP
already exist is skipped, so an interrupted sweep resumes where it stopped.

    python scripts/build_variants.py --snapshot-root outputs/variants
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# name -> (defense config path, rationale shown in the summary)
VARIANTS = {
    "precision": (
        "configs/defense_purified.yaml",
        "accuracy/privacy first: band-gated soft recenter (threshold 3.0). "
        "Best main-task accuracy; trigger cancellation engages on 5 of 6 "
        "(attack, dataset) pairs except dataset_02's VILLAIN.",
    ),
    "balanced": (
        "configs/defense_purified_balanced.yaml",
        "middle point: same mechanism with a tighter detection threshold (2.0), "
        "so more batches are treated as perturbed.",
    ),
    "robust": (
        "configs/defense_purified_robust.yaml",
        "robustness first: unconditional recenter (threshold 0), which cancels "
        "the trigger regardless of magnitude. Expect much lower backdoor ASR and "
        "a large main-task accuracy cost on dataset_02.",
    ),
}


def run(command, logPath):
    logPath.parent.mkdir(parents=True, exist_ok=True)
    with logPath.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return process.returncode


def snapshot(snapshotRoot):
    """Copy each scenario's current metrics into the snapshot directory.

    The metrics file lives at ``outputs/scenarios/<dataset>/<scenario>/current/
    metrics.json``, so the dataset and scenario names are the third- and
    second-from-last path components of that layout.  Reading them from
    ``parents[1]`` / ``parent`` silently collapses every dataset into one
    directory (the dataset level), so the indices are spelled out explicitly here.
    """
    count = 0
    for metricsPath in sorted(
        PROJECT_ROOT.glob("outputs/scenarios/*/*/current/metrics.json")
    ):
        parts = metricsPath.parts
        # ... outputs / scenarios / <dataset> / <scenario> / current / metrics.json
        scenario = parts[-3]
        dataset = parts[-4]
        destination = snapshotRoot / dataset / scenario
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(metricsPath, destination / "metrics.json")
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANTS),
        choices=list(VARIANTS),
    )
    parser.add_argument("--snapshot-root", dest="snapshotRoot", default="outputs/variants")
    parser.add_argument("--team-id", dest="teamId", default="team001")
    parser.add_argument("--log-root", dest="logRoot", default="outputs/variant_logs")
    arguments = parser.parse_args()

    pythonExecutable = sys.executable
    snapshotRoot = PROJECT_ROOT / arguments.snapshotRoot
    logRoot = PROJECT_ROOT / arguments.logRoot
    results = {}

    for name in arguments.variants:
        defenseConfig, rationale = VARIANTS[name]
        variantSnapshot = snapshotRoot / name
        zipPath = PROJECT_ROOT / "outputs" / f"{arguments.teamId}__{name}.zip"
        existing = (
            sorted(variantSnapshot.glob("*/*/metrics.json"))
            if variantSnapshot.is_dir()
            else []
        )
        if len(existing) >= 40 and zipPath.is_file():
            print(f"[skip] {name}: snapshot and package already present", flush=True)
            results[name] = {"skipped": True, "accuracy": None, "asr": None, "lra": None}
            continue

        print(f"[run ] {name} <- {defenseConfig}", flush=True)
        cacheRoot = PROJECT_ROOT / "outputs" / f"variant_cache_{name}"
        code = run(
            [
                pythonExecutable,
                "scripts/build_submission.py",
                "--run-all",
                "--force",
                "--no-progress",
                "--defense-config",
                defenseConfig,
                "--results-dir",
                str(cacheRoot),
                "--team-id",
                arguments.teamId,
                "--model-version",
                name,
            ],
            logRoot / f"{name}.log",
        )
        if code != 0:
            print(f"[fail] {name}: pipeline exited {code}; see {logRoot / (name + '.log')}", flush=True)
            results[name] = {"failed": code, "accuracy": None, "asr": None, "lra": None}
            continue

        count = snapshot(variantSnapshot)
        print(f"[done] {name}: snapshotted {count} scenarios, zip={zipPath.name}", flush=True)
        results[name] = {"scenarios": count, "zip": zipPath.name, "rationale": rationale}

    summaryPath = PROJECT_ROOT / arguments.snapshotRoot / "variants_summary.json"
    summaryPath.parent.mkdir(parents=True, exist_ok=True)
    with summaryPath.open("w", encoding="utf-8") as handle:
        json.dump({"variants": VARIANTS, "results": results}, handle, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
