"""运行正式场景、更新结果缓存并生成统一提交 ZIP。"""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.submission_builder import (  # noqa: E402
    DATASET_NAMES,
    SCENARIO_DEFINITIONS,
    normalizeSubmissionIdentity,
    packageCachedSubmission,
    resolveGitState,
    runScenarioToCache,
    scenarioCacheIsReusable,
)
from core.config import loadDefenseConfig, resolveDefenseConfigPath  # noqa: E402


def buildParser():
    parser = argparse.ArgumentParser(
        description=(
            "Run VFL competition scenarios into an incremental cache and "
            "build one validated submission ZIP"
        ),
        allow_abbrev=False,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--run-all",
        dest="runAll",
        action="store_true",
        help="Run every missing or stale formal scenario before packaging",
    )
    mode.add_argument(
        "--package-only",
        dest="packageOnly",
        action="store_true",
        help="Validate and package the existing cache without training",
    )
    parser.add_argument("--dataset", choices=DATASET_NAMES)
    parser.add_argument("--scenario", choices=tuple(SCENARIO_DEFINITIONS))
    parser.add_argument(
        "--defense-config",
        dest="defenseConfig",
        default="configs/defense.yaml",
        help=(
            "Participant defense-only YAML inside configs/ "
            "(default: configs/defense.yaml). NOTE: the repository's own defense "
            "lives in configs/defense_purified.yaml; the bare default selects the "
            "bundled robust_vfl baseline instead, so pass it explicitly when "
            "reproducing the submitted results."
        ),
    )
    parser.add_argument(
        "--results-dir",
        dest="resultsDir",
        default="outputs/submission_cache",
        help="Incremental scenario result cache",
    )
    parser.add_argument(
        "--output",
        help=(
            "Destination ZIP path; defaults to "
            "outputs/<team_id>__<model_version>.zip"
        ),
    )
    parser.add_argument(
        "--team-id",
        dest="teamId",
        help="Competition-platform team identifier written to manifest.json",
    )
    parser.add_argument(
        "--model-version",
        dest="modelVersion",
        help="Team-defined model version written to manifest.json",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="With --run-all, rerun valid cached scenarios too",
    )
    parser.add_argument(
        "--retention",
        type=int,
        choices=(1, 2),
        default=2,
        help=(
            "Number of complete local result generations to keep per "
            "scenario; default: 2 (current and previous)"
        ),
    )
    parser.add_argument(
        "--no-package",
        dest="noPackage",
        action="store_true",
        help="Update the selected cache entries without building a ZIP",
    )
    parser.add_argument(
        "--no-progress",
        dest="noProgress",
        action="store_true",
        help="Disable training and attack progress bars for executed scenarios",
    )
    return parser


def validateArguments(arguments, parser):
    selectedSingleScenario = (
        arguments.dataset is not None or arguments.scenario is not None
    )
    if arguments.packageOnly:
        if selectedSingleScenario:
            parser.error("--package-only cannot be combined with --dataset/--scenario")
        if arguments.force or arguments.noPackage:
            parser.error("--package-only cannot be combined with --force/--no-package")
    elif arguments.runAll:
        if selectedSingleScenario:
            parser.error("--run-all cannot be combined with --dataset/--scenario")
    else:
        if arguments.dataset is None or arguments.scenario is None:
            parser.error(
                "choose --run-all, --package-only, or provide both "
                "--dataset and --scenario"
            )
        if arguments.force:
            parser.error("--force is only needed with --run-all")

    # Warn loudly when the bare default is used.  The repository's own defense is
    # configs/defense_purified.yaml; configs/defense.yaml still points at the bundled
    # robust_vfl baseline, so silently accepting the default would run the wrong
    # defense and quietly invalidate any comparison against the submitted results.
    normalizedDefense = str(arguments.defenseConfig).replace("\\", "/")
    if normalizedDefense == "configs/defense.yaml":
        print(
            "[warning] --defense-config was not given, so the bundled "
            "configs/defense.yaml (robust_vfl baseline) will be used.\n"
            "          To reproduce the results reported in docs/ pass:\n"
            "            --defense-config configs/defense_purified.yaml\n"
            "          Other recorded operating points:\n"
            "            --defense-config configs/defense_purified_villain_gate.yaml\n"
            "            --defense-config configs/defense_purified_nojitter.yaml\n",
            file=sys.stderr,
        )

    if not arguments.noPackage:
        for fieldName, optionName in (
            ("teamId", "--team-id"),
            ("modelVersion", "--model-version"),
        ):
            value = getattr(arguments, fieldName)
            if value is None:
                parser.error(f"{optionName} is required when building a ZIP")
            manifestFieldName = (
                "team_id" if fieldName == "teamId" else "model_version"
            )
            try:
                normalizedValue = normalizeSubmissionIdentity(
                    value,
                    manifestFieldName,
                )
            except ValueError as error:
                parser.error(f"{optionName}: {error}")
            setattr(arguments, fieldName, normalizedValue)

        expectedArchiveName = (
            f"{arguments.teamId}__{arguments.modelVersion}.zip"
        )
        if (
            arguments.output is not None
            and Path(arguments.output).name != expectedArchiveName
        ):
            parser.error(
                "--output filename must match "
                f"<team_id>__<model_version>.zip: {expectedArchiveName}"
            )


def resolvePath(pathValue):
    candidate = Path(pathValue)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def printJson(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def runAllScenarios(
    cacheRoot,
    force,
    retention,
    progressEnabled=None,
    defenseConfig=None,
    defenseConfigPath=None,
):
    completed = []
    reused = []
    currentGitState = resolveGitState(PROJECT_ROOT)
    dataFileHashCache = {}
    for datasetName in DATASET_NAMES:
        for scenarioName in SCENARIO_DEFINITIONS:
            label = f"{datasetName}/{scenarioName}"
            if (
                not force
                and scenarioCacheIsReusable(
                    PROJECT_ROOT,
                    cacheRoot,
                    datasetName,
                    scenarioName,
                    currentGitState=currentGitState,
                    dataFileHashCache=dataFileHashCache,
                    defenseConfig=defenseConfig,
                    defenseConfigPath=defenseConfigPath,
                )
            ):
                print(f"[reuse] {label}", flush=True)
                reused.append(label)
                continue
            print(f"[run] {label}", flush=True)
            runScenarioToCache(
                PROJECT_ROOT,
                cacheRoot,
                datasetName,
                scenarioName,
                retention=retention,
                progressEnabled=progressEnabled,
                dataFileHashCache=dataFileHashCache,
                defenseConfig=defenseConfig,
                defenseConfigPath=defenseConfigPath,
            )
            completed.append(label)
    return {"completed": completed, "reused": reused}


def main():
    parser = buildParser()
    arguments = parser.parse_args()
    validateArguments(arguments, parser)
    defenseConfigPath = resolveDefenseConfigPath(
        PROJECT_ROOT,
        arguments.defenseConfig,
    )
    defenseConfig = loadDefenseConfig(defenseConfigPath)
    cacheRoot = resolvePath(arguments.resultsDir).resolve()
    outputPath = None
    if not arguments.noPackage:
        outputValue = arguments.output or (
            f"outputs/{arguments.teamId}__{arguments.modelVersion}.zip"
        )
        outputPath = resolvePath(outputValue).resolve()
    execution = {"completed": [], "reused": []}

    if arguments.runAll:
        execution = runAllScenarios(
            cacheRoot,
            arguments.force,
            arguments.retention,
            False if arguments.noProgress else None,
            defenseConfig,
            defenseConfigPath,
        )
    elif not arguments.packageOnly:
        label = f"{arguments.dataset}/{arguments.scenario}"
        print(f"[run] {label}", flush=True)
        runScenarioToCache(
            PROJECT_ROOT,
            cacheRoot,
            arguments.dataset,
            arguments.scenario,
            retention=arguments.retention,
            progressEnabled=False if arguments.noProgress else None,
            defenseConfig=defenseConfig,
            defenseConfigPath=defenseConfigPath,
        )
        execution["completed"].append(label)

    result = {
        "cacheDirectory": str(cacheRoot),
        "localResultRetention": arguments.retention,
        "defenseConfig": str(defenseConfigPath),
        "execution": execution,
    }
    if not arguments.noPackage:
        print("[package] validating complete cache", flush=True)
        packageResult = packageCachedSubmission(
            PROJECT_ROOT,
            cacheRoot,
            outputPath,
            arguments.teamId,
            arguments.modelVersion,
            defenseConfig=defenseConfig,
            defenseConfigPath=defenseConfigPath,
        )
        result["submission"] = packageResult["archive"]
        result["validationWarnings"] = packageResult["issues"]
    printJson(result)


if __name__ == "__main__":
    main()
