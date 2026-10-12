import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import uuid
import zipfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .submission_schema import (
    BACKDOOR_FILE,
    DATASET_NAMES,
    LABEL_INFERENCE_FILE,
    MAIN_TASK_FILE,
    SCENARIO_DEFINITIONS,
    SUBMISSION_MANIFEST_FILE,
    getExpectedFiles,
    getExpectedRelativeFiles,
    getLabelInferenceColumns,
    getScenarioDefinition,
    isValidSubmissionIdentifier,
)
from .submission_validation import buildIssue, validatePredictionFile

from .config import (
    calculateDefenseConfigHash,
    loadConfig,
    resolveDefenseConfigPath,
)
from .experiment import runExperiment
from .formal_config import (
    calculateFormalConfigHash,
    resolveFormalConfigLockPath,
)


RUN_MANIFEST_FILE = "run_manifest.json"
SUBMISSION_SCHEMA_VERSION = "1.0"


def utcTimestamp():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def calculateFileSha256(filePath):
    digest = hashlib.sha256()
    with Path(filePath).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def calculateConfigHash(config):
    stableConfig = deepcopy(config)
    stableConfig.pop("configPath", None)
    payload = json.dumps(
        stableConfig,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def buildDefenseConfigRecord(projectRoot, defenseConfig, defenseConfigPath):
    if defenseConfig is None and defenseConfigPath is None:
        return {"path": None, "sha256": None}
    if defenseConfig is None or defenseConfigPath is None:
        raise ValueError(
            "defenseConfig and defenseConfigPath must be provided together"
        )
    projectPath = Path(projectRoot).resolve()
    resolvedPath = resolveDefenseConfigPath(projectPath, defenseConfigPath)
    relativePath = resolvedPath.relative_to(projectPath).as_posix()
    return {
        "path": relativePath,
        "sha256": calculateDefenseConfigHash(defenseConfig),
    }


def getScenarioDataFiles(projectRoot, config):
    """Return the exact input files used by one formal scenario.

    Roles, rather than absolute paths, are used as the stable identity of each
    file so that an unchanged dataset keeps the same fingerprint after the
    repository is moved to another directory.
    """
    projectPath = Path(projectRoot).resolve()
    datasetName = str(config["dataset"]["name"]).lower()
    configuredDataRoot = Path(config["dataset"]["dataRoot"])
    if not configuredDataRoot.is_absolute():
        configuredDataRoot = projectPath / configuredDataRoot
    datasetDirectory = (configuredDataRoot / datasetName).resolve()

    relativeDataFiles = (
        "train/party_A_train.csv",
        "train/party_B_train.csv",
        "validation/party_A_validation.csv",
        "validation/party_B_validation.csv",
        "test_hidden/party_A_test.csv",
        "test_hidden/party_B_test.csv",
    )
    files = [
        {
            "role": f"dataset/{relativePath}",
            "path": datasetDirectory / relativePath,
        }
        for relativePath in relativeDataFiles
    ]

    backdoorName = str(config["backdoor"].get("name", "none")).lower()
    if backdoorName != "none":
        for partyName in ("A", "B"):
            relativePath = (
                f"test_backdoor/{backdoorName}/party_{partyName}_test.csv"
            )
            files.append(
                {
                    "role": f"dataset/{relativePath}",
                    "path": datasetDirectory / relativePath,
                }
            )

    labelInferenceName = str(
        config["labelInference"].get("name", "none")
    ).lower()
    if labelInferenceName != "none":
        auxiliaryPath = Path(
            config["labelInference"]["auxiliaryLabelsPath"]
        )
        if not auxiliaryPath.is_absolute():
            auxiliaryPath = (
                Path(config["configPath"]).resolve().parent / auxiliaryPath
            )
        files.append(
            {
                "role": "label_inference/auxiliary_labels.csv",
                "path": auxiliaryPath.resolve(),
            }
        )
    return files


def calculateScenarioDataFingerprint(projectRoot, config, fileHashCache=None):
    """Hash every data file used by a scenario and return an aggregate record."""
    projectPath = Path(projectRoot).resolve()
    hashCache = fileHashCache if fileHashCache is not None else {}
    records = []
    for fileSpec in getScenarioDataFiles(projectPath, config):
        path = Path(fileSpec["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"Scenario data file does not exist: {path}"
            )
        stat = path.stat()
        cacheKey = (str(path), int(stat.st_size), int(stat.st_mtime_ns))
        fileSha256 = hashCache.get(cacheKey)
        if fileSha256 is None:
            fileSha256 = calculateFileSha256(path)
            hashCache[cacheKey] = fileSha256
        try:
            manifestPath = path.relative_to(projectPath).as_posix()
        except ValueError:
            manifestPath = path.as_posix()
        records.append(
            {
                "role": fileSpec["role"],
                "path": manifestPath,
                "sizeBytes": int(stat.st_size),
                "sha256": fileSha256,
            }
        )

    records.sort(key=lambda record: record["role"])
    stableRecords = [
        {
            "role": record["role"],
            "sizeBytes": record["sizeBytes"],
            "sha256": record["sha256"],
        }
        for record in records
    ]
    payload = json.dumps(
        stableRecords,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "files": records,
    }


def calculateCodeSha256(projectRoot):
    projectPath = Path(projectRoot).resolve()
    try:
        result = subprocess.run(
            [
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            cwd=str(projectPath),
            check=True,
            capture_output=True,
        )
        relativePaths = [
            Path(rawPath.decode("utf-8"))
            for rawPath in result.stdout.split(b"\0")
            if rawPath
        ]
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError):
        excludedDirectories = {
            ".git",
            "__pycache__",
            "data",
            "outputs",
            "runs",
            "logs",
            ".venv",
            "venv",
            ".vs",
            ".idea",
        }
        relativePaths = []
        for path in projectPath.rglob("*"):
            if not path.is_file():
                continue
            relativePath = path.relative_to(projectPath)
            if any(part in excludedDirectories for part in relativePath.parts):
                continue
            relativePaths.append(relativePath)

    digest = hashlib.sha256()
    for relativePath in sorted(relativePaths, key=lambda item: item.as_posix()):
        path = projectPath / relativePath
        if not path.is_file():
            continue
        normalizedPath = relativePath.as_posix().encode("utf-8")
        digest.update(len(normalizedPath).to_bytes(8, "big"))
        digest.update(normalizedPath)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    return digest.hexdigest()


def writeJson(filePath, value):
    path = Path(filePath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def readJson(filePath):
    with Path(filePath).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def readOrderedIds(csvPath):
    path = Path(csvPath)
    if not path.is_file():
        raise FileNotFoundError(f"Published ID file does not exist: {path}")
    frame = pd.read_csv(
        path,
        usecols=["ID"],
        dtype={"ID": str},
        keep_default_na=False,
        encoding="utf-8",
    )
    ids = frame["ID"].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError(f"Published ID file contains duplicate IDs: {path}")
    return ids


def getScenarioConfigPath(projectRoot, datasetName, scenarioName):
    return (
        Path(projectRoot)
        / "configs"
        / "scenarios"
        / f"{datasetName}_{scenarioName}.yaml"
    )


def getExpectedPredictionSpecs(
    projectRoot,
    datasetName,
    scenarioName,
    config=None,
    defenseConfig=None,
):
    projectPath = Path(projectRoot).resolve()
    scenarioConfig = config or loadConfig(
        getScenarioConfigPath(projectPath, datasetName, scenarioName),
        defenseConfig=defenseConfig,
    )
    scenarioDefinition = getScenarioDefinition(datasetName, scenarioName)
    configuredBackdoor = str(scenarioConfig["backdoor"]["name"]).lower()
    configuredLabelInference = str(
        scenarioConfig["labelInference"]["name"]
    ).lower()
    expectedBackdoor = scenarioDefinition["backdoor"] or "none"
    expectedLabelInference = scenarioDefinition["labelInference"] or "none"
    if configuredBackdoor != expectedBackdoor:
        raise ValueError(
            f"{datasetName}/{scenarioName} config backdoor is "
            f"{configuredBackdoor}, expected {expectedBackdoor}"
        )
    if configuredLabelInference != expectedLabelInference:
        raise ValueError(
            f"{datasetName}/{scenarioName} config label inference is "
            f"{configuredLabelInference}, expected {expectedLabelInference}"
        )
    dataRoot = (
        projectPath
        / scenarioConfig["dataset"]["dataRoot"]
        / datasetName
    ).resolve()
    specs = {
        MAIN_TASK_FILE: {
            "ids": readOrderedIds(
                dataRoot / "test_hidden" / "party_A_test.csv"
            ),
            "columns": ("ID", "Prediction"),
        }
    }
    backdoorName = scenarioDefinition["backdoor"]
    if backdoorName is not None:
        specs[BACKDOOR_FILE] = {
            "ids": readOrderedIds(
                dataRoot
                / "test_backdoor"
                / backdoorName
                / "party_A_test.csv"
            ),
            "columns": ("ID", "Prediction"),
        }
    labelInferenceName = scenarioDefinition["labelInference"]
    if labelInferenceName is not None:
        labelIdsPath = dataRoot / "test_hidden" / "party_A_test.csv"
        specs[LABEL_INFERENCE_FILE] = {
            "ids": readOrderedIds(labelIdsPath),
            "columns": getLabelInferenceColumns(labelInferenceName),
        }
    return specs


def validateScenarioPredictions(
    predictionDirectory,
    projectRoot,
    datasetName,
    scenarioName,
    config=None,
    defenseConfig=None,
):
    specs = getExpectedPredictionSpecs(
        projectRoot,
        datasetName,
        scenarioName,
        config=config,
        defenseConfig=defenseConfig,
    )
    issues = []
    frames = {}
    for fileName, spec in specs.items():
        frame, fileIssues = validatePredictionFile(
            Path(predictionDirectory) / fileName,
            spec["ids"],
            spec["columns"],
            datasetName,
            scenarioName,
        )
        issues.extend(fileIssues)
        if frame is not None:
            frames[fileName] = frame
    return frames, issues


def resolveGitState(projectRoot):
    projectPath = Path(projectRoot).resolve()
    codeSha256 = calculateCodeSha256(projectPath)
    packagedRevision = os.environ.get("VFL_PACKAGED_SOURCE_REVISION")
    packagedClean = os.environ.get("VFL_PACKAGED_SOURCE_CLEAN")
    if packagedRevision is not None or packagedClean is not None:
        if not packagedRevision or packagedClean not in {"0", "1"}:
            raise ValueError(
                "VFL_PACKAGED_SOURCE_REVISION and VFL_PACKAGED_SOURCE_CLEAN "
                "must be set together; clean must be 0 or 1"
            )
        return {
            "revision": packagedRevision,
            "workingTreeDirty": packagedClean != "1",
            "codeSha256": codeSha256,
        }
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(projectPath),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(projectPath),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return {
            "revision": revision,
            "workingTreeDirty": bool(status),
            "codeSha256": codeSha256,
        }
    except (OSError, subprocess.SubprocessError):
        return {
            "revision": "unavailable",
            "workingTreeDirty": None,
            "codeSha256": codeSha256,
        }


def buildPredictionFileRecords(predictionDirectory, expectedFiles):
    records = {}
    for fileName in expectedFiles:
        path = Path(predictionDirectory) / fileName
        frame = pd.read_csv(path)
        records[fileName] = {
            "sha256": calculateFileSha256(path),
            "rowCount": int(len(frame)),
        }
    return records


def buildRunManifest(
    datasetName,
    scenarioName,
    config,
    configPath,
    predictionDirectory,
    gitState,
    retention,
    dataFingerprint,
    defenseConfigRecord=None,
):
    expectedFiles = getExpectedFiles(scenarioName)
    formalConfigLockPath = resolveFormalConfigLockPath(configPath)
    defenseRecord = defenseConfigRecord or {"path": None, "sha256": None}
    return {
        "schemaVersion": SUBMISSION_SCHEMA_VERSION,
        "dataset": datasetName,
        "scenario": scenarioName,
        "createdAtUtc": utcTimestamp(),
        "seed": int(config["training"]["seed"]),
        "configPath": Path(configPath).relative_to(
            Path(configPath).resolve().parents[2]
        ).as_posix(),
        "configSha256": calculateConfigHash(config),
        "formalConfigSha256": calculateFormalConfigHash(config),
        "formalConfigLockSha256": calculateFileSha256(formalConfigLockPath),
        "defenseConfigPath": defenseRecord["path"],
        "defenseConfigSha256": defenseRecord["sha256"],
        "codeRevision": gitState["revision"],
        "workingTreeDirty": gitState["workingTreeDirty"],
        "codeSha256": gitState["codeSha256"],
        "dataFingerprint": dataFingerprint,
        "localResultRetention": int(retention),
        "predictionFiles": buildPredictionFileRecords(
            predictionDirectory,
            expectedFiles,
        ),
    }


def replaceScenarioCache(preparedDirectory, targetDirectory):
    preparedPath = Path(preparedDirectory).resolve()
    targetPath = Path(targetDirectory).resolve()
    targetPath.parent.mkdir(parents=True, exist_ok=True)
    backupPath = targetPath.parent / (
        f".{targetPath.name}.backup-{uuid.uuid4().hex}"
    )
    hadExistingTarget = targetPath.exists()
    if hadExistingTarget:
        os.replace(str(targetPath), str(backupPath))
    try:
        os.replace(str(preparedPath), str(targetPath))
    except Exception:
        if hadExistingTarget and backupPath.exists() and not targetPath.exists():
            os.replace(str(backupPath), str(targetPath))
        raise
    if backupPath.exists():
        shutil.rmtree(backupPath)


def validateRetention(retention):
    resolvedRetention = int(retention)
    if resolvedRetention not in (1, 2):
        raise ValueError("retention must be 1 or 2")
    return resolvedRetention


def resolveScenarioOutputRoot(projectRoot, formalConfig):
    configuredPath = Path(formalConfig["output"]["directory"])
    if configuredPath.is_absolute():
        return configuredPath.resolve()
    return (Path(projectRoot).resolve() / configuredPath).resolve()


def removeDirectory(directory):
    path = Path(directory)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def rollbackLocalOutputRotation(rotationState):
    currentDirectory = rotationState["currentDirectory"]
    previousDirectory = rotationState["previousDirectory"]
    discardedDirectory = rotationState["discardedDirectory"]
    if rotationState["newCurrentInstalled"] and currentDirectory.exists():
        os.replace(str(currentDirectory), str(discardedDirectory))

    if rotationState["retention"] == 2:
        if (
            rotationState["oldCurrentMovedToPrevious"]
            and previousDirectory.exists()
        ):
            os.replace(str(previousDirectory), str(currentDirectory))
    else:
        backupCurrent = rotationState["backupCurrent"]
        if backupCurrent.exists():
            os.replace(str(backupCurrent), str(currentDirectory))

    backupPrevious = rotationState["backupPrevious"]
    if backupPrevious.exists():
        os.replace(str(backupPrevious), str(previousDirectory))
    removeDirectory(discardedDirectory)


def beginLocalOutputRotation(newCurrentDirectory, scenarioOutputRoot, retention):
    resolvedRetention = validateRetention(retention)
    scenarioRoot = Path(scenarioOutputRoot).resolve()
    scenarioRoot.mkdir(parents=True, exist_ok=True)
    currentDirectory = scenarioRoot / "current"
    previousDirectory = scenarioRoot / "previous"
    transactionId = uuid.uuid4().hex
    rotationState = {
        "retention": resolvedRetention,
        "currentDirectory": currentDirectory,
        "previousDirectory": previousDirectory,
        "backupCurrent": scenarioRoot / f".current.backup-{transactionId}",
        "backupPrevious": scenarioRoot / f".previous.backup-{transactionId}",
        "discardedDirectory": scenarioRoot / f".discarded-{transactionId}",
        "oldCurrentMovedToPrevious": False,
        "newCurrentInstalled": False,
    }
    try:
        if resolvedRetention == 2:
            if currentDirectory.exists():
                if previousDirectory.exists():
                    os.replace(
                        str(previousDirectory),
                        str(rotationState["backupPrevious"]),
                    )
                os.replace(str(currentDirectory), str(previousDirectory))
                rotationState["oldCurrentMovedToPrevious"] = True
        else:
            if currentDirectory.exists():
                os.replace(
                    str(currentDirectory),
                    str(rotationState["backupCurrent"]),
                )
            if previousDirectory.exists():
                os.replace(
                    str(previousDirectory),
                    str(rotationState["backupPrevious"]),
                )

        os.replace(str(newCurrentDirectory), str(currentDirectory))
        rotationState["newCurrentInstalled"] = True
        return rotationState
    except Exception:
        rollbackLocalOutputRotation(rotationState)
        raise


def finalizeLocalOutputRotation(rotationState):
    removeDirectory(rotationState["backupCurrent"])
    removeDirectory(rotationState["backupPrevious"])
    removeDirectory(rotationState["discardedDirectory"])


def runScenarioToCache(
    projectRoot,
    cacheRoot,
    datasetName,
    scenarioName,
    retention=2,
    progressEnabled=None,
    experimentRunner=runExperiment,
    dataFileHashCache=None,
    defenseConfig=None,
    defenseConfigPath=None,
):
    if datasetName not in DATASET_NAMES:
        raise ValueError(f"Unsupported dataset: {datasetName}")
    if scenarioName not in SCENARIO_DEFINITIONS:
        raise ValueError(f"Unsupported scenario: {scenarioName}")
    resolvedRetention = validateRetention(retention)

    projectPath = Path(projectRoot).resolve()
    cachePath = Path(cacheRoot).resolve()
    cachePath.mkdir(parents=True, exist_ok=True)
    cacheStagingParent = cachePath / ".staging"
    cacheStagingParent.mkdir(parents=True, exist_ok=True)
    # tempfile.mkdtemp creates a directory whose ACL the DSH sandbox then locks (mkdir and
    # rmtree inside it fail with WinError 5), so use a plain mkdir with a unique name instead.
    cacheStagingRoot = cacheStagingParent / f"{datasetName}-{scenarioName}-{uuid.uuid4().hex}"
    cacheStagingRoot.mkdir(parents=True, exist_ok=False)
    configPath = getScenarioConfigPath(projectPath, datasetName, scenarioName)
    defenseConfigRecord = buildDefenseConfigRecord(
        projectPath,
        defenseConfig,
        defenseConfigPath,
    )
    formalConfig = loadConfig(configPath, defenseConfig=defenseConfig)
    dataFingerprint = calculateScenarioDataFingerprint(
        projectPath,
        formalConfig,
        fileHashCache=dataFileHashCache,
    )
    scenarioOutputRoot = resolveScenarioOutputRoot(projectPath, formalConfig)
    scenarioOutputRoot.mkdir(parents=True, exist_ok=True)
    localStagingRoot = scenarioOutputRoot / f".staging-{uuid.uuid4().hex}"
    localStagingRoot.mkdir(parents=True, exist_ok=False)
    runOutput = localStagingRoot / "new_current"
    preparedDirectory = cacheStagingRoot / "prepared"
    rotationState = None
    try:
        runOverrides = {"output": {"directory": str(runOutput)}}
        if progressEnabled is not None:
            runOverrides["training"] = {
                "progressEnabled": bool(progressEnabled),
            }
        runConfig = loadConfig(
            configPath,
            runOverrides,
            defenseConfig=defenseConfig,
        )
        summary = experimentRunner(runConfig, projectPath)
        frames, issues = validateScenarioPredictions(
            runOutput,
            projectPath,
            datasetName,
            scenarioName,
            config=formalConfig,
        )
        if issues:
            messages = "; ".join(issue["message"] for issue in issues)
            raise ValueError(
                f"{datasetName}/{scenarioName} prediction validation failed: "
                f"{messages}"
            )
        manifest = buildRunManifest(
            datasetName,
            scenarioName,
            formalConfig,
            configPath,
            runOutput,
            resolveGitState(projectPath),
            resolvedRetention,
            dataFingerprint,
            defenseConfigRecord,
        )
        writeJson(runOutput / RUN_MANIFEST_FILE, manifest)
        preparedDirectory.mkdir(parents=True)
        for fileName in getExpectedFiles(scenarioName):
            shutil.copy2(runOutput / fileName, preparedDirectory / fileName)
        writeJson(preparedDirectory / RUN_MANIFEST_FILE, manifest)
        targetDirectory = cachePath / datasetName / scenarioName
        rotationState = beginLocalOutputRotation(
            runOutput,
            scenarioOutputRoot,
            resolvedRetention,
        )
        try:
            replaceScenarioCache(preparedDirectory, targetDirectory)
        except Exception:
            rollbackLocalOutputRotation(rotationState)
            rotationState = None
            raise
        finalizeLocalOutputRotation(rotationState)
        rotationState = None
        return {
            "summary": summary,
            "manifest": manifest,
            "validatedFiles": sorted(frames),
            "cacheDirectory": str(targetDirectory),
            "localCurrentDirectory": str(scenarioOutputRoot / "current"),
            "localPreviousDirectory": (
                str(scenarioOutputRoot / "previous")
                if (scenarioOutputRoot / "previous").is_dir()
                else None
            ),
            "retention": resolvedRetention,
        }
    finally:
        if rotationState is not None:
            rollbackLocalOutputRotation(rotationState)
        removeDirectory(localStagingRoot)
        removeDirectory(cacheStagingRoot)


def validateRunManifest(
    scenarioDirectory,
    datasetName,
    scenarioName,
    expectedConfigHash,
    expectedFormalConfigHash,
    expectedFormalConfigLockHash,
    currentGitState,
    expectedDataFingerprint,
    expectedDefenseConfigPath=None,
    expectedDefenseConfigHash=None,
):
    directory = Path(scenarioDirectory)
    manifestPath = directory / RUN_MANIFEST_FILE
    issues = []
    if not manifestPath.is_file():
        return None, [
            buildIssue(
                "error",
                "missing_run_manifest",
                f"Cached scenario is missing {RUN_MANIFEST_FILE}",
                datasetName,
                scenarioName,
                RUN_MANIFEST_FILE,
            )
        ]
    try:
        manifest = readJson(manifestPath)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return None, [
            buildIssue(
                "error",
                "invalid_run_manifest",
                f"Could not read cached run manifest: {error}",
                datasetName,
                scenarioName,
                RUN_MANIFEST_FILE,
            )
        ]
    if manifest.get("dataset") != datasetName or manifest.get("scenario") != scenarioName:
        issues.append(
            buildIssue(
                "error",
                "run_manifest_identity_mismatch",
                "Cached run manifest dataset or scenario does not match its directory",
                datasetName,
                scenarioName,
                RUN_MANIFEST_FILE,
            )
        )
    if manifest.get("configSha256") != expectedConfigHash:
        issues.append(
            buildIssue(
                "error",
                "stale_scenario_config",
                "Cached predictions were generated from a different scenario configuration",
                datasetName,
                scenarioName,
                RUN_MANIFEST_FILE,
            )
        )
    if manifest.get("formalConfigSha256") != expectedFormalConfigHash:
        issues.append(
            buildIssue(
                "error",
                "formal_config_lock_mismatch",
                "Cached predictions were generated from a different formal protocol configuration",
                datasetName,
                scenarioName,
                RUN_MANIFEST_FILE,
            )
        )
    if manifest.get("formalConfigLockSha256") != expectedFormalConfigLockHash:
        issues.append(
            buildIssue(
                "error",
                "formal_config_lock_file_mismatch",
                "Cached predictions refer to a different formal configuration lock file",
                datasetName,
                scenarioName,
                RUN_MANIFEST_FILE,
            )
        )
    if manifest.get("defenseConfigPath") != expectedDefenseConfigPath:
        issues.append(
            buildIssue(
                "error",
                "defense_config_path_mismatch",
                "Cached predictions refer to a different defense configuration file",
                datasetName,
                scenarioName,
                RUN_MANIFEST_FILE,
            )
        )
    if manifest.get("defenseConfigSha256") != expectedDefenseConfigHash:
        issues.append(
            buildIssue(
                "error",
                "stale_defense_config",
                "Cached predictions were generated with different defense parameters",
                datasetName,
                scenarioName,
                RUN_MANIFEST_FILE,
            )
        )
    manifestDataFingerprint = manifest.get("dataFingerprint")
    if (
        not isinstance(manifestDataFingerprint, dict)
        or not isinstance(manifestDataFingerprint.get("sha256"), str)
        or not isinstance(manifestDataFingerprint.get("files"), list)
    ):
        issues.append(
            buildIssue(
                "error",
                "missing_data_fingerprint",
                "Cached run manifest has no valid scenario data fingerprint",
                datasetName,
                scenarioName,
                RUN_MANIFEST_FILE,
            )
        )
    elif (
        manifestDataFingerprint["sha256"]
        != expectedDataFingerprint["sha256"]
    ):
        issues.append(
            buildIssue(
                "error",
                "stale_scenario_data",
                "Cached predictions were generated from different scenario data",
                datasetName,
                scenarioName,
                RUN_MANIFEST_FILE,
            )
        )
    manifestRevision = manifest.get("codeRevision")
    currentRevision = currentGitState.get("revision")
    if (
        manifestRevision
        and currentRevision
        and manifestRevision != "unavailable"
        and currentRevision != "unavailable"
        and manifestRevision != currentRevision
    ):
        issues.append(
            buildIssue(
                "warning",
                "different_code_revision",
                (
                    f"Cached scenario uses code revision {manifestRevision}; "
                    f"current revision is {currentRevision}"
                ),
                datasetName,
                scenarioName,
                RUN_MANIFEST_FILE,
            )
        )
    manifestCodeHash = manifest.get("codeSha256")
    currentCodeHash = currentGitState.get("codeSha256")
    if manifestCodeHash is None:
        issues.append(
            buildIssue(
                "warning",
                "missing_code_checksum",
                "Cached run manifest does not record a code checksum",
                datasetName,
                scenarioName,
                RUN_MANIFEST_FILE,
            )
        )
    elif currentCodeHash and manifestCodeHash != currentCodeHash:
        issues.append(
            buildIssue(
                "warning",
                "different_code_checksum",
                "Cached scenario was generated from different source contents",
                datasetName,
                scenarioName,
                RUN_MANIFEST_FILE,
            )
        )
    fileRecords = manifest.get("predictionFiles")
    if not isinstance(fileRecords, dict):
        issues.append(
            buildIssue(
                "error",
                "missing_prediction_checksums",
                "Cached run manifest has no prediction file checksums",
                datasetName,
                scenarioName,
                RUN_MANIFEST_FILE,
            )
        )
        return manifest, issues
    for fileName in getExpectedFiles(scenarioName):
        path = directory / fileName
        record = fileRecords.get(fileName)
        recordIsValid = isinstance(record, dict)
        fileExists = path.is_file()
        if not recordIsValid:
            issues.append(
                buildIssue(
                    "error",
                    "missing_prediction_checksums",
                    f"Cached run manifest has no checksum for {fileName}",
                    datasetName,
                    scenarioName,
                    RUN_MANIFEST_FILE,
                )
            )
        if not fileExists:
            issues.append(
                buildIssue(
                    "error",
                    "missing_prediction_file",
                    f"Cached scenario is missing {fileName}",
                    datasetName,
                    scenarioName,
                    fileName,
                )
            )
        if not recordIsValid or not fileExists:
            continue
        if record.get("sha256") != calculateFileSha256(path):
            issues.append(
                buildIssue(
                    "error",
                    "prediction_checksum_mismatch",
                    f"Cached {fileName} no longer matches its run manifest",
                    datasetName,
                    scenarioName,
                    fileName,
                )
            )
    return manifest, issues


def validateCachedSubmission(
    projectRoot,
    cacheRoot,
    defenseConfig=None,
    defenseConfigPath=None,
):
    projectPath = Path(projectRoot).resolve()
    cachePath = Path(cacheRoot).resolve()
    currentGitState = resolveGitState(projectPath)
    issues = []
    scenarioRecords = []
    dataFileHashCache = {}
    defenseConfigRecord = buildDefenseConfigRecord(
        projectPath,
        defenseConfig,
        defenseConfigPath,
    )
    for datasetName in DATASET_NAMES:
        for scenarioName in SCENARIO_DEFINITIONS:
            scenarioDirectory = cachePath / datasetName / scenarioName
            configPath = getScenarioConfigPath(
                projectPath,
                datasetName,
                scenarioName,
            )
            config = loadConfig(configPath, defenseConfig=defenseConfig)
            dataFingerprint = calculateScenarioDataFingerprint(
                projectPath,
                config,
                fileHashCache=dataFileHashCache,
            )
            frames, predictionIssues = validateScenarioPredictions(
                scenarioDirectory,
                projectPath,
                datasetName,
                scenarioName,
                config=config,
            )
            issues.extend(predictionIssues)
            manifest, manifestIssues = validateRunManifest(
                scenarioDirectory,
                datasetName,
                scenarioName,
                calculateConfigHash(config),
                calculateFormalConfigHash(config),
                calculateFileSha256(resolveFormalConfigLockPath(configPath)),
                currentGitState,
                dataFingerprint,
                defenseConfigRecord["path"],
                defenseConfigRecord["sha256"],
            )
            issues.extend(manifestIssues)
            if (
                manifest is not None
                and not predictionIssues
                and not any(
                    issue["severity"] == "error"
                    for issue in manifestIssues
                )
            ):
                scenarioRecords.append(
                    {
                        "dataset": datasetName,
                        "scenario": scenarioName,
                        "seed": manifest.get("seed"),
                        "configSha256": manifest.get("configSha256"),
                        "formalConfigSha256": manifest.get("formalConfigSha256"),
                        "formalConfigLockSha256": manifest.get(
                            "formalConfigLockSha256"
                        ),
                        "codeRevision": manifest.get("codeRevision"),
                        "workingTreeDirty": manifest.get("workingTreeDirty"),
                        "codeSha256": manifest.get("codeSha256"),
                        "dataFingerprint": manifest.get("dataFingerprint"),
                        "predictionFiles": manifest.get("predictionFiles"),
                        "validatedFiles": sorted(frames),
                    }
                )
    return scenarioRecords, issues


def hasErrors(issues):
    return any(issue.get("severity") == "error" for issue in issues)


def normalizeSubmissionIdentity(value, fieldName):
    normalizedValue = value.strip() if isinstance(value, str) else value
    if not isValidSubmissionIdentifier(normalizedValue):
        raise ValueError(
            f"{fieldName} must use 1-64 ASCII letters, digits, dots, "
            "underscores, or hyphens, and must start with a letter or digit"
        )
    return normalizedValue


def buildSubmissionManifest(scenarioRecords, teamId, modelVersion):
    publicScenarioRecords = []
    for record in scenarioRecords:
        predictionFiles = {
            fileName: {"sha256": fileRecord["sha256"]}
            for fileName, fileRecord in sorted(
                record["predictionFiles"].items()
            )
        }
        publicScenarioRecords.append(
            {
                "dataset": record["dataset"],
                "scenario": record["scenario"],
                "predictionFiles": predictionFiles,
            }
        )
    return {
        "schemaVersion": SUBMISSION_SCHEMA_VERSION,
        "team_id": normalizeSubmissionIdentity(teamId, "team_id"),
        "model_version": normalizeSubmissionIdentity(
            modelVersion,
            "model_version",
        ),
        "datasetCount": len(DATASET_NAMES),
        "scenarioCount": len(publicScenarioRecords),
        "predictionFileCount": sum(
            len(record["predictionFiles"])
            for record in publicScenarioRecords
        ),
        "scenarios": publicScenarioRecords,
    }


def createSubmissionArchive(
    cacheRoot,
    outputPath,
    submissionManifest,
    relativePredictionFiles=None,
):
    cachePath = Path(cacheRoot).resolve()
    destination = Path(outputPath).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    expectedRelativeFiles = sorted(
        relativePredictionFiles or getExpectedRelativeFiles()
    )
    temporaryPath = destination.parent / (
        f".{destination.name}.tmp-{uuid.uuid4().hex}"
    )
    try:
        with zipfile.ZipFile(
            temporaryPath,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            archive.writestr(
                SUBMISSION_MANIFEST_FILE,
                json.dumps(
                    submissionManifest,
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
            )
            for relativePath in expectedRelativeFiles:
                sourcePath = cachePath / Path(relativePath)
                if not sourcePath.is_file():
                    raise FileNotFoundError(
                        f"Submission cache is missing: {relativePath}"
                    )
                archive.write(sourcePath, arcname=relativePath)
        os.replace(str(temporaryPath), str(destination))
    finally:
        if temporaryPath.exists():
            temporaryPath.unlink()
    return {
        "outputPath": str(destination),
        "sha256": calculateFileSha256(destination),
        "predictionFileCount": len(expectedRelativeFiles),
        "archiveFileCount": len(expectedRelativeFiles) + 1,
    }


def packageCachedSubmission(
    projectRoot,
    cacheRoot,
    outputPath,
    teamId,
    modelVersion,
    defenseConfig=None,
    defenseConfigPath=None,
):
    scenarioRecords, issues = validateCachedSubmission(
        projectRoot,
        cacheRoot,
        defenseConfig=defenseConfig,
        defenseConfigPath=defenseConfigPath,
    )
    if hasErrors(issues):
        errorMessages = [
            issue["message"]
            for issue in issues
            if issue.get("severity") == "error"
        ]
        preview = "; ".join(errorMessages[:10])
        remaining = len(errorMessages) - 10
        if remaining > 0:
            preview += f"; ... and {remaining} more errors"
        raise ValueError(f"Submission cache validation failed: {preview}")
    expectedScenarioCount = len(DATASET_NAMES) * len(SCENARIO_DEFINITIONS)
    if len(scenarioRecords) != expectedScenarioCount:
        raise ValueError(
            f"Submission cache must contain {expectedScenarioCount} valid scenarios"
        )
    manifest = buildSubmissionManifest(
        scenarioRecords,
        teamId,
        modelVersion,
    )
    archive = createSubmissionArchive(cacheRoot, outputPath, manifest)
    return {
        "archive": archive,
        "manifest": manifest,
        "issues": issues,
    }


def scenarioCacheIsReusable(
    projectRoot,
    cacheRoot,
    datasetName,
    scenarioName,
    currentGitState=None,
    dataFileHashCache=None,
    defenseConfig=None,
    defenseConfigPath=None,
):
    projectPath = Path(projectRoot).resolve()
    configPath = getScenarioConfigPath(projectPath, datasetName, scenarioName)
    defenseConfigRecord = buildDefenseConfigRecord(
        projectPath,
        defenseConfig,
        defenseConfigPath,
    )
    config = loadConfig(configPath, defenseConfig=defenseConfig)
    dataFingerprint = calculateScenarioDataFingerprint(
        projectPath,
        config,
        fileHashCache=dataFileHashCache,
    )
    gitState = currentGitState or resolveGitState(projectPath)
    scenarioDirectory = Path(cacheRoot) / datasetName / scenarioName
    frames, predictionIssues = validateScenarioPredictions(
        scenarioDirectory,
        projectPath,
        datasetName,
        scenarioName,
        config=config,
    )
    manifest, manifestIssues = validateRunManifest(
        scenarioDirectory,
        datasetName,
        scenarioName,
        calculateConfigHash(config),
        calculateFormalConfigHash(config),
        calculateFileSha256(resolveFormalConfigLockPath(configPath)),
        gitState,
        dataFingerprint,
        defenseConfigRecord["path"],
        defenseConfigRecord["sha256"],
    )
    fatalIssues = predictionIssues + [
        issue
        for issue in manifestIssues
        if issue.get("severity") == "error"
    ]
    cacheIsReusable = (
        manifest is not None
        and len(frames) == len(getExpectedFiles(scenarioName))
        and not fatalIssues
    )
    if not cacheIsReusable:
        return False

    currentDirectory = (
        resolveScenarioOutputRoot(projectPath, config) / "current"
    )
    requiredLocalFiles = {
        "checkpoint.pt",
        "metrics.json",
        "attack_audit.json",
        RUN_MANIFEST_FILE,
        *getExpectedFiles(scenarioName),
    }
    if any(
        not (currentDirectory / fileName).is_file()
        for fileName in requiredLocalFiles
    ):
        return False
    localFrames, localPredictionIssues = validateScenarioPredictions(
        currentDirectory,
        projectPath,
        datasetName,
        scenarioName,
        config=config,
    )
    localManifest, localManifestIssues = validateRunManifest(
        currentDirectory,
        datasetName,
        scenarioName,
        calculateConfigHash(config),
        calculateFormalConfigHash(config),
        calculateFileSha256(resolveFormalConfigLockPath(configPath)),
        gitState,
        dataFingerprint,
        defenseConfigRecord["path"],
        defenseConfigRecord["sha256"],
    )
    localFatalIssues = localPredictionIssues + [
        issue
        for issue in localManifestIssues
        if issue.get("severity") == "error"
    ]
    return (
        localManifest is not None
        and len(localFrames) == len(getExpectedFiles(scenarioName))
        and not localFatalIssues
    )
