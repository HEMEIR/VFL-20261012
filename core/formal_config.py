"""Deterministic protocol locks for formal scenario configurations.

The lock covers the resolved, effective configuration rather than the YAML
source bytes.  Operational paths, presentation-only runtime controls and the
participant defense section are deliberately excluded; every other current or
future field is locked by default.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path


FORMAL_CONFIG_LOCK_SCHEMA_VERSION = 1
FORMAL_CONFIG_LOCK_FILE_NAME = "formal_config_lock.json"
FORMAL_CONFIG_EXCLUDED_PATHS = (
    "configPath",
    "dataset.dataRoot",
    "defense",
    "output",
    "training.device",
    "training.progressEnabled",
    "training.progressMinInterval",
)

_EXCLUDED_ROOT_FIELDS = {"configPath", "defense", "output"}
_EXCLUDED_NESTED_FIELDS = {
    "dataset": {"dataRoot"},
    "training": {"device", "progressEnabled", "progressMinInterval"},
}


def canonicalJsonBytes(value, pretty=False):
    """Serialize JSON-compatible data with stable ordering and no NaN values."""
    options = {
        "ensure_ascii": False,
        "sort_keys": True,
        "allow_nan": False,
    }
    if pretty:
        options["indent"] = 2
        return (json.dumps(value, **options) + "\n").encode("utf-8")
    options["separators"] = (",", ":")
    return json.dumps(value, **options).encode("utf-8")


def buildLockedConfigProjection(config):
    """Return the formal-protocol portion of one resolved configuration."""
    if not isinstance(config, dict):
        raise TypeError("Formal configuration must be a mapping")
    projection = deepcopy(config)
    for fieldName in _EXCLUDED_ROOT_FIELDS:
        projection.pop(fieldName, None)
    for sectionName, excludedFields in _EXCLUDED_NESTED_FIELDS.items():
        section = projection.get(sectionName)
        if isinstance(section, dict):
            for fieldName in excludedFields:
                section.pop(fieldName, None)
    return projection


def calculateFormalConfigHash(config):
    projection = buildLockedConfigProjection(config)
    return hashlib.sha256(canonicalJsonBytes(projection)).hexdigest()


def buildFormalConfigLock(scenarioConfigs):
    """Build a deterministic manifest from ``{file_name: resolved_config}``."""
    if not isinstance(scenarioConfigs, dict) or not scenarioConfigs:
        raise ValueError("Formal scenario configurations must be a non-empty mapping")
    records = {}
    for scenarioName in sorted(scenarioConfigs):
        normalizedName = Path(str(scenarioName)).name
        if normalizedName != str(scenarioName):
            raise ValueError(
                "Formal scenario lock keys must be file names without directories"
            )
        if normalizedName in records:
            raise ValueError(f"Duplicate formal scenario lock key: {normalizedName}")
        records[normalizedName] = {
            "sha256": calculateFormalConfigHash(scenarioConfigs[scenarioName]),
        }
    return {
        "schemaVersion": FORMAL_CONFIG_LOCK_SCHEMA_VERSION,
        "algorithm": "sha256",
        "excludedPaths": list(FORMAL_CONFIG_EXCLUDED_PATHS),
        "scenarioCount": len(records),
        "scenarios": records,
    }


def serializeFormalConfigLock(lockManifest):
    return canonicalJsonBytes(lockManifest, pretty=True)


def isFormalScenarioConfigPath(configPath):
    path = Path(configPath).resolve()
    return (
        path.suffix.lower() in {".yaml", ".yml"}
        and path.parent.name == "scenarios"
        and path.parent.parent.name == "configs"
    )


def resolveFormalConfigLockPath(configPath):
    path = Path(configPath).resolve()
    return path.parent.parent / FORMAL_CONFIG_LOCK_FILE_NAME


def readFormalConfigLock(lockPath):
    path = Path(lockPath).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Formal configuration lock file does not exist: {path}"
        )
    try:
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Formal configuration lock is not valid JSON: {path}"
        ) from error
    if not isinstance(manifest, dict):
        raise ValueError("Formal configuration lock root must be a mapping")
    if manifest.get("schemaVersion") != FORMAL_CONFIG_LOCK_SCHEMA_VERSION:
        raise ValueError("Unsupported formal configuration lock schemaVersion")
    if manifest.get("algorithm") != "sha256":
        raise ValueError("Formal configuration lock algorithm must be sha256")
    if manifest.get("excludedPaths") != list(FORMAL_CONFIG_EXCLUDED_PATHS):
        raise ValueError("Formal configuration lock exclusion policy does not match code")
    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, dict) or not scenarios:
        raise ValueError("Formal configuration lock scenarios must be a non-empty mapping")
    if manifest.get("scenarioCount") != len(scenarios):
        raise ValueError("Formal configuration lock scenarioCount is inconsistent")
    for scenarioName, record in scenarios.items():
        if Path(str(scenarioName)).name != str(scenarioName):
            raise ValueError("Formal configuration lock contains an invalid scenario key")
        if not isinstance(record, dict):
            raise ValueError(
                f"Formal configuration lock record must be a mapping: {scenarioName}"
            )
        digest = record.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                f"Formal configuration lock has an invalid sha256: {scenarioName}"
            )
    return manifest


def validateFormalConfigLock(configPath, config, lockPath=None):
    """Fail closed unless a formal scenario matches its checked-in lock entry."""
    scenarioPath = Path(configPath).resolve()
    resolvedLockPath = (
        resolveFormalConfigLockPath(scenarioPath)
        if lockPath is None
        else Path(lockPath).resolve()
    )
    manifest = readFormalConfigLock(resolvedLockPath)
    scenarioName = scenarioPath.name
    record = manifest["scenarios"].get(scenarioName)
    if record is None:
        raise ValueError(
            f"Formal configuration lock has no entry for scenario: {scenarioName}"
        )
    expectedDigest = record["sha256"]
    actualDigest = calculateFormalConfigHash(config)
    if actualDigest != expectedDigest:
        raise ValueError(
            "Formal configuration protocol lock mismatch for "
            f"{scenarioName}: expected {expectedDigest}, got {actualDigest}"
        )
    return actualDigest
