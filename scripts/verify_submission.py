"""Independently verify a submission ZIP: scenario coverage, file hashes, and distinctness.

Recomputes every prediction file's SHA-256 against the manifest rather than trusting the
builder's own report, and checks that two packages are genuinely different rather than
accidental copies.

    <python> scripts/verify_submission.py outputs/team001__cc_ramp100.zip
    <python> scripts/verify_submission.py --compare a.zip b.zip
"""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def inspect(path):
    archive = zipfile.ZipFile(path)
    manifest = json.loads(archive.read("manifest.json"))
    scenarios = len(manifest["scenarios"])
    mismatches = []
    total = 0
    for record in manifest["scenarios"]:
        dataset = record["dataset"]
        scenario = record["scenario"]
        for name, info in record["predictionFiles"].items():
            member = f"{dataset}/{scenario}/{name}"
            try:
                data = archive.read(member)
            except KeyError:
                mismatches.append((member, "missing"))
                continue
            total += 1
            digest = hashlib.sha256(data).hexdigest()
            if digest != info["sha256"]:
                mismatches.append((member, "sha mismatch"))
    return {
        "path": path,
        "scenarios": scenarios,
        "files": total,
        "mismatches": mismatches,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "modelVersion": manifest.get("model_version"),
        "teamId": manifest.get("team_id"),
        "scenarioCount": manifest.get("scenarioCount"),
        "fileCount": manifest.get("predictionFileCount"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packages", nargs="+")
    parser.add_argument("--compare", nargs=2, default=None)
    arguments = parser.parse_args()

    for name in arguments.packages:
        path = Path(name)
        if not path.is_file():
            print("  %-44s MISSING" % path.name)
            continue
        result = inspect(path)
        status = "OK" if not result["mismatches"] else "**问题**"
        print("  %-32s 场景 %2d  文件 %3d  SHA不符 %d  %s"
              % (path.name, result["scenarios"], result["files"],
                 len(result["mismatches"]), status))
        print("      model_version=%s  team_id=%s  scenarioCount=%s  fileCount=%s"
              % (result["modelVersion"], result["teamId"],
                 result["scenarioCount"], result["fileCount"]))
        for member, why in result["mismatches"][:5]:
            print("        问题: %s (%s)" % (member, why))

    if arguments.compare:
        left, right = Path(arguments.compare[0]), Path(arguments.compare[1])
        a, b = zipfile.ZipFile(left), zipfile.ZipFile(right)
        differing = 0
        compared = 0
        for record in json.loads(a.read("manifest.json"))["scenarios"]:
            dataset, scenario = record["dataset"], record["scenario"]
            for name in record["predictionFiles"]:
                member = f"{dataset}/{scenario}/{name}"
                try:
                    compared += 1
                    if a.read(member) != b.read(member):
                        differing += 1
                except KeyError:
                    differing += 1
        print()
        print("  对照 %s vs %s：比较 %d 个文件，其中 %d 个不同"
              % (left.name, right.name, compared, differing))


if __name__ == "__main__":
    main()
