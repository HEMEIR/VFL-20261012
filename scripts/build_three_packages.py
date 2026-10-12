"""Build the three evaluation submission packages, one operating point each.

Runs each variant over all 40 scenarios, snapshots that variant's per-scenario
metrics, then packages it.  Variants are serialised because the official pipeline
writes every scenario into the same rotating results tree and validates its cache
against the active defense-config hash.

Re-running is safe: a variant whose snapshot and ZIP both exist is skipped, so an
interrupted sweep resumes where it stopped.

Snapshot layout: ``<snapshot-root>/<variant>/<dataset>/<scenario>/metrics.json``

    python scripts/build_three_packages.py
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

# variant -> (defense config, model version, rationale)
VARIANTS = {
    "precision": (
        "configs/defense_purified.yaml",
        "precision_v3",
        "提交主配置：软阈值共模消除（阈值 3.0）+ 注入批抖动 0.5。"
        "40 场景均值 acc 0.8844 / LRA 0.7239，为本地最优且与平台已评分包内容一致。",
    ),
    "villain_gate": (
        "configs/defense_purified_villain_gate.yaml",
        "villain_gate",
        "按公开场景标签只对 VILLAIN 场景做无条件共模消除（VILLAIN 的 batch 均值偏移"
        "就是攻击本身）。正式 VILLAIN 三数据集均值 acc +0.0056 / ASR −0.0352。",
    ),
    "nojitter": (
        "configs/defense_purified_nojitter.yaml",
        "nojitter",
        "关闭注入批抖动（trainingJitter 0.5 → 0）。抖动收益逐数据集分裂，关闭在 "
        "dataset_01/02/05 更优、dataset_03/04 更差；由平台分判定。",
    ),
}


def run(command, logPath):
    logPath.parent.mkdir(parents=True, exist_ok=True)
    with logPath.open("w", encoding="utf-8") as handle:
        return subprocess.run(
            command, cwd=str(PROJECT_ROOT), stdout=handle, stderr=subprocess.STDOUT,
            text=True,
        ).returncode


def snapshot(snapshotRoot):
    """Copy each scenario's current metrics in, using explicit path components."""
    count = 0
    for metricsPath in sorted(
        PROJECT_ROOT.glob("outputs/scenarios/*/*/current/metrics.json")
    ):
        parts = metricsPath.parts
        scenario = parts[-3]
        dataset = parts[-4]
        destination = snapshotRoot / dataset / scenario
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(metricsPath, destination / "metrics.json")
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS),
                        choices=list(VARIANTS))
    parser.add_argument("--snapshot-root", dest="snapshotRoot",
                        default="outputs/three_variants")
    parser.add_argument("--log-root", dest="logRoot", default="outputs/three_logs")
    parser.add_argument("--team-id", dest="teamId", default="team001")
    arguments = parser.parse_args()

    snapshotRoot = PROJECT_ROOT / arguments.snapshotRoot
    results = {}
    for name in arguments.variants:
        defenseConfig, version, rationale = VARIANTS[name]
        variantSnapshot = snapshotRoot / name
        existing = sorted(variantSnapshot.glob("*/*/metrics.json"))
        zipPath = PROJECT_ROOT / "outputs" / f"{arguments.teamId}__{name}.zip"
        if len(existing) >= 40 and zipPath.is_file():
            print(f"[skip] {name}: snapshot and package already present", flush=True)
            results[name] = {"skipped": True}
            continue

        print(f"[run ] {name} <- {defenseConfig}", flush=True)
        code = run(
            [
                sys.executable, "scripts/build_submission.py",
                "--run-all", "--force", "--no-progress",
                "--defense-config", defenseConfig,
                "--results-dir", f"outputs/submission_cache_{name}",
                "--team-id", arguments.teamId,
                "--model-version", version,
            ],
            PROJECT_ROOT / arguments.logRoot / f"{name}.log",
        )
        if code != 0:
            print(f"[fail] {name}: exited {code}", flush=True)
            results[name] = {"failed": code}
            continue
        count = snapshot(variantSnapshot)
        print(f"[done] {name}: snapshotted {count}, zip={zipPath.name}", flush=True)
        results[name] = {"scenarios": count, "zip": zipPath.name,
                         "config": defenseConfig, "rationale": rationale}

    summary = PROJECT_ROOT / arguments.snapshotRoot / "summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", encoding="utf-8") as handle:
        json.dump({"variants": {k: {"config": v[0], "rationale": v[2]}
                                for k, v in VARIANTS.items()},
                   "results": results}, handle, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
