"""Run one variant across all 40 scenarios and snapshot its metrics atomically.

Why this exists
---------------
The official pipeline rotates ``outputs/scenarios/<ds>/<sc>/{current,previous}`` and
keeps only two generations, while the submission cache stores prediction files but
NOT ``metrics.json``.  Running several variants in sequence therefore destroys the
earlier variants' per-scenario metrics: after variant N, variant N-2's metrics are
unrecoverable.  That has cost this project its comparison data twice, so the
snapshot has to be part of the run, not an afterthought.

The snapshot is written to ``<snapshot-root>/<variant>/<dataset>/<scenario>/metrics.json``
and is taken immediately after the pipeline returns, before anything else can rotate
the tree.

    python scripts/run_variant_snapshot.py --variant precision_v3
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

VARIANTS = {
    "precision_v3": ("configs/defense_purified.yaml", "precision_v3"),
    "villain_gate": ("configs/defense_purified_villain_gate.yaml", "villain_gate"),
    "nojitter": ("configs/defense_purified_nojitter.yaml", "nojitter"),
    "jitter_villain": (
        "configs/defense_purified_jitter_villain.yaml",
        "jitter_villain",
    ),
    "per_sample": (
        "configs/defense_purified_per_sample.yaml",
        "per_sample",
    ),
    "norm_excess": (
        "configs/defense_purified_norm_excess.yaml",
        "norm_excess",
    ),
    "g": (
        "configs/defense_purified_g.yaml",
        "g",
    ),
    "h": (
        "configs/defense_purified_h.yaml",
        "h",
    ),
    "i": (
        "configs/defense_purified_i.yaml",
        "i",
    ),
    "j": (
        "configs/defense_purified_j.yaml",
        "j",
    ),
    "k": (
        "configs/defense_purified_k.yaml",
        "k",
    ),
    "l": (
        "configs/defense_purified_l.yaml",
        "l",
    ),
    "m": (
        "configs/defense_purified_m.yaml",
        "m",
    ),
    "n": (
        "configs/defense_purified_n.yaml",
        "n",
    ),
    "o": (
        "configs/defense_purified_o.yaml",
        "o",
    ),
    "b0": (
        "configs/defense_matrix_b0.yaml",
        "b0",
    ),
    "b1_clip_uniform": (
        "configs/defense_matrix_b1_clip_uniform.yaml",
        "b1_clip_uniform",
    ),
    "b2_clip_adaptive": (
        "configs/defense_matrix_b2_clip_adaptive.yaml",
        "b2_clip_adaptive",
    ),
    "b3_quant_uniform": (
        "configs/defense_matrix_b3_quant_uniform.yaml",
        "b3_quant_uniform",
    ),
    "b4_quant_adaptive": (
        "configs/defense_matrix_b4_quant_adaptive.yaml",
        "b4_quant_adaptive",
    ),
    "b5_sparsify_adaptive": (
        "configs/defense_matrix_b5_sparsify_adaptive.yaml",
        "b5_sparsify_adaptive",
    ),
    "p": (
        "configs/defense_purified_p.yaml",
        "p",
    ),
    "softcenter": (
        "configs/defense_purified_softcenter.yaml",
        "softcenter",
    ),
    "reassign_sym": (
        "configs/defense_purified_reassign_sym.yaml",
        "reassign_sym",
    ),
}


def snapshotMetrics(snapshotRoot):
    """Copy every scenario's current metrics; returns the number copied."""
    import glob

    count = 0
    for path in sorted(
        glob.glob(str(PROJECT_ROOT / "outputs/scenarios/*/*/current/metrics.json"))
    ):
        parts = Path(path).parts
        dataset, scenario = parts[-4], parts[-3]
        destination = snapshotRoot / dataset / scenario
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination / "metrics.json")
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=list(VARIANTS))
    parser.add_argument("--snapshot-root", dest="snapshotRoot",
                        default="outputs/three_variants")
    parser.add_argument("--cache-root", dest="cacheRoot", default=None)
    parser.add_argument("--team-id", dest="teamId", default="team001")
    parser.add_argument("--no-package", dest="noPackage", action="store_true")
    arguments = parser.parse_args()

    defenseConfig, version = VARIANTS[arguments.variant]
    cacheRoot = arguments.cacheRoot or f"outputs/cache_{arguments.variant}"
    snapshotRoot = PROJECT_ROOT / arguments.snapshotRoot / arguments.variant

    existing = sorted(snapshotRoot.glob("*/*/metrics.json"))
    if len(existing) >= 40:
        print(f"[skip] {arguments.variant}: snapshot already complete "
              f"({len(existing)} scenarios)")
        return

    command = [
        sys.executable, "scripts/build_submission.py",
        "--run-all", "--no-progress",
        "--defense-config", defenseConfig,
        "--results-dir", cacheRoot,
        "--team-id", arguments.teamId,
        "--model-version", version,
    ]
    if arguments.noPackage:
        command.append("--no-package")

    logPath = PROJECT_ROOT / "outputs" / "three_logs" / f"{arguments.variant}.log"
    logPath.parent.mkdir(parents=True, exist_ok=True)
    with logPath.open("w", encoding="utf-8") as handle:
        code = subprocess.run(command, cwd=str(PROJECT_ROOT), stdout=handle,
                              stderr=subprocess.STDOUT, text=True).returncode

    # Snapshot even on failure: whatever completed is still useful comparison data.
    count = snapshotMetrics(snapshotRoot)
    print(f"[{arguments.variant}] pipeline exit={code}, snapshotted {count} scenarios")
    summary = PROJECT_ROOT / arguments.snapshotRoot / "snapshot_source.json"
    record = {}
    if summary.is_file():
        record = json.loads(summary.read_text(encoding="utf-8"))
    record[arguments.variant] = {"exitCode": code, "scenarios": count,
                                 "config": defenseConfig}
    summary.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                       encoding="utf-8")


if __name__ == "__main__":
    main()
