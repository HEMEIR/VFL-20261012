"""Compare two variants on the scenarios where the poisoningActive gate differs.

Candidate M extends candidate h by reassigning gradients during the LFBA attack phase of
the composite scenarios, targeting the measured win on dataset_02/composite
(ASR 0.9812 -> 0.0300 with accuracy rising 0.8439 -> 0.8976).  Candidate L applied the
same idea more broadly and lost 1.343 formula points, almost entirely in the performance
term, because it also fired during the LFBA-only scenarios.

This reports, for the composite and LFBA scenarios, whether accuracy moved back above the
0.9 * A_clean constraint line - which matters because that factor gates BOTH the privacy
and the robustness term, so dropping below it is penalised twice.

    python scripts/compare_composite_gate.py --baseline h --candidate m
"""

import argparse
import glob
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN = {
    "dataset_01": 0.7308,
    "dataset_02": 0.9699,
    "dataset_03": 0.9783,
    "dataset_04": 0.8552,
    "dataset_05": 0.9874,
}
MINIMUM = {"dataset_05": 1.0 / 6.0}


def minimum(dataset):
    return MINIMUM.get(dataset, 0.5)


def load(variant):
    records = {}
    for path in glob.glob(str(PROJECT_ROOT / "outputs" / "three_variants" / variant
                              / "*" / "*" / "metrics.json")):
        parts = Path(path).parts
        validation = json.load(open(path, encoding="utf-8"))["validationResults"]
        records[(parts[-3], parts[-2])] = {
            "acc": validation["mainTask"]["accuracy"],
            "asr": (validation.get("backdoor") or {}).get("attackSuccessRate"),
            "lra": (validation.get("labelInference") or {}).get("labelRecoveryAccuracy"),
        }
    return records


def privacyContribution(row, dataset, baselineLra):
    if row.get("lra") is None or baselineLra is None:
        return None
    amin = minimum(dataset)
    span = baselineLra - amin
    attack = max(0.0, min(1.0, (baselineLra - row["lra"]) / span)) if span > 0 else 0.0
    guard = max(0.0, min(1.0, (row["acc"] - amin) / (0.9 * CLEAN[dataset] - amin)))
    return guard * attack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="h")
    parser.add_argument("--candidate", default="m")
    arguments = parser.parse_args()

    baselineLra = {}
    for path in glob.glob(str(PROJECT_ROOT / "outputs" / "scenarios" / "*" / "*"
                              / "metrics.json")):
        parts = Path(path).parts
        validation = json.load(open(path, encoding="utf-8"))["validationResults"]
        baselineLra[(parts[-3], parts[-2])] = (
            validation.get("labelInference") or {}
        ).get("labelRecoveryAccuracy")

    base = load(arguments.baseline)
    cand = load(arguments.candidate)
    if not cand:
        print("no snapshot for", arguments.candidate)
        return

    interesting = [
        key for key in sorted(set(base) & set(cand))
        if key[1] == "composite" or key[1].startswith("lfba")
    ]
    print("%-24s %18s %18s %10s %12s" % ("scenario", arguments.baseline,
                                         arguments.candidate, "dAcc", "dQp"))
    print("-" * 88)
    totalQp = 0.0
    accWins = accLosses = 0
    for key in interesting:
        dataset = key[0]
        line = 0.9 * CLEAN[dataset]
        b, c = base[key], cand[key]
        qb = privacyContribution(b, dataset, baselineLra.get(key))
        qc = privacyContribution(c, dataset, baselineLra.get(key))
        dq = (qc - qb) if (qb is not None and qc is not None) else 0.0
        totalQp += dq
        dacc = c["acc"] - b["acc"]
        if dacc > 0:
            accWins += 1
        elif dacc < 0:
            accLosses += 1
        mark = ""
        if b["acc"] < line <= c["acc"]:
            mark = "  <-- 回到约束线以上"
        elif c["acc"] < line <= b["acc"]:
            mark = "  <-- 跌破约束线"
        print("%-24s %8.4f/%-8.4f %8.4f/%-8.4f %+10.4f %+12.4f%s"
              % ("%s/%s" % key, b["acc"], b["lra"] if b["lra"] is not None else float("nan"),
                 c["acc"], c["lra"] if c["lra"] is not None else float("nan"),
                 dacc, dq, mark))
    print()
    print("精度改善 %d 个场景 / 变差 %d 个场景" % (accWins, accLosses))
    print("Q_p 合计变化 %+.4f  ->  隐私项 %+.2f 公式分"
          % (totalQp, 20.0 * totalQp / 15.0))
    print("（平台分 = 0.6090 * 公式分 + 15.4837，故约 %+.2f 平台分）"
          % (0.6090 * 20.0 * totalQp / 15.0))


if __name__ == "__main__":
    main()
