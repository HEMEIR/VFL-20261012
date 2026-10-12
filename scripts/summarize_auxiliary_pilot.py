"""Summarise one auxiliary-sample pilot: baseline against the intervention, per scenario.

The pilots that matter here are the ones that target the samples B holds public labels for,
with each dataset's own auxiliary IDs (an earlier version passed one fixed file to every
scenario, which silently targeted the wrong samples).

    <python> scripts/summarize_auxiliary_pilot.py --tag N1a
"""

import argparse
import glob
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def loadCell(directory):
    records = {}
    for path in glob.glob(str(directory / "*" / "metrics.json")):
        parts = Path(path).parts
        v = json.load(open(path, encoding="utf-8"))["validationResults"]
        records[parts[-2]] = {
            "acc": float(v["mainTask"]["accuracy"]),
            "lra": (v.get("labelInference") or {}).get("labelRecoveryAccuracy"),
            "asr": (v.get("backdoor") or {}).get("attackSuccessRate"),
        }
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="grid tag, e.g. N1a")
    arguments = parser.parse_args()

    root = PROJECT_ROOT / "outputs" / "grid" / arguments.tag
    if not root.is_dir():
        print("no results for tag", arguments.tag)
        return

    baseDir = root / "BASE"
    if not baseDir.is_dir():
        print("no BASE cell in", root)
        return
    base = loadCell(baseDir)

    print("=" * 96)
    print("grid tag: %s" % arguments.tag)
    print("=" * 96)
    print("  %-24s %18s %18s %10s" % ("场景", "基线 精度/恢复率", "干预 精度/恢复率", "Δ恢复率"))
    print("  " + "-" * 92)

    for cell in sorted(p.name for p in root.iterdir() if p.is_dir()):
        if cell == "BASE":
            continue
        data = loadCell(root / cell)
        for scenario in sorted(base):
            if scenario not in data:
                print("  %-24s %18s %18s %10s"
                      % (scenario, "-", "未完成", "-"))
                continue
            a, b = base[scenario], data[scenario]
            if a["lra"] is not None and b["lra"] is not None:
                marker = "  <- 恢复率下降" if b["lra"] < a["lra"] - 1e-9 else ""
                print("  %-24s %8.4f/%8.4f %8.4f/%8.4f %+10.4f%s"
                      % (scenario, a["acc"], a["lra"], b["acc"], b["lra"],
                         b["lra"] - a["lra"], marker))
            elif a["asr"] is not None and b["asr"] is not None:
                print("  %-24s %8.4f/ASR%6.4f %8.4f/ASR%6.4f   ΔASR %+.4f"
                      % (scenario, a["acc"], a["asr"], b["acc"], b["asr"],
                         b["asr"] - a["asr"]))
            else:
                print("  %-24s %8.4f %18.4f" % (scenario, a["acc"], b["acc"]))
        print()

    print("  说明：Δ恢复率 < 0 表示攻击被削弱（有益）；ΔASR < 0 表示后门被削弱（有益）。")


if __name__ == "__main__":
    main()
