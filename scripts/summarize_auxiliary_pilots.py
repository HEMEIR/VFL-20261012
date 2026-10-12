"""Summarise the auxiliary-sample pilots: base against every intervention.

Assumes the ``outputs/grid/W_*`` layout produced by ``scripts/run_auxiliary_pilots.py``,
where each variant ran exactly one scenario per point (matched to that scenario's own
auxiliary IDs).

    <python> scripts/summarize_auxiliary_pilots.py
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRID = PROJECT_ROOT / "outputs" / "grid"

SCENARIOS = [
    "dataset_02_pmc",
    "dataset_03_pmc",
    "dataset_02_amc",
    "dataset_05_pmc",
    "dataset_03_villain",
]

# tag -> (label, point name prefix or None for the single-point baseline)
VARIANTS = [
    ("W_base", "基线", "T_base"),
    ("W_sideA", "A 侧剔除（只剔 A）", "T_"),
    ("W_sideB", "B 侧剔除（只剔 B）", "T_"),
    ("W_sideAB", "双侧剔除", "T_"),
    ("W_overfit3", "过拟合 strength=3", "T_"),
    ("W_overfit9", "过拟合 strength=9", "T_"),
    ("W_lossmax2", "最大化损失 strength=2", "T_"),
]


def readCell(tag, scenario, pointIsBase):
    if pointIsBase:
        path = GRID / tag / "T_base" / scenario / "metrics.json"
    else:
        path = GRID / tag / f"T_{scenario}" / scenario / "metrics.json"
    if not path.is_file():
        return None
    v = json.load(open(path, encoding="utf-8"))["validationResults"]
    return {
        "acc": float(v["mainTask"]["accuracy"]),
        "lra": (v.get("labelInference") or {}).get("labelRecoveryAccuracy"),
        "asr": (v.get("backdoor") or {}).get("attackSuccessRate"),
    }


def main():
    base = {s: readCell("W_base", s, True) for s in SCENARIOS}
    missing = [s for s, r in base.items() if r is None]
    if missing:
        print("baseline missing for:", missing)
        return

    for tag, label, prefix in VARIANTS:
        if tag == "W_base":
            continue
        print("=" * 104)
        print("%s    (%s)" % (label, tag))
        print("=" * 104)
        print("  %-22s %20s %20s %11s" % ("场景", "基线 精度/恢复率", "干预 精度/恢复率", "Δ恢复率"))
        print("  " + "-" * 100)
        deltas = []
        for scenario in SCENARIOS:
            a = base[scenario]
            b = readCell(tag, scenario, False)
            if b is None:
                print("  %-22s %20s %20s %11s"
                      % (scenario, "%.4f" % a["acc"], "未完成", "-"))
                continue
            if a["lra"] is not None and b["lra"] is not None:
                delta = b["lra"] - a["lra"]
                deltas.append(delta)
                marker = "  改善" if delta < -1e-9 else ("  恶化" if delta > 1e-9 else "")
                print("  %-22s %9.4f/%9.4f %9.4f/%9.4f %+11.4f%s"
                      % (scenario, a["acc"], a["lra"], b["acc"], b["lra"], delta, marker))
            else:
                print("  %-22s %9.4f/ASR%6.4f %9.4f/ASR%6.4f   ΔASR %+.4f"
                      % (scenario, a["acc"], a["asr"], b["acc"], b["asr"],
                         b["asr"] - a["asr"]))
        if deltas:
            improved = sum(1 for d in deltas if d < -1e-9)
            print("  %-22s 标签推断场景 %d 个：改善 %d，恶化 %d，平均 Δ恢复率 %+.4f"
                  % ("小结", len(deltas), improved, len(deltas) - improved,
                     sum(deltas) / len(deltas)))
        print()


if __name__ == "__main__":
    main()
