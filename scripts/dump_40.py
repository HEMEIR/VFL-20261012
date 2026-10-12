"""Dump all 40 scenarios' acc/lra/asr for reassign_sym, then score with the OFFICIAL formula.

    <python> scripts/dump_40.py
"""

import glob
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS = [f"dataset_0{i}" for i in range(1, 6)]
SCENARIOS = ["clean", "lfba_p05", "lfba_p10", "lfba_p20", "villain", "pmc", "amc", "composite"]

# Official A_clean / A_attack constants are lost (outputs/scenarios deleted), so print raw
# values and, where a baseline is recoverable, mark the loss.  A_attack is recovered from the
# earlier score_breakdown output; A_clean is approximated by O's snapshot clean accuracy.
A_CLEAN = {
    "dataset_01": 0.743333, "dataset_02": 0.970063, "dataset_03": 0.975588,
    "dataset_04": 0.852375, "dataset_05": 0.989320,
}
A_ATTACK = {
    ("dataset_01", "amc"): 0.6633, ("dataset_01", "composite"): 0.5525, ("dataset_01", "pmc"): 0.6708,
    ("dataset_02", "amc"): 0.8897, ("dataset_02", "composite"): 0.9453, ("dataset_02", "pmc"): 0.9419,
    ("dataset_03", "amc"): 0.6338, ("dataset_03", "composite"): 0.6447, ("dataset_03", "pmc"): 0.6609,
    ("dataset_04", "amc"): 0.7076, ("dataset_04", "composite"): 0.6951, ("dataset_04", "pmc"): 0.7179,
    ("dataset_05", "amc"): 0.6194, ("dataset_05", "composite"): 0.8214, ("dataset_05", "pmc"): 0.8330,
}


def minimum(d):
    return 1.0 / 6.0 if d == "dataset_05" else 0.5


def clip(v):
    return max(0.0, min(1.0, v))


def load():
    out = {}
    for p in glob.glob(str(PROJECT_ROOT / "outputs" / "three_variants" / "reassign_sym" / "*" / "*" / "metrics.json")):
        parts = Path(p).parts
        v = json.load(open(p, encoding="utf-8"))["validationResults"]
        out[(parts[-3], parts[-2])] = {
            "acc": v["mainTask"]["accuracy"],
            "asr": (v.get("backdoor") or {}).get("attackSuccessRate"),
            "lra": (v.get("labelInference") or {}).get("labelRecoveryAccuracy"),
        }
    return out


def main():
    rec = load()
    print("=== 40 场景 raw 值 ===")
    print("  %-22s %8s %8s %8s" % ("场景", "acc", "lra", "asr"))
    print("  " + "-" * 52)
    for d in DATASETS:
        for s in SCENARIOS:
            r = rec.get((d, s))
            if not r:
                continue
            lra = r["lra"] if isinstance(r["lra"], float) else None
            asr = r["asr"] if isinstance(r["asr"], float) else None
            print("  %-22s %8.4f %8s %8s" % (
                "%s/%s" % (d, s), r["acc"],
                "%.4f" % lra if lra is not None else "-",
                "%.4f" % asr if asr is not None else "-",
            ))

    # Privacy decomposition
    print()
    print("=== 隐私项逐场景（15）===")
    print("  %-22s %8s %8s %7s %7s %7s" % ("场景", "A_attack", "A_rec", "P", "G_p", "Q_p"))
    rows = []
    for d in DATASETS:
        for s in ("pmc", "amc", "composite"):
            r = rec.get((d, s))
            base = A_ATTACK.get((d, s))
            if not r or base is None or not isinstance(r["lra"], float):
                continue
            a = minimum(d)
            p = clip((base - r["lra"]) / (base - a)) if (base - a) > 0 else 0.0
            g = clip((r["acc"] - a) / (0.9 * A_CLEAN[d] - a))
            q = g * p
            rows.append((q, d, s, base, r["lra"], p, g))
    for q, d, s, base, rec_, p, g in sorted(rows):
        tag = "  <== 丢分" if p <= 1e-9 or q < 0.5 else ""
        print("  %-22s %8.4f %8.4f %7.4f %7.4f %7.4f%s" % ("%s/%s" % (d, s), base, rec_, p, g, q, tag))
    mean_qp = sum(r[0] for r in rows) / len(rows)
    print("  mean(Q_p)=%.6f -> S_priv=%.6f" % (mean_qp, 20 * mean_qp))

    # Robustness decomposition
    print()
    print("=== 鲁棒项逐场景（25）===")
    print("  %-22s %8s %8s %7s %7s" % ("场景", "acc", "asr", "G_r", "Q_r"))
    sec = []
    for key in sorted(rec):
        r = rec[key]
        if not isinstance(r["asr"], float):
            continue
        d = key[0]
        a = minimum(d)
        g = clip((r["acc"] - a) / (0.9 * A_CLEAN[d] - a))
        rr = 1.0 - r["asr"]
        q = 2 * g * rr / (g + rr) if (g + rr) > 0 else 0.0
        sec.append((q, key, r["acc"], r["asr"], g))
    for q, key, acc, asr, g in sorted(sec):
        tag = "  <== 丢分" if q < 0.8 else ""
        print("  %-22s %8.4f %8.4f %7.4f %7.4f%s" % ("%s/%s" % key, acc, asr, g, q, tag))
    mean_qr = sum(r[0] for r in sec) / len(sec)
    print("  mean(Q_r)=%.6f -> S_sec=%.6f" % (mean_qr, 20 * mean_qr))


if __name__ == "__main__":
    main()
