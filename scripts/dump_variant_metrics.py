"""Print raw per-scenario metrics for several variants side by side.

Used to verify a comparison before trusting it: an earlier comparison script silently
reported identical numbers for two different candidates, so this dumps the values
straight from each snapshot for direct inspection.

    python scripts/dump_variant_metrics.py --variants h l m --order composite lfba
"""

import argparse
import glob
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=["h", "l", "m"])
    parser.add_argument("--order", default="composite",
                        help="substring filter on the scenario name (empty = all)")
    arguments = parser.parse_args()

    tables = {v: load(v) for v in arguments.variants}
    keys = sorted(set().union(*[set(t) for t in tables.values()]))
    if arguments.order:
        keys = [k for k in keys if arguments.order in k[1]]
    if not keys:
        print("no scenarios matched", arguments.order)
        return

    header = "%-24s" % "scenario"
    for v in arguments.variants:
        header += " %22s" % v
    print(header)
    print("-" * len(header))
    for key in keys:
        row = "%-24s" % ("%s/%s" % key)
        for v in arguments.variants:
            rec = tables[v].get(key)
            if rec is None:
                row += " %22s" % "-"
                continue
            lra = rec["lra"]
            asr = rec["asr"]
            cell = "%7.4f/%s" % (
                rec["acc"],
                ("%7.4f" % lra) if lra is not None else "   --   ",
            )
            if asr is not None:
                cell += "/%6.3f" % asr
            else:
                cell += "/  --  "
            row += " %22s" % cell
        print(row)
    print()
    print("格式: 精度 / 标签恢复率 [/ 后门ASR]")


if __name__ == "__main__":
    main()
