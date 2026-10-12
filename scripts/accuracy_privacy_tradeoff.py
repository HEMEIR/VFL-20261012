"""How much room is there to trade accuracy for privacy under the official formula?

The published score couples the two: both the privacy and the robustness term use the
SAME accuracy constraint factor

    G_p = G_r = clip((A - A_min) / (0.9 * A_clean - A_min))

so pushing accuracy below 0.9 * A_clean weakens privacy and robustness TOGETHER rather
than converting accuracy into privacy.  This prints candidate h's headroom against that
line per dataset, and the marginal exchange rates.

    python scripts/accuracy_privacy_tradeoff.py
"""

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


def main():
    records = {}
    for path in glob.glob(str(PROJECT_ROOT / "outputs" / "three_variants" / "h"
                              / "*" / "*" / "metrics.json")):
        parts = Path(path).parts
        validation = json.load(open(path, encoding="utf-8"))["validationResults"]
        records[(parts[-3], parts[-2])] = validation["mainTask"]["accuracy"]

    print("H 的精度相对 0.9*A_clean 约束线的余量")
    print()
    for dataset in sorted(CLEAN):
        line = 0.9 * CLEAN[dataset]
        rows = sorted((s, a) for (d, s), a in records.items() if d == dataset)
        below = [(s, a) for s, a in rows if a < line]
        worst = min(rows, key=lambda pair: pair[1])
        if below:
            note = "低于线的场景: " + ", ".join(
                "%s(%.3f)" % (s, a) for s, a in below
            )
        else:
            note = "全部在线以上"
        print("  %-11s 约束线 %.4f  最低 %.4f (%s)  %s"
              % (dataset, line, worst[1], worst[0], note))

    print()
    print("交换率（公式口径）")
    print("  性能项: 单场景归一化精度 -1  ->  -%.3f 分" % (20.0 / 25.0))
    print("  隐私项: 单场景 Q_p +1        ->  +%.3f 分" % (20.0 / 15.0))
    print("  无约束因子时的净收益         =   +%.3f 分/场景"
          % (20.0 / 15.0 - 20.0 / 25.0))
    print()
    print("但 G_p 与 G_r 共用分母 0.9*A_clean，所以：")
    print("  * 精度高于线 -> G_p = G_r = 1，性能项是唯一的精度代价，交易划算；")
    print("  * 精度低于线 -> 隐私项与鲁棒项同时乘上 <1 的因子，")
    print("    性能项的损失之外还多付一份代价，交易很可能不划算。")


if __name__ == "__main__":
    main()
