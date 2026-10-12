"""Calibrated mapping from the local official-formula score to the platform score.

The local formula reproduces the published rules but is NOT the platform's own
computation: it uses validation-split metrics, while the platform scores hidden-test
predictions against private labels.  Four submissions give the empirical mapping:

    variant          formula    platform     residual
    precision_v3     36.858     37.9599      +1.102
    k                41.928     40.6421      -1.286
    h                42.409     41.6341      -0.775
    i                41.994     41.07465     -0.919

A least-squares line through those points is

    platform = 0.6090 * formula + 15.4837        mean |residual| = 0.187

against 1.054 for the raw formula, so this correction is worth roughly a factor of 5.6
in prediction accuracy.  The slope below 1 means the platform's score range is
COMPRESSED relative to the formula: a 1-point formula gain is worth only about 0.61
points on the platform, which is why component changes worth 0.4-0.5 in the formula
have repeatedly failed to move the leaderboard.

    python scripts/calibration.py --formula 42.409
"""

import argparse

SLOPE = 0.7196
INTERCEPT = 11.1888
CALIBRATION_POINTS = [
    ("precision_v3", 36.858, 37.9599),
    ("k", 41.928, 40.6421),
    ("h", 42.409, 41.6341),
    ("i", 41.994, 41.07465),
    ("o", 43.420, 43.313455),
    # Candidate p: local formula is the HIGHEST (44.068) yet the platform scored it
    # BELOW o and below h.  Kept in the table because it is the counterexample that
    # bounds how the formula may be used.
    ("p", 44.068, 41.840633),
    # cc_ramp100: the class-collapse mechanism.  Its platform breakdown was published, and
    # it is the key calibration datum: the PERFORMANCE and SECURITY components transfer
    # almost exactly (ratios 0.994 and 0.990) while the PRIVACY component does not
    # (ratio 0.810).  Both p and cc_ramp100 inflate the local formula by about 2.1 points
    # and both are mechanisms that retune the LOCAL label-recovery rate directly.
    ("cc_ramp100", 45.009091, 42.796321),
]


def toPlatform(formulaValue):
    return SLOPE * formulaValue + INTERCEPT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formula", type=float, default=None,
                        help="local formula total to convert to a platform estimate")
    arguments = parser.parse_args()

    print("标定直线:  平台分 = %.4f x 公式分 + %.4f" % (SLOPE, INTERCEPT))
    print()
    print("%-14s %9s %11s %10s %9s" % ("变体", "公式", "平台实测", "拟合预测", "残差"))
    total = 0.0
    for name, formula, platform in CALIBRATION_POINTS:
        predicted = toPlatform(formula)
        total += abs(platform - predicted)
        print("%-14s %9.3f %11.5f %10.3f %+9.3f"
              % (name, formula, platform, predicted, platform - predicted))
    print()
    print("拟合后 绝对误差均值 = %.3f   （原始公式为 1.054）"
          % (total / len(CALIBRATION_POINTS)))
    print()
    print("含义：平台的分数范围被压缩到公式的 %.1f%%，" % (SLOPE * 100))
    print("      即 **公式增加 1.00 分 ≈ 平台增加 %.2f 分**。" % SLOPE)
    print("      要让平台 +1 分，需要公式 +%.2f 分。" % (1.0 / SLOPE))

    if arguments.formula is not None:
        print()
        print("公式 %.3f  ->  预估平台分 %.3f" % (arguments.formula,
                                                toPlatform(arguments.formula)))


if __name__ == "__main__":
    main()
