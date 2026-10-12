"""Generate the six defence-config files for the experiment matrix.

Each file is candidate O's parameter set with exactly one axis changed, so any score
difference is attributable to that axis alone.

    <python> scripts/make_matrix_configs.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"

HEADER = """defense:
  name: purified_vfl
  parameters:
    # 生成自 scripts/make_matrix_configs.py
    # 基座 = 候选 O（平台实测 43.3135），仅改动本文件标题所述的一个轴。
{params}
"""

BASE = """    trainingRecenterStrength: 0.5
    inferenceRecenterStrength: 1.0
    referenceMomentum: 0.99
    recenterMode: soft
    recenterDetectionThreshold: 3.0

    jitterForAttacks: villain

    normExcessStrength: 1.0
    normExcessFactor: 2.5

    gradientShuffleStrength: 1.0
    gradientShuffleLabelInferenceOnly: true
    gradientShufflePoisoningActiveOnly: true
    gradientShufflePoisoningBackdoors: ["lfba"]
    gradientShufflePoisonedOnly: false
    gradientShuffleBackdoorOnly: false

    partyDropoutRate: 0.2
    partyDropoutLabelInferenceOnly: true

    trainingJitter: 0.5
    jitterPoisonedOnly: true
    jitterWithLabelInference: false
"""

# name -> (extra yaml block, one-line description used in the report)
MATRIX = {
    "matrix_b0": ("", "基线（无裁剪/量化/稀疏化）"),
    "matrix_b1_clip_uniform": (
        "    gradientClipQuantile: 0.9\n",
        "梯度裁剪（统一，分位 0.9）"),
    "matrix_b2_clip_adaptive": (
        "    gradientClipQuantile: 0.9\n"
        "    gradientClipAdaptiveStrength: 1.0\n",
        "梯度裁剪（误差自适应，分位 0.9）"),
    "matrix_b3_quant_uniform": (
        "    gradientQuantizeStep: 0.5\n"
        "    gradientQuantizeUniform: true\n",
        "梯度量化（统一，步长 0.5·RMS）"),
    "matrix_b4_quant_adaptive": (
        "    gradientQuantizeStep: 0.5\n"
        "    gradientQuantizeAdaptiveStrength: 1.0\n",
        "梯度量化（误差自适应，步长 0.5·RMS）"),
    "matrix_b5_sparsify_adaptive": (
        "    gradientSparsifyRate: 0.5\n"
        "    gradientSparsifyAdaptive: true\n",
        "梯度稀疏化（误差自适应，保留 50%）"),
}


def main():
    for name, (extra, description) in MATRIX.items():
        body = BASE
        if extra:
            body = extra + "\n" + body
        path = CONFIG_DIR / f"defense_{name}.yaml"
        path.write_text(HEADER.format(params=body), encoding="utf-8")
        print("wrote %-34s %s" % (path.name, description))


if __name__ == "__main__":
    main()
