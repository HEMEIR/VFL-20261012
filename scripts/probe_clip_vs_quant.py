"""Demonstrate the mathematical difference between gradient clipping and quantization.

Both take a per-sample scale, but clipping is a multiplicative, selective rescale that
preserves direction, while quantization is an additive-grid rounding that changes
direction. This prints the operation on one concrete gradient row so the difference is
visible rather than asserted.

    <python> scripts/probe_clip_vs_quant.py
"""

import sys

import torch

sys.path.insert(0, '.')
from defense import buildPartyADefense

torch.manual_seed(7)
row = torch.tensor([0.004, -0.012, 0.030, -0.002, 0.011, -0.025, 0.001, 0.018])
gradient = row.reshape(1, -1).repeat(4, 1)
gradient[1] *= 3.0          # row 1: a large-norm sample
gradient[2] *= 0.1          # row 2: a small-norm sample

clip = buildPartyADefense({'name': 'purified_vfl',
                           'parameters': {'gradientClipQuantile': 0.5,
                                          'gradientClipNorm': 0.03}})
quant = buildPartyADefense({'name': 'purified_vfl',
                            'parameters': {'gradientQuantizeStep': 1.0,
                                           'gradientQuantizeUniform': True}})

clipped = clip._clipGradients(gradient)
quantized = quant._adaptiveQuantize(gradient, 1.0, 0.0, 0.9, uniform=True)


def show(name, tensor):
    print("  %-14s" % name, " ".join("%+8.5f" % v for v in tensor[0].tolist()))


print("输入梯度（第 0 行；第 1 行放大 3 倍，第 2 行缩小 10 倍）")
show("原始", gradient)
show("裁剪后", clipped)
show("量化后", quantized)
print()

print("关键量的对比")
print("  %-14s %10s %10s %10s %10s" % ("方法", "行范数比", "方向余弦", "非零个数", "被改坐标数"))
for name, out in (("裁剪", clipped), ("量化", quantized)):
    norms_in = gradient.norm(dim=1)
    norms_out = out.norm(dim=1)
    ratio = (norms_out / norms_in.clamp_min(1e-12)).tolist()
    cos = torch.nn.functional.cosine_similarity(out, gradient, dim=1).tolist()
    nonzero = int((out != 0).sum(dim=1).min().item())
    changed = int((out != gradient).sum().item())
    print("  %-14s %10s %10s %10d %10d"
          % (name,
             " ".join("%.3f" % r for r in ratio),
             " ".join("%.4f" % c for c in cos),
             nonzero, changed))
print()
print("裁剪：只按比例缩小超限的行，方向余弦恒为 1.0000（方向不变）")
print("量化：把每个坐标吸附到网格，方向余弦小于 1（方向被改变）")
