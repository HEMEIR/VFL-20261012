import sys
import torch

sys.path.insert(0, '.')
from defense import buildPartyADefense

d = buildPartyADefense({'name': 'purified_vfl',
                        'parameters': {'gradientQuantizeStep': 0.5,
                                       'gradientQuantizeAdaptiveStrength': 1.0}})
print('  量化幅度诊断（width = step * rms * (1+w)，与梯度 RMS 成正比，故尺度无关）')
print()
for scale, label in [(1.0, 'O(1)'), (1e-4, '约1e-4'), (1e-6, '1e-6')]:
    torch.manual_seed(0)
    g = torch.randn(8, 64) * scale
    out = d._adaptiveQuantize(g, 0.5, 1.0, 0.9)
    rel = float((g - out).abs().max() / g.abs().max())
    print('  %-8s 梯度max=%.3e  RMS=%.3e  最大相对变化=%.4f  %s'
          % (label, float(g.abs().max()), float(g.pow(2).mean().sqrt()), rel,
             '有效' if rel > 0.05 else '几乎无效果'))

print()
print('  对照：直接看 width 与单个坐标的比例')
torch.manual_seed(0)
g = torch.randn(8, 64)
rms = float(g.pow(2).mean().sqrt())
w = d._errorWeights(g, 0.9)
factor = float((1.0 + 1.0 * w).clamp_min(1e-3).mean())
print('    RMS = %.4f, 平均 factor = %.4f, width = %.4f, 坐标典型值 %.4f'
      % (rms, factor, 0.5 * rms * factor, float(g.abs().mean())))
print('    => width 与坐标同量级，量化应当有效。')
