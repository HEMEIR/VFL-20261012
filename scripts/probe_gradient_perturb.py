"""Which structured gradient perturbation actually hides the label?

The label leaks through the gradient because, with hard labels,
    gradientB[i] = softmax(logits_i) - onehot(y_i)
so the MINIMUM entry (argmin) of gradientB[i] is the label dimension y_i.  Zero-mean Gaussian
noise does not change argmin in expectation, which is why it was measured to be inert.

This measures, for several STRUCTURED (direction-changing) perturbations, whether argmin still
recovers the label, AND how much the gradient's useful signal (for B to learn) survives.

Perturbations:
  baseline   gradientB unchanged
  gaussian   zero-mean Gaussian scaled to gradient RMS (the current, measured-inert method)
  signflip   flip the sign of the WHOLE gradient for a fraction p of samples
  labelzero  set the label dimension to 0 (A knows y from the hook's labels)
  recenter   subtract the per-class mean gradient (class centering)

    <python> scripts/probe_gradient_perturb.py
"""

import torch

N, C = 2000, 8  # N samples, C classes


def makeGradients():
    torch.manual_seed(0)
    logits = torch.randn(N, C)
    probs = torch.softmax(logits, dim=1)
    y = torch.randint(0, C, (N,))
    onehot = torch.nn.functional.one_hot(y, C).float()
    grad = probs - onehot  # [N, C]
    return grad, y


def argminRecovery(grad, y):
    return float((grad.argmin(1) == y).float().mean())


def main():
    grad, y = makeGradients()
    print("  %-22s %14s %14s" % ("扰动", "argmin恢复标签", "梯度相关性"))
    print("  " + "-" * 54)

    # baseline
    print("  %-22s %14.4f %14.4f" % ("无扰动", argminRecovery(grad, y),
                                     float(torch.nn.functional.cosine_similarity(grad, grad).mean())))

    # gaussian
    rms = grad.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-12)
    noise = torch.randn(grad.shape) * rms * 0.5
    g = grad + noise
    print("  %-22s %14.4f %14.4f" % ("零均值高斯 scale=0.5", argminRecovery(g, y),
                                     float(torch.nn.functional.cosine_similarity(g, grad).mean())))

    # sign flip
    for p in (0.5, 1.0):
        mask = (torch.rand(N) < p).float().unsqueeze(1)
        g = grad * (1 - 2 * mask)  # +1 or -1
        print("  %-22s %14.4f %14.4f" % ("符号翻转 p=%.1f" % p, argminRecovery(g, y),
                                         float(torch.nn.functional.cosine_similarity(g, grad).mean())))

    # label zero
    g = grad.clone()
    g[torch.arange(N), y] = 0.0
    print("  %-22s %14.4f %14.4f" % ("标签维度置零", argminRecovery(g, y),
                                     float(torch.nn.functional.cosine_similarity(g, grad).mean())))

    # class centering (subtract per-class mean)
    g = grad.clone()
    for c in range(C):
        m = (y == c)
        g[m] = grad[m] - grad[m].mean(dim=0, keepdim=True)
    print("  %-22s %14.4f %14.4f" % ("类中心去除", argminRecovery(g, y),
                                     float(torch.nn.functional.cosine_similarity(g, grad).mean())))

    # label-dimension flip: flip ONLY the label dimension's sign
    g = grad.clone()
    g[torch.arange(N), y] = -g[torch.arange(N), y]
    print("  %-22s %14.4f %14.4f" % ("标签维度取反", argminRecovery(g, y),
                                     float(torch.nn.functional.cosine_similarity(g, grad).mean())))

    print()
    print("  判读：")
    print("  * argmin恢复标签 越低 = 标签越难从梯度读出（目标：随机 1/C=%.3f）" % (1.0/C))
    print("  * 梯度相关性 越高 = B 学习所需的信号保留越多（目标：接近 1）")
    print("  * 需要同时满足：argmin 恢复率接近随机，且相关性高。")


if __name__ == "__main__":
    main()
