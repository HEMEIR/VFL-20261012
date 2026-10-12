"""Normal-shaped soft labels with random peak positions: learnability vs attacker recovery.

The proposal: for the 6-class case, map each hard label to a NORMAL-shaped soft label (one
prominent peak, falling away to the sides).  The peak position is RANDOM per sample, class
centres are shuffled uniformly, peaks within a class are spread, and peaks across classes
overlap - while A records the peak position so it can invert.

This toy test answers the two properties that must hold SIMULTANEOUSLY:

  A_ACC   A's inverse accuracy (does the model learn enough for A to invert exactly?)
  ATTACK  attacker recovery from the frozen embedding (should approach chance 1/C)

The key design question is whether "random peak position + class-centre weak correlation"
can be learnable for A but not recoverable for an attacker.

    <python> scripts/probe_normal_softlabel.py
"""

import torch

N, WIDTH, EMB, C = 4000, 32, 64, 6
SPLIT = 3000
AUX = 60
M = 36  # output width (C * K with K=6)


def makePeaks(truth, sigma=1.0, jitter=2.0, shuffle=True):
    """Assign each sample a peak position: class centre (shuffled) + per-sample jitter."""
    torch.manual_seed(0)
    # class centre = one of M positions, shuffled uniformly across classes
    if shuffle:
        centres = torch.randperm(M)[:C]  # C centres, uniform across M
    else:
        centres = torch.arange(C) * (M // C)
    peaks = centres[truth].float() + torch.randn(N) * jitter
    return peaks


def trainBottom(features, truth, peaks, sigma):
    """Train an embedding + head to emit the normal soft label with the given peaks."""
    torch.manual_seed(1)
    model = torch.nn.Sequential(
        torch.nn.Linear(WIDTH, 128), torch.nn.ReLU(), torch.nn.Linear(128, EMB))
    head = torch.nn.Linear(EMB, M)
    params = list(model.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)

    for _ in range(1200):
        optimizer.zero_grad()
        logits = head(model(features))
        # normal-shaped soft target centred at each sample's peak
        positions = torch.arange(M, dtype=torch.float32)
        target = torch.exp(-((positions[None, :] - peaks[:, None]) ** 2) / (2 * sigma ** 2))
        target = target / target.sum(dim=1, keepdim=True)
        logProb = torch.log_softmax(logits, dim=1)
        loss = -(target * logProb).sum(dim=1).mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        return model(features).detach()


def fitHead(embedding, truth, auxOnly=False):
    torch.manual_seed(1)
    head = torch.nn.Linear(embedding.shape[1], C)
    optimizer = torch.optim.Adam(head.parameters(), lr=5e-3)
    xs, ys = (embedding[:AUX], truth[:AUX]) if auxOnly else (embedding[:SPLIT], truth[:SPLIT])
    for _ in range(600):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(head(xs), ys)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        pred = head(embedding[SPLIT:]).argmax(1)
    return float((pred == truth[SPLIT:]).float().mean())


def main():
    torch.manual_seed(0)
    features = torch.randn(N, WIDTH)
    truth = (features @ torch.randn(WIDTH, C)).argmax(dim=1)

    print("  %-34s %10s %14s" % ("配置", "A逆映射精度", "攻击者恢复率"))
    print("  " + "-" * 60)
    print("  随机水平：攻击者恢复率 = 1/C = %.4f" % (1.0 / C))

    for sigma in (1.0, 2.0, 3.0, 5.0):
        peaks = makePeaks(truth, sigma=sigma, jitter=2.0, shuffle=True)
        embedding = trainBottom(features, truth, peaks, sigma)
        # A inverts by matching the peak position (recorded) back to its class
        # A's inverse accuracy: can the model's argmax recover the true class via the peak?
        # Simplify: A fits a full-labelled head on the embedding (same info A has).
        aAcc = fitHead(embedding, truth, auxOnly=False)
        att = fitHead(embedding, truth, auxOnly=True)
        print("  sigma=%.1f (峰宽) 峰随机打乱       %10.4f %14.4f" % (sigma, aAcc, att))

    print()
    print("  判读：若存在 sigma 使 A 精度高且攻击者恢复率接近 0.167，则设计成立；")
    print("  若攻击者恢复率始终远高于随机，则随机峰位置仍可被还原。")


if __name__ == "__main__":
    main()
