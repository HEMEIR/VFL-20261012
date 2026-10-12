"""Does spreading the peak positions (larger inter-class distance) weaken PMC?

The user's insight: symmetric peaks are arranged CONTINUOUSLY within each layer
(class c -> position c), so adjacent classes (esp. middle ones 2,3,4) sit distance 1 apart
and stay easy to separate - the class identity is still encoded in the peak position.

This compares three peak-ARRANGEMENT schemes (same symmetric idea, different inter-class
spacing) on PMC recovery from the frozen embedding:

  contiguous : class c -> position c            (current; inter-class distance 1)
  spread     : class c -> position c * gap      (inter-class distance gap)
  shuffled   : class c -> a random permutation  (class identity NOT a simple function)

    <python> scripts/probe_peak_spacing.py
"""

import torch

N, WIDTH, EMB, C = 4000, 32, 64, 6
SPLIT = 3000
AUX = 60


def trainBottom(features, truth, peaks, M, sigma=1.5):
    torch.manual_seed(1)
    model = torch.nn.Sequential(
        torch.nn.Linear(WIDTH, 128), torch.nn.ReLU(), torch.nn.Linear(128, EMB))
    head = torch.nn.Linear(EMB, M)
    params = list(model.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)
    positions = torch.arange(M, dtype=torch.float32)
    for _ in range(1200):
        optimizer.zero_grad()
        logits = head(model(features))
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

    print("  %-20s %12s %14s" % ("峰排列", "A逆映射精度", "攻击者(PMC)恢复率"))
    print("  " + "-" * 48)
    print("  随机水平 = %.4f" % (1.0 / C))

    # contiguous: class c -> position c (per-layer), symmetric pairs
    # spread: class c -> c*gap (per layer), M scales with gap
    # shuffled: class c -> perm[c] (random, but fixed per class)

    torch.manual_seed(0)
    perm = torch.randperm(C)

    schemes = {}
    # contiguous symmetric, K=2 (one left + one right), M = 2*C
    M = 2 * C
    pick = torch.randint(0, 2, (N,))
    left = truth
    right = M - 1 - truth
    schemes["contiguous"] = (torch.where(pick == 0, left, right).float(), M)

    # spread symmetric, gap=3: class c -> {3c, M-1-3c}, M = 3*(C-1)*2 + 1... use M = 6*C
    M2 = 6 * C
    left2 = truth * 3
    right2 = M2 - 1 - truth * 3
    schemes["spread gap=3"] = (torch.where(pick == 0, left2, right2).float(), M2)

    # shuffled symmetric: class c -> {perm[c], M-1-perm[c]}
    M3 = 2 * C
    left3 = perm[truth]
    right3 = M3 - 1 - perm[truth]
    schemes["shuffled"] = (torch.where(pick == 0, left3, right3).float(), M3)

    for name, (peaks, M) in schemes.items():
        embedding = trainBottom(features, truth, peaks, M)
        aAcc = fitHead(embedding, truth, auxOnly=False)
        att = fitHead(embedding, truth, auxOnly=True)
        print("  %-20s %12.4f %14.4f" % (name, aAcc, att))

    print()
    print("  判读：若 spread/shuffled 的攻击者恢复率明显低于 contiguous，")
    print("  则用户的洞察成立：中间类相邻是 PMC 仍能区分的原因。")


if __name__ == "__main__":
    main()
