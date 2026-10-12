"""Symmetric (edge-to-centre) peak assignment vs block and interleaved.

The user's rule: class 0 takes the two EXTREME peaks, class 1 the next pair inward, ... so
every class is a symmetric pair whose centre is the middle of the space - no class centre,
but a deterministic, invertible rule (inverse: c = min(p, M-1-p) for K=2).

    <python> scripts/probe_symmetric.py
"""

import torch

N, WIDTH, EMB, C = 4000, 32, 64, 6
SPLIT = 3000
AUX = 60
K = 2
M = C * K  # 12


def trainBottom(features, truth, peaks, sigma=2.0):
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

    schemes = {
        "block": lambda t, pick: (t * K + pick).float(),
        "interleaved": lambda t, pick: (t + pick * C).float(),
        "symmetric": lambda t, pick: (t + pick * (M - 1 - 2 * t)).float(),
    }
    inverses = {
        "block": lambda p: p // K,
        "interleaved": lambda p: p % C,
        "symmetric": lambda p: torch.minimum(p, M - 1 - p),
    }

    print("  %-12s %14s %14s" % ("方案", "A查找表逆映射", "攻击者恢复率"))
    print("  " + "-" * 42)
    print("  随机水平 = %.4f" % (1.0 / C))

    for name in ("block", "interleaved", "symmetric"):
        torch.manual_seed(0)
        pick = torch.randint(0, K, (N,))
        peaks = schemes[name](truth, pick)
        embedding = trainBottom(features, truth, peaks)

        # A's inverse via the lookup rule
        torch.manual_seed(1)
        head = torch.nn.Linear(EMB, M)
        optimizer = torch.optim.Adam(head.parameters(), lr=5e-3)
        positions = torch.arange(M, dtype=torch.float32)
        for _ in range(600):
            optimizer.zero_grad()
            logits = head(embedding[:SPLIT])
            target = torch.exp(-((positions[None, :] - peaks[:SPLIT, None]) ** 2) / (2 * 2.0 ** 2))
            target = target / target.sum(dim=1, keepdim=True)
            logProb = torch.log_softmax(logits, dim=1)
            loss = -(target * logProb).sum(dim=1).mean()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            predPos = head(embedding[SPLIT:]).argmax(1)
        aAcc = float((inverses[name](predPos) == truth[SPLIT:]).float().mean())

        att = fitHead(embedding, truth, auxOnly=True)
        print("  %-12s %14.4f %14.4f" % (name, aAcc, att))

    print()
    print("  判读：symmetric 应同时满足 A 逆映射高（规律可逆）且攻击者低。")


if __name__ == "__main__":
    main()
