"""Sweep the shared-peak ratio: each class shares ONE peak with exactly ONE other class.

The goal is the total score = performance + privacy.  A class has K peaks, of which K-1 are
PRIVATE (exact inverse) and 1 is SHARED with its partner class (50% inverse).  The sweep over
K varies the shared fraction 1/K and measures both A's inverse accuracy and PMC recovery,
so we can find the K with the best trade-off.

    <python> scripts/probe_share_sweep.py
"""

import torch

N, WIDTH, EMB, C = 4000, 32, 64, 6
SPLIT = 3000
AUX = 60


def makePeaks(truth, K):
    """Class c -> (K-1) private peaks + 1 shared peak with class c^1."""
    torch.manual_seed(0)
    M = C * (K - 1) + C // 2  # private + shared positions
    private = torch.stack([truth * (K - 1) + t for t in range(K - 1)], dim=1)  # [N, K-1]
    shared = C * (K - 1) + (truth // 2)  # class c and c^1 share the same position
    allPeaks = torch.cat([private, shared.unsqueeze(1)], dim=1)  # [N, K]
    pick = torch.randint(0, K, (N,))
    return allPeaks[torch.arange(N), pick].float(), M


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

    print("  %-14s %14s %14s %14s %14s" % ("K(共享比例)", "A逆映射", "PMC恢复", "A损失", "PMC改善"))
    print("  " + "-" * 72)
    print("  随机水平 = %.4f" % (1.0 / C))

    base = None
    results = []
    for K in (2, 3, 4, 6, 8):
        peaks, M = makePeaks(truth, K)
        embedding = trainBottom(features, truth, peaks, M)
        att = fitHead(embedding, truth, auxOnly=True)

        # A inverse: reproduce peaks then invert (shared peak -> 50% guess)
        torch.manual_seed(1)
        head = torch.nn.Linear(EMB, M)
        optimizer = torch.optim.Adam(head.parameters(), lr=5e-3)
        positions = torch.arange(M, dtype=torch.float32)
        for _ in range(600):
            optimizer.zero_grad()
            logits = head(embedding[:SPLIT])
            target = torch.exp(-((positions[None, :] - peaks[:SPLIT, None]) ** 2) / (2 * 1.5 ** 2))
            target = target / target.sum(dim=1, keepdim=True)
            logProb = torch.log_softmax(logits, dim=1)
            loss = -(target * logProb).sum(dim=1).mean()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            predPos = head(embedding[SPLIT:]).argmax(1)
        private = predPos < C * (K - 1)
        exact = (private & (predPos // (K - 1) == truth[SPLIT:])).float().mean()
        sharedGuess = ((~private) & ((predPos - C * (K - 1)) == truth[SPLIT:] // 2)).float()
        aAcc = float(exact + 0.5 * sharedGuess.mean())

        if base is None:
            base = (aAcc, att)
        aLoss = base[0] - aAcc
        attGain = base[1] - att
        results.append((K, aAcc, att, aLoss, attGain))
        print("  %-14s %14.4f %14.4f %14.4f %14.4f" % (K, aAcc, att, aLoss, attGain))

    print()
    print("  判读：K=2 是 50%% 共享；K 越大共享越少（A 损失越小）。")
    print("  找 PMC改善 - A损失 最大的 K，即净隐私收益最大。")


if __name__ == "__main__":
    main()
