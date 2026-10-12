"""Label-smoothing sweep: how SOFT must the alias encoding be to hide the label?

Current "soft" encoding is [0,...,0, 0.5, 0.5, 0,...,0] - only the block is non-zero, so the
block membership is still exactly exposed.  This sweep adds a leak epsilon to the OUTSIDE of
the block, turning it into genuine label smoothing:

    class c -> (1-eps) mass split inside block [c*K, c*K+K), eps mass split over the OTHER
    (outDim-K) positions.

Two things are measured together, because the whole question is whether there is ANY eps
where privacy improves without losing accuracy:

    A_ACC   A's own inverse accuracy (argmax // K, which is what A does at inference)
    ATTACK  attacker recovery: fit a 2-class head on the frozen embedding with a small
            auxiliary set of TRUE labels, then score held-out data.

    <python> scripts/probe_alias_smoothing.py
"""

import torch

N, WIDTH, EMB = 4000, 32, 64
SPLIT = 3000
AUX = 40
CLASSES = 2


def buildEmbedding(softTarget, outDim):
    """Train a frozen embedding + alias head on the soft target."""
    torch.manual_seed(0)
    features = torch.randn(N, WIDTH)
    model = torch.nn.Sequential(
        torch.nn.Linear(WIDTH, 128), torch.nn.ReLU(), torch.nn.Linear(128, EMB))
    head = torch.nn.Linear(EMB, outDim)
    params = list(model.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)
    for _ in range(1200):
        optimizer.zero_grad()
        logProb = torch.log_softmax(head(model(features)), dim=1)
        loss = -(softTarget * logProb).sum(dim=1).mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return model(features).detach()


def attackAccuracy(embedding, trueLabels):
    """Fit a 2-class head on AUX true-labelled samples, score the rest."""
    torch.manual_seed(1)
    head = torch.nn.Linear(embedding.shape[1], CLASSES)
    optimizer = torch.optim.Adam(head.parameters(), lr=5e-3)
    auxX, auxY = embedding[:AUX], trueLabels[:AUX]
    for _ in range(600):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(head(auxX), auxY)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        pred = head(embedding[SPLIT:]).argmax(1)
    return float((pred == trueLabels[SPLIT:]).float().mean())


def main():
    torch.manual_seed(0)
    features = torch.randn(N, WIDTH)
    truth = (features @ torch.randn(WIDTH, CLASSES)).argmax(dim=1)

    print("  %-28s %10s %10s %10s" % ("编码(eps=块外泄漏)", "A精度", "攻击者恢复", "随机?"))
    print("  " + "-" * 62)

    K = 2
    outDim = 2 * CLASSES * K  # 8, with offset base = CLASSES*K = 4
    base = CLASSES * K

    for eps in (0.0, 0.1, 0.2, 0.4, 0.6):
        softTarget = torch.zeros(N, outDim)
        for c in range(CLASSES):
            mask = (truth == c)
            block = torch.arange(c * K + base, c * K + base + K)
            inside = (1.0 - eps) / K
            outside = eps / (outDim - K)
            for i in range(N):
                if not mask[i]:
                    continue
                softTarget[i, :] = outside
                softTarget[i, block] = inside

        embedding = buildEmbedding(softTarget, outDim)

        # A's inverse accuracy: argmax // K after removing the base
        head = torch.nn.Linear(EMB, outDim)
        optimizer = torch.optim.Adam(head.parameters(), lr=5e-3)
        for _ in range(600):
            optimizer.zero_grad()
            logProb = torch.log_softmax(head(embedding[:SPLIT]), dim=1)
            loss = -(softTarget[:SPLIT] * logProb).sum(dim=1).mean()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            pred = head(embedding[SPLIT:]).argmax(1)
        inverted = (pred - base).clamp(min=0) // K
        aAcc = float((inverted == truth[SPLIT:]).float().mean())

        att = attackAccuracy(embedding, truth)
        print("  eps=%.1f (块外泄漏)          %10.4f %10.4f" % (eps, aAcc, att))

    print()
    print("  判读：若某 eps 使攻击者恢复率下降到接近 0.5（随机）而 A 精度保持，")
    print("  则真 label smoothing 值得实现；若攻击者恢复率始终 > 随机，则")
    print("  「软编码」无法靠调软硬程度来隐藏标签。")


if __name__ == "__main__":
    main()
