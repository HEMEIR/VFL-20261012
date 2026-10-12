"""Correct A-inverse (lookup table) + attacker, sweeping K.

The user's two points, tested correctly:
  1. A inverts via a LOOKUP TABLE (argmax position -> class), not by fitting a head on the
     embedding, so A's accuracy is whether argmax lands in the right class's position set.
  2. Increase K (soft labels per class) to shrink cross-class soft-label difference and push
     the attacker's recovery toward chance.

Interleaved scheme: class c owns positions {c, c+C, c+2C, ...} (K of them, stride C), so
argmax position p inverts exactly by c = p % C.

    <python> scripts/probe_interleave_sweep.py
"""

import torch

N, WIDTH, EMB, C = 4000, 32, 64, 6
SPLIT = 3000
AUX = 60


def trainBottom(features, truth, peaks, M, sigma=2.0):
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

    print("  %-12s %14s %14s" % ("K(每类软标签)", "A查找表逆映射", "攻击者恢复率"))
    print("  " + "-" * 44)
    print("  随机水平 = 1/C = %.4f" % (1.0 / C))

    for K in (2, 4, 6, 12, 24):
        M = C * K
        torch.manual_seed(0)
        # interleaved: class c -> positions {c, c+C, c+2C, ...}, per-sample random pick
        pick = torch.randint(0, K, (N,))
        peaks = (truth + pick * C).float()
        embedding = trainBottom(features, truth, peaks, M)

        # A's correct inverse: argmax position % C
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
        aAcc = float((predPos % C == truth[SPLIT:]).float().mean())

        att = fitHead(embedding, truth, auxOnly=True)
        print("  %-12d %14.4f %14.4f" % (K, aAcc, att))

    print()
    print("  判读：A 查找表逆映射应接近 1.0（用户论点 1）；")
    print("  攻击者恢复率应随 K 增大而下降，目标是接近随机 %.4f（用户论点 2）。" % (1.0 / C))


if __name__ == "__main__":
    main()
