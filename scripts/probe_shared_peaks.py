"""Does SHARING peaks between classes weaken PMC at acceptable A-inverse cost?

The user's idea: let two classes share some peaks, so the peak set is NOT class-unique, which
should blur the class identity the embedding encodes.  The trade-off is that a shared peak
cannot be inverted to a single class, so A's own inverse accuracy must also drop - the
question is whether PMC falls MORE than A.

Design: C=6 classes.  Each class has ONE private peak at position c, and ONE shared peak at
position C + c//2 (so classes 2k and 2k+1 share the peak C+k).  A inverts a private peak
exactly; on a shared peak it must guess between the two sharing classes (1/2 correct).

    <python> scripts/probe_shared_peaks.py
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

    print("  %-24s %12s %14s" % ("方案", "A逆映射精度", "PMC恢复率"))
    print("  " + "-" * 52)
    print("  随机水平 = %.4f" % (1.0 / C))

    # no sharing: class c -> {c, c+C} (two private peaks), M = 2C
    M1 = 2 * C
    pick = torch.randint(0, 2, (N,))
    peaks1 = torch.where(pick == 0, truth, truth + C).float()

    # sharing: class c -> {c (private), C + c//2 (shared by c and c^1)}, M = C + C//2
    M2 = C + C // 2
    peaks2 = torch.where(pick == 0, truth, C + (truth // 2)).float()

    for name, peaks, M, shared in (("不共享(私有峰)", peaks1, M1, False),
                                    ("共享(每2类共享1峰)", peaks2, M2, True)):
        embedding = trainBottom(features, truth, peaks, M)
        att = fitHead(embedding, truth, auxOnly=True)

        # A inverse: train a head to reproduce the peaks, then invert
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
        if shared:
            # private peak (< C) -> exact; shared peak (>= C) -> 50% correct (guess)
            private = predPos < C
            exact = (private & (predPos == truth[SPLIT:])).float().mean()
            sharedOk = ((~private) & (predPos - C == truth[SPLIT:] // 2)).float()
            # on a shared peak, guess the two sharing classes -> 0.5 expected
            aAcc = float(exact + 0.5 * sharedOk.mean())
        else:
            aAcc = float((predPos % C == truth[SPLIT:]).float().mean())
        print("  %-24s %12.4f %14.4f" % (name, aAcc, att))

    print()
    print("  判读：若共享峰的 PMC 下降幅度 > A 逆映射下降幅度，则共享峰净收益为正。")


if __name__ == "__main__":
    main()
