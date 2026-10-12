"""Does removing the class CENTRE actually hide the class?

The user's argument: if same-class soft labels differ as much as different-class soft labels
(no class centre), there is nothing for an attacker to cluster.

This tests three peak-assignment schemes, keeping the NORMAL peak shape and per-sample
random peak choice identical, and varying only the class -> position mapping:

  A. block      : class c -> contiguous block [c*K, (c+1)*K)   (has a class centre)
  B. interleaved: class c -> positions {c, c+K, c+2K, ...}     (no centre, but the mapping
                  class -> position SET is still deterministic)
  C. random     : class c -> a per-sample random position, unrelated to class (the mapping
                  class -> position is NOT a function)

    <python> scripts/probe_no_centre.py
"""

import torch

N, WIDTH, EMB, C = 4000, 32, 64, 6
SPLIT = 3000
AUX = 60
K = 6
M = C * K  # 36


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


def makePeaks(truth, scheme):
    torch.manual_seed(0)
    if scheme == "block":
        # class c -> contiguous block [c*K, (c+1)*K); per-sample random pick inside
        base = truth * K
        pick = torch.randint(0, K, (N,))
        return (base + pick).float()
    if scheme == "interleaved":
        # class c -> positions {c, c+K, c+2K, ...}; per-sample random pick inside
        pick = torch.randint(0, K, (N,))
        return (truth + pick * C).float()
    # random: position unrelated to class
    return torch.randint(0, M, (N,)).float()


def main():
    torch.manual_seed(0)
    features = torch.randn(N, WIDTH)
    truth = (features @ torch.randn(WIDTH, C)).argmax(dim=1)

    print("  %-14s %12s %14s" % ("峰位置方案", "A逆映射精度", "攻击者恢复率"))
    print("  " + "-" * 42)
    print("  随机水平 = 1/C = %.4f" % (1.0 / C))

    for scheme in ("block", "interleaved", "random"):
        peaks = makePeaks(truth, scheme)
        embedding = trainBottom(features, truth, peaks)
        aAcc = fitHead(embedding, truth, auxOnly=False)
        att = fitHead(embedding, truth, auxOnly=True)
        print("  %-14s %12.4f %14.4f" % (scheme, aAcc, att))

    print()
    print("  判读：")
    print("  * block: 有类中心，攻击者可聚类 -> 恢复率高")
    print("  * interleaved: 无类中心(同类分散/类间交错)，但类->位置集合仍确定")
    print("  * random: 类->位置完全无关，模型不可学 -> A 逆映射掉到随机")


if __name__ == "__main__":
    main()
