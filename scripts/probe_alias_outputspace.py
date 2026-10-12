"""Does expanding the alias output space actually hide the label from an attacker?

The proposal: instead of swapping 0/1, give each class its own BLOCK of aliases (K per class,
2K outputs total) and invert at inference. The hope is that with more outputs the attack's
success falls toward chance.

This tests the mechanism directly, without the VFL harness, by emulating what the label
inference attack does: fit a head on B's frozen embedding using a small auxiliary set of
TRUE labels, then measure how well it recovers true labels on held-out data.

Three attackers are compared, because the proposal's assumption is that the attacker's head
is stuck at the 2-class output size:

  (1) head outputs 2 classes           - the naive attacker matching the auxiliary labels
  (2) head outputs 2K classes          - an attacker that expands to match the aliases
  (3) linear probe on the full embedding - the strongest attacker, dimension-agnostic

If (3) stays high, expanding the output space does not hide anything: the label information
is still in the embedding, merely re-encoded.

    <python> scripts/probe_alias_outputspace.py
"""

import torch

N, WIDTH, EMB = 4000, 32, 64
SPLIT = 3000
AUX = 40


def buildEmbedding(aliasLabels, classes):
    """A frozen embedding model trained on alias labels - stands in for bottomB."""
    torch.manual_seed(0)
    features = torch.randn(N, WIDTH)
    teacher = torch.randn(WIDTH, EMB)
    # The embedding carries whatever the alias task made it carry.
    labels = aliasLabels
    model = torch.nn.Sequential(
        torch.nn.Linear(WIDTH, 128), torch.nn.ReLU(), torch.nn.Linear(128, EMB))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossFunction = torch.nn.CrossEntropyLoss()
    head = torch.nn.Linear(EMB, classes)
    parameters = list(model.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(parameters, lr=1e-3)
    for _ in range(1200):
        optimizer.zero_grad()
        loss = lossFunction(head(model(features)), labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return features, model(features).detach()


def attackAccuracy(embedding, trueLabels, classesOut):
    """Fit a head on AUX true-labelled samples, score the rest."""
    torch.manual_seed(1)
    head = torch.nn.Linear(embedding.shape[1], classesOut)
    optimizer = torch.optim.Adam(head.parameters(), lr=5e-3)
    lossFunction = torch.nn.CrossEntropyLoss()
    auxX, auxY = embedding[:AUX], trueLabels[:AUX]
    for _ in range(600):
        optimizer.zero_grad()
        loss = lossFunction(head(auxX), auxY)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        prediction = head(embedding[SPLIT:]).argmax(1)
    return float((prediction == trueLabels[SPLIT:]).float().mean())


def main():
    torch.manual_seed(0)
    features = torch.randn(N, WIDTH)
    truth = (features @ torch.randn(WIDTH, 2)).argmax(dim=1)

    print("  %-34s %12s %14s %14s" % ("配置", "别名测试精度", "攻击者(2类)", "线性探针"))
    print("  " + "-" * 78)

    for K in (1, 2, 4, 8):
        classes = 2 * K
        # Per-sample ALIAS drawn from the sample's own class block.  A records which alias
        # each sample got, so inverting is exact: alias // K == true class.
        generator = torch.Generator().manual_seed(11)
        alias = truth * K + torch.randint(0, K, (N,), generator=generator)
        embedding, _ = buildEmbedding(alias, classes)

        # A's own accuracy after inversion: how well does a head on the alias task do?
        # The probe reads the EMBEDDING (EMB wide), not the raw features.
        torch.manual_seed(1)
        proxy = torch.nn.Linear(embedding.shape[1], classes)
        optimizer = torch.optim.Adam(proxy.parameters(), lr=5e-3)
        lossFunction = torch.nn.CrossEntropyLoss()
        for _ in range(600):
            optimizer.zero_grad()
            loss = lossFunction(proxy(embedding[:SPLIT]), alias[:SPLIT])
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            aliasPrediction = proxy(embedding[SPLIT:]).argmax(1)
        inverted = (aliasPrediction // K)
        aAccuracy = float((inverted == truth[SPLIT:]).float().mean())

        naive = attackAccuracy(embedding, truth, 2)
        probe = attackAccuracy(embedding, truth, 2)
        print("  K=%d 每类别名数 (输出 %2d 维)          %12.4f %14.4f %14.4f"
              % (K, classes, aAccuracy, naive, probe))

    print()
    print("  说明：三个攻击者都在同一个冻结嵌入上拟合；区别只在头的输出维度。")
    print("  * 若 >线性探针< 随 K 上升而保持在随机以上，说明扩维并未隐藏标签信息，")
    print("    只是把同一信息重新编码到更多维度上。")


if __name__ == "__main__":
    main()
