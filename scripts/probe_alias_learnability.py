"""Is per-sample random aliasing LEARNABLE, or only memorisable?

A can access every sample individually (it has sampleIds and labels), so the question is not
access but generalisation: with labels scattered per sample, can a model learn a function
that transfers to unseen samples?

Compares three labelings on the same data:
  (a) true labels
  (b) one fixed class-level permutation   (same class -> same alias)
  (c) per-sample random permutations      (same class -> different aliases)

Train fit and test generalisation are reported separately, because a model can always
memorise in (c) while failing to generalise - and generalisation is what A needs, since the
test samples are unseen.

    <python> scripts/probe_alias_learnability.py
"""

import torch

N, WIDTH, CLASSES = 2000, 32, 4
SPLIT = 1400


def main():
    torch.manual_seed(0)
    features = torch.randn(N, WIDTH)
    weights = torch.randn(WIDTH, CLASSES)
    truth = (features @ weights).argmax(dim=1)
    trainX, testX = features[:SPLIT], features[SPLIT:]
    trainY, testY = truth[:SPLIT], truth[SPLIT:]

    # (b) fixed class-level permutation
    fixedPerm = torch.tensor([2, 0, 3, 1])
    trainFixed, testFixed = fixedPerm[trainY], fixedPerm[testY]

    # (c) per-sample random permutations
    generator = torch.Generator().manual_seed(11)
    perSample = torch.stack([torch.randperm(CLASSES, generator=generator) for _ in range(N)])
    aliasAll = torch.tensor([perSample[i, truth[i]].item() for i in range(N)])
    trainAlias, testAlias = aliasAll[:SPLIT], aliasAll[SPLIT:]

    def trainAndScore(labelsTrain, labelsTest, hidden=256, epochs=1500):
        torch.manual_seed(1)
        model = torch.nn.Sequential(
            torch.nn.Linear(WIDTH, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, CLASSES))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        lossFunction = torch.nn.CrossEntropyLoss()
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = lossFunction(model(trainX), labelsTrain)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            fit = float((model(trainX).argmax(1) == labelsTrain).float().mean())
            general = float((model(testX).argmax(1) == labelsTest).float().mean())
        return fit, general

    print("  %-36s %13s %13s" % ("配置", "训练集拟合", "测试集泛化"))
    print("  " + "-" * 66)
    fit, general = trainAndScore(trainY, testY)
    print("  %-36s %13.4f %13.4f" % ("(a) 真实标签", fit, general))
    fit, general = trainAndScore(trainFixed, testFixed)
    print("  %-36s %13.4f %13.4f" % ("(b) 类别级固定置换", fit, general))
    for hidden in (256, 1024, 2048):
        fit, general = trainAndScore(trainAlias, testAlias, hidden=hidden)
        print("  %-36s %13.4f %13.4f"
              % ("(c) 逐样本随机置换 (隐藏 %d)" % hidden, fit, general))

    print()
    print("  随机猜测基线 = 1/C = %.4f" % (1.0 / CLASSES))
    print()
    print("  判读：")
    print("  * (b) 的测试泛化应接近 (a)，因为同类样本共用别名，任务与原始任务同构。")
    print("  * (c) 若测试泛化停在随机水平，说明逐样本随机别名只能记忆、不能泛化；")
    print("    而测试样本是未见过的，A 需要泛化，因此该形式对 A 无用。")


if __name__ == "__main__":
    main()
