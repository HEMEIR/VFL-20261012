"""Test the "several aliases per class" design for learnability.

The proposal: in a binary problem, do NOT swap 0 and 1. Instead create several alias labels
per class, map each class's samples onto its own group of aliases, and invert at inference
from the recorded mapping.

Written as an EXPLICIT hypothesis test, because the deciding question is learnability and
that is where two earlier designs failed:

  H1  With K aliases per class the task stays learnable, i.e. test generalisation is close to
      the un-aliased baseline - and inverting recovers the baseline accuracy.
  H2  The alias marginal is flattened enough to defeat frequency sorting.

Both are measured, and the two-level structure (which aliases belong to which true class) is
varied independently of the number of aliases, so the two effects can be told apart:

  * "grouped"   : alias is a function of (class, per-sample hash) - the MODEL cannot infer the
                  grouping from the input alone, because within one class the same input
                  region maps to several aliases.
  * "class-only": alias is a function of class only (K = 1), the already-measured baseline
                  design that is learnable but sortable.

    <python> scripts/probe_alias_groups.py
"""

import collections
import random
import torch

N, WIDTH = 2400, 32
SPLIT = 1700


def trainAndScore(trainX, trainY, testX, classes, epochs=1500, hidden=256):
    torch.manual_seed(1)
    model = torch.nn.Sequential(
        torch.nn.Linear(WIDTH, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossFunction = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = lossFunction(model(trainX), trainY)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return model(testX).argmax(1)


def main():
    torch.manual_seed(0)
    features = torch.randn(N, WIDTH)
    truth = (features @ torch.randn(WIDTH, 2)).argmax(dim=1)
    trainX, testX = features[:SPLIT], features[SPLIT:]
    trainTrue, testTrue = truth[:SPLIT], truth[SPLIT:]

    # Baseline
    prediction = trainAndScore(trainX, trainTrue, testX, 2)
    baseline = float((prediction == testTrue).float().mean())
    print("  基线（真实标签，2 类）     测试精度 %.4f" % baseline)
    print()

    print("  %-38s %12s %12s %12s" % ("配置", "训练拟合", "测试泛化", "逆映射后"))
    print("  " + "-" * 80)

    for K in (1, 2, 4, 8):
        classes = 2 * K
        rng = random.Random(11)
        # Grouped design: each sample draws its alias uniformly from ITS class's block.
        aliasTrain = []
        for value in trainTrue.tolist():
            aliasTrain.append(value * K + rng.randrange(K))
        aliasTest = []
        rngTest = random.Random(11)
        testAliases = []
        for value in testTrue.tolist():
            testAliases.append(value * K + rngTest.randrange(K))

        aliasTrainTensor = torch.tensor(aliasTrain)
        prediction = trainAndScore(trainX, aliasTrainTensor, testX, classes)
        # Invert: alias // K is the true class, so the inverse is exact and needs only the
        # block structure that A recorded.
        inverted = torch.tensor([int(v) // K for v in prediction.tolist()])
        # Train-set fit, for reference.
        fitPrediction = trainAndScore(trainX, aliasTrainTensor, trainX, classes, epochs=1)
        invertedAccuracy = float((inverted == testTrue).float().mean())
        counts = collections.Counter(aliasTrain)
        total = len(aliasTrain)
        spread = max(counts.values()) / total - min(counts.values()) / total
        print("  K=%d 每类别名数 (共 %2d 类)              %12s %12.4f %12.4f"
              % (K, classes, "-", float((prediction == torch.tensor(testAliases)).float().mean()),
                 invertedAccuracy))
        label = "  别名频率极差 %.4f" % spread
        print("  %-38s %s" % ("", label))

    print()
    print("  判读：")
    print("  * >测试泛化< 是对该样本自己别名的准确率；>逆映射后< 才是 A 的实际精度。")
    print("  * K>1 时同一真实类的样本落在 K 个不同别名上，模型无法从输入判断用哪个别名，")
    print("    这正是此前被证伪的逐样本随机结构；K=1 退化为可学但可排序的固定置换。")
    print("  * >别名频率极差< 越小说明分布越均匀（越难被频率排序还原）。")


if __name__ == "__main__":
    main()
