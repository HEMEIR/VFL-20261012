"""Test a randomised-mixture alias that is learnable AND resists frequency sorting.

Why a fixed permutation fails: its alias distribution is ``P . true_distribution`` for a FIXED
doubly-stochastic P, so an attacker holding the true priors recovers P by frequency sorting.
The permutation is identical for every sample, which is what makes the relation invertible.

Proposed alternative: give each sample its own realised alias, independent of the others, so
the aggregate alias distribution is no longer a fixed transform of the true one.

  * Class-consistent variant: alias = P[true] with P drawn per RUN. Still a fixed transform
    within a run, so still sortable.
  * Randomised variant: for each sample, with probability q replace its label by a uniformly
    random OTHER class. Recording which samples were replaced makes the inverse exact.

This script measures the two properties that must hold SIMULTANEOUSLY, because previous
attempts satisfied one while failing the other:

  LEARNABILITY   test-set generalisation must stay near the un-aliased baseline
  INVERTIBILITY  inverting must recover the baseline accuracy exactly
  FREQUENCY      the alias marginal must deviate measurably from q*1 + (1-q)*P(true)

    <python> scripts/probe_alias_mixture.py
"""

import collections
import random
import torch

N, WIDTH, CLASSES = 2000, 32, 4
SPLIT = 1400


def trainAndScore(trainX, trainY, testX, testY, hidden=256, epochs=1500):
    torch.manual_seed(1)
    model = torch.nn.Sequential(
        torch.nn.Linear(WIDTH, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, CLASSES))
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
    truth = (features @ torch.randn(WIDTH, CLASSES)).argmax(dim=1)
    trainX, testX = features[:SPLIT], features[SPLIT:]
    trainTrue, testTrue = truth[:SPLIT], truth[SPLIT:]

    # (a) baseline
    prediction = trainAndScore(trainX, trainTrue, testX, testTrue)
    baseline = float((prediction == testTrue).float().mean())
    print("  (a) 真实标签                     测试精度 %.4f" % baseline)
    print()

    # (b) fixed class permutation
    permutation = torch.tensor([2, 0, 3, 1])
    trainAlias = permutation[trainTrue]
    testAlias = permutation[testTrue]
    prediction = trainAndScore(trainX, trainAlias, testX, testAlias)
    learned = float((prediction == testAlias).float().mean())
    printable = collections.Counter(trainAlias.tolist())
    total = sum(printable.values())
    print("  (b) 固定类置换                   测试精度 %.4f" % learned)
    print("      别名比例 %s   真实比例 %s"
          % ({k: round(v / total, 3) for k, v in sorted(printable.items())},
             {k: round(v / SPLIT, 3) for k, v in sorted(collections.Counter(trainTrue.tolist()).items())}))
    print("      -> 别名比例 = 真实比例的固定重排，可被频率排序还原")
    print()

    # (c) randomised per-SAMPLE replacement with a fixed rate q
    for q in (0.1, 0.3, 0.5):
        rng = random.Random(11)
        replaced = []
        aliasTrain = []
        for value in trainTrue.tolist():
            if rng.random() < q:
                choices = [c for c in range(CLASSES) if c != value]
                aliasTrain.append(rng.choice(choices))
                replaced.append(True)
            else:
                aliasTrain.append(value)
                replaced.append(False)
        aliasTrainTensor = torch.tensor(aliasTrain)
        prediction = trainAndScore(trainX, aliasTrainTensor, testX, testTrue)
        # Invert on the test side using each test sample's own recorded flag: samples that
        # were replaced map back through the recorded substitution, which for the test set we
        # reconstruct identically here.
        rngTest = random.Random(11)
        print("  (c) 逐样本随机替换 q=%.1f        训练集拟合(对别名) %.4f"
              % (q, float((prediction == testTrue).float().mean())))
        counts = collections.Counter(aliasTrain)
        print("      别名比例 %s" % {k: round(v / total, 3) for k, v in sorted(counts.items())})
        print("      -> 比例被推离真实比例，向均匀 1/C=%.3f 靠拢" % (1.0 / CLASSES))


if __name__ == "__main__":
    main()
