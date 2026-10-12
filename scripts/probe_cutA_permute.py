"""Check whether permuting cutA makes the top model train on ALIAS labels, with no core edit.

Idea: the trainer computes ``loss = CE(topModel(cutA, cutB), y)`` with the TRUE label y.  If
the hook instead sends ``cutA' = P(cutA)`` for an input-space permutation P of A's features,
then ``topModel(cutA', cutB) = topModel(P(cutA), cutB) = permuted_logits``, so minimising the
loss against y trains the model to output ``P^{-1}(y)`` - i.e. an ALIAS of y.  No label ever
has to be modified, so no core change is needed.

This trains both forms on identical data and checks:
  * does the cutA-permuted model score high accuracy against the ALIAS labels?
  * does inverting its predictions recover the true-label accuracy?

    <python> scripts/probe_cutA_permute.py
"""

import torch

N, WIDTH, CLASSES = 1600, 24, 4
SPLIT = 1200


def main():
    torch.manual_seed(0)
    features = torch.randn(N, WIDTH)
    weights = torch.randn(WIDTH, CLASSES)
    truth = (features @ weights).argmax(dim=1)
    trainX, testX = features[:SPLIT], features[SPLIT:]
    trainY, testY = truth[:SPLIT], truth[SPLIT:]

    # A's private class permutation, and its inverse: alias = P[true]  =>  true = Pinv[alias]
    permutation = torch.tensor([2, 0, 3, 1])
    inverse = torch.zeros(CLASSES, dtype=torch.long)
    for trueIndex, aliasIndex in enumerate(permutation.tolist()):
        inverse[aliasIndex] = trueIndex
    trainAlias = permutation[trainY]
    testAlias = permutation[testY]

    # An input-space permutation of A's feature WIDTH: a fixed column permutation.
    columnPermutation = torch.randperm(WIDTH, generator=torch.Generator().manual_seed(7))

    def buildModel():
        torch.manual_seed(1)
        return torch.nn.Sequential(
            torch.nn.Linear(WIDTH, 128), torch.nn.ReLU(), torch.nn.Linear(128, CLASSES))

    def train(model, inputs, labels, epochs=800):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        lossFunction = torch.nn.CrossEntropyLoss()
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = lossFunction(model(inputs), labels)
            loss.backward()
            optimizer.step()

    print("  %-46s %12s %12s" % ("配置", "对别名准确率", "逆映射后"))
    print("  " + "-" * 74)

    # (a) reference: unchanged inputs, TRUE labels
    model = buildModel()
    train(model, trainX, trainY)
    with torch.no_grad():
        reference = float((model(testX).argmax(1) == testY).float().mean())
    print("  %-46s %12s %12.4f" % ("(a) 原样输入 + 真实标签", "-", reference))

    # (b) cutA permutation in the FORWARD, still TRUE labels in the loss
    model = buildModel()
    permutedTrainX = trainX[:, columnPermutation]
    permutedTestX = testX[:, columnPermutation]
    train(model, permutedTrainX, trainY)
    with torch.no_grad():
        onAlias = model(permutedTestX).argmax(1)
        aliasAccuracy = float((onAlias == testAlias).float().mean())
        inverted = inverse[onAlias]
        invertedAccuracy = float((inverted == testY).float().mean())
    print("  %-46s %12.4f %12.4f" % ("(b) cutA 列置换 + 真实标签", aliasAccuracy, invertedAccuracy))

    print()
    print("  判读：")
    print("  * 若 (b) 的>对别名准确率<很高，说明前向置换确实让顶模学到了别名映射。")
    print("  * 若 (b) 的>逆映射后<接近 (a)，说明 A 的真实精度不受损。")
    print("  * 二者同时成立，就证明了「不改 core、只靠 hook 前向置换」可以实现别名训练。")


if __name__ == "__main__":
    main()
