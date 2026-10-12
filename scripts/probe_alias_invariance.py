"""Standalone check that aliasing every label preserves the task exactly.

Claim under test: because each sample's alias map is a bijection, a model trained on alias
labels and then inverted must recover EXACTLY the same accuracy as a model trained on true
labels.  "Not collapsed" is not enough - the requirement is equality.

This trains the same small classifier twice on identical data (true labels vs aliased
labels), then inverts the aliased model's predictions and compares accuracies.

    <python> scripts/probe_alias_invariance.py
"""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from defense import buildPartyADefense  # noqa: E402


def makeData(rows, width, classes, seed):
    generator = torch.Generator().manual_seed(seed)
    weights = torch.randn(width, classes, generator=generator)
    features = torch.randn(rows, width, generator=generator)
    labels = (features @ weights).argmax(dim=1)
    return features, labels


def trainAndScore(features, labels, classes, seed):
    torch.manual_seed(seed)
    model = torch.nn.Linear(features.shape[1], classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    lossFunction = torch.nn.CrossEntropyLoss()
    for _ in range(300):
        optimizer.zero_grad()
        loss = lossFunction(model(features), labels)
        loss.backward()
        optimizer.step()
    return model(features).argmax(dim=1)


def main():
    rows, width, classes = 800, 32, 4
    features, trueLabels = makeData(rows, width, classes, seed=0)
    sampleIds = ["S%05d" % index for index in range(rows)]

    print("  %-28s %10s %10s %10s" % ("配置", "训练用标签", "准确率", "往返一致"))
    print("  " + "-" * 62)

    # (a) control: trained on true labels
    controlPredictions = trainAndScore(features, trueLabels, classes, seed=1)
    controlAccuracy = float((controlPredictions == trueLabels).float().mean())
    print("  %-28s %10s %10.4f %10s"
          % ("(a) 真实标签训练", "真实", controlAccuracy, "-"))

    # (b) trained on aliases, scored WITHOUT inversion
    defense = buildPartyADefense({
        "name": "purified_vfl",
        "parameters": {"pseudoLabelStrength": 1.0, "pseudoLabelClasses": classes,
                       "pseudoLabelSeed": 11},
    })
    aliasLabels = defense._pseudoLabel(trueLabels, sampleIds)
    aliasPredictions = trainAndScore(features, aliasLabels, classes, seed=1)
    aliasAccuracy = float((aliasPredictions == trueLabels).float().mean())
    print("  %-28s %10s %10.4f %10s"
          % ("(b) 别名标签训练（不逆映射）", "别名", aliasAccuracy, "-"))

    # (c) same model, predictions inverted
    restored = defense.inverseMapPredictions(aliasPredictions.tolist(), sampleIds)
    restoredTensor = torch.tensor(restored)
    restoredAccuracy = float((restoredTensor == trueLabels).float().mean())
    equal = restoredAccuracy == controlAccuracy
    print("  %-28s %10s %10.4f %10s"
          % ("(c) 别名标签训练（逆映射后）", "别名", restoredAccuracy,
             "是" if equal else "**否**"))

    print()
    print("  (b) 应接近随机水平 1/C = %.4f，说明别名确实遮蔽了真实标签" % (1.0 / classes))
    print("  (c) 应与 (a) 完全相等，说明双射保持任务、逆映射无损")
    print()
    print("  结论: 逆映射后与基线完全一致 = %s" % equal)
    print("        别名训练后不逆映射 = %.4f（随机水平 %.4f）"
          % (aliasAccuracy, 1.0 / classes))

    # Same check for the binary case, where all samples share one permutation.
    print()
    print("  === 二分类（C=2，全部样本共用同一置换）===")
    binaryTrue = (trueLabels % 2)
    binaryIds = sampleIds
    defense2 = buildPartyADefense({
        "name": "purified_vfl",
        "parameters": {"pseudoLabelStrength": 1.0, "pseudoLabelClasses": 2,
                       "pseudoLabelSeed": 11},
    })
    binaryAlias = defense2._pseudoLabel(binaryTrue, binaryIds)
    binaryAliasPredictions = trainAndScore(features, binaryAlias, 2, seed=1)
    binaryControl = trainAndScore(features, binaryTrue, 2, seed=1)
    binaryControlAccuracy = float((binaryControl == binaryTrue).float().mean())
    binaryRestored = torch.tensor(
        defense2.inverseMapPredictions(binaryAliasPredictions.tolist(), binaryIds))
    binaryRestoredAccuracy = float((binaryRestored == binaryTrue).float().mean())
    print("    基线准确率        = %.4f" % binaryControlAccuracy)
    print("    不逆映射          = %.4f"
          % float((binaryAliasPredictions == binaryTrue).float().mean()))
    print("    逆映射后          = %.4f" % binaryRestoredAccuracy)
    print("    往返一致 = %s" % (binaryRestoredAccuracy == binaryControlAccuracy))


if __name__ == "__main__":
    main()
