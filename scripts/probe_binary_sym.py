"""Diagnose why binary symmetric (K=1) still collapses accuracy.

Reproduces the exact labelTransform + trainer loss + inverseMapPredictions path for a binary
dataset, printing the soft targets and the final accuracy.
"""

import sys
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from defense import buildPartyADefense  # noqa: E402

N, WIDTH = 2400, 32
SPLIT = 1700


def main():
    torch.manual_seed(0)
    features = torch.randn(N, WIDTH)
    truth = (features @ torch.randn(WIDTH, 2)).argmax(dim=1)
    trainX, testX = features[:SPLIT], features[SPLIT:]
    trainTrue, testTrue = truth[:SPLIT], truth[SPLIT:]

    defense = buildPartyADefense({"name": "purified_vfl", "parameters": {
        "pseudoLabelStrength": 1.0,
        "pseudoLabelClasses": 2,
        "pseudoTrueClassCount": 2,
        "labelAliasFactor": 1,
        "labelAliasSymmetric": True,
        "labelAliasSigma": 1.0,
        "pseudoLabelSeed": 11,
    }})
    soft = defense.labelTransform(trainTrue)
    print("  软标签形状:", tuple(soft.shape))
    print("  类0 soft:", ['%.3f' % v for v in soft[0].tolist()])
    print("  类1 soft:", ['%.3f' % v for v in soft[1].tolist()])

    # 训练（模拟 trainer 的 2D loss 路径）
    torch.manual_seed(1)
    model = torch.nn.Linear(WIDTH, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(1500):
        optimizer.zero_grad()
        logits = model(trainX)
        logProb = torch.log_softmax(logits, dim=1)
        loss = -(soft * logProb).sum(dim=1).mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        pred = model(testX).argmax(1).tolist()
    inv = defense.inverseMapPredictions(pred)
    acc = float((torch.tensor(inv) == testTrue).float().mean())
    print("  逆映射后精度: %.4f" % acc)
    print("  不逆映射(argmax)精度: %.4f" % float((torch.tensor(pred) == testTrue).float().mean()))

    # 对照：硬 one-hot 训练
    torch.manual_seed(1)
    model2 = torch.nn.Linear(WIDTH, 2)
    optimizer = torch.optim.Adam(model2.parameters(), lr=1e-3)
    for _ in range(1500):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model2(trainX), trainTrue)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        pred2 = model2(testX).argmax(1)
    print("  对照(硬标签)精度: %.4f" % float((pred2 == testTrue).float().mean()))


if __name__ == "__main__":
    main()
