"""End-to-end check that soft alias encoding works with the REAL defence code.

Trains a tiny model on the defence's soft target, then applies the defence's own
inverseMapPredictions to the argmax and confirms the accuracy is recovered.

    <python> scripts/probe_alias_soft_e2e.py
"""

import sys
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from defense import buildPartyADefense  # noqa: E402


def main():
    torch.manual_seed(0)
    N, WIDTH = 2400, 32
    SPLIT = 1700
    CLASSES = 2

    features = torch.randn(N, WIDTH)
    truth = (features @ torch.randn(WIDTH, CLASSES)).argmax(dim=1)
    trainX, testX = features[:SPLIT], features[SPLIT:]
    trainTrue, testTrue = truth[:SPLIT], truth[SPLIT:]

    print("  %-40s %10s" % ("配置", "逆映射后精度"))
    print("  " + "-" * 52)

    for K in (2, 4):
        outDim = 2 * CLASSES * K
        defense = buildPartyADefense({"name": "purified_vfl", "parameters": {
            "pseudoLabelStrength": 1.0,
            "pseudoLabelClasses": outDim,
            "pseudoTrueClassCount": CLASSES,
            "labelAliasFactor": K,
            "labelAliasSoft": True,
            "pseudoLabelSeed": 11,
        }})
        softTarget = defense.labelTransform(trainTrue)

        torch.manual_seed(1)
        model = torch.nn.Sequential(
            torch.nn.Linear(WIDTH, 256), torch.nn.ReLU(), torch.nn.Linear(256, outDim))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(1500):
            optimizer.zero_grad()
            logProb = torch.log_softmax(model(trainX), dim=1)
            loss = -(softTarget * logProb).sum(dim=1).mean()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            prediction = model(testX).argmax(1).tolist()
        inverted = defense.inverseMapPredictions(prediction)
        accuracy = float((torch.tensor(inverted) == testTrue).float().mean())
        print("  K=%d 软编码(输出 %2d 维)          %10.4f" % (K, outDim, accuracy))


if __name__ == "__main__":
    main()
