"""Soft vs hard alias encoding: learnability and invertibility.

The proposal: instead of mapping each sample to ONE hard alias (hard encoding), map it to a
SOFT distribution over its class's alias block (e.g. class 0 -> uniform over {4,5}).

Hard encoding was measured to have two real problems in the VFL pilot:
  * dataset_02_amc K=2: accuracy collapsed 0.8951 -> 0.7779
  * dataset_05_pmc (6-class) K=4: LRA worsened 0.6495 -> 0.7738

This toy test answers, BEFORE touching the harness, whether soft encoding is more learnable
and still invertible.  It compares:

  (a) hard true labels        (baseline)
  (b) HARD alias, K per class : target = one random alias inside the class's block
  (c) SOFT alias, K per class : target = uniform over the class's block

The inverse for both (b) and (c) is: softmax -> sum probabilities inside each block -> argmax.
This is what A would do at inference.

    <python> scripts/probe_alias_soft.py
"""

import torch

N, WIDTH = 2400, 32
SPLIT = 1700
CLASSES = 2


def trainAndScore(trainX, targets, testX, nOut, epochs=1500, hidden=256):
    """targets: either integer tensor (hard) or float tensor [N, nOut] (soft)."""
    torch.manual_seed(1)
    model = torch.nn.Sequential(
        torch.nn.Linear(WIDTH, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, nOut))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(trainX)
        if targets.dim() == 2:
            # soft target: KL divergence with log-softmax
            logProb = torch.log_softmax(logits, dim=1)
            loss = -(targets * logProb).sum(dim=1).mean()
        else:
            loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return torch.softmax(model(testX), dim=1)


def main():
    torch.manual_seed(0)
    features = torch.randn(N, WIDTH)
    truth = (features @ torch.randn(WIDTH, CLASSES)).argmax(dim=1)
    trainX, testX = features[:SPLIT], features[SPLIT:]
    trainTrue, testTrue = truth[:SPLIT], truth[SPLIT:]

    # (a) baseline
    probs = trainAndScore(trainX, trainTrue, testX, CLASSES)
    baseline = float((probs.argmax(1) == testTrue).float().mean())
    print("  (a) 硬真实标签           精度 %.4f" % baseline)
    print()

    print("  %-34s %10s %10s %10s" % ("配置", "块内 argmax", "块内求和", "备注"))
    print("  " + "-" * 68)

    for K in (2, 4):
        nOut = CLASSES * K

        # (b) hard alias: random pick inside block
        g = torch.Generator().manual_seed(11)
        hardAlias = (trainTrue * K + torch.randint(0, K, (SPLIT,), generator=g))
        probs = trainAndScore(trainX, hardAlias, testX, nOut)
        # inverse by block-argmax: predicted block = argmax over block-sums
        blockSums = torch.stack([probs[:, c * K:(c + 1) * K].sum(1) for c in range(CLASSES)], dim=1)
        hardInv = float((blockSums.argmax(1) == testTrue).float().mean())
        hardMax = float((probs.argmax(1) // K == testTrue).float().mean())

        # (c) soft alias: uniform over block
        softTarget = torch.zeros(SPLIT, nOut)
        for c in range(CLASSES):
            mask = (trainTrue == c)
            softTarget[mask, c * K:(c + 1) * K] = 1.0 / K
        probs = trainAndScore(trainX, softTarget, testX, nOut)
        blockSums = torch.stack([probs[:, c * K:(c + 1) * K].sum(1) for c in range(CLASSES)], dim=1)
        softInv = float((blockSums.argmax(1) == testTrue).float().mean())

        print("  K=%d 硬编码(每类%d别名)     %10.4f %10.4f %s" % (
            K, K, hardMax, hardInv, "随机pick"))
        print("  K=%d 软编码(每类%d别名)     %10.4f %10.4f %s" % (
            K, K, float((probs.argmax(1) // K == testTrue).float().mean()), softInv, "soft target"))

    print()
    print("  判读：")
    print("  * >块内求和 argmax< 是 A 实际可用的逆映射（对软编码最自然）。")
    print("  * 若软编码的逆映射精度 >= 硬编码，且更稳定，则软编码值得实现。")


if __name__ == "__main__":
    main()
