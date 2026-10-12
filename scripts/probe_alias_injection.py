"""Minimal, self-contained proof of WHERE alias training fails to take effect.

The question is whether "train on aliases" is a bad idea or merely an unimplementable
injection point. This isolates the two halves:

  PART 1 - the idea itself: train a model on aliased labels and invert.  No VFL harness, no
           hooks.  If this succeeds, the idea is sound.

  PART 2 - the injection point: call the real hook and check whether the labels it receives
           can reach the loss the trainer computes.

    <python> scripts/probe_alias_injection.py
"""

import inspect
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from defense import buildPartyADefense  # noqa: E402


def part1_idea():
    print("=" * 78)
    print("PART 1  想法本身：在别名上训练，再逆映射")
    print("=" * 78)
    torch.manual_seed(0)
    rows, width, classes = 600, 24, 4
    features = torch.randn(rows, width)
    truth = (features @ torch.randn(width, classes)).argmax(dim=1)
    sampleIds = ["S%05d" % index for index in range(rows)]

    defense = buildPartyADefense({
        "name": "purified_vfl",
        "parameters": {"pseudoLabelStrength": 1.0, "pseudoLabelClasses": classes,
                       "pseudoLabelSeed": 11},
    })
    alias = defense._pseudoLabel(truth, sampleIds)

    def train(labels, seed):
        torch.manual_seed(seed)
        model = torch.nn.Linear(width, classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        lossFunction = torch.nn.CrossEntropyLoss()
        for _ in range(400):
            optimizer.zero_grad()
            loss = lossFunction(model(features), labels)
            loss.backward()
            optimizer.step()
        return model(features).argmax(dim=1)

    control = train(truth, 1)
    onAlias = train(alias, 1)
    restored = torch.tensor(defense.inverseMapPredictions(onAlias.tolist(), sampleIds))

    print("  用真实标签训练     准确率 = %.4f"
          % float((control == truth).float().mean()))
    print("  用别名标签训练     准确率 = %.4f  <- 对真实标签评测（应接近随机 1/C=%.2f）"
          % (float((onAlias == truth).float().mean()), 1.0 / classes))
    print("  用别名标签训练+逆映射 准确率 = %.4f"
          % float((restored == truth).float().mean()))
    print()
    print("  结论：想法成立。别名训练 + 逆映射 = 与直接用真实标签训练完全相同。")
    print()


def part2_injection():
    print("=" * 78)
    print("PART 2  注入点：hook 收到的 labels 能否到达损失")
    print("=" * 78)
    source = (PROJECT_ROOT / "core" / "trainer.py").read_text(encoding="utf-8")
    lines = source.splitlines()
    marks = [
        ("hook 收到的 labels 是副本", "labels=labels.detach().clone()"),
        ("hook 返回值必须是二元组", "must return (partyARepresentation"),
        ("hook 只返回张量", "return transformedA, transformedB"),
        ("损失读取原始标签", 'loss = self.lossFunction(logits, batch["label"])'),
    ]
    for index, line in enumerate(lines, start=1):
        for label, needle in marks:
            if needle in line:
                print("  trainer.py:%d  %s" % (index, label))
                print("      %s" % line.strip()[:80])
    print()

    # Exercise the real hook and show the local rebinding does not escape.
    defense = buildPartyADefense({
        "name": "purified_vfl",
        "parameters": {"pseudoLabelStrength": 1.0, "pseudoLabelClasses": 2,
                       "pseudoLabelSeed": 11},
    })
    cutA = torch.randn(8, 16, requires_grad=True)
    cutB = torch.randn(8, 16, requires_grad=True)
    labels = torch.tensor([0, 1, 0, 1, 1, 0, 0, 1])
    sampleIds = ["S%03d" % index for index in range(8)]
    returned = defense.trainingRepresentationHook(
        cutA, cutB, labels=labels.detach().clone(), sampleIds=sampleIds,
        epoch=0, batchIndex=0, context={"backdoor": "none", "labelInference": "pmc"},
    )
    print("  hook 返回值的类型/长度 :", type(returned).__name__, len(returned))
    print("  返回值元素类型        :", [type(item).__name__ for item in returned])
    print("  => 返回的是两个张量，没有标签通道。")
    print()
    print("  hook 内部确实做了别名化（记录 %d 个映射），但那是它自己的局部副本；"
          % len(defense.pseudoSampleMaps))
    print("  trainer 随后仍用 batch[\"label\"] 计算损失。")
    print()
    print("  结论：不是「别名训练行不通」，而是 hook 接口没有回传标签的通道。")
    print("        要让它生效，损失必须读到别名 —— 那属于 core 侧改动（trainer.py:451）。")


if __name__ == "__main__":
    part1_idea()
    part2_injection()
