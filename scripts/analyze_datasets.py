"""Analyse why defences transfer unevenly: A/B feature label-information per dataset.

    <python> scripts/analyze_datasets.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from datasets import loadDataset
from core.config import loadConfig


def main():
    print("=== A/B 特征标签信息对比（逻辑回归，训练/验证）===")
    print("  %-11s %-6s %-6s %-8s %-8s %-8s %-10s" % (
        "数据集", "类别", "A维", "B维", "A单独acc", "B单独acc", "B/A信息比"))
    for i in range(1, 6):
        d = f"dataset_0{i}"
        cfg = loadConfig(
            PROJECT_ROOT / "configs" / f"{d}.yaml",
            {"training": {"epochs": 1, "device": "cpu", "progressEnabled": False}},
            enforceFormalLock=False,
        )
        bundle = loadDataset(cfg["dataset"], PROJECT_ROOT / "data" / "datasets")
        Xa = bundle.train.partyA.numpy()
        Xb = bundle.train.partyB.numpy()
        y = bundle.train.labels.numpy().ravel()
        cc = bundle.classCount
        # Collapse any leading singleton dims to 2D [n, d].
        Xa = Xa.reshape(Xa.shape[0], -1)
        Xb = Xb.reshape(Xb.shape[0], -1)

        def fit_acc(X):
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(Xs, y)
            pred = clf.predict(Xs)
            return float((pred == y).mean())

        acc_a = fit_acc(Xa)
        acc_b = fit_acc(Xb)
        random_acc = 1.0 / cc
        # 归一化信息：超出随机的部分
        info_a = (acc_a - random_acc) / (1.0 - random_acc) if acc_a > random_acc else 0.0
        info_b = (acc_b - random_acc) / (1.0 - random_acc) if acc_b > random_acc else 0.0
        ratio = info_b / info_a if info_a > 0 else float("inf")
        print("  %-11s %-6s %-6s %-8s %-8.4f %-8.4f %-10.3f" % (
            d, cc, Xa.shape[1], Xb.shape[1], acc_a, acc_b, ratio))


if __name__ == "__main__":
    main()
