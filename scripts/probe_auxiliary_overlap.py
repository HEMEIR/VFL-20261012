"""Check whether the auxiliary (public-label) samples are part of the main task too.

This decides whether A can degrade those samples' embeddings without hurting the main
task: if the same 20 samples also carry the joint cross-entropy loss, any damage A
inflicts on them is paid for twice.

    <python> scripts/probe_auxiliary_overlap.py
"""

import csv
import io
import glob
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def idsOf(path):
    with io.open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return [row[0] for row in reader]


def main():
    for dataset in sorted(glob.glob(str(PROJECT_ROOT / "data" / "datasets" / "dataset_0*"))):
        name = Path(dataset).name
        auxPath = PROJECT_ROOT / "configs" / "auxiliary_labels" / f"{name}_pmc.csv"
        if not auxPath.is_file():
            continue
        with io.open(auxPath, encoding="utf-8-sig", newline="") as handle:
            aux = {row["ID"]: row["Label"] for row in csv.DictReader(handle)}
        trainPath = Path(dataset) / "train" / "party_A_train.csv"
        if not trainPath.is_file():
            continue
        trainIds = set(idsOf(trainPath))
        labelPath = Path(dataset) / "train" / "party_A_train.csv"
        # A's own label column: find it by header
        with io.open(labelPath, encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames
            labelColumn = header[-1]
        with io.open(labelPath, encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            own = {row["ID"]: row[labelColumn] for row in reader}
        overlap = sorted(set(aux) & set(own))
        agree = sum(1 for k in overlap if str(own[k]) == str(aux[k]))
        print("  %-11s 辅助集 %2d 个样本, 全在 A 训练集: %s, 标签与 A 自身一致: %d/%d"
              % (name, len(aux), len(overlap) == len(aux), agree, len(overlap)))
        print("              A 侧标签列名: %s" % labelColumn)


if __name__ == "__main__":
    main()
