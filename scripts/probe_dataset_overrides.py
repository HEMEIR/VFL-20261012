import csv
import io
import json
import sys

import torch

sys.path.insert(0, '.')
from defense import buildPartyADefense

OVERRIDES = json.dumps({'dataset_0%d' % i: {'partyDropoutRate': 0.0} for i in range(1, 6)})


def load(dataset, n=8):
    ids = []
    rows = []
    path = 'data/datasets/%s/train/party_A_train.csv' % dataset
    with io.open(path, encoding='utf-8-sig', newline='') as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for index, row in enumerate(reader):
            ids.append(row[0])
            rows.append([float(value) for value in row[1:]])
            if index >= n - 1:
                break
    return ids, torch.tensor(rows)


def main():
    probe = buildPartyADefense({'name': 'purified_vfl',
                                'parameters': {'datasetOverrides': OVERRIDES}})
    signatures = probe._buildDatasetSignatures()
    print('  签名宽度:', {k: v[0] for k, v in signatures.items()})
    print()
    print('  === 各数据集能否被正确判别 ===')
    for dataset in ['dataset_0%d' % i for i in range(1, 6)]:
        ids, tensor = load(dataset)
        instance = buildPartyADefense({'name': 'purified_vfl',
                                       'parameters': {'datasetOverrides': OVERRIDES}})
        got = instance._detectDataset(ids, tensor)
        mark = 'OK' if got == dataset else '**错误**'
        print('    %s 宽%4d -> %-12s %s' % (dataset, tensor.shape[1], got, mark))

    print()
    print('  === 错配保护：用 dataset_04 的值 + 只配 dataset_02 ===')
    onlyTwo = json.dumps({'dataset_02': {'partyDropoutRate': 0.0}})
    ids, tensor = load('dataset_04')
    guard = buildPartyADefense({'name': 'purified_vfl',
                                'parameters': {'partyDropoutRate': 0.2,
                                               'datasetOverrides': onlyTwo}})
    got = guard._detectDataset(ids, tensor)
    print('    判别结果: %s （应为 None，即拒绝错误匹配）' % got)
    print('    原因: %s' % guard.lastDetectionNote)
    print('    partyDropoutRate 未被污染，仍为全局值 0.2: %s'
          % (guard.partyDropoutRate == 0.2))

    print()
    print('  === 正确匹配时覆盖生效 ===')
    overrideAll = json.dumps({'dataset_04': {'partyDropoutRate': 0.0}})
    applied = buildPartyADefense({'name': 'purified_vfl',
                                  'parameters': {'partyDropoutRate': 0.2,
                                                 'datasetOverrides': overrideAll}})
    applied._detectDataset(ids, tensor)
    print('    partyDropoutRate: 0.2 -> %s' % applied.partyDropoutRate)
    print('    已应用: %s' % applied.lastOverrideApplied)
    print('    被拒: %s' % applied.lastOverrideRejected)


if __name__ == '__main__':
    main()
