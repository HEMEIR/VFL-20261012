import os
import random

import numpy as np
import torch


def setRandomSeed(seed):
    """统一固定训练所使用的 Python、NumPy、PyTorch 和 CUDA 随机源。"""
    normalizedSeed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(normalizedSeed)
    random.seed(normalizedSeed)
    np.random.seed(normalizedSeed)
    torch.manual_seed(normalizedSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(normalizedSeed)
        torch.cuda.manual_seed_all(normalizedSeed)

    # 关闭自动算法选择，避免 cuDNN 使用非确定性实现。
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {
        "seed": normalizedSeed,
        "pythonHashSeed": os.environ["PYTHONHASHSEED"],
        "cudnnBenchmark": bool(torch.backends.cudnn.benchmark),
        "cudnnDeterministic": bool(torch.backends.cudnn.deterministic),
    }


def createDataLoaderGenerator(seed):
    """为训练集洗牌提供独立生成器，避免其他随机操作改变 batch 顺序。"""
    return torch.Generator().manual_seed(int(seed))


def seedDataLoaderWorker(workerId):
    """让未来启用的多进程 DataLoader worker 继承可复现的随机状态。"""
    del workerId
    workerSeed = int(torch.initial_seed() % (2**32))
    random.seed(workerSeed)
    np.random.seed(workerSeed)
