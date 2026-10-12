from copy import deepcopy
from pathlib import Path

import torch


def captureModelState(models):
    return {
        "bottomA": deepcopy(models.bottomA.state_dict()),
        "bottomB": deepcopy(models.bottomB.state_dict()),
        "topModel": deepcopy(models.topModel.state_dict()),
    }


def restoreModelState(models, state):
    models.bottomA.load_state_dict(state["bottomA"])
    models.bottomB.load_state_dict(state["bottomB"])
    models.topModel.load_state_dict(state["topModel"])


def saveCheckpoint(path, models, runMetadata):
    checkpointPath = Path(path)
    checkpointPath.parent.mkdir(parents=True, exist_ok=True)
    # OneDrive 同步目录中，PyTorch 的底层路径写入器偶尔无法立即看到刚创建的目录。
    # 先由 Python 打开文件句柄，再交给 torch.save，可避免该文件系统可见性竞争。
    with checkpointPath.open("wb") as handle:
        torch.save({"models": captureModelState(models), "metadata": runMetadata}, handle)
    return checkpointPath
