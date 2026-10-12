import sys

from tqdm import tqdm


def isProgressEnabled(progressConfig):
    """只有配置显式开启时才显示进度，避免测试和库调用产生额外输出。"""
    return bool((progressConfig or {}).get("progressEnabled", False))


def createProgressBar(iterable, progressConfig, description, unit):
    """创建统一样式的单层进度条；调用方负责在循环内更新 postfix。"""
    config = progressConfig or {}
    return tqdm(
        iterable,
        desc=description,
        unit=unit,
        leave=False,
        dynamic_ncols=True,
        mininterval=float(config.get("progressMinInterval", 0.5)),
        disable=not isProgressEnabled(config),
    )


def writeProgress(progressConfig, message):
    """通过 tqdm.write 输出永久日志，避免破坏正在刷新的进度条。"""
    if isProgressEnabled(progressConfig):
        # 与 tqdm 默认进度条同写 stderr，保留 main.py 的 stdout 纯 JSON 输出。
        tqdm.write(str(message), file=sys.stderr)
