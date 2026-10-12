from dataclasses import dataclass, field


@dataclass
class ModelBundle:
    bottomA: object
    bottomB: object
    topModel: object


@dataclass
class TrainingHistory:
    epochs: list = field(default_factory=list)
    bestEpoch: int = -1
    bestValidationAccuracy: float = float("-inf")
    bestValidationAuc: float = float("-inf")
    bestSelectionScore: float = float("-inf")


class TrainingBatchResult(float):
    """兼容旧 float 返回值，同时携带统一 batch 执行器的附加结果。"""

    def __new__(cls, loss, passiveCutGradient=None, metadata=None):
        instance = super().__new__(cls, float(loss))
        instance.loss = float(loss)
        instance.passiveCutGradient = passiveCutGradient
        instance.metadata = dict(metadata or {})
        return instance

    def __init__(self, loss, passiveCutGradient=None, metadata=None):
        # float 是不可变类型，全部初始化已在 __new__ 中完成。
        pass
