"""A 方防御模块集合。

防御只接入 ``VflTrainer.configurePartyADefenseHooks`` 提供的三个入口，
不修改 B 方底模、B 方私有特征、攻击实现或正式场景协议。
"""

from .purified_vfl import PurifiedVflDefense, buildPartyADefense
from .robust_vfl import RobustVflDefense

__all__ = ["RobustVflDefense", "PurifiedVflDefense", "buildPartyADefense"]
