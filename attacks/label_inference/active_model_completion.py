from dataclasses import dataclass

import torch

from .model_completion import runPassiveModelCompletion


@dataclass
class GradientScaleSummary:
    step: int
    meanFactor: float
    minimumFactor: float
    maximumFactor: float
    acceleratedFraction: float


class AdaptiveMaliciousGradientScaler:
    """在 Adam 更新前执行 AMC 的逐参数自适应梯度放大机制。"""

    def __init__(self, parameters, config):
        self.parameters = [parameter for parameter in parameters if parameter.requires_grad]
        self.momentum = float(config.get("maliciousMomentum", 0.9))
        self.dampening = float(config.get("maliciousDampening", 0.0))
        self.gamma = float(config.get("gradientScaleGamma", 1.0))
        self.minimumScale = float(config.get("minimumGradientScale", 1.0))
        self.maximumScale = float(config.get("maximumGradientScale", 5.0))
        self.epsilon = float(config.get("gradientScaleEpsilon", 1e-7))
        # 配置使用一基轮次：1 表示从首轮启用，2 表示首轮保持干净更新、
        # 从第二轮开始执行 AMC 梯度放大。
        self.activeStartEpoch = int(config.get("activeStartEpoch", 1))
        self.lastUnscaledDirections = {}
        self.stepCount = 0
        self.factorSum = 0.0
        self.factorElementCount = 0
        self.acceleratedElementCount = 0
        self.observedMinimumScale = float("inf")
        self.observedMaximumScale = float("-inf")
        self.firstStepRecords = []
        self.lastStepRecords = []

    def isActiveAtEpoch(self, epoch):
        """判断零基训练轮次是否已经进入 AMC 主动攻击阶段。"""
        return int(epoch) + 1 >= self.activeStartEpoch

    def scaleGradients(self):
        factorSum = 0.0
        factorElementCount = 0
        acceleratedElementCount = 0
        minimumFactor = float("inf")
        maximumFactor = float("-inf")
        for parameterIndex, parameter in enumerate(self.parameters):
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach()
            self.ensureFinite(gradient, parameterIndex, "input gradient")
            previousDirection = self.lastUnscaledDirections.get(parameterIndex)
            if previousDirection is None:
                # 第一次更新只记录方向，不进行放大。
                currentDirection = gradient.clone()
                factor = torch.ones_like(gradient)
            else:
                currentDirection = (
                    self.momentum * previousDirection
                    + (1.0 - self.dampening) * gradient
                )
                self.ensureFinite(
                    currentDirection,
                    parameterIndex,
                    "unscaled momentum direction",
                )
                factor = 1.0 + self.gamma * (
                    currentDirection / (previousDirection + self.epsilon)
                )
                # 接近零的坐标可能产生非有限比例；按配置上下界处理正负无穷，
                # 无法确定方向的 NaN 坐标不进行放大。
                factor = torch.nan_to_num(
                    factor,
                    nan=1.0,
                    posinf=self.maximumScale,
                    neginf=self.minimumScale,
                )
                factor = torch.clamp(
                    factor,
                    min=self.minimumScale,
                    max=self.maximumScale,
                )
            scaledGradient = gradient * factor
            self.ensureFinite(
                scaledGradient,
                parameterIndex,
                "scaled Adam input gradient",
            )
            parameter.grad.copy_(scaledGradient)
            # 只缓存未放大的方向。若缓存 scaledGradient，下一批会再次包含
            # 上一批放大结果，从而形成与 Adam 状态叠加的递归放大。
            self.lastUnscaledDirections[parameterIndex] = (
                currentDirection.detach().clone()
            )

            factorSum += float(factor.double().sum().item())
            factorElementCount += factor.numel()
            acceleratedElementCount += int((factor > 1.0).sum().item())
            minimumFactor = min(minimumFactor, float(factor.min().item()))
            maximumFactor = max(maximumFactor, float(factor.max().item()))

        self.stepCount += 1
        if factorElementCount == 0:
            summary = GradientScaleSummary(self.stepCount, 1.0, 1.0, 1.0, 0.0)
        else:
            summary = GradientScaleSummary(
                step=self.stepCount,
                meanFactor=factorSum / factorElementCount,
                minimumFactor=minimumFactor,
                maximumFactor=maximumFactor,
                acceleratedFraction=acceleratedElementCount / factorElementCount,
            )
            self.factorSum += factorSum
            self.factorElementCount += factorElementCount
            self.acceleratedElementCount += acceleratedElementCount
            self.observedMinimumScale = min(self.observedMinimumScale, minimumFactor)
            self.observedMaximumScale = max(self.observedMaximumScale, maximumFactor)
        self.recordStep(summary)
        return summary

    def ensureFinite(self, values, parameterIndex, valueName):
        if torch.isfinite(values).all():
            return
        nonFiniteCount = int((~torch.isfinite(values)).sum().item())
        raise FloatingPointError(
            "AMC produced non-finite "
            f"{valueName} at step {self.stepCount + 1}, "
            f"parameter {parameterIndex}; nonFiniteCount={nonFiniteCount}"
        )

    def recordStep(self, summary):
        record = {
            "step": summary.step,
            "meanFactor": summary.meanFactor,
            "minimumFactor": summary.minimumFactor,
            "maximumFactor": summary.maximumFactor,
            "acceleratedFraction": summary.acceleratedFraction,
        }
        if len(self.firstStepRecords) < 5:
            self.firstStepRecords.append(record)
        self.lastStepRecords.append(record)
        if len(self.lastStepRecords) > 5:
            self.lastStepRecords.pop(0)

    def buildAuditRecords(self):
        divisor = max(self.factorElementCount, 1)
        return {
            "maliciousParty": "B, the passive party without labels",
            "affectedModel": "bottomB only",
            "baseOptimizer": "Adam",
            "optimizerAdaptation": (
                "the adjacent-update ratio is estimated from an unscaled malicious "
                "momentum direction; the factor scales only the current raw bottomB "
                "gradient before the Adam step"
            ),
            "historyState": "unscaledMomentumDirection",
            "scaledStateIsNotReused": True,
            "nonFinitePolicy": "failImmediatelyBeforeAdamStep",
            "maliciousMomentum": self.momentum,
            "maliciousDampening": self.dampening,
            "gradientScaleGamma": self.gamma,
            "minimumGradientScale": self.minimumScale,
            "maximumGradientScale": self.maximumScale,
            "gradientScaleEpsilon": self.epsilon,
            "activeStartEpoch": self.activeStartEpoch,
            "stepCount": self.stepCount,
            "observedMeanScale": self.factorSum / divisor,
            "observedMinimumScale": (
                self.observedMinimumScale if self.factorElementCount else 1.0
            ),
            "observedMaximumScale": (
                self.observedMaximumScale if self.factorElementCount else 1.0
            ),
            "acceleratedElementFraction": self.acceleratedElementCount / divisor,
            "firstSteps": self.firstStepRecords,
            "lastSteps": self.lastStepRecords,
        }


def runActiveModelCompletion(
    bottomB,
    bundle,
    labelInferenceConfig,
    configPath,
    device,
    seed,
    gradientScaler,
    progressConfig=None,
):
    """AMC 在恶意 VFL 更新后执行与 PMC 相同的半监督模型补全。"""
    result = runPassiveModelCompletion(
        bottomB,
        bundle,
        labelInferenceConfig,
        configPath,
        device,
        seed,
        progressConfig,
        "AMC",
    )
    result.auditRecords.update(
        {
            "attack": "amc",
            "threatModel": (
                "active malicious bottomB optimization during VFL training, followed "
                "by post-training model completion"
            ),
            "mutatesVflBottomB": True,
            "modelCompletionUsesBottomBCopy": True,
            "activeTrainingChangesBottomBUpdates": True,
            "activeTraining": gradientScaler.buildAuditRecords(),
        }
    )
    return result
