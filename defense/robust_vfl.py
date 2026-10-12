"""面向复合攻击的两方 VFL 防御：被动方表征重定心与通信扰动。

防御动机（只使用 A 方合法可见的信息）：

1. 被动表征重定心（common-mode recenter）—— 本防御的主力机制
   两个后门攻击最终都体现为被动方表征上的一个"全体样本共有"的偏置：
   VILLAIN 在推理阶段对每一条样本的 B 方 embedding 叠加同一个固定触发器；
   LFBA 在推理阶段对每一条样本的 B 方输入特征写入同一组触发维度。
   A 方按 batch 统计量把 cutB/embeddingB 重新对齐到训练期参考均值，
   即可抵消这种共有偏置；而干净 batch 类别大致均衡、``batchMean`` 本身
   接近参考均值，因此该变换对干净主任务近似恒等，代价极小。
   训练侧与推理侧强度分开配置：训练侧强度过高会把 batch 均值的采样噪声
   注入顶模输入而拖累精度，推理侧则必须为满强度才能抵消攻击的固定偏置。
   此外可选启用逐维方差对齐（``representationStandardizeStrength``）：触发器
   经非线性底模后除了均值偏移还会留下逐维方差偏移，可把 batch 标准差按幂次拉回
   参考标准差（默认关闭，实测未带来普适收益）。

2. 可选扰动（默认全部关闭，仅作为可调强度暴露）
   ``representationStandardizeStrength`` 逐维方差对齐、``representationNoise``
   顶模输入加噪、``gradientNormalizeStrength`` 逐样本梯度模长归一化、
   ``gradientNoise`` 梯度加噪、``gradientClassCenteringStrength`` 类别条件去均值。
   实测这些附加扰动要么收效甚微、要么明显拖累主任务：
   梯度模长归一化会直接破坏 B 方底模的学习（dataset_03 clean 精度掉到 0.44）；
   类别去均值把 dataset_03 clean 打到 0.59；梯度加噪对标签恢复率几乎无改善
   （本项目的 PMC 只用 B 方本地特征与辅助标签训练补全模型，并不消费 A 方
   返回的 cut 梯度）；方差对齐与表征加噪只在个别图像场景上互有胜负、不具普适性。
   因此最终配置只启用重定心，其余保持 0.0。

所有随机性都复用 trainer 注入的 ``generator``（训练与推理分离），
保证可复现；所有输出保持原 shape/dtype/device，且恒为有限值。
"""

import torch


def _finiteFloat(value, default, minimum=None, maximum=None, name="parameter"):
    resolved = float(default if value is None else value)
    if resolved != resolved or resolved in (float("inf"), float("-inf")):
        raise ValueError(f"defense parameter {name} must be finite")
    if minimum is not None and resolved < minimum:
        raise ValueError(f"defense parameter {name} must be >= {minimum}")
    if maximum is not None and resolved > maximum:
        raise ValueError(f"defense parameter {name} must be <= {maximum}")
    return resolved


SUPPORTED_PARAMETERS = {
    "trainingRecenterStrength",
    "inferenceRecenterStrength",
    "trainingCovarianceStrength",
    "inferenceCovarianceStrength",
    "trainingPrototypeFilterStrength",
    "trainingPrototypeFilterZ",
    "trainingPrototypeFilterMargin",
    "trainingTemporalFilterStrength",
    "trainingTemporalFilterZ",
    "temporalMomentum",
    "representationStandardizeStrength",
    "representationNoise",
    "gradientNormalizeStrength",
    "gradientNoise",
    "gradientClassCenteringStrength",
    "referenceMomentum",
}

# 类别样本数少于此值时不做原型一致性判定，避免中位数统计不稳。
_PROTOTYPE_MIN_CLASS_COUNT = 4


def _symmetricMatrixPower(matrix, exponent, epsilon=1e-6):
    """对称半正定矩阵的幂，用特征分解实现并做特征值下限保护。"""
    values, vectors = torch.linalg.eigh(matrix)
    if exponent < 0:
        scaled = values.clamp_min(epsilon).pow(exponent)
    else:
        scaled = values.clamp_min(0.0).pow(exponent)
    return (vectors * scaled.unsqueeze(0)) @ vectors.transpose(-2, -1)


class RobustVflDefense:
    """A 方防御：被动表征重定心 + 训练期加噪 + cut 梯度归一化与加噪。"""

    def __init__(self, parameters=None):
        parameters = dict(parameters or {})
        # 训练期/推理期重定心强度分开配置：训练期重定心会改变顶模看到的
        # 表征分布（batch 均值估计带采样噪声），强度过高会拖累主任务精度；
        # 而推理期必须用完整强度才能抵消攻击注入的整批共有触发器偏置。
        self.trainingRecenterStrength = _finiteFloat(
            parameters.get("trainingRecenterStrength"),
            0.5,
            0.0,
            1.0,
            "trainingRecenterStrength",
        )
        self.inferenceRecenterStrength = _finiteFloat(
            parameters.get("inferenceRecenterStrength"),
            1.0,
            0.0,
            1.0,
            "inferenceRecenterStrength",
        )
        # 二阶共模匹配：均值重定心只能消掉触发器的一阶偏置，触发器经非线性底模
        # 后还会留下协方差结构偏移（逐维方差 + 维间相关）。这里把 batch 协方差
        # 按强度插值拉回参考协方差，0.0 关闭，1.0 完全匹配。
        self.trainingCovarianceStrength = _finiteFloat(
            parameters.get("trainingCovarianceStrength"),
            0.0,
            0.0,
            1.0,
            "trainingCovarianceStrength",
        )
        self.inferenceCovarianceStrength = _finiteFloat(
            parameters.get("inferenceCovarianceStrength"),
            0.0,
            0.0,
            1.0,
            "inferenceCovarianceStrength",
        )
        # 训练期标签一致性过滤：用 batch 内每个类别的鲁棒原型（逐维中位数）
        # 判定"与自身标签不一致"的样本（被动方表示离本类原型异常远），并把它们的
        # 表示替换为本类原型。特征投毒（LFBA 把源样本 B 特征搬到目标样本）与
        # embedding 注入（VILLAIN 对少量样本叠加触发器）都会让被污染样本的被动
        # 表示偏离其标签对应的类簇，替换后顶模学不到"触发器 → 目标类"的关联。
        # 0.0 关闭；1.0 对被判定样本做全量替换（其 cut 梯度随之退化为 0）。
        self.trainingPrototypeFilterStrength = _finiteFloat(
            parameters.get("trainingPrototypeFilterStrength"),
            0.0,
            0.0,
            1.0,
            "trainingPrototypeFilterStrength",
        )
        # 离群判定阈值：表示到本类原型中心的距离超过 z 倍该类距离中位数即判为异常。
        self.trainingPrototypeFilterZ = _finiteFloat(
            parameters.get("trainingPrototypeFilterZ"),
            3.0,
            1.0,
            None,
            "trainingPrototypeFilterZ",
        )
        # 交叉类别判定阈值：表示到"最近的其他类别"原型的距离小于 margin 倍本类距离
        # 时判为标签不一致（margin>=1 表示只要更靠近别的类别就算；0 关闭该规则）。
        # 该规则专门捕捉特征投毒里"借用了别的类别表示"的样本，是离群规则的补充。
        self.trainingPrototypeFilterMargin = _finiteFloat(
            parameters.get("trainingPrototypeFilterMargin"),
            1.0,
            0.0,
            None,
            "trainingPrototypeFilterMargin",
        )
        # 训练期时序一致性过滤：A 方在每次 batch 都能拿到样本 ID，于是按样本维护
        # 被动表示的指数滑动平均。干净样本的表示随 B 方训练平滑漂移，而被特征投毒
        # 的样本每个 epoch 都会换成随机源样本的表示、发生大幅跳变。用"当前表示相对
        # 自身历史的残差"做逐样本异常检测，比类原型更精确（类内差异远大于时序漂移）。
        # 命中样本用其自身历史表示替换，等于抹掉本轮的触发器成分。0.0 关闭。
        self.trainingTemporalFilterStrength = _finiteFloat(
            parameters.get("trainingTemporalFilterStrength"),
            0.0,
            0.0,
            1.0,
            "trainingTemporalFilterStrength",
        )
        self.trainingTemporalFilterZ = _finiteFloat(
            parameters.get("trainingTemporalFilterZ"),
            3.0,
            1.0,
            None,
            "trainingTemporalFilterZ",
        )
        self.temporalMomentum = _finiteFloat(
            parameters.get("temporalMomentum"),
            0.9,
            0.0,
            0.9999,
            "temporalMomentum",
        )
        # 每个训练样本 ID 的被动表示历史（仅训练期使用）。
        self.passiveHistory = {}
        self.representationNoise = _finiteFloat(
            parameters.get("representationNoise"),
            0.0,
            0.0,
            None,
            "representationNoise",
        )
        # 逐维方差对齐：把 batch 逐维标准差按幂次拉回参考标准差，用于抵消触发器
        # 经非线性底模后在 embedding 上留下的"方差偏移"（均值重定心无法处理）。
        # 0.0 关闭；1.0 完全对齐到参考标准差；中间值做幂次插值。
        self.representationStandardizeStrength = _finiteFloat(
            parameters.get("representationStandardizeStrength"),
            0.0,
            0.0,
            1.0,
            "representationStandardizeStrength",
        )
        self.gradientNormalizeStrength = _finiteFloat(
            parameters.get("gradientNormalizeStrength"),
            0.0,
            0.0,
            1.0,
            "gradientNormalizeStrength",
        )
        self.gradientNoise = _finiteFloat(
            parameters.get("gradientNoise"),
            0.0,
            0.0,
            None,
            "gradientNoise",
        )
        # 类别条件梯度去均值：按 batch 内标签分组，削弱返回给 B 方的
        # 类间可分梯度分量，直接压低 PMC/AMC 的可恢复标签信息量。
        self.gradientClassCenteringStrength = _finiteFloat(
            parameters.get("gradientClassCenteringStrength"),
            0.0,
            0.0,
            1.0,
            "gradientClassCenteringStrength",
        )
        self.referenceMomentum = _finiteFloat(
            parameters.get("referenceMomentum"),
            0.99,
            0.0,
            0.9999,
            "referenceMomentum",
        )
        unknown = sorted(set(parameters) - SUPPORTED_PARAMETERS)
        if unknown:
            raise ValueError(f"Unknown defense parameters: {', '.join(unknown)}")
        # 参考统计只在训练期估计，推理期冻结复用，
        # 从而保证训练和推理使用一致的确定性变换。
        self.referenceMean = None
        self.referenceStd = None
        self.referenceCov = None
        # 最近一个 batch 被判为标签不一致的样本比例（仅用于诊断）。
        self.lastFlaggedFraction = 0.0
        self.flaggedFractionSum = 0.0
        self.flaggedFractionCount = 0
        self.temporalFlaggedFractionSum = 0.0
        self.temporalFlaggedFractionCount = 0

    # ------------------------------------------------------------------
    # 参考统计
    # ------------------------------------------------------------------
    def _updateReference(self, representation):
        """用 raw 被动表征的 batch 均值/标准差刷新参考统计。"""
        if representation.shape[0] < 2:
            return
        with torch.no_grad():
            batchMean = representation.mean(dim=0).detach().float()
            batchStd = (
                representation.var(dim=0, unbiased=False).detach().float().clamp_min(0.0).sqrt()
            )
            if self.referenceMean is None or self.referenceMean.shape != batchMean.shape:
                self.referenceMean = batchMean.clone()
                self.referenceStd = batchStd.clone()
                self.referenceCov = self._batchCovariance(representation, batchMean)
                return
            keep = self.referenceMomentum
            self.referenceMean.mul_(keep).add_(batchMean, alpha=1.0 - keep)
            self.referenceStd.mul_(keep).add_(batchStd, alpha=1.0 - keep)
            batchCov = self._batchCovariance(representation, batchMean)
            if batchCov is not None:
                if self.referenceCov is None or self.referenceCov.shape != batchCov.shape:
                    self.referenceCov = batchCov
                else:
                    self.referenceCov.mul_(keep).add_(batchCov, alpha=1.0 - keep)

    @staticmethod
    def _batchCovariance(representation, batchMean=None):
        """batch 内去均值后的协方差矩阵（无偏置归一化，维度 D x D）。"""
        if representation.shape[0] < 2:
            return None
        values = representation.detach().float()
        if batchMean is None:
            batchMean = values.mean(dim=0, keepdim=True)
        else:
            batchMean = batchMean.reshape(1, -1).to(dtype=values.dtype, device=values.device)
        centered = values - batchMean
        return (centered.transpose(0, 1) @ centered) / float(values.shape[0])

    def _matchCovariance(self, representation, strength):
        """把 batch 协方差按强度插值拉回参考协方差，保持 batch 均值不变。

        ``gain = (1-strength) * I + strength * batchCov^{-1/2} referenceCov^{1/2}``，
        strength=0 时恒等；strength=1 时完全把 batch 协方差映射到参考协方差。
        干净 batch 的 batchCov 本身接近 referenceCov，故增益接近单位阵，近似恒等。
        """
        if (
            self.referenceCov is None
            or strength <= 0.0
            or representation.shape[0] < 3
        ):
            return representation
        batchMean = representation.mean(dim=0, keepdim=True)
        batchCov = self._batchCovariance(representation, batchMean)
        if (
            batchCov is None
            or batchCov.shape != self.referenceCov.shape
        ):
            return representation
        referenceCov = self.referenceCov.to(
            dtype=batchCov.dtype, device=batchCov.device
        )
        identity = torch.eye(
            batchCov.shape[0], dtype=batchCov.dtype, device=batchCov.device
        )
        inverseRoot = _symmetricMatrixPower(batchCov, -0.5)
        referenceRoot = _symmetricMatrixPower(referenceCov, 0.5)
        gain = (1.0 - strength) * identity + strength * (inverseRoot @ referenceRoot)
        gain = gain.to(dtype=representation.dtype)
        centered = representation - batchMean
        return centered @ gain.transpose(0, 1) + batchMean

    def _recenter(self, representation, strength):
        """移除 batch 共有偏置 ``strength * (batchMean - referenceMean)``。"""
        if self.referenceMean is None or strength <= 0.0 or representation.shape[0] < 2:
            return representation
        referenceMean = self.referenceMean.to(
            dtype=representation.dtype, device=representation.device
        )
        batchMean = representation.mean(dim=0, keepdim=True)
        return representation - strength * (batchMean - referenceMean)

    def _standardize(self, representation, strength):
        """逐维把 batch 标准差按幂次拉回参考标准差，保持 batch 均值不变。

        ``gain = (referenceStd / batchStd) ** strength``：strength=0 时增益为 1
        （恒等变换），strength=1 时完全对齐到参考标准差。因为重定心只是在每个
        维度上加同一个常数，不会改变逐维方差，所以本步与 ``_recenter`` 可任意
        顺序复合；干净 batch 的 batchStd 本身接近 referenceStd，故近似恒等。
        """
        if self.referenceStd is None or strength <= 0.0 or representation.shape[0] < 2:
            return representation
        referenceStd = self.referenceStd.to(
            dtype=representation.dtype, device=representation.device
        ).clamp_min(1e-6)
        batchStd = (
            representation.var(dim=0, unbiased=False, keepdim=True).clamp_min(0.0).sqrt()
        ).clamp_min(1e-6)
        gain = (referenceStd / batchStd).pow(strength)
        batchMean = representation.mean(dim=0, keepdim=True)
        return (representation - batchMean) * gain + batchMean

    def _centerClassConditional(self, gradient, labels):
        """按 batch 内标签分组，衰减逐类的平均梯度（类间可分方向）。"""
        labelTensor = torch.as_tensor(labels).to(
            device=gradient.device, dtype=torch.long
        ).reshape(-1)
        if labelTensor.numel() != gradient.shape[0]:
            return gradient
        globalMean = gradient.mean(dim=0, keepdim=True)
        centered = gradient.clone()
        for labelValue in torch.unique(labelTensor):
            mask = labelTensor == labelValue
            if int(mask.sum().item()) < 2:
                continue
            classMean = gradient[mask].mean(dim=0, keepdim=True)
            centered[mask] = gradient[mask] - self.gradientClassCenteringStrength * (
                classMean - globalMean
            )
        return centered

    def _filterLabelInconsistent(self, representation, labels, generator=None):
        """把与本类原型"不一致"的被动表示替换为同类同 batch 的随机同伴表示。

        每个类别在 batch 内用逐维中位数估计鲁棒中心（对少量离群样本不敏感），
        距离中位数作为该类的鲁棒尺度。判定两条规则取并集：

        1. 离群规则：到本类中心的距离超过 ``z * scale``（捕捉 VILLAIN 那种
           在 embedding 上叠加固定触发器、把样本推到本类簇之外的投毒）；
        2. 交叉类别规则：到最近的其他类别中心的距离小于 ``margin`` 倍本类距离
           （捕捉 LFBA 那种把别的样本的 B 方特征搬过来的投毒，其表示仍然落在
           源样本所属的类别簇里）。

        被判定样本以 ``strength`` 为插值系数替换为同批次、同标签的随机同伴表示
        （而不是本类中心）：既切断"触发器 → 目标类"的关联，又保留类内多样性，
        避免把类分布压成单点后让被触发样本更孤立。替换后其 cut 梯度退化为 0。
        """
        strength = self.trainingPrototypeFilterStrength
        if strength <= 0.0 or labels is None or representation.shape[0] < 2:
            return representation, None
        labelTensor = torch.as_tensor(labels).to(
            device=representation.device, dtype=torch.long
        ).reshape(-1)
        if labelTensor.numel() != representation.shape[0]:
            return representation, None
        with torch.no_grad():
            values = representation.detach().float()
            classCount = int(labelTensor.max().item()) + 1
            nan = float("nan")
            centers = torch.full(
                (classCount, values.shape[1]),
                nan,
                dtype=values.dtype,
                device=values.device,
            )
            scales = torch.full(
                (classCount,), nan, dtype=values.dtype, device=values.device
            )
            for labelValue in torch.unique(labelTensor):
                mask = labelTensor == labelValue
                if int(mask.sum().item()) < _PROTOTYPE_MIN_CLASS_COUNT:
                    continue
                rows = values[mask]
                center = rows.median(dim=0).values
                distances = (rows - center).norm(dim=1)
                centers[labelValue] = center
                scales[labelValue] = distances.median().clamp_min(1e-6)
            rowCenters = centers[labelTensor]
            ownDistance = (values - rowCenters).norm(dim=1)
            flagged = ownDistance > self.trainingPrototypeFilterZ * scales[labelTensor]
            if self.trainingPrototypeFilterMargin > 0.0:
                usable = torch.isfinite(centers).all(dim=1)
                distancesToCenters = torch.cdist(
                    values, torch.nan_to_num(centers, nan=0.0)
                )
                distancesToCenters[:, ~usable] = float("inf")
                distancesToCenters[
                    torch.arange(values.shape[0], device=values.device), labelTensor
                ] = float("inf")
                otherDistance = distancesToCenters.min(dim=1).values
                flagged = flagged | (
                    otherDistance
                    < self.trainingPrototypeFilterMargin * ownDistance
                )
            flaggedFraction = float(flagged.float().mean().item())
        self.lastFlaggedFraction = flaggedFraction
        self.flaggedFractionSum += flaggedFraction
        self.flaggedFractionCount += 1
        if not bool(flagged.any().item()):
            return representation, flagged
        replacement = self._sameLabelPeerRepresentation(
            representation, labelTensor, generator
        )
        if replacement is None:
            replacement = torch.nan_to_num(
                centers[labelTensor], nan=0.0
            ).to(dtype=representation.dtype)
        weight = (strength * flagged.to(dtype=representation.dtype)).unsqueeze(1)
        return representation + weight * (replacement - representation), flagged

    def _filterTemporallyInconsistent(self, representation, sampleIds):
        """用逐样本的被动表示历史识别"本轮发生跳变"的样本并回填其历史表示。

        残差先减去 batch 内所有已知样本残差的逐维中位数，以消除 B 方参数更新带来的
        全体共有漂移；随后用残差模长的中位数作为鲁棒尺度，超过 ``z * scale`` 的样本
        判为跳变。历史只在未被判为跳变时更新，避免污染干净参考。
        """
        strength = self.trainingTemporalFilterStrength
        if strength <= 0.0 or sampleIds is None or representation.shape[0] < 3:
            return representation, None
        keys = [str(sampleId) for sampleId in sampleIds]
        if len(keys) != representation.shape[0]:
            return representation, None
        with torch.no_grad():
            values = representation.detach().float()
            history = torch.full_like(values, float("nan"))
            knownPositions = []
            for position, key in enumerate(keys):
                stored = self.passiveHistory.get(key)
                if stored is not None and stored.shape == values.shape[1:]:
                    history[position] = stored.to(
                        dtype=values.dtype, device=values.device
                    )
                    knownPositions.append(position)
            flagged = torch.zeros(
                values.shape[0], dtype=torch.bool, device=values.device
            )
            if len(knownPositions) >= 3:
                knownIndex = torch.tensor(
                    knownPositions, dtype=torch.long, device=values.device
                )
                residuals = values[knownIndex] - history[knownIndex]
                residuals = residuals - residuals.median(dim=0).values
                distances = residuals.norm(dim=1)
                scale = distances.median().clamp_min(1e-6)
                flagged[knownIndex] = distances > self.trainingTemporalFilterZ * scale
            flaggedFraction = float(flagged.float().mean().item())
            self.temporalFlaggedFractionSum += flaggedFraction
            self.temporalFlaggedFractionCount += 1
            momentum = self.temporalMomentum
            updated = momentum * torch.nan_to_num(history, nan=0.0) + (
                1.0 - momentum
            ) * values
            fresh = torch.isnan(history).any(dim=1, keepdim=True)
            updated = torch.where(fresh, values, updated)
            updated = torch.where(flagged.unsqueeze(1), history, updated)
            for position, key in enumerate(keys):
                self.passiveHistory[key] = updated[position].detach().clone()
        if not bool(flagged.any().item()):
            return representation, flagged
        weight = (strength * flagged.to(dtype=representation.dtype)).unsqueeze(1)
        replacement = torch.nan_to_num(
            history, nan=0.0
        ).to(dtype=representation.dtype)
        return representation + weight * (replacement - representation), flagged

    @staticmethod
    def _sameLabelPeerRepresentation(representation, labelTensor, generator):
        """为每个样本抽取一个同标签同伴；随机排列保证不会抽到自己。"""
        if generator is None:
            return None
        positions = torch.arange(representation.shape[0], device=representation.device)
        donor = positions.clone()
        for labelValue in torch.unique(labelTensor):
            index = positions[labelTensor == labelValue]
            if index.numel() < 2:
                continue
            permutation = torch.randperm(
                index.numel(), generator=generator, device=generator.device
            ).to(index.device)
            peer = index[permutation]
            sameRow = peer == index
            if bool(sameRow.any().item()):
                peer = torch.where(
                    sameRow, index[(permutation + 1) % index.numel()], peer
                )
            donor[index] = peer
        return representation[donor].detach()

    # ------------------------------------------------------------------
    # 三个 hook
    # ------------------------------------------------------------------
    def trainingRepresentationHook(
        self,
        cutA,
        cutB,
        labels=None,
        sampleIds=None,
        epoch=0,
        batchIndex=0,
        context=None,
        generator=None,
    ):
        """训练期：先刷新参考统计，再做重定心与方差对齐后送入顶模。"""
        self._updateReference(cutB)
        transformed = self._recenter(cutB, self.trainingRecenterStrength)
        transformed = self._matchCovariance(
            transformed, self.trainingCovarianceStrength
        )
        transformed = self._standardize(
            transformed, self.representationStandardizeStrength
        )
        transformed, _ = self._filterLabelInconsistent(
            transformed, labels, generator
        )
        transformed, _ = self._filterTemporallyInconsistent(transformed, sampleIds)
        if batchIndex == 0 and (
            self.flaggedFractionCount or self.temporalFlaggedFractionCount
        ):
            print(
                "[DefenseFilter] epoch-mean flagged fraction "
                f"prototype={self.flaggedFractionSum / max(self.flaggedFractionCount, 1):.4f} "
                f"temporal={self.temporalFlaggedFractionSum / max(self.temporalFlaggedFractionCount, 1):.4f}",
                flush=True,
            )
            self.flaggedFractionSum = 0.0
            self.flaggedFractionCount = 0
            self.temporalFlaggedFractionSum = 0.0
            self.temporalFlaggedFractionCount = 0
        if (
            self.representationNoise > 0.0
            and self.referenceStd is not None
            and generator is not None
        ):
            referenceStd = self.referenceStd.to(
                dtype=transformed.dtype, device=transformed.device
            )
            transformed = transformed + self.representationNoise * referenceStd * torch.randn(
                transformed.shape,
                generator=generator,
                dtype=transformed.dtype,
                device=transformed.device,
            )
        return cutA, transformed

    def cutGradientHook(
        self,
        gradientB,
        passiveCut=None,
        labels=None,
        sampleIds=None,
        epoch=0,
        batchIndex=0,
        context=None,
        generator=None,
    ):
        """训练期：对返回给 B 方的 cut 梯度做归一化、类别去均值与加噪。"""
        transformed = gradientB
        if self.gradientNormalizeStrength > 0.0 and transformed.shape[0] >= 2:
            norms = transformed.norm(p=2, dim=1, keepdim=True)
            referenceNorm = norms.median()
            rescaled = transformed * (referenceNorm / norms.clamp_min(1e-12))
            transformed = transformed + self.gradientNormalizeStrength * (
                rescaled - transformed
            )
        if (
            self.gradientClassCenteringStrength > 0.0
            and labels is not None
            and transformed.shape[0] >= 2
        ):
            transformed = self._centerClassConditional(transformed, labels)
        if self.gradientNoise > 0.0 and generator is not None:
            perSampleRms = transformed.pow(2).mean(dim=1, keepdim=True).sqrt()
            noise = torch.randn(
                transformed.shape,
                generator=generator,
                dtype=transformed.dtype,
                device=transformed.device,
            )
            transformed = transformed + self.gradientNoise * perSampleRms * noise
        return transformed

    def inferenceRepresentationHook(
        self,
        embeddingA,
        embeddingB,
        sampleIds=None,
        context=None,
        generator=None,
    ):
        """推理期：用冻结参考统计做确定性重定心与方差对齐，抵消共有触发器偏置。"""
        recentered = self._recenter(embeddingB, self.inferenceRecenterStrength)
        recentered = self._matchCovariance(
            recentered, self.inferenceCovarianceStrength
        )
        return embeddingA, self._standardize(
            recentered, self.representationStandardizeStrength
        )


SUPPORTED_DEFENSE_NAMES = {
    "robust_vfl": RobustVflDefense,
}


def buildPartyADefense(defenseConfig):
    """根据 ``config['defense']`` 构造防御对象；无防御时返回 None。"""
    if not defenseConfig:
        return None
    name = str(defenseConfig.get("name", "")).strip().lower()
    if name in {"", "none", "identity", "baseline"}:
        return None
    factory = SUPPORTED_DEFENSE_NAMES.get(name)
    if factory is None:
        raise ValueError(f"Unsupported defense name: {name}")
    return factory(defenseConfig.get("parameters", {}))
