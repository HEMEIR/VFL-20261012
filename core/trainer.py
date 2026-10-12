import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from .checkpoint import captureModelState, restoreModelState
from .evaluator import evaluateModels, predictModels
from .interfaces import TrainingBatchResult, TrainingHistory
from .progress import createProgressBar, writeProgress
from .randomness import createDataLoaderGenerator, seedDataLoaderWorker


class VflTrainer:
    """无防御两方 VFL 训练器，显式模拟 cut-layer 梯度返回。"""

    def __init__(self, models, trainingConfig, device):
        self.models = models
        self.config = trainingConfig
        self.device = device
        self.lossFunction = nn.CrossEntropyLoss()
        # Optional label transform applied ONLY to the objective the trainer minimises.
        # Defaults to identity, so every existing configuration is bit-identical.
        # This exists because the defence hooks cannot reach the loss: they receive a
        # detached copy of the labels and may only return cut tensors (see
        # applyPartyATrainingRepresentationHook), so a class-relabelling defence needs this
        # single, explicit seam.  Set by the party-A defence when it wants the model to
        # optimise ALIAS labels.
        self.labelTransform = None
        # Optional inverse transform applied to predicted class indices.  Symmetric to
        # labelTransform: if the objective was trained on ALIAS labels, the model emits
        # aliases and the caller must map them back before comparing with true labels.
        self.predictionTransform = None
        learningRate = float(trainingConfig["learningRate"])
        weightDecay = float(trainingConfig.get("weightDecay", 0.0))
        # 三部分模型统一使用 Adam，不提供 SGD 选项。
        self.optimizerA = torch.optim.Adam(models.bottomA.parameters(), lr=learningRate, weight_decay=weightDecay)
        self.optimizerB = torch.optim.Adam(models.bottomB.parameters(), lr=learningRate, weight_decay=weightDecay)
        self.optimizerTop = torch.optim.Adam(models.topModel.parameters(), lr=learningRate, weight_decay=weightDecay)
        # 正式配置显式提供 seed；最小化测试或外部直接调用时仍采用确定性的 0。
        seed = int(trainingConfig.get("seed", 100))
        self.dataLoaderGenerator = createDataLoaderGenerator(seed)
        # 防御随机性与 DataLoader、攻击及全局 torch RNG 分离；训练和推理也互不推进。
        # CUDA 场景使用同设备 generator，hook 可直接按通信张量的 device 采样。
        defenseGeneratorDevice = self.device if self.device.type == "cuda" else "cpu"
        self.partyADefenseTrainingGenerator = torch.Generator(
            device=defenseGeneratorDevice
        ).manual_seed(
            seed + 4000007
        )
        self.partyADefenseInferenceGenerator = torch.Generator(
            device=defenseGeneratorDevice
        ).manual_seed(
            seed + 5000011
        )
        self.amcGradientScaler = None
        # 每轮 batch 数，供防御 hook 的进度调度使用；由 trainEpoch 填写。
        self.batchesPerEpoch = 0
        # 基座默认不启用任何防御。三个 hook 只提供统一的 A 方扩展入口，
        # 确保 clean、LFBA、VILLAIN 及复合场景不会因独立攻击循环而绕过防御。
        self.partyATrainingRepresentationHook = None
        self.partyACutGradientHook = None
        self.partyAInferenceRepresentationHook = None
        self.scenarioContext = {"backdoor": "none", "labelInference": "none"}

    def configureScenarioContext(self, backdoor="none", labelInference="none"):
        """记录所有统一训练/推理入口均可见的公开场景信息。"""
        self.scenarioContext = {
            "backdoor": str(backdoor).lower(),
            "labelInference": str(labelInference).lower(),
        }

    def configureLabelTransform(self, transform=None):
        """Install a label transform for the training objective, or None for identity.

        Why this seam exists: the party-A hooks receive a DETACHED COPY of the labels and may
        only return cut tensors (see applyPartyATrainingRepresentationHook), so a defence that
        needs the model to optimise a RELABELLED objective cannot reach the loss from inside a
        hook.  Measured consequence: aliasing the labels inside the representation hook changed
        nothing at all - four settings produced bit-identical results - because
        ``lossFunction(logits, batch["label"])`` reads the original labels.

        Strictly optional and identity by default, so a configuration that never calls this
        behaves exactly as before.
        """
        if transform is not None and not callable(transform):
            raise TypeError("labelTransform must be callable or None")
        self.labelTransform = transform

    def configurePredictionTransform(self, transform=None):
        """Install an inverse transform for predicted indices, or None.

        Needed whenever a label transform was installed: the model then emits ALIAS indices,
        and every downstream consumer (accuracy, the attack metrics, VILLAIN's minimum-accuracy
        checkpoint check) expects TRUE labels.  Measured: without this, dataset_03/villain
        aborted with "VILLAIN did not produce a checkpoint meeting minimumMainTaskAccuracy"
        because the aliased accuracy never reaches the un-aliased threshold, and the per-scenario
        accuracies read as 0.13 / 0.06 instead of 0.87 / 0.94.
        """
        if transform is not None and not callable(transform):
            raise TypeError("predictionTransform must be callable or None")
        self.predictionTransform = transform

    def _applyPredictionTransform(self, result):
        """Map predicted indices back through the inverse alias table, in place."""
        if self.predictionTransform is None or not result:
            return result
        predictions = result.get("predictions")
        if not predictions:
            return result
        ids = result.get("ids") or []
        result["predictions"] = list(self.predictionTransform(list(predictions), list(ids)))
        return result

    def configurePartyADefenseHooks(
        self,
        trainingRepresentationHook=None,
        cutGradientHook=None,
        inferenceRepresentationHook=None,
    ):
        """注册 A 方合法可见通信上的防御 hook；传入 None 即保持无防御基线。"""
        for name, hook in (
            ("trainingRepresentationHook", trainingRepresentationHook),
            ("cutGradientHook", cutGradientHook),
            ("inferenceRepresentationHook", inferenceRepresentationHook),
        ):
            if hook is not None and not callable(hook):
                raise TypeError(f"{name} must be callable or None")
        self.partyATrainingRepresentationHook = trainingRepresentationHook
        self.partyACutGradientHook = cutGradientHook
        self.partyAInferenceRepresentationHook = inferenceRepresentationHook

    @staticmethod
    def validateHookTensor(name, reference, transformed, checkFinite=False):
        if not torch.is_tensor(transformed):
            raise TypeError(f"{name} must return a torch.Tensor")
        if transformed.shape != reference.shape:
            raise ValueError(
                f"{name} changed tensor shape from {tuple(reference.shape)} "
                f"to {tuple(transformed.shape)}"
            )
        if transformed.dtype != reference.dtype:
            raise ValueError(
                f"{name} changed tensor dtype from {reference.dtype} "
                f"to {transformed.dtype}"
            )
        if transformed.device != reference.device:
            raise ValueError(
                f"{name} changed tensor device from {reference.device} "
                f"to {transformed.device}"
            )
        if checkFinite and not bool(torch.isfinite(transformed).all().item()):
            raise FloatingPointError(f"{name} produced non-finite values")
        return transformed

    def applyPartyATrainingRepresentationHook(
        self,
        cutA,
        cutB,
        labels,
        sampleIds,
        epoch,
        batchIndex,
        trainingContext,
    ):
        """在攻击完成 B 方通信后、顶模前向前执行 A 方训练期防御。"""
        hook = self.partyATrainingRepresentationHook
        if hook is None:
            return cutA, cutB
        transformed = hook(
            cutA,
            cutB,
            labels=labels.detach().clone(),
            sampleIds=sampleIds,
            epoch=int(epoch),
            batchIndex=int(batchIndex),
            context=dict(trainingContext),
            generator=self.partyADefenseTrainingGenerator,
        )
        if not isinstance(transformed, tuple) or len(transformed) != 2:
            raise TypeError(
                "trainingRepresentationHook must return (partyARepresentation, partyBRepresentation)"
            )
        transformedA = self.validateHookTensor(
            "trainingRepresentationHook party A output",
            cutA,
            transformed[0],
            checkFinite=True,
        )
        transformedB = self.validateHookTensor(
            "trainingRepresentationHook party B output",
            cutB,
            transformed[1],
            checkFinite=True,
        )
        return transformedA, transformedB

    def applyPartyACutGradientHook(
        self,
        gradientB,
        cutB,
        labels,
        sampleIds,
        epoch,
        batchIndex,
        trainingContext,
    ):
        """处理 A 方准备返回给 B 方的 cut-layer 梯度。"""
        hook = self.partyACutGradientHook
        if hook is None:
            return gradientB
        transformed = hook(
            gradientB,
            passiveCut=cutB.detach().clone(),
            labels=labels.detach().clone(),
            sampleIds=sampleIds,
            epoch=int(epoch),
            batchIndex=int(batchIndex),
            context=dict(trainingContext),
            generator=self.partyADefenseTrainingGenerator,
        )
        return self.validateHookTensor(
            "cutGradientHook output",
            gradientB,
            transformed,
            checkFinite=True,
        )

    def applyPartyAInferenceRepresentationHook(
        self,
        embeddingA,
        embeddingB,
        sampleIds,
        inferenceContext,
    ):
        """在攻击完成 B 方推理通信后、顶模前向前执行 A 方推理期防御。"""
        hook = self.partyAInferenceRepresentationHook
        if hook is None:
            return embeddingA, embeddingB
        transformed = hook(
            embeddingA,
            embeddingB,
            sampleIds=sampleIds,
            context=dict(inferenceContext),
            generator=self.partyADefenseInferenceGenerator,
        )
        if not isinstance(transformed, tuple) or len(transformed) != 2:
            raise TypeError(
                "inferenceRepresentationHook must return (partyARepresentation, partyBRepresentation)"
            )
        transformedA = self.validateHookTensor(
            "inferenceRepresentationHook party A output",
            embeddingA,
            transformed[0],
            checkFinite=True,
        )
        transformedB = self.validateHookTensor(
            "inferenceRepresentationHook party B output",
            embeddingB,
            transformed[1],
            checkFinite=True,
        )
        return transformedA, transformedB

    def enableAmcGradientScaling(self, gradientScaler):
        """只为 AMC 打开 B 方底模的恶意梯度变换。"""
        self.amcGradientScaler = gradientScaler

    def applyAmcGradientScaling(self, epoch=0):
        if (
            self.amcGradientScaler is not None
            and self.amcGradientScaler.isActiveAtEpoch(epoch)
        ):
            return self.amcGradientScaler.scaleGradients()
        return None

    def moveModelsToDevice(self):
        self.models.bottomA.to(self.device)
        self.models.bottomB.to(self.device)
        self.models.topModel.to(self.device)

    def buildTrainingLoader(self, dataset):
        """所有训练路径共用同一个独立生成器，保证 batch 洗牌顺序可复现。"""
        return DataLoader(
            dataset,
            batch_size=int(self.config["batchSize"]),
            shuffle=True,
            generator=self.dataLoaderGenerator,
            worker_init_fn=seedDataLoaderWorker,
        )

    def predict(self, dataset, partyBEmbeddingTransform=None, inferenceContext=None):
        context = {
            **self.scenarioContext,
            "phase": "inference",
            "triggered": False,
        }
        if inferenceContext is not None:
            context.update(dict(inferenceContext))
        result = predictModels(
            self.models,
            dataset,
            int(self.config["batchSize"]),
            self.device,
            partyBEmbeddingTransform=partyBEmbeddingTransform,
            partyARepresentationHook=self.applyPartyAInferenceRepresentationHook,
            inferenceContext=context,
        )
        return self._applyPredictionTransform(result)

    def evaluate(self, dataset, partyBEmbeddingTransform=None, inferenceContext=None):
        context = {
            **self.scenarioContext,
            "phase": "inference",
            "triggered": False,
        }
        if inferenceContext is not None:
            context.update(dict(inferenceContext))
        return evaluateModels(
            self.models,
            dataset,
            int(self.config["batchSize"]),
            self.device,
            partyBEmbeddingTransform=partyBEmbeddingTransform,
            partyARepresentationHook=self.applyPartyAInferenceRepresentationHook,
            inferenceContext=context,
            # Passed INTO the evaluator so the alias inverse is applied before the metrics are
            # scored; applying it on the returned dict was measured to be too late.
            predictionTransform=self.predictionTransform,
        )

    def train(self, trainDataset, validationDataset):
        self.moveModelsToDevice()
        history = TrainingHistory()
        bestState = captureModelState(self.models)
        patience = int(self.config.get("earlyStoppingPatience", 0))
        epochsWithoutImprovement = 0
        totalEpochs = int(self.config["epochs"])
        writeProgress(
            self.config,
            (
                f"[Train] started | epochs={totalEpochs} | "
                f"batch_size={int(self.config['batchSize'])} | device={self.device}"
            ),
        )

        for epoch in range(totalEpochs):
            epochStartTime = time.perf_counter()
            trainingLoss = self.trainEpoch(trainDataset, epoch)
            validation = self.evaluate(validationDataset)
            validationAccuracy = float(validation["metrics"]["ba"])
            validationAuc = validation["metrics"]["auc"]
            history.epochs.append(
                {
                    "epoch": epoch,
                    "trainingLoss": trainingLoss,
                    "validationBa": validationAccuracy,
                    "validationAuc": validationAuc,
                }
            )
            # 正式主任务指标固定为 Accuracy，最佳模型与早停也按同一口径选择。
            if validationAccuracy > history.bestValidationAccuracy:
                history.bestValidationAccuracy = validationAccuracy
                history.bestValidationAuc = (
                    validationAuc
                    if validationAuc is not None
                    else float("-inf")
                )
                history.bestSelectionScore = validationAccuracy
                history.bestEpoch = epoch
                bestState = captureModelState(self.models)
                epochsWithoutImprovement = 0
            else:
                epochsWithoutImprovement += 1
            elapsedSeconds = time.perf_counter() - epochStartTime
            patienceText = (
                f"{epochsWithoutImprovement}/{patience}"
                if patience > 0
                else "off"
            )
            writeProgress(
                self.config,
                (
                    f"Epoch {epoch + 1}/{totalEpochs} | loss={trainingLoss:.4f} | "
                    f"val_acc={validationAccuracy:.4f} | "
                    f"best_val_acc={history.bestValidationAccuracy:.4f} | "
                    f"patience={patienceText} | time={elapsedSeconds:.1f}s"
                ),
            )
            if patience > 0 and epochsWithoutImprovement >= patience:
                writeProgress(
                    self.config,
                    (
                        f"[EarlyStop] stopped at epoch {epoch + 1}; "
                        f"best_epoch={history.bestEpoch + 1} | "
                        f"best_val_acc={history.bestValidationAccuracy:.4f}"
                    ),
                )
                break

        restoreModelState(self.models, bestState)
        writeProgress(
            self.config,
            (
                f"[Train] completed | best_epoch={history.bestEpoch + 1} | "
                f"best_val_acc={history.bestValidationAccuracy:.4f}"
            ),
        )
        return history

    def trainEpoch(self, trainDataset, epoch=0):
        self.models.bottomA.train()
        self.models.bottomB.train()
        self.models.topModel.train()
        loader = self.buildTrainingLoader(trainDataset)
        totalLoss = 0.0
        totalSamples = 0
        totalEpochs = int(self.config["epochs"])
        self.batchesPerEpoch = max(
            1, -(-len(trainDataset) // int(self.config["batchSize"]))
        )
        progressBar = createProgressBar(
            loader,
            self.config,
            f"Epoch {epoch + 1}/{totalEpochs}",
            "batch",
        )
        for batchIndex, originalBatch in enumerate(progressBar):
            batch = self.moveBatchToDevice(originalBatch)
            result = self.trainBatch(
                batch,
                epoch,
                batchIndex,
                {
                    # Must not hardcode "none": trainBatch merges this over
                    # scenarioContext, so a literal here would erase the public
                    # scenario label that defenses use to tell a backdoor scenario
                    # from a clean one during training.
                    "backdoor": self.scenarioContext.get("backdoor", "none"),
                    "poisoningActive": False,
                },
            )
            batchSize = int(batch["label"].shape[0])
            totalLoss += float(result) * batchSize
            totalSamples += batchSize
            progressBar.set_postfix(
                loss=f"{totalLoss / max(totalSamples, 1):.4f}",
                refresh=False,
            )
        return totalLoss / max(totalSamples, 1)

    def trainBatch(
        self,
        batch,
        epoch=0,
        batchIndex=0,
        trainingContext=None,
        partyBEmbeddingTransform=None,
        capturePassiveCutGradient=False,
    ):
        """执行唯一的两方训练 batch，攻击只能在 B 方 embedding 边界注入行为。"""
        context = {
            **self.scenarioContext,
            "phase": "train",
            "poisoningActive": False,
        }
        if trainingContext is not None:
            context.update(dict(trainingContext))
        # 防御 hook 的进度调度需要知道总轮数与每轮 batch 数；两者都由配置和数据集
        # 直接决定，因此在这里以纯元数据形式公开，不泄漏任何 B 方私有信息。
        context["totalEpochs"] = int(self.config["epochs"])
        context["batchesPerEpoch"] = getattr(self, "batchesPerEpoch", 0)
        sampleIds = tuple(str(sampleId) for sampleId in batch.get("id", ()))

        self.optimizerA.zero_grad()
        self.optimizerB.zero_grad()
        self.optimizerTop.zero_grad()

        embeddingA = self.models.bottomA(batch["partyA"])
        embeddingB = self.models.bottomB(batch["partyB"])
        metadata = {}
        if partyBEmbeddingTransform is not None:
            transformed = partyBEmbeddingTransform(embeddingB)
            if not isinstance(transformed, tuple) or len(transformed) != 2:
                raise TypeError(
                    "partyBEmbeddingTransform must return (embeddingB, metadata)"
                )
            embeddingB = self.validateHookTensor(
                "partyBEmbeddingTransform output",
                embeddingB,
                transformed[0],
                checkFinite=True,
            )
            metadata = dict(transformed[1] or {})

        # 顶模只接收脱离底模计算图的 cut-layer 输出。
        cutA = embeddingA.detach().requires_grad_(True)
        cutB = embeddingB.detach().requires_grad_(True)
        topInputA, topInputB = self.applyPartyATrainingRepresentationHook(
            cutA,
            cutB,
            batch["label"],
            sampleIds,
            epoch,
            batchIndex,
            context,
        )
        logits = self.models.topModel(topInputA, topInputB)
        objectiveLabels = batch["label"]
        if self.labelTransform is not None:
            objectiveLabels = self.labelTransform(objectiveLabels)
        # Soft alias encoding returns a 2-D target (a distribution over the alias block), for
        # which CrossEntropyLoss is inapplicable.  Use the cross-entropy against the soft
        # target directly, equivalent to KL(log_softmax(logits) || target).
        if objectiveLabels is not None and objectiveLabels.dim() == 2:
            logProb = torch.log_softmax(logits, dim=1)
            loss = -(objectiveLabels * logProb).sum(dim=1).mean()
        else:
            loss = self.lossFunction(logits, objectiveLabels)
        loss.backward()
        if cutA.grad is None or cutB.grad is None:
            raise RuntimeError(
                "A party training representation hook detached a cut-layer gradient path"
            )
        gradientA = cutA.grad.detach().clone()
        gradientB = cutB.grad.detach().clone()

        gradientB = self.applyPartyACutGradientHook(
            gradientB,
            cutB,
            batch["label"],
            sampleIds,
            epoch,
            batchIndex,
            context,
        )

        # 主动方将两段 cut-layer 梯度分别返回，各方只更新自己的底模。
        embeddingA.backward(gradientA)
        embeddingB.backward(gradientB)
        self.applyAmcGradientScaling(epoch)
        # 所有 hook、张量校验和底模反向均成功后才提交三部分参数更新，
        # 避免防御异常留下“顶模已更新、底模未更新”的半个 batch。
        self.optimizerTop.step()
        self.optimizerA.step()
        self.optimizerB.step()
        lossValue = float(loss.detach().item())
        passiveCutGradient = (
            gradientB.detach().cpu().clone() if capturePassiveCutGradient else None
        )
        return TrainingBatchResult(
            loss=lossValue,
            passiveCutGradient=passiveCutGradient,
            metadata=metadata,
        )

    def moveBatchToDevice(self, batch):
        moved = {}
        for key, value in batch.items():
            moved[key] = value.to(self.device) if torch.is_tensor(value) else value
        return moved
