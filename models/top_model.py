import torch
from torch import nn


class TopModel(nn.Module):
    """将 A、B 的 embedding 按固定顺序拼接后执行三层分类。"""

    def __init__(self, embeddingDim=64, classCount=2, hiddenDims=None):
        super().__init__()
        resolvedHiddenDims = [128, 64] if hiddenDims is None else list(hiddenDims)
        if len(resolvedHiddenDims) != 2:
            raise ValueError("Top model requires exactly two hidden dimensions")
        if any(int(dimension) <= 0 for dimension in resolvedHiddenDims):
            raise ValueError("Top model hidden dimensions must be positive")
        firstHiddenDim, secondHiddenDim = (
            int(dimension) for dimension in resolvedHiddenDims
        )
        self.hiddenDims = resolvedHiddenDims
        self.network = nn.Sequential(
            nn.Linear(embeddingDim * 2, firstHiddenDim),
            nn.ReLU(),
            nn.Linear(firstHiddenDim, secondHiddenDim),
            nn.ReLU(),
            nn.Linear(secondHiddenDim, classCount),
        )

    def forward(self, embeddingA, embeddingB):
        return self.network(torch.cat([embeddingA, embeddingB], dim=1))
