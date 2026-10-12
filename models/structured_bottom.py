from torch import nn


class StructuredBottomModel(nn.Module):
    """使用三个可配置隐藏层和固定 embedding 输出层的四层 MLP。"""

    def __init__(self, inputDim, hiddenDims=None, embeddingDim=64):
        super().__init__()
        resolvedHiddenDims = [128, 64, 32] if hiddenDims is None else list(hiddenDims)
        if len(resolvedHiddenDims) != 3:
            raise ValueError("Structured bottom model requires exactly three hidden dimensions")
        if any(int(dimension) <= 0 for dimension in resolvedHiddenDims):
            raise ValueError("Structured bottom hidden dimensions must be positive")

        layerDimensions = [
            int(inputDim),
            *(int(dimension) for dimension in resolvedHiddenDims),
            int(embeddingDim),
        ]
        layers = []
        for layerIndex in range(len(layerDimensions) - 1):
            layers.append(
                nn.Linear(layerDimensions[layerIndex], layerDimensions[layerIndex + 1])
            )
            if layerIndex < len(layerDimensions) - 2:
                layers.append(nn.ReLU())
        self.hiddenDims = resolvedHiddenDims
        self.embeddingDim = int(embeddingDim)
        self.network = nn.Sequential(*layers)

    def forward(self, features):
        return self.network(features)
