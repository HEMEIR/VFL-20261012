from torch import nn
from torchvision.models import resnet18


class ImageBottomModel(nn.Module):
    """两方各自持有一半图像，并使用独立参数的 ResNet18。"""

    def __init__(self, embeddingDim=64):
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embeddingDim)

    def forward(self, features):
        return self.backbone(features)
