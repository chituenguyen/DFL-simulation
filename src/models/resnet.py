"""
ResNet-18 Model for CIFAR-10
Adapted for Federated Learning
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class ResNet18(nn.Module):
    """
    ResNet-18 model optimized for CIFAR-10 (32x32 images)
    """

    def __init__(self, num_classes: int = 10, pretrained: bool = False):
        super(ResNet18, self).__init__()

        # Load base ResNet-18
        self.model = models.resnet18(pretrained=pretrained)

        # Adapt for CIFAR-10 (smaller input size)
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.model.maxpool = nn.Identity()  # Remove maxpool for 32x32 images

        # Replace final FC layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_parameters(self) -> dict:
        """Get model parameters as dict"""
        return {name: param.data.clone() for name, param in self.named_parameters()}

    def set_parameters(self, params: dict) -> None:
        """Set model parameters from dict"""
        model_dict = self.state_dict()
        model_dict.update(params)
        self.load_state_dict(model_dict)


def create_model(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """Factory function to create ResNet-18 model"""
    return ResNet18(num_classes=num_classes, pretrained=pretrained)