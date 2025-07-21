import torch
import torch.nn as nn
from typing import List, Callable
import torchvision.models as models

from .base_model import BaseModel


class ResNetFer(BaseModel):
    """
    A ResNet-18 model fine-tuned for FER.
    This class has a clean __init__ signature and a correct forward pass.
    """

    # The __init__ method now ONLY asks for what it truly needs.
    def __init__(self, num_classes: int, dropout: float):
        super().__init__()

        # Load the pre-trained model into a temporary local variable
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # 1. Explicitly define the feature extractor
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        # 2. Explicitly define the new classifier head
        num_ftrs = resnet.fc.in_features
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        """Defines the correct forward pass for the refactored model."""
        # 1. Pass the input through the feature extractor
        features = self.feature_extractor(x)
        
        # 2. Flatten the output for the classifier
        features = torch.flatten(features, 1)
        
        # 3. Pass the flattened features through the new classifier
        output = self.classifier(features)
        
        return output
        
