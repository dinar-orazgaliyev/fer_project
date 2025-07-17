import torch
import torch.nn as nn
from typing import List, Callable
import torchvision.models as models

from .base_model import BaseModel


class ResNetFer(BaseModel):

    def __init__(
        self,
        num_classes: int,
        activation: Callable,
        dropout: float,
        norm_layer: Callable = None,
        input_shape: List[int] = [1, 48, 48],
    ):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.num_classes = num_classes
        self.activation = activation
        self.dropout = dropout
        self.norm_layer = norm_layer
        self.input_channels, self.input_height, self.input_width = input_shape
        self.model.fc = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.model.fc.in_features, self.num_classes),
        )
        


    def forward(self, x):
        return self.model(x)
        
