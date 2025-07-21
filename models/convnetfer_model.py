import torch
import torch.nn as nn
from typing import List, Callable
import logging

from .base_model import BaseModel
logger = logging.getLogger(__name__)

class ConvNetFer(BaseModel):
    def __init__(
        self,
        num_classes: int,
        activation: Callable,
        dropout: float,
        norm_layer: Callable = None,
        input_shape: List[int] = [1,48,48], 
    ):
        super().__init__()
        self.input_channels, self.input_height, self.input_width = input_shape
        self.num_classes = num_classes
        #self.activation = activation
        self.dropout = dropout
        self.norm_layer = norm_layer
        
        self.maxpool_size = (2, 2)
        self.stride_size = (2, 2)

        self.__build_model()
        logger.info(self.layers)
        logger.info(self.classifier)

    def __build_model(self):
        self.layers = nn.Sequential(
            # --- Conv Block 1: 48x48 -> 24x24 ---
            # in_channels=1, out_channels=64
            nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout),
            # --- Conv Block 2: 24x24 -> 12x12 ---
            # in_channels=64, out_channels=128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout),
            # --- Conv Block 3: 12x12 -> 6x6 ---
            # in_channels=128, out_channels=256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout),
            # --- Conv Block 4: 6x6 -> 3x3 ---
            # in_channels=256, out_channels=512
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout),
        )

        # After the conv layers, the feature map is (B, 512, 3, 3).
        # We need to calculate the flattened size for the linear layer.
        # Flattened size = 512 * 3 * 3 = 4608
        self.classifier = nn.Sequential(
            nn.Linear(4608, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # A stronger dropout is recommended for the dense classifier part
            # to prevent overfitting.
            nn.Dropout(0.4),
            nn.Linear(256, self.num_classes),
        )
        

    def forward(self, x):
        out = self.layers(x)

        #print(f"Output shape1: {out.shape}")

        out = out.view(out.size(0), -1)

        #print(f"Output shape2: {out.shape}")
        
        out = self.classifier(out)

        return out