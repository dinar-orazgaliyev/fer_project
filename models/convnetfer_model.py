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
            # Block 1: in_channels=1, out_channels=64, kernel=12x12, POOL
            nn.Conv2d(
                self.input_channels, 128, kernel_size=(5, 5), padding=2
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout),
            # Block 2: in_channels=64, out_channels=128, kernel=8x8, POOL
            nn.Conv2d(128, 128, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout),
            # Block 3: in_channels=128, out_channels=256, kernel=6x6, POOL
            nn.Conv2d(128, 128, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout),
            # Block 4: in_channels=256, out_channels=512, kernel=3x3, NO POOL
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            # Block 5: in_channels=512, out_channels=512, kernel=3x3, NO POOL
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            # Block 6: in_channels=512, out_channels=512, kernel=3x3, NO POOL
            nn.Conv2d(64, 16, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
        )

        

        self.classifier = nn.Sequential(
            nn.Linear(576, 24),
            nn.LayerNorm(24),
            nn.ReLU(),
            nn.Linear(24, self.num_classes)
        )
        

    def forward(self, x):
        out = self.layers(x)

        #print(f"Output shape1: {out.shape}")

        out = out.view(out.size(0), -1)

        #print(f"Output shape2: {out.shape}")
        
        out = self.classifier(out)

        return out
