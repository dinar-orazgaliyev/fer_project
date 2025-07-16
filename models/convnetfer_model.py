import torch
import torch.nn as nn
from typing import List, Callable

from .base_model import BaseModel

class ConvNetFer(BaseModel):
    def __init__(
        self,
              # [channels, height, width]
        num_classes: int,
        hidden_layers: List[int],
        activation: Callable,
        dropout: float,
        norm_layer: Callable = None,
        input_shape: List[int] = [1,48,48], 
    ):
        super().__init__()
        self.input_channels, self.input_height, self.input_width = input_shape
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout = dropout
        self.norm_layer = norm_layer
        self.kernel_size = (3, 3)
        self.maxpool_size = (1, 1)
        self.stride_size = (2, 2)

        self.__build_model()

    def __build_model(self):
        layers = []

        for idx, hidden_size in enumerate(self.hidden_layers):
            in_channels = self.input_channels if idx == 0 else self.hidden_layers[idx - 1]
            layers.append(nn.Conv2d(in_channels, hidden_size, kernel_size=self.kernel_size, padding=1))

            if self.norm_layer:
                layers.append(self.norm_layer(num_features=hidden_size))

            if idx < len(self.hidden_layers) - 1:
                layers.append(nn.MaxPool2d(kernel_size=self.maxpool_size, stride=self.stride_size))

            layers.append(self.activation())

            if self.dropout and self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))

        self.layers = nn.Sequential(*layers)
        print(layers)
        # Dummy forward pass to determine fc input size
        dummy_input = torch.zeros(1, self.input_channels, self.input_height, self.input_width)
        print(f"Dummy input shape: {dummy_input.shape}")
        self.layers.eval()
        with torch.no_grad():
            out = self.layers(dummy_input)
        fc_input_size = out.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(fc_input_size, 100),
            #nn.BatchNorm1d(100),
            nn.LayerNorm(100),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(100, self.num_classes)
        )

    def forward(self, x):
        out = self.layers(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
