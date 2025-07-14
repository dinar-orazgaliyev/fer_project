import numpy as np 
import torch.nn as nn

from typing import List, Callable

from .base_model import BaseModel

class ConvNetFer(BaseModel):

    def __init__(self,input_shape:List[int], num_classes:int, hidden_layers:List[int], activation,dropout,norm_layer):
        super().__init__()
        self.input_channels = input_shape
        self.num_classes = num_classes 
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout = dropout
        self.norm_layer = norm_layer
        self.kernel_size = (3,3)
        self.maxpool_size = (2,2)
        self.stride_size = (2,2)
        self.__build_model()
    
    def __build_model(self):
        layers = []
        for id,hidden_size in enumerate(self.hidden_layers):
            if id == 0:
                layers.append(nn.Conv2d(in_channels = 1, out_channels = hidden_size, kernel_size=self.kernel_size,padding=1))
            else:
                layers.append(nn.Conv2d(in_channels = self.hidden_layers[id-1], out_channels = hidden_size, kernel_size = self.kernel_size,padding=1))
            if self.norm_layer:
                layers.append(self.norm_layer(num_features = hidden_size))
            if id < len(self.hidden_layers) - 1:
                layers.append(nn.MaxPool2d(kernel_size = self.maxpool_size, stride = self.stride_size))
            
            layers.append(self.activation())
            if self.dropout and self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))

        fc_input_size = self.hidden_layers[-1] 
        self.classifier = nn.Sequential(nn.Linear(fc_input_size,100), nn.BatchNorm1d(100), nn.ReLU(),nn.Dropout(self.dropout), nn.Linear(100,self.num_classes))
        self.layers = nn.Sequential(*layers) 

    def forward(self,x):
        out = self.layers(x)
        out = out.view(out.size(0),-1)
        out = self.classifier(out)
        return out
