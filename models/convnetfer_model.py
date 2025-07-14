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
