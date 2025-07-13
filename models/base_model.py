import torch.nn as nn 
import numpy as np 
from abc import abstractmethod

class BaseModel(nn.Module):

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic
        """
        raise NotImplementedError
    
    def __str__(self):
        """
        Model prints with number of trainable parameters
        """

        ret_str = super().__str__()
        
        #### TODO #######################################
        # Print the number of **trainable** parameters  #
        # by appending them to ret_str                  #
        #################################################     

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        ret_str += f"\nTrainable parameters: {num_params}" 
           
        return ret_str