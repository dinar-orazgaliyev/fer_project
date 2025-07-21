import torch
import torch.nn.functional as F
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)  
        ce_loss = -log_prob.gather(1, target.unsqueeze(1)).squeeze()
        focal_loss = (1 - torch.exp(-ce_loss)) ** self.gamma * ce_loss
        
        if self.weight is not None:
            focal_loss = focal_loss * self.weight[target]
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# # Then instantiate it like this:
# self.criterion = FocalLoss(gamma=2.0, reduction='mean')  # optionally set weight if you want class weights
