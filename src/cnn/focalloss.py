'''This module estimates the focal loss function'''
import torch
import torch.nn.functional as F
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FocalLossMultiClass(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        """
        Initializes the multi-class focal loss function.

        Parameters:
        alpha (list of floats): Weighting factor for each class
        gamma (float): Focusing parameter (default=2)
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum' (default='mean')
        """
        super(FocalLossMultiClass, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass for the multi-class focal loss calculation.

        Parameters:
        inputs (tensor): Predicted logits of shape (N, C) where C is the number of classes
        targets (tensor): Ground truth class indices of shape (N,)
        
        Returns:
        tensor: Computed focal loss
        """

        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        CE_loss = CE_loss.to(device)
        CE_loss = CE_loss.float()

        pt = torch.exp(-CE_loss)
        pt = pt.to(device)
        pt = pt.float()

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            alpha_t = alpha_t.to(device)
            alpha_t = alpha_t.float()

        else:
            alpha_t = 1
            alpha_t = alpha_t.to(device)
            alpha_t = alpha_t.float()
        
        F_loss = alpha_t * ((1 - pt) ** self.gamma )* CE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss