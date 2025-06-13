import torch
import torch.nn as nn
from torchsurv.loss.cox import neg_partial_log_likelihood


class CoxLoss(nn.Module):
    def __init__(self, ties_method='efron', reduction='mean'):
        super().__init__()
        self.ties_method = ties_method
        self.reduction = reduction

    def __call__(self, x, y_event, y_time):
        return neg_partial_log_likelihood(x, y_event, y_time, ties_method=self.ties_method, reduction=self.reduction)
