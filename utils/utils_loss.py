import torch
from torch.nn.modules.loss import _Loss

class BandLoss(_Loss):
    def __init__(self, size_average = None, reduce = None, reduction = 'mean'):
        super().__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return torch.sum(torch.pow(torch.abs(input - target)/torch.max(torch.abs(target)), 2))/torch.numel(target)
    

class BandLoss_MAE_Norm(_Loss):
    def __init__(self, size_average = None, reduce = None, reduction = 'mean'):
        super().__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return torch.sum(torch.abs(input - target)/torch.max(torch.abs(target)))/torch.numel(target)
