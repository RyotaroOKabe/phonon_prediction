from sympy import im
import torch
from torch.nn.modules.loss import _Loss
from torch_scatter import scatter
from torch_geometric.loader import DataLoader
from e3nn.o3 import Irrep, Irreps, spherical_harmonics, TensorProduct, FullyConnectedTensorProduct
from e3nn.nn import Gate, FullyConnectedNet
from e3nn.math import soft_one_hot_linspace
import matplotlib.pyplot as plt
import pandas as pd
import math
import time
from utils.utils_plot import generate_dafaframe, plot_bands
torch.autograd.set_detect_anomaly(True)

class BandLoss(_Loss):
    def __init__(self, size_average = None, reduce = None, reduction: str = 'mean') -> None:
        super(BandLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return torch.sum(torch.pow(torch.abs(input - target)/torch.max(torch.abs(target)), 2)) \
               /torch.numel(target)

def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = Irreps(irreps_in1).simplify()
    irreps_in2 = Irreps(irreps_in2).simplify()
    ir_out = Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False

class CustomCompose(torch.nn.Module):
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second
        self.irreps_in = self.first.irreps_in
        self.irreps_out = self.second.irreps_out

    def forward(self, *input):
        x = self.first(*input)
        self.first_out = x.clone()
        x = self.second(x)
        self.second_out = x.clone()
        return x

class GraphNetwork(torch.nn.Module):
    pass



def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    pass

def loglinspace(rate, step, end=None):
    t = 0
    while end is None or t <= end:
        yield t
        t = int(t + 1 + step*(1 - math.exp(-t*rate/step)))

def train(model,
          opt,
          tr_set,
          tr_nums,
          te_set,
          loss_fn,
          run_name,
          max_iter,
          scheduler,
          device,
          batch_size,
          k_fold):
    pass



