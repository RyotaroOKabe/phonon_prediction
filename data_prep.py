#%%
##########################

# Import

##########################
import torch
import time
import pickle as pkl
import os
import random

from utils.utils_load import load_band_structure_data
from utils.utils_data import generate_gamma_data_dict
from utils.utils_model import BandLoss, GraphNetworkVVN, train  #! update
from utils.utils_plot import generate_dafaframe, plot_gphonons, plot_element_count_stack    #! update

torch.set_default_dtype(torch.float64)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
import pandas as pd

# data visualization
import matplotlib as mpl
from ase.visualize.plot import plot_atoms
palette = ['#43AA8B', '#F8961E', '#F94144']
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

#%%
##########################

# Parameters (ToDo: move some parameters to the section of GNN model set up)

##########################

run_name = time.strftime('%y%m%d-%H%M%S', time.localtime())
# run_name = '221028-172544'
model_dir = './models'
data_dir = './data'
raw_dir = './data/DFPT_band_structure'
data_file = 'DFPT_band_structure.pkl'

print('torch device: ', device)
print('model name: ', run_name)
print('data_file: ', data_file)

tr_ratio = 0.9
batch_size = 1
k_fold = 5

print('\ndata parameters')
print('method: ', k_fold, '-fold cross validation')
print('training ratio: ', tr_ratio)
print('batch size: ', batch_size)

#%%
##########################

# Parameters (ToDo: move some parameters to the section of GNN model set up)

##########################
# phonon_11320.out
max_iter = 1 #200
lmax = 2 #2
mul = 4 #4
nlayers = 2 #5
r_max = 4 #4
number_of_basis = 10 #10
radial_layers = 1 #1
radial_neurons = 100 #100
node_dim = 118
node_embed_dim = 32 #32
input_dim = 118
input_embed_dim = 32 #32
irreps_out = '1x0e'
option='vvn'

print('\nmodel parameters')
print('max iteration: ', max_iter)
print('max l: ', lmax)
print('multiplicity: ', mul)
print('convolution layer: ', nlayers)
print('cut off radius for neighbors: ', r_max)
print('radial distance bases: ', number_of_basis)
print('radial embedding layers: ', radial_layers)
print('radial embedding neurons per layer: ', radial_neurons)
print('node attribute dimension: ', node_dim)
print('node attribute embedding dimension: ', node_embed_dim)
print('input dimension: ', input_dim)
print('input embedding dimension: ', input_embed_dim)
print('irreduceble output representation: ', irreps_out)
print('Model option: ', option)

#%%
loss_fn = BandLoss()
lr = 0.005 # random.uniform(0.001, 0.05) #0.005
weight_decay = 0.05 # random.uniform(0.01, 0.5) #0.05
schedule_gamma = 0.96 # random.uniform(0.85, 0.99) #0.96

print('\noptimization parameters')
print('loss function: ', loss_fn)
print('optimization function: AdamW')
print('learning rate: ', lr)
print('weight decay: ', weight_decay)
print('learning rate scheduler: exponentialLR')
print('schedule factor: ', schedule_gamma)

#%%
##########################

# Load data from pkl or csv
# (DFPT, Kyoto)

##########################

data = load_band_structure_data(data_dir, raw_dir, data_file)
# data = load_data(f'./data/data.csv')
# pkl.dump(data, open(data_dir + f'/data.pkl', 'wb'))

data_dict = generate_gamma_data_dict(data_dir, run_name, data, r_max)   #TODO: complete build_data_vvn function

num = len(data_dict)
tr_nums = [int((num * tr_ratio)//k_fold)] * k_fold
te_num = num - sum(tr_nums)
tr_set, te_set = torch.utils.data.random_split(list(data_dict.values()), [num - te_num, te_num])

#%%
os.system(f'rm -r {data_dir}/9850858*')
os.system(f'rm -r {data_dir}/phonon/')
#%%
os.system(f'cd {data_dir}; wget https://figshare.com/ndownloader/files/9850858')
#%%
# os.system(f'cd {data_dir}; mv 9850858 data.tar.bz2; tar -xf data.tar.bz2; rm -r data.tar.bz2')
os.system(f'cd {data_dir}; tar -xf 9850858; rm -r data.tar.bz2')

#%%
