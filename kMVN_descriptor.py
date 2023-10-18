#%%
import torch
import time
import pickle as pkl
import os
from sklearn.model_selection import train_test_split
from utils.utils_load import load_band_structure_data   #, load_data
from utils.utils_data import generate_band_structure_data_dict
from utils.utils_model import BandLoss, GraphNetwork, train
from utils.utils_plot import generate_dafaframe, plot_bands, plot_element_count_stack
torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
seed=None #42
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
import pandas as pd
import matplotlib as mpl
from ase.visualize.plot import plot_atoms
palette = ['#43AA8B', '#F8961E', '#F94144']
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

#%%
file_name = os.path.basename(__file__)
print("File Name:", file_name)
run_name = time.strftime('%y%m%d-%H%M%S', time.localtime())
run_name_idx = "kmvn_230103-023724"
model_name = run_name

model_dir = './models'
data_dir = './data'
raw_dir = './data/phonon'
data_file = 'DFPT_band_structure.pkl'

print('torch device: ', device)
print('run name: ', run_name)
print('run name (indices to split train/test data): ', run_name_idx)
print('model name: ', model_name)
print('data_file: ', data_file)

tr_ratio = 0.9
batch_size = 1
k_fold = 5

print('\ndata parameters')
print('method: ', k_fold, '-fold cross validation')
print('training ratio: ', tr_ratio)
print('batch size: ', batch_size)

#%%
max_iter = 200 #200
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
irreps_out = '2x0e+2x1e+2x2e'
descriptor = 'non' # descriptor. Select from ['mass', 'number', 'radius', 'en', 'ie', 'dp', 'non']

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
print('Descriptor for node features: ', descriptor) 

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
download_data = False
if download_data:
    os.system(f'rm -r {data_dir}/9850858*')
    os.system(f'rm -r {data_dir}/phonon/')
    os.system(f'cd {data_dir}; wget --no-verbose https://figshare.com/ndownloader/files/9850858')
    os.system(f'cd {data_dir}; tar -xf 9850858')
    os.system(f'rm -r {data_dir}/9850858*')

#%%
data = load_band_structure_data(data_dir, raw_dir, data_file)
data_dict = generate_band_structure_data_dict(data_dir, run_name, data, r_max, descriptor)

#%%
num = len(data_dict)
tr_nums = [int((num * tr_ratio)//k_fold)] * k_fold
te_num = num - sum(tr_nums)
if run_name_idx is not None:
    with open(f'./data/idx_{run_name_idx}_tr.txt', 'r') as f: idx_tr = [int(i.split('\n')[0]) for i in f.readlines()]
    with open(f'./data/idx_{run_name_idx}_te.txt', 'r') as f: idx_te = [int(i.split('\n')[0]) for i in f.readlines()]
    print(f'./data/idx_{run_name_idx}_tr.txt')
    print(f'./data/idx_{run_name_idx}_te.txt')
else:
    idx_tr, idx_te = train_test_split(range(num), test_size=te_num, random_state=seed)
    with open(f'./data/idx_{run_name}_tr.txt', 'w') as f: 
        for idx in idx_tr: f.write(f"{idx}\n")
    with open(f'./data/idx_{run_name}_te.txt', 'w') as f: 
        for idx in idx_te: f.write(f"{idx}\n")
    print(f'./data/idx_{run_name}_tr.txt')
    print(f'./data/idx_{run_name}_te.txt')

#%%
# activate this tab to load train/valid/test indices
# run_name_idx = "221226-011042"
# with open(f'./data/idx_{run_name_idx}_tr.txt', 'r') as f: idx_tr = [int(i.split('\n')[0]) for i in f.readlines()]
# with open(f'./data/idx_{run_name_idx}_te.txt', 'r') as f: idx_te = [int(i.split('\n')[0]) for i in f.readlines()]


#%%
data_set = torch.utils.data.Subset(list(data_dict.values()), range(len(data_dict)))
tr_set, te_set = torch.utils.data.Subset(data_set, idx_tr), torch.utils.data.Subset(data_set, idx_te)

#%%
model = GraphNetwork(mul,
                     irreps_out,
                     lmax,
                     nlayers,
                     number_of_basis,
                     radial_layers,
                     radial_neurons,
                     node_dim,
                     node_embed_dim,
                     input_dim,
                     input_embed_dim)
print(model)

#%%
opt = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma = schedule_gamma)

#%%
train(model,
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
      k_fold,
      option='kmvn')

#%%
model_file = f'./models/{model_name}.torch'
model.load_state_dict(torch.load(model_file)['state'])
model = model.to(device)

#%%
# Generate Data Loader
tr_loader = DataLoader(tr_set, batch_size = batch_size)
te1_loader = DataLoader(te_set, batch_size = batch_size)

# Generate Data Frame
df_tr = generate_dafaframe(model, tr_loader, loss_fn, device)
df_te = generate_dafaframe(model, te1_loader, loss_fn, device)

#%%
# Plot the bands of TRAIN data
plot_bands(df_tr, header='./models/' + model_name, title='TRAIN', n=6, m=2, palette=palette)
# Plot the bands of TEST data
plot_bands(df_te, header='./models/' + model_name, title='TEST', n=6, m=2, palette=palette)

# %%