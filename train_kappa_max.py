#%%
import torch
import time
import pickle as pkl
import os
from sklearn.model_selection import train_test_split
from utils.utils_load import load_band_structure_data   #, load_data
from utils.utils_data_kappa import generate_gru_data_dict
from utils.utils_model_gru import BandLoss, GraphNetworkGru, train
from utils.utils_plot_gru import generate_dafaframe_scalar, plot_scalar, plot_element_count_stack
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
import random
palette = ['#43AA8B', '#F8961E', '#F94144']
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

#%%
file_name = os.path.basename(__file__)
print("File Name:", file_name)
run_name = time.strftime('%y%m%d-%H%M%S', time.localtime())
model_dir = './models'
data_dir = './data'
raw_dir = './data/phonon'
data_file = 'DFPT_band_structure.pkl'

print('torch device: ', device)
print('model name: ', run_name)
print('data_file: ', data_file)

tr_ratio = 1.0
batch_size = 1
k_fold = 5

print('\ndata parameters')
print('method: ', k_fold, '-fold cross validation')
print('training ratio: ', tr_ratio)
print('batch size: ', batch_size)

#%%
max_iter = 200 #200
lmax = 2 #random.randint(1, 3) #2
mul = 15 #random.randint(3, 32) #4
nlayers = 1 #random.randint(1, 5) #2
r_max = 4 #random.randint(3, 6) #4
number_of_basis = 6 #random.randint(5, 20) #10
radial_layers = 1 #1
radial_neurons = 121 #random.randint(30, 150) #100
node_dim = 118
node_embed_dim = 78 #random.randint(4, 128) #32
input_dim = 118
input_embed_dim = node_embed_dim #32
irreps_out = '1x0e' #'2x0e+2x1e+2x2e'
temp_dim = 50
temp_idx_start = 30
temp_idx_end = temp_idx_start + temp_dim
temp_skip =2
factor = 2000
remove_above_factor=True

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
print('Div, mul factor: ', factor)
print('remove_above_factor: ', remove_above_factor)
print('temperature start, end, dim, skip: ', (temp_idx_start, temp_idx_end, temp_dim, temp_skip))

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
download_data = True
if download_data:
    os.system(f'rm -r {data_dir}/9850858*')
    os.system(f'rm -r {data_dir}/phonon/')
    os.system(f'cd {data_dir}; wget --no-verbose https://figshare.com/ndownloader/files/9850858')
    os.system(f'cd {data_dir}; tar -xf 9850858')
    os.system(f'rm -r {data_dir}/9850858*')
data = load_band_structure_data(data_dir, raw_dir, data_file)
file = './data/anharmonic_fc_2.pkl'
anharmonic = pkl.load(open(file, 'rb'))
anharmonic['mpid'] = anharmonic['mpid'].map(lambda x: 'mp-' + str(x))
anharmonic['structure'] = anharmonic['mpid'].map(lambda x: 0)
mpids = list(anharmonic['mpid'])
for i in range(len(anharmonic)):
    mpid = anharmonic.iloc[i]['mpid']
    row = data[data['id']==mpid]
    anharmonic['structure'][i]=row['structure'].item()
anharmonic['gru']=anharmonic['kappa'].map(lambda x: np.max(np.sum(x[temp_idx_start:temp_idx_end, :3], axis=-1)/3))
len0 = len(anharmonic['gru'])
if remove_above_factor:
    anharmonic = anharmonic[anharmonic['gru']<=factor]
    len1 = len(anharmonic['gru'])
    print('(length0, length1): ', [len0, len1])
keys = anharmonic.keys()

#%%
# data = load_band_structure_data(data_dir, raw_dir, data_file)
data_dict = generate_gru_data_dict(data_dir, run_name, anharmonic, r_max, factor)

#%%
num = len(data_dict)
tr_nums = [int((num * tr_ratio)//k_fold)] * k_fold
te_num = num - sum(tr_nums)
idx_tr, idx_te = train_test_split(range(num), test_size=te_num, random_state=seed)
with open(f'./data/idx_{run_name}_tr.txt', 'w') as f: 
    for idx in idx_tr: f.write(f"{idx}\n")
with open(f'./data/idx_{run_name}_te.txt', 'w') as f: 
    for idx in idx_te: f.write(f"{idx}\n")

#%%
# activate this tab to load train/valid/test indices
# run_name_idx = "221226-011042"
# with open(f'./data/idx_{run_name_idx}_tr.txt', 'r') as f: idx_tr = [int(i.split('\n')[0]) for i in f.readlines()]
# with open(f'./data/idx_{run_name_idx}_te.txt', 'r') as f: idx_te = [int(i.split('\n')[0]) for i in f.readlines()]


#%%
data_set = torch.utils.data.Subset(list(data_dict.values()), range(len(data_dict)))
tr_set, te_set = torch.utils.data.Subset(data_set, idx_tr), torch.utils.data.Subset(data_set, idx_te)

#%%
model = GraphNetworkGru(mul,
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
      factor,
      )


#%%
model_name =run_name
# Generate Data Loader
tr_loader = DataLoader(tr_set, batch_size = batch_size)
te1_loader = DataLoader(te_set, batch_size = batch_size)

# Generate Data Frame
df_tr = generate_dafaframe_scalar(model, tr_loader, loss_fn, device, factor)
df_te = generate_dafaframe_scalar(model, te1_loader, loss_fn, device, factor)

# Plot the bands of TRAIN data
plot_scalar(df_tr, color=palette[0], header='./models/' + model_name, title='TRAIN', name='kmax', size=15, r2=True)

# Plot the bands of TEST data
plot_scalar(df_te, color=palette[0], header='./models/' + run_name, title='TEST', name='kmax', size=15, r2=True)

# %%
