#%%
import os
os.environ['PYTHONPATH'] = '/home/rokabe/anaconda3/envs/pdos/lib/python3.9/site-packages/pymatgen'
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
import pandas as pd
import matplotlib as mpl
from ase.visualize.plot import plot_atoms
import sys
from pymatgen.core.structure import Structure
sys.executable

#%%
# !pip install pymatgen
# !pip install torch
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
run_name = time.strftime('%y%m%d-%H%M%S', time.localtime())
model_dir = './models'
data_dir = './data'
raw_dir = './data/phonon'
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
max_iter = 1#200
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
# with open(('./data/kyoto/df_40.pkl'), 'rb') as f:
with open(('/home/rokabe/data1/phonon/g_phonon_v1/data/df_40.pkl'), 'rb') as f:
    data = pkl.load(f)

# remove the column: structure_pm
data = data.drop('structure_pm', axis=1)
# rename the key structure_ase to structure
data = data.rename(columns={'structure_ase': 'structure', 'band': 'band_structure'})


num = len(data)
for i in range(num):
    row = data.iloc[i]
    astruct, real, qpts = row['structure'], row['band_structure'], row['qpts']
    # if len(astruct)!=58:
    # print(i, len(astruct), real.shape, qpts.shape)
    if real.shape[0]!=qpts.shape[0]:
        print(i, 'real.shape[0]!=qpts.shape[0')
    if len(astruct)*3!=real.shape[1]:
        print(i, 'len(astruct)*3!=real.shape[1]')


# data = load_band_structure_data(data_dir, raw_dir, data_file)
# data['fmin'] = data['band_structure'].map(lambda x: np.min(x))

data_dict = generate_band_structure_data_dict(data_dir, run_name, data, r_max)
use_idx_split=True  # True if you want to load train/valid/test indices
num = len(data_dict)
# tr_nums = [int((num * tr_ratio)//k_fold)] * k_fold
te_num = num #- sum(tr_nums)
idx_te = list(range(te_num))


#%%
data_set = torch.utils.data.Subset(list(data_dict.values()), range(len(data_dict)))
# tr_set, te_set = torch.utils.data.Subset(data_set, idx_tr), torch.utils.data.Subset(data_set, idx_te)
te_set = torch.utils.data.Subset(data_set, idx_te)

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

opt = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma = schedule_gamma)

#%%
model_name = "kmvn_230103-023724"      # pre-trained model. Rename if you want to use the model you trained in the tabs above. 
model_file = f'./models/{model_name}.torch'
model.load_state_dict(torch.load(model_file)['state'])
model = model.to(device)

#%%
# Generate Data Loader
# tr_loader = DataLoader(tr_set, batch_size = batch_size)
te1_loader = DataLoader(te_set, batch_size = batch_size)

# Generate Data Frame
# df_tr = generate_dafaframe(model, tr_loader, loss_fn, device)
df_te = generate_dafaframe(model, te1_loader, loss_fn, device)

# Plot the bands of TEST data
plot_bands(df_te, header='./models/' + model_name, title='TEST_ky40', n=6, m=2, palette=palette)

# save dataframe as pkl
df_te.to_pickle(f'./data/source/04_kmvn/{model_name}_df_ky40.pkl')
data.to_pickle(f'./data/source/04_kmvn/{model_name}_data_ky40.pkl')







