#%%
import torch
import time
import pickle as pkl
import os
from sklearn.model_selection import train_test_split
from utils.utils_load import load_band_structure_data   #, load_data
from utils.utils_data import generate_band_structure_data_dict, pkl_load, CombinedDataset
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
target = ['H', 'Li', 'F']
# file_extra ='./data/kyoto/df_40_low.pkl'
file_extra ='./data/kyoto/df_40.pkl'
model_name0 = "kmvn_230103-023724"
print('target elements: ', target)
print('file_extras: ', file_extra)
print('model_name0: ', model_name0)

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
download_data = True
if download_data:
    os.system(f'rm -r {data_dir}/9850858*')
    os.system(f'rm -r {data_dir}/phonon/')
    os.system(f'cd {data_dir}; wget --no-verbose https://figshare.com/ndownloader/files/9850858')
    os.system(f'cd {data_dir}; tar -xf 9850858')
    # os.system(f'rm -r {data_dir}/9850858*')

#%%
data = load_band_structure_data(data_dir, raw_dir, data_file)
data_dict = generate_band_structure_data_dict(data_dir, run_name, data, r_max)

#%%
num0 = len(data_dict)
tr_nums0 = [int((num0 * tr_ratio)//k_fold)] * k_fold
te_num0 = num0 - sum(tr_nums0)
# idx_tr, idx_te = train_test_split(range(num), test_size=te_num, random_state=seed)
# with open(f'./data/idx_{run_name}_tr.txt', 'w') as f: 
#     for idx in idx_tr: f.write(f"{idx}\n")
# with open(f'./data/idx_{run_name}_te.txt', 'w') as f: 
#     for idx in idx_te: f.write(f"{idx}\n")

# activate this tab to load train/valid/test indices
run_name_idx = model_name0 #"kmvn_230103-023724"
with open(f'./data/idx_{run_name_idx}_tr.txt', 'r') as f: idx_tr = [int(i.split('\n')[0]) for i in f.readlines()]
with open(f'./data/idx_{run_name_idx}_te.txt', 'r') as f: idx_te = [int(i.split('\n')[0]) for i in f.readlines()]


#%%
data_set0 = torch.utils.data.Subset(list(data_dict.values()), range(len(data_dict)))
tr_set0, te_set0 = torch.utils.data.Subset(data_set0, idx_tr), torch.utils.data.Subset(data_set0, idx_te)

#%%
df_kyoto = pkl_load(file_extra)
# some lines: change the index name, filter by atomic elements get train and test
# Index(['id', 'structure', 'qpts', 'band_structure'], dtype='object')
df_kyoto = df_kyoto.rename(columns={'structure_ase': 'structure'})
df_kyoto = df_kyoto.rename(columns={'band': 'band_structure'})
df_kyoto = df_kyoto.rename(columns={'structure_ase': 'structure'})
df_kyoto.drop(columns=['formula', 'sites', 'species', 'structure_pm', 'g_phs', 'g_phs_max', 'g_phs_min', 'sym_pts'], inplace=True)
# df_kyoto = df_kyoto[df_kyoto['structure'].apply(lambda x: target in x.symbols)].reset_index(drop=True)
df_kyoto = df_kyoto[df_kyoto['structure'].apply(lambda x: any([t in x.symbols for t in target]))].reset_index(drop=True)
data_dict_ky = generate_band_structure_data_dict(data_dir, run_name, df_kyoto, r_max)

num_ky = len(data_dict_ky)
tr_nums_ky = [int(num_ky//k_fold)] * k_fold
tr_num_ky = sum(tr_nums_ky)
te_num_ky = num_ky - sum(tr_nums_ky)

data_set_ky = torch.utils.data.Subset(list(data_dict_ky.values()), range(tr_num_ky))
tr_set, te_set = CombinedDataset([tr_set0, data_set_ky]), te_set0

tr_nums= [int(len(tr_set)//k_fold)] * k_fold
print('df_kyoto(target): ', len(df_kyoto))
print('tr_set0: ', len(tr_set0))
print('te_set0: ', len(te_set0))
print('data_set_ky: ', len(data_set_ky))
print('tr_set: ', len(tr_set))
print('tr_nums: ', len(tr_nums))

#%%
# get histograms
plot_element_count_stack(tr_set0, te_set0, header=None, title=None)
plot_element_count_stack(tr_set, te_set, header=None, title=None)

#%%
model0 = GraphNetwork(mul,  # original 
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
model = GraphNetwork(mul,   # with kkyoto data
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

#%%
opt = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma = schedule_gamma)

#%%
# train(model,
#       opt,
#       tr_set,
#       tr_nums,
#       te_set,
#       loss_fn,
#       run_name,
#       max_iter,
#       scheduler,
#       device,
#       batch_size,
#       k_fold,
#       option='kmvn')


#%%
# prediction
model_name0 = "kmvn_230103-023724"      # pre-trained model. Rename if you want to use the model you trained in the tabs above. 
model_file0 = f'./models/{model_name0}.torch'
model0.load_state_dict(torch.load(model_file0)['state'])
model0 = model0.to(device)

#%%
# Generate Data Loader
tr_loader0 = DataLoader(tr_set0, batch_size = batch_size)
te1_loader0 = DataLoader(te_set0, batch_size = batch_size)

# Generate Data Frame
df_tr0 = generate_dafaframe(model0, tr_loader0, loss_fn, device)
df_te0 = generate_dafaframe(model0, te1_loader0, loss_fn, device)

# Plot the bands of TRAIN data
plot_bands(df_tr0, header='./models/' + model_name0, title='TRAIN_KY', n=6, m=2, palette=palette)
plot_bands(df_te0, header='./models/' + model_name0, title='TEST_KY', n=6, m=2, palette=palette)

#%%
# with kyoto data
model_name = run_name #'230628-215029'       # pre-trained model. Rename if you want to use the model you trained in the tabs above. 
model_file = f'./models/{model_name}.torch'
model.load_state_dict(torch.load(model_file)['state'])
model = model.to(device)

# Generate Data Loader
tr_loader = DataLoader(tr_set, batch_size = batch_size)
te1_loader = DataLoader(te_set, batch_size = batch_size)

# Generate Data Frame
df_tr = generate_dafaframe(model, tr_loader, loss_fn, device)
df_te = generate_dafaframe(model, te1_loader, loss_fn, device)

# Plot the bands of TRAIN data
plot_bands(df_tr, header='./models/' + model_name, title='TRAIN_add', n=6, m=2, palette=palette)
plot_bands(df_te, header='./models/' + model_name, title='TEST_add', n=6, m=2, palette=palette)

#%%
# compare individuals
# df_te0 and df_te
# same mpid (or same index)
# plot the difference of the band structures
mpid_test = 'mp-697133'
df_target0 = df_te0[df_te0['name'].apply(lambda x: any([t in x for t in target]))]
df_target = df_te[df_te['name'].apply(lambda x: any([t in x for t in target]))]
# rband =df_te0[df_te0['id']==mpid_test]['real_band'].item()
# band0 =df_te0[df_te0['id']==mpid_test]['output_test'].item()
# band =df_te[df_te['id']==mpid_test]['output_test'].item()
# nqpts = rband.shape[0]

# bcat = np.concatenate([rband, band0, band])
# bmin, bmax = np.min(bcat), np.max(bcat)
# height = bmax - bmin
# formula = data[data['id']==mpid_test]['structure'].item().symbols
# fig, axs = plt.subplots(1,2, figsize=(12, 6))
# ax0, ax1 = axs[0], axs[1]
# ax0.plot(range(nqpts), rband, color='k', linewidth=1.2)
# ax0.plot(range(nqpts), band0, color=palette[2], linewidth=1.5)
# ax0.set_xticks([])
# ax0.set_ylim([-height*0.05, height*1.05])
# ax0.set_title('Default', fontsize=15)
# ax1.plot(range(nqpts), rband, color='k', linewidth=1.2)
# ax1.plot(range(nqpts), band, color=palette[0], linewidth=1.5)
# ax1.set_xticks([])
# ax1.set_ylim([-height*0.05, height*1.05])
# ax1.set_title(f'Default + Togo ({len(df_kyoto)} materials with {target})', fontsize=15)
# fig.suptitle(f'{mpid_test}: {formula}', fontsize=18)

#%%
# get
folder_path = './models/' + model_name
if not os.path.exists(folder_path):
    # Create the folder if it doesn't exist
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created successfully.")
else:
    print(f"Folder '{folder_path}' already exists.")

mpids_target = list(df_target0['id'])

for mpid_test in mpids_target:
    rband =df_te0[df_te0['id']==mpid_test]['real_band'].item()
    band0 =df_te0[df_te0['id']==mpid_test]['output_test'].item()
    band =df_te[df_te['id']==mpid_test]['output_test'].item()
    nqpts = rband.shape[0]

    bcat = np.concatenate([rband, band0, band])
    bmin, bmax = np.min(bcat), np.max(bcat)
    height = bmax - bmin
    formula = data[data['id']==mpid_test]['structure'].item().symbols
    fig, axs = plt.subplots(1,2, figsize=(12, 6))
    ax0, ax1 = axs[0], axs[1]
    ax0.plot(range(nqpts), rband, color='k', linewidth=1.2)
    ax0.plot(range(nqpts), band0, color=palette[2], linewidth=1.5)
    ax0.set_xticks([])
    ax0.set_ylim([-height*0.05, height*1.05])
    ax0.set_title('Default', fontsize=15)
    ax1.plot(range(nqpts), rband, color='k', linewidth=1.2)
    ax1.plot(range(nqpts), band, color=palette[0], linewidth=1.5)
    ax1.set_xticks([])
    ax1.set_ylim([-height*0.05, height*1.05])
    ax1.set_title(f'Default + Togo ({len(df_kyoto)} materials with {target})', fontsize=15)
    fig.suptitle(f'{mpid_test}: {formula}', fontsize=18)
    fig.savefig(os.path.join(folder_path, mpid_test+'.png'))
    fig.savefig(os.path.join(folder_path, mpid_test+'.pdf'))
    


#%%