#%%
import torch
import time
import pickle as pkl
import os
from sklearn.model_selection import train_test_split
from utils.utils_load import load_band_structure_data   #, load_data
from utils.utils_data_kappa import generate_kappa_data_dict, generate_gru_data_dict, pkl_load
from utils.utils_model_kappa import BandLoss, GraphNetworkKappa, train
from utils.utils_plot_kappa import generate_dafaframe, plot_kappa
from utils.utils_model_gru import BandLoss, GraphNetworkGru  #!
from utils.utils_model_gru import train as train_scalar #!
from utils.utils_plot_gru import plot_scalar, generate_dafaframe_scalar #!
from copy import copy, deepcopy
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
import pickle as pkl
from ase import Atom, Atoms
palette = ['#43AA8B', '#F8961E', '#F94144']
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

#%%
file_name = os.path.basename(__file__)
print("File Name:", file_name)
run_name = "230616-130148"   #time.strftime('%y%m%d-%H%M%S', time.localtime())
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
print('parameters for kappa shape')
max_iter = 100 #200
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
temp_dim = 30
temp_idx_start = 30
temp_idx_end = temp_idx_start + temp_dim
temp_skip =2
kappa_normalize = True
irreps_out = f'{temp_dim//temp_skip}x0e' #'2x0e+2x1e+2x2e'
factor = 1
remove_above_factor=False

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
print('temperature start, end, dim, skip: ', (temp_idx_start, temp_idx_end, temp_dim, temp_skip))
print('kappa_normalize: ', kappa_normalize)
print('Div, mul factor: ', factor)
print('remove_above_factor: ', remove_above_factor)
#%%
print('parameters for kappa max')
max_iter1 = 50 #200
lmax1 = 2 #random.randint(1, 3) #2
mul1 = 27   #random.randint(3, 32) #4
nlayers1 = 2    #random.randint(1, 5) #2
r_max1 = 4 #random.randint(3, 6) #4
number_of_basis1 = 9 # random.randint(5, 20) #10
radial_layers1 = 1 #1
radial_neurons1 = 45 #random.randint(30, 150) #100
node_dim1 = 118
node_embed_dim1 = 111   #random.randint(4, 128) #32
input_dim1 = 118
input_embed_dim1 = node_embed_dim #32
temp_dim1 = temp_dim
temp_idx_start1 = temp_idx_start
temp_idx_end1 = temp_idx_end
temp_skip1 =temp_skip
kappa_normalize1 = kappa_normalize
irreps_out1 = '1x0e'
factor1 = 2000
remove_above_factor1=False

print('\nmodel parameters')
print('max iteration: ', max_iter1)
print('max l: ', lmax1)
print('multiplicity: ', mul1)
print('convolution layer: ', nlayers1)
print('cut off radius for neighbors: ', r_max1)
print('radial distance bases: ', number_of_basis1)
print('radial embedding layers: ', radial_layers1)
print('radial embedding neurons per layer: ', radial_neurons1)
print('node attribute dimension: ', node_dim1)
print('node attribute embedding dimension: ', node_embed_dim1)
print('input dimension: ', input_dim1)
print('input embedding dimension: ', input_embed_dim1)
print('irreduceble output representation: ', irreps_out1)
print('temperature start, end, dim, skip: ', (temp_idx_start1, temp_idx_end1, temp_dim1, temp_skip1))
print('kappa_normalize: ', kappa_normalize1)
print('Div, mul factor (max): ', factor1)
print('remove_above_factor: ', remove_above_factor)

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
# 1.5k materials
download_data = True
if download_data:
    os.system(f'rm -r {data_dir}/9850858*')
    os.system(f'rm -r {data_dir}/phonon/')
    os.system(f'cd {data_dir}; wget --no-verbose https://figshare.com/ndownloader/files/9850858')
    os.system(f'cd {data_dir}; tar -xf 9850858')
    os.system(f'rm -r {data_dir}/9850858*')
data = load_band_structure_data(data_dir, raw_dir, data_file)

# load anharmonic phonon data from abinit calculation
file = './data/anharmonic_fc_2.pkl'
anharmonic = pkl.load(open(file, 'rb'))
anharmonic['mpid'] = anharmonic['mpid'].map(lambda x: 'mp-' + str(x))
anharmonic['structure'] = anharmonic['mpid'].map(lambda x: 0)
anharmonic['temperature'] = anharmonic['temperature'].map(lambda x: x[temp_idx_start:temp_idx_end][::temp_skip])   #!
anharmonic['kappa'] = anharmonic['kappa'].map(lambda x: x[temp_idx_start:temp_idx_end, :][::temp_skip, :])    #!
mpids = list(anharmonic['mpid'])
for i in range(len(anharmonic)):
    mpid = anharmonic.iloc[i]['mpid']
    row = data[data['id']==mpid]
    anharmonic['structure'][i]=row['structure'].item()
anharmonic['gru']=anharmonic['kappa'].map(lambda x: np.max(np.sum(x[:, :3], axis=-1)/3))    # For predicting the highest point of kappa(T), we use the same model as Gruneisen parameter prediction. 
len0 = len(anharmonic['gru'])
if remove_above_factor1:
    anharmonic = anharmonic[anharmonic['gru']<=factor1]
    len1 = len(anharmonic['gru'])
    print('(length0, length1): ', [len0, len1])
keys = anharmonic.keys()

#%%
# load material structures from MP
mp_data = pkl_load('./data/mp_full.pkl')
mpids = sorted(list(mp_data.keys()))

df = pd.DataFrame({})
for i, mpid in enumerate(mpids[:200] + ['mp-1639', 'mp-66']):
    row = dict()
    pstruct = mp_data[mpid]
    astruct = Atoms(list(map(lambda x: x.symbol, pstruct.species)) , # list of symbols got from pymatgen
            positions=pstruct.cart_coords.copy(),
            cell=pstruct.lattice.matrix.copy(), pbc=True) 
    row['mpid'] = mpid
    row['structure'] = [astruct]
    row['temperature'] = [anharmonic['temperature'][0]]
    row['kappa'] = [np.zeros((temp_dim//temp_skip, 3))]
    row['loss'] = np.random.rand(1)
    dfn = pd.DataFrame(data = row)
    df = pd.concat([df, dfn], ignore_index = True)

df['gru']=df['kappa'].map(lambda x: np.max(np.sum(x[:, :3], axis=-1)/3))

# data = load_band_structure_data(data_dir, raw_dir, data_file)
mpdata_dict = generate_kappa_data_dict(data_dir, run_name, df, r_max, factor, kappa_normalize)
mpdata_dict1 = generate_gru_data_dict(data_dir, run_name, df, r_max, factor1)

# for normalized kappa
idx_te = list(range(len(df)))
data_set = torch.utils.data.Subset(list(mpdata_dict.values()), range(len(mpdata_dict)))
te_set = torch.utils.data.Subset(data_set, idx_te)

# for kappa max
data_set1 = torch.utils.data.Subset(list(mpdata_dict1.values()), range(len(mpdata_dict1)))
te_set1 = torch.utils.data.Subset(data_set1, idx_te)

#%%
model = GraphNetworkKappa(mul,
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

model1 = GraphNetworkGru(mul1,
                     irreps_out1,
                     lmax1,
                     nlayers1,
                     number_of_basis1,
                     radial_layers1,
                     radial_neurons1,
                     node_dim1,
                     node_embed_dim1,
                     input_dim1,
                     input_embed_dim1)
print(model1)

#%%
opt = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma = schedule_gamma)
opt1 = torch.optim.AdamW(model1.parameters(), lr = lr, weight_decay = weight_decay)
scheduler1 = torch.optim.lr_scheduler.ExponentialLR(opt1, gamma = schedule_gamma)

#%%
# train(model,opt,tr_set,tr_nums,te_set,loss_fn,run_name+'curve',max_iter,scheduler,device,batch_size,k_fold,factor,)
model_name =run_name+'curve'
print('model name: ', model_name)
model_file = f'./models/{model_name}.torch'
model.load_state_dict(torch.load(model_file)['state'])
model = model.to(device)

# Generate Data Loader
te_loader = DataLoader(te_set, batch_size = batch_size)
# Generate Data Frame
df_te = generate_dafaframe(model, te_loader, loss_fn, device, factor)
lent = len(df_te)
for i in range(lent):
    df_te['loss'][i] = np.random.randn()
# Plot the bands of TEST data
plot_kappa(df_te, header='./models/' + model_name, title='TEST_FD', n=4, m=1, lwidth=0.6, windowsize=(3, 2.5), palette=palette, formula=True)
#%%
model_name1 =run_name+'max'
print('model name: ', model_name1)
model_file1 = f'./models/{model_name1}.torch'
model1.load_state_dict(torch.load(model_file1)['state'])
model1 = model1.to(device)

# Generate Data Loader
te_loader1 = DataLoader(te_set1, batch_size = batch_size)
# Generate Data Frame
df_te1 = generate_dafaframe_scalar(model1, te_loader1, loss_fn, device, factor1)
lent1 = len(df_te1)
for i in range(lent1):
    df_te1['loss'][i] = np.random.randn()
# Plot the bands of TEST data
plot_scalar(df_te1, color=palette[0], header='./models/' + model_name1, title='TEST_FD', name='kmax', size=15, r2=True, log=False)
# %%
# combine the result
df_te2 = df_te.copy()
df_te2['real'] = df_te2['id'].map(lambda x: df_te[df_te['id']==x]['real'].item()*df_te1[df_te1['id']==x]['real'].item())
df_te2['output_test'] = df_te2['id'].map(lambda x: df_te[df_te['id']==x]['output_test'].item()*df_te1[df_te1['id']==x]['output_test'].item())
df_te2['loss'] = df_te2.apply(lambda row: np.array(loss_fn(torch.tensor(row['output_test']), torch.tensor(row['real']))), axis=1)
lent2 = len(df_te2)
for i in range(lent2):
    df_te2['loss'][i] = np.random.randn()
#%%
# plot the absolute curve 
model_name2 =run_name+'total'
# Plot the bands of TEST data
plot_kappa(df_te2, header='./models/' + model_name2, title='TEST_FD', n=4, m=1, lwidth=0.6, windowsize=(3, 2.5), palette=palette, formula=True)
plt.close()

df_save = pd.DataFrame({})

for index, row0 in df_te.iterrows():
    row = dict()
    mpid = row0['id']
    formula = mp_data[mpid].composition.reduced_formula
    gruneisen = row0['output_test'][0] # Calculate mean of np.array
    row['mpid']=mpid
    row['formula']=formula
    row['k300']=[gruneisen]
    dfn = pd.DataFrame(data = row)
    df_save = pd.concat([df_save, dfn], ignore_index = True)
df_save = df_save.sort_values(by='k300', ascending=True).reset_index(drop=True)
print(df_save)
df_save.to_csv(f'./data/rokabe/kappa{len(df_save)}.csv', index=False)
# df_save2 = pd.read_csv(f'./data/rokabe/gruneisen{len(df_save)}.csv')

plt.hist(df_save['k300'], bins=len(df_save)//10)
plt.xlabel('kappa at 300K (W/(m K))')
# %%
