#%%
"""
https://www.notion.so/230402-zeolite-phonon-ba985314a45441749f258234ec7931c8

"""
import torch
import time
import pickle as pkl
import os
from sklearn.model_selection import train_test_split
from utils.utils_load import load_band_structure_data   #, load_data
from utils.utils_data import generate_band_structure_data_dict
from utils.utils_model import BandLoss, GraphNetwork, train
from utils.utils_plot import generate_dafaframe, plot_bands, plot_element_count_stack
from utils.utils_plot_path import plot_bands_qlabels, simname
from utils.utils_plot import get_spectra
torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = 'cuda:7'
else:
    device = 'cpu'
seed=None #42
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
import pandas as pd
import matplotlib as mpl
from ase.visualize.plot import plot_atoms
from ase import Atom, Atoms
from utils.utils_path import get_path
palette = ['#43AA8B', '#F8961E', '#F94144']
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

#%%
run_name = time.strftime('%y%m%d-%H%M%S', time.localtime())
model_dir = './models'
data_dir = './data'

print('torch device: ', device)
print('model name: ', run_name)

tr_ratio = 0.9
batch_size = 1
k_fold = 5

print('\ndata parameters')
print('method: ', k_fold, '-fold cross validation')
print('training ratio: ', tr_ratio)
print('batch size: ', batch_size)

#%%
# max_iter = 200 #200
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
# print('max iteration: ', max_iter)
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
# load zeolite data 
import glob
from os.path import join as opj
zeo_dir = '/data1/rokabe/zeolite'   #!
NPZPATH = opj(zeo_dir, 'npz')
files = glob.glob(opj(NPZPATH, '*.npz'))
# v = np.load(files[0])
# print(v['xyz'].shape)
# print(v['lattice'].shape)
# print(v['numbers'])

#%%
df = pd.DataFrame({})
idx_zeo = -1
for i, file in enumerate(files):
    Data = dict()
    name = file[len(NPZPATH)+1:-4]
    npdata = np.load(file)
    ccoords = npdata['xyz'][idx_zeo, :, :]
    lattice = npdata['lattice'][idx_zeo, :,:]
    species = npdata['numbers']
    natms = len(species)
    symbols = [Atom(nb).symbol for nb in species]
    atoms = Atoms(symbols, positions = ccoords,
                                cell = lattice,
                                pbc=True) 
    out_dict = get_path(atoms)
    qpts = np.array(out_dict['qpts'])
    qticks = out_dict['qticks']
    Data['id'] = name
    Data['structure'] = [atoms]
    Data['qpts'] = [qpts]
    Data['qticks'] = [qticks]
    Data['band_structure'] = [np.zeros((len(qpts), 3*natms))]
    dfn = pd.DataFrame(data = Data)
    df = pd.concat([df, dfn], ignore_index = True)

#%%
# orthrombic cell
orth_point_coords = {
    'Gamma': [0.0, 0.0, 0.0],
    'X': [0.5, 0.0, 0.0],
    'Y': [0.0, 0.5, 0.0],
    'Z': [0.0, 0.0, 0.5],
    'T': [0.0, 0.5, 0.5],
    'U': [0.5, 0.0, 0.5],
    'S': [0.5, 0.5, 0.0],
    'R': [0.5, 0.5, 0.5],
}
orth_path = [
    ('Z', 'Gamma'), 
    ('Gamma', 'S'),
    ('S', 'R'),
    ('R', 'Z'),
    ('Z', 'T'),
    ('T', 'Y'),
    ('Y', 'Gamma')
]

orth_path_points = [20 for _ in orth_path]

# hexagonal cell
hex_point_coords = {
    'Gamma': [0.0, 0.0, 0.0],
    'A': [0.0, 0.0, 0.5],
    'K': [2/3, 1/3, 0.0],
    'H': [2/3, 1/3, 1/2],
    'M': [0.5, 0.0, 0.0],
    'L': [0.5, 0.0, 0.5],
}

hex_path = [
    ('A', 'H'),
    ('H', 'L'),
    ('L', 'A'),
    ('A', 'Gamma'),
    ('Gamma', 'K'),
    ('K', 'M'),
    ('M', 'Gamma'),
]

hex_path_points =[19,11,15,36,29,15,17]

def qpts_path(pcoords, path, npoints):
    qpts = []
    qticks = []
    for i in range(len(path)):
        sym_start = path[i][0]
        sym_end = path[i][-1]
        qpt_start = np.array(pcoords[sym_start])
        qpt_end = np.array(pcoords[sym_end])
        pathway = qpt_end-qpt_start
        npts = npoints[i]
        # qpts.append(qpt_start)
        for j in range(npts):
            qpt = qpt_start + pathway * j / npts
            qpts.append(qpt)
            if j==0:
                if sym_start=='Gamma':
                    qticks.append("$\Gamma$")
                else: 
                    qticks.append(sym_start)
            else:
                qticks.append('')
    qpts.append(qpt_end)
    if sym_end=='Gamma':
        qticks.append("$\Gamma$")
    else: 
        qticks.append(sym_end)
        
    return np.stack(qpts, axis=0), qticks


#%%
boz_set = qpts_path(orth_point_coords, orth_path, orth_path_points)
etr_set = qpts_path(hex_point_coords, hex_path, hex_path_points)
targets = {'BOZ': boz_set, 'ETR': etr_set}
df1 = pd.DataFrame({})
for k, v in targets.items():
    row = dict()
    row0 = df[df['id']==k]
    row['id'] = k
    row['structure'] = [row0['structure'].item()]
    row['qpts'] = [targets[k][0]]
    row['qticks'] = [targets[k][1]]
    row['band_structure'] = [row0['band_structure'].item()]
    dfn = pd.DataFrame(data = row)
    df1 = pd.concat([df1, dfn], ignore_index = True)

data = df1 #df[df['id'].isin(['BOZ', 'ETR'])].reset_index()
# r_max = 3   #!
data_dict = generate_band_structure_data_dict(data_dir, run_name, data, r_max)

#%%
num = len(data_dict)
tr_nums = 0 #[int((num * tr_ratio)//k_fold)] * k_fold
te_num = num # - sum(tr_nums)
# idx_tr, idx_te = train_test_split(range(num), test_size=te_num, random_state=seed)
idx_tr = []
idx_te = list(range(num))

#%%
data_set = torch.utils.data.Subset(list(data_dict.values()), range(len(data_dict)))
tr_set, te_set = torch.utils.data.Subset(data_set, idx_tr), torch.utils.data.Subset(data_set, idx_te)

#%%""
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
tr_loader = DataLoader(tr_set, batch_size = batch_size)
te1_loader = DataLoader(te_set, batch_size = batch_size)

def generate_dafaframe(model, dataloader, loss_fn, device, option='kmvn'):
    with torch.no_grad():
        df = pd.DataFrame(columns=['id', 'name', 'loss', 'real_band', 'output_test'])
        for d in dataloader:
            d.to(device)
            # if len(d.pos) > 60:
            #     continue
            if option in ['kmvn', 'mvn']:
                Hs, shifts = model(d)
                output = get_spectra(Hs, shifts, d.qpts)
            else:
                output = model(d)
            loss = loss_fn(output, d.y).cpu()
            real = d.y.cpu().numpy()*1000
            pred = output.cpu().numpy()*1000
            rrr = {'id': d.id, 'name': d.symbol, 'loss': loss.item(), 'real_band': list(real), 'output_test': list(np.array([pred]))}
            df0 = pd.DataFrame(data = rrr)
            df = pd.concat([df, df0], ignore_index=True)
    return df

# Generate Data Frame
df_tr = generate_dafaframe(model, tr_loader, loss_fn, device)
df_te = generate_dafaframe(model, te1_loader, loss_fn, device)
df_te.to_pickle('./models/' + model_name + '_zeolite_GT'.pkl)

#%%
lent = len(df_te)
print('df_te: ', lent)
for i in range(lent):
    df_te['loss'][i] = np.random.randn()
#%%
# Plot the bands of TEST data
zpalette = ['#43AA8B' for _ in range(3)]
# plot_bands(df_te, header='./models/' + model_name, title='TEST', n=1, m=1, palette=zpalette, gtruth=False)
# plot_bands_qlabels(df_te, header='./models/' + model_name + '_zeolite', 
#            title='TEST', n=1, m=2, windowsize=(4,3), palette=zpalette, 
#            gtruth=False, datadf=data)

fig, axs = plt.subplots(1,2, figsize=(10, 5))
ds = df_te
fontsize = 10
header = './models/' + model_name + '_zeolite_GT'
title = 'TEST'
for i in range(len(ds)):
    ax = axs[i]
    realb = ds.iloc[i]['real_band']
    predb = ds.iloc[i]['output_test']
    xpts = realb.shape[0]
    ax.plot(range(xpts), predb, color='#43AA8B', linewidth=1)
    ax.set_title(f"[${ds.iloc[i]['id']}$] {simname(ds.iloc[i]['name']).translate(sub)}", fontsize=fontsize*1.8)
    min_y1, max_y1 = np.min(realb), np.max(realb)
    min_y2, max_y2 = np.min(predb), np.max(predb)
    min_y = min([min_y1, min_y2])
    max_y = max([max_y1, max_y2])
    width_y = max_y - min_y
    ax.set_ylim(min_y-0.05*width_y, max_y+0.05*width_y)
    labelsize = fontsize*1.5
    ax.tick_params(axis='y', which='major', labelsize=labelsize)
    ax.tick_params(axis='y', which='minor', labelsize=labelsize)
    # ax.set_xticks([])
    qlabels = df1[df1['id']==ds.iloc[i]['id']]['qticks'].item()
        # print(qlabels)
    ax.set_xticks(range(xpts), qlabels, fontsize=labelsize)
    ax.tick_params(bottom = False)

fig.tight_layout()
fig.subplots_adjust(hspace=0.6)
fig.patch.set_facecolor('white')
if title: fig.suptitle(title, ha='center', y=1., fontsize=fontsize)
fig.savefig(f"{header}_{title}_bands.png")
fig.savefig(f"{header}_{title}_bands.pdf")
print(f"{header}_{title}_bands.pdf")


# %%
