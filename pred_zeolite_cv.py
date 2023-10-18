#%%
"""
https://www.notion.so/230911-zeolite-work-ab2d3c2489064f62abf2bf24229cb34a

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
from utils.utils_plot_path import plot_bands_qlabels
from utils.utils_dos import plot_dos, DoS_data, ddf, dos_from_band, dos_from_band_gauss, dos2cv, plot_cv
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
from os.path import join
# zeo_dir = '/data1/rokabe/zeolite'   #!
# NPZPATH = opj(zeo_dir, 'npz')
# files = glob.glob(opj(NPZPATH, '*.npz'))
zeo_dir = '/home/rokabe/data1/phonon/phonon_prediction/data/zeo_DFT_calculations'
dos_dir = join(zeo_dir, 'DFT_dos')
struct_dir = join(zeo_dir, 'DFT_structures')
cp_file = join(zeo_dir, 'DFT_site_projected_cp.csv')
tr_file = join(zeo_dir, 'DFT_trainset.csv')
te_file = join(zeo_dir, 'DFT_testset.csv')

#%%
# load dataframe fro csv files (tr + te)
csv_file0 = tr_file
csv_file1 = te_file
# Read the CSV file into a DataFrame
df_dft0 = pd.read_csv(csv_file0, index_col=False)
df_dft1 = pd.read_csv(csv_file1, index_col=False)
df_dft = pd.concat([df_dft0, df_dft1], ignore_index=True)
df_dft = df_dft.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])
column_to_front = 'name'
column_to_insert = df_dft.pop(column_to_front)
df_dft.insert(0, column_to_front, column_to_insert)

#%%
# load structures
from pymatgen.io.cif import CifParser
# Specify the path to your CIF file
struct_files = [s for s in os.listdir(struct_dir) if s.endswith('cif') ]
struct_dict = {}
for sf in struct_files: 
    name = sf[:-4]
    cif_file_path = join(struct_dir, sf)  # Replace 'your_file.cif' with the actual file path
    # Create a CIF parser and parse the CIF file
    parser = CifParser(cif_file_path)
    # Get the structure from the parser
    struct = parser.get_structures()[0]  # Assuming there's only one structure in the CIF file
    struct_dict[name] = struct

#%%
# get dataframe
temps = [250, 275, 300, 325, 350, 375, 400]
df = pd.DataFrame({})
idx_zeo = -1
use_highsympath = False
for i in range(len(df_dft)):
    Data = dict()
    row = df_dft.iloc[i]
    name = row['name']
    pstruct = struct_dict[name]
    natms = len(pstruct)
    atoms = Atoms(list(map(lambda x: x.symbol, pstruct.species)) , # list of symbols got from pymatgen
            positions=pstruct.cart_coords.copy(),
            cell=pstruct.lattice.matrix.copy(), pbc=True) 
    if use_highsympath:
        out_dict = get_path(atoms)
        qpts = np.array(out_dict['qpts'])
        qticks = out_dict['qticks']
    else:
        qpts = np.zeros((1,3))
        qticks = ["$\Gamma$"]
    band = np.zeros((len(qpts), 3*natms))
    cv = np.array([1000*row[f'Cv_gravimetric_{"{:.{}f}".format(te, 2)}'] for te in temps])
    Data['id'] = name
    Data['structure'] = [atoms]
    Data['qpts'] = [qpts]
    Data['qticks'] = [qticks]
    Data['band_structure'] = [band]
    Data['cv'] = [cv]
    dfn = pd.DataFrame(data = Data)
    df = pd.concat([df, dfn], ignore_index = True)

#%%
data = df
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

# Generate Data Frame
df_tr = generate_dafaframe(model, tr_loader, loss_fn, device)
df_te = generate_dafaframe(model, te1_loader, loss_fn, device)
#%%
lent = len(df_te)
for i in range(lent):
    df_te['loss'][i] = np.random.randn()
#%%
# Plot the bands of TEST data
zpalette = ['#43AA8B' for _ in range(3)]
# plot_bands(df_te, header='./models/' + model_name, title='TEST', n=1, m=1, palette=zpalette, gtruth=False)
plot_bands_qlabels(df_te, header='./models/' + model_name + '_zeo', 
           title='TEST_moosavi_g', n=5, m=1, windowsize=(4,3), palette=zpalette, 
           gtruth=False, datadf=data)

# %%

# Plot the heat capacity of TEST data
zpalette = ['#43AA8B' for _ in range(3)]
T_lst = temps
plot_cv_gt(df_te, T_lst, data, header='./models/' + model_name + '_zeo', 
         fnum=100, title='TESTcv_moosavi_g', 
         n=6, m=1, lwidth=1.2, windowsize=(2.5, 2.2), palette=zpalette)

#%%
