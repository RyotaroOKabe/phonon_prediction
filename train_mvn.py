#%%
##########################

# Import

##########################
import torch
import time
import pickle as pkl
import os
from sklearn.model_selection import train_test_split
from utils.utils_load import load_band_structure_data   #, load_data
from utils.utils_data import generate_data_dict
from utils.utils_model import BandLoss, GraphNetwork_MVN, train
from utils.helpers import make_dict
torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
seedn=42
palette = ['#43AA8B', '#F8961E', '#F94144']
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

#%%
##########################

# Parameters 

##########################

file_name = os.path.basename(__file__)
print("File Name:", file_name)
run_name = time.strftime('%y%m%d-%H%M%S', time.localtime())
model_dir = './models'
data_dir = './data'
raw_dir = './data/phonon'
data_file = 'DFPT_band_structure.pkl'
tr_ratio = 0.9
batch_size = 1
k_fold = 5

max_iter = 200 
lmax = 2 
mul = 4 
nlayers = 2 
r_max = 4 
number_of_basis = 10 
radial_layers = 1
radial_neurons = 100 
node_dim = 118
node_embed_dim = 32 
input_dim = 118
input_embed_dim = 32 
irreps_out = '2x0e+2x1e+2x2e'
option = 'mvn'
descriptor = 'mass'
factor = 1000

loss_fn = BandLoss()
lr = 0.005
weight_decay = 0.05 
schedule_gamma = 0.96 

conf_dict = make_dict([run_name, model_dir, data_dir, raw_dir, data_file, tr_ratio, batch_size, k_fold, 
                       max_iter, lmax, mul, nlayers, r_max, number_of_basis, radial_layers, radial_neurons, 
                       node_dim, node_embed_dim, input_dim, input_embed_dim, irreps_out, option, 
                       loss_fn, lr, weight_decay, schedule_gamma, device, seedn])

for k, v in conf_dict.items():
    print(f'{k}: {v}')

#%%
##########################

# Load data from pkl or csv

##########################

download_data = True
if download_data:
    os.system(f'rm -r {data_dir}/9850858*')
    os.system(f'rm -r {data_dir}/phonon/')
    os.system(f'cd {data_dir}; wget --no-verbose https://figshare.com/ndownloader/files/9850858')
    os.system(f'cd {data_dir}; tar -xf 9850858')
    os.system(f'rm -r {data_dir}/9850858*')

#%%
data = load_band_structure_data(data_dir, raw_dir, data_file)
data_dict = generate_data_dict(data_dir=data_dir, run_name=run_name, data=data, r_max=r_max, descriptor=descriptor, option=option, factor=factor)

#%%
num = len(data_dict)
tr_nums = [int((num * tr_ratio)//k_fold)] * k_fold
te_num = num - sum(tr_nums)
idx_tr, idx_te = train_test_split(range(num), test_size=te_num, random_state=seedn)
with open(f'./data/idx_{run_name}_tr.txt', 'w') as f: 
    for idx in idx_tr: f.write(f"{idx}\n")
with open(f'./data/idx_{run_name}_te.txt', 'w') as f: 
    for idx in idx_te: f.write(f"{idx}\n")

#%%
data_set = torch.utils.data.Subset(list(data_dict.values()), range(len(data_dict)))
tr_set, te_set = torch.utils.data.Subset(data_set, idx_tr), torch.utils.data.Subset(data_set, idx_te)

#%%
##########################

# Set up the GNN model

##########################

model = GraphNetwork_MVN(mul, 
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
##########################

# Train the GNN model

##########################

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
    option=option,
    factor=factor,
    conf_dict=conf_dict)  #!)

