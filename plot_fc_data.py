#%%
import torch
import time
import pickle as pkl
import os
from sklearn.model_selection import train_test_split
from utils.utils_load import load_band_structure_data   #, load_data
from utils.utils_data import generate_band_structure_data_dict
from utils.utils_model_nomass import BandLoss, GraphNetwork, train
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

folder = './data/230404-150605_FC_Test'
te_file = '230404-150605_te_set.pkl'
file_path = os.path.join(folder, te_file)

with open(file_path, 'rb') as f:
    data = pkl.load(f)







#%%



