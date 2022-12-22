import torch
import glob
import os
import pandas as pd
import numpy as np
import pickle as pkl
from ase.neighborlist import neighbor_list
from torch_geometric.data import Data
from ase import Atom

torch.set_default_dtype(torch.float64)

# def doub(array):
#     return np.concatenate([array]*2, axis = 0)

def create_virtual_nodes_vvn(edge_src, edge_dst, edge_shift, edge_vec, edge_len):
    # diagonal virtual nodes
    # option for atom choices. 
    pass

# def get_node_attr(atomic_numbers, n):
#     z = []
#     for atomic_number in atomic_numbers:
#         atomic = [0.0] * 118
#         atomic[atomic_number - 1] = 1
#         z.append(atomic)
#     temp = []
#     for atomic_number in atomic_numbers:
#         atomic = [0.0] * 118
#         atomic[atomic_number - 1] = 1
#         temp += [atomic] * len(atomic_numbers)
#     z += temp * n
#     return torch.from_numpy(np.array(z, dtype = np.float64))

# def get_input(atomic_numbers, n):
#     x = []
#     for atomic_number in atomic_numbers:
#         atomic = [0.0] * 118
#         atomic[atomic_number - 1] = Atom(atomic_number).mass
#         x.append(atomic)
#     temp = []
#     for atomic_number in atomic_numbers:
#         atomic = [0.0] * 118
#         atomic[atomic_number - 1] = Atom(atomic_number).mass
#         temp += [atomic] * len(atomic_numbers)
#     x += temp * n
#     return torch.from_numpy(np.array(x, dtype = np.float64))

def build_data_vvn(id, structure, qpts, band_structure, r_max):
    # use the create_virtual_nodes
    pass


def generate_gamma_data_dict(data_dir, run_name, data, r_max):
    data_dict_path = os.path.join(data_dir, f'data_dict_{run_name}.pkl')
    if len(glob.glob(data_dict_path)) == 0: 
        data_dict = dict()
        ids = data['id']
        structures = data['structure']
        qptss = data['qpts']
        band_structures = data['band_structure']
        for id, structure, qpts, band_structure in zip(ids, structures, qptss, band_structures):
            print(id)
            gi = np.argmin(np.abs(np.linalg.norm(qpts - np.array([0, 0, 0]), axis = 1)), axis = 0)
            data_dict[id] = build_data_vvn(id, structure, qpts[gi], band_structure[gi])
        # pkl.dump(data_dict, open(data_dict_path, 'wb'))
    else:
        data_dict  = pkl.load(open(data_dict_path, 'rb'))
    return data_dict