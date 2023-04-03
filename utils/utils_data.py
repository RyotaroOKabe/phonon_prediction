import torch
import glob
import os
import pandas as pd
import numpy as np
import pickle as pkl
from ase.neighborlist import neighbor_list
from torch_geometric.data import Data
from ase import Atom
import itertools
from copy import copy

default_dtype = torch.float64
torch.set_default_dtype(default_dtype)

def doub(array):
    return np.concatenate([array]*2, axis = 0)

def create_virtual_nodes(edge_src, edge_dst, edge_shift, edge_vec, edge_len):
    N = max(edge_src) + 1
    shift_lengths = np.max(edge_shift, axis = 0) - np.min(edge_shift, axis = 0) + 1
    ucs = UnitCellShift(shift_lengths)
    shift_dst = ucs.shift_indices[tuple(np.array(edge_shift).T.tolist())]
    return doub(edge_src), np.concatenate([edge_dst, N * (edge_dst + 1) + edge_src + N ** 2 * shift_dst], axis = 0), doub(edge_shift), doub(edge_vec), doub(edge_len), ucs

class UnitCellShift():
    def __init__(self, shift_lengths):
        self.shift_lengths = shift_lengths
        self.shift_arrays = [np.array(list(range(length))[int((length-1)/2):]+list(range(length))[:int((length-1)/2)]) - int((length-1)/2) for length in shift_lengths]
        self.shift_indices = np.array(list(range(np.prod(self.shift_lengths)))).reshape(self.shift_lengths)
        self.shift_reverse = np.meshgrid(*self.shift_arrays, indexing='ij')
        self.shift_reverse = np.concatenate([shift.reshape((-1, 1)) for shift in self.shift_reverse], axis = 1)

def get_node_deg(edge_dst, n):
    node_deg = np.zeros((n, 1), dtype = np.float64)
    for dst in edge_dst:
        node_deg[dst] += 1
    node_deg += node_deg == 0
    return torch.from_numpy(node_deg)

def get_node_attr(atomic_numbers, n):
    z = []
    for atomic_number in atomic_numbers:
        atomic = [0.0] * 118
        atomic[atomic_number - 1] = 1
        z.append(atomic)
    temp = []
    for atomic_number in atomic_numbers:
        atomic = [0.0] * 118
        atomic[atomic_number - 1] = 1
        temp += [atomic] * len(atomic_numbers)
    z += temp * n
    return torch.from_numpy(np.array(z, dtype = np.float64))

def get_input(atomic_numbers, n):
    x = []
    for atomic_number in atomic_numbers:
        atomic = [0.0] * 118
        atomic[atomic_number - 1] = Atom(atomic_number).mass
        x.append(atomic)
    temp = []
    for atomic_number in atomic_numbers:
        atomic = [0.0] * 118
        atomic[atomic_number - 1] = Atom(atomic_number).mass
        temp += [atomic] * len(atomic_numbers)
    x += temp * n
    return torch.from_numpy(np.array(x, dtype = np.float64))

def build_data(id, structure, qpts, band_structure, r_max):
    symbols = structure.symbols
    positions = torch.from_numpy(structure.positions.copy())
    numb = len(positions)
    lattice = torch.from_numpy(structure.cell.array.copy()).unsqueeze(0)
    edge_src, edge_dst, edge_shift, edge_vec, edge_len = neighbor_list("ijSDd", a = structure, cutoff = r_max, self_interaction = True)
    edge_src, edge_dst, edge_shift, edge_vec, edge_len, ucs = create_virtual_nodes(edge_src, edge_dst, edge_shift, edge_vec, edge_len)
    z = get_node_attr(structure.arrays['numbers'], len(ucs.shift_reverse))
    x = get_input(structure.arrays['numbers'], len(ucs.shift_reverse))
    node_deg = get_node_deg(edge_dst, len(x))
    y = torch.from_numpy(band_structure/1000).unsqueeze(0)
    data = Data(id = id,
                pos = positions,
                lattice = lattice,
                symbol = symbols,
                z = z,
                x = x,
                y = y,
                node_deg = node_deg,
                edge_index = torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim = 0),
                edge_shift = torch.tensor(edge_shift, dtype = torch.float64),
                edge_vec = torch.tensor(edge_vec, dtype = torch.float64),
                edge_len = torch.tensor(edge_len, dtype = torch.float64),
                qpts = torch.tensor(qpts, dtype = torch.float64),
                band_structure = torch.from_numpy(band_structure).unsqueeze(0),
                r_max = r_max,
                ucs = ucs,
                numb = numb)
    return data

def generate_band_structure_data_dict(data_dir, run_name, data, r_max):
    data_dict_path = os.path.join(data_dir, f'data_dict_{run_name}.pkl')
    if len(glob.glob(data_dict_path)) == 0: 
        data_dict = dict()
        ids = data['id']
        structures = data['structure']
        qptss = data['qpts']
        band_structures = data['band_structure']
        for id, structure, qpts, band_structure in zip(ids, structures, qptss, band_structures):
            # print(id)
            data_dict[id] = build_data(id, structure, qpts, band_structure, r_max)
        # pkl.dump(data_dict, open(data_dict_path, 'wb'))
    else:
        data_dict  = pkl.load(open(data_dict_path, 'rb'))
    return data_dict

#!
def append_diag_vn(struct, element="Fe"):
    # diagonal virtual nodes
    # option for atom choices. 
    """_summary_
    Args:
        struct (ase.atoms.Atoms): original ase.atoms.Atoms object
    Returns:
        ase.atoms.Atoms: Atoms affter appending additonal nodes
    """
    cell = struct.get_cell()
    num_sites = struct.get_positions().shape[0]
    total_len = 3*num_sites
    struct2 = struct.copy()
    for i in range(total_len):
        vec = i*(cell[0]  + cell[1] + cell[2])/total_len
        struct2.append(Atom(element, (vec[0], vec[1], vec[2])))
    return struct2

def create_virtual_nodes_vvn(structure0, vnelem, edge_src0, edge_dst0, edge_shift0):
    structure = append_diag_vn(structure0, element=vnelem)
    positions = torch.from_numpy(structure.get_positions().copy())
    positions0 = torch.from_numpy(structure0.get_positions().copy())
    numb = len(positions0)
    lattice = torch.from_numpy(structure.cell.array.copy()).unsqueeze(0)
    idx_real, idx_virt = range(numb), range(numb, 4*numb)
    rv_pairs = list(itertools.product(idx_real, idx_virt))
    vv_pairs = list(itertools.product(idx_virt, idx_virt))
    edge_src = copy(edge_src0)
    edge_dst = copy(edge_dst0)
    edge_shift = copy(edge_shift0)
    for i in range(len(rv_pairs)):
        edge_src = np.append(edge_src, np.array([rv_pairs[i][0]]))
        edge_dst = np.append(edge_dst, np.array([rv_pairs[i][1]]))
        edge_shift = np.concatenate((edge_shift, np.array([[0, 0, 0]])), axis=0)
    for j in range(len(vv_pairs)):
        edge_src = np.append(edge_src, np.array([vv_pairs[j][0]]))
        edge_dst = np.append(edge_dst, np.array([vv_pairs[j][1]]))
        edge_shift = np.concatenate((edge_shift, np.array([[0, 0, 0]])), axis=0)
    edge_batch = positions.new_zeros(positions.shape[0], dtype=torch.long)[torch.from_numpy(edge_src)]
    edge_vec = (positions[torch.from_numpy(edge_dst)]
                - positions[torch.from_numpy(edge_src)]
                + torch.einsum('ni,nij->nj', torch.tensor(edge_shift, dtype=default_dtype), lattice[edge_batch]))
    edge_len = np.around(edge_vec.norm(dim=1).numpy(), decimals=2)
    return edge_src, edge_dst, edge_shift, edge_vec, edge_len, structure

def get_node_attr_vvn(atomic_numbers):
    z = []
    for atomic_number in atomic_numbers:
        node_attr = [0.0] * 118
        node_attr[atomic_number - 1] = 1
        z.append(node_attr)
    return torch.from_numpy(np.array(z, dtype = np.float64))

def get_node_feature_vvn(atomic_numbers):
    x = []
    for atomic_number in atomic_numbers:
        node_feature = [0.0] * 118
        node_feature[atomic_number - 1] = Atom(atomic_number).mass
        x.append(node_feature)
    return torch.from_numpy(np.array(x, dtype = np.float64))

def build_data_vvn(id, structure, qpts, gphonon, r_max, vnelem='Fe'):
    symbols = list(structure.symbols).copy()
    positions = torch.from_numpy(structure.get_positions().copy())
    numb = len(positions)
    lattice = torch.from_numpy(structure.cell.array.copy()).unsqueeze(0)
    _edge_src, _edge_dst, _edge_shift, _, _ = neighbor_list("ijSDd", a = structure, cutoff = r_max, self_interaction = True)
    edge_src, edge_dst, edge_shift, edge_vec, edge_len, structure_vn = create_virtual_nodes_vvn(structure, vnelem, _edge_src, _edge_dst, _edge_shift)
    z = get_node_attr_vvn(structure_vn.arrays['numbers'])
    x =  get_node_feature_vvn(structure_vn.arrays['numbers'])
    node_deg = get_node_deg(edge_dst, len(x))
    y = torch.from_numpy(gphonon/1000).unsqueeze(0)
    data = Data(id = id,
                pos = positions,
                lattice = lattice,
                symbol = symbols,
                x = x,
                z = z,
                y = y,
                node_deg = node_deg,
                edge_index = torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim = 0),
                edge_shift = torch.tensor(edge_shift, dtype = torch.float64),
                edge_vec = torch.tensor(edge_vec, dtype = torch.float64),
                edge_len = torch.tensor(edge_len, dtype = torch.float64),
                qpts = torch.from_numpy(qpts).unsqueeze(0),
                gphonon = torch.from_numpy(gphonon).unsqueeze(0),
                r_max = r_max,
                # ucs = None,
                numb = numb)
    
    return data


def generate_gamma_data_dict(data_dir, run_name, data, r_max, vn_an=26):
    data_dict_path = os.path.join(data_dir, f'data_dict_{run_name}.pkl')
    vnelem = Atom(vn_an).symbol #!
    if len(glob.glob(data_dict_path)) == 0:
        data_dict = dict()
        ids = data['id']
        structures = data['structure']
        qptss = data['qpts']
        band_structures = data['band_structure']
        for id, structure, qpts, band_structure in zip(ids, structures, qptss, band_structures):
            # print(id)
            gi = np.argmin(np.abs(np.linalg.norm(qpts - np.array([0, 0, 0]), axis = 1)), axis = 0)
            data_dict[id] = build_data_vvn(id, structure, qpts[gi], band_structure[gi], r_max, vnelem)
        # pkl.dump(data_dict, open(data_dict_path, 'wb'))
    else:
        data_dict  = pkl.load(open(data_dict_path, 'rb'))
    return data_dict
