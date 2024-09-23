import torch
import glob
import os
import numpy as np
import pickle as pkl
from ase.neighborlist import neighbor_list
from torch_geometric.data import Data
from ase import Atom
import mendeleev as md
import itertools
from tqdm import tqdm
# from copy import copy

default_dtype = torch.float64
torch.set_default_dtype(default_dtype)

class MD():
    """
    Class to store atomic properties like radius, electronegativity, ionization energy, and dipole polarizability.
    These properties are retrieved from the mendeleev library for all elements.
    """
    def __init__(self):
        self.radius, self.pauling, self.ie, self.dip = {}, {}, {}, {}
        for atomic_number in range(1, 119):
            ele = md.element(atomic_number)
            # print(str(ele))
            self.radius[atomic_number] = ele.atomic_radius
            self.pauling[atomic_number] = ele.en_pauling
            ie_dict = ele.ionenergies
            self.ie[atomic_number] = ie_dict[min(list(ie_dict.keys()))] if len(ie_dict)>0 else 0
            self.dip[atomic_number] = ele.dipole_polarizability

md_class = MD()

def pkl_load(filename):
    """
    Load a pickle file.
    Args:
        filename (str): Path to the pickle file.
    Returns:
        object: The loaded object from the pickle file.
    """
    with open(filename, 'rb') as file:
        loaded_dict = pkl.load(file)
    return loaded_dict

def get_lattice_parameters(data):
    a = []
    len_data = len(data)
    for i in range(len_data):
        d = data.iloc[i]
        a.append(d.structure.cell.cellpar()[:3])
    return np.stack(a)

def doub(array):
    """
    Concatenate an array with itself along axis 0.
    Args:
        array (np.ndarray): Input array.
    Returns:
        np.ndarray: Concatenated array.
    """
    return np.concatenate([array]*2, axis = 0)

class UnitCellShift():
    """
    Class to handle unit cell shifts based on provided shift lengths.
    """
    def __init__(self, shift_lengths):
        self.shift_lengths = shift_lengths
        self.shift_arrays = [np.array(list(range(length))[int((length-1)/2):]+list(range(length))[:int((length-1)/2)]) - int((length-1)/2) for length in shift_lengths]
        self.shift_indices = np.array(list(range(np.prod(self.shift_lengths)))).reshape(self.shift_lengths)
        self.shift_reverse = np.meshgrid(*self.shift_arrays, indexing='ij')
        self.shift_reverse = np.concatenate([shift.reshape((-1, 1)) for shift in self.shift_reverse], axis = 1)


def create_virtual_nodes_kmvn(edge_src, edge_dst, edge_shift, edge_vec, edge_len):
    """
    Create virtual nodes for the 'kmvn' method.
    Args:
        edge_src (np.ndarray): Source edge indices.
        edge_dst (np.ndarray): Destination edge indices.
        edge_shift (np.ndarray): Edge shifts.
        edge_vec (np.ndarray): Edge vectors.
        edge_len (np.ndarray): Edge lengths.
    Returns:
        tuple: Updated edges, shifts, vectors, lengths, and unit cell shifts.
    """
    N = max(edge_src) + 1
    shift_lengths = np.max(edge_shift, axis = 0) - np.min(edge_shift, axis = 0) + 1
    ucs = UnitCellShift(shift_lengths)
    shift_dst = ucs.shift_indices[tuple(np.array(edge_shift).T.tolist())]
    return doub(edge_src), np.concatenate([edge_dst, N * (edge_dst + 1) + edge_src + N ** 2 * shift_dst], axis = 0), doub(edge_shift), doub(edge_vec), doub(edge_len), ucs

def create_virtual_node_mvn(edge_src, edge_dst, edge_shift, edge_vec, edge_len):
    """
    Create virtual nodes for the 'mvn' method.
    Args:
        edge_src (np.ndarray): Source edge indices.
        edge_dst (np.ndarray): Destination edge indices.
        edge_shift (np.ndarray): Edge shifts.
        edge_vec (np.ndarray): Edge vectors.
        edge_len (np.ndarray): Edge lengths.
    Returns:
        tuple: Updated edges, shifts, vectors, lengths.
    """
    N = max(edge_src) + 1
    return doub(edge_src), np.concatenate([edge_dst, N * (edge_dst + 1) + edge_src], axis = 0), doub(edge_shift), doub(edge_vec), doub(edge_len)

def get_node_deg(edge_dst, n):
    """
    Compute node degrees from the destination edges.
    Args:
        edge_dst (np.ndarray): Destination edges.
        n (int): Number of nodes.
    Returns:
        torch.Tensor: Node degrees.
    """
    node_deg = np.zeros((n, 1), dtype = np.float64)
    for dst in edge_dst:
        node_deg[dst] += 1
    node_deg += node_deg == 0
    return torch.from_numpy(node_deg)


# def get_node_attr(atomic_numbers, n): #TODO: delete later


def atom_feature(atomic_number: int, descriptor):
    """
    Get atomic features based on the descriptor.
    Args:
        atomic_number (int): Atomic number of the element.
        descriptor (str): Type of descriptor. Can be 'mass', 'number', 'radius', 'en', 'ie', 'dp', 'non'.
    Returns:
        float: The atomic feature value.
    """
    if descriptor=='mass':  # Atomic Mass (amu)
        feature = Atom(atomic_number).mass
    elif descriptor=='number':  # atomic number
        feature = atomic_number
    else:
        # ele = md.element(atomic_number) # use mendeleev
        if descriptor=='radius':    # Atomic Radius (pm)
            feature = md_class.radius[atomic_number]
        elif descriptor=='en': # Electronegativity (Pauling)
            feature = md_class.pauling[atomic_number]
        elif descriptor=='ie':  # Ionization Energy (eV)
            feature = md_class.ie[atomic_number]
        elif descriptor=='dp':  # Dipole Polarizability (Å^3)
            feature = md_class.dip[atomic_number]
        else:   # no feature
            feature = 1
    return feature


# def get_input(atomic_numbers, n, descriptor='mass'):  #TODO delete later


def create_node_input(atomic_numbers, n=None, descriptor='mass', option='kmvn'):
    """
    Create node input features for a list of atomic numbers.
    Args:
        atomic_numbers (list): List of atomic numbers.
        n (int, optional): Scaling factor for repetition. Defaults to None.
        descriptor (str, optional): Descriptor for the node features. Defaults to 'mass'.
        option (str, optional): Option for repeating vectors. Defaults to 'kmvn'.
    Returns:
        torch.Tensor: Tensor of node input features.
    """
    x = []
    temp = []
    for atomic_number in atomic_numbers:
        atomic = [0.0] * 118         
        atomic[atomic_number - 1] = atom_feature(int(atomic_number), descriptor)
        x.append(atomic)
        if option in ['kmvn', 'mvn']:
            temp += [atomic] * len(atomic_numbers)
    if n is not None:
        x += temp * n
    return torch.from_numpy(np.array(x, dtype = np.float64))


def append_diag_vvn(structure, element="Fe"):
    """
    Append diagonal virtual nodes to the structure.
    Args:
        structure (ase.atoms.Atoms): Original atomic structure.
        element (str, optional): Element type for virtual nodes. Defaults to "Fe".
    Returns:
        ase.atoms.Atoms: Structure with added virtual nodes.
    """
    cell = structure.get_cell()
    num_sites = structure.get_positions().shape[0]
    total_len_vvn = 3*num_sites
    structure_vvn = structure.copy()
    for i in range(total_len_vvn):
        vec = i*(cell[0]  + cell[1] + cell[2])/total_len_vvn
        structure_vvn.append(Atom(element, (vec[0], vec[1], vec[2])))
    return structure_vvn

def create_virtual_nodes_vvn(structure, edge_src, edge_dst, edge_shift, vn_elem='Fe'):
    """
    Create virtual nodes for the 'vvn' method.
    Args:
        structure (ase.atoms.Atoms): Original atomic structure.
        edge_src (np.ndarray): Source edges.
        edge_dst (np.ndarray): Destination edges.
        edge_shift (np.ndarray): Edge shifts.
        vn_elem (str, optional): Element type for virtual nodes. Defaults to 'Fe'.
    Returns:
        tuple: Updated edges, shifts, vectors, lengths, and virtual node structure.
    """
    structure_vvn = append_diag_vvn(structure, element=vn_elem)
    positions_vvn = torch.from_numpy(structure_vvn.get_positions().copy())
    positions = torch.from_numpy(structure.get_positions().copy())
    numb = len(positions)
    lattice_vvn = torch.from_numpy(structure_vvn.cell.array.copy()).unsqueeze(0)
    idx_real, idx_virt = range(numb), range(numb, 4*numb)
    rv_pairs = list(itertools.product(idx_real, idx_virt))
    vv_pairs = list(itertools.product(idx_virt, idx_virt))
    # edge_src_vvn = copy(edge_src)
    # edge_dst_vvn = copy(edge_dst)
    # edge_shift_vvn = copy(edge_shift)
    
    edge_src_vvn = np.append(edge_src, [pair[0] for pair in rv_pairs + vv_pairs])
    edge_dst_vvn = np.append(edge_dst, [pair[1] for pair in rv_pairs + vv_pairs])
    edge_shift_vvn = np.concatenate([edge_shift, np.zeros((len(rv_pairs) + len(vv_pairs), 3))])
    
    # for i in range(len(rv_pairs)):
    #     edge_src_vvn = np.append(edge_src_vvn, np.array([rv_pairs[i][0]]))
    #     edge_dst_vvn = np.append(edge_dst_vvn, np.array([rv_pairs[i][1]]))
    #     edge_shift_vvn = np.concatenate((edge_shift_vvn, np.array([[0, 0, 0]])), axis=0)
    # for j in range(len(vv_pairs)):
    #     edge_src_vvn = np.append(edge_src_vvn, np.array([vv_pairs[j][0]]))
    #     edge_dst_vvn = np.append(edge_dst_vvn, np.array([vv_pairs[j][1]]))
    #     edge_shift_vvn = np.concatenate((edge_shift_vvn, np.array([[0, 0, 0]])), axis=0)
    
    # edge_batch = positions_vvn.new_zeros(positions_vvn.shape[0], dtype=torch.long)[torch.from_numpy(edge_src_vvn)]   #TODO: check the code line
    # edge_vec_vvn = (positions_vvn[torch.from_numpy(edge_dst_vvn)]
    #             - positions_vvn[torch.from_numpy(edge_src_vvn)]
    #             + torch.einsum('ni,nij->nj', torch.tensor(edge_shift_vvn, dtype=default_dtype), lattice_vvn[edge_batch]))   #TODO: check the code line
    edge_vec_vvn = (positions_vvn[torch.from_numpy(edge_dst_vvn)]
                    - positions_vvn[torch.from_numpy(edge_src_vvn)]
                    + torch.einsum('ni,nij->nj', torch.tensor(edge_shift_vvn, dtype=default_dtype), lattice_vvn))
    edge_len_vvn = np.around(edge_vec_vvn.norm(dim=1).numpy(), decimals=2)
    return edge_src_vvn, edge_dst_vvn, edge_shift_vvn, edge_vec_vvn, edge_len_vvn, structure_vvn

# def get_node_attr_vvn(atomic_numbers):    #TODO: delete later
# def get_node_feature_vvn(atomic_numbers, descriptor='mass'):  #TODO: delete later


def build_data(mpid, structure, real, r_max, qpts, descriptor='mass', option='kmvn', factor=1000, vn_elem='Fe', **kwargs):
    """
    Build data object for graph-based learning models.
    Args:
        mpid (str): Material project ID.
        structure (ase.atoms.Atoms): Atomic structure.
        real (np.ndarray): Real values (e.g., band structure).
        r_max (float): Cutoff radius for neighbor list.
        qpts (np.ndarray): q-points.
        descriptor (str, optional): Descriptor for node features. Defaults to 'mass'.
        option (str, optional): Option for virtual node creation. Defaults to 'kmvn'.
        factor (int, optional): Scaling factor for real values. Defaults to 1000.
        vn_elem (str, optional): Element type for virtual nodes. only for VVN. Defaults to 'Fe'.
    Returns:
        torch_geometric.data.Data: Data object for PyTorch Geometric.
    """
    symbols = structure.symbols
    positions = torch.from_numpy(structure.positions.copy())
    numb = len(positions)
    lattice = torch.from_numpy(structure.cell.array.copy()).unsqueeze(0)
    edge_src, edge_dst, edge_shift, edge_vec, edge_len = neighbor_list("ijSDd", a = structure, cutoff = r_max, self_interaction = True)
    ucs = None
    if option == 'vvn':
        edge_src, edge_dst, edge_shift, edge_vec, edge_len, structure = create_virtual_nodes_vvn(structure, edge_src, edge_dst, edge_shift, vn_elem)
        len_usc = None
    else: 
        if option == 'kmvn':
            edge_src, edge_dst, edge_shift, edge_vec, edge_len, ucs = create_virtual_nodes_kmvn(edge_src, edge_dst, edge_shift, edge_vec, edge_len)
            len_usc = len(ucs.shift_reverse)
        elif option == 'mvn':
            edge_src, edge_dst, edge_shift, edge_vec, edge_len = create_virtual_node_mvn(edge_src, edge_dst, edge_shift, edge_vec, edge_len)
            len_usc = 1

    z = create_node_input(structure.arrays['numbers'], len_usc, descriptor='one_hot', option=option)   # node attribute
    x = create_node_input(structure.arrays['numbers'], len_usc, descriptor=descriptor, option=option)  # init node feature
    y = torch.from_numpy(real/factor).unsqueeze(0)
    node_deg = get_node_deg(edge_dst, len(x))
    
    data_dict = {'id': mpid, 'pos': positions, 'lattice': lattice, 'symbol': symbols, 'z': z, 'x': x, 'y': y, 'node_deg': node_deg, 
                 'edge_index': torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim = 0),
                 'edge_shift': torch.tensor(edge_shift, dtype = torch.float64), 'edge_vec': torch.tensor(edge_vec, dtype = torch.float64),
                 'edge_len': torch.tensor(edge_len, dtype = torch.float64), 'qpts': torch.tensor(qpts, dtype = torch.float64), 
                 'r_max': r_max, 'numb': numb, 'ucs': ucs}
    
    data = Data(**data_dict)
    return data

def generate_data_dict(data_dir, run_name, data, r_max, descriptor='mass', option='kmvn', factor=1000, vn_elem='Fe', **kwargs):
    """
    Generate a dictionary of band structure data.
    Args:
        data_dir (str): Directory to store the data.
        run_name (str): Name of the run.
        data (dict): Dictionary containing data to process.
        r_max (float): Cutoff radius for neighbor list.
        descriptor (str, optional): Descriptor for node features. Defaults to 'mass'.
        option (str, optional): Option for virtual node creation. Defaults to 'kmvn'.
        factor (int, optional): Scaling factor for real values. Defaults to 1000.
        vn_elem (str): Element type for virtual nodes (only for VVN). Defaults to 'Fe'.
    Returns:
        dict: Data dictionary containing band structure information.
    """
    data_dict_path = os.path.join(data_dir, f'data_dict_{run_name}.pkl')
    if len(glob.glob(data_dict_path)) == 0: 
        data_dict = dict()
        ids = data['id']
        structures = data['structure']
        qptss = data['qpts']
        reals = data['real_band']  # data['band_structure']
        for id, structure, real, qpts in tqdm(zip(ids, structures, reals, qptss), total = len(ids)):
            # print(id)
            if option in ['vvn', 'mvn']:
                gamma_idx = np.argmin(np.abs(np.linalg.norm(qpts - np.array([0, 0, 0]), axis = 1)), axis = 0)
                real = real[gamma_idx]
                qpts = qpts[gamma_idx]
            data_dict[id] = build_data(id, structure, real, r_max, qpts, descriptor, option, factor, vn_elem, **kwargs)
        # pkl.dump(data_dict, open(data_dict_path, 'wb'))
    else:
        data_dict  = pkl.load(open(data_dict_path, 'rb'))
    return data_dict
