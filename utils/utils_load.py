import glob
import os
import json
import pandas as pd
import numpy as np
from ase import Atoms
from pymatgen.core.structure import Structure

def load_band_structure_data(data_dir, raw_dir, data_file):
    data_path = os.path.join(data_dir, data_file)
    if len(glob.glob(data_path)) == 0:
        df = pd.DataFrame({})
        for file_path in glob.glob(os.path.join(raw_dir, '*.json')):
            Data = dict()
            with open(file_path) as f:
                data = json.load(f)
            structure = Structure.from_str(data['metadata']['structure'], fmt = 'cif')
            print(structure.__dict__)
            atoms = Atoms(list(map(lambda x: x.symbol, structure.species)),
                            positions = structure.cart_coords.copy(),
                            cell = structure.lattice.matrix.copy(), 
                            pbc=True)
            Data['id'] = data['metadata']['material_id']
            Data['structure'] = [atoms]
            Data['qpts'] = [np.array(data['phonon']['qpts'])]
            Data['band_structure'] = [np.array(data['phonon']['ph_bandstructure'])]
            dfn = pd.DataFrame(data = Data)
            df = pd.concat([df, dfn], ignore_index = True)
        df.to_pickle(data_path)
        return df
    else:
        return pd.read_pickle(data_path)

# def load_data(filename):
#     df = pd.read_csv(filename, index_col=0)
#     try:
#         # structure provided as Atoms object
#         df['structure'] = df['structure'].apply(eval).progress_map(lambda x: Atoms.fromdict(x))
    
#     except:
#         # no structure provided
#         pass

#     else:
#         df['formula'] = df['structure'].map(lambda x: x.get_chemical_formula())
#         df['species'] = df['structure'].map(lambda x: list(set(x.get_chemical_symbols())))
#         species = sorted(list(set(df['species'].sum())))

#     df['qpts'] = df['qpts'].apply(eval).apply(np.array)
#     df['band_structure'] = df['band_structure'].apply(eval).apply(np.array)

#     return df