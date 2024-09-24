import numpy as np
import seekpath
Greek_letters = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta', 'Iota', 'Kappa', 'Lambda', 'Mu', 'Nu', 'Xi', 'Omicron', 'Pi', 'Rho', 'Sigma', 'Tau', 'Upsilon', 'Phi', 'Chi', 'Psi', 'Omega']

def get_structure(astruct):
    """
    Extracts structure data (cell, positions, atomic numbers) from an ASE atoms object.
    
    Args:
        astruct (ase.Atoms): ASE crystal structure.

    Returns:
        tuple: Cell, positions, and atomic numbers.
    """
    return astruct.cell.tolist(), astruct.get_positions().tolist(), astruct.numbers.tolist()

def symbol_latex(symbol):
    """
    Converts a chemical symbol to LaTeX format.
    
    Args:
        symbol (str): Chemical symbol.

    Returns:
        str: LaTeX formatted chemical symbol.
    """
    return f'\{symbol.capitalize()}' if symbol.split('_')[0].capitalize() in Greek_letters else symbol


def get_qpts(astruct, res=0.05, threshold=1e-07, symprec=1e-05, angle_tolerance=-1.0, nongamma=False):
    """
    Computes the high-symmetry k-point path and q-points for a given structure.
    
    Args:
        astruct (ase.Atoms): ASE crystal structure.
        res (float, optional): Resolution of q-points. Defaults to 0.05.
        threshold (float, optional): Symmetry threshold. Defaults to 1e-07.
        symprec (float, optional): Symmetry precision. Defaults to 1e-05.
        angle_tolerance (float, optional): Angle tolerance. Defaults to -1.0.
        nongamma (bool, optional): If True, show non-Gamma points. Defaults to False.

    Returns:
        dict: Contains symmetry points, paths, q-points, and labels for plotting.
    """
    # Get the structure data (cell, positions, atomic numbers) from the ASE object
    struct = get_structure(astruct)
    # Use Seekpath to calculate the high-symmetry path and coordinates
    getpath = seekpath.getpaths.get_path(
        struct, 
        with_time_reversal=True, 
        recipe='hpkot', 
        threshold=threshold, 
        symprec=symprec, 
        angle_tolerance=angle_tolerance
    )
    
    # Convert point coordinates and path points to LaTeX symbols (if needed) using symbol_latex function
    pcoords = {symbol_latex(k): v for k, v in getpath['point_coords'].items()}
    path = [(symbol_latex(start), symbol_latex(end)) for start, end in getpath['path']]
    # Initialize empty lists to store symmetry paths, point coordinates, and other necessary data
    path_set, point_list, sym_points, break_from = [], [], [], []
    p_end0 = None  # Keeps track of the end point from the previous loop iteration

    # Loop through each path segment (start, end) and construct path sets
    for i, (p_start, p_end) in enumerate(path):
        # length = len(path_set)
        
        # Check if the start point of the current path segment is the same as the previous end point
        if p_end0 == p_start: 
            path_set[-1].append(pcoords[p_end])  # Extend the previous path set
            sym_points.append(p_end)  # Add the endpoint to the symmetry points list
            point_list.append(pcoords[p_end])  # Add the coordinates of the endpoint
        else:
            # If a new path segment starts, create a new path set
            path_set.append([pcoords[p_start], pcoords[p_end]])  
            sym_points.extend([p_start, p_end])  # Add both the start and end points
            point_list.extend([pcoords[p_start], pcoords[p_end]])  # Add both the start and end coordinates
            break_from.append(i)  # Mark the point where the path breaks
        p_end0 = p_end  # Update the previous endpoint for the next iteration
    
    # Remove the first element from break_from (it only stores breaks after the first segment)
    break_from.remove(0)

    # Generate q-points between the symmetry points
    qpts, dist_list = [], []
    for i in range(len(point_list) - 1):
        start, end = np.array(point_list[i]), np.array(point_list[i + 1])
        dist = np.linalg.norm(end - start)  # Calculate the distance between two points
        dist_list.append(dist)  # Store the distance
        dig = int(dist // res)  # Determine the number of q-points between the two points based on resolution

        if i not in break_from:  # If not a break point, evenly distribute q-points along the path
            for j in range(dig):
                kvec = ((dig - j) * start + j * end) / dig  # Linearly interpolate between points
                qpts.append(kvec.tolist())  # Store q-point coordinates
        else:
            qpts.append(start.tolist())  # Store the start point if it's a break point
    
    qpts.append(end.tolist())  # Add the final endpoint to the q-points
    qpts = np.array(qpts)  # Convert q-points list to a NumPy array

    # Generate labels for the q-points based on their proximity to symmetry points
    qticks = []
    for qpt in qpts:
        high_symmetry = 0
        for label, vec in pcoords.items():
            if np.allclose(qpt, vec, atol=1e-4):  # Check if q-point is close to a high-symmetry point
                qticks.append(label)  # Assign the label of the high-symmetry point
                high_symmetry = 1
                break
        if high_symmetry == 0:
            qticks.append("")  # If not a high-symmetry point, leave label blank
    # print('path:', len(path), path)
    # print('point_list:', len(point_list), point_list)
    # print('pcoords:', len(pcoords), pcoords)
    # print('qpts:', len(qpts))
    # print('qticks:', len(qticks), qticks)


    return {
        'pcoords': pcoords,
        'path': path,
        'path_set': path_set,
        'point_list': point_list,
        'sym_points': sym_points,
        'break_from': break_from,
        'qpts': qpts,
        'dist_list': dist_list,
        'qticks': qticks
    }