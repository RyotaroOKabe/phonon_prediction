import numpy as np
import seekpath
def get_structure(astruct):
    cell = astruct.cell.tolist()
    positions = astruct.get_positions().tolist()
    numbers = astruct.numbers.tolist()
    return (cell, positions, numbers)

def get_path(astruct, res=0.05, threshold=1e-07, symprec=1e-05, angle_tolerance=-1.0):
    """
    If you do not have 

    Args:
        astruct (ase.ase.Atoms): structure
        res (float, optional): resolution of qpts. Defaults to 0.05.

    Returns:
        _type_: _description_
    """
    struct = get_structure(astruct)
    getpath = seekpath.getpaths.get_path(struct, with_time_reversal=True, recipe='hpkot', threshold=threshold, symprec=symprec, angle_tolerance=angle_tolerance)

    pcoords = getpath['point_coords']
    path = getpath['path']
    path_set = [] 
    point_list = []
    sym_points = []
    p_end0 = 0
    break_from  = []
    for i in range(len(path)):
        length = len(path_set)
        p_start = path[i][0]
        p_end = path[i][1]
        # print(p_start, pcoords[p_start])
        # print(p_end, pcoords[p_end])
        if p_end0 == p_start:
            path_set[length-1].append(pcoords[p_end])
            sym_points.append(p_end)
            point_list.append(pcoords[p_end])
        else:
            path_set.append([])
            path_set[length].append(pcoords[p_start])
            path_set[length].append(pcoords[p_end])
            sym_points.append(p_start)
            sym_points.append(p_end)
            point_list.append(pcoords[p_start])
            point_list.append(pcoords[p_end])
            break_from.append(i)
        p_end0 = p_end
    break_from.remove(0)

    qpts = []
    dist_list = []
    for i in range(len(point_list)-1):
        start = np.array(point_list[i])
        end = np.array(point_list[i+1])
        vec = end-start
        dist = np.linalg.norm(vec)
        dist_list.append(dist)
        dig = int(dist//res)
        if i not in break_from:
            for j in range(dig):
                kvec = ((dig-j)*start + j*end)/dig
                # print(kvec)
                qpts.append(kvec.tolist())
        else:
            # print(start)
            qpts.append(start.tolist())
    qpts.append(end.tolist())
    #!
    lenq = len(qpts)
    qlabels = list(pcoords.keys())
    qorders = {}
    for j in range(lenq):
        qpt = qpts[j]
        for qlabel in qlabels:
            qvec = pcoords[qlabel]
            if np.allclose(qpt, qvec, atol=1e-05):
                qorders[j] = qlabel
    q_idx = list(qorders.keys())
    qticks = []
    # greeks = ["GAMMA"]
    for x in range(lenq):
        if x in q_idx:
            qlabel = qorders[x]
            if qlabel=="GAMMA":
                qticks.append("$\Gamma$")
            else: 
                qticks.append("") 
                
        else: 
            qticks.append("") 
    #!

    out_dict = {'pcoords': pcoords, 'path': path, 'path_set': path_set, 
                'point_list': point_list, 'sym_points': sym_points, 'break_from': break_from,
                'qpts': qpts, 'dist_list':dist_list, 'qticks': qticks}
    return out_dict
    

# def get_band_from_qpts(qpts):
    
#     pass_str = str(qpts).replace('[', '').replace(']', '').replace(',', '')
#     os.system(f"phonopy -p phonopy.conf -c POSCAR-unitcell --qpoints {pass_str}")
#     my_file = open(f"qpoints.yaml", "r")
#     data = my_file.read()
#     data_into_list = data.replace(' ', '').split("\n")
#     my_file.close()

#     gph_list = []
#     for line in data_into_list:
#         if line.startswith('freq'):
#             gph_list.append(float(line[10:]))

#     p_data = np.array(gph_list).reshape((len(qpts), -1))
#     return p_data

# def update_df(df_in, mpid, res=0.01):
#     astruct = df_in[df_in['id']==mpid]['structure'].item()
#     out_dict = get_path(astruct, res)
#     qpts = out_dict['kvecs']
#     band = get_band_from_qpts(qpts)
#     return band, np.array(qpts)




