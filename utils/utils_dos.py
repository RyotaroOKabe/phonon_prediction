import numpy as np 
import matplotlib.pyplot as plt 
import glob
import pandas as pd
import json
import os
from utils.utils_plot import simname, loss_dist
palette = ['#43AA8B', '#F8961E', '#F94144']
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

def ddf(x):
    res = abs(x[0]-x[1])
    sig=1/(res+1e-05)
    val = np.zeros_like(x)
    val[(-(1/(2*sig))<=x) & (x<=(1/(2*sig)))] = 1
    return val

def dos_from_band(band, x):
    nq, nb = band.shape
    res = abs(x[0]-x[1])
    cum = np.zeros_like(x)
    for q in range(nq):
        for b in range(nb):
            f = band[q,b]
            delta = ddf(x-f)
            cum += delta
    integ = np.sum(res*cum)
    return cum*nb/integ

def dos_from_band_gauss(band, x, sigma):
    nq, nb = band.shape
    res = abs(x[0]-x[1])
    cum = np.zeros_like(x)
    for q in range(nq):
        for b in range(nb):
            f = band[q,b]
            delta = np.exp(-(x-f)**2/(2*sigma**2))*(1/(sigma*np.sqrt(2*np.pi)))
            cum += delta
    integ = np.sum(res*cum)
    return cum*nb/integ

# def DoS_dict(data_dir, raw_dir, data_file):
def DoS_data(raw_dir):
    # data_path = os.path.join(data_dir, data_file)
    df = pd.DataFrame({})
    for file_path in glob.glob(os.path.join(raw_dir, '*.json')):
        Data = dict()
        with open(file_path) as f:
            data = json.load(f)
        
        Data['id'] = data['metadata']['material_id']
        Data['dos'] = [np.array(data['phonon']['ph_dos'])]
        Data['freq'] = [np.array(data['phonon']['dos_frequencies'])]
        dfn = pd.DataFrame(data = Data)
        df = pd.concat([df, dfn], ignore_index = True)
        dfn = pd.DataFrame(data = Data)
        df = pd.concat([df, dfn], ignore_index = True)
    return df

def plot_dos(df_in, header, fnum=100, title=None, n=5, m=1, lwidth=0.5, windowsize=(3, 2), palette=palette, formula=True, gtruth=True):
    """_summary_

    Args:
        df_in (pandas.core.frame.DataFrame): _description_
        struct_data (pandas.core.frame.DataFrame): _description_
        header (str): _description_
        title (str, optional): _description_. Defaults to None.
        n (int, optional): _description_. Defaults to 5.
        m (int, optional): _description_. Defaults to 1.
        lwidth (float, optional): _description_. Defaults to 0.5.
        windowsize (tuple, optional): _description_. Defaults to (4, 1.8).
        palette (list, optional): _description_. Defaults to palette.
    """
    fontsize = 10
    i_mse = np.argsort(df_in['loss'])
    ds = df_in.iloc[i_mse][['id', 'name', 'real_band', 'output_test', 'loss']].reset_index(drop=True)
    tiles = (1/3, 2/3, 1.)
    num = len(tiles)
    xtiles = np.quantile(ds['loss'].values, tiles)
    iq = [0] + [np.argmin(np.abs(ds['loss'].values - k)) for k in xtiles]
    s = np.concatenate([np.sort(np.random.choice(np.arange(iq[k-1], iq[k], 1), size=m*n, replace=False)) for k in range(1,num+1)])
    fig, axs = plt.subplots(num*m,n+1, figsize=((n+1)*windowsize[1], num*m*windowsize[0]), gridspec_kw={'width_ratios': [0.7] + [1]*n})
    gs = axs[0,0].get_gridspec()
    # remove the underlying axes
    for ax in axs[:,0]:
        ax.remove()
    # add long axis
    axl = fig.add_subplot(gs[:,0])
    axl, cols=loss_dist(axl, ds, num, palette, xtiles, fontsize)

    cols = np.repeat(cols, n*m)
    axs = axs[:,1:].ravel()
    id_list = []
    for k in range(num*m*n):
        ax = axs[k]
        i = s[k]
        # mpid = ds.iloc[i]['id']
        # dos_data_idx = dos_data[dos_data['id']==mpid].iloc[0]
        # dos = dos_data_idx['dos']
        # freq = dos_data_idx['freq']
        rband = ds.iloc[i]['real_band']
        pband = ds.iloc[i]['output_test']
        fpmax, fpmin = np.max(pband), np.min(pband)
        frmax, frmin = np.max(rband), np.min(rband)
        fmax, fmin = max(frmax, fpmax), min(frmin, fpmin)
        frange = fmax-fmin
        r_ends = 0.15
        x_from = fmin-frange*r_ends
        x_to = fmax+frange*r_ends
        x = np.linspace(x_from, x_to, fnum)
        xres = np.abs(x[0]-x[1])
        sigma = xres*2
        pr_dos = dos_from_band(pband, x)
        re_dos = dos_from_band(rband, x)
        # pr_dos = dos_from_band_gauss(pband, x, sigma)
        # re_dos = dos_from_band_gauss(rband, x, sigma)
        # xpts = realb.shape[0]
        if gtruth:
            ax.plot(re_dos, x, color='k', linewidth=lwidth*0.8)
        ax.plot(pr_dos, x, color=cols[k], linewidth=lwidth)
        id_list.append(ds.iloc[i]['id'])
        if formula:
            ax.set_title(simname(ds.iloc[i]['name']).translate(sub), fontsize=fontsize*1.8)
        else:
            ax.set_title(ds.iloc[i]['id'].translate(sub), fontsize=fontsize*1.8)
        # min_y1, max_y1 = np.min(realb), np.max(realb)
        # min_y2, max_y2 = np.min(predb), np.max(predb)
        # min_y = min([min_y1, min_y2])
        # max_y = max([max_y1, max_y2])
        # width_y = max_y - min_y
        # ax.set_ylim(min_y-0.05*width_y, max_y+0.05*width_y)
        ax.tick_params(axis='y', which='major', labelsize=fontsize)
        ax.tick_params(axis='y', which='minor', labelsize=fontsize)
        ax.set_xticks([])
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.6)
    fig.patch.set_facecolor('white')
    if title: fig.suptitle(title, ha='center', y=1., fontsize=fontsize)
    fig.savefig(f"{header}_{title}_dos.png")
    fig.savefig(f"{header}_{title}_dos.pdf")
    print(id_list)