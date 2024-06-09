import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from scipy.stats import gaussian_kde
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit
from matplotlib.ticker import FormatStrFormatter
from ase import Atoms, Atom
from copy import copy
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
import matplotlib as mpl
import matplotlib.pyplot as plt
palette = ['#90BE6D', '#277DA1', '#F8961E', '#F94144']
datasets = ['train', 'valid', 'test']
colors = dict(zip(datasets, palette[:-1]))
cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', [palette[k] for k in [0,2,1]])

def simname(symbol):
    count = 0
    prev = ''
    name = ''
    for s in symbol:
        if s != prev:
            if name != '':
                name += str(count)
            name += s
            prev = s
            count = 1
        else:
            count += 1
    name += str(count)
    return name

def loss_dist(axl, ds, num, palette, xtiles, fontsize):
    """_summary_

    Args:
        axl (_type_): _description_
        ds (_type_): _description_
        num (_type_): _description_
        palette (_type_): _description_
        xtiles (_type_): _description_
        fontsize (_type_): _description_

    Returns:
        _type_: _description_
    """
    y_min, y_max = ds['loss'].min(), ds['loss'].max()
    y = np.linspace(y_min, y_max, 5000)
    kde = gaussian_kde(list(ds['loss']))
    p = kde.pdf(y)
    axl.plot(p, y, color='black')
    cols = palette[:num]
    cols_rev = copy(cols)
    cols_rev.reverse()
    qs =  list(xtiles)[::-1] + [0]
    for i in range(len(qs)-1):
        axl.fill_between([p.min(), p.max()], y1=[qs[i], qs[i]], y2=[qs[i+1], qs[i+1]], color=cols_rev[i], lw=0, alpha=0.5)
    axl.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    axl.invert_yaxis()
    axl.set_xticks([])
    axl.set_yscale('log')
    axl.tick_params(axis='y', which='major', labelsize=fontsize)
    axl.tick_params(axis='y', which='minor', labelsize=fontsize)
    axl.yaxis.set_minor_formatter(FormatStrFormatter("%.5f"))
    axl.yaxis.set_major_formatter(FormatStrFormatter("%.5f"))
    return axl, cols

def plot_bands_qlabels(df_in, header, title=None, 
                       n=5, m=1, lwidth=0.5, 
                       windowsize=(3, 2), 
                       palette=palette, formula=True, 
                       gtruth=True, datadf=None, qticks=True):
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
        realb = ds.iloc[i]['real_band']
        predb = ds.iloc[i]['output_test']
        xpts = realb.shape[0]
        if gtruth:
            ax.plot(range(xpts), realb, color='k', linewidth=lwidth*0.8)
        ax.plot(range(xpts), predb, color=cols[k], linewidth=lwidth)
        id_list.append(ds.iloc[i]['id'])
        if formula:
            ax.set_title(f"[${ds.iloc[i]['id']}$] {simname(ds.iloc[i]['name']).translate(sub)}", fontsize=fontsize*1.8)
        else:
            ax.set_title(f"${ds.iloc[i]['id']}$", fontsize=fontsize*1.8)
        min_y1, max_y1 = np.min(realb), np.max(realb)
        min_y2, max_y2 = np.min(predb), np.max(predb)
        min_y = min([min_y1, min_y2])
        max_y = max([max_y1, max_y2])
        width_y = max_y - min_y
        ax.set_ylim(min_y-0.05*width_y, max_y+0.05*width_y)
        labelsize = fontsize*1.5
        ax.tick_params(axis='y', which='major', labelsize=labelsize)
        ax.tick_params(axis='y', which='minor', labelsize=labelsize)
        # ax.set_xticks([])
        if qticks:
            qlabels = datadf[datadf['id']==ds.iloc[i]['id']]['qticks'].item()
            # print(qlabels)
            ax.set_xticks(range(xpts), qlabels, fontsize=labelsize)
        ax.tick_params(bottom = False)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.6)
    fig.patch.set_facecolor('white')
    if title: fig.suptitle(title, ha='center', y=1., fontsize=fontsize)
    fig.savefig(f"{header}_{title}_bands.png")
    fig.savefig(f"{header}_{title}_bands.pdf")
    print(id_list)