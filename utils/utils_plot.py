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
import sklearn
palette = ['#90BE6D', '#277DA1', '#F8961E', '#F94144']
datasets = ['train', 'valid', 'test']
colors = dict(zip(datasets, palette[:-1]))
cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', [palette[k] for k in [0,2,1]])
import sklearn

def loss_plot(model_file, device, fig_file):
    history = torch.load(model_file + '.torch', map_location = device)['history']
    steps = [d['step'] + 1 for d in history]
    loss_train = [d['train']['loss'] for d in history]
    loss_valid = [d['valid']['loss'] for d in history]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(steps, loss_train, 'o-', label='Training')
    ax.plot(steps, loss_valid, 'o-', label='Validation')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.legend()
    fig.savefig(fig_file  + '_loss_train_valid.png')
    plt.close()

def loss_test_plot(model, device, fig_file, dataloader, loss_fn, option='kmvn'):
    loss_test = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for d in dataloader:
            d.to(device)
            if option in ['kmvn', 'mvn']:   #!
                Hs, shifts = model(d)
                output = get_spectra(Hs, shifts, d.qpts)
            else:   #!
                output = model(d)
            loss = loss_fn(output, d.y).cpu()
            loss_test.append(loss)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(np.array(loss_test), label = 'testing loss: ' + str(np.mean(loss_test)))
    ax.set_ylabel('loss')
    ax.legend()
    fig.savefig(fig_file + '_loss_test.png')
    plt.close()

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

def get_spectra(Hs, shifts, qpts):
    H = torch.sum(torch.mul(Hs.unsqueeze(1), torch.exp(2j*math.pi*torch.matmul(shifts, qpts.type(torch.complex128).t())).unsqueeze(-1).unsqueeze(-1)), dim = 0)
    eigvals = torch.linalg.eigvals(H)
    abx = torch.abs(eigvals)
    try:
        epsilon = torch.min(abx[abx > 0])/100
    except:
        epsilon = 1E-8
    eigvals = torch.sqrt(eigvals + epsilon)
    return torch.sort(torch.real(eigvals))[0]

def generate_dafaframe(model, dataloader, loss_fn, device, option='kmvn'):
    with torch.no_grad():
        df = pd.DataFrame(columns=['id', 'name', 'loss', 'real_band', 'output_test'])
        for d in dataloader:
            d.to(device)
            if len(d.pos) > 60:
                continue
            if option in ['kmvn', 'mvn']:
                Hs, shifts = model(d)
                output = get_spectra(Hs, shifts, d.qpts)
            else:
                output = model(d)
            loss = loss_fn(output, d.y).cpu()
            real = d.y.cpu().numpy()*1000
            pred = output.cpu().numpy()*1000
            rrr = {'id': d.id, 'name': d.symbol, 'loss': loss.item(), 'real_band': list(real), 'output_test': list(np.array([pred]))}
            df0 = pd.DataFrame(data = rrr)
            df = pd.concat([df, df0], ignore_index=True)
    return df

def plot_bands(df_in, header, 
               title=None, 
               n=5, m=1, lwidth=0.5, 
               windowsize=(3, 2), palette=palette, 
               formula=True, gtruth=True):
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
            ax.set_title(simname(ds.iloc[i]['name']).translate(sub), fontsize=fontsize*1.8)
        else:
            ax.set_title(ds.iloc[i]['id'].translate(sub), fontsize=fontsize*1.8)
        min_y1, max_y1 = np.min(realb), np.max(realb)
        min_y2, max_y2 = np.min(predb), np.max(predb)
        min_y = min([min_y1, min_y2])
        max_y = max([max_y1, max_y2])
        width_y = max_y - min_y
        ax.set_ylim(min_y-0.05*width_y, max_y+0.05*width_y)
        ax.tick_params(axis='y', which='major', labelsize=fontsize)
        ax.tick_params(axis='y', which='minor', labelsize=fontsize)
        ax.set_xticks([])
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.6)
    fig.patch.set_facecolor('white')
    if title: fig.suptitle(title, ha='center', y=1., fontsize=fontsize)
    fig.savefig(f"{header}_{title}_bands.png")
    fig.savefig(f"{header}_{title}_bands.pdf")
    print(id_list)

def plot_gphonons(df_in, header, title=None, n=5, m=1, lwidth=0.5, windowsize=(4, 2), palette=palette, formula=True):
    # function to plot gamma phonons of the tertiles. 
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
        realg = ds.iloc[i]['real_band'].reshape(-1)
        predg = ds.iloc[i]['output_test'].reshape(-1)
        for j in range(ds.iloc[i]['real_band'].shape[0]):
            ax.hlines(realg[j], 0.2, 1.2,color='k', linewidth=lwidth*0.6)
            ax.hlines(predg[j], 1.5, 2.5,color=cols[k], linewidth=lwidth)
        if formula:
            ax.set_title(simname(ds.iloc[i]['name']).translate(sub), fontsize=fontsize*1.8)
        else:
            ax.set_title(ds.iloc[i]['id'].translate(sub), fontsize=fontsize*1.8)
        id_list.append(ds.iloc[i]['id'])
        min_x, max_x = 0.2, 2.5
        min_y1, max_y1, min_y2, max_y2 = np.min(realg), np.max(realg), np.min(predg), np.max(predg)
        min_y, max_y = min([min_y1, min_y2]), max([max_y1, max_y2])
        width_x, width_y = max_x - min_x, max_y - min_y
        ax.set_xlim(min_x-0.05*width_x, max_x+0.05*width_x)
        ax.set_ylim(min_y-0.05*width_y, max_y+0.05*width_y)
        ax.tick_params(axis='y', which='major', labelsize=12)
        ax.tick_params(axis='y', which='minor', labelsize=10)
        ax.set_xticks([])
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.6)
    fig.patch.set_facecolor('white')
    if title: fig.suptitle(title, ha='center', y=1., fontsize=fontsize)
    fig.savefig(f"{header}_{title}_gphonons.png")
    fig.savefig(f"{header}_{title}_gphonons.pdf")
    print(id_list)


def compare_corr(df1, df2, color1, color2, header, size=5, r2=False):
    """_summary_

    Args:
        df1 (pandas.core.frame.DataFrame): Dataframe1
        df2 (pandas.core.frame.DataFrame): Dataframe2
        color1 (str): Hex color for df1's correlation plot
        color2 (str): Hex color for df1's correlation plot
        header (str): header as the save dir and file name
        size (int, optional): Size of the data points. Defaults to 5.
    """
    re_out1 = np.concatenate([df1.iloc[i]['real_band'].flatten() for i in range(len(df1))])
    pr_out1 = np.concatenate([df1.iloc[i]['output_test'].flatten() for i in range(len(df1))])
    re_out2 = np.concatenate([df2.iloc[i]['real_band'].flatten() for i in range(len(df2))])
    pr_out2 = np.concatenate([df2.iloc[i]['output_test'].flatten() for i in range(len(df2))])
    min_x1, max_x1 = np.min(re_out1), np.max(re_out1)
    min_y1, max_y1 = np.min(pr_out1), np.max(pr_out1)
    min_x2, max_x2 = np.min(re_out2), np.max(re_out2)
    min_y2, max_y2 = np.min(pr_out2), np.max(pr_out2)
    minimum = min(min_x1, min_x2, min_y1, min_y2)
    maximum = max(max_x1, max_x2,max_y1, max_y2)
    width = maximum - minimum
    fig, ax = plt.subplots(1,1, figsize=(6, 6))
    ax.plot([minimum-0.01*width, maximum+0.01*width], [minimum-0.01*width, maximum+0.01*width], color='k')
    ax.set_xlim(minimum-0.01*width, maximum+0.01*width)
    ax.set_ylim(minimum-0.01*width, maximum+0.01*width)

    ax.scatter(re_out1, pr_out1, s=size, marker='.', color=color1)
    ax.scatter(re_out2, pr_out2, s=size, marker='.', color=color2)

    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.set_xlabel('True $\omega$ [$cm^{-1}$]', fontsize=16)
    ax.set_ylabel('Predicted $\omega$ [$cm^{-1}$]', fontsize=16)
    if r2:
        R2_1 = sklearn.metrics.r2_score(y_true=re_out1, y_pred=pr_out1)
        R2_2 = sklearn.metrics.r2_score(y_true=re_out2, y_pred=pr_out2)
        ax.set_title(f"$R^2 [1] {R2_1} [2] {R2_2}$", fontsize=16)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.6)
    fig.patch.set_facecolor('white')
    fig.savefig(f"{header}scatter_compare.png")
    fig.savefig(f"{header}scatter_compare.pdf")
    fig.savefig(f"{header}scatter_compare.svg")

def compare_loss(df1, df2, color1, color2, header, labels=('Model1', 'Model2'), lw=3):
    """_summary_

    Args:
        df1 (pandas.core.frame.DataFrame): Dataframe1
        df2 (pandas.core.frame.DataFrame): Dataframe2
        color1 (str): Hex color for df1's loss plot
        color2 (str): Hex color for df1's loss plot
        header (str): header as the save dir and file name
        labels (tuple, optional): Legends. Defaults to ('Model1', 'Model2').
        lw (int, optional): Line width. Defaults to 3.
    """
    fig, ax = plt.subplots(1,1, figsize=(6, 6))
    min_x1, max_x1 = df1['loss'].min(), df1['loss'].max()
    x1 = np.linspace(min_x1, max_x1, 500)
    kde1 = gaussian_kde(df1['loss'])
    p1 = kde1.pdf(x1)
    ax.plot(x1, p1, color=color1, label=labels[0], lw=lw)
    min_x2, max_x2 = df2['loss'].min(), df2['loss'].max()
    x2 = np.linspace(min_x2, max_x2, 500)
    kde2 = gaussian_kde(df2['loss'])
    p2 = kde2.pdf(x2)
    ax.plot(x2, p2, color=color2, label=labels[1], lw=lw)
    ax.set_xscale('log')
    min_y1, max_y1 = np.min(p1), np.max(p1)
    min_y2, max_y2 = np.min(p2), np.max(p2)
    min_x, max_x = min([min_x1, min_x2]), max([max_x1, max_x2])
    min_y, max_y = min([min_y1, min_y2]), max([max_y1, max_y2])
    width_x = max_x - min_x
    width_y = max_y - min_y
    ax.set_xlim(min_x-0.08*width_x, max_x+0.08*width_x)
    ax.set_ylim(min_y-0.08*width_y, max_y+0.08*width_y)
    ax.tick_params(axis='x', which='major', labelsize=15)
    ax.tick_params(axis='x', which='minor', labelsize=13)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.legend()
    ax.set_xlabel('Loss', fontsize=16)
    ax.set_ylabel('Probability density', fontsize=16)
    # fig.tight_layout()
    # fig.subplots_adjust(hspace=0.6)
    fig.patch.set_facecolor('white')
    if labels: fig.suptitle(f'{labels[0]}_{labels[1]}', ha='center', y=1., fontsize=16)
    fig.savefig(f"{header}_{labels[0]}_{labels[1]}_loss_compare.png")
    fig.savefig(f"{header}_{labels[0]}_{labels[1]}_loss_compare.pdf")


def get_element_statistics(data_set):
    """_summary_

    Args:
        data_set (torch.utils.data.dataset.Subset): tr_set or te_set

    Returns:
        pandas.core.frame.DataFrame: Dataframe of the counts on how many mterials have each elememnt.
    """
    species = []
    for Z in range(1, 119):
        species.append(Atom(Z).symbol)
    # create dictionary indexed by element names storing index of samples containing given element
    species_dict = {k: [] for k in species}
    len_data = len(data_set)
    for i in range(len_data):
        data = data_set[i]
        d_species = set([data.symbol[j] for j in range(len(data.symbol))])

        for d_specie in d_species:
            species_dict[d_specie].append(i)

    # create dataframe of element statistics
    stats = pd.DataFrame({'symbol': species})
    stats['data'] = stats['symbol'].astype('object')
    for specie in species:
        stats.at[stats.index[stats['symbol'] == specie].values[0], 'data'] = species_dict[specie]
    stats['count'] = stats['data'].apply(len)
    return stats


def plot_element_count_stack(data_set1, data_set2, header=None, title=None, 
                             bar_colors=['#90BE6D', '#277DA1']):
    """_summary_

    Args:
        data_set1 (torch.utils.data.dataset.Subset): tr_set or te_set
        data_set2 (torch.utils.data.dataset.Subset): tr_set or te_set
        header (str): header as the save dir and file name Defaults to None.
        title (str, optional): Figure title. Defaults to None.
        bar_colors (list, optional): [color1, color2]. Defaults to ['#90BE6D', '#277DA1'].
    """
    rows=2
    stats1 = get_element_statistics(data_set1)
    stats1_elems = set(stats1[stats1['count']>0]['symbol'])
    # stats1=stats1[stats1['count']>0].reset_index(drop=True)
    stats2 = get_element_statistics(data_set2)
    stats2_elems =set(stats2[stats2['count']>0]['symbol'])
    # stats1=stats1[stats1['count']>0].reset_index(drop=True)
    elems_common = stats1_elems.union(stats2_elems)
    idx_stats1 = []
    idx_stats2 = []
    for i in range(len(stats1)):
        if stats1['symbol'][i] in elems_common:
            idx_stats1.append(i)
    for i in range(len(stats2)):
        if stats2['symbol'][i] in elems_common:
            idx_stats2.append(i)
    stats1=stats1.iloc[idx_stats1].reset_index(drop=True)
    stats2=stats2.iloc[idx_stats2].reset_index(drop=True)

    anums1 = {}
    for l in range(len(stats1)):
        # if stats['count'][l] > 0:
        anums1[stats1['symbol'][l]]=stats1['count'][l]
    anums2 = {}
    for l in range(len(stats2)):
        # if stats['count'][l] > 0:
        anums2[stats2['symbol'][l]]=stats2['count'][l]
    fig, axs = plt.subplots(rows,1, figsize=(27, 10*rows)) 
    # for j in range(rows):
    bar_max = max(anums1.values())+max(anums2.values())
    if rows==2:
        ax0=axs[0]
        cols = max(len(anums1), len(anums2))//rows
        ax0.bar(range(cols), list(anums1.values())[:cols], width=0.6, color=bar_colors[0], label='Train')
        ax0.bar(range(cols), list(anums2.values())[:cols], bottom=list(anums1.values())[:cols], width=0.6, color=bar_colors[1], label='Test')
        ax0.set_xticks(np.arange(cols))
        ax0.set_xticklabels(list(anums1.keys())[:cols], fontsize = 27)
        ax0.set_ylim(0, bar_max*1.05)
        ax0.tick_params(axis='y', which='major', labelsize=23)
        ax0.legend()
        ax0.set_ylabel('Counts', fontsize=24)
        ax1=axs[1]
        ax1.bar(range(cols, len(anums1)), list(anums1.values())[cols:], width=0.6, color=bar_colors[0], label='Train')
        ax1.bar(range(cols, len(anums2)), list(anums2.values())[cols:], bottom=list(anums1.values())[cols:], width=0.6, color=bar_colors[1], label='Test')
        ax1.set_xticks(np.arange(cols, len(anums1)))
        ax1.set_xticklabels(list(anums1.keys())[cols:], fontsize = 27)
        ax1.tick_params(axis='y', which='major', labelsize=23)
        ax1.set_ylim(0, bar_max*1.05)
        ax1.legend()
        ax1.set_ylabel('Counts', fontsize=24)
    if title: fig.suptitle(title, ha='center', y=1., fontsize=20)
    fig.patch.set_facecolor('white')
    fig.savefig(f'{header}_element_count_{title}.png')
