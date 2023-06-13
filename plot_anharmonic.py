#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

#%%
file = './data/anharmonic_fc_2.pkl'
data = pkl.load(open(file, 'rb'))
data['kappa_sc'] = data['kappa'].map(lambda x: np.sum(x[:, :3], axis=-1)/3)
data['kappa_sc_mean'] = data['kappa_sc'].map(lambda x:x.mean())
print('mean of kappa (scalar): ', data['kappa_sc_mean'].mean())
keys = data.keys()

#%%
idx = 37
fig, ax = plt.subplots(1,1,figsize=(5,5))
row = data.iloc[idx]
mpid = row['mpid']
kappa = row['kappa']
temp = row['temperature']
m_gru = row['gruneisen']
freq = row['frequency']
weight = row['weight']
kappa_sc = np.sum(kappa[:, :3], axis=-1)/3
ax.plot(temp, kappa_sc)
ax.set_xlabel('Temperature [K]')
ax.set_ylabel('kappa [W/mK]')
ax.set_title(mpid)


# nband = freq.shape[-1]
# A = 0
# for i in range(nband):
#     A += m_gru[:, i]*freq[:, i]@weight
# B = 0
# for i in range(nband):
#     B += freq[:, i]@weight
nband = freq.shape[-1]
A = np.einsum('ij, ij, i->', m_gru, freq, weight)
B = np.einsum('ij, i->', freq, weight)
# print('A: ', A)
# print('B: ', B)
# print('A/B: ', A/B)
gru = A/B
print(f'[{mpid}] gru: ', gru)

#%%
data['gru'] = data.apply(lambda row: np.einsum('ij, ij, i->', row['gruneisen'], row['frequency'], row['weight']) / 
                                  np.einsum('ij, i->', row['frequency'], row['weight']), axis=1)

#%%
num = 4
fig, axs = plt.subplots(1,num,figsize=(6*num,5))
for idx in range(num):
    ax = axs[idx]
    row = data.iloc[idx]
    mpid = row['mpid']
    kappa = row['kappa']
    temp = row['temperature']
    kappa_sc = np.sum(kappa[:, :3], axis=-1)/3
    ax.plot(temp, kappa_sc)
    ax.set_xlabel('Temperature [K]')
    ax.set_ylabel('kappa [W/m-K]')
    ax.set_xscale('log')
    ax.set_title(mpid)

#%%
num = 4
start = 30
fig, axs = plt.subplots(1,num,figsize=(6*num,5))
for i in range(num):
    ax = axs[i]
    idx = start + i
    row = data.iloc[idx]
    mpid = row['mpid']
    kappa = row['kappa']
    temp = row['temperature']
    kappa_sc = np.sum(kappa[:, :3], axis=-1)/3
    print('kappa_sc mean: ', kappa_sc.mean())
    ax.plot(temp, kappa_sc)
    ax.set_xlabel('Temperature [K]')
    ax.set_ylabel('kappa [W/m-K}]')
    ax.set_title(mpid)

#%%
idx = 10
check_dim=False
if check_dim:
    for k in keys:
        out = data[k][0]
        if type(out)!=str:
            print(k, data[k][0].shape)
        else:
            print(k, data[k][0])

# %%
