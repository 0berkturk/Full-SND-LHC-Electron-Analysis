import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import os
import numpy
cmap = plt.get_cmap('plasma')
cmap.set_under('white')
import h5py

def plot_2d_im(scifi_hits, j,out_name,energy):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    for i, ax in enumerate(axes):
        data = scifi_hits[j,i]  # shape (15, 512)
        #print(data.shape)
        nrows, ncols = data.shape

        masked_data = np.ma.masked_where(data == 0, data)
        #for x in range(ncols+ 1):
           # ax.axvline(x, color='black', linewidth=0.5)
        for y in range(nrows + 1):
            ax.axhline(y, color='black', linewidth=0.5)
        #im = ax.imshow(masked_data,
                      # cmap="inferno",
                     #  origin="lower",
                    #   aspect="auto",
                    #   interpolation="nearest")
        im = ax.imshow(masked_data, interpolation='nearest', aspect="auto",cmap=cmap)

        ax.set_title(f"SciFi Projection {['X', 'Y'][i]},({energy} GeV)")
        ax.set_xlabel("PMTs")
        if i == 0:
            ax.set_ylabel("Z (plane index)")

    # Move colorbar to the right side of both plots
    cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.04)
    cbar.set_label("Signal")

    plt.tight_layout(rect=[0, 0, 0.95, 1])  # leave space for colorbar
    plt.savefig("signals/"+ out_name+ str(j) + ".png", dpi=200)
    plt.clf()

def plot_2d_im_single(data, j,out_name,energy):
    data=data[j]
    nrows, ncols = data.shape
    masked_data = np.ma.masked_where(data == 0, data)
    #for x in range(ncols+ 1):
       # ax.axvline(x, color='black', linewidth=0.5)
    #for y in range(nrows + 1):
        #plt.hlines(y, color='black', linewidth=0.5)
    #im = ax.imshow(masked_data,
                  # cmap="inferno",
                 #  origin="lower",
                #   aspect="auto",
                #   interpolation="nearest")
    plt.imshow(masked_data, interpolation='nearest', aspect="auto",cmap=cmap)

    plt.title(f"DS SiPMs Projection ({energy} GeV)")
    plt.xlabel("PMT")
    plt.ylabel("Z (plane index)")

    # Move colorbar to the right side of both plots
    plt.colorbar()
    #plt.la("Signal")

    #plt.tight_layout(rect=[0, 0, 0.95, 1])  # leave space for colorbar
    plt.savefig("signals/"+ out_name+ str(j) + ".png", dpi=200)
    plt.clf()

def find_shower_max(data):
    # shower maximum by mean ? then theere is angle
    #DATA is Nx2x5x1536
    #first find layer, then go that layer find max x and y.
    summed_xy= torch.sum(data,(1,3))
    max_point_layer = torch.argmax(summed_xy,dim=1)
    batch_size = data.size(0)
    batch_indices = torch.arange(batch_size)

    data_at_z = data[batch_indices,:,max_point_layer,:] ## 2x1536
    max_point_hor = torch.argmax(data_at_z[:,0],dim=1)
    max_point_ver = torch.argmax(data_at_z[:,1],dim=1)
    return max_point_layer, max_point_hor, max_point_ver

TEST_DATA_DIR = ["/eos/user/b/beturk/snd/MonteCarlo/particle_gun_advanced_v1/merge_electron_gun_0_50/test_combined_electron_0_50_2025.pt"]
h5f=torch.load(TEST_DATA_DIR[0])
print(h5f["y"])
scifi = torch.tensor(h5f["scifi_signals"])
us = torch.tensor(h5f["us_signals"])
ds_h = torch.tensor(h5f["ds_horizontal"])
ds_v = torch.tensor(h5f["ds_vertical"])
energy = torch.tensor(h5f["en3d"])
print(scifi.shape)

max_point_layer, max_point_hor, max_point_ver = find_shower_max(scifi)
print(max_point_layer.shape)
print(max_point_layer)

at_layer_5 = max_point_layer== 4
scifi = scifi[at_layer_5]

N=20
scifi = scifi[:N]
print(scifi.shape)
for i in range(0,N):
    plot_2d_im(scifi,i,"scifi",int(energy[i]))
    plot_2d_im_single(ds_h,i,"ds hor",int(energy[i]) )
    plot_2d_im_single(ds_v,i,"ds ver",int(energy[i]) )
    plot_2d_im_single(us[:,0],i,"us",int(energy[i]) )
    

