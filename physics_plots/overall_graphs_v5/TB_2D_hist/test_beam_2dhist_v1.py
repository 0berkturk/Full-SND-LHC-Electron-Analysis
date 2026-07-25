import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import os
import numpy
import torch.nn.functional as F
import glob


cmap = plt.get_cmap('plasma')
cmap.set_under('white')


def plot_2d_hist(true_en_list, qdc_energy_list,
                 bins_x=50, bins_y=50,
                 out_name="qdc_vs_true", xlabel="True Energy [GeV]",
                 ylabel='QDC Energy [GeV]', title="QDC vs True Energy",
                 outdir="qdc_comparison"):
    
    plt.figure(figsize=(8,6))

    # Binleri hesapla x_min, x_max, y_min, y_max, 
    #bins_x_arr = np.linspace(x_min, x_max, bins_x+1)
    #bins_y_arr = np.linspace(y_min, y_max, bins_y+1)


    true_en = true_en_list
    qdc_en = qdc_energy_list

    # 2D histogram
    hist = plt.hist2d(true_en, qdc_en, bins=[bins_x, bins_y],
                        cmap=cmap, alpha=0.8, vmin=1)  # vmin=1 ile 0 değerleri alt değer sayılır

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar()
    plt.grid(True)

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    plt.savefig(f"{outdir}/new_{out_name}.png", dpi=300)
    plt.clf()


def load_as_lists(file_list):
    scifi_list = []
    us_list = []
    ds_hor_list = []
    ds_ver_list = []
    #torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    for fname in file_list:
        print(fname)
        data = torch.load(fname, weights_only=False)
        cut = torch.ones_like(data["scifi_05usualtime_hits_per_layer"])
        hitx = torch.as_tensor(data["scifi_hitx_in_64r"], dtype=torch.float32)
        hity = torch.as_tensor(data["scifi_hity_in_64r"], dtype=torch.float32)
        cut = (hitx > 15) & (hity > 15)

        if PLOT_MEAN_HIT_NUMBERS:
            scifi_prop=data["scifi_05usualtime_hits_per_layer"][cut]
            us_prop=data["us_3usualtime_hits_per_layer"][cut]
            ds_hor_prop=data["dsh_notime_hits_per_layer"][cut]
            ds_ver_prop=data["dsv_notime_hits_per_layer"][cut]
        elif PLOT_MEAN_QDC_ENERGY:
            scifi_prop=data["scifi_05usualtime_qdc_per_layer"][cut]
            us_prop=data["us_3usualtime_qdc_per_layer"][cut]
            ds_hor_prop=data["dsh_notime_qdc_per_layer"][cut]
            ds_ver_prop=data["dsv_notime_qdc_per_layer"][cut]
        

    return scifi_prop, us_prop, ds_hor_prop, ds_ver_prop

TEST_DATA_DIR_DATA=[]
#TEST_DATA_DIR_DATA.append(glob.glob(f"/eos/experiment/sndlhc/users/beturk/TB/TB_Data/2023/*pions_180GeV_2Fe*.pt")[0])
#TEST_DATA_DIR_DATA.append(glob.glob(f"/eos/experiment/sndlhc/users/beturk/TB/TB_Data/2024/*pions_180GeV_1Fe*.pt")[0])
#TEST_DATA_DIR_DATA.append(glob.glob(f"/eos/experiment/sndlhc/users/beturk/TB/TB_Data/2024/*electrons_50GeV*.pt")[0])
#TEST_DATA_DIR_DATA.append(glob.glob(f"/eos/experiment/sndlhc/users/beturk/TB/TB_Data/2024/*electrons_150GeV*.pt")[0])
#TEST_DATA_DIR_DATA.append(glob.glob(f"/eos/experiment/sndlhc/users/beturk/TB/TB_Data/2024/*electrons_300GeV*.pt")[0])
PLOT_MEAN_HIT_NUMBERS=False
PLOT_MEAN_QDC_ENERGY=True
scifi_prop, us_prop, ds_hor_prop, ds_ver_prop = load_as_lists(TEST_DATA_DIR_DATA)

plot_2d_hist(scifi_prop[:,1], scifi_prop[:,2], bins_x=50, bins_y=50,
                 out_name="pion2023_scifi_1_2", xlabel="QDC of Station 1",
                 ylabel='QDC of Station 2', title="2D histogram of QDC energy deposition in SciFi Station 1 vs. Station 2",
                 outdir="qdc_comparison")


scifi=scifi_prop.sum(dim=(1))
ds = ds_hor_prop.sum(dim=(1))+ds_ver_prop.sum(dim=(1))
plot_2d_hist(ds, scifi, bins_x=50, bins_y=50,
                 out_name="pion2023_scifi_ds", xlabel="QDC of 1 Layer DS",
                 ylabel='QDC of Total SciFi', title="2D histogram of QDC energy deposition in SciFi vs. 1 Layer DS",
                 outdir="qdc_comparison")


us = us_prop.sum(dim=(1))
plot_2d_hist(us, scifi, bins_x=50, bins_y=50,
                 out_name="pion2023_scifi_us", xlabel="QDC of Total US",
                 ylabel='QDC of Total SciFi', title="2D histogram of QDC energy deposition in SciFi vs. US",
                 outdir="qdc_comparison")

plot_2d_hist(us_prop[:,0], scifi, bins_x=50, bins_y=50,
                 out_name="pion2023_scifi_us_1layer", xlabel="QDC of 1 Layer US",
                 ylabel='QDC of Total SciFi', title="2D histogram of QDC energy deposition in SciFi vs. 1 Layer US",
                 outdir="qdc_comparison")