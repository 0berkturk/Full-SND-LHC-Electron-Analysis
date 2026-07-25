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

def plot_multiple_hist(qdc_energy_list,N,xmin,xmax,x_label, title, label_str, outdir,alpha_list,name="EMPTY"):
    for i in range(len(qdc_energy_list)):
        qdc_energy=qdc_energy_list[i]
        if N!=None:
            bins = np.linspace(xmin, xmax, N)
            plt.hist(qdc_energy,bins=bins,label=label_str[i],alpha=alpha_list[i],histtype='step',density=True )
            plt.xlim(xmin, xmax)
        else:
            plt.hist(qdc_energy,bins=40,label=label_str[i],alpha=alpha_list[i],histtype='step',density=True)
        # Log scale
    plt.yscale('log')
    plt.xlabel(x_label)
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    # Save
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/{name}.png", dpi=300)
    plt.close()

def load_as_lists(fname):
    scifi_list = []
    us_list = []
    ds_hor_list = []
    ds_ver_list = []
    #torch.serialization.add_safe_globals([np.core.multiarray.scalar])

    print(fname)
    data = torch.load(fname, weights_only=False)
    cut = torch.ones_like(data["scifi_05usualtime_hits_per_layer"])
    hitx = torch.as_tensor(data["scifi_hitx_in_64r"], dtype=torch.float32)
    hity = torch.as_tensor(data["scifi_hity_in_64r"], dtype=torch.float32)
    print(hitx)
    cut = (hitx > 15) & (hity > 15)


    scifi_prop=data["scifi_05usualtime_qdc_per_layer"][cut]
    us_prop=data["us_3usualtime_qdc_per_layer"][cut]
    ds_hor_prop=data["dsh_notime_qdc_per_layer"][cut]
    ds_ver_prop=data["dsv_notime_qdc_per_layer"][cut]


    return scifi_prop.sum(dim=(1)), us_prop[:,0], ds_hor_prop[:,0] , ds_ver_prop[:,0]

TEST_DATA_DIR_DATA=[]
#TEST_DATA_DIR_DATA.append(glob.glob(f"/eos/experiment/sndlhc/users/beturk/TB/TB_Data/2024/*pions_180GeV_W*.pt")[0])
#TEST_DATA_DIR_DATA.append(glob.glob(f"/eos/experiment/sndlhc/users/beturk/TB/TB_Data/2024/*electrons_50GeV*.pt")[0])
#TEST_DATA_DIR_DATA.append(glob.glob(f"/eos/experiment/sndlhc/users/beturk/TB/TB_Data/2024/*electrons_150GeV*.pt")[0])
# W vs 2Fe in 2024 maybe add vs 23 3Fe
# 24 vs 23 for each 1Fe, 2Fe

GROUP1=True
if GROUP1:
    scifi_24, us_24,ds_24_hor,ds_24_ver = load_as_lists(glob.glob(f"/eos/experiment/sndlhc/users/beturk/TB/TB_Data/2024/*pions_180GeV_W*.pt")[0])

    scifi_24_2fe, us_24_2fe, ds_24_hor_2fe, ds_24_ver_2fe = load_as_lists(glob.glob(f"/eos/experiment/sndlhc/users/beturk/TB/TB_Data/2024/*pions_180GeV_2Fe*.pt")[0])

    scifi_23_3fe, us_23_3fe,ds_23_hor_3fe,ds_23_ver_3fe = load_as_lists(glob.glob(f"/eos/experiment/sndlhc/users/beturk/TB/TB_Data/2023/*pions_180GeV_3Fe*.pt")[0])
    scifi_list = [scifi_24, scifi_24_2fe, scifi_23_3fe]
    us_or_ds = [ds_24_hor+ds_24_ver, ds_24_hor_2fe+ds_24_ver_2fe , us_23_3fe]
    label_str=["2W(2024)","2Fe(2024)","3Fe(2023)"]
    name="Wall_Comp"
    

else:
    scifi_24_2fe, us_24_2fe, ds_24_hor_2fe, ds_24_ver_2fe = load_as_lists(glob.glob(f"/eos/experiment/sndlhc/users/beturk/TB/TB_Data/2024/*pions_180GeV_2Fe*.pt")[0])

    scifi_24_1fe, us_24_1fe, ds_24_hor_1fe, ds_24_ver_1fe = load_as_lists(glob.glob(f"/eos/experiment/sndlhc/users/beturk/TB/TB_Data/2024/*pions_180GeV_1Fe*.pt")[0])

    scifi_23_2fe, us_23_2fe, ds_23_hor_2fe, ds_23_ver_2fe = load_as_lists(glob.glob(f"/eos/experiment/sndlhc/users/beturk/TB/TB_Data/2023/*pions_180GeV_2Fe*.pt")[0])

    scifi_23_1fe, us_23_1fe, ds_23_hor_1fe, ds_23_ver_1fe = load_as_lists(glob.glob(f"/eos/experiment/sndlhc/users/beturk/TB/TB_Data/2023/*pions_180GeV_1Fe*.pt")[0])

    scifi_list = [scifi_24_2fe, scifi_24_1fe, scifi_23_2fe, scifi_23_1fe]
    us_or_ds = [ds_24_hor_2fe+ds_24_ver_2fe, ds_24_hor_1fe+ds_24_ver_1fe , us_23_2fe,us_23_1fe]
    label_str=["2Fe(2024)","1Fe(2024)","2Fe(2023)","1Fe(2023)"]
    name="Year_Comp"


alpha_list=[0.9, 0.9, 0.9, 0.9]
plot_multiple_hist(scifi_list, None ,None,None, "Total SciFi QDC", "Total SciFi QDC Distribution", label_str, "plots",alpha_list,name)