import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import os
import numpy
import torch.nn.functional as F

cmap = plt.get_cmap('plasma')
cmap.set_under('white')


N=100000

def plot_qdc_energy(true_en_list, qdc_energy_list,us_list,ds_list,label_list,bins ,out_name):
    plt.figure()
# Sum over channels, width, and height dimensions (dimensions 1, 2, 3)
    for j in range(len(qdc_energy_list)):
        scifi=qdc_energy_list[j]
        us=us_list[j]
        ds=ds_list[j]
        true_en=true_en_list[j]

        total_qdc_scifi = scifi.sum(dim=(1, 2, 3))
        total_qdc_us = us.sum(dim=(1, 2, 3))
        total_qdc_ds = ds.sum(dim=(1, 2, 3))
        total_qdc = total_qdc_scifi + total_qdc_us + total_qdc_ds

        average_qdc = []
        average_scifi_qdc = []
        average_us_qdc = []
        average_ds_qdc = []

        std_qdc = []
        std_scifi_qdc = []
        std_us_qdc = []
        std_ds_qdc = []
        for i in range(len(bins)-1):
            en_min = bins[i]
            en_max = bins[i+1]
            index = (true_en>=en_min) & (true_en<en_max)
            average_qdc.append(total_qdc[index].mean().item())
            average_scifi_qdc.append(total_qdc_scifi[index].mean().item())
            average_us_qdc.append(total_qdc_us[index].mean().item())
            average_ds_qdc.append(total_qdc_ds[index].mean().item())
            std_qdc.append(total_qdc[index].std().item())
            std_scifi_qdc.append(total_qdc_scifi[index].std().item())
            std_us_qdc.append(total_qdc_us[index].std().item())
            std_ds_qdc.append(total_qdc_ds[index].std().item())


        plt.errorbar((bins[:-1]+bins[1:])/2, average_qdc, yerr=std_qdc, fmt='o-', label=f'Total QDC ({label_list[j]})',alpha=0.7)
        plt.errorbar((bins[:-1]+bins[1:])/2, average_scifi_qdc, yerr=std_scifi_qdc, fmt='s-', label=f'SciFi QDC({label_list[j]})',alpha=0.7)
        plt.errorbar((bins[:-1]+bins[1:])/2, average_us_qdc, yerr=std_us_qdc, fmt='^-', label=f'US QDC({label_list[j]})',alpha=0.7)
        plt.errorbar((bins[:-1]+bins[1:])/2, average_ds_qdc, yerr=std_ds_qdc, fmt='d-', label=f'DS QDC({label_list[j]})',alpha=0.7)

    
    plt.xlabel('True Energy [GeV]')
    plt.ylabel('Average QDC')
    plt.title(f'Average QDC vs True Energy')
    plt.legend()
    plt.grid()
    if os.path.exists("qdc_comparision") is False:
        os.mkdir("qdc_comparision")
    plt.savefig("qdc_comparision/"+out_name+"_average_qdc_energy.png", dpi=300)
    plt.clf()

    return total_qdc



def plot_1d_graphs(true_en_list, qdc_energy_list, label_list,bins,out_name ,xlabel="True Energy [GeV]",ylabel='Average QDC Energy[GeV]',title="Average QDC vs True Energy",outdir="qdc_comparision",show_ideal=False):
    plt.figure()
# Sum over channels, width, and height dimensions (dimensions 1, 2, 3)
    for j in range(len(qdc_energy_list)):
        total_qdc_scifi=qdc_energy_list[j]
        true_en=true_en_list[j]

        

        average_scifi_qdc = []

        std_scifi_qdc = []

        for i in range(len(bins)-1):
            en_min = bins[i]
            en_max = bins[i+1]
            index = (true_en>=en_min) & (true_en<en_max)
            average_scifi_qdc.append(total_qdc_scifi[index].mean().item())
            std_scifi_qdc.append(total_qdc_scifi[index].std().item())

        plt.errorbar((bins[:-1]+bins[1:])/2, average_scifi_qdc, yerr=std_scifi_qdc, fmt='s-', label=f'SciFi QDC({label_list[j]})',alpha=0.7)
    if show_ideal:
        ax = plt.gca()
        ax.axline((0, 0), slope=1, linestyle='--', color='black', label='Ideal')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    if os.path.exists(outdir) is False:
        os.mkdir(outdir)
    plt.savefig(f"{outdir}/{out_name}_{title}.png", dpi=300)
    plt.clf()



def load_as_lists(file_list):
    qdc_energy_list = []
    us_list = []
    ds_list = []
    true_en_list = []

    for fname in file_list:
        h5f = torch.load(fname, map_location="cpu")
        if "Neutrinos" in fname:
            label=h5f["y"]
            cut = (label == 0) | (label == 3)

        else:
            cut = torch.ones(h5f["y"].shape[0], dtype=torch.bool)

        scifi = h5f["scifi_signals"][cut]
        us = h5f["us_signals"][cut]
        ds_h = h5f["ds_horizontal"][cut]
        ds_v = h5f["ds_vertical"][cut]
        energy = h5f["en3d"][cut]

        ds_h_padded = F.pad(ds_h, (0, 0, 0, 1))
        ds = torch.stack([ds_h_padded, ds_v], dim=1)  # (N, 2, 4, 60)

        qdc_energy_list.append(scifi.sum(dim=(1, 2, 3))*0.059)
        us_list.append(us)
        ds_list.append(ds)
        true_en_list.append(energy)

    return qdc_energy_list, us_list, ds_list, true_en_list


def load_as_lists_scifi(file_list):
    qdc_energy_list = []
    us_list = []
    ds_list = []
    true_en_list = []
    hit_number_list=[]

    for fname in file_list:
        print(fname)
        h5f = torch.load(fname, map_location="cpu")
        for key in h5f:
            h5f[key] = h5f[key][:N]
        if "neutrino" in fname:
            label=h5f["y"]
            cut = (label == 0) | (label == 3)
        elif "electron" in fname:
            us_sum = h5f["us_signals"].sum(dim=(1, 2, 3))
            ds_sum_v = h5f["ds_vertical"].sum(dim=(1, 2))
            ds_sum_h = h5f["ds_horizontal"].sum(dim=(1, 2))
            cut =  (ds_sum_v == 0) & (ds_sum_h==0) & (us_sum == 0)

        else:
            cut = torch.ones(h5f["y"].shape[0], dtype=torch.bool)

        scifi = h5f["scifi_signals"][cut]
        scifi = torch.clamp_min(scifi, 0)
        energy = h5f["en3d"][cut]
        hit_number_list.append((scifi!=0).sum((1,2,3)).float())
        qdc_energy_list.append(scifi.sum(dim=(1, 2, 3))*0.059)
        true_en_list.append(energy)

    return qdc_energy_list,true_en_list,hit_number_list

"""TEST_DATA_DIR =["/eos/user/b/beturk/snd/MonteCarlo/create_datasets_of_snd/neutrons_QGSP_BERT_HP_PEN/test_combined_neutrons_QGSP_BERT_HP.pt",    
"/eos/user/b/beturk/snd/MonteCarlo/onlyscifi_withcuts/onlyscifi__sim_neutrino_2024_new.pt",
"/eos/user/b/beturk/snd/MonteCarlo/create_datasets_of_snd/kaons_FTFP_BERT/test_combined_kaons_FTFP_BERT.pt",
"/eos/user/b/beturk/snd/MonteCarlo/merge_electron_gun(biased)/test_combined_electrons_20_09_2025_v0.pt",
"/eos/user/b/beturk/snd/MonteCarlo/onlyscifi_withcuts/onlyscifi__combined_kshort_0_100_2025.pt",
"/eos/user/b/beturk/snd/MonteCarlo/magnetic_moment_mc/magnetic_mom_mc_0.pt"]
qdc_energy_list, true_en_list = load_as_lists_scifi(TEST_DATA_DIR)

label_list=["SND Kaons","My Electrons", "My Kaons","Cemal's Electrons"]
bins=np.linspace(0,100,20)
out_name="sndkaon_myelectrons"
#plot_qdc_energy(true_en_list, qdc_energy_list,us_list,ds_list,label_list,bins ,out_name)
plot_qdc_scifi_energy(true_en_list, qdc_energy_list,label_list,bins ,out_name)"""

TEST_DATA_DIR = ["/eos/user/b/beturk/snd/MonteCarlo/create_datasets_of_snd/neutrons_QGSP_BERT_HP_PEN/test_combined_neutrons_QGSP_BERT_HP.pt", 
"/eos/user/b/beturk/snd/MonteCarlo/onlyscifi_withcuts/onlyscifi__combined_neutron_0_100_2025.pt",

"/eos/user/b/beturk/snd/MonteCarlo/create_datasets_of_snd/kaons_FTFP_BERT/test_combined_kaons_FTFP_BERT.pt",
"/eos/user/b/beturk/snd/MonteCarlo/onlyscifi_withcuts/onlyscifi__combined_kshort_0_100_2025.pt",
"/eos/user/b/beturk/snd/MonteCarlo/onlyscifi_withcuts/onlyscifi__combined_klong_0_100_2025.pt",

"/eos/user/b/beturk/snd/MonteCarlo/onlyscifi_withcuts/onlyscifi__sim_neutrino_2024_new.pt",
"/eos/user/b/beturk/snd/MonteCarlo/merge_electron_gun(biased)/test_combined_electrons_20_09_2025_v0.pt",
]


qdc_energy_list, true_en_list,hit_number_list = load_as_lists_scifi(TEST_DATA_DIR)

label_list=["Neutrons QGSP_BERT_HP_PEN","My Neutrons", "Kaons FTFP_BERT" , "My K-short","My K-long","Neutrinos", "My Electrons"]
bins=np.linspace(0,100,20)
out_name="all_v1"
#plot_qdc_energy(true_en_list, qdc_energy_list,us_list,ds_list,label_list,bins ,out_name)
plot_1d_graphs(true_en_list, qdc_energy_list,label_list,bins ,out_name,show_ideal=True)
plot_1d_graphs(true_en_list, hit_number_list, label_list,bins, out_name ,xlabel="True Energy [GeV]",ylabel='Hit Number',title="Hit Number vs. True Energy",outdir="qdc_comparision")

"""
TEST_DATA_DIR =[
"/eos/user/b/beturk/snd/MonteCarlo/create_datasets_of_snd/neutrons_QGSP_BERT_HP_PEN/test_combined_neutrons_QGSP_BERT_HP.pt",
"/eos/user/b/beturk/snd/MonteCarlo/create_datasets_of_snd/NEUTRONS_FTFP_BERT/test_combined_NEUTRONS_FTFP_BERT.pt",

"/eos/user/b/beturk/snd/MonteCarlo/create_datasets_of_snd/kaons_QGSP_BERT_HP_PEN/test_combined_kaons_QGSP_BERT_HP_PEN.pt",
"/eos/user/b/beturk/snd/MonteCarlo/create_datasets_of_snd/kaons_FTFP_BERT/test_combined_kaons_FTFP_BERT.pt"
]
qdc_energy_list, true_en_list = load_as_lists_scifi(TEST_DATA_DIR)

label_list=["Neutrons QGSP_BERT_HP_PEN","Neutrons FTFP_BERT", "Kaons QGSP_BERT_HP_PEN","Kaons FTFP_BERT" ]
bins=np.linspace(0,100,20)
out_name="all_snd"
#plot_qdc_energy(true_en_list, qdc_energy_list,us_list,ds_list,label_list,bins ,out_name)
plot_qdc_scifi_energy(true_en_list, qdc_energy_list,label_list,bins ,out_name)"""
# hit number, done
# weighted hit denstiy
# qdc weighted hit wtr shower max 
# shower length
# shower radius
# shower center of qdc values.
#chatgptye sordum. oraya bak
