import torch


def load_data(path1,en1,en2):
    data1 = torch.load(path1)
    print(data1["y"].shape,data1["y"])
    label = data1["y"]
    energy = data1["en3d"]
    cut = (energy<en2 ) & (energy>en1) #& (label==3)

    energy = energy[cut]
    total_particles = len(energy)
    print("total",total_particles)
    #scifi = data1["scifi_signals"][cut]
    us = data1["us_signals"][cut]
    ds_h = data1["ds_horizontal"][cut]
    ds_v = data1["ds_vertical"][cut]
    
    ds_h_padded = torch.nn.functional.pad(ds_h, (0, 0, 0, 1))
    ds = torch.stack([ds_h_padded, ds_v], dim=1) #2x4x60

    #scifi_sum = scifi_data.sum(dim=(1,2,3))
    us_sum = us.sum(dim=(1,2,3))
    ds_sum = ds.sum(dim=(1,2,3))
    
    list_us =[]
    list_ds = []

    thresholds = [0,1,2,3,4,5,6,7,10,15,20,25,30]
    for threshold in thresholds:
        cut_us = us_sum>threshold 
        cut_ds = ds_sum>threshold
    
        n_ds_zero = cut_ds.sum()
        n_us_zero = cut_us.sum()

        list_us.append(n_us_zero/total_particles)
        list_ds.append(n_ds_zero/total_particles)
        print(list_us)

    return list_us, list_ds

def load_datav2(path1,en1,en2):
    data1 = torch.load(path1)
    print(data1["y"].shape,data1["y"])
    label = data1["y"]
    energy = data1["en3d"]
    cut = (energy<en2 ) & (energy>en1) #& (label==3)

    energy = energy[cut]
    total_particles = len(energy)
    print("total",total_particles)
    #scifi = data1["scifi_signals"][cut]
    us = data1["us_signals"][cut]
    ds_h = data1["ds_horizontal"][cut]
    ds_v = data1["ds_vertical"][cut]
    
    ds_h_padded = torch.nn.functional.pad(ds_h, (0, 0, 0, 1))
    ds = torch.stack([ds_h_padded, ds_v], dim=1) #2x4x60

    #scifi_sum = scifi_data.sum(dim=(1,2,3))
    us_sum = us.sum(dim=(1,2,3))
    ds_sum = ds.sum(dim=(1,2,3))
    
    list_us =[]
    list_ds = []

    thresholds = [0,1,2,3,4,5,6,7,10,15,20,25,30]
    for threshold in thresholds:
        cut_us = (us_sum>threshold ) & (ds_sum==0)
        
        n_us_zero = cut_us.sum()

        list_us.append(n_us_zero/total_particles)
        print(list_us)

    return list_us, list_ds
    
import matplotlib.pyplot as plt

def plot_graph(bkg_list_us, bkg_list_ds, sig_list_us, sig_list_ds, save_path="signal_vs_background.png"):
    plt.figure(figsize=(8,6))
    
    # Scatter plots
    plt.scatter(bkg_list_us, sig_list_us, color='blue', alpha=0.8, label='US')
    #plt.scatter(bkg_list_ds, sig_list_ds, color='red', alpha=0.8, label='DS')
    
    # Labels and title
    plt.xlabel("Background", fontsize=12)
    plt.ylabel("Signal", fontsize=12)
    plt.title("Signal Eff. vs Background Eff. by Appliying SciFi QDC Cut ", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save the figure
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved as: {save_path}")


path2="/eos/user/b/beturk/snd/MonteCarlo/particle_gun_advanced_v1/merge_electron_gun_0_50/test_combined_electron_0_50_2025.pt"

path1="/eos/user/b/beturk/snd/MonteCarlo/particle_gun_advanced_v1/merge_kshort_gun_0_100/test_combined_kshort_0_100_2025.pt"

#path="/eos/user/b/beturk/snd/MonteCarlo/particle_gun_advanced_v1/merge_muon_gun_0_10/test_combined_muon_0_10_2025.pt"

#path = "/eos/user/b/beturk/snd/MonteCarlo/particle_gun_advanced_v1/merge_muon_gun_0_1000/test_combined_muon_0_1000_2025.pt"

#path = "/eos/user/b/beturk/snd/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/test_sim_neutrino_2024_new.pt"
sig_list_us, sig_list_ds = load_datav2(path2,0,50)
bkg_list_us, bkg_list_ds = load_datav2(path1,0,50)


plot_graph(bkg_list_us, bkg_list_ds, sig_list_us, sig_list_ds)


## background efficiency vs signal eff.


