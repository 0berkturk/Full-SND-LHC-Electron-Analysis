import torch
import numpy as np

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

def get_efficiency_vs_energy(path, threshold, energy_bins):
    """
    Belirli bir threshold değeri için enerji aralıklarına (bin) göre efficiency hesaplar.
    """
    data = torch.load(path)
    energy = data["en3d"]
    
    # Sinyalleri çek
    us = data["us_signals"]
    ds_h = data["ds_horizontal"]
    ds_v = data["ds_vertical"]
    
    # DS padding ve birleştirme işlemleri
    ds_h_padded = torch.nn.functional.pad(ds_h, (0, 0, 0, 1))
    ds = torch.stack([ds_h_padded, ds_v], dim=1) 
    
    # Toplamları al
    us_sum = us.sum(dim=(1,2,3))
    ds_sum = ds.sum(dim=(1,2,3))
    
    efficiencies = []
    bin_centers = []

    # Enerji bin'leri üzerinde döngü
    for i in range(len(energy_bins) - 1):
        en_low = energy_bins[i]
        en_high = energy_bins[i+1]
        
        # Bu enerji aralığındaki parçacıklar
        in_bin_mask = (energy >= en_low) & (energy < en_high)
        total_in_bin = in_bin_mask.sum().item()
        
        if total_in_bin == 0:
            # Eğer bu aralıkta hiç parçacık yoksa verimlilik 0 (veya None yapabilirsin)
            efficiencies.append(0.0)
        else:
            # load_datav2'deki cut şartın: US toplamı threshold'dan büyük VE DS toplamı 0
            cut_mask = (us_sum <= threshold) & (ds_sum <= threshold)
            
            # Hem enerji aralığında olan hem de cut'ı geçenler
            passed_mask = in_bin_mask & cut_mask
            passed_count = passed_mask.sum().item()
            
            efficiencies.append(passed_count / total_in_bin)
            
        bin_centers.append((en_low + en_high) / 2.0)
        
    return bin_centers, efficiencies

def plot_eff_vs_energy(bin_centers, sig_eff, bkg_eff, threshold, save_path="efficiency_vs_energy.png"):
    """
    Sinyal ve arka plan verimliliklerini enerjiye göre aynı grafikte çizer.
    """
    plt.figure(figsize=(8,6))
    
    # Çizgiler ve noktalar
    plt.plot(bin_centers, sig_eff, marker='o', color='blue', label='Electrons', linewidth=2)
    plt.plot(bin_centers, bkg_eff, marker='s', color='red', label='Neutrino', linewidth=2)
    
    # Eksenler ve başlık
    plt.xlabel("Energy (GeV)", fontsize=12)
    plt.ylabel(f"Efficiency (Threshold = {threshold})", fontsize=12)
    plt.title("Efficiency vs Energy", fontsize=14)
    plt.legend()
    plt.yscale("log")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Ekseni biraz düzenlemek istersen:
    plt.ylim(0, 1.05) 
    
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved as: {save_path}")

path2="/eos/user/b/beturk/snd/MonteCarlo/particle_gun_advanced_v1/merge_electron_gun_0_50/test_combined_electron_0_50_2025.pt"

#path1="/eos/user/b/beturk/snd/MonteCarlo/particle_gun_advanced_v1/merge_kshort_gun_0_100/test_combined_kshort_0_100_2025.pt"

#path="/eos/user/b/beturk/snd/MonteCarlo/particle_gun_advanced_v1/merge_muon_gun_0_10/test_combined_muon_0_10_2025.pt"

#path1 = "/eos/user/b/beturk/snd/MonteCarlo/particle_gun_advanced_v1/merge_muon_gun_0_1000/test_combined_muon_0_1000_2025.pt"

path1 = "/eos/user/b/beturk/snd/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/test_sim_neutrino_2024_new.pt"

#sig_list_us, sig_list_ds = load_datav2(path2,0,50)
#bkg_list_us, bkg_list_ds = load_datav2(path1,0,50)


#plot_graph(bkg_list_us, bkg_list_ds, sig_list_us, sig_list_ds)




chosen_threshold = 0.1
# 0'dan 50'ye kadar 10'ar birimlik enerji aralıkları (bin) oluştur (0, 10, 20, 30, 40, 50)
# İsteğine göre numpy ile aralıkları daha hassas belirleyebilirsin, örneğin 5'erlik aralıklar için np.arange(0, 55, 5)
energy_bins = np.linspace(0, 1000, 100) # [0, 5, 10, ..., 50] şeklinde böler
# Signal için hesapla
bin_centers, sig_eff_energy = get_efficiency_vs_energy(path2, chosen_threshold, energy_bins)

# Background için hesapla
_, bkg_eff_energy = get_efficiency_vs_energy(path1, chosen_threshold, energy_bins)
# Grafiği çizdir
plot_eff_vs_energy(bin_centers, sig_eff_energy, bkg_eff_energy, chosen_threshold, save_path=f"eff_vs_energy_thr{chosen_threshold}.png")
## background efficiency vs signal eff.


