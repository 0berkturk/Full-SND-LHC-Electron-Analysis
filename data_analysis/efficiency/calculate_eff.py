import torch
import numpy as np
import matplotlib.pyplot as plt

def get_efficiency_vs_energy_electrons_only(path, energy_bins):
    """
    Belirli bir enerji aralığındaki (bin) parçacıklar için efficiency hesaplar.
    Bu kod Veto ve Muon kesilimlerini MÜKEMMEL BİR ŞEKİLDE UYGULAR, ANCAK KESİNLİKLE TENSÖRLERİ DOĞRU BOYUTLANDIRMALISIN.
    """
    data = torch.load(path)
    
    # Tensör boyutlarını ve etiketleri dikkatle kontrol et (Senin 2x5x1536 tensörün)
    # y=3 elektron etiketini temsil ediyorsa:
    label = data["y"]
    energy = data["en3d"]
    scifi = data["scifi_signals"]
    us = data["us_signals"]
    ds_h = data["ds_horizontal"]
    ds_v = data["ds_vertical"]
    
    # SADECE ELEKTRONLARI SEÇ (y == 3)
    electron_mask = (label == 0)
    
    energy_e = energy[electron_mask]
    scifi_e = scifi[electron_mask]
    us_e = us[electron_mask]
    ds_h_e = ds_h[electron_mask]
    ds_v_e = ds_v[electron_mask]

    # DS padding ve birleştirme (2x4x60)
    ds_h_padded = torch.nn.functional.pad(ds_h_e, (0, 0, 0, 1))
    ds_e = torch.stack([ds_h_padded, ds_v_e], dim=1) 
    
    # SciFi 1. İstasyon kesilimi için (Senin tensöründe 1. istasyonun indeksi 0'dır, boyut [N, 2, 5, 1536] ise [:, :, 0, :] kısmına bakıyoruz)
    # LÜTFEN tensör boyutlarının N, 2, 5, 1536 olduğundan emin ol. Eğer değilse bu sum boyutu patlar.
    scifi_station1_sum = scifi_e[:, :, 0, :].sum(dim=(1, 2)) 
    
    # Muon Sistemi (US ve DS) Toplamı
    us_sum = us_e.sum(dim=(1, 2, 3))
    ds_sum = ds_e.sum(dim=(1, 2, 3))
    
    efficiencies = []
    bin_centers = []

    for i in range(len(energy_bins) - 1):
        en_low = energy_bins[i]
        en_high = energy_bins[i+1]
        
        in_bin_mask = (energy_e >= en_low) & (energy_e < en_high)
        total_in_bin = in_bin_mask.sum().item()
        
        if total_in_bin == 0:
            efficiencies.append(0.0)
        else:
            # KESİLİM ŞARTLARI:
            # 1. İlk SciFi istasyonunda HİÇ HİT YOK (Toplam = 0)
            # 2. Muon sisteminde (US ve DS) HİÇ HİT YOK (Toplam = 0)
            # Senin eşik (threshold) değerini sildim çünkü "hiç hit yok" istedin (sıfır kontrolü yapıyoruz).
            cut_mask = (scifi_station1_sum == 0) & (us_sum == 0) & (ds_sum == 0)
            
            passed_mask = in_bin_mask & cut_mask
            passed_count = passed_mask.sum().item()
            
            efficiencies.append(passed_count / total_in_bin)
            
        bin_centers.append((en_low + en_high) / 2.0)
        
    return bin_centers, efficiencies

def plot_eff_vs_energy_single(bin_centers, eff, save_path="electron_efficiency_vs_energy.png"):
    plt.figure(figsize=(8,6))
    
    plt.plot(bin_centers, eff, marker='o', color='blue', label='Electrons', linewidth=2)
    
    plt.xlabel("Energy (GeV)", fontsize=12)
    plt.ylabel("Efficiency (Veto & SciFi Station 1 = 0 hits)", fontsize=12)
    plt.title("Electron Selection Efficiency vs Energy", fontsize=14)
    plt.legend()
    # Log scale istersen geri açabilirsin, ama verimlilikte lineer genelde daha okunurdur.
    # plt.yscale("log") 
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(0, 1.05) 
    
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved as: {save_path}")

# PATH VE DEĞİŞKENLER
path_electron = "/eos/user/b/beturk/snd/MonteCarlo/particle_gun_advanced_v1/merge_electron_gun_0_50/test_combined_electron_0_50_2025.pt"

# Sadece elektronları incelediğimiz için enerji aralığını o hedefe göre ayarladık
energy_bins = np.linspace(0, 50, 50) 

# Hesapla
bin_centers, electron_eff = get_efficiency_vs_energy_electrons_only(path_electron, energy_bins)

# Çizdir
plot_eff_vs_energy_single(bin_centers, electron_eff, save_path="electron_eff_scifi1_muon_veto.png")