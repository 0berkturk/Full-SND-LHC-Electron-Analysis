import torch
import numpy as np
import matplotlib.pyplot as plt

def analyze_and_plot_scifi_cuts(path, energy_bins, save_prefix="scifi_analysis"):
    # 1. Veri Yükleme ve Elektron Filtreleme
    print("opening path", path)
    data = torch.load(path)
    label = data["y"]
    print("opened data")
    
    # Sadece elektronları seçiyoruz (Not: Nötrino datası için 0, electron gun için 3 olabilir. Kontrol et!)
    electron_mask = (label == 0)
    energy_e = data["en3d"][electron_mask].numpy()
    total_events = len(energy_e)
    
    scifi_e = data["scifi_signals"][electron_mask]
    
    # 1.5 Muon Sistemi Kontrolü (Dinamik)
    if "us_signals" in data and "ds_horizontal" in data and "ds_vertical" in data:
        print("Muon (US/DS) verisi bulundu. Kesilim (US=0, DS=0) uygulanıyor...")
        us_e = data["us_signals"][electron_mask]
        ds_h_e = data["ds_horizontal"][electron_mask]
        ds_v_e = data["ds_vertical"][electron_mask]
        
        ds_h_padded = torch.nn.functional.pad(ds_h_e, (0, 0, 0, 1))
        ds_e = torch.stack([ds_h_padded, ds_v_e], dim=1)

        us_sum = us_e.sum(dim=(1, 2, 3))
        ds_sum = ds_e.sum(dim=(1, 2, 3))
        
        muon_veto_mask = (us_sum == 0) & (ds_sum == 0)
    else:
        print("Bilgi: Muon (US/DS) verisi bulunamadı! Veri setinin zaten Muon-Veto'dan geçtiği varsayılıyor.")
        # Zaten filtrelenmişse, tüm olaylar bu kesilimi geçmiş kabul edilir.
        muon_veto_mask = torch.ones(total_events, dtype=torch.bool)

    # 2. Tensör Toplamları ve İstasyon Analizi
    # scifi: Boyut [N, 2, 5, 1536] -> İstasyon bazında toplam almak için plane(1) ve channel(3) boyutlarını topluyoruz
    # Sonuç boyutu: [N, 5] (Her event için 5 istasyonun hit sayıları)
    hits_per_station = scifi_e.sum(dim=(1, 3))
    scifi_total_hits = hits_per_station.sum(dim=1)
    
    # Her event için kaç istasyonun >25 ve >0 hit aldığını hesaplıyoruz
    n_stations_gt_25 = (hits_per_station > 25).sum(dim=1)
    n_stations_gt_0 = (hits_per_station > 0).sum(dim=1)

    # 3. Kesilim (Cut) Maskelerinin Oluşturulması
    # Cut 1: Sadece SciFi içinde olan (US=0, DS=0, SciFi>0)
    cut1_mask = muon_veto_mask & (scifi_total_hits > 0)
    
    # Cut 2: Sadece 2 istasyon >25 hit. Ekstra maksimum 1 istasyona daha izin verilir (toplam >0 hitli istasyon <= 3)
    cut2_mask = cut1_mask & (n_stations_gt_25 == 2) & (n_stations_gt_0 <= 3)
    
    # Cut 3: Sadece 3 istasyon >25 hit. Ekstra maksimum 1 istasyona daha izin verilir (toplam >0 hitli istasyon <= 4)
    cut3_mask = cut1_mask & (n_stations_gt_25 == 3) & (n_stations_gt_0 <= 4)
    
    # Cut 4: Sadece 4 istasyon >25 hit. Ekstra maksimum 1 istasyona daha izin verilir (toplam >0 hitli istasyon <= 5)
    cut4_mask = cut1_mask & (n_stations_gt_25 == 4) & (n_stations_gt_0 <= 5)

    # Numpy array'e çevirme (Histogram ve Bin işlemleri için)
    cut1 = cut1_mask.numpy()
    cut2 = cut2_mask.numpy()
    cut3 = cut3_mask.numpy()
    cut4 = cut4_mask.numpy()

    # 3.5 Toplam Verimlilikleri (Total Efficiencies) Hesapla ve Yazdır
    print("\n--- Toplam Verimlilikler (Total Efficiencies) ---")
    print(f"Toplam Elektron Sayısı (Pre-cut): {total_events}")
    
    if total_events > 0:
        print(f"Cut 1 (Sadece SciFi): {cut1.sum() / total_events:.4f} ({cut1.sum()} event)")
        print(f"Cut 2 (2 İstasyon >25 hit): {cut2.sum() / total_events:.4f} ({cut2.sum()} event)")
        print(f"Cut 3 (3 İstasyon >25 hit): {cut3.sum() / total_events:.4f} ({cut3.sum()} event)")
        print(f"Cut 4 (4 İstasyon >25 hit): {cut4.sum() / total_events:.4f} ({cut4.sum()} event)")
    else:
        print("Veri setinde elektron bulunamadı.")
    print("-----------------------------------------------\n")

    # 4. Verimlilik (Efficiency) Hesaplaması
    bin_centers = []
    eff_cut1, eff_cut2, eff_cut3, eff_cut4 = [], [], [], []

    for i in range(len(energy_bins) - 1):
        en_low = energy_bins[i]
        en_high = energy_bins[i+1]
        
        in_bin_mask = (energy_e >= en_low) & (energy_e < en_high)
        total_in_bin = in_bin_mask.sum()
        
        if total_in_bin == 0:
            eff_cut1.append(0.0)
            eff_cut2.append(0.0)
            eff_cut3.append(0.0)
            eff_cut4.append(0.0)
        else:
            eff_cut1.append((in_bin_mask & cut1).sum() / total_in_bin)
            eff_cut2.append((in_bin_mask & cut2).sum() / total_in_bin)
            eff_cut3.append((in_bin_mask & cut3).sum() / total_in_bin)
            eff_cut4.append((in_bin_mask & cut4).sum() / total_in_bin)
            
        bin_centers.append((en_low + en_high) / 2.0)

    # 5. Çizim - Histogram
    plt.figure(figsize=(10, 7))
    plt.hist(energy_e, bins=50, range=(0, 50), alpha=0.3, color='black', label='All Electrons (Pre-cut)')
    plt.hist(energy_e[cut1], bins=50, range=(0, 50), histtype='step', linewidth=2, color='blue', label='Cut 1: Only in SciFi')
    plt.hist(energy_e[cut2], bins=50, range=(0, 50), histtype='step', linewidth=2, color='green', label='Cut 2: 2 Stations > 25 hits')
    plt.hist(energy_e[cut3], bins=50, range=(0, 50), histtype='step', linewidth=2, color='orange', label='Cut 3: 3 Stations > 25 hits')
    plt.hist(energy_e[cut4], bins=50, range=(0, 50), histtype='step', linewidth=2, color='red', label='Cut 4: 4 Stations > 25 hits')
    
    plt.xlabel("Energy (GeV)", fontsize=12)
    plt.ylabel("Number of Events", fontsize=12)
    plt.title("Electron Energy Distribution: SciFi Station Cuts", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f"{save_prefix}_histogram.png", dpi=300)
    plt.close()

    # 6. Çizim - Verimlilik (Efficiency) Eğrileri
    plt.figure(figsize=(10, 7))
    plt.plot(bin_centers, eff_cut1, marker='o', color='blue', label='Cut 1: Only in SciFi', linewidth=2)
    plt.plot(bin_centers, eff_cut2, marker='s', color='green', label='Cut 2: 2 Stations > 25 hits', linewidth=2)
    plt.plot(bin_centers, eff_cut3, marker='^', color='orange', label='Cut 3: 3 Stations > 25 hits', linewidth=2)
    plt.plot(bin_centers, eff_cut4, marker='d', color='red', label='Cut 4: 4 Stations > 25 hits', linewidth=2)
    
    plt.xlabel("Energy (GeV)", fontsize=12)
    plt.ylabel("Efficiency", fontsize=12)
    plt.title("Selection Efficiency vs Energy", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(0, 1.05)
    plt.savefig(f"{save_prefix}_efficiency.png", dpi=300)
    plt.close()

    print(f"Grafikler kaydedildi: {save_prefix}_histogram.png, {save_prefix}_efficiency.png")

# ÇALIŞTIRMA KOMUTLARI
path_electron = "/eos/user/b/beturk/snd/MonteCarlo/onlyscifi_withcuts/onlyscifi__sim_neutrino_2024_new.pt"
energy_bins = np.linspace(0, 2000, 50)
analyze_and_plot_scifi_cuts(path_electron, energy_bins, save_prefix="scifi_station_cuts")