import os
import sys
import torch
import ROOT
import SndlhcGeo

print("=== SND@LHC GEO-LUT BUILDER BAŞLADI ===", flush=True)

def get_physical_position(Scifi, detID, n_vert):
    """
    SND@LHC geometrisinden ilgili detID'ye sahip fiberin X, Y, Z koordinatlarını çeker.
    """
    A = ROOT.TVector3()
    B = ROOT.TVector3()
    
    # Donanım hizalamalı SiPM pozisyonunu alt kütüphaneden çek
    Scifi.GetSiPMPosition(detID, A, B)
    
    hit_z = A.Z() # Z ekseni fiber boyunca sabittir
    
    if n_vert == 1: 
        hit_val = A.X() # Dikey fiberler X eksenini ölçer
    else:           
        hit_val = A.Y() # Yatay fiberler Y eksenini ölçer
        
    return hit_val, hit_z

def main():
    # İşlenecek nominal geometri dosyalarının yolları
    GEO_FILES = {
        "2022": "/eos/experiment/sndlhc/convertedData/physics/2022/geofile_sndlhc_TI18_V4_2022.root",
        "2023": "/eos/experiment/sndlhc/convertedData/physics/2023/geofile_sndlhc_TI18_V3_2023.root",
        "2024": "/eos/experiment/sndlhc/convertedData/physics/2024/geofile_sndlhc_TI18_V12_2024.root",
        "2025": "/eos/experiment/sndlhc/convertedData/physics/2025/geofile_sndlhc_TI18_V8_2025.root",
        "2026": "/eos/experiment/sndlhc/convertedData/physics/2026/geofile_sndlhc_TI18_V4_2026.root"
    }

    OUTPUT_DIR = "./geo_luts"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for year, geo_path in GEO_FILES.items():
        if not os.path.exists(geo_path):
            print(f"Uyarı: {year} yılına ait geometri dosyası bulunamadı, atlanıyor. ({geo_path})", flush=True)
            continue
            
        print(f"\n[{year}] Geometrisi yükleniyor...", flush=True)
        
        # Hafıza çakışmalarını önlemek için önceki GeoManager'ı temizle
        if ROOT.gGeoManager:
            ROOT.gGeoManager.Clear()
        
        # Geometri arayüzünü başlat
        geo = SndlhcGeo.GeoInterface(geo_path)
        Scifi = geo.modules['Scifi']
        
        # Hedef Tensör Yapısı: [Orientation (2), Plane (5), Channel (1536), Koordinat/Z (2)]
        current_lut = torch.zeros((2, 5, 1536, 2), dtype=torch.float32)
        
        print(f"  -> {year} yılı için sabit kanal haritası oluşturuluyor...", flush=True)
        
        # Tüm teorik kanalları tarayarak donanım ID'lerini (STMRFFF) hesapla
        for n_vert in range(2):            # 0: Yatay (Horizontal), 1: Dikey (Vertical)
            for n_plane in range(1, 6):    # Detektör istasyonları: 1, 2, 3, 4, 5
                tensor_plane_idx = n_plane - 1 # PyTorch tensör indeksi: 0, 1, 2, 3, 4
                
                for n_chan in range(1536): # Toplam kanal sayısı: 3 Mat * 4 SiPM * 128 Kanal
                    
                    # scifi_array_id fonksiyonunun donanımsal ters matris eşlemesi (STMRFFF)
                    mat_no = n_chan // 512
                    sipm_no = (n_chan % 512) // 128
                    local_chan = (n_chan % 512) % 128
                    
                    # Bit kaydırma kurallarına göre benzersiz detID üretimi
                    detID = (n_plane * 1000000) + (n_vert * 100000) + (mat_no * 10000) + (sipm_no * 1000) + local_chan
                    
                    # Geometriden cm cinsinden konumları oku
                    hit_val, hit_z = get_physical_position(Scifi, detID, n_vert)
                    
                    # Verileri tensör hücelerine yerleştir
                    current_lut[n_vert, tensor_plane_idx, n_chan, 0] = hit_val
                    current_lut[n_vert, tensor_plane_idx, n_chan, 1] = hit_z
        print(current_lut)

        # Çıktıyı PyTorch formatında diske yazma
        save_path = os.path.join(OUTPUT_DIR, f'geo_lut_{year}.pt')
        torch.save(current_lut, save_path)
        print(f"  -> Başarıyla kaydedildi: {save_path}", flush=True)

    print("\n[BİTTİ] Tüm yıllar için sabit geometri LUT dosyaları oluşturuldu.", flush=True)

if __name__ == "__main__":
    main()