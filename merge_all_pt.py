import os
import torch
from pathlib import Path

def merge_pt_files(input_dir, output_filename):
    """
    Belirtilen dizindeki tüm .pt dosyalarını birleştirir ve kaydeder.
    """
    merged_data = {}
    directory = Path(input_dir)
    
    # Dizindeki tüm .pt dosyalarını bul
    pt_files = list(directory.glob("*.pt"))
    
    if not pt_files:
        print(f"Uyarı: '{input_dir}' dizininde hiç .pt dosyası bulunamadı.")
        return

    print(f"Toplam {len(pt_files)} adet .pt dosyası bulundu. Birleştiriliyor...\n")

    for file_path in sorted(pt_files):
        try:
            # Sadece ağırlık/tensör verisi içeriyorsa güvenliği artırmak için weights_only=True eklenebilir
            data = torch.load(file_path, weights_only=False) 
            
            # Orijinal dosya adını uzantısız olarak al (örneğin: 'orjinaldosya_1')
            key_name = file_path.stem 
            
            # Veriyi sözlüğe ekle
            merged_data[key_name] = data
            print(f"[+] Eklendi: {file_path.name}")
            
        except Exception as e:
            print(f"[-] Hata: {file_path.name} yüklenirken bir sorun oluştu. Detay: {e}")

    # Çıktı dosyasının tam yolu
    output_path = directory / output_filename
    
    # Birleştirilmiş veriyi kaydet
    torch.save(merged_data, output_path)
    print(f"\nİşlem tamamlandı! Birleştirilen dosya şuraya kaydedildi:\n{output_path}")

# ==========================================
# KULLANIM (Bu kısımları kendin belirleyebilirsin)
# ==========================================

# .pt dosyalarının bulunduğu dizin
HEDEF_DIZIN = "/eos/experiment/sndlhc/users/beturk/Data" 

# Oluşturulacak yeni dosyanın adı (dizin içine kaydedilecek)
CIKTI_DOSYA_ADI = "merged_all_data_2023.pt" 

if __name__ == "__main__":
    merge_pt_files(HEDEF_DIZIN, CIKTI_DOSYA_ADI)