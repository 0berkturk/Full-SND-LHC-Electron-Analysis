import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np

def plot_genie_cross_sections(xml_filename, filters=None):
    """
    GENIE XML dosyasından tesir kesiti verilerini okur ve grafiğe döker.
    
    xml_filename: Okunacak XML dosyasının yolu.
    filters: Çizdirilmesini istediğin spline isimlerini içeren kelimeler (liste).
             Örn: ["QES", "DIS", "nu:14"] (Sadece içinde bu kelimeler geçenleri çizer).
             Eğer None ise, ilk bulduğu 5 tanesini örnek olarak çizer.
    """
    print(f"'{xml_filename}' dosyası okunuyor, lütfen bekleyin...")
    
    try:
        tree = ET.parse(xml_filename)
        root = tree.getroot()
    except Exception as e:
        print(f"Dosya okuma hatası: {e}")
        return

    plt.figure(figsize=(10, 7))
    
    plot_count = 0
    max_plots = 10 # Grafiğin çorbaya dönmemesi için maksimum 10 eğri çizdiriyoruz
    
    # XML içindeki tüm <spline> etiketlerini bul
    for spline in root.findall('.//spline'):
        spline_name = spline.get('name')
        
        # Eğer filtre verilmişse, ismin içinde filtrenin geçip geçmediğini kontrol et
        if filters:
            # Filtre listesindeki tüm kelimelerin ismin içinde geçmesini istiyorsak:
            match = all(f in spline_name for f in filters)
            if not match:
                continue
        
        # İlgili spline için Enerji ve Tesir Kesiti listelerini hazırla
        E_vals = []
        xsec_vals = []
        
        for knot in spline.findall('knot'):
            E = float(knot.find('E').text)
            xsec = float(knot.find('xsec').text)
            E_vals.append(E)
            xsec_vals.append(xsec)
            
        # Eğer veri boş değilse grafiğe ekle
        if len(E_vals) > 0:
            # İsmi çok uzun olduğu için grafikte düzgün görünmesi adına biraz kısaltıyoruz
            short_name = spline_name.replace("genie::", "").split("/Default/")[1] if "/Default/" in spline_name else spline_name
            
            plt.plot(E_vals, xsec_vals, label=short_name, linewidth=2)
            plot_count += 1
            
        if plot_count >= max_plots:
            print(f"\nMaksimum çizim limitine ({max_plots}) ulaşıldı.")
            break

    if plot_count == 0:
        print("Filtrelerine uygun hiçbir tesir kesiti (spline) bulunamadı!")
        return

    # Grafik Ayarları
    plt.title("GENIE Neutrino Cross-Sections", fontsize=14, fontweight='bold')
    plt.xlabel("Neutrino Energy $E_{\\nu}$ [GeV]", fontsize=12)
    plt.ylabel("Cross Section $\\sigma$ [$cm^2/nucleon$]", fontsize=12)
    
    # Tesir kesitleri genelde çok büyük farklara sahip olduğu için Log-Log scale kullanılır
    plt.xscale('log')
    plt.yscale('log')
    
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(loc='best', fontsize=8, framealpha=0.9)
    plt.tight_layout()
    
    # Grafiği kaydet ve göster
    plt.savefig("cross_sections_plot.png", dpi=300)
    print("Grafik 'cross_sections_plot.png' olarak kaydedildi.")
    plt.show()

if __name__ == "__main__":
    # Senin yüklediğin dosyanın tam adını buraya yazıyoruz
    xml_file = "genie_splines_GENIE_v32_SNDG18_02a_00_000.xml"
    
    # --- FİLTRELEME ÖRNEKLERİ ---
    # Sadece Müon Nötrinosu (nu:14) Charged Current (CC) Quasi-Elastic (QES) etkileşimini görmek istersen:
    # filters = ["nu:14", "Weak[CC]", "QES"]
    
    # Bir önceki sorunda sorduğun Quasi-Elastic (QES) ve Deep Inelastic (DIS) kıyaslaması için
    # filtreleri biraz daha geniş tutabilirsin. (Örneğin sadece QES görmek için:)
    
    aranacak_terimler = ["Weak[CC]", "QES", "nu:14"] 
    
    plot_genie_cross_sections(xml_file, filters=aranacak_terimler)