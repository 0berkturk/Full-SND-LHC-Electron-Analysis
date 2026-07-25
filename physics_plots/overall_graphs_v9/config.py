"""
config.py
Master configuration file. 
Tamamen yeni TestBeamPlotter mimarisine göre sadeleştirilmiştir.
"""

from cuts import *  # cuts.py içindeki sınırları dataloader için içeri aktarır

# ==========================================
# 1. GENERAL PROCESSING SETTINGS (Genel Ayarlar)
# ==========================================
qdc_threshold_value_scifi_all_same_final_mc=-5
noise_sigma=1
q_max=13
qdc_threshold_value_scifi_mc=0 ## en başta dataloaderda uygulanan cut. aşağıdakiler ise dataloaderdan çıktıktan sonra uygulanıyor
qdc_threshold_value_scifi_data=-5#qdc_threshold_value_scifi_all_same_final_mc

#threshold -1 
#sigma 2

plot_dir="plots_only_electrons_tuning_new_q19"

SHOWER_WIDTH        = None   # Sadece merkeze yakın kanalları tutar (örn: 512). None ise uygulanmaz.
USE_HIGHEST_N_LAYER = None   # Sadece en yüksek hite sahip N layer'ı tutar. None ise uygulanmaz.
LOGARITHMIC_SCALING = False
IS_MC_TUNING        = True   # Zaman grafiklerinin (2D Time Diff) çalışması için True kalmalıdır!


EN_MIN=0
EN_MAX=2000

TB_RECALIBRATION_S2Y=True
TB_USE_SAME_XY_S2=False

# Enerji Kalibrasyon Çarpanları (Makaleden)
SCIFI_QDC_2_GEV = 1          # Makale değeri: 0.059
US_DS_QDC_2_GEV = 1          # Makale değeri: 0.0145

# ==========================================
# 2. DATALOADER KEYS (Hangi dedektörler okunacak?)
# ==========================================
# İhtiyaç duyduğunda ilgili dedektörü True yapman yeterlidir
USE_US = False 
USE_DS = False 

# Anahtarlar otomatik olarak inşa edilir
KEYS_FOR_DATA_LOADER = ["scifi_signals"]
if IS_MC_TUNING:
    KEYS_FOR_DATA_LOADER.append("scifi_time_diff")
if USE_US:
    KEYS_FOR_DATA_LOADER.append("us_signals")
if USE_DS:
    KEYS_FOR_DATA_LOADER.append("ds")

print(f"Active Dataloader Keys: {KEYS_FOR_DATA_LOADER}")

# ==========================================
# 3. PLOTTING METADATA (Çizim Parametreleri)
# ==========================================
TOTAL_TEST_SIZE = 5000  # Dataloader'ın tek seferde çekeceği maksimum event sayısı

# Yeni Plotter sistemi renkleri ve enerjileri buradan otomatik eşleştirir.
# (Ekstra enerji eklersen buraya renk ekleyebilirsin)
beam_en_list = [50, 100, 150, 200, 250, 300]
color_list   = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

