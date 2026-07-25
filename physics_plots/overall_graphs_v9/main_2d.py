import config
import sys
import os
import glob

# Yeni yazdığımız çizim kütüphanesini içeri aktarıyoruz
from test_beam_plotting import RUN_FINAL_MEGA_COMP_ALL_V8, LOAD_PLOT_ALL_2D_COMBINATIONS

def main():
    # ---------------------------------------------------------
    # 1. DOSYA LİSTELERİNİ HAZIRLAMA (50, 100, 150 GeV vb.)
    # ---------------------------------------------------------
    TEST_DATA_DIR_DATA = []
    

    for energy in  [50,100,150,200,250,300]:
        #TEST_DATA_DIR_DATA.append([1,glob.glob(f"/eos/experiment/sndlhc/users/beturk/TB/TB_Small/small_scifi_us_ds_2024_electrons_{energy}GeV_*")[0]])
        TEST_DATA_DIR_DATA.append([1,glob.glob(f"/eos/experiment/sndlhc/users/beturk/TB/TB_MC_New/Sparse_Datasets_2024/shuffled_smaller_MCEB_TB_MC_2024_electron_{energy}GeV*")[0]])




    # ---------------------------------------------------------
    # 3. YENİ 2D KOMBİNASYONLARINI ÇALIŞTIRMA (Kanal vs QDC vs Zaman)
    # ---------------------------------------------------------
    print("\n===========================================")
    print("STARTING 2D COMBINATIONS (Heatmaps)")
    print("===========================================\n")
    
    # 2D grafiklerin kaydedileceği ana klasör
    outdir_2d = "2d_hist_MC_q13"
    
    # Sadece 2D fonksiyonunu çağırıyoruz. File list olarak yukarıda oluşturduğumuz listeyi veriyoruz.
    LOAD_PLOT_ALL_2D_COMBINATIONS(
        config=config,
        outdir=outdir_2d,
        file_list=TEST_DATA_DIR_DATA,
        N=config.TOTAL_TEST_SIZE,  # İşlenecek event sayısı
        layers=[1, 2],          # İstediğin layerlar
        planes=[0, 1],             # İstediğin planeler (Oryantasyon)
        thresholds=[-3],           # QDC Threshold denemeleri
        time_window_max_mc=[1],
        time_window_min_mc=[-1],
        time_window_max_data=[2.3],
        time_window_min_data=[-0.5]
    )

if __name__ == "__main__":
    main()