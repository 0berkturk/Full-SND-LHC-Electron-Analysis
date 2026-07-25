import os
import glob
import torch
import config 
from dl_recon_core_sparse.data_loader import SNDSparseDataset
import numpy as np
# Çekilecek özelliklerin listesi
keys_list = [
    "scifi_hitx_in_64r", "scifi_hity_in_64r", "scifi_hitx_in_128r", "scifi_hity_in_128r",
    "max_total_hdw", "station_hdw", "max_hor_hdw", "max_ver_hdw",
    "scifi_notime_total_hits", "scifi_notime_total_qdc",
    "scifi_notime_hits_per_layer", "scifi_notime_qdc_per_layer",
    "scifi_05usualtime_total_hits", "scifi_05usualtime_total_qdc",
    "scifi_05usualtime_hits_per_layer", "scifi_05usualtime_qdc_per_layer",
    "scifi_05_18_total_hits", "scifi_05_18_total_qdc",
    "scifi_05_18_hits_per_layer", "scifi_05_18_qdc_per_layer",
    "scifi_05_22_total_hits", "scifi_05_22_total_qdc",
    "scifi_05_22_hits_per_layer", "scifi_05_22_qdc_per_layer",
    "scifi_05_23_total_hits", "scifi_05_23_total_qdc",
    "scifi_05_23_hits_per_layer", "scifi_05_23_qdc_per_layer",
    "us_notime_total_hits", "us_notime_total_qdc",
    "us_notime_hits_per_layer", "us_notime_qdc_per_layer",
    "us_3usualtime_total_hits", "us_3usualtime_total_qdc",
    "us_3usualtime_hits_per_layer", "us_3usualtime_qdc_per_layer",
    "dsh_notime_total_hits", "dsh_notime_total_qdc",
    "dsh_notime_hits_per_layer", "dsh_notime_qdc_per_layer",
    "dsv_notime_total_hits", "dsv_notime_total_qdc",
    "dsv_notime_hits_per_layer", "dsv_notime_qdc_per_layer",
    "run_id", "event_number", "event_time"
]

def extract_passed_data_simple(file_names):
    for file_path in file_names:
        print(f"Processing file: {file_path}")

        file_dir = os.path.dirname(file_path)
        file_basename = os.path.splitext(os.path.basename(file_path))[0]
        
        # Sadece extract edilmiş veriler için ayrı bir klasör
        output_dir = file_dir + "_extracted_props"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{file_basename}_extracted.pt")

        # Dosya zaten varsa hiç dataseti yüklemeden direkt atla
        if os.path.exists(save_path):
            print(f"File {save_path} already exists, skipping.")
            continue

        # Dataset'i başlat (Bu işlem __init__ içindeki tüm cut'ları uygulayacaktır)
        ith_dataset = SNDSparseDataset([0, file_path], EN_MIN=0, EN_MAX=20000, is_lhcdata=True)
        
        if ith_dataset.num_events == 0:
            print("Zero data passed cuts, skipping.")
            continue

        # 1. Dataset'in tuttuğu orijinal ham veri
        raw_data = ith_dataset.data
        
        # 2. Dataset'in içine eklediğimiz, sadece cut'tan geçenlerin indexleri
        idx = ith_dataset.valid_indices
        #print("idx ",idx)
        print("length", len(idx))
        
        extracted_dict = {}
        
        for key in keys_list:
            if key in raw_data:
                
                extracted_dict[key] = torch.from_numpy(np.array(raw_data[key]))[idx]
                #print(key,extracted_dict[key].shape)

        extracted_dict["idx"] = idx

        torch.save(extracted_dict, save_path)
        print(f"Saved extracted properties to: {save_path}\n")

if __name__ == "__main__":
    year="2023"
    target_files = glob.glob(f"/eos/experiment/sndlhc/users/beturk/Data/PT/Data_{year}/batched_data_{year}_*.pt")

    print(target_files)
    extract_passed_data_simple(target_files)