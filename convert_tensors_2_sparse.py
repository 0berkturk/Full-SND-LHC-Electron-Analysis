import torch
import os
from tqdm import tqdm

def convert_minimal_dense_to_sparse(file_list, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for filename in file_list:
        print(f"\nProcessing {filename}...")
        
        # 1. Load the old data (Strictly just SciFi and US)
        old_data = torch.load(filename)
        
        if isinstance(old_data, dict):
            sf_dense = old_data["scifi_signals"]
            us_dense = old_data["us_signals"]

        N_events = sf_dense.shape[0]
        print(f"Found {N_events} events. Converting to sparse...")
        
        # 2. Initialize the exact dictionary your new DataLoader needs
        new_data = {
            "scifi_indices": [], "scifi_signals": [], "scifi_hit_time": [],
            "us_indices": [], "us_signals": [], "us_signals_time": [],
            
            # Event-level cut tensors
            "scifi_notime_total_hits": torch.zeros(N_events, dtype=torch.float32),
            "scifi_notime_total_qdc": torch.zeros(N_events, dtype=torch.float32),
            "scifi_notime_hits_per_layer": torch.zeros((N_events, 5), dtype=torch.float32),
            "scifi_notime_qdc_per_layer": torch.zeros((N_events, 5), dtype=torch.float32),
            
            "us_notime_total_hits": torch.zeros(N_events, dtype=torch.float32),
            "us_notime_total_qdc": torch.zeros(N_events, dtype=torch.float32),
            "us_notime_hits_per_layer": torch.zeros((N_events, 5), dtype=torch.float32),
            "us_notime_qdc_per_layer": torch.zeros((N_events, 5), dtype=torch.float32),
        }

        # 3. Fast extraction loop
        for i in tqdm(range(N_events)):
            
            # --- SCIFI EXTRACTION ---
            sf_ev = sf_dense[i]
            sf_nonzero = torch.nonzero(sf_ev) 
            
            if len(sf_nonzero) > 0:
                sf_vals = sf_ev[sf_nonzero[:, 0], sf_nonzero[:, 1], sf_nonzero[:, 2]]
                
                new_data["scifi_indices"].append(sf_nonzero.tolist())
                new_data["scifi_signals"].append(sf_vals.tolist())
                
                # FAKE TIME GENERATION: Fill with 0.0 so the DataLoader doesn't crash!
                new_data["scifi_hit_time"].append([0.0] * len(sf_vals)) 
                
                # Physics Cuts Calculation
                new_data["scifi_notime_total_hits"][i] = len(sf_vals)
                new_data["scifi_notime_total_qdc"][i] = sf_vals.sum().item()
                
                for layer in range(5):
                    layer_mask = (sf_nonzero[:, 1] == layer)
                    new_data["scifi_notime_hits_per_layer"][i, layer] = layer_mask.sum().item()
                    new_data["scifi_notime_qdc_per_layer"][i, layer] = sf_vals[layer_mask].sum().item()
            else:
                new_data["scifi_indices"].append([])
                new_data["scifi_signals"].append([])
                new_data["scifi_hit_time"].append([])

            # --- US EXTRACTION ---
            us_ev = us_dense[i]
            us_nonzero = torch.nonzero(us_ev)
            
            if len(us_nonzero) > 0:
                us_vals = us_ev[us_nonzero[:, 0], us_nonzero[:, 1], us_nonzero[:, 2]]
                
                new_data["us_indices"].append(us_nonzero.tolist())
                new_data["us_signals"].append(us_vals.tolist())
                
                # FAKE TIME GENERATION
                new_data["us_signals_time"].append([0.0] * len(us_vals))
                
                new_data["us_notime_total_hits"][i] = len(us_vals)
                new_data["us_notime_total_qdc"][i] = us_vals.sum().item()
                
                for layer in range(5):
                    layer_mask = (us_nonzero[:, 1] == layer)
                    new_data["us_notime_hits_per_layer"][i, layer] = layer_mask.sum().item()
                    new_data["us_notime_qdc_per_layer"][i, layer] = us_vals[layer_mask].sum().item()
            else:
                new_data["us_indices"].append([])
                new_data["us_signals"].append([])
                new_data["us_signals_time"].append([])

        # 4. Save
        base_name = os.path.basename(filename)
        save_path = os.path.join(out_dir, base_name)
        
        torch.save(new_data, save_path)
        print(f"Saved optimized sparse file to: {save_path}")


file_list=[
#"/eos/user/b/beturk/snd/MonteCarlo/merge_electron_gun(biased)/train_combined_electrons_20_09_2025_v0.pt",

"/eos/user/b/beturk/snd/MonteCarlo/particle_gun_advanced_v1/merge_electron_gun_0_50/test_combined_electron_0_50_2025.pt",
"/eos/user/b/beturk/snd/MonteCarlo/particle_gun_advanced_v1/merge_electron_gun_0_50/train_combined_electron_0_50_2025_pt1.pt",
"/eos/user/b/beturk/snd/MonteCarlo/particle_gun_advanced_v1/merge_electron_gun_0_50/train_combined_electron_0_50_2025_pt2.pt",
"/eos/user/b/beturk/snd/MonteCarlo/particle_gun_advanced_v1/merge_electron_gun_0_50/val_combined_electron_0_50_2025.pt",

"/eos/user/b/beturk/snd/MonteCarlo/particle_gun_advanced_v1/merge_electron_gun_0_2000/test_combined_electron_0_2000_2025_v2.pt",
"/eos/user/b/beturk/snd/MonteCarlo/particle_gun_advanced_v1/merge_electron_gun_0_2000/train_combined_electron_0_2000_2025_v1.pt",
"/eos/user/b/beturk/snd/MonteCarlo/particle_gun_advanced_v1/merge_electron_gun_0_2000/val_combined_electron_0_2000_2025_v2.pt"]


convert_minimal_dense_to_sparse(file_list,"/eos/experiment/sndlhc/users/beturk/MC/old_PG")
