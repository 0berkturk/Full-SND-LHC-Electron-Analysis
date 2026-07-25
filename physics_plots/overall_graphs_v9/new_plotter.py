import torch
import numpy as np
import matplotlib.pyplot as plt
import config

def plot_histograms():
    plt.figure(figsize=(10, 6))
    all_energies = sorted(list(set(data_dict.keys()).union(set(mc_dict.keys()))))
    
        for en in all_energies:
            color = self.get_color(en)
            
            # Plot Data (Solid line)
            if en in data_dict:
                vals = data_dict[en].float().cpu().numpy()
                if len(vals) > 0:
                    plt.hist(vals, bins=50, histtype='step', color=color, linestyle='-', 
                             linewidth=2, label=f'Data {en} GeV', density=True, alpha=0.8)
            
            # Plot MC (Dashed line)
            if en in mc_dict:
                vals = mc_dict[en].float().cpu().numpy()
                if len(vals) > 0:
                    plt.hist(vals, bins=50, histtype='step', color=color, linestyle='--', 
                             linewidth=2, label=f'MC {en} GeV', density=True, alpha=0.8)

        plt.yscale('log')
        plt.xlabel(feature_name, fontsize=12)
        plt.ylabel('Normalized Density', fontsize=12)
        plt.title(f'{feature_name} Distribution (MC vs Data)', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, which='both', linestyle='--', alpha=0.4)
        plt.tight_layout()
        
        os.makedirs(f"{self.outdir}/histograms", exist_ok=True)
        clean_name = feature_name.replace(' ', '_')
        plt.savefig(f"{self.outdir}/histograms/Hist_{clean_name}.png", dpi=300, bbox_inches='tight')
        plt.close()


def extract_energy_from_name(fname):
    """Helper to pull energy float directly from path."""
    match = re.search(r'_(\d+)GeV', fname)
    return float(match.group(1)) if match else 0.0

def feature_extractor(scifi):
    cut = scifi!=-999

    hits_per_station = cut.sum(dim=(1,3))
    total_hits = hits_per_station.sum(dim=1)

    clean_hits = scifi[cut]
    qdc_per_station = clean_hits.sum(dim=(1,3))
    total_qdc = qdc_per_station.sum(dim=1)

    features = {
        "Total Hits": total_hits,
        "Total QDC": total_qdc,
    }
    
    # Automatically add per-station metrics
    for i in range(5):
        features[f"Station {i+1} Hits"] = hits_per_station[:, i]
        features[f"Station {i+1} QDC"] = qdc_per_station[:, i]
        
    return features

def dataloader(scifi):
    datasets = []
    for test_data_name in datalist:
        path = test_data_name[1] if isinstance(test_data_name, list) else test_data_name
        datasets.append(SNDSparseDataset([0, path], perc=config.TOTAL_TEST_SIZE))

    for TB_RECALIBRATION_S2Y in dict_cuts.get("TB_RECALIBRATION_S2Y", [False]):
        for qdc_data in dict_cuts.get("qdc_threshold_value_scifi_data", [-10]):
            for qdc_mc in dict_cuts.get("qdc_threshold_value_scifi_mc", [0]):
                for t_win_data in dict_cuts.get("t_window_data", [(2.3, 0.5)]):
                    for t_win_mc in dict_cuts.get("t_window_mc", [(5, 5)]):

                        dir_data = f"S2Ycal{TB_RECALIBRATION_S2Y}_qdcD{qdc_data}_twinD{t_win_data[0]}_{t_win_data[1]}"
                        dir_mc = f"qdcM{qdc_mc}_twinM{t_win_mc[0]}_{t_win_mc[1]}"
                        outdir = f"plots/{dir_data}__VS__{dir_mc}"
                        
                        features_data = {} 
                        features_mc = {}

                        print(f"Processing cuts: {outdir}")
                        for i, test_data in enumerate(datalist):
                            dataset = datasets[i]
                            path = test_data[1] if isinstance(test_data, list) else test_data
                            is_mc = "MC" in path
                            energy = extract_energy_from_name(path)

                            # 1. Update Cuts dynamically
                            if is_mc:
                                dataset.update_hit_cuts(
                                    t_window_high_mc=t_win_mc[0], 
                                    t_window_low_mc=t_win_mc[1], 
                                    qdc_thresh_mc=qdc_mc
                                )
                            else:
                                dataset.update_hit_cuts(
                                    t_window_high_data=t_win_data[0], 
                                    t_window_low_data=t_win_data[1], 
                                    qdc_thresh_data=qdc_data,
                                    TB_RECALIBRATION_S2Y=TB_RECALIBRATION_S2Y
                                )
                            
                            loader = DataLoader(dataset, batch_size=config.TOTAL_TEST_SIZE, shuffle=False)
                            batch = next(iter(loader))
                            scifi_sig = batch[0] 

                            extracted_metrics = extract_event_features(scifi_sig)
                            target_dict = features_mc if is_mc else features_data
                            
                            for feat_name, feat_tensor in extracted_metrics.items():
                                if feat_name not in target_dict:
                                    target_dict[feat_name] = {}
                                target_dict[feat_name][energy] = feat_tensor
                        
                        
                            

