import os
import re
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader

# Import your dataloader classes (Ensure path is correct)
from dl_recon_core_sparse.data_loader import SNDSparseDataset

# ==========================================
# HELPER: LABEL & COLOR GENERATORS
# ==========================================

def get_base_label(path, idx, custom_labels):
    """Extracts base configuration (Particle, Year, Wall) to ensure consistent grouping."""
    if custom_labels and idx < len(custom_labels) and custom_labels[idx]:
        return custom_labels[idx]
        
    path_lower = str(path).lower()
    
    # Identify Particle
    particle = "Electrons" if "electron" in path_lower else "Pions" if "pion" in path_lower else "Unknown"
    
    # Identify Year
    year = "2023" if "2023" in path_lower else "2024" if "2024" in path_lower else ""
    
    # Identify Wall (Electrons and 'W' are 2W; otherwise default to Fe structure logic)
    if particle == "Electrons" or "_w_" in path_lower or "_w." in path_lower or "2w" in path_lower:
        wall = "2W"
    else:
        match = re.search(r'(\d*)fe', path_lower)
        if match:
            prefix = match.group(1)
            wall = f"{prefix}Fe" if prefix else "Fe"
        else:
            wall = "Fe" # Fallback for 24 pions without explicit wall string
            
    return f"{particle} {year} {wall}".strip()

def hdw_all_fast_conv(scifi_qdc, us=None, ds=None, delta_ch=1):
    """
    scifi_qdc: (N, 2, 5, 1536)
    delta_ch: Neighborhood width for horizontal/vertical definitions
    """
    N = scifi_qdc.shape[0]
    hits = (scifi_qdc != -999).float()    

    kernel_size = 2 * delta_ch + 1
    kernel = torch.ones(1, 1, kernel_size, device=scifi_qdc.device)
    kernel[:, :, delta_ch] = 0.0      

    plane_hdw = torch.zeros(N, 2, 5, device=scifi_qdc.device)

    for station in range(5):      
        for plane in range(2):            
            x = hits[:, plane, station, :]     
            x = x.unsqueeze(1)                 

            neighbors = F.conv1d(
                x,
                kernel,
                padding=delta_ch
            ).squeeze(1)                        

            wi = neighbors * hits[:, plane, station, :] 
            plane_hdw[:, plane, station] = wi.sum(dim=-1)

    station_hdw = plane_hdw.sum(dim=1)          
    event_hdw, max_index = station_hdw.max(dim=1)

    idx = max_index.view(N, 1, 1).expand(-1, 2, 1)   

    plane_hdw_max = plane_hdw.gather(dim=2, index=idx).squeeze(2)

    hor_hdw = plane_hdw_max[:, 0]
    ver_hdw = plane_hdw_max[:, 1]

    return station_hdw, event_hdw, hor_hdw, ver_hdw


# ==========================================
# 1. FEATURE EXTRACTOR
# ==========================================
def extract_event_features(scifi_sig, us=None):
    valid_mask = (scifi_sig != -999)
    hits_per_station = torch.sum(valid_mask, dim=(1, 3)) 
    total_hits = torch.sum(hits_per_station, dim=1)      
    
    clean_qdc = torch.where(valid_mask, scifi_sig, torch.tensor(0.0, device=scifi_sig.device))    
    qdc_per_station = torch.sum(clean_qdc, dim=(1, 3))   
    total_qdc = torch.sum(qdc_per_station, dim=1)      

    hdw_station_list, max_hdw_station, _, _ = hdw_all_fast_conv(scifi_sig, delta_ch=60)
    total_hdw = hdw_station_list.sum(1)

    topK_values_qdc, _ = torch.topk(qdc_per_station, k=2, dim=1)
    frac_abs_qdc = torch.log(torch.where(topK_values_qdc[:, 1] > 0, topK_values_qdc[:, 0] / topK_values_qdc[:, 1], torch.tensor(1.0, device=scifi_sig.device)))

    topK_values_hits, _ = torch.topk(hits_per_station, k=2, dim=1)
    frac_abs_hits = torch.log(torch.where(topK_values_hits[:, 1] > 0, topK_values_hits[:, 0].float() / topK_values_hits[:, 1].float(), torch.tensor(1.0, device=scifi_sig.device)))

    features = {
        "Total Hits": total_hits,
        "Total QDC": total_qdc,
        "Total Sum of HDW": total_hdw,
        "Max. HDW Station": max_hdw_station,
        "Log of Fraction Abs QDC": frac_abs_qdc,
        "Log of Fraction Abs Hits": frac_abs_hits
    }

    for i in range(5):
        features[f"Station {i+1} Hits"] = hits_per_station[:, i]
        features[f"Station {i+1} QDC"] = qdc_per_station[:, i]
        features[f"Station {i+1} HDW"] = hdw_station_list[:, i]
        
    return features

# ==========================================
# 2. SMART PLOTTER CLASS
# ==========================================
class TestBeamPlotter:
    def __init__(self, config, outdir):
        self.config = config
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        
        self.color_map = {}
        self.default_colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#17becf']

    def get_color(self, base_label):
        """Automatically assigns color exclusively based on the configuration label (Particle, Year, Wall)."""
        if base_label not in self.color_map:
            self.color_map[base_label] = self.default_colors[len(self.color_map) % len(self.default_colors)]
        return self.color_map[base_label]

    def _generate_legend(self, data_dict, mc_dict):
        """Simplified legend: Only shows Colors for Walls/Particles, and Lines for Data/MC."""
        legend_elements = [
            Line2D([0], [0], color='k', linestyle='-', label='Data'),
            Line2D([0], [0], color='k', linestyle='--', label='MC')
        ]
        
        all_configs = {}
        for d in (data_dict, mc_dict):
            for e, lbl, color in d.keys():
                all_configs[lbl] = color
                
        # Add color swatches for each base label (e.g., "Pions 2024 2W")
        for lbl in sorted(all_configs.keys()):
            legend_elements.append(Line2D([0], [0], marker='s', color='w', label=lbl, markerfacecolor=all_configs[lbl], markersize=10))
            
        return legend_elements

    def plot_histograms(self, feature_name, data_dict, mc_dict):
        plt.figure(figsize=(10, 6))
        
        for (en, label, color), vals in data_dict.items():
            v = vals.float().cpu().numpy()
            if len(v) > 0:
                hist_kwargs = {'bins': 50, 'range': None}
                if np.max(v) == np.min(v): 
                    hist_kwargs['bins'] = 1
                    hist_kwargs['range'] = (v[0] - 0.5, v[0] + 0.5)

                # Histograms keep verbose legends (with Energy) as requested for plots "except the histograms"
                plt.hist(v, histtype='step', color=color, linestyle='-', 
                         linewidth=2, label=f'Data {label} {en} GeV', density=True, alpha=0.8, **hist_kwargs)
                         
        for (en, label, color), vals in mc_dict.items():
            v = vals.float().cpu().numpy()
            if len(v) > 0:
                hist_kwargs = {'bins': 50, 'range': None}
                if np.max(v) == np.min(v):
                    hist_kwargs['bins'] = 1
                    hist_kwargs['range'] = (v[0] - 0.5, v[0] + 0.5)

                plt.hist(v, histtype='step', color=color, linestyle='--', 
                         linewidth=2, label=f'MC {label} {en} GeV', density=True, alpha=0.8, **hist_kwargs)

        plt.yscale('log')
        plt.xlabel(feature_name, fontsize=12)
        plt.ylabel('Normalized Density', fontsize=12)
        plt.title(f'{feature_name} Distribution', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, which='both', linestyle='--', alpha=0.4)
        plt.tight_layout()
        
        os.makedirs(f"{self.outdir}/histograms", exist_ok=True)
        clean_name = feature_name.replace(' ', '_')
        plt.savefig(f"{self.outdir}/histograms/Hist_{clean_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_mean_vs_energy(self, feature_name, data_dict, mc_dict):
        plt.figure(figsize=(9, 6))
        
        for (en, label, color), vals in data_dict.items():
            vals = vals.float()
            if len(vals) == 0: continue
            mean_val, std_val = vals.mean().item(), vals.std().item()
            plt.errorbar(en, mean_val, yerr=std_val, fmt='-o', color=color, 
                         markersize=6, capsize=4, elinewidth=1.5, alpha=0.8)
                         
        for (en, label, color), vals in mc_dict.items():
            vals = vals.float()
            if len(vals) == 0: continue
            mean_val, std_val = vals.mean().item(), vals.std().item()
            offset_en = en * 1.03 
            plt.errorbar(offset_en, mean_val, yerr=std_val, fmt='--o', color=color, 
                         markerfacecolor='white', markeredgewidth=1.5, markersize=6, capsize=4, elinewidth=1.5, alpha=0.8)

        plt.xlabel('Beam Energy [GeV]', fontsize=12)
        plt.ylabel(f'Mean {feature_name}', fontsize=12)
        plt.title(f'Mean {feature_name} vs. Beam Energy', fontsize=14)
        
        plt.legend(handles=self._generate_legend(data_dict, mc_dict), bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        os.makedirs(f"{self.outdir}/means", exist_ok=True)
        clean_name = feature_name.replace(' ', '_')
        plt.savefig(f"{self.outdir}/means/Mean_{clean_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_combined_layers_mean(self, features_data, features_mc, metric="QDC", layers=[2, 3, 4]):
        plt.figure(figsize=(10, 7))
        all_configs = {}
        
        for layer in layers:
            feat_name = f"Station {layer} {metric}"
            
            # --- LAYER X DATA ---
            if feat_name in features_data:
                d_dict = features_data[feat_name]
                labels = set([lbl for en, lbl, color in d_dict.keys()])
                for lbl in labels:
                    ens = sorted([e for e, l, c in d_dict.keys() if l == lbl])
                    l_color = next((c for e, l, c in d_dict.keys() if l == lbl), 'k')
                    all_configs[lbl] = l_color
                    
                    means, stds = [], []
                    for e in ens:
                        for k, v in d_dict.items():
                            if k[0] == e and k[1] == lbl:
                                means.append(v.float().mean().item())
                                stds.append(v.float().std().item())
                    # Solid line for data, uniform shape (o) for everything
                    plt.errorbar(ens, means, yerr=stds, fmt='-o', color=l_color,
                                 markersize=6, capsize=4, elinewidth=1.5, alpha=0.85)
            
            # --- LAYER X MC ---
            if feat_name in features_mc:
                m_dict = features_mc[feat_name]
                labels = set([lbl for en, lbl, color in m_dict.keys()])
                for lbl in labels:
                    ens = sorted([e for e, l, c in m_dict.keys() if l == lbl])
                    l_color = next((c for e, l, c in m_dict.keys() if l == lbl), 'k')
                    all_configs[lbl] = l_color
                    
                    means, stds = [], []
                    for e in ens:
                        for k, v in m_dict.items():
                            if k[0] == e and k[1] == lbl:
                                means.append(v.float().mean().item())
                                stds.append(v.float().std().item())
                    
                    offset_ens = [e * 1.03 for e in ens]
                    # Dashed line for MC, uniform shape (o) for everything
                    plt.errorbar(offset_ens, means, yerr=stds, fmt='--o', color=l_color, markerfacecolor='white', markeredgewidth=1.5,
                                 markersize=6, capsize=4, elinewidth=1.5, alpha=0.85)

        plt.xlabel('Beam Energy [GeV]', fontsize=12)
        plt.ylabel(f'Mean {metric}', fontsize=12)
        plt.title(f'Mean {metric} vs. Beam Energy (Layers {", ".join(map(str, layers))})', fontsize=14)
        
        # Build the exact legend requested
        legend_elements = [
            Line2D([0], [0], color='k', linestyle='-', label='Data'),
            Line2D([0], [0], color='k', linestyle='--', label='MC')
        ]
        for lbl in sorted(all_configs.keys()):
            legend_elements.append(Line2D([0], [0], marker='s', color='w', label=lbl, markerfacecolor=all_configs[lbl], markersize=10))
            
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        os.makedirs(f"{self.outdir}/means_combined", exist_ok=True)
        plt.savefig(f"{self.outdir}/means_combined/Combined_Layers_Mean_{metric}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _get_highest_bin(self, vals_tensor, num_bins=50):
        vals = vals_tensor.float().cpu().numpy()
        if len(vals) == 0:
            return 0.0, 0.0
            
        if np.max(vals) == np.min(vals):
            return float(vals[0]), 0.0
            
        counts, bin_edges = np.histogram(vals, bins=num_bins)
        max_bin_idx = np.argmax(counts)
        highest_bin_val = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2.0
        std_dev = vals_tensor.float().std().item()
        
        return highest_bin_val, std_dev

    def plot_highest_bin_vs_energy(self, feature_name, data_dict, mc_dict):
        plt.figure(figsize=(9, 6))
        
        for (en, label, color), vals in data_dict.items():
            peak_val, peak_err = self._get_highest_bin(vals)
            if peak_val == 0.0: continue
            plt.errorbar(en, peak_val, yerr=peak_err, fmt='-o', color=color, 
                         markersize=6, capsize=4, elinewidth=1.5, alpha=0.8)
                         
        for (en, label, color), vals in mc_dict.items():
            peak_val, peak_err = self._get_highest_bin(vals)
            if peak_val == 0.0: continue
            offset_en = en * 1.03
            plt.errorbar(offset_en, peak_val, yerr=peak_err, fmt='--o', color=color, 
                         markerfacecolor='white', markeredgewidth=1.5, markersize=6, capsize=4, elinewidth=1.5, alpha=0.8)

        plt.xlabel('Beam Energy [GeV]', fontsize=12)
        plt.ylabel(f'Highest Bin {feature_name}', fontsize=12)
        plt.title(f'Highest Bin (Peak) {feature_name} vs. Beam Energy', fontsize=14)
        
        plt.legend(handles=self._generate_legend(data_dict, mc_dict), bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        os.makedirs(f"{self.outdir}/highest_bins", exist_ok=True)
        clean_name = feature_name.replace(' ', '_')
        plt.savefig(f"{self.outdir}/highest_bins/HighestBin_{clean_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_combined_layers_highest_bin(self, features_data, features_mc, metric="QDC", layers=[2, 3, 4]):
        plt.figure(figsize=(10, 7))
        all_configs = {}
        
        for layer in layers:
            feat_name = f"Station {layer} {metric}"
            
            # --- LAYER X DATA ---
            if feat_name in features_data:
                d_dict = features_data[feat_name]
                labels = set([lbl for en, lbl, c in d_dict.keys()])
                for lbl in labels:
                    ens = sorted([e for e, l, c in d_dict.keys() if l == lbl])
                    l_color = next((c for e, l, c in d_dict.keys() if l == lbl), 'k')
                    all_configs[lbl] = l_color
                    
                    peaks_and_errs = []
                    for e in ens:
                        for k, v in d_dict.items():
                            if k[0] == e and k[1] == lbl:
                                peaks_and_errs.append(self._get_highest_bin(v))
                    
                    plt.errorbar(ens, [p[0] for p in peaks_and_errs], yerr=[p[1] for p in peaks_and_errs], 
                                 fmt='-o', color=l_color, markersize=6, capsize=4, elinewidth=1.5, alpha=0.85)
            
            # --- LAYER X MC ---
            if feat_name in features_mc:
                m_dict = features_mc[feat_name]
                labels = set([lbl for en, lbl, c in m_dict.keys()])
                for lbl in labels:
                    ens = sorted([e for e, l, c in m_dict.keys() if l == lbl])
                    l_color = next((c for e, l, c in m_dict.keys() if l == lbl), 'k')
                    all_configs[lbl] = l_color
                    
                    peaks_and_errs = []
                    for e in ens:
                        for k, v in m_dict.items():
                            if k[0] == e and k[1] == lbl:
                                peaks_and_errs.append(self._get_highest_bin(v))

                    offset_ens = [e * 1.03 for e in ens]
                    plt.errorbar(offset_ens, [p[0] for p in peaks_and_errs], yerr=[p[1] for p in peaks_and_errs], 
                                 fmt='--o', color=l_color, markerfacecolor='white', markeredgewidth=1.5,
                                 markersize=6, capsize=4, elinewidth=1.5, alpha=0.85)

        plt.xlabel('Beam Energy [GeV]', fontsize=12)
        plt.ylabel(f'Highest Bin {metric}', fontsize=12)
        plt.title(f'Highest Bin (Peak) {metric} vs. Beam Energy (Layers {", ".join(map(str, layers))})', fontsize=14)
        
        legend_elements = [
            Line2D([0], [0], color='k', linestyle='-', label='Data'),
            Line2D([0], [0], color='k', linestyle='--', label='MC')
        ]
        for lbl in sorted(all_configs.keys()):
            legend_elements.append(Line2D([0], [0], marker='s', color='w', label=lbl, markerfacecolor=all_configs[lbl], markersize=10))
            
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        os.makedirs(f"{self.outdir}/highest_bins_combined", exist_ok=True)
        plt.savefig(f"{self.outdir}/highest_bins_combined/Combined_Layers_HighestBin_{metric}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_2d_hist(self, x_data, y_data, out_name, xlabel, ylabel, title, subfolder="2d_combinations", bins_x=50, bins_y=50):
        plt.figure(figsize=(9, 7))
        if torch.is_tensor(x_data): x_data = x_data.cpu().numpy()
        if torch.is_tensor(y_data): y_data = y_data.cpu().numpy()
        
        cmap = plt.get_cmap('plasma')
        cmap.set_under('white')
        
        plt.hist2d(x_data, y_data, bins=[bins_x, np.linspace(-2, 15, 50)], cmap=cmap, alpha=0.85, cmin=1)  
        
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14)
        plt.colorbar(label='Counts')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        target_dir = f"{self.outdir}/{subfolder}"
        os.makedirs(target_dir, exist_ok=True)
        plt.savefig(f"{target_dir}/{out_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

# ==========================================
# 3. MEGA COMPARISON RUNNER
# ==========================================
def extract_energy_from_name(fname):
    match = re.search(r'_(\d+)GeV', fname)
    return float(match.group(1)) if match else 0.0

def RUN_FINAL_MEGA_COMP_ALL_V8(config, datalist, dict_cuts, custom_labels=None, custom_colors=None):
    print("Initializing Datasets...")
    datasets = []
    for test_data_name in datalist:
        path = test_data_name[1] if isinstance(test_data_name, list) else test_data_name
        datasets.append(SNDSparseDataset([0, path], perc=config.TOTAL_TEST_SIZE))

    for TB_RECALIBRATION_S2Y in dict_cuts.get("TB_RECALIBRATION_S2Y", [False]):
        for qdc_data in dict_cuts.get("qdc_threshold_value_scifi_data", [-10]):
            for qdc_mc in dict_cuts.get("qdc_threshold_value_scifi_mc", [0]):
                for t_win_data in dict_cuts.get("t_window_data", [(2.3, 0.5)]):
                    for t_win_mc in dict_cuts.get("t_window_mc", [(5, 5)]):
                        for noise_sigma in dict_cuts.get("noise_sigma", [0]):
                            for q_max in dict_cuts.get("q_max", [13]):

                                dir_data = f"S2Ycal{TB_RECALIBRATION_S2Y}_qdcD{qdc_data}_twinD{t_win_data[0]}_{t_win_data[1]}"
                                dir_mc = f"qdcM{qdc_mc}_twinM{t_win_mc[0]}_{t_win_mc[1]}_{noise_sigma}_{q_max}"
                                outdir = f"{config.plot_dir}/{dir_data}__VS__{dir_mc}"
                                
                                plotter = TestBeamPlotter(config, outdir)
                                
                                features_data = {} 
                                features_mc = {}

                                print(f"Processing cuts: {outdir}")
                                for i, test_data in enumerate(datalist):
                                    dataset = datasets[i]
                                    path = test_data[1] if isinstance(test_data, list) else test_data
                                    is_mc = "MC" in path
                                    energy = extract_energy_from_name(path)
                                    
                                    # Fetch dynamic group label and specific config color
                                    base_label = get_base_label(path, i, custom_labels)
                                    
                                    # Use overriding custom color if provided, otherwise standard grouped color
                                    if custom_colors and i < len(custom_colors) and custom_colors[i]:
                                        color = custom_colors[i]
                                    else:
                                        color = plotter.get_color(base_label)

                                    if is_mc:
                                        dataset.update_hit_cuts(
                                            t_window_high_mc=t_win_mc[0], 
                                            t_window_low_mc=t_win_mc[1], 
                                            qdc_thresh_mc=qdc_mc,
                                            noise_sigma=noise_sigma,
                                            q_max=q_max
                                        )
                                    else:
                                        dataset.update_hit_cuts(
                                            t_window_high_data=t_win_data[0], 
                                            t_window_low_data=t_win_data[1], 
                                            qdc_thresh_data=qdc_data,
                                            TB_RECALIBRATION_S2Y=TB_RECALIBRATION_S2Y
                                        )

                                    loader = DataLoader(dataset, batch_size=config.TOTAL_TEST_SIZE, shuffle=False, num_workers=4, pin_memory=True)
                                    try:
                                        batch = next(iter(loader))
                                        scifi_sig = batch[0]
                                    except StopIteration:
                                        continue
                                    
                                    extracted_metrics = extract_event_features(scifi_sig)
                                    target_dict = features_mc if is_mc else features_data
                                    
                                    for feat_name, feat_tensor in extracted_metrics.items():
                                        if feat_name not in target_dict:
                                            target_dict[feat_name] = {}
                                        target_dict[feat_name][(energy, base_label, color)] = feat_tensor

                                all_features = set(features_data.keys()).union(set(features_mc.keys()))
                                
                                for feat_name in all_features:
                                    d_dict = features_data.get(feat_name, {})
                                    m_dict = features_mc.get(feat_name, {})
                                    
                                    plotter.plot_histograms(feat_name, d_dict, m_dict)
                                    plotter.plot_mean_vs_energy(feat_name, d_dict, m_dict)
                                    plotter.plot_highest_bin_vs_energy(feat_name, d_dict, m_dict)

                                plotter.plot_combined_layers_mean(features_data, features_mc, metric="QDC", layers=[2, 3, 4])
                                plotter.plot_combined_layers_mean(features_data, features_mc, metric="Hits", layers=[2, 3, 4])
                                plotter.plot_combined_layers_highest_bin(features_data, features_mc, metric="QDC", layers=[2, 3, 4])
                                plotter.plot_combined_layers_highest_bin(features_data, features_mc, metric="Hits", layers=[2, 3, 4])
                                plotter.plot_combined_layers_mean(features_data, features_mc, metric="HDW", layers=[2, 3, 4])
                                plotter.plot_combined_layers_highest_bin(features_data, features_mc, metric="HDW", layers=[2, 3, 4])
                                print(f"Finished plotting for {outdir}\n")


def LOAD_PLOT_ALL_2D_COMBINATIONS(config, outdir, file_list, N, layers=[0, 1, 2], planes=[0, 1], thresholds=[-3], time_window_max_mc=[1], time_window_min_mc=[-1], time_window_max_data=[0.5], time_window_min_data=[-0.5], custom_labels=None, custom_colors=None):
    
    print(f"\n--- Starting 2D Combinations for {len(file_list)} files ---")
    plotter = TestBeamPlotter(config, outdir)
    
    from dl_recon_core_sparse.data_loader import data_loader
    
    for k, file_entry in enumerate(file_list):
        fname = file_entry[1] if isinstance(file_entry, list) else file_entry
        
        match = re.search(r'_(\d+)GeV_', fname)
        energy_str = f"{match.group(1)}GeV" if match else "UnknownEnergy"
        
        print(f"Generating 2D plots for: {os.path.basename(fname)} (Energy: {energy_str})")
        
        dataloader, _, _ = data_loader([[0, fname]], N, N, "cpu", is_train=False)
        
        try:
            batch = next(iter(dataloader))
        except StopIteration:
            print(f"WARNING: No valid events found in {fname} after dataloader cuts. Skipping.")
            continue
            
        scifi_sig = batch[0]
        scifi_hittime_diff = batch[1] if len(batch) > 1 else None
        
        if scifi_hittime_diff is None:
            print("WARNING: Time diff tensor missing! Ensure config.IS_MC_TUNING = True")
            continue
              
        is_mc = "MC" in fname
        base_label = get_base_label(fname, k, custom_labels)
        
        if is_mc:
            time_window_min = time_window_min_mc
            time_window_max = time_window_max_mc
        else:
            time_window_min = time_window_min_data
            time_window_max = time_window_max_data
            
        for plane in planes:
            for layer in layers:
                for threshold in thresholds:
                    for i in range(len(time_window_max)):
                        
                        t_min = time_window_min[i]
                        t_max = time_window_max[i]
                        
                        hist_qdc = scifi_sig[:, plane, layer, :]
                        hist_time = scifi_hittime_diff[:, plane, layer, :]

                        cut = (hist_qdc != -999) & (hist_qdc > threshold) & (hist_time > t_min) & (hist_time < t_max)
                        
                        if not cut.any():
                            continue 
                            
                        filtered_qdc = hist_qdc[cut]
                        filtered_time_diff = hist_time[cut]

                        coords = cut.nonzero()
                        hist_sipm_indices = coords[:, -1]

                        timing_name = f"T_{t_max}_{t_min}"
                        base_name = f"{energy_str}_L{layer}_P{plane}_Thresh{threshold}_{timing_name}_{base_label}_{config.qdc_threshold_value_scifi_mc}_{config.noise_sigma}_{config.q_max}"
                        title_suffix = f"({energy_str} {base_label} | Layer {layer} Ori {plane})"
                        
                        plotter.plot_2d_hist(
                            x_data=hist_sipm_indices, 
                            y_data=filtered_qdc,
                            out_name=f"QDC_vs_Channel_{base_name}", 
                            xlabel="Channel Index",
                            ylabel="QDC", 
                            title=f"QDC vs Channel {title_suffix}",
                            subfolder="2d_combinations/QDC_vs_Channel"
                        )
                        
                        plotter.plot_2d_hist(
                            x_data=filtered_time_diff, 
                            y_data=filtered_qdc,
                            out_name=f"QDC_vs_Time_{base_name}", 
                            xlabel="Hit Time Diff [Clock Cyc.]",
                            ylabel="QDC", 
                            title=f"QDC vs Hit Time Diff {title_suffix}",
                            subfolder="2d_combinations/QDC_vs_Time"
                        )
                        
                        plotter.plot_2d_hist(
                            x_data=hist_sipm_indices, 
                            y_data=filtered_time_diff,
                            out_name=f"Time_vs_Channel_{base_name}", 
                            xlabel="Channel Index",
                            ylabel="Hit Time Diff [Clock Cyc.]", 
                            title=f"Time Diff vs Channel {title_suffix}",
                            subfolder="2d_combinations/Time_vs_Channel"
                        )