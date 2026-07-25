import os
import glob
import torch
import numpy as np
import config 
from dl_recon_core_sparse.data_loader import SNDSparseDataset

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

def calculate_time_differences(p_idx, hit_times, signals):
    """
    Her layer (station) için zamanları ve zaman farklarını hesaplar.
    Hem 'ortalama (mean)' zamanı hem de 'ilk kaydedilen (first - minimum)' zamanı baz alır.
    """
    # İsim, Threshold, Hesaplama Modu
    configs = [
        ('no_thr_mean', None, 'mean'),
        ('thr_0_mean', 0, 'mean'),
        ('thr_1_mean', 1, 'mean'),
        ('thr_2_mean', 2, 'mean'),
        ('no_thr_first', None, 'first')  # Threshold olmadan sadece ilk hitler
    ]
    
    results = {}
    
    for thr_name, thr_val, mode in configs:
        layer_times = {p: [] for p in range(5)}
        
        # Sinyalleri eşik değerine göre filtreleyip zamanları layer bazlı grupla
        for p, t, sig in zip(p_idx, hit_times, signals):
            if thr_val is None or sig > thr_val:
                layer_times[p].append(t)
        
        # Her layer için hesaplanmış zaman (mean veya min)
        computed_layer_times = {}
        for p in range(5):
            if len(layer_times[p]) > 0:
                if mode == 'mean':
                    computed_layer_times[p] = np.mean(layer_times[p])
                elif mode == 'first':
                    computed_layer_times[p] = np.min(layer_times[p]) # En erken zamanı ilk hit kabul ediyoruz
        
        diff_last_first = -999.0
        consecutive_diffs = []
        
        hit_layers = sorted(list(computed_layer_times.keys()))
        
        if len(hit_layers) >= 2:
            # Sadece hit alınan son ve ilk layer arasındaki fark
            diff_last_first = computed_layer_times[hit_layers[-1]] - computed_layer_times[hit_layers[0]]
            
            # Sadece ardışık (consecutive) layerlar arasındaki farkları topla
            for i in range(1, len(hit_layers)):
                p_curr = hit_layers[i]
                p_prev = hit_layers[i-1]
                if p_curr - p_prev == 1: 
                    consecutive_diffs.append(computed_layer_times[p_curr] - computed_layer_times[p_prev])
                    
        avg_consec_diff = np.mean(consecutive_diffs) if len(consecutive_diffs) > 0 else -999.0
        
        results[thr_name] = {
            'computed_layer_times': computed_layer_times,
            'diff_last_first': diff_last_first,
            'avg_consecutive_diff': avg_consec_diff
        }
        for key in results:
            print(key,results[key])
        
    return results

def calculate_advanced_angles(plane_spatial_data, plane_signals):
    step_thresholds = [0, 1, 2, 5, 10]
    linear_thresholds = [0.1, 0.2, 0.5, 1.0]
    
    results = {'X': {'step': {}, 'linear': {}, 'unweighted': None}, 
               'Y': {'step': {}, 'linear': {}, 'unweighted': None}}

    def get_weighted_mean(vals, sigs, threshold, mode='step'):
        if not vals or not sigs: return None
        sigs = np.array(sigs)
        weights = np.where(sigs >= threshold, sigs, 0) if mode == 'step' else np.where(sigs < threshold, threshold, sigs)
        if weights.sum() == 0: return np.mean(vals)
        return np.sum(np.array(vals) * weights) / weights.sum()

    for axis in ['X', 'Y']:
        # Unweighted
        z_pts_unw, v_pts_unw = [], []
        for p in range(1, 6):
            vals = plane_spatial_data[axis][p]['val']
            if len(vals) > 0:
                v_pts_unw.append(np.mean(vals))
                z_pts_unw.append(np.mean(plane_spatial_data[axis][p]['z']))

        results[axis]['unweighted'] = np.polyfit(z_pts_unw, v_pts_unw, 1)[0] if len(z_pts_unw) >= 2 else -999.0

        # Thresholds
        for thr in step_thresholds:
            z, v = [], []
            for p in range(1, 6):
                if len(plane_spatial_data[axis][p]['val']) == len(plane_signals[p][axis]):
                    val = get_weighted_mean(plane_spatial_data[axis][p]['val'], plane_signals[p][axis], thr, 'step')
                    if val is not None and not np.isnan(val):
                        z.append(np.mean(plane_spatial_data[axis][p]['z']))
                        v.append(val)
            results[axis]['step'][f'thr_{thr}'] = np.polyfit(z, v, 1)[0] if len(z) >= 2 else -999.0

        for thr in linear_thresholds:
            z, v = [], []
            for p in range(1, 6):
                if len(plane_spatial_data[axis][p]['val']) == len(plane_signals[p][axis]):
                    val = get_weighted_mean(plane_spatial_data[axis][p]['val'], plane_signals[p][axis], thr, 'linear')
                    if val is not None and not np.isnan(val):
                        z.append(np.mean(plane_spatial_data[axis][p]['z']))
                        v.append(val)
            results[axis]['linear'][f'thr_{thr}'] = np.polyfit(z, v, 1)[0] if len(z) >= 2 else -999.0
            
    return results

def calculate_ideal_neutrino_angle(plane_spatial_data, TUNGSTEN_LENGTH):
    IP1_DISTANCE_CM = 48000.0
    first_plane = next((p for p in range(1, 6) if len(plane_spatial_data['X'][p]['val']) > 0 and len(plane_spatial_data['Y'][p]['val']) > 0), None)
    
    if first_plane is None or first_plane == 1:
        return 999.0, 999.0
        
    z_hit = (np.mean(plane_spatial_data['X'][first_plane]['z']) + np.mean(plane_spatial_data['Y'][first_plane]['z'])) / 2.0
    vertex_z = z_hit - (TUNGSTEN_LENGTH / 2.0)
    
    return np.mean(plane_spatial_data['X'][first_plane]['val']) / (vertex_z + IP1_DISTANCE_CM), \
           np.mean(plane_spatial_data['Y'][first_plane]['val']) / (vertex_z + IP1_DISTANCE_CM)

def extract_passed_data_simple(file_names, year):
    geo_lut = torch.load(f"/afs/cern.ch/work/b/beturk/private/snd/data_analysis/angles/geo_luts/geo_lut_{year}.pt")
    TUNGSTEN_LENGTH = abs(geo_lut[0, 1, 768, 1].item() - geo_lut[0, 0, 768, 1].item())
    step_thr, lin_thr = [0, 1, 2, 5, 10], [0.1, 0.2, 0.5, 1.0]
    
    # Zaman için kullanılacak anahtar isimleri
    time_configs = ['no_thr_mean', 'thr_0_mean', 'thr_1_mean', 'thr_2_mean', 'no_thr_first']

    for file_path in file_names:
        save_path = os.path.join(os.path.dirname(file_path) + "_extracted_props", f"{os.path.basename(file_path).replace('.pt', '')}_extracted.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.exists(save_path): continue

        dataset = SNDSparseDataset([0, file_path], EN_MIN=0, EN_MAX=20000, is_lhcdata=True)
        print(".  ")
        if dataset.num_events == 0: continue

        raw_data, idx = dataset.data, dataset.valid_indices
        extracted_dict = {key: torch.from_numpy(np.array(raw_data[key]))[idx] for key in keys_list if key in raw_data}
        extracted_dict["idx"] = idx

        angle_results = {'incoming_neutrino_x': [], 'incoming_neutrino_y': [], 'unweighted_x': [], 'unweighted_y': [], 
                         **{f'step_{t}': {'x': [], 'y': []} for t in step_thr},
                         **{f'linear_{t}': {'x': [], 'y': []} for t in lin_thr}}

        # Zaman analiz sonuçları için dictionary oluştur
        time_results_dict = {
            thr: {
                'diff_last_first': [],
                'avg_consecutive_diff': [],
                **{f'layer_{p}_time': [] for p in range(5)}
            } for thr in time_configs
        }

        for i in idx:
            v_idx, p_idx, c_idx = raw_data["scifi_indices"][i][:, 0].long(), raw_data["scifi_indices"][i][:, 1].long(), raw_data["scifi_indices"][i][:, 2].long()
            pos = geo_lut[v_idx, p_idx, c_idx].numpy()
            plane_data = {'X': {p: {'val': [], 'z': []} for p in range(1, 6)}, 'Y': {p: {'val': [], 'z': []} for p in range(1, 6)}}
            plane_sig = {p: {'X': [], 'Y': []} for p in range(1, 6)}

            for v, p, val, z, sig in zip(v_idx.numpy(), p_idx.numpy(), pos[:, 0], pos[:, 1], raw_data["scifi_signals"][i]):
                axis, plane = ('X' if v == 1 else 'Y'), p + 1
                plane_data[axis][plane]['val'].append(val); plane_data[axis][plane]['z'].append(z); plane_sig[plane][axis].append(sig)

            # Zaman analizini çalıştır
            if "scifi_hit_time" in raw_data:
                time_metrics = calculate_time_differences(p_idx.numpy(), raw_data["scifi_hit_time"][i], raw_data["scifi_signals"][i])
                for thr in time_configs:
                    time_results_dict[thr]['diff_last_first'].append(time_metrics[thr]['diff_last_first'])
                    time_results_dict[thr]['avg_consecutive_diff'].append(time_metrics[thr]['avg_consecutive_diff'])
                    for pl in range(5):
                        val = time_metrics[thr]['computed_layer_times'].get(pl, -999.0)
                        time_results_dict[thr][f'layer_{pl}_time'].append(val)

            adv = calculate_advanced_angles(plane_data, plane_sig)
            ix, iy = calculate_ideal_neutrino_angle(plane_data, TUNGSTEN_LENGTH)
            
            angle_results['incoming_neutrino_x'].append(ix); angle_results['incoming_neutrino_y'].append(iy)
            angle_results['unweighted_x'].append(adv['X']['unweighted']); angle_results['unweighted_y'].append(adv['Y']['unweighted'])

            def _filtered_mean(vals):
                good = [v for v in vals if v != -999.0]
                return np.mean(good) if good else -999.0

            step_vals_x = [adv['X']['step'][f'thr_{t}'] for t in step_thr]
            step_vals_y = [adv['Y']['step'][f'thr_{t}'] for t in step_thr]
            lin_vals_x = [adv['X']['linear'][f'thr_{t}'] for t in lin_thr]
            lin_vals_y = [adv['Y']['linear'][f'thr_{t}'] for t in lin_thr]

            angle_results.setdefault('avg_step_x', []).append(_filtered_mean(step_vals_x))
            angle_results.setdefault('avg_step_y', []).append(_filtered_mean(step_vals_y))
            angle_results.setdefault('avg_linear_x', []).append(_filtered_mean(lin_vals_x))
            angle_results.setdefault('avg_linear_y', []).append(_filtered_mean(lin_vals_y))

            for t in step_thr:
                angle_results[f'step_{t}']['x'].append(adv['X']['step'][f'thr_{t}']); angle_results[f'step_{t}']['y'].append(adv['Y']['step'][f'thr_{t}'])
            for t in lin_thr:
                angle_results[f'linear_{t}']['x'].append(adv['X']['linear'][f'thr_{t}']); angle_results[f'linear_{t}']['y'].append(adv['Y']['linear'][f'thr_{t}'])
            
        for k, v in angle_results.items():
            if isinstance(v, list): extracted_dict[f"angle_{k}"] = torch.tensor(v)
            else:
                for axis in ['x', 'y']: extracted_dict[f"angle_{k}_{axis}"] = torch.tensor(v[axis])
                
        # Zaman sonuçlarını extracted_dict içerisine aktar
        if "scifi_hit_time" in raw_data:
            for thr, metrics in time_results_dict.items():
                for metric_name, values in metrics.items():
                    extracted_dict[f"time_{thr}_{metric_name}"] = torch.tensor(values)

        torch.save(extracted_dict, save_path)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    year = "2022"
    filelists=glob.glob(f"/eos/experiment/sndlhc/users/beturk/Data/PT/Data_{str(year)}/batched_data_{year}_*.pt")
    extract_passed_data_simple(filelists, year)