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
        #print("unweighted")
        #print(v_pts_unw)
        #print(z_pts_unw)
        results[axis]['unweighted'] = np.polyfit(z_pts_unw, v_pts_unw, 1)[0] if len(z_pts_unw) >= 2 else -999.0
        #print(results[axis]['unweighted'])
        # Thresholds
        for thr in step_thresholds:
            z, v = [], []
            for p in range(1, 6):
                if len(plane_spatial_data[axis][p]['val']) == len(plane_signals[p][axis]):
                    val = get_weighted_mean(plane_spatial_data[axis][p]['val'], plane_signals[p][axis], thr, 'step')
                    if val is not None and not np.isnan(val):
                        z.append(np.mean(plane_spatial_data[axis][p]['z']))
                        v.append(val)
            #print("step threshold")
            #print(z)
            #print(v)
            results[axis]['step'][f'thr_{thr}'] = np.polyfit(z, v, 1)[0] if len(z) >= 2 else -999.0
            #print(results[axis]['step'][f'thr_{thr}'])
        for thr in linear_thresholds:
            z, v = [], []
            for p in range(1, 6):
                if len(plane_spatial_data[axis][p]['val']) == len(plane_signals[p][axis]):
                    val = get_weighted_mean(plane_spatial_data[axis][p]['val'], plane_signals[p][axis], thr, 'linear')
                    if val is not None and not np.isnan(val):
                        z.append(np.mean(plane_spatial_data[axis][p]['z']))
                        v.append(val)
            results[axis]['linear'][f'thr_{thr}'] = np.polyfit(z, v, 1)[0] if len(z) >= 2 else -999.0
            #print("linear thresolhd")
            #print(results[axis]['linear'][f'thr_{thr}'])
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

    for file_path in file_names:
        #print(file_names)
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

        for i in idx:
            v_idx, p_idx, c_idx = raw_data["scifi_indices"][i][:, 0].long(), raw_data["scifi_indices"][i][:, 1].long(), raw_data["scifi_indices"][i][:, 2].long()
            pos = geo_lut[v_idx, p_idx, c_idx].numpy()
            plane_data = {'X': {p: {'val': [], 'z': []} for p in range(1, 6)}, 'Y': {p: {'val': [], 'z': []} for p in range(1, 6)}}
            plane_sig = {p: {'X': [], 'Y': []} for p in range(1, 6)}

            for v, p, val, z, sig in zip(v_idx.numpy(), p_idx.numpy(), pos[:, 0], pos[:, 1], raw_data["scifi_signals"][i]):
                axis, plane = ('X' if v == 1 else 'Y'), p + 1
                plane_data[axis][plane]['val'].append(val); plane_data[axis][plane]['z'].append(z); plane_sig[plane][axis].append(sig)

            adv = calculate_advanced_angles(plane_data, plane_sig)
            ix, iy = calculate_ideal_neutrino_angle(plane_data, TUNGSTEN_LENGTH)
            #print("neutrino angles",ix,iy)
            
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
                
        #for key in extracted_dict:
            #print(key,extracted_dict[key][:5])
        #mean hesapla?
        torch.save(extracted_dict, save_path)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    year = "2023"
    filelists=glob.glob(f"/eos/experiment/sndlhc/users/beturk/Data/PT/Data_{str(year)}/batched_data_{year}_*.pt")
    #print(filelists)
    extract_passed_data_simple(filelists, year)
    #print("opened")
