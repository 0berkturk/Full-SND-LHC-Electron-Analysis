import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset, ConcatDataset
import config
import os
import torch.nn as nn
import re
#### target data has unlabeled data. test data has labeled data.
##### source data must be train data. No need to add other stuff.
### Model will learn from target data. I have to add new unseen data to target data .Put all ISS(labeled+unlabeled). Maybe I can add Test Beam and MC(>1TeV) as well ??
##### labels for source->train and target
### fine for (MC+TB <1TeV) train, validate (MC>1TeV), validate (ISS). For other stuffs, make it manuel.

def extract_energy_from_path(path):
    match = re.search(r'_(\d+)GeV_', path)
    if match:
        return float(match.group(1))
    return -1.0

class EnergyResLossV1(nn.Module):
    def __init__(self,bins):
        super().__init__()
        self.register_buffer("bins", torch.tensor(bins, dtype=torch.float32)) ## storing bins as buffer so they automatically goes to gpu
        
    def forward(self, predicted_en, true_en):
        # both predicted_en and true_en are torch tensors
        device = predicted_en.device

        # bin edges on correct device
        bins_t = self.bins.to(device)

        # get bin index (torch equivalent of np.digitize)
        bin_idx = torch.bucketize(true_en, bins_t) - 1
        nbins = bins_t.numel() - 1

        rel_resolution = []

        for i in range(nbins):
            idx = (bin_idx == i)

            # skip bins with less than 2 samples
            if idx.sum() < 2:
                continue
            
            t = true_en[idx]
            p = predicted_en[idx]

            # relative residual (t - p)/t, gradient friendly
            rel = (t - p) / t

            # use unbiased=False to match numpy default if desired
            rel_resolution.append(torch.std(rel))

        # no valid bins case
        if len(rel_resolution) == 0:
            return torch.tensor(0.0, device=device)

        # sum of std across bins
        loss = torch.stack(rel_resolution).mean()
        return loss

def apply_sigmoid(logits):
    """Apply sigmoid to logits to normalize them between 0 and 1."""
    return torch.sigmoid(torch.tensor(logits))
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossBinary(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        """
        Binary Focal Loss
        alpha: weight for the positive class (float or None)
        gamma: focusing parameter
        reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: logits, shape (N, 1)
        targets: float tensor, shape (N, 1), values 0 or 1
        """
        # BCE with logits
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Compute pt = probability of true class
        pt = torch.exp(-bce_loss)

        # Focal loss factor
        focal_loss = (1 - pt) ** self.gamma * bce_loss

        # Apply alpha weighting if provided
        if self.alpha is not None:
            # alpha for positive class, 1-alpha for negative
            alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            focal_loss = alpha_factor * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha  # class weights (tensor or None)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: logits (N, C)
        targets: int64 labels (N,)
        """
        ce_loss = torch.nn.functional.cross_entropy(
            inputs, targets, weight=self.alpha, reduction="none"
        )
        pt = torch.exp(-ce_loss)  # probability of true class
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
import matplotlib.pyplot as plt
from datetime import datetime
cmap = plt.get_cmap('plasma')
cmap.set_under('white')
def plot_2d_im(scifi_hits, out_name):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    for i, ax in enumerate(axes):
        data = scifi_hits[i]  # shape (15, 512)
        #print(data.shape)
        nrows, ncols = data.shape

        masked_data = np.ma.masked_where(data == 0, data)
        #for x in range(ncols+ 1):
           # ax.axvline(x, color='black', linewidth=0.5)
        for y in range(nrows + 1):
            ax.axhline(y, color='black', linewidth=0.5)
        #im = ax.imshow(masked_data,
                      # cmap="inferno",
                     #  origin="lower",
                    #   aspect="auto",
                    #   interpolation="nearest")
        im = ax.imshow(masked_data, interpolation='nearest', aspect="auto",cmap=cmap)

        ax.set_title(f"SciFi Signals on {['X', 'Y'][i]}Z Plane")
        ax.set_xlabel("Fiber index")
        if i == 0:
            ax.set_ylabel("Z (Station index)")

    # Move colorbar to the right side of both plots
    cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.04)
    cbar.set_label("Signal")

    plt.tight_layout(rect=[0, 0, 0.95, 1])  # leave space for colorbar
    plt.savefig(out_name, dpi=300)
    plt.clf()


class SNDSparseDataset(Dataset):
    def __init__(self,label_fname,perc=None):
        print(f"Loading sparse data from {label_fname}...")

        #self.qdc_threshold_value_scifi = 0

        #self.smear_sigma = 0

        self.label=label_fname[0]
        self.path=label_fname[1]
        self.perc = perc
        

        data = torch.load(self.path, weights_only=False)
        total_events_in_file = len(data['scifi_notime_total_hits'])
        cut_mask = torch.ones(total_events_in_file, dtype=torch.bool)

        hit_hdw_prefixes =["scifi_hitx_in_64r", "scifi_hity_in_64r",
        "scifi_hitx_in_128r", "scifi_hity_in_128r",
        "max_total_hdw"]

        for key in hit_hdw_prefixes:
            if key in data:
                vals = torch.as_tensor(data[key], dtype=torch.float32)
                cut_mask &= (vals >= getattr(config, f"cut_min_{key}", 0))
                cut_mask &= (vals <= getattr(config, f"cut_max_{key}", float('inf')))

        global_prefixes = [
            "scifi_notime", "scifi_05usualtime",
            "us_notime", "us_3usualtime",
            "dsh_notime", "dsh_3usualtime",
            "dsv_notime", "dsv_3usualtime"
        ]
        print(torch.sum(cut_mask))

        for prefix in global_prefixes:
            # --- Hits ---
            hits_key = f"{prefix}_total_hits"
            if hits_key in data:
                vals = data[hits_key]
                cut_mask &= (vals >= getattr(config, f"cut_min_{hits_key}", -50000000))
                cut_mask &= (vals <= getattr(config, f"cut_max_{hits_key}", float('inf')))

            # --- QDC ---
            qdc_key = f"{prefix}_total_qdc"
            if qdc_key in data:
                vals = data[qdc_key]
                cut_mask &= (vals >= getattr(config, f"cut_min_{qdc_key}", -50000000))
                cut_mask &= (vals <= getattr(config, f"cut_max_{qdc_key}", float('inf')))
        print(torch.sum(cut_mask))


        layer_configs = [
            ("scifi_notime", 5), ("scifi_05usualtime", 5),
            ("us_notime", 5), ("us_3usualtime", 5),
            ("dsh_notime", 3), ("dsh_3usualtime", 3),
            ("dsv_notime", 4), ("dsv_3usualtime", 4)
        ]

        for prefix, n_layers in layer_configs:
            hits_key = f"{prefix}_hits_per_layer"
            qdc_key = f"{prefix}_qdc_per_layer"
            
            # --- Hits Per Layer ---
            if hits_key in data:
                vals = data[hits_key]  # Shape: (N_events, n_layers)
                min_limits = getattr(config, f"cut_min_{hits_key}", [-50000000] * n_layers)
                max_limits = getattr(config, f"cut_max_{hits_key}", [float('inf')] * n_layers)
                
                for i in range(n_layers):
                    cut_mask &= (vals[:, i] >= min_limits[i])
                    cut_mask &= (vals[:, i] <= max_limits[i])
            
            # --- QDC Per Layer ---
            if qdc_key in data:
                vals = data[qdc_key]   # Shape: (N_events, n_layers)
                min_limits = getattr(config, f"cut_min_{qdc_key}", [-50000000] * n_layers)
                max_limits = getattr(config, f"cut_max_{qdc_key}", [float('inf')] * n_layers)
                
                for i in range(n_layers):
                    cut_mask &= (vals[:, i] >= min_limits[i])
                    cut_mask &= (vals[:, i] <= max_limits[i])

        print(torch.sum(cut_mask))

        # apply max cuts too. reject naturally using the max cuts.
        # continue to apply cut on us and ds hits. 

        valid_indices = torch.where(cut_mask)[0]
        if self.perc is not None:
            if self.perc < 1.0:
                limit = int(self.perc * len(valid_indices))
            else:
                limit = min(int(self.perc), len(valid_indices))
            valid_indices = valid_indices[:limit]

        valid_mask_final = torch.zeros(total_events_in_file, dtype=torch.bool)
        valid_mask_final[valid_indices] = True
        
        print(f"File {self.path}: {len(valid_indices)}/{total_events_in_file} passed cuts.")
        self.num_events = len(valid_indices)
        if self.num_events == 0:
            return

        print(self.num_events,self.label )
        self.labels = torch.full((self.num_events,), self.label, dtype=torch.long)

        
        if "en3d" not in data:
            energy = extract_energy_from_path(self.path)
            self.energies = torch.full((self.num_events,), energy, dtype=torch.float32)
        else:
            self.energies =data["en3d"][valid_indices]

        # 5. cannot work with self.energies =data["en3d"][valid_indices] becuase below is a just list with different lengths
        self.mean_time_scifi=[False] * self.num_events
        self.mean_time_ds=[False] * self.num_events
        self.mean_time_us=[False] * self.num_events


        valid_idx_list = valid_indices.tolist()
        self.scifi_idx = [data["scifi_indices"][i] for i in valid_idx_list]
        self.scifi_sig = [data["scifi_signals"][i] for i in valid_idx_list]
        self.scifi_time = [data["scifi_hit_time"][i] for i in valid_idx_list]

        if "scifi_mean_hit_time" in data:
            self.mean_time_scifi = [data["scifi_mean_hit_time"][i] for i in valid_idx_list]


        if "us_signals" in config.KEYS_FOR_DATA_LOADER:
            self.us_idx = [data["us_indices"][i] for i in valid_idx_list]
            self.us_sig = [data["us_signals"][i] for i in valid_idx_list]
            self.us_time = [data["us_signals_time"][i] for i in valid_idx_list]

            if "us_mean_hit_time" in data:
                self.mean_time_us = [data["us_mean_hit_time"][i] for i in valid_idx_list]

        
        if "ds" in config.KEYS_FOR_DATA_LOADER:
            self.dsh_idx = [data["ds_h_indices"][i] for i in valid_idx_list]
            self.dsh_sig = [data["ds_h_signals"][i] for i in valid_idx_list]
            self.dsh_time = [data["ds_h_times"][i] for i in valid_idx_list]
            
            self.dsv_idx = [data["ds_v_indices"][i] for i in valid_idx_list]
            self.dsv_sig = [data["ds_v_signals"][i] for i in valid_idx_list]
            self.dsv_time = [data["ds_v_times"][i] for i in valid_idx_list]

            if "ds_mean_hit_time" in data:
                self.mean_time_ds = [data["ds_mean_hit_time"][i] for i in valid_idx_list]

        print(f"Loaded {self.num_events} events.")

    """def update_cut(self, qdc_threshold_value_scifi, smear_sigma):
        #self.qdc_threshold_value_scifi = qdc_threshold_value_scifi
        #self.smear_sigma = smear_sigma"""
        
    def __len__(self):
        return self.num_events

    def _find_highest_bin(self, values, num_bins=128,range1=(0,16)):
        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy()
        else:
            values = np.array(values)
        valid_mask = np.isfinite(values)
        clean_values = values[valid_mask]

        if len(values) == 0:
            return 0.0
        if np.min(values) == np.max(values):
            return float(values[0])
        
        if range1==(0,16):
            counts, bin_edges = np.histogram(values, bins=num_bins, range=range1)
        elif range1=="no_range":
            counts, bin_edges = np.histogram(values, bins=num_bins)

        max_bin_idx = np.argmax(counts)
        return float((bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2.0)

    def _apply_time_cut(self, idx_tensor, sig_tensor, time_tensor, highest_bin_time, t_window_high_end,t_window_low_end):
        """ Helper to apply hit-level time cuts on the fly inside workers """ 
        # 1. Type-Safe Conversion: Force everything into PyTorch Tensors instantly!
        if not isinstance(idx_tensor, torch.Tensor):
            idx_tensor = torch.tensor(idx_tensor, dtype=torch.long)
        if not isinstance(sig_tensor, torch.Tensor):
            sig_tensor = torch.tensor(sig_tensor, dtype=torch.float32)
        if not isinstance(time_tensor, torch.Tensor):
            time_tensor = torch.tensor(time_tensor, dtype=torch.float32)

        # 2. Fast bypass for empty events
        if len(time_tensor) == 0 or t_window_high_end is None:
            return idx_tensor, sig_tensor

        # 3. Fast bypass for MC dummy data
        # If the first hit time is 0.0, we know it's dummy MC data. Skip the math!
        if time_tensor[0].item() == 0.0:
            return idx_tensor, sig_tensor

        # 4. Calculate highest bin if not already cached
        #if highest_bin_time is False:  use n=128 bins, previous one is wrong.
        highest_bin_time = self._find_highest_bin(time_tensor)

        # 5. Apply the cut (Now mathematically safe because time_tensor is a real Tensor)
        valid_time_mask = (time_tensor >= highest_bin_time - t_window_low_end) & (time_tensor <= highest_bin_time + t_window_high_end)        

        return idx_tensor[valid_time_mask], sig_tensor[valid_time_mask]
        
    def _crop_shower(self, dense_tensor, width):
        """Replaces the old get_shower_cluster on a 3D (2, 5, 1536) tensor."""
        summed_xy = torch.sum(dense_tensor, dim=(0, 2)) # Sum over orientation and channels to find max plane
        max_layer = torch.argmax(summed_xy)
        
        # Find max horizontal and vertical on that layer
        layer_data = dense_tensor[:, max_layer, :]
        max_hor = torch.argmax(layer_data[0])
        max_ver = torch.argmax(layer_data[1])
        
        # Calculate boundaries
        ver_start = max(0, max_ver - width)
        ver_end = min(1536, max_ver + width)
        hor_start = max(0, max_hor - width)
        hor_end = min(1536, max_hor + width)
        
        # Adjust if hitting borders
        if ver_end == 1536: ver_start = 1536 - 2*width
        if ver_start == 0: ver_end = 2*width
        if hor_end == 1536: hor_start = 1536 - 2*width
        if hor_start == 0: hor_end = 2*width
        
        cropped = torch.zeros((2, 5, 2*width), dtype=torch.float32)
        cropped[0, :, :] = dense_tensor[0, :, hor_start:hor_end]
        cropped[1, :, :] = dense_tensor[1, :, ver_start:ver_end]
        return cropped
    
    def _crop_layer(self, x, K):
        layer_sums = x.sum(dim=(0, 2))      
        topK_indices = torch.topk(layer_sums, k=K).indices 
        topK_indices, _ = torch.sort(topK_indices) 
        return x[:, topK_indices, :]
    
    def _qdc_threshold(self,sf_idx, sf_sig, threshold):
        #print(threshold)
        cut = sf_sig >= threshold
        return sf_idx[cut], sf_sig[cut]
    
    def _add_noise(self, mc_signal_list, smear_sigma, scale_factor_data_mc):
        scaled_tensor = mc_signal_list * scale_factor_data_mc        
        noise = torch.randn_like(scaled_tensor) * smear_sigma
        smeared_tensor = scaled_tensor + noise
        return smeared_tensor

    def __getitem__(self, idx):
        # call scifi us ds 
        # call in time hits cut tensor
        # apply cuts
        # make dict.
        out_dict = {
            "y": self.labels[idx],
            "en3d": self.energies[idx]
        }

        if "MC" in self.path:
            t_window_high_end = getattr(config, 't_window_high_end_mc', 16)
            t_window_low_end = getattr(config, 't_window_low_end_mc', 16)

            time_window_max_us_ds = getattr(config, 'time_window_max_us_ds_mc', 16)
            time_window_min_us_ds = getattr(config, 'time_window_min_us_ds_mc', 16)

            qdc_threshold_value_scifi = getattr(config,"qdc_threshold_value_scifi_mc",-10000) #deleted for now
        
        else:
            t_window_high_end = getattr(config, 't_window_high_end_data', 16)
            t_window_low_end = getattr(config, 't_window_low_end_data', 16)

            time_window_max_us_ds = getattr(config, 'time_window_max_us_ds_data', 16)
            time_window_min_us_ds = getattr(config, 'time_window_min_us_ds_data', 16)

            qdc_threshold_value_scifi = getattr(config,"qdc_threshold_value_scifi_data",-1110) # deleted for now.


        sf_sig_dense = torch.zeros((2, 5, 1536), dtype=torch.float32)
        #sf_idx, sf_sig = self.scifi_idx[idx], self.scifi_sig[idx]
        sf_idx, sf_sig = self._apply_time_cut(self.scifi_idx[idx], self.scifi_sig[idx], self.scifi_time[idx], self.mean_time_scifi[idx] , t_window_high_end, t_window_low_end)
        #apply noise too.
        """if "MC" in self.path:
            sf_sig = self._add_noise(sf_sig, getattr(config, 'smear_sigma', 0), self.scale_factor_data_mc)"""
        
        #print("fname and qdc",self.path, self.qdc_threshold_value_scifi)
        sf_idx, sf_sig = self._qdc_threshold(sf_idx, sf_sig, qdc_threshold_value_scifi)


        if len(sf_idx) > 0:
            sf_sig_dense[sf_idx[:, 0].long(), sf_idx[:, 1].long(), sf_idx[:, 2].long()] = sf_sig
            
        if getattr(config, 'BINARY_QDC_VALUES', False):
            sf_sig_dense = (sf_sig_dense > 0).float()

        if getattr(config, 'SHOWER_WIDTH', None) is not None:
            print("USING SHOWER WIDTH")
            sf_sig_dense = self._crop_shower(sf_sig_dense, config.SHOWER_WIDTH)
        
        if getattr(config, 'USE_HIGHEST_N_LAYER', None) is not None: 
            sf_sig_dense = self._crop_layer(sf_sig_dense, config.USE_HIGHEST_N_LAYER)

        out_dict["scifi_signals"] = sf_sig_dense

        if "us_signals" in config.KEYS_FOR_DATA_LOADER:
            us_sig_dense = torch.zeros((2, 5, 80), dtype=torch.float32)
            us_idx, us_sig = self._apply_time_cut(self.us_idx[idx], self.us_sig[idx], self.us_time[idx], self.mean_time_us[idx] ,time_window_max_us_ds,time_window_min_us_ds)
            
            if len(us_idx) > 0:
                us_sig_dense[us_idx[:, 0].long(), us_idx[:, 1].long(), us_idx[:, 2].long()] = us_sig
                
            if getattr(config, 'BINARY_QDC_VALUES', False): 
                us_sig_dense = (us_sig_dense > 0).float()
            out_dict["us_signals"] = us_sig_dense

        
        # --- DS (Direct Scatter Method) ---
        if "ds" in config.KEYS_FOR_DATA_LOADER:
            # Shape: (3 Orientations [H0, H1, V], 4 Planes, 60 Channels)
            ds_dense = torch.zeros((3, 4, 60), dtype=torch.float32)

            if self.mean_time_ds[idx]==False:
                combined_time = torch.cat((self.dsv_time[idx], self.dsh_time[idx]))
                self.mean_time_ds[idx] = self._find_highest_bin(combined_time)


            # DS Horizontal
            dsh_idx, dsh_sig = self._apply_time_cut(self.dsh_idx[idx], self.dsh_sig[idx], self.dsh_time[idx], self.mean_time_ds[idx], time_window_max_us_ds,time_window_min_us_ds)
            if len(dsh_idx) > 0:
                ds_dense[dsh_idx[:, 0].long(), dsh_idx[:, 1].long(), dsh_idx[:, 2].long()] = dsh_sig

            # DS Vertical (Scatter directly into Orientation Index 2)
            dsv_idx, dsv_sig = self._apply_time_cut(self.dsv_idx[idx], self.dsv_sig[idx], self.dsv_time[idx], self.mean_time_ds[idx], time_window_max_us_ds,time_window_min_us_ds)
            if len(dsv_idx) > 0:
                ds_dense[2, dsv_idx[:, 0].long(), dsv_idx[:, 1].long()] = dsv_sig

            if getattr(config, 'BINARY_QDC_VALUES', False): 
                ds_dense = (ds_dense > 0).float()
                
            out_dict["ds"] = ds_dense

        # Package only what the config asks for, in the exact order
        return tuple(out_dict[key] for key in config.KEYS_FOR_DATA_LOADER)



def data_loader(data_list, perc, batch_size, device, is_train=True):
    datasets = []          # FIX 1: Matched the list name
    label_counts = {}      # FIX 1: Initialized the dictionary
    for label_fname in data_list:
        ds = SNDSparseDataset(label_fname, perc)  
        if len(ds) > 0:
            datasets.append(ds)
            lbl = label_fname[0]
            label_counts[lbl] = label_counts.get(lbl, 0) + len(ds)
            
    if len(datasets) == 0:
        return None, None, label_counts

    combined_dataset = ConcatDataset(datasets)
    
    dataloader = DataLoader(
        combined_dataset, 
        batch_size=batch_size,   # FIX 3: Matched capitalization
        shuffle=is_train,        # FIX 2: Uses the new function argument
        num_workers=4, 
        pin_memory=True
    )
    return dataloader, combined_dataset, label_counts
    
def loss_function_cls(label_counts,device):
    n_classes = max(label_counts.keys()) + 1
    counts = torch.zeros(n_classes, dtype=torch.float32)
    for lbl, count in label_counts.items():
        counts[lbl] = count
    # 3. The math stays EXACTLY the same as your original code
    weights = counts.sum() / (counts + 1e-8)
    weights = weights / weights.mean()
    weights = weights.to(device)
    print(weights)
    if config.IS_BINARY:
        pos_weight1 = torch.tensor([weights[1] / weights[0]], device=device)
        print("pos weight BCE", pos_weight1)

        if config.focal_loss:
            # Focal loss for binary
            if config.pos_weight:
                lossfun = FocalLossBinary(alpha=pos_weight1, gamma=config.gamma, reduction="none")  # adapt FocalLoss to binary

            else:
                lossfun = FocalLossBinary(alpha=None, gamma=config.gamma, reduction="none")  # adapt FocalLoss to binary
                print("Using focal loss (binary), gamma =", config.gamma)
        else:
            if config.pos_weight:
                lossfun = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight1, reduction="none")
                print("Using weighted BCEWithLogitsLoss")
            else:
                lossfun = torch.nn.BCEWithLogitsLoss(reduction="none")
                print("Using plain BCEWithLogitsLoss")
    else:
        if config.focal_loss:
            if config.pos_weight:
                lossfun = FocalLoss(alpha=weights, gamma=config.gamma, reduction="none")
                print("Using weighted focal loss, gamma =", config.gamma)
            else:
                lossfun = FocalLoss(alpha=None, gamma=config.gamma, reduction="none")
                print("Using focal loss, gamma =", config.gamma)
        else:
            if config.pos_weight:
                lossfun = torch.nn.CrossEntropyLoss(weight=weights, reduction="none")
                print("Using weighted cross entropy")
            else:
                lossfun = torch.nn.CrossEntropyLoss(reduction="none")
                print("Using plain cross entropy")
    return lossfun

def loss_function_energy(device):
    if config.LOSS_FUNC_TYPE == "HUBER":
        lossfun = nn.HuberLoss(reduction="mean")
    elif config.LOSS_FUNC_TYPE == "L1":
        lossfun = nn.L1Loss(reduction="mean")
    elif config.LOSS_FUNC_TYPE == "MSE":
        lossfun = nn.MSELoss(reduction="mean")
    elif config.LOSS_FUNC_TYPE=="EnergyResLossV1":
        lossfun = EnergyResLossV1(config.TRANING_VAL_BINS)
        print("Using my EnergyResLossV1, using bins",config.TRANING_VAL_BINS)
    else:
        print("LOSS FUNCTION NOT DEFINED, EXIT")
        os._exit(1)
    return lossfun


def combine_encut(data_list,en_min,en_max):
    print("UPDATE THIS FUNCTION,WRONG, DATA_LOADER.PY LINE 96")
    x= None
    for f in data_list:
        print(f)
        data = torch.load(f)
        index = (data["en3d"] > en_min) & (data["en3d"] < en_max)
        data = data["x"][index]
        if x is None:
            x = data
        else:
            x = torch.cat((x, torch.load(f)["x"]))
    return x.reshape(-1,1,18,72)


def data_loader_loss_train_val_target(device):
    ###### train loader #####
    print("\ntrain")
    train_data_loader, train_data_set, label_counts = data_loader(config.TRAINING_FILE, config.PERC_TRAIN, config.BATCH_SIZE,device)
    if config.IS_ENERGY_RECON:
        lossfun_train = loss_function_energy(device)
    elif config.IS_CLS:
        lossfun_train = loss_function_cls(label_counts,device)    
    print("\n finished\n\n\n ")

    ##### end of train loader  #####

    ## validation loader
    print("\nval")
    val_data_loader, val_data_set, label_counts = data_loader(config.VALIDATION_FILE, config.PERC_VAL, config.BATCH_SIZE, device)
    if config.IS_ENERGY_RECON:
        lossfun_val = loss_function_energy(device)
    elif config.IS_CLS:
        lossfun_val = loss_function_cls(label_counts,device)    
    print("\n finished\n\n\n ")
    ## end of validation loader

    if config.TARGET_DATA_DIR=="val":
        target_x0 = val_data_set.tensors[0]
        print("Shape target", target_x0.shape)
        target_y0 = torch.zeros(len(target_x0))

        target_x1 = train_data_set.tensors[0]
        target_y1 = torch.ones(len(target_x1))

    elif config.TARGET_DATA_DIR == None:
        label0_train_domain_adapt_dataset=0
        label1_train_domain_adapt_dataset=0

    else:
        target_x0 = combine_encut(config.TARGET_DATA_DIR, en_min_target, en_max_target)
        print("Shape target", target_x0.shape)
        target_y0 = torch.zeros(len(target_x0))

        target_x1 = train_data_set.tensors[0]
        target_y1 = torch.ones(len(target_x1))

    ## update bce losses ???
    if (config.DA_1_DATALOADER):
        target_x=torch.cat((target_x0,target_x1))
        target_y=torch.cat((target_y0,target_y1))
        dataset = TensorDataset(target_x,target_y)
        label0_train_domain_adapt_dataset = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        n_pr = target_y0.shape[0]
        n_el = target_y1.shape[0]
        lossfun_domain = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_el / n_pr], device=device))
        label1_train_domain_adapt_dataset=0

    elif config.TARGET_DATA_DIR == None:
        label0_train_domain_adapt_dataset=0
        label1_train_domain_adapt_dataset=0
        lossfun_domain = 0

    else:
        target = TensorDataset(target_x0,target_y0)
        label0_train_domain_adapt_dataset = DataLoader(target, batch_size=BATCH_SIZE, shuffle=True)

        label1_train_domain_adapt_dataset = TensorDataset(target_x1, target_y1)
        lossfun_domain = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.5], device=device))


    return train_data_loader, val_data_loader, label0_train_domain_adapt_dataset, label1_train_domain_adapt_dataset, lossfun_train, lossfun_val, lossfun_domain

