import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import os
import numpy
import torch.nn.functional as F

def hdw_plane(qdc_plane, delta_ch=80):
    """
    qdc_plane: (1536,) QDC values for one plane, one station
    delta_ch : channel window corresponding to 1 cm
    """
    hit_idx = torch.nonzero(qdc_plane > 0, as_tuple=False).squeeze(1)

    if hit_idx.numel() == 0:
        return 0.0

    # pairwise channel distance
    dist = hit_idx[:, None] - hit_idx[None, :] # create a matrix.

    # exclude self (j != i)
    mask = (dist != 0) & (dist.abs() <= delta_ch)

    # wi = number of neighbors
    wi = mask.sum(dim=1)

    # sum of hit-density weights
    return wi.sum().item()

def hdw_station(qdc_station, delta_ch=80):
    """
    qdc_station: (2, 1536) → [XZ, YZ]
    """
    hdw = 0.0
    for plane in range(2):
        hdw += hdw_plane(qdc_station[plane,:], delta_ch)
    return hdw
def hdw_event(scifi_event, delta_ch=1):
    """
    scifi_event: (2, 5, 1536)
    returns: maximum HDW among the 5 stations
    """
    hdw_stations = []
    for station in range(5):
        hdw_stations.append(
            hdw_station(scifi_event[:, station], delta_ch)
        )
    return max(hdw_stations)

def compute_hdw_all(scifi_qdc, delta_ch=1):
    """
    scifi_qdc: (N, 2, 5, 1536)
    """
    hdw = torch.zeros(len(scifi_qdc))
    for i in range(len(scifi_qdc)):
        hdw[i] = hdw_event(scifi_qdc[i], delta_ch)
    return hdw


import torch.nn.functional as F
def hdw_all_fast(scifi_qdc, delta_ch=80):
    """
    scifi_qdc: (N, 2, 5, 1536)
    returns  : (N,)
    """

    device = scifi_qdc.device
    N = scifi_qdc.shape[0]

    hits = (scifi_qdc > 0).int()    # (N,2,5,1536)

    plane_hdw = torch.zeros(N, 2, 5, device=device)

    for station in range(5):
        for plane in range(2):

            h = hits[:, plane, station, :]   # (N,1536)

            for n in range(N):
                hit_idx = torch.nonzero(h[n], as_tuple=False).squeeze(1)
                if hit_idx.numel() == 0:
                    continue

                # pairwise distance in channel index
                dist = hit_idx[:, None] - hit_idx[None, :]

                mask = (dist != 0) & (dist.abs() <= delta_ch)
                plane_hdw[n, plane, station] = mask.sum()

    # station HDW = sum of two planes
    station_hdw = plane_hdw.sum(dim=1)          # (N,5)
    print(station_hdw.shape)
    # event HDW = max station (paper tanımı)
    event_hdw = station_hdw.max(dim=1).values   # (N,)

    return event_hdw




def hdw_all_fast_conv(scifi_qdc, delta_ch=60):
    """
    scifi_qdc: (N, 2, 5, 1536)
    """
    N = scifi_qdc.shape[0]
    hits = (scifi_qdc !=0).float()    # (N,2,5,1536)

    # conv1d kernel
    kernel_size = 2 * delta_ch + 1
    kernel = torch.ones(1, 1, kernel_size, device=scifi_qdc.device)
    kernel[:, :, delta_ch] = 0.0      # j ≠ i

    plane_hdw = torch.zeros(N, 2, 5, device=scifi_qdc.device)

    for station in range(5):      # 5 stations
        for plane in range(2):            # XZ / YZ  
            x = hits[:, plane, station, :]     # (N,1536)
            # conv1d input
            x = x.unsqueeze(1)                 # (N,1,1536)

            neighbors = F.conv1d(
                x,
                kernel,
                padding=delta_ch
            ).squeeze(1)                        # (N,1536)

            wi = neighbors * hits[:, plane, station, :] 
            print("wi shape",wi.shape)
            plane_hdw[:, plane, station] = wi.sum(dim=-1)

    station_hdw = plane_hdw.sum(dim=1)          # (N, 5)
    print(station_hdw.shape)
    event_hdw, max_index = station_hdw.max(dim=1)

    idx = max_index.view(N, 1, 1).expand(-1, 2, 1)   # (N,2,1)

    plane_hdw_max = plane_hdw.gather(dim=2, index=idx).squeeze(2)

    hor_hdw = plane_hdw_max[:, 0]
    ver_hdw = plane_hdw_max[:, 1]

    return event_hdw, hor_hdw, ver_hdw

name= "scifi_ds_2024_electrons_300GeV_run_100927"
i=0
TEST_DATA_DIR = [f"/eos/user/b/beturk/snd/test_beam/2024/{name}_{i}.pt"]
scifi_signals = torch.load(TEST_DATA_DIR[0])["scifi_signals"]
hdw,hor_hdw, ver_hdw = hdw_all_fast_conv(scifi_signals,40)
print(ver_hdw.shape)
print(hor_hdw.shape)

plt.figure()

#plt.hist(hdw, bins=10, alpha=0.5, label="HDW")
plt.hist(hor_hdw, bins=40, alpha=0.7, label="Horizontal plane")
plt.hist(ver_hdw, bins=40, alpha=0.7, label="Vertical plane")
plt.yscale("log")
plt.title("Histogram of HDW(50 GeV Electrons RUN=100927)")
plt.legend()
plt.savefig(f"1d_HDW/{name}.png", dpi=300, bbox_inches="tight")
plt.close()