import sys
import glob
import os
import ROOT
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from decorators import *
import uproot
import awkward as ak
import numpy as np
import time
import torch 
from pathlib import Path
import SndlhcGeo
import shipunit
import torch.nn.functional as F

cmap = plt.get_cmap('plasma')
cmap.set_under('white')

def hdw_all_fast_conv(scifi_qdc, HDW_CHANNEL=40):
    """
    scifi_qdc: (N, 2, 5, 1536)
    """
    delta_ch = HDW_CHANNEL
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
            #print("wi shape",wi.shape)
            plane_hdw[:, plane, station] = wi.sum(dim=-1)

    station_hdw = plane_hdw.sum(dim=1)          # (N, 5)
    #print(station_hdw.shape)
    event_hdw, max_index = station_hdw.max(dim=1)

    idx = max_index.view(N, 1, 1).expand(-1, 2, 1)   # (N,2,1)

    plane_hdw_max = plane_hdw.gather(dim=2, index=idx).squeeze(2)

    hor_hdw = plane_hdw_max[:, 0]
    ver_hdw = plane_hdw_max[:, 1]

    return station_hdw, event_hdw, hor_hdw, ver_hdw

def plot_2d_im(scifi_hits, j,out_name):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    for i, ax in enumerate(axes):
        data = scifi_hits[j,i]  # shape (15, 512)
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

        ax.set_title(f"SciFi Projection {['X', 'Y'][i]}")
        ax.set_xlabel("Fiber index")
        if i == 0:
            ax.set_ylabel("Z (plane index)")

    # Move colorbar to the right side of both plots
    cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.04)
    cbar.set_label("Signal")

    plt.tight_layout(rect=[0, 0, 0.95, 1])  # leave space for colorbar
    plt.savefig("signals/"+ out_name+ str(j) + ".png", dpi=200)
    plt.clf()

def plot_2d_im_single(data, j,out_name):
    data=data[j]
    nrows, ncols = data.shape
    masked_data = np.ma.masked_where(data == 0, data)
    #for x in range(ncols+ 1):
       # ax.axvline(x, color='black', linewidth=0.5)
    #for y in range(nrows + 1):
        #plt.hlines(y, color='black', linewidth=0.5)
    #im = ax.imshow(masked_data,
                  # cmap="inferno",
                 #  origin="lower",
                #   aspect="auto",
                #   interpolation="nearest")
    plt.imshow(masked_data, interpolation='nearest', aspect="auto",cmap=cmap)

    plt.title("DS SiPMs Projection")
    plt.xlabel("Fiber index")
    plt.ylabel("Z (plane index)")

    # Move colorbar to the right side of both plots
    plt.colorbar()
    #plt.la("Signal")

    #plt.tight_layout(rect=[0, 0, 0.95, 1])  # leave space for colorbar
    plt.savefig("signals/"+ out_name+ str(j) + ".png", dpi=200)
    plt.clf()

def scifi_array_id(detID):
    # /* STMRFFF
    # First digit S: 		station # within the sub-detector
    # Second digit T: 		type of the plane: 0-horizontal fiber plane, 1-vertical fiber plane
    # Third digit M: 		determines the mat number 0-2
    # Fourth digit S: 		SiPM number  0-3
    # Last three digits F: 	local SiPM channel number in one mat  0-127
    # */
    # print('detID: ', detID)
    #one plane consists of 3 mat. 1 mat consists of 4 SiPMs. 1 SiPMs has 128 channel.
    n_plane = (detID // 1000000) % 10
    n_vert = (detID // 100000) % 10
    n_chan = detID % 1000 ## position of chann in a given SiPMs
    n_chan += 128 * ((detID // 1000) % 10) ## channel in a given mat (adding position of SiPMs)
    #print(detID)
    #print(n_vert, n_plane - 1, n_chan)
    n_chan += 128 * 4 * ((detID // 10000) % 10) ### final channel (adding position of mat)
    #print(n_vert, n_plane - 1, n_chan,"\n")

    # print('n_vert: {}, n_plane: {}, n_chan: {}'.format(n_vert, n_plane, n_chan))
    return n_vert, n_plane - 1, n_chan

def get_scifi_signals(tree,N):
    scifi_hits = np.zeros((N, 2, 15, 512))
    for i in range(0, N):
        tree.GetEvent(i)
        for aHit in tree.Digi_ScifiHits:
            #print(aHit.GetDetectorID(), aHit.GetStation(), aHit.GetMat(), aHit.GetSiPM(), aHit.GetSiPMChan(), aHit.GetChannelID(),aHit.GetSignal(),aHit.GetTime())
            n_plane, n_vert,n_chan = scifi_array_id(aHit.GetDetectorID())
            scifi_hits[i, n_vert, n_plane,n_chan] = aHit.GetSignal()
    return scifi_hits



def mufi_array_id(detID, vert=True):
    # int subsystem     = floor(fDetectorID/10000);
    # int plane             = floor(fDetectorID/1000) - 10*subsystem;
    # int bar_number   = fDetectorID%1000;
    # print('detID: ', detID)
    n_sys = detID // 10000
    n_plane = (detID // 1000) % 10
    n_chan = detID % 1000
    if n_sys == 3 and vert:
        # print("\nchanged one",n_chan,"\n")
        n_chan -= 60  ### probably, in ds, total number of chanell is n_chan_vert+n_chan_hor=120. so we need to extract them.
    if n_sys == 2 and vert:  ## no vertical in us.
        print('\n\t2 VERT\n')
    n_vert = int(vert)
    # works for only ds, not for us.
    # print('n_sys: {}, n_vert: {}, n_plane: {}, n_chan: {}'.format(n_sys, n_vert, n_plane, n_chan))

    return n_sys, n_vert, n_plane, n_chan

def get_ds_signals(tree,N):
    ds_horizontal = np.zeros((N, 3, 60))  # 60 channels, 3 layers.
    ds_vertical = np.zeros((N, 4, 60))  # 2 is reading side, 60 channels, 4 layers.
    for i in range(N):
        tree.GetEvent(i)
        for aHit in tree.Digi_MuFilterHits:
            if aHit.GetSystem() == 3 :
                hit_id = aHit.GetDetectorID()  # first digit is system number, second digit is plane number(layer
                #hit_system = aHit.GetSystem()
                #hit_plane_2 = aHit.GetPlane()
                is_vertical = aHit.isVertical()
                is_vertical = int(is_vertical)
                # loop over all channels per bars(bars are side by side)
                # print(is_vertical,aHit.GetnSides(),aHit.GetnSiPMs())
                side=0
                for channel in range(aHit.GetnSiPMs()):  # loop over channels per side. it is 1 and constant always for ds
                    ch = 8 * side + channel  ## fiberin iki uçtan toplanıyor. chanellar 8li aslında. 0'dan 8'e resim. 8'den 16'ya aynı resmin diğer taraftan toplanan signal resmi
                    # probably 8 must stay because every object has 16 size.
                    #print(ch, hit_ch, n_chan)
                    ###print(aHit.Getchannel(3),aHit.GetSignal(4)) this works but no assigned values.
                    #hit_ch = aHit.Getchannel(ch)  #getchannel works with fdaqid.
                    n_sys, n_vert, n_plane, n_chan = mufi_array_id(hit_id, aHit.isVertical())
                    signal = aHit.GetSignal(ch)
                    if signal!=-999.0:
                        #if side==0 and is_vertical==0:
                          #  print(signal) prints -1.
                          #  print(side, is_vertical,n_plane, n_chan) ##side1,horizontal,      side0, vert+horizontal
                        #print("hit on,",i,n_sys, n_vert, n_plane, n_chan,signal)
                        if is_vertical==0:
                            ds_horizontal[i, n_plane, n_chan] = signal
                        else:
                            ds_vertical[i, n_plane, n_chan] = signal
                            ##side'a göre +8 daha ekleniyor. yanlış. ?? eklenmiyor
    return ds_vertical,ds_horizontal


def get_us_signal(tree,N):
    us_signals = np.zeros((N, 2, 5, 10))
    for i in range(N):
        tree.GetEvent(i)
        for aHit in tree.Digi_MuFilterHits:
            hit_id = aHit.GetDetectorID()  ## get the hit on some part on detector id or tells system,layer,bar that hit happened.
            bar = (hit_id % 1000)
            hit_system = aHit.GetSystem()
            hit_plane_2 = aHit.GetPlane()
            #is_vertical = aHit.isVertical()
            if hit_system == 2:
                ## now, loop over each simps.
                # make 5layer x 10bar x 8pmts x 2readout ### along yz plane
                # convert this into 2x5x80 , 2 is the read out of the same fibers from both sides.

                side = 0
                for channel in range(1):  # this always give 16 for us. #this is not number of channel actually, this is ordering of channels because signals stored as a list.
                    # this loops over all SiPMs. some of them are empty....  these functions get the detector information from the aHit. nothing else happens.
                    # after that, this code one by one get the signal depositions.
                    ch = 8 * side + channel  ## fiberin iki uçtan toplanıyor. chanellar 8li aslında. 0'dan 8'e resim. 8'den 16'ya aynı resmin diğer taraftan toplanan signal resmi
                    real_channel = bar # aHit.Getchannel(ch)
                    # print(hit_id, hit_system, hit_plane_2, side, channel, aHit.Getchannel(ch), real_channel, aHit.GetSignal(ch)) # in here I dont need to use real_channel becuase ahit tells where is the hit. funciton goes there and find specific channel.
                    signal=aHit.GetSignal(ch)
                    if signal != -999.0:
                        us_signals[i, side, hit_plane_2, real_channel] = signal
                side = 1
                for channel in range(1):
                    ch = 8 * side + channel
                    real_channel = bar
                    # print(hit_id, hit_system, hit_plane_2, side, channel, aHit.Getchannel(ch), real_channel, aHit.GetSignal(ch))
                    signal=aHit.GetSignal(ch)
                    #print(i, side, hit_plane_2, real_channel)
                    if signal != -999.0:
                        us_signals[i, side, hit_plane_2, real_channel] = signal

                # geo.modules['MuFilter'].GetPosition(detID, A, B)
    return us_signals


def get_hcal_signals(tree,N):
    ds_horizontal = np.zeros((N,2, 3, 60)).astype(np.float32)  # 60 channels, 3 layers.
    ds_vertical = np.zeros((N, 4, 60)).astype(np.float32)  # 2 is reading side, 60 channels, 4 layers. one side is working in mc
    us_signals = np.zeros((N, 2, 5, 80)).astype(np.float32) #  2 reading side, 10 bar, each bar has 8 sipms,makes 80 channel, 5 layer

    ds_horizontal_time = np.zeros((N,2, 3, 60)).astype(np.float32)  # 60 channels, 3 layers.
    ds_vertical_time = np.zeros((N, 4, 60)).astype(np.float32)  # 2 is reading side, 60 channels, 4 layers. one side is working in mc
    us_signals_time = np.zeros((N, 2, 5, 80)).astype(np.float32) #  2 reading side, 10 bar, each bar has 8 sipms,makes 80 channel, 5 layer


    for i in range(N):
        for aHit in tree.Digi_MuFilterHits:
            hit_id = aHit.GetDetectorID()  ## get the hit on some part on detector id or tells system,layer,bar that hit happened.
            bar = (hit_id % 1000)
            hit_system = aHit.GetSystem()
            hit_plane_2 = aHit.GetPlane()
            if hit_system==1: ## veto
                print("hit in veto system")
                continue ## exit and look at other ahits.

            elif hit_system==2: ## us system
                for side in range(aHit.GetnSides()): #loop over sides left/right or top
                    for channel in range(aHit.GetnSiPMs()): #loop over channels per side
                        ch = 8*side+channel
                        real_channel = bar*10 + channel
                        if aHit.GetSignal(ch)!= -999.0:
                            us_signals[N,side,hit_plane_2, real_channel] = aHit.GetSignal(ch)
                            print("signal in us ",i, hit_id, us_signals[N,side,hit_plane_2, real_channel])
                            us_signals_time[N,side,hit_plane_2, real_channel] = aHit.GetTime(ch)

            elif hit_system==3: ## us system
                for side in range(aHit.GetnSides()): #loop over sides left/right or top
                    for channel in range(aHit.GetnSiPMs()): #loop over channels per side
                        ch = 8*side+channel
                        if aHit.GetSignal(ch)!= -999.0:
                            if aHit.isVertical():
                                real_channel=bar-60
                                ds_vertical[N, hit_plane_2, real_channel] = aHit.GetSignal(ch)
                                ds_vertical_time[N, hit_plane_2, real_channel] = aHit.GetTime(ch)
                            else:
                                real_channel=bar
                                ds_horizontal[N, side, hit_plane_2, real_channel] = aHit.GetSignal(ch)
                                ds_horizontal_time[N, side, hit_plane_2, real_channel] = aHit.GetTime(ch)
#ROOT.gSystem.Load("libSndAnalysisTools.so")
import numpy as np
import torch

def find_highest_bin(values, num_bins=40):
    if len(values) == 0:
        return 0.0
    if np.min(values) == np.max(values):
        return float(values[0])
        
    counts, bin_edges = np.histogram(values, bins=num_bins, range=(0, 16))
    max_bin_idx = np.argmax(counts)
    return float((bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2.0)


def get_event_aggregates(indices, signals, times,  num_layers, layer_dim=1,mean_time=None ,time_window_high=None,time_window_low=None):
    """
    Calculates total hits and QDC (global and per-layer).
    If time_window is provided, applies the time cut dynamically first.
    """
    idx_arr = np.array(indices, dtype=np.int16)
    sig_arr = np.array(signals, dtype=np.float32)
    time_arr = np.array(times, dtype=np.float32)
    
    # Apply Time Cut if requested
    if time_window_high is not None and len(time_arr) > 0:
        time_mask = (time_arr >= (mean_time - time_window_low)) & (time_arr <= (mean_time + time_window_high))
        
        idx_arr = idx_arr[time_mask]
        sig_arr = sig_arr[time_mask]
    

    # Global Aggregates
    total_hits = len(sig_arr)
    total_qdc = np.sum(sig_arr) if total_hits > 0 else 0.0

    # Per-Layer Aggregates
    hits_per_layer = np.zeros(num_layers, dtype=np.int32)
    qdc_per_layer = np.zeros(num_layers, dtype=np.float32)

    if total_hits > 0:
        layers = idx_arr[:, layer_dim] # Extract the plane/layer column
        for i in range(num_layers):
            layer_mask = (layers == i)
            hits_per_layer[i] = np.sum(layer_mask)
            qdc_per_layer[i] = np.sum(sig_arr[layer_mask])

    return total_hits, total_qdc, hits_per_layer, qdc_per_layer

def calculate_hdw(idx_list,sig_list,hdw_channel=40):
    """if len(idx_list) <= (min_hits * 2):
        return False"""

    idx_arr = np.array(idx_list, dtype=int)
    sig_arr = np.array(sig_list, dtype=np.float32)

    dense = np.zeros((2, 5, 1536), dtype=np.float32)
    dense[idx_arr[:, 0], idx_arr[:, 1], idx_arr[:, 2]] = sig_arr

    dense = torch.from_numpy(dense).unsqueeze(0)

    return hdw_all_fast_conv(dense, hdw_channel)


def passes_shower_density_cut(idx_list, sig_list, radius=64, min_hits=10):
    """
    Finds the shower core and checks if there are strictly more than 
    'min_hits' inside the core +/- 'radius' for BOTH horizontal and vertical planes.
    Maintains a strict 2*radius window size even at the detector edges.
    """
    """if len(idx_list) <= (min_hits * 2):
        return False
        """
    idx_arr = np.array(idx_list, dtype=int)
    sig_arr = np.array(sig_list, dtype=np.float32)

    dense = np.zeros((2, 5, 1536), dtype=np.float32)
    dense[idx_arr[:, 0], idx_arr[:, 1], idx_arr[:, 2]] = sig_arr

    max_layer = np.argmax(np.sum(dense, axis=(0, 2)))
    max_hor = np.argmax(dense[0, max_layer, :])
    max_ver = np.argmax(dense[1, max_layer, :])

    # 1. Initial boundaries clamped to detector limits
    hor_start = max(0, max_hor - radius)
    hor_end   = min(1536, max_hor + radius)
    ver_start = max(0, max_ver - radius)
    ver_end   = min(1536, max_ver + radius)

    # 2. SHIFT LOGIC: Guarantee the window is exactly 2*radius wide
    if hor_start == 0:
        hor_end = 2 * radius
    elif hor_end == 1536:
        hor_start = 1536 - (2 * radius)

    if ver_start == 0:
        ver_end = 2 * radius
    elif ver_end == 1536:
        ver_start = 1536 - (2 * radius)

    # 3. Fast NumPy masking for both orientations
    hor_mask = (idx_arr[:, 0] == 0) & (idx_arr[:, 2] >= hor_start) & (idx_arr[:, 2] < hor_end)
    ver_mask = (idx_arr[:, 0] == 1) & (idx_arr[:, 2] >= ver_start) & (idx_arr[:, 2] < ver_end)
    
    hits_in_radius_h = np.sum(hor_mask)
    hits_in_radius_v = np.sum(ver_mask)

    return hits_in_radius_h, hits_in_radius_v #(hits_in_radius_h > min_hits) and (hits_in_radius_v > min_hits)
import ROOT
import numpy as np

def get_physical_position(scifiDet, detID, n_vert):
    """
    SND@LHC geometrisinden fiberin X, Y, Z koordinatlarını çeker.
    Dikey fiberler (n_vert=1) X eksenini, yatay fiberler (n_vert=0) Y eksenini kusursuz ölçer.
    """
    A = ROOT.TVector3()
    B = ROOT.TVector3()
    scifiDet.GetPosition(detID, A, B)
    
    hit_z = A.Z() # Z ekseni fiber boyunca sabittir
    
    if n_vert == 1: # Vertical fiberler X ölçer
        hit_val = A.X()
    else:           # Horizontal fiberler Y ölçer
        hit_val = A.Y()
        
    return hit_val, hit_z

import numpy as np

def calculate_advanced_angles(plane_spatial_data, plane_signals):
    """
    QDC ağırlıklı ve threshold bazlı yöntemlerle açıları hesaplar.
    """
    step_thresholds = [0, 1, 2, 5, 10]
    linear_thresholds = [0.1, 0.2, 0.5, 1.0]
    
    # Dict yapısı eksenlere göre ayrıldı ki daha kolay okunsun
    results = {'X': {'step': {}, 'linear': {}}, 'Y': {'step': {}, 'linear': {}}}

    def get_weighted_mean(vals, sigs, threshold, mode='step'):
        if not vals or not sigs: return None
        sigs = np.array(sigs)
        
        if mode == 'step':
            weights = np.where(sigs >= threshold, sigs, 0)
        else:
            weights = np.where(sigs < threshold, threshold, sigs)
        
        if weights.sum() == 0: 
            return np.mean(vals)
        return np.sum(np.array(vals) * weights) / weights.sum()

    for axis in ['X', 'Y']:
        # Step Thresholds
        for thr in step_thresholds:
            z_pts, v_pts = [], []
            for p in range(1, 6):
                vals = plane_spatial_data[axis][p]['val']
                sigs = plane_signals[p][axis]
                
                # Güvenlik: Hit sayısı ile Signal sayısı eşit mi?
                if len(vals) > 0 and len(vals) == len(sigs):
                    val = get_weighted_mean(vals, sigs, thr, 'step')
                    z_pts.append(np.mean(plane_spatial_data[axis][p]['z']))
                    v_pts.append(val)
                    
            # Polyfit için en az 2 istasyon dolu olmalı
            results[axis]['step'][f'thr_{thr}'] = np.polyfit(z_pts, v_pts, 1)[0] if len(z_pts) >= 2 else -999.0

        # Linear Thresholds
        for thr in linear_thresholds:
            z_pts, v_pts = [], []
            for p in range(1, 6):
                vals = plane_spatial_data[axis][p]['val']
                sigs = plane_signals[p][axis]
                
                if len(vals) > 0 and len(vals) == len(sigs):
                    val = get_weighted_mean(vals, sigs, thr, 'linear')
                    z_pts.append(np.mean(plane_spatial_data[axis][p]['z']))
                    v_pts.append(val)
                    
            results[axis]['linear'][f'thr_{thr}'] = np.polyfit(z_pts, v_pts, 1)[0] if len(z_pts) >= 2 else -999.0
            
    return results

def calculate_ideal_neutrino_angle(plane_spatial_data):
    #### KULLANMAYA GEREK YOK.
    """
    LHC (IP1) noktasından gelen nötrinonun ideal geliş açısını hesaplar.
    Vertex, ilk hit'in olduğu istasyondan bir miktar geride (Tungsten'in içinde) kabul edilir.
    IP1'in dedektöre olan nominal uzaklığı Z_IP1 = -480 metre (-48000 cm)'dir.
    """
    # Z_IP1 = 0 kabul ederek lokal açıyı buluyoruz (Çünkü detektör orijini IP1 hizasındadır)
    IP1_DISTANCE_CM = 48000.0 
    
    # SND@LHC'de bir duvarın (Tungsten + boşluklar) yaklaşık kalınlığı
    # Bu değeri geo dosyasından kesin olarak kontrol edebilirsin, genelde ~10-13 cm arasıdır.
    HALF_WALL_THICKNESS = 6.5 # cm (Tahmini Tungsten merkezi) AMA YANLIŞ DEĞER. DÜZELTMEDİM ÇÜNKÜ YANLIŞ

    first_plane = None
    
    # Hangi istasyonun "ilk" vurulduğunu bul (Hem X hem Y eksenine bakarak)
    for p in range(1, 6):
        if len(plane_spatial_data['X'][p]['val']) > 0 or len(plane_spatial_data['Y'][p]['val']) > 0:
            first_plane = p
            break
            
    if first_plane is None:
        return -999.0, -999.0 # Event boş
        
    # Vertex X ve Y'si ilk istasyonun ağırlıksız aritmetik ortalaması kabul edilir
    # (Eğer o eksende hit yoksa 0 kabul et)
    if len(plane_spatial_data['X'][first_plane]['val']) > 0:
        vertex_x = np.mean(plane_spatial_data['X'][first_plane]['val'])
        z_x_hit = np.mean(plane_spatial_data['X'][first_plane]['z'])
    else:
        vertex_x = 0.0
        z_x_hit = None

    if len(plane_spatial_data['Y'][first_plane]['val']) > 0:
        vertex_y = np.mean(plane_spatial_data['Y'][first_plane]['val'])
        z_y_hit = np.mean(plane_spatial_data['Y'][first_plane]['z'])
    else:
        vertex_y = 0.0
        z_y_hit = None

    # Vertex Z: İlk hit'ten Tungsten kalınlığının yarısı kadar gerisi
    # (Hangi eksende hit bulduysak onun Z'sini kullan)
    base_z = z_x_hit if z_x_hit is not None else z_y_hit
    vertex_z = base_z - HALF_WALL_THICKNESS
    
    # Nötrinonun (veya IP1'den gelen hayali çizginin) Eğimi (dx/dz ve dy/dz)
    # Dedektör koordinat sistemi IP1'i baz aldığı için direkt vertex / mesafe yapabiliriz.
    # Ancak gerçek bir 3B hesap: Angle = (Vertex - IP1) / (Z_vertex - Z_IP1)
    
    theta_x = vertex_x / (vertex_z + IP1_DISTANCE_CM)
    theta_y = vertex_y / (vertex_z + IP1_DISTANCE_CM)
    
    return theta_x, theta_y

def combined_all_signals(tree):
    if PROCCES_ALL_EVENTS == True:
        size_tchain = tree.GetEntries()
    else:
        size_tchain = PROCCES_ALL_EVENTS

    # 1. Global lists to hold the sparse event data
    scifi_indices_all, scifi_signals_all, scifi_times_all = [], [], []
    us_indices_all, us_signals_all, us_times_all = [], [], []
    ds_h_indices_all, ds_h_signals_all, ds_h_times_all = [], [], []
    ds_v_indices_all, ds_v_signals_all, ds_v_times_all = [], [], []


    # Scifi
    sf_h_all, sf_q_all, sf_hl_all, sf_ql_all = [], [], [], []
    sf_h_tc_all, sf_q_tc_all, sf_hl_tc_all, sf_ql_tc_all = [], [], [], []
    sf_h_tc23_all, sf_q_tc23_all, sf_hl_tc23_all, sf_ql_tc23_all = [], [], [], []
    sf_h_tc22_all, sf_q_tc22_all, sf_hl_tc22_all, sf_ql_tc22_all = [], [], [], []
    sf_h_tc18_all, sf_q_tc18_all, sf_hl_tc18_all, sf_ql_tc18_all = [], [], [], []
    # US
    us_h_all, us_q_all, us_hl_all, us_ql_all = [], [], [], []
    us_h_tc_all, us_q_tc_all, us_hl_tc_all, us_ql_tc_all = [], [], [], []
    # DS Horizontal
    dsh_h_all, dsh_q_all, dsh_hl_all, dsh_ql_all = [], [], [], []
    dsh_h_tc_all, dsh_q_tc_all, dsh_hl_tc_all, dsh_ql_tc_all = [], [], [], []
    # DS Vertical
    dsv_h_all, dsv_q_all, dsv_hl_all, dsv_ql_all = [], [], [], []
    dsv_h_tc_all, dsv_q_tc_all, dsv_hl_tc_all, dsv_ql_tc_all = [], [], [], []

    run_id_all, event_number_all, event_time_all = [], [], []
    past_time_diff_all, next_time_diff_all = [], []

    mean_time_scifi_all=[]
    mean_time_us_all=[]
    mean_time_ds_all=[]

    hit_x_64=[]
    hit_y_64=[]
    hit_x_128=[]
    hit_y_128=[]

    station_hdw=[]
    max_total_hdw=[]
    max_ver_hdw=[]
    max_hor_hdw=[]
    energy_list=[]


    previous_tree_time = None
    pending_event_index = -1
    pending_current_time = None
    event_count = 0  

    for i in range(size_tchain):
        tree.GetEntry(i)

        #print("en3d ",tree.MCTrack.At(0).GetEnergy())

        SKIP_EVENT=False
        
        if not hasattr(tree, 'Digi_ScifiHits'):
            print(i)
            continue
        if not IS_MC:
            hdr = tree.EventHeader  # veya eventdigi.EventHeader
        # -------------------------------------------------------------
        # Sequential Time Tracking
        # -------------------------------------------------------------
            current_time = tree.EventHeader.GetEventTime()

            if pending_event_index != -1: 
                next_time_diff_all[pending_event_index] = current_time - pending_current_time
                pending_event_index = -1

            if previous_tree_time is None:
                time_diff_past = 999999 
            else:
                time_diff_past = current_time - previous_tree_time

            previous_tree_time = current_time

            if not IS_MC and (time_diff_past * 6.25 < 625):
                continue

        if len(tree.Digi_ScifiHits) < 10:
            continue
        if (hdr.GetBeamMode()==1):
            print("passed some cuts",hdr.GetBeamMode(),hdr.isIP1())
        if (IS_LHC_DATA ) and (hdr.GetBeamMode() != 11 or not hdr.isIP1()):
            continue
        print("passed some cuts",i)


        if hasattr(tree, 'EventHeader') and not IS_MC:
            scifi_geometry.InitEvent(tree.EventHeader)

        # -------------------------------------------------------------
        # Extract Hits (Sparse Format)
        # -------------------------------------------------------------
        evt_scifi_idx, evt_scifi_sig, evt_scifi_time = [], [], []
        event_vetoed = False
        layer_channels = {}
        plane_spatial_data = {
            'X': {1: {'val': [], 'z': []}, 2: {'val': [], 'z': []}, 3: {'val': [], 'z': []}, 4: {'val': [], 'z': []}, 5: {'val': [], 'z': []}},
            'Y': {1: {'val': [], 'z': []}, 2: {'val': [], 'z': []}, 3: {'val': [], 'z': []}, 4: {'val': [], 'z': []}, 5: {'val': [], 'z': []}}
        }
        
        plane_signals = {1: {'X': [], 'Y': []}, 2: {'X': [], 'Y': []}, 3: {'X': [], 'Y': []}, 4: {'X': [], 'Y': []}, 5: {'X': [], 'Y': []}}

        #scifiDet = scifi_geometry.modules['Scifi'] # buna gerek var mı zaten aşağıda bu şekilde tanımlandı
        for aHit in tree.Digi_ScifiHits:
            if not aHit.isValid(): continue
            detID = aHit.GetDetectorID() # DÜZELTME
            n_vert, n_plane, n_chan = scifi_array_id(detID)

            if n_plane==0:
                event_vetoed = True
                break

            sig = aHit.GetSignal()
            #print(sig)
            #if sig <= -150: continue 
            """is_vertical = (n_vert == 1)
            is_horizontal = (n_vert == 0)
            
            vert_violation = is_vertical and (n_chan <= 200 or n_chan >= 1200)
            horiz_violation = is_horizontal and (n_chan <= 250 or n_chan >= 1320)
            #find highest plane and check it with cris metod. other than that station, if there is hit on that region, reject them.
            
            if vert_violation or horiz_violation:
                event_vetoed = True
                #print("event_vetoed, passed to next event")
                break"""

            rawtime = aHit.GetTime()

            if IS_MC:
                corrected_time = rawtime
            else:
                corrected_time = scifi_geometry.GetCorrectedTime(aHit.GetDetectorID(), rawtime*shipunit.snd_TDC2ns, 0) / shipunit.snd_TDC2ns

            evt_scifi_idx.append([n_vert, n_plane, n_chan])
            evt_scifi_sig.append(sig)
            evt_scifi_time.append(corrected_time)

            layer_key = (n_plane, n_vert)
            if layer_key not in layer_channels:
                layer_channels[layer_key] = []
            layer_channels[layer_key].append(n_chan)

            """            ## positions are added
            hit_val, hit_z = get_physical_position(scifi_geometry, detID, n_vert)
            
            axis = 'X' if n_vert == 1 else 'Y'
            plane_spatial_data[axis][n_plane]['val'].append(hit_val)
            plane_spatial_data[axis][n_plane]['z'].append(hit_z)
            plane_signals[n_plane][axis].append(sig)"""
            ## positions done
        if event_vetoed: # <--- Veto yediyse event'i tamamen çöpe at
            continue
        if len(evt_scifi_idx) <= 10:
            continue
        print(i,"passed at scifi")
        valid_planes = []
        all_planes_in_event = set(plane for plane, vert in layer_channels.keys())
        
        for p in all_planes_in_event:
            if (p, 0) in layer_channels and (p, 1) in layer_channels:
                valid_planes.append(p)
                
        unique_planes = sorted(valid_planes)
        
        has_consecutive = False
        for i in range(len(unique_planes) - 1):
            if unique_planes[i+1] == unique_planes[i] + 1:
                has_consecutive = True
                break
                
        if not has_consecutive:
            continue 
        print(i,"passed consectuive")
        
        for (n_plane, n_vert), channels in layer_channels.items():
            avg_chan = sum(channels) / len(channels)
            
            is_vertical = (n_vert == 1)
            is_horizontal = (n_vert == 0)
            
            vert_violation = is_vertical and (avg_chan <= 200 or avg_chan >= 1200)
            horiz_violation = is_horizontal and (avg_chan <= 300 or avg_chan >= 1336)
            
            # Bir tane layer bile sınırı geçerse olayı reddet ve döngüyü kır
            if vert_violation or horiz_violation:
                event_vetoed = True
                break

        if event_vetoed:
            continue
        print(i,"passed fi volume")

        """advanced_angles = calculate_advanced_angles(plane_spatial_data, plane_signals)
        measured_theta_y = advanced_angles['Y']['step']['thr_2'] # Örnek olarak Thr=2 kullandık

        ideal_theta_x, ideal_theta_y = calculate_ideal_neutrino_angle(plane_spatial_data)"""

        """if abs(measured_theta_y - ideal_theta_y) > 0.020: # 20 miliradyan tolerans
            continue # Nötrino değil, kaya nötronu!"""

        evt_us_idx, evt_us_sig, evt_us_time = [], [], []
        evt_ds_h_idx, evt_ds_h_sig, evt_ds_h_time = [], [], []
        evt_ds_v_idx, evt_ds_v_sig, evt_ds_v_time = [], [], []
        evt_ds_v_raw_time,evt_ds_h_raw_time = [],[]

        for aHit in tree.Digi_MuFilterHits:
            hit_id = aHit.GetDetectorID()
            bar = (hit_id % 1000)
            hit_system = aHit.GetSystem()
            hit_plane_2 = aHit.GetPlane()

            if (IS_LHC_DATA) and (hit_system==1):
                SKIP_EVENT = True
                break ## must exit and passes to next event

            elif hit_system == 2:  # US system
                for side in range(aHit.GetnSides()):
                    for channel in range(aHit.GetnSiPMs()):
                        ch = 8 * side + channel
                        real_channel = bar * 8 + channel
                        sig = aHit.GetSignal(ch)
                        if sig > 0:
                            evt_us_idx.append([side, hit_plane_2, real_channel])
                            evt_us_sig.append(sig)
                            evt_us_time.append(aHit.GetTime(ch))

            elif hit_system == 3:  # DS system
                for side in range(aHit.GetnSides()):
                    for channel in range(aHit.GetnSiPMs()):
                        ch = 8 * side + channel
                        sig = aHit.GetSignal(ch)
                        if sig > 0:
                            if aHit.isVertical():
                                real_channel = bar - 60
                                evt_ds_v_idx.append([hit_plane_2, real_channel])
                                evt_ds_v_sig.append(sig)

                                evt_ds_v_time.append(aHit.GetTime(ch))

                                """t = aHit.GetTime(0)*shipunit.snd_TDC2ns
                                if IS_TB_Data:
                                    t = MuFi_geometry.GetCorrectedTime(aHit.GetDetectorID(), 0, t, 0)
                                evt_ds_v_time.append(t/shipunit.snd_TDC2ns)"""

                            else:
                                real_channel = bar
                                evt_ds_h_idx.append([side, hit_plane_2, real_channel])
                                evt_ds_h_sig.append(sig)

                                evt_ds_h_time.append(aHit.GetTime(ch))
                                
                                """t = [aHit.GetTime(i)*shipunit.snd_TDC2ns for i in range(2)]
                                if IS_TB_Data:
                                    for i in range(2) :
                                        t[i] = MuFi_geometry.GetCorrectedTime(aHit.GetDetectorID(), i, t[i], 0)
                                t = np.mean(t)
                                evt_ds_h_time.append(t/shipunit.snd_TDC2ns)"""
        # -------------------------------------------------------------
        # CALCULATE AND APPEND AGGREGATES
        # -------------------------------------------------------------
        # Scifi (5 layers, dim=1)
        if SKIP_EVENT:
            continue

        """if IS_LHC_DATA:
            if len(evt_us_sig) > 10:
                continue
            if len(evt_ds_h_sig) > 5:
                continue
            if len(evt_ds_v_sig) > 5:
                continue"""

        if not passes_shower_density_cut(evt_scifi_idx, evt_scifi_sig):
            continue
        ith_hit_x_64, ith_hit_y_64 = passes_shower_density_cut(evt_scifi_idx, evt_scifi_sig,64)
        ith_hit_x_128, ith_hit_y_128 = passes_shower_density_cut(evt_scifi_idx, evt_scifi_sig,128)
        ith_station_hdw, ith_max_total_hdw, ith_max_hor_hdw, ith_max_ver_hdw = calculate_hdw(evt_scifi_idx,evt_scifi_sig,40)

        hit_x_64.append(ith_hit_x_64)
        hit_y_64.append(ith_hit_y_64)
        hit_x_128.append(ith_hit_x_128)
        hit_y_128.append(ith_hit_y_128)

        station_hdw.append(ith_station_hdw)
        max_total_hdw.append(ith_max_total_hdw)
        max_ver_hdw.append(ith_max_ver_hdw)
        max_hor_hdw.append(ith_max_hor_hdw)


        mean_time_scifi = find_highest_bin(evt_scifi_time)
        mean_time_us = find_highest_bin(evt_us_time)
        mean_time_ds = find_highest_bin(evt_ds_h_time + evt_ds_v_time)


        h, q, hl, ql = get_event_aggregates(evt_scifi_idx, evt_scifi_sig, evt_scifi_time, num_layers=5, layer_dim=1)
        sf_h_all.append(h); sf_q_all.append(q); sf_hl_all.append(hl); sf_ql_all.append(ql)
        
        h, q, hl, ql = get_event_aggregates(evt_scifi_idx, evt_scifi_sig, evt_scifi_time, num_layers=5, layer_dim=1,mean_time=mean_time_scifi, time_window_low=0.5,time_window_high=0.5)
        sf_h_tc_all.append(h); sf_q_tc_all.append(q); sf_hl_tc_all.append(hl); sf_ql_tc_all.append(ql)

        h, q, hl, ql = get_event_aggregates(evt_scifi_idx, evt_scifi_sig, evt_scifi_time, num_layers=5, layer_dim=1,mean_time=mean_time_scifi, time_window_low=0.5,time_window_high=2.3)
        sf_h_tc23_all.append(h); sf_q_tc23_all.append(q); sf_hl_tc23_all.append(hl); sf_ql_tc23_all.append(ql)

        h, q, hl, ql = get_event_aggregates(evt_scifi_idx, evt_scifi_sig, evt_scifi_time, num_layers=5, layer_dim=1,mean_time=mean_time_scifi, time_window_low=0.5,time_window_high=2.2)
        sf_h_tc22_all.append(h); sf_q_tc22_all.append(q); sf_hl_tc22_all.append(hl); sf_ql_tc22_all.append(ql)

        h, q, hl, ql = get_event_aggregates(evt_scifi_idx, evt_scifi_sig, evt_scifi_time, num_layers=5, layer_dim=1,mean_time=mean_time_scifi, time_window_low=0.5,time_window_high=1.8)
        sf_h_tc18_all.append(h); sf_q_tc18_all.append(q); sf_hl_tc18_all.append(hl); sf_ql_tc18_all.append(ql)

        # US (5 layers, dim=1)
        h, q, hl, ql = get_event_aggregates(evt_us_idx, evt_us_sig, evt_us_time, num_layers=5, layer_dim=1)
        us_h_all.append(h); us_q_all.append(q); us_hl_all.append(hl); us_ql_all.append(ql)
        
        h, q, hl, ql = get_event_aggregates(evt_us_idx, evt_us_sig, evt_us_time, num_layers=5, layer_dim=1, mean_time=mean_time_us ,time_window_low=3,time_window_high=3)
        us_h_tc_all.append(h); us_q_tc_all.append(q); us_hl_tc_all.append(hl); us_ql_tc_all.append(ql)

        # DS Horizontal (3 layers, dim=1)
        h, q, hl, ql = get_event_aggregates(evt_ds_h_idx, evt_ds_h_sig, evt_ds_h_time, num_layers=3, layer_dim=1)
        dsh_h_all.append(h); dsh_q_all.append(q); dsh_hl_all.append(hl); dsh_ql_all.append(ql)
        
        #h, q, hl, ql = get_event_aggregates(evt_ds_h_idx, evt_ds_h_sig, evt_ds_h_time, num_layers=3, layer_dim=1, time_window=3.0)
        #dsh_h_tc_all.append(h); dsh_q_tc_all.append(q); dsh_hl_tc_all.append(hl); dsh_ql_tc_all.append(ql)

        # DS Vertical (4 layers, dim=0) -> Remember layer is the 0th column here!
        h, q, hl, ql = get_event_aggregates(evt_ds_v_idx, evt_ds_v_sig, evt_ds_v_time, num_layers=4, layer_dim=0)
        dsv_h_all.append(h); dsv_q_all.append(q); dsv_hl_all.append(hl); dsv_ql_all.append(ql)

        #""""h, q, hl, ql = get_event_aggregates(evt_ds_v_idx, evt_ds_v_sig, evt_ds_v_time, num_layers=4, layer_dim=0, time_window=3.0)
        #dsv_h_tc_all.append(h); dsv_q_tc_all.append(q); dsv_hl_tc_all.append(hl); dsv_ql_tc_all.append(ql)
        

        # -------------------------------------------------------------
        # Save Sparse Coordinate Arrays
        # -------------------------------------------------------------
        scifi_indices_all.append(torch.tensor(evt_scifi_idx, dtype=torch.int16))
        scifi_signals_all.append(torch.tensor(evt_scifi_sig, dtype=torch.float32))
        scifi_times_all.append(torch.tensor(evt_scifi_time, dtype=torch.float32))

        us_indices_all.append(torch.tensor(evt_us_idx, dtype=torch.int16))
        us_signals_all.append(torch.tensor(evt_us_sig, dtype=torch.float32))
        us_times_all.append(torch.tensor(evt_us_time, dtype=torch.float32))

        ds_h_indices_all.append(torch.tensor(evt_ds_h_idx, dtype=torch.int16))
        ds_h_signals_all.append(torch.tensor(evt_ds_h_sig, dtype=torch.float32))
        ds_h_times_all.append(torch.tensor(evt_ds_h_time, dtype=torch.float32))

        ds_v_indices_all.append(torch.tensor(evt_ds_v_idx, dtype=torch.int16))
        ds_v_signals_all.append(torch.tensor(evt_ds_v_sig, dtype=torch.float32))
        ds_v_times_all.append(torch.tensor(evt_ds_v_time, dtype=torch.float32))

        mean_time_scifi_all.append(float(mean_time_scifi))
        mean_time_us_all.append(float(mean_time_us))
        mean_time_ds_all.append(float(mean_time_ds))
        
        if not IS_MC:
            run_id_all.append(tree.EventHeader.GetRunId())
            event_number_all.append(tree.EventHeader.GetEventNumber())

            event_time_all.append(current_time)

            past_time_diff_all.append(time_diff_past)
            next_time_diff_all.append(0)  

            pending_event_index = event_count
            pending_current_time = current_time

        if IS_MC:
            energy_list.append(tree.MCTrack.At(0).GetEnergy())
        event_count += 1
        #print(event_count,i)
        
    # -------------------------------------------------------------
    # Package into Dictionary
    # -------------------------------------------------------------
    
    # Save to .pt file
    print("event number",len(mean_time_scifi_all))
    """print(run_id_all)
    print(station_hdw)
    print(max_total_hdw)
    print(max_hor_hdw)
    print(max_ver_hdw)
    print(hit_x_64,hit_y_64)
    print(hit_x_128,hit_y_128)"""

    event_dict = {
        # Sparse Data
        "scifi_indices": scifi_indices_all, "scifi_signals": scifi_signals_all, "scifi_hit_time": scifi_times_all,
        "us_indices": us_indices_all, "us_signals": us_signals_all, "us_signals_time": us_times_all,
        "ds_h_indices": ds_h_indices_all, "ds_h_signals": ds_h_signals_all, "ds_h_times": ds_h_times_all,
        "ds_v_indices": ds_v_indices_all, "ds_v_signals": ds_v_signals_all, "ds_v_times": ds_v_times_all,

        "scifi_mean_hit_time": mean_time_scifi_all,
        "us_mean_hit_time": mean_time_us_all,
        "ds_mean_hit_time": mean_time_ds_all,

        "scifi_hitx_in_64r":hit_x_64,
        "scifi_hity_in_64r":hit_y_64,
        "scifi_hitx_in_128r":hit_x_128,
        "scifi_hity_in_128r":hit_y_128,

        "max_total_hdw":max_total_hdw,
        "station_hdw":station_hdw,
        "max_hor_hdw":max_hor_hdw,
        "max_ver_hdw":max_ver_hdw,

        
        # Scifi Aggregates
        "scifi_notime_total_hits": torch.tensor(sf_h_all, dtype=torch.int32),
        "scifi_notime_total_qdc": torch.tensor(sf_q_all, dtype=torch.float32),
        "scifi_notime_hits_per_layer": torch.from_numpy(np.array(sf_hl_all, dtype=np.int32)),
        "scifi_notime_qdc_per_layer": torch.from_numpy(np.array(sf_ql_all, dtype=np.float32)),
        "scifi_05usualtime_total_hits": torch.tensor(sf_h_tc_all, dtype=torch.int32),
        "scifi_05usualtime_total_qdc": torch.tensor(sf_q_tc_all, dtype=torch.float32),
        "scifi_05usualtime_hits_per_layer": torch.from_numpy(np.array(sf_hl_tc_all, dtype=np.int32)),
        "scifi_05usualtime_qdc_per_layer": torch.from_numpy(np.array(sf_ql_tc_all, dtype=np.float32)),
        
        "scifi_05_18_total_hits": torch.tensor(sf_h_tc18_all, dtype=torch.int32),
        "scifi_05_18_total_qdc": torch.tensor(sf_q_tc18_all, dtype=torch.float32),
        "scifi_05_18_hits_per_layer": torch.from_numpy(np.array(sf_hl_tc18_all, dtype=np.int32)),
        "scifi_05_18_qdc_per_layer": torch.from_numpy(np.array(sf_ql_tc18_all, dtype=np.float32)),


        "scifi_05_22_total_hits": torch.tensor(sf_h_tc22_all, dtype=torch.int32),
        "scifi_05_22_total_qdc": torch.tensor(sf_q_tc22_all, dtype=torch.float32),
        "scifi_05_22_hits_per_layer": torch.from_numpy(np.array(sf_hl_tc22_all, dtype=np.int32)),
        "scifi_05_22_qdc_per_layer": torch.from_numpy(np.array(sf_ql_tc22_all, dtype=np.float32)),


        "scifi_05_23_total_hits": torch.tensor(sf_h_tc23_all, dtype=torch.int32),
        "scifi_05_23_total_qdc": torch.tensor(sf_q_tc23_all, dtype=torch.float32),
        "scifi_05_23_hits_per_layer": torch.from_numpy(np.array(sf_hl_tc23_all, dtype=np.int32)),
        "scifi_05_23_qdc_per_layer": torch.from_numpy(np.array(sf_ql_tc23_all, dtype=np.float32)),
        
        # US Aggregates
        "us_notime_total_hits": torch.tensor(us_h_all, dtype=torch.int32),
        "us_notime_total_qdc": torch.tensor(us_q_all, dtype=torch.float32),
        "us_notime_hits_per_layer": torch.from_numpy(np.array(us_hl_all, dtype=np.int32)),
        "us_notime_qdc_per_layer": torch.from_numpy(np.array(us_ql_all, dtype=np.float32)),
        "us_3usualtime_total_hits": torch.tensor(us_h_tc_all, dtype=torch.int32),
        "us_3usualtime_total_qdc": torch.tensor(us_q_tc_all, dtype=torch.float32),
        "us_3usualtime_hits_per_layer": torch.from_numpy(np.array(us_hl_tc_all, dtype=np.int32)),
        "us_3usualtime_qdc_per_layer": torch.from_numpy(np.array(us_ql_tc_all, dtype=np.float32)),

        # DS_H Aggregates
        "dsh_notime_total_hits": torch.tensor(dsh_h_all, dtype=torch.int32),
        "dsh_notime_total_qdc": torch.tensor(dsh_q_all, dtype=torch.float32),
        "dsh_notime_hits_per_layer": torch.from_numpy(np.array(dsh_hl_all, dtype=np.int32)),
        "dsh_notime_qdc_per_layer": torch.from_numpy(np.array(dsh_ql_all, dtype=np.float32)),
        #"dsh_3usualtime_total_hits": torch.tensor(dsh_h_tc_all, dtype=torch.int32),
        #"dsh_3usualtime_total_qdc": torch.tensor(dsh_q_tc_all, dtype=torch.float32),
        #"dsh_3usualtime_hits_per_layer": torch.tensor(dsh_hl_tc_all, dtype=torch.int32),
        #"dsh_3usualtime_qdc_per_layer": torch.tensor(dsh_ql_tc_all, dtype=torch.float32),

        # DS_V Aggregates
        "dsv_notime_total_hits": torch.tensor(dsv_h_all, dtype=torch.int32),
        "dsv_notime_total_qdc": torch.tensor(dsv_q_all, dtype=torch.float32),
        "dsv_notime_hits_per_layer": torch.from_numpy(np.array(dsv_hl_all, dtype=np.int32)),
        "dsv_notime_qdc_per_layer": torch.from_numpy(np.array(dsv_ql_all, dtype=np.float32)),
        #"dsv_3usualtime_total_hits": torch.tensor(dsv_h_tc_all, dtype=torch.int32),
        #"dsv_3usualtime_total_qdc": torch.tensor(dsv_q_tc_all, dtype=torch.float32),
        #"dsv_3usualtime_hits_per_layer": torch.tensor(dsv_hl_tc_all, dtype=torch.int32),
        #"dsv_3usualtime_qdc_per_layer": torch.tensor(dsv_ql_tc_all, dtype=torch.float32),
    }
    if not IS_MC:
        event_dict.update({
            "run_id": torch.from_numpy(np.array(run_id_all, dtype=np.int32)),
            "event_number": torch.from_numpy(np.array(event_number_all, dtype=np.int32)),
            "event_time": torch.from_numpy(np.array(event_time_all, dtype=np.int64)),
            "past_consecutive_time_diff": torch.from_numpy(np.array(past_time_diff_all, dtype=np.int32)),
            "next_consecutive_time_diff": torch.from_numpy(np.array(next_time_diff_all, dtype=np.int32))})
    if IS_MC:
        event_dict.update({"en3d":torch.from_numpy(np.array(energy_list, dtype=np.float32))})
    return event_dict

#FOR DATA, LOOK AT VETO, REJECT US AND DS,

IS_LHC_DATA=True
IS_TB_MC=False
IS_TB_Data=False
IS_PG_MC=False
PROCCES_ALL_EVENTS=True ## Number or True 

if IS_TB_MC or IS_PG_MC:
    IS_MC=True
else:
    IS_MC=False

MERGE_ALL_ONE_FILE=False
#f_name="nominal_entry_points"
#out_name="/eos/user/b/beturk/snd/test_beam/2024/TB_MC_2024_"+f"electrons_{f_name}"   # number of splits
#all_files = glob.glob(f"/eos/experiment/sndlhc/MonteCarlo/testbeam2024/300GeV_11/nominal_entry_points/*/*digCPP.root", recursive=True) 

if IS_LHC_DATA:
    MERGE_ALL_ONE_FILE=True # Since, we batch, we must merge, otherwise no meaning
    out_name=sys.argv[1]       # number of splits
    i = int(sys.argv[2]) 
    batch_number=i
    Divide = int(sys.argv[3])             # which batch to use (0 or 1)
    filelist_path =  sys.argv[4]   # path to the text file

    # --- Read file list ---
    with open(filelist_path, "r") as f:
        filelist = [line.strip() for line in f if line.strip()]

    N = len(filelist)
    print("total size1:", N)

    # --- Divide into batches ---
    batch_size = N // Divide
    remainder = N % Divide
    print("batch_size:", batch_size)
    print("remainder:", batch_size)

    # --- Select i-th batch ---
    start = 0
    batches = []
    for j in range(Divide):
        extra = 1 if j < remainder else 0
        end = start + batch_size + extra
        batches.append(filelist[start:end])
        print(f"Batch {j}: {len(batches[-1])} files")
        start = end

    # --- Select the batch you want ---
    if i >= Divide:
        raise ValueError(f"i={i} is out of range (0–{Divide-1})")

    all_files = batches[i]
    print(f"\nSelected batch {i} with {len(all_files)} files")
    total_files = len(all_files)

    ex_file = all_files[0]
    if "2022" in ex_file:
        geo_path ="/eos/experiment/sndlhc/convertedData/physics/2022/geofile_sndlhc_TI18_V4_2022.root"
    elif "2023" in ex_file:
        geo_path = "/eos/experiment/sndlhc/convertedData/physics/2023/geofile_sndlhc_TI18_V3_2023.root"
    elif "2024" in ex_file:
        geo_path = "/eos/experiment/sndlhc/convertedData/physics/2024/geofile_sndlhc_TI18_V12_2024.root"
    elif "2025" in ex_file:
        geo_path = "/eos/experiment/sndlhc/convertedData/physics/2025/geofile_sndlhc_TI18_V8_2025.root"
    elif "2026" in ex_file:
        geo_path = "/eos/experiment/sndlhc/convertedData/physics/2026/geofile_sndlhc_TI18_V4_2026.root"
    
    print(geo_path)
    geo = SndlhcGeo.GeoInterface(geo_path)
    scifi_geometry = geo.modules['Scifi']
    MuFi_geometry = geo.modules['MuFilter']

if IS_PG_MC:
    MERGE_ALL_ONE_FILE=True # Since, we batch, we must merge, otherwise no meaning
    out_name=sys.argv[1]       # number of splits
    i = int(sys.argv[2])
    batch_number=i
    Divide = int(sys.argv[3])             # which batch to use (0 or 1)
    filelist_path =  sys.argv[4]   # path to the text file

    # --- Read file list ---
    with open(filelist_path, "r") as f:
        filelist = [line.strip() for line in f if line.strip()]

    N = len(filelist)
    print("total size1:", N)

    # --- Divide into batches ---
    batch_size = N // Divide
    remainder = N % Divide
    print("batch_size:", batch_size)
    print("remainder:", batch_size)

    # --- Select i-th batch ---
    start = 0
    batches = []
    for j in range(Divide):
        extra = 1 if j < remainder else 0
        end = start + batch_size + extra
        batches.append(filelist[start:end])
        print(f"Batch {j}: {len(batches[-1])} files")
        start = end

    # --- Select the batch you want ---
    if i >= Divide:
        raise ValueError(f"i={i} is out of range (0–{Divide-1})")

    all_files = batches[i]
    print(f"\nSelected batch {i} with {len(all_files)} files")
    total_files = len(all_files)


if IS_TB_Data:
    print("Runing for the TB data")
    run_number=sys.argv[1]
    particle_name_beam_energy = sys.argv[2]
    which_TB_year = sys.argv[3]
    print(run_number)
    print(particle_name_beam_energy)
    print(which_TB_year)

    if which_TB_year=="2023":
        if "3Fe" in particle_name_beam_energy:
            wall_number = "3walls.root"
        elif "2Fe" in particle_name_beam_energy:
            wall_number = "2walls.root"
        elif "1Fe" in particle_name_beam_energy:
            wall_number = "1wall.root"
        
        print("wall number for 2023 TB",wall_number )
        out_name="/eos/experiment/sndlhc/users/beturk/TB/TB_Data/2023/"+f"scifi_us_ds_2023_{particle_name_beam_energy}_{run_number}"      # number of splits
        all_files = glob.glob(f"/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/{run_number}/*.root", recursive=True)
        geo_path = f"/eos/experiment/sndlhc/convertedData/commissioning/testbeam_June2023_H8/geofile_sndlhc_H8_2023_"+wall_number
        # geo files is only 1,2,3walls

    elif which_TB_year=="2024":
        if "electron" in particle_name_beam_energy:
            wall_number= "geofile_sndlhc_H4_2024_W_2walls.root"
        if "W" in particle_name_beam_energy:
            wall_number= "geofile_sndlhc_H4_2024_W_2walls.root"
        elif "1Fe" in particle_name_beam_energy:
            wall_number = "geofile_sndlhc_H4_2024_Fe_1wall.root"
        print(particle_name_beam_energy)
        print("wall number for 2024 TB",wall_number )
        out_name="/eos/experiment/sndlhc/users/beturk/TB/TB_Data/2024/"+f"scifi_us_ds_2024_{particle_name_beam_energy}_{run_number}"      # number of splits
        all_files = glob.glob(f"/eos/experiment/sndlhc/convertedData/commissioning/testbeam_24/{run_number}/*.root", recursive=True)
        geo_path = "/eos/experiment/sndlhc/convertedData/commissioning/testbeam_24/"+wall_number

    geo = SndlhcGeo.GeoInterface(geo_path)
    scifi_geometry = geo.modules['Scifi']
    MuFi_geometry = geo.modules['MuFilter']

if IS_TB_MC:
    print("Running for the TB MC")
    nominal_or_not=sys.argv[1]
    which_TB_year = sys.argv[3]
    particle_name_beam_energy = sys.argv[2]
    print(nominal_or_not)
    print(particle_name_beam_energy)
    print(which_TB_year)

    if which_TB_year=="2023":
        out_name="/eos/experiment/sndlhc/users/beturk/TB/TB_MC/2024/"+f"OLD_TB_MC_scifi_us_ds_2023_{particle_name_beam_energy}"      # number of splits
        all_files = glob.glob(f"/eos/experiment/sndlhc/MonteCarlo/testbeam2023/{particle_name_beam_energy}/*CPP.root", recursive=True)
    
    elif which_TB_year=="2024":
        all_files = glob.glob(f"/eos/experiment/sndlhc/MonteCarlo/testbeam2024/{particle_name_beam_energy}/{nominal_or_not}/*/*CPP.root", recursive=True)
        out_name="/eos/experiment/sndlhc/users/beturk/TB/TB_MC/2024/"+f"OLD_TB_MC_scifi_us_ds_2024_{particle_name_beam_energy}_{nominal_or_not}"      # number of splits
        print(all_files)
        print(len(all_files))

#all_files=all_files[:1]
file_length = len(all_files)
print("total OF FİLES size1:", file_length)


if IS_MC:
    TREE_NAME="cbmsim"
else:
    TREE_NAME="rawConv"
print("tree name is ",TREE_NAME )

if MERGE_ALL_ONE_FILE:
    total_files = len(all_files)
    print(f"Found {total_files} files")
    tchain = ROOT.TChain(TREE_NAME)
    good_files = 0
    deleted_files = 0
    for j, file in enumerate(all_files, 1):
        try:
            f = ROOT.TFile.Open(file, "READ")
            print(f.Get(TREE_NAME))
            if not f or f.IsZombie():
                print(f"[DELETE] Zombie or cannot open: {file}")
                deleted_files += 1
                continue

            tree1=f.Get(TREE_NAME)
            print("tree1",tree1)
            if f.TestBit(ROOT.TFile.kRecovered) or not tree1:
                print(f"[DELETE] Missing StreamerInfo or cbmsim or rawConv be careful: {file}")
                f.Close()
                deleted_files += 1
                continue
            
            if not tree1.GetBranch("Digi_ScifiHits") or not tree1.GetBranch("Digi_MuFilterHits"):
                print(f"[DELETE] Missing Digi_ScifiHits or Digi_MuFilterHits: {file}")
                f.Close()
                deleted_files += 1
                continue

            print(f"[GOOD] Added to chain: {file}")
            tchain.Add(file)
            N_events = tchain.GetEntries()
            #print("N events:", N_events)
            good_files += 1
            f.Close()

        except Exception as e:
            print(f"[DELETE] Exception {e} for {file}")
            deleted_files += 1



    print(f"\n✅ Added {good_files} good files to chain")
    print(f"🗑️ Deleted {deleted_files} bad files")
    N_events = tchain.GetEntries()
    print("N events:", N_events)
    timex=time.time()
    save_dict = combined_all_signals(tchain)
    timey=time.time()
    print("total time", timex-timey, "average time", (timey-timex)/(N_events+1))
    if IS_LHC_DATA or IS_PG_MC:
        torch.save(save_dict, f"{out_name}_{batch_number}.pt")
        print("saved",  f"{out_name}_{batch_number}.pt")
    else:
        torch.save(save_dict, f"{out_name}_all_files.pt")
        print("saved,",  f"{out_name}_all_files.pt")
    print("finished and saved to:",f"{out_name}_9.pt")


else:
    for i in range(len(all_files)):
        print("\n\n\n\n",i)
        file = all_files[i]
        print("total size1:", len(all_files))
        print(all_files)
        tchain = ROOT.TChain(TREE_NAME)
        good_files = 0
        deleted_files = 0    
        try:
            f = ROOT.TFile.Open(file, "READ")
            if not f or f.IsZombie():
                print(f"[DELETE] Zombie or cannot open: {file}")
                deleted_files += 1
                continue
            # require both StreamerInfo and cbmsim tree
            tree1=f.Get(TREE_NAME)
            print("tree1",tree1)
            if f.TestBit(ROOT.TFile.kRecovered) or not tree1:
                print(f"[DELETE] Missing StreamerInfo or cbmsim or rawConv be careful: {file}")
                f.Close()
                deleted_files += 1
                continue
            
            if not tree1.GetBranch("Digi_ScifiHits") or not tree1.GetBranch("Digi_MuFilterHits"):
                print(f"[DELETE] Missing Digi_ScifiHits or Digi_MuFilterHits: {file}")
                f.Close()
                deleted_files += 1
                continue

            print(f"[GOOD] Added to chain: {file}")
            tchain.Add(file) 
            good_files += 1
            f.Close()

        except Exception as e:
            print(f"[DELETE] Exception {e} for {file}")
            deleted_files += 1

        print(f"\n✅ Added {good_files} good files to chain")
        print(f"🗑️ Deleted {deleted_files} bad files")
        
        N_events = tchain.GetEntries()
        print("N events:", N_events)
        save_dict = combined_all_signals(tchain)
        torch.save(save_dict, f"{out_name}_{i}.pt")
        print("finished and saved to:",f"{out_name}_{i}.pt")

