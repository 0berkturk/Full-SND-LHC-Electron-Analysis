import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import os
import numpy
import torch.nn.functional as F

# just use one of them. otherwise, it will call another prop.
#from TB_hit_numbers_electrons.config_TB_hits_electrons import *
#from TB_QDC_electrons.config_TB_QDC_electrons import *
from TB_HDW_electrons.config_TB_HDW_electrons import *
#from TB_width_shower_fraction_electron.config_TB_frac_width_shower_electrons import *
#from TB_QDC_hadrons.config_TB_QDC_hadrons import *
#from TB_hit_numbers_hadrons.config_TB_hits_hadrons import *
#from TB_HDW_hadrons.config_TB_HDW_hadrons import *


cmap = plt.get_cmap('plasma')
cmap.set_under('white')
def hdw_all_fast_conv(scifi_qdc, us=None,ds=None):
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
            print("wi shape",wi.shape)
            plane_hdw[:, plane, station] = wi.sum(dim=-1)

    station_hdw = plane_hdw.sum(dim=1)          # (N, 5)
    print(station_hdw.shape)
    event_hdw, max_index = station_hdw.max(dim=1)

    idx = max_index.view(N, 1, 1).expand(-1, 2, 1)   # (N,2,1)

    plane_hdw_max = plane_hdw.gather(dim=2, index=idx).squeeze(2)

    hor_hdw = plane_hdw_max[:, 0]
    ver_hdw = plane_hdw_max[:, 1]

    return station_hdw, event_hdw, hor_hdw, ver_hdw

def find_shower_starting_layer(scifi_data, threshold=50):
    layer_number = torch.zeros(scifi_data.shape[0], dtype=torch.int64)
    layer_1_index = scifi_data[:,:,0,:].sum(dim=(1,2)) > threshold
    layer_number[layer_1_index] = 1
    print("Layer 1 index sum:", layer_1_index.sum().item())

    layer_2_index = (scifi_data[:,:,1,:].sum(dim=(1,2)) > threshold) & (layer_number == 0)
    layer_number[layer_2_index] = 2
    print("Layer 2 index sum:", layer_2_index.sum().item())

    layer_3_index = (scifi_data[:,:,2,:].sum(dim=(1,2)) > threshold) & (layer_number == 0)
    layer_number[layer_3_index] = 3
    print("Layer 3 index sum:", layer_3_index.sum().item())

    layer_4_index = (scifi_data[:,:,3,:].sum(dim=(1,2)) > threshold) & (layer_number == 0)
    layer_number[layer_4_index] = 4
    print("Layer 4 index sum:", layer_4_index.sum().item())

    layer_5_index = (scifi_data[:,:,4,:].sum(dim=(1,2)) > threshold) & (layer_number == 0)
    layer_number[layer_5_index] = 5
    print("Layer 5 index sum:", layer_5_index.sum().item())

    return layer_number

def find_shower_max(data):
    # shower maximum by mean ? then theere is angle
    #DATA is Nx2x5x1536
    #first find layer, then go that layer find max x and y.
    summed_xy= torch.sum(data,(1,3))
    max_point_layer = torch.argmax(summed_xy,dim=1)
    batch_size = data.size(0)
    batch_indices = torch.arange(batch_size)

    data_at_z = data[batch_indices,:,max_point_layer,:] ## 2x1536
    max_point_hor = torch.argmax(data_at_z[:,0],dim=1)
    max_point_ver = torch.argmax(data_at_z[:,1],dim=1)
    return max_point_layer, max_point_hor, max_point_ver


def scifi_qdc_distr(scifi,label_str):
    z,h,v = find_shower_max(scifi)
    x = range(1,6)
    plt.figure()
    for i in range(5):
        ith_layer_index = z==i
        total_qdc = torch.sum(scifi[ith_layer_index],(1,2,3))
        layer_qdc_dist = torch.mean(torch.sum(scifi[ith_layer_index],(1,3))/total_qdc.unsqueeze(1),dim=0)
        print(layer_qdc_dist.shape)
        plt.plot(x, layer_qdc_dist,label=f"Shower Max. is at Layer {i+1}")
    plt.legend()
    plt.xlabel('SciFi Layer Number', fontsize=12)
    plt.ylabel('QDC Fraction', fontsize=12)
    plt.title('SCI-FI QDC Distribution in SciFi Layers', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(x)
    plt.tight_layout()
    outdir = "shower_max_scifi_dist"
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/{label_str}_shower_max_distr.png",dpi=300)
    plt.close()

def call_cluster_size(v,width):
    image_pos_ver_big = (v + width).clamp(max=1535)
    index_at_big_border = image_pos_ver_big==1535

    image_pos_ver_small = (v - width).clamp(min=0)
    index_at_small_border = image_pos_ver_small==0

    image_pos_ver_small[index_at_big_border] = 1535-2*width
    image_pos_ver_big[index_at_small_border] = 2*width
    return image_pos_ver_big, image_pos_ver_small

def call_cluster_image(data,width, v, h, batch_size):
    ver_pos_big,ver_pos_small = call_cluster_size(v,width)
    hor_pos_big, hor_pos_small = call_cluster_size(h,width)
    
    cluster_data=torch.zeros(batch_size,2,5,2*width)
    for i in range(batch_size):
       # print(data[i,1,:, hor_pos_small[i]:hor_pos_big[i] ].shape)
        #print(hor_pos_small[i],hor_pos_big[i])
        cluster_data[i,1,:,:] = data[i,1,:, ver_pos_small[i]:ver_pos_big[i] ]
        cluster_data[i,0,:,:] = data[i,0,:, hor_pos_small[i]:hor_pos_big[i] ]

    sum_in_xy = torch.sum(cluster_data,(1,2,3))
    total_qdc = torch.sum(data[:batch_size],(1,2,3))
    frac_layer = sum_in_xy/total_qdc
    return frac_layer

def calc_fraction_of_shower_wtr_pmt_width(scifi,arb1,arb2):
    z,h,v = find_shower_max(scifi)
    batch_size = scifi.size(0)
    batch_indices = torch.arange(batch_size)
    #data_at_z = data[batch_indices,:,z]
    width = [10, 50, 100,200, 300, 400, 600,700,768] # vary this and see frac of qdc in there
    frac_layer = np.array([call_cluster_image(scifi, w, v, h, batch_size) for w in width])
    return frac_layer,width,0,0



def get_shower_cluster(data,width):# widht is radius
    batch_size = data.size(0)
    _, h, v = find_shower_max(data)

    ver_pos_big,ver_pos_small = call_cluster_size(v,width)
    hor_pos_big, hor_pos_small = call_cluster_size(h,width)
    
    cluster_data=torch.zeros(batch_size,2,5,2*width)
    for i in range(batch_size):
       # print(data[i,1,:, hor_pos_small[i]:hor_pos_big[i] ].shape)
        #print(hor_pos_small[i],hor_pos_big[i])
        cluster_data[i,1,:,:] = data[i,1,:, ver_pos_small[i]:ver_pos_big[i] ]
        cluster_data[i,0,:,:] = data[i,0,:, hor_pos_small[i]:hor_pos_big[i] ]

    return cluster_data

def find_highest_bin(scifi_timehits, xmin, xmax, N):

    bin_centers = []
    bins = np.linspace(xmin, xmax, N)

    for i in range(scifi_timehits.shape[0]):
        data = scifi_timehits[i].reshape(-1).cpu().numpy()

        # optional: ignore zero / invalid hits
        data = data[data > 0]
        if len(data) == 0:
            bin_centers.append(0.0)
            continue

        counts, bin_edges = np.histogram(data, bins=bins)
        max_bin = np.argmax(counts)

        center = 0.5 * (bin_edges[max_bin] + bin_edges[max_bin + 1])
        bin_centers.append(center)
    if len(scifi_timehits) > 0:
        plt.clf()
        plt.hist(data, bins=bins)
        plt.savefig("hittime.png")


    return bin_centers

def apply_test_beam_cuts(scifi,past_consecutive_time_diff,fname):
    scifi = get_shower_cluster(scifi,cluster_radius)

    hit_cut = ((scifi[:,0]>0).sum((1,2)) > hitx) & ((scifi[:,1]>0).sum((1,2)) > hity) ## 50 gev el: 7,  250gev 25
    if "TB_MC" in fname:
        print("it is mc, passibg consc. time diff cut",fname)
        time_cut = torch.ones_like(hit_cut)
    else:
        time_cut = past_consecutive_time_diff>150   ## plot histogram from scifi_timehits find max bin, reject events smaller than 3ns.
    return hit_cut & time_cut

def pmt_cut_inside_event(signals, times, min_qdc, t_min, t_max):
    # 1. Clean Inputs: Remove noise (low/negative QDC)
    if INCLUDE_NEG_QDC:
        positive_mask = signals !=0
    else:
        positive_mask=signals>min_pmt_qdc_value

    signals = signals * positive_mask
    times = times * positive_mask

    # 2. Calculate Mean Time
    # (Assumes find_highest_bin returns a tensor of shape (N,))
    mean_time = find_highest_bin(times, 0, 16, 100)
    
    # --- AUTO-RESHAPING FOR BROADCASTING ---
    # This automatically fits 3D (Vertical) or 4D (Horizontal/Scifi) inputs
    # If signals is (N, 2, 3, 60), view_shape becomes (-1, 1, 1, 1)
    # If signals is (N, 4, 60),    view_shape becomes (-1, 1, 1)
    view_shape = [-1] + [1] * (signals.ndim - 1)
    
    mean_time = torch.as_tensor(
        mean_time, device=signals.device, dtype=times.dtype
    ).view(*view_shape)

    # 3. Define Time Window
    t_start = torch.clamp_min(mean_time - t_min, 0)
    t_end = mean_time + t_max

    # 4. Apply Time Cut
    time_mask = (times > t_start) & (times < t_end)
    signals = signals * time_mask

    return signals


def reject_events_that_leaked_to_us_ds(us, ds_horizontal,ds_vertical):
    # US mask
    if IS_THERE_US_IN_DATA and REJECT_US:
        zero_mask_us = torch.logical_not(us.any(dim=(1, 2, 3)))
    else:
        zero_mask_us = torch.ones(len(ds_horizontal), dtype=torch.bool, device=ds_horizontal.device)

    # DS mask
    if IS_THERE_DS_IN_DATA and REJECT_DS:
        print("ds shapes",ds_horizontal.shape, ds_vertical.shape)
        zero_mask_ds = (
                torch.logical_not(ds_horizontal.any(dim=(1, 2, 3)))
                & torch.logical_not(ds_vertical.any(dim=(1, 2)))
            )
    else:
        zero_mask_ds = torch.ones(len(ds_horizontal), dtype=torch.bool, device=ds_horizontal.device)

    return zero_mask_us & zero_mask_ds


def plot_2d_hist(true_en_list, qdc_energy_list,
                 x_min, x_max, y_min, y_max, bins_x=50, bins_y=50,
                 out_name="qdc_vs_true", xlabel="True Energy [GeV]",
                 ylabel='QDC Energy [GeV]', title="QDC vs True Energy",
                 outdir="qdc_comparison"):
    
    plt.figure(figsize=(8,6))

    # Binleri hesapla
    bins_x_arr = np.linspace(x_min, x_max, bins_x+1)
    bins_y_arr = np.linspace(y_min, y_max, bins_y+1)

    for j in range(len(qdc_energy_list)):
        true_en = true_en_list[j]
        qdc_en = qdc_energy_list[j]

        # 2D histogram
        hist = plt.hist2d(true_en, qdc_en, bins=[bins_x_arr, bins_y_arr],
                          cmap=cmap, alpha=0.8, vmin=1)  # vmin=1 ile 0 değerleri alt değer sayılır

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar()
    plt.grid(True)

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    plt.savefig(f"{outdir}/new_{out_name}.png", dpi=300)
    plt.clf()


def plot_qdc_energy(true_en_list, qdc_energy_list,us_list,ds_list,label_list,bins ,out_name):
    plt.figure()
# Sum over channels, width, and height dimensions (dimensions 1, 2, 3)
    for j in range(len(qdc_energy_list)):
        scifi=qdc_energy_list[j]
        us=us_list[j]
        ds=ds_list[j]
        true_en=true_en_list[j]

        total_qdc_scifi = scifi.sum(dim=(1, 2, 3))
        total_qdc_us = us.sum(dim=(1, 2, 3))
        total_qdc_ds = ds.sum(dim=(1, 2, 3))
        total_qdc = total_qdc_scifi + total_qdc_us + total_qdc_ds

        average_qdc = []
        average_scifi_qdc = []
        average_us_qdc = []
        average_ds_qdc = []

        std_qdc = []
        std_scifi_qdc = []
        std_us_qdc = []
        std_ds_qdc = []
        for i in range(len(bins)-1):
            en_min = bins[i]
            en_max = bins[i+1]
            index = (true_en>=en_min) & (true_en<en_max)
            average_qdc.append(total_qdc[index].mean().item())
            average_scifi_qdc.append(total_qdc_scifi[index].mean().item())
            average_us_qdc.append(total_qdc_us[index].mean().item())
            average_ds_qdc.append(total_qdc_ds[index].mean().item())
            std_qdc.append(total_qdc[index].std().item())
            std_scifi_qdc.append(total_qdc_scifi[index].std().item())
            std_us_qdc.append(total_qdc_us[index].std().item())
            std_ds_qdc.append(total_qdc_ds[index].std().item())


        plt.errorbar((bins[:-1]+bins[1:])/2, average_qdc, yerr=std_qdc, fmt='o-', label=f'Total QDC ({label_list[j]})',alpha=0.7)
        plt.errorbar((bins[:-1]+bins[1:])/2, average_scifi_qdc, yerr=std_scifi_qdc, fmt='s-', label=f'SciFi QDC({label_list[j]})',alpha=0.7)
        plt.errorbar((bins[:-1]+bins[1:])/2, average_us_qdc, yerr=std_us_qdc, fmt='^-', label=f'US QDC({label_list[j]})',alpha=0.7)
        plt.errorbar((bins[:-1]+bins[1:])/2, average_ds_qdc, yerr=std_ds_qdc, fmt='d-', label=f'DS QDC({label_list[j]})',alpha=0.7)

    
    plt.xlabel('True Energy [GeV]')
    plt.ylabel('Average QDC')
    plt.title(f'Average QDC vs True Energy')
    plt.legend()
    plt.grid()
    if os.path.exists("qdc_comparision") is False:
        os.mkdir("qdc_comparision")
    plt.savefig("qdc_comparision/"+out_name+"_average_qdc_energy.png", dpi=300)
    plt.clf()

    return total_qdc



def plot_multiple_hist(qdc_energy_list,N,xmin,xmax,x_label, title, label_str, outdir,alpha_list,name="EMPTY"):
    for i in range(len(qdc_energy_list)):
        qdc_energy=qdc_energy_list[i]
        if N!=None:
            bins = np.linspace(xmin, xmax, N)
            plt.hist(qdc_energy,bins=bins,label=label_str[i],alpha=alpha_list[i],histtype='step',density=True )
            plt.xlim(xmin, xmax)
        else:
            plt.hist(qdc_energy,bins=40,label=label_str[i],alpha=alpha_list[i],histtype='step',density=True )

    
        # Log scale
    plt.yscale('log')
    plt.xlabel(x_label)
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    # Save
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/{name}.png", dpi=300)
    plt.close()

def plot_1d_graphs(true_en_list, qdc_energy_list, label_list,bins,out_name ,xlabel="True Energy [GeV]",ylabel='Average QDC Energy[GeV]',title="Average QDC vs True Energy",outdir="qdc_comparision",show_ideal=False):
    plt.figure()
# Sum over channels, width, and height dimensions (dimensions 1, 2, 3)
    for j in range(len(qdc_energy_list)):
        total_qdc_scifi=qdc_energy_list[j]
        true_en=true_en_list[j]
        average_scifi_qdc = []

        std_scifi_qdc = []

        for i in range(len(bins)-1):
            en_min = bins[i]
            en_max = bins[i+1]
            index = (true_en>=en_min) & (true_en<en_max)
            average_scifi_qdc.append(total_qdc_scifi[index].mean().item())
            std_scifi_qdc.append(total_qdc_scifi[index].std().item())

        plt.errorbar((bins[:-1]+bins[1:])/2, average_scifi_qdc, yerr=std_scifi_qdc, fmt='s-', label=f'SciFi QDC({label_list[j]})',alpha=0.7)
    if show_ideal:
        ax = plt.gca()
        ax.axline((0, 0), slope=1, linestyle='--', color='black', label='Ideal')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    if os.path.exists(outdir) is False:
        os.mkdir(outdir)
    plt.savefig(f"{outdir}/{out_name}_{title}.png", dpi=300)
    plt.clf()

def plot_1d_beam_energy_graphs(true_en_list, qdc_energy_list, label_list,out_name ,xlabel="True Energy [GeV]",ylabel='Average QDC Energy[GeV]',title="Average QDC vs True Energy",outdir="qdc_comparision",show_ideal=False):
    plt.figure()
    # Sum over channels, width, and height dimensions (dimensions 1, 2, 3)

    for j in range(len(qdc_energy_list)):
        total_qdc_scifi=qdc_energy_list[j].float()
        true_en=true_en_list[j]

        average_scifi_qdc = []
        std_scifi_qdc = []

        average_scifi_qdc.append(total_qdc_scifi.mean().item())
        std_scifi_qdc.append(total_qdc_scifi.std().item())

        plt.errorbar(true_en, average_scifi_qdc, yerr=std_scifi_qdc, fmt='s-', label=label_list[j],alpha=0.7)

    if show_ideal:
        ax = plt.gca()
        ax.axline((0, 0), slope=1, linestyle='--', color='black', label='y=x')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=6)
    plt.grid()
    if os.path.exists(outdir) is False:
        os.mkdir(outdir)
    plt.savefig(f"{outdir}/{out_name}_{title}.png", dpi=300)
    plt.clf()


def plot_1d_compare_2_domains_beam_energy_graphs(true_en_list, first_index_layer_second_index_energy_array, label_list,out_name ,xlabel="True Energy [GeV]",ylabel='Average QDC Energy[GeV]',title="Average QDC vs True Energy",outdir="qdc_comparision",show_ideal=False):
    plt.figure()
    # Sum over channels, width, and height dimensions (dimensions 1, 2, 3)
    for k,qdc_energy_list in enumerate(first_index_layer_second_index_energy_array):

        average_scifi_qdc = []
        std_scifi_qdc = []

        for j in range(len(qdc_energy_list)):
            total_qdc_scifi = qdc_energy_list[j].float()
            true_en = true_en_list

            average_scifi_qdc.append(total_qdc_scifi.mean().item())
            std_scifi_qdc.append(total_qdc_scifi.std().item())
        plt.errorbar(true_en, average_scifi_qdc, yerr=std_scifi_qdc, fmt='s-', label=label_list[k],alpha=0.7)

    

    if show_ideal:
        ax = plt.gca()
        ax.axline((0, 0), slope=1, linestyle='--', color='black', label='y=x')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=6)
    plt.grid()
    if os.path.exists(outdir) is False:
        os.mkdir(outdir)
    plt.savefig(f"{outdir}/{out_name}_{title}.png", dpi=300)
    plt.clf()


def plot_1d_compare_3_domains_beam_energy_graphs(true_en_list, mc_data_first_index, label_list,out_name ,xlabel="True Energy [GeV]",ylabel='Average QDC Energy[GeV]',title="Average QDC vs True Energy",outdir="qdc_comparision",show_ideal=False,skip_some=list):
    plt.figure()
    # Sum over channels, width, and height dimensions (dimensions 1, 2, 3)
    a=0
    for first_index_layer_second_index_energy_array in mc_data_first_index:
        for k,qdc_energy_list in enumerate(first_index_layer_second_index_energy_array):

            average_scifi_qdc = []
            std_scifi_qdc = []

            for j in range(len(qdc_energy_list)):
                total_qdc_scifi = qdc_energy_list[j].float()
                true_en = true_en_list

                average_scifi_qdc.append(total_qdc_scifi.mean().item())
                std_scifi_qdc.append(total_qdc_scifi.std().item())
            if skip_some[a]==0:
                plt.errorbar(true_en, average_scifi_qdc, yerr=std_scifi_qdc, fmt='s-', label=label_list[a],alpha=0.7)
            a+=1

    

    if show_ideal:
        ax = plt.gca()
        ax.axline((0, 0), slope=1, linestyle='--', color='black', label='y=x')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=6)
    plt.grid()
    if os.path.exists(outdir) is False:
        os.mkdir(outdir)
    plt.savefig(f"{outdir}/{out_name}_{title}.png", dpi=300)
    plt.clf()


def calculate_qdc_energy(scifi, use_us, use_ds):

    # --- SciFi ---
    scifi_energy = scifi.sum((1,3)) * SCIFI_QDC_2_GEV

    # --- US ---
    if use_us is False:
        us_energy = torch.zeros_like(scifi_energy)
        print("NOT USING US")
    else:
        us_energy = use_us.sum((1,3)) * US_DS_QDC_2_GEV

    # --- DS ---
    if use_ds is False:
        ds_hor_energy = torch.zeros_like(scifi_energy)
        ds_ver_energy = torch.zeros_like(scifi_energy)
        print("NOT USING DS")
    else:
        ds_horizontal, ds_vertical = use_ds

        ds_hor_energy = ds_horizontal.sum((1,3)) * US_DS_QDC_2_GEV
        ds_ver_energy = ds_vertical.sum((2)) * US_DS_QDC_2_GEV

    return scifi_energy, us_energy, ds_hor_energy, ds_ver_energy


def calculate_hit_number(scifi,use_us,use_ds):
    if INCLUDE_NEG_QDC:
        scifi=scifi!=0
    else:
        scifi=scifi>min_pmt_qdc_value

    scifi_energy = scifi.sum((1,3))

    if use_us is False:
        us_energy=torch.zeros_like(scifi_energy)
        print("NOT USING US")
    else:
        use_us=use_us>0
        us_energy = use_us.sum((1,3))

    if use_ds is False:
        ds_energy=torch.zeros_like(scifi_energy)
        print("NOT USING DS")
    else:
        ds_horizontal,ds_vertical = use_ds
        ds_horizontal=ds_horizontal>0
        ds_vertical=ds_vertical>0
        ds_hor_energy =  ds_horizontal.sum((1,3)) # side,layer,pmt
        ds_ver_energy= ds_vertical.sum((2)) # layer,pmt
    return scifi_energy, us_energy, ds_hor_energy, ds_ver_energy

def load_as_lists(file_list,N):
    scifi_list = []
    us_list = []
    ds_hor_list = []
    ds_ver_list = []

    for fname in file_list:
        fname, use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us = fname
        print(fname)
        save_name, ext = os.path.splitext(fname)
        if scifi_cluster_radius!=None:
            save_name=save_name+"_Radius_"+str(scifi_cluster_radius)

        if PLOT_MEAN_QDC_ENERGY:
            save_name= save_name+"qdc_"+str(time_window_min)+"_"+str(time_window_max) +pos_and_neg_qdc_name+".pt"
            calculate_shower_prop = calculate_qdc_energy
        elif PLOT_MEAN_HIT_NUMBERS:
            save_name= save_name +"hit_numbers_"+str(time_window_min)+"_"+str(time_window_max) +pos_and_neg_qdc_name+".pt"
            calculate_shower_prop = calculate_hit_number
        elif PLOT_HDW:
            save_name= save_name +f"HDW_{str(HDW_CHANNEL)}_"+str(time_window_min)+"_"+str(time_window_max) +pos_and_neg_qdc_name+".pt"
            calculate_shower_prop = hdw_all_fast_conv
        
        elif PLOT_FRACTION_SHOWER_WIDTH:
            calculate_shower_prop = calc_fraction_of_shower_wtr_pmt_width

        if os.path.exists(save_name):
            data = torch.load(save_name)
            scifi_prop, us_prop, ds_hor_prop,ds_ver_prop = data["scifi"],data["us"],data["ds_hor"],data["ds_ver"]
        

        else:
            first_data = torch.load(fname, map_location="cpu", mmap=True)
            data={}
            for key in first_data:
                data[key] = first_data[key][:N].clone()
            print("1",data[key].shape)
            scifi = data["scifi_signals"]
            scifi_hit_time = data["scifi_hit_time"]
            past_consecutive_time_diff = data["past_consecutive_time_diff"]*6.25

            if use_us:
                us_signals=data["us_signals"]
                us_signals_time=data["us_signals_time"]
            
            if use_ds:
                ds_horizontal = data["ds_horizontal"]
                ds_horizontal_time=data["ds_horizontal_time"]

                ds_vertical=data["ds_vertical"]
                ds_vertical_time=data["ds_vertical_time"]

            ## cuts
            scifi  = pmt_cut_inside_event(scifi,scifi_hit_time,min_pmt_qdc_value, time_window_min, time_window_max)
            cut = apply_test_beam_cuts(scifi, past_consecutive_time_diff,fname)
            scifi=scifi[cut]#[:,:,1:3]
            print("2",scifi.shape)
            #next_consecutive_time_diff=next_consecutive_time_diff[cut]
            past_consecutive_time_diff=past_consecutive_time_diff[cut]
            if use_ds:
                ds_horizontal=ds_horizontal[cut]
                ds_horizontal_time=ds_horizontal_time[cut]
                ds_vertical=ds_vertical[cut]
                ds_vertical_time=ds_vertical_time[cut]
                ds_horizontal = pmt_cut_inside_event(ds_horizontal, ds_horizontal_time, min_pmt_qdc_value_us, time_window_min_us, time_window_max_us)
                ds_vertical = pmt_cut_inside_event(ds_vertical, ds_vertical_time, min_pmt_qdc_value_us, time_window_min_us, time_window_max_us)
                use_ds = (ds_horizontal,ds_vertical)

            if use_us:
                us_signals=us_signals[cut]
                us_signals_time=us_signals_time[cut]
                use_us = pmt_cut_inside_event(us_signals,us_signals_time, min_pmt_qdc_value_us, time_window_min_us, time_window_max_us)
                ## use_us is equal to us_signals. it is not mistake. dont delete

            ##calculating shower prop.
            if scifi_cluster_radius!=None:
                scifi = get_shower_cluster(scifi, scifi_cluster_radius)
                
            scifi_prop, us_prop, ds_hor_prop, ds_ver_prop = calculate_shower_prop(scifi, use_us, use_ds)
            dict_energy={}
            # names are wrong for hdw, it is not us ds etc.
            dict_energy["scifi"] = scifi_prop
            dict_energy["us"] = us_prop
            dict_energy["ds_hor"] = ds_hor_prop
            dict_energy["ds_ver"] = ds_ver_prop

            if PLOT_FRACTION_SHOWER_WIDTH==False: 
                torch.save(dict_energy , save_name)

        print(scifi_prop, scifi_prop.shape, ds_hor_prop, ds_ver_prop ,"\n")
        scifi_list.append(scifi_prop)
        us_list.append(us_prop)
        ds_hor_list.append(ds_hor_prop)
        ds_ver_list.append(ds_ver_prop)

    return scifi_list, us_list, ds_hor_list,ds_ver_list

def plot_for_frac_width_shower(TEST_DATA_DIR, name, IS_MC_DATA_TITLE):
    x_min, x_max=0,1300
    y_min, y_max=0,6000
    bins_x, bins_y = 40,40

    N=50000
    energies_width_fraction,list2,list3,list4 = load_as_lists(TEST_DATA_DIR,N)
    # list1 is energies, width, fraction sample -> 6,9, N where N is different for each energies.
    # Frac vs width.
    label_list_en = ["50 GeV","100 GeV","150 GeV","200 GeV","250 GeV","300 GeV"]
    width_list = list2[0]
    print(width_list)
    print(len(energies_width_fraction))
    j=0
    for width_frac in energies_width_fraction: # for each energies:
        print(len(width_frac))
        print(width_frac.shape)
        width_frac = torch.tensor(width_frac)
        means = torch.mean(width_frac,dim=1)
        std = torch.std(width_frac,dim=1)
        print(means.shape,std.shape)
        errors = np.clip(std,0,1)
        plt.errorbar(width_list,
                means,
                yerr=errors,
                fmt='s-', label=label_list_en[j])
        j+=1    

        plt.ylim(0, 1.05) 

    plt.xlabel('S', fontsize=12)
    plt.ylabel('QDC in a Cluster / Total QDC', fontsize=12)
    plt.title('QDC Fraction in a Cluster vs. Radius of SiPMs'+IS_MC_DATA_TITLE)
    plt.grid()
    outdir = "cluster_scifi_dist"
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/{name}_cluster_distr.png",dpi=300)
    plt.close()



    #plot_1d_beam_energy_graphs(width_list, list1, label_list_en , "mean_total_"+name, show_ideal=False, xlabel="Radius of SiPMs",ylabel='QDC Fraction in a Cluster',title="QDC Fraction in a Cluster vs. Radius of SiPMs"+IS_MC_DATA_TITLE, outdir=outdirname)


def plot_for_hit_number(TEST_DATA_DIR,name, IS_MC_DATA_TITLE):
    x_min, x_max=0,1300
    y_min, y_max=0,6000
    bins_x, bins_y = 40,40

    N=50000
    list1,list2,list3,list4 = load_as_lists(TEST_DATA_DIR,N)
    total_list1 = sums = [tensor.sum(dim=(1)) for tensor in list1]
    total_list2 = sums = [tensor.sum(dim=(1)) for tensor in list2]
    total_list3 = sums = [tensor.sum(dim=(1)) for tensor in list3]
    total_list4 = sums = [tensor.sum(dim=(1)) for tensor in list4]

    total_energy = []

    label_list_en = ["50 GeV","100 GeV","150 GeV","200 GeV","250 GeV","300 GeV",]
    alpha_list=[0.7,0.7,0.7,0.7,0.7,0.7]

    xlabel="SciFi Hit Number"
    name= "hits_scifi"+name

    plot_multiple_hist(total_list1, None , None, None, xlabel,
                    "SciFi HDW Histogram"+IS_MC_DATA_TITLE, label_list_en, outdir=outdirname,alpha_list=[1,1,1,1,1,1], name="hist_total_"+name)

    plot_1d_beam_energy_graphs(beam_en_list, total_list1, label_list_en , "mean_total_"+name, show_ideal=False, xlabel="Beam Energy [GeV]",ylabel='SciFi Total Hit Number',title="SciFi Total Hit Number vs. Beam Energy"+IS_MC_DATA_TITLE, outdir=outdirname)

    first_index_layer_second_index_energy_array=[]
    for i in range(5):
        ith_layer = [wtrenergy_tensor[:,i] for wtrenergy_tensor in list1]
        first_index_layer_second_index_energy_array.append(ith_layer)

    label_list=["Layer 1","Layer 2","Layer 3","Layer 4","Layer 5"]
    plot_1d_compare_2_domains_beam_energy_graphs(beam_en_list, first_index_layer_second_index_energy_array, label_list , "all_layers_hits"+name, show_ideal=False, xlabel="Beam Energy [GeV]",ylabel='SciFi Total Hit Number',title=f"SciFi Hit Number vs. Beam Energy"+IS_MC_DATA_TITLE, outdir=outdirname)

    for i in range(5):
        plot_1d_beam_energy_graphs(beam_en_list, [tensor[:,i] for tensor in list1], label_list_en , str(i+1)+name, show_ideal=False, xlabel="Beam Energy [GeV]",ylabel='SciFi Total Hit Number',title=f"SciFi Hit Number at the Station {i+1} vs. Beam Energy"+IS_MC_DATA_TITLE, outdir=outdirname)

        plot_multiple_hist([tensor[:,i] for tensor in list1], None , None, None, xlabel,
                    f"SciFi Hit Number at the Station {i+1} Histogram"+IS_MC_DATA_TITLE, label_list_en, outdir=outdirname,alpha_list=[1,1,1,1,1,1], name="hist_layer_"+str(i+1)+name)

    return first_index_layer_second_index_energy_array

def plot_for_HDW(TEST_DATA_DIR,name, IS_MC_DATA_TITLE):
    x_min, x_max=0,1300
    y_min, y_max=0,6000
    bins_x, bins_y = 40,40

    N=50000
    list1,list2,list3,list4 = load_as_lists(TEST_DATA_DIR,N)
    ## list1, HDW for each layer
    # list2, max HDW in a layer
    # list3, max HDW hor in a layer
    #list4, max HDW ver in a layer
    total_HDW_in_all_layers = sums = [tensor.sum(dim=(1)) for tensor in list1]
    
    label_list_en = ["50 GeV","100 GeV","150 GeV","200 GeV","250 GeV","300 GeV"]
    alpha_list=[0.7,0.7,0.7,0.7,0.7,0.7]

    xlabel="HDW"
    name= "hdw_scifi"+name

    plot_multiple_hist(total_HDW_in_all_layers, None , None, None, xlabel,
                    "HDW Distribution (All SciFi Stations)"+IS_MC_DATA_TITLE, label_list_en, outdir=outdirname,alpha_list=[1,1,1,1,1,1], name="hist_all_scifi_"+name)

    plot_1d_beam_energy_graphs(beam_en_list, total_HDW_in_all_layers, label_list_en , "mean_all_scifi_"+name, show_ideal=False, xlabel="Beam Energy [GeV]",ylabel='HDW',title="HDW Distribution (All SciFi Stations) vs. Beam Energy"+IS_MC_DATA_TITLE, outdir=outdirname)

    plot_multiple_hist(list2, None , None, None, xlabel,
                    "Max HDW Distribution(Hor. and Ver.)"+IS_MC_DATA_TITLE, label_list_en, outdir=outdirname,alpha_list=[1,1,1,1,1,1], name="hist_max_hdw_"+name)
    plot_1d_beam_energy_graphs(beam_en_list, list2, label_list_en , "mean_max_hdw_"+name, show_ideal=False, xlabel="Beam Energy [GeV]",ylabel='HDW',title="Max HDW Distribution(Hor. and Ver.) vs. Beam Energy"+IS_MC_DATA_TITLE, outdir=outdirname)

    plot_multiple_hist(list3, None , None, None, xlabel,
                    "Max HDW Distribution (Hor.)"+IS_MC_DATA_TITLE, label_list_en, outdir=outdirname,alpha_list=[1,1,1,1,1,1], name="hist_max_hdw_hor_"+name)
    plot_1d_beam_energy_graphs(beam_en_list, list3, label_list_en , "mean_max_hdw_hor_"+name, show_ideal=False, xlabel="Beam Energy [GeV]",ylabel='HDW',title="Max HDW Distribution(Hor.) vs. Beam Energy"+IS_MC_DATA_TITLE, outdir=outdirname)

    plot_multiple_hist(list4, None , None, None, xlabel,
                    "Max HDW Distribution(Ver.)"+IS_MC_DATA_TITLE, label_list_en, outdir=outdirname,alpha_list=[1,1,1,1,1,1], name="hist_max_hdw_ver_"+name)
    plot_1d_beam_energy_graphs(beam_en_list, list4, label_list_en , "mean_max_hdw_ver_"+name, show_ideal=False, xlabel="Beam Energy [GeV]",ylabel='HDW',title="Max HDW Distribution(Ver.) vs. Beam Energy"+IS_MC_DATA_TITLE, outdir=outdirname)


    first_index_layer_second_index_energy_array=[]
    for i in range(5):
        ith_layer = [wtrenergy_tensor[:,i] for wtrenergy_tensor in list1]
        first_index_layer_second_index_energy_array.append(ith_layer)

    label_list=["Layer 1","Layer 2","Layer 3","Layer 4","Layer 5"]
    plot_1d_compare_2_domains_beam_energy_graphs(beam_en_list, first_index_layer_second_index_energy_array, label_list , "all_layers_HDW"+name, show_ideal=False, xlabel="Beam Energy [GeV]",ylabel='HDW',title=f"SciFi HDW vs. Beam Energy"+IS_MC_DATA_TITLE, outdir=outdirname)

    for i in range(5):
        plot_1d_beam_energy_graphs(beam_en_list, [tensor[:,i] for tensor in list1], label_list_en , str(i+1)+name, show_ideal=False, xlabel="Beam Energy [GeV]",ylabel='SciFi Total Hit Number',title=f"SciFi HDW at the Station {i+1} vs. Beam Energy"+IS_MC_DATA_TITLE, outdir=outdirname)

        plot_multiple_hist([tensor[:,i] for tensor in list1], None , None, None, xlabel,
                    f"SciFi HDW at the Station {i+1} Histogram"+IS_MC_DATA_TITLE, label_list_en, outdir=outdirname,alpha_list=[1,1,1,1,1,1], name="hist_layer_"+str(i+1)+name)

    return first_index_layer_second_index_energy_array

def plot_for_QDC(TEST_DATA_DIR,name, IS_MC_DATA_TITLE):
    x_min, x_max=0,13000
    y_min, y_max=0,60000
    bins_x, bins_y = 40,40

    N=50000
    list1,list2,list3,list4 = load_as_lists(TEST_DATA_DIR,N)
    total_list1 = sums = [tensor.sum(dim=(1)) for tensor in list1]
    total_list2 = sums = [tensor.sum(dim=(1)) for tensor in list2]
    total_list3 = sums = [tensor.sum(dim=(1)) for tensor in list3]
    total_list4 = sums = [tensor.sum(dim=(1)) for tensor in list4]

    total_energy=[]
    for i in range(len(list1)):
        total_energy.append(total_list1[i]+total_list2[i]+total_list3[i]+total_list4[i])

    us_ds_list=[]
    for i in range(len(list1)):
        us_ds_list.append(total_list2[i]+total_list3[i]+total_list4[i])


    
    alpha_list=[0.7,0.7,0.7,0.7,0.7,0.7]

    xlabel="QDC"
    name= "QDC_"+name

    plot_multiple_hist(total_list1, None , None, None, xlabel,
                    "QDC Distribution (All SciFi Stations)"+IS_MC_DATA_TITLE, label_list_en, outdir=outdirname,alpha_list=[1,1,1,1,1,1], name="hist_total_scifi_"+name)
    plot_multiple_hist(total_list2, None , None, None, xlabel,
                    "QDC Distribution (All US Stations)"+IS_MC_DATA_TITLE, label_list_en, outdir=outdirname,alpha_list=[1,1,1,1,1,1], name="hist_total_US_"+name)
    plot_multiple_hist(total_list3, None , None, None, xlabel,
                    "QDC Distribution (All DS Hor. Stations)"+IS_MC_DATA_TITLE, label_list_en, outdir=outdirname,alpha_list=[1,1,1,1,1,1], name="hist_total_DS_HOR_"+name)
    plot_multiple_hist(total_list4, None , None, None, xlabel,
                    "QDC Distribution (All DS Ver. Stations)"+IS_MC_DATA_TITLE, label_list_en, outdir=outdirname,alpha_list=[1,1,1,1,1,1], name="hist_total_DS_VER_"+name)
    plot_multiple_hist(us_ds_list, None , None, None, xlabel,
                    "QDC Distribution (US and DS)"+IS_MC_DATA_TITLE, label_list_en, outdir=outdirname,alpha_list=[1,1,1,1,1,1], name="hist_total_ALL_scifi_us_ds_"+name)
    plot_multiple_hist(total_energy, None , None, None, xlabel,
                    "QDC Distribution (SciFi, US, and DS)"+IS_MC_DATA_TITLE, label_list_en, outdir=outdirname,alpha_list=[1,1,1,1,1,1], name="hist_total_ALL_scifi_us_ds_"+name)


    
    beam_xlabel = "Beam Energy [GeV]"
    
    # 1. SciFi (total_list1)
    plot_1d_beam_energy_graphs(beam_en_list, total_list1, label_list_en, 
                               "mean_total_scifi_" + name, show_ideal=False, 
                               xlabel=beam_xlabel, ylabel='Mean QDC', 
                               title="Mean QDC vs. Beam Energy (All SciFi Stations)" + IS_MC_DATA_TITLE, 
                               outdir=outdirname)

    # 2. US Stations (total_list2)
    plot_1d_beam_energy_graphs(beam_en_list, total_list2, label_list_en, 
                               "mean_total_US_" + name, show_ideal=False, 
                               xlabel=beam_xlabel, ylabel='Mean QDC', 
                               title="Mean QDC vs. Beam Energy(All US Stations)" + IS_MC_DATA_TITLE, 
                               outdir=outdirname)

    # 3. DS Horizontal (total_list3)
    plot_1d_beam_energy_graphs(beam_en_list, total_list3, label_list_en, 
                               "mean_total_DS_HOR_" + name, show_ideal=False, 
                               xlabel=beam_xlabel, ylabel='Mean QDC', 
                               title="Mean QDC vs. Beam Energy(DS Hor.)" + IS_MC_DATA_TITLE, 
                               outdir=outdirname)

    # 4. DS Vertical (total_list4)
    plot_1d_beam_energy_graphs(beam_en_list, total_list4, label_list_en, 
                               "mean_total_DS_VER_" + name, show_ideal=False, 
                               xlabel=beam_xlabel, ylabel='Mean QDC', 
                               title="Mean QDC vs. Beam Energy(DS Ver.)" + IS_MC_DATA_TITLE, 
                               outdir=outdirname)

    # 5. US + DS Combined (us_ds_list)
    plot_1d_beam_energy_graphs(beam_en_list, us_ds_list, label_list_en, 
                               "mean_total_US_DS_combined_" + name, show_ideal=False, 
                               xlabel=beam_xlabel, ylabel='Mean QDC', 
                               title="Mean QDC vs. Beam Energy: US & DS Combined" + IS_MC_DATA_TITLE, 
                               outdir=outdirname)

    # 6. Global / All (total_energy)
    plot_1d_beam_energy_graphs(beam_en_list, total_energy, label_list_en, 
                               "mean_total_ALL_global_" + name, show_ideal=False, 
                               xlabel=beam_xlabel, ylabel='Mean QDC', 
                               title="Mean QDC vs. Beam Energy: Global (SciFi, US, and DS)" + IS_MC_DATA_TITLE, 
                               outdir=outdirname)


    first_index_layer_second_index_energy_array=[]
    for i in range(5):
        ith_layer = [wtrenergy_tensor[:,i] for wtrenergy_tensor in list1]
        first_index_layer_second_index_energy_array.append(ith_layer)

    label_list=["Layer 1","Layer 2","Layer 3","Layer 4","Layer 5"]
    plot_1d_compare_2_domains_beam_energy_graphs(beam_en_list, first_index_layer_second_index_energy_array, label_list , "all_layers_hits"+name, show_ideal=False, xlabel="Beam Energy [GeV]",ylabel='SciFi QDC',title=f"SciFi QDC vs. Beam Energy"+IS_MC_DATA_TITLE, outdir=outdirname)

    for i in range(5):
        plot_1d_beam_energy_graphs(beam_en_list, [tensor[:,i] for tensor in list1], label_list_en , str(i+1)+name, show_ideal=False, xlabel="Beam Energy [GeV]",ylabel='SciFi QDC',title=f"SciFi QDC at the Station {i+1} vs. Beam Energy"+IS_MC_DATA_TITLE, outdir=outdirname)

        plot_multiple_hist([tensor[:,i] for tensor in list1], None , None, None, xlabel,
                    f"SciFi QDC at the Station {i+1} Histogram"+IS_MC_DATA_TITLE, label_list_en, outdir=outdirname,alpha_list=[1,1,1,1,1,1], name="hist_layer_"+str(i+1)+name)

    total_scifi_qdc_energy_wtr_beam_energy = [arr1*0.053 for arr1 in total_list1]
    
    total_us_ds_qdc_energy_wtr_beam_energy = [arr2*0.0151 for arr2 in us_ds_list]


    total_qdc_energy_wtr_beam_energy=[]
    print(len(total_scifi_qdc_energy_wtr_beam_energy))
    print(len(total_us_ds_qdc_energy_wtr_beam_energy))
    for i in range(len(total_scifi_qdc_energy_wtr_beam_energy)):
        """print(total_scifi_qdc_energy_wtr_beam_energy[i])
        print(total_us_ds_qdc_energy_wtr_beam_energy[i])
        print(len(total_scifi_qdc_energy_wtr_beam_energy[i]))
        print(len(total_us_ds_qdc_energy_wtr_beam_energy[i]))"""
        total_qdc_energy_wtr_beam_energy.append(total_scifi_qdc_energy_wtr_beam_energy[i] + total_us_ds_qdc_energy_wtr_beam_energy[i])


    return first_index_layer_second_index_energy_array,  [total_qdc_energy_wtr_beam_energy, total_scifi_qdc_energy_wtr_beam_energy, total_us_ds_qdc_energy_wtr_beam_energy]

"""

SCIFI_QDC_2_GEV=1 ## IN PAPER,0.059
US_DS_QDC_2_GEV=1 # ın paper, 0.0145
scifi_cluster_radius=128 # this is for the plots, this does not apply cut.
INCLUDE_NEG_QDC=False
if INCLUDE_NEG_QDC:
     pos_and_neg_qdc_name = "_pos_and_neg_qdc"
     outdirname="new_hit_numbers_pos_and_neg_qdc"
else:
    pos_and_neg_qdc_name=""
    outdirname="new_hit_numbers"
if scifi_cluster_radius!=None:
    outdirname=outdirname+"_Radius_"+str(scifi_cluster_radius)


min_pmt_qdc_value=0
min_ds_pmt_qdc_value=0

time_window_max=0.5 ## 0.41
time_window_min=0.5
time_window_ds=3

min_pmt_qdc_value_us=0
time_window_min_us=3
time_window_max_us=3

hitx,hity=15,15
cluster_radius=64

use_us=True
use_ds=True

PLOT_MEAN_QDC_ENERGY=False
PLOT_MEAN_HIT_NUMBERS= True
N=10000

def plot_for_single_domain(TEST_DATA_DIR,name):
    x_min, x_max=0,1300
    y_min, y_max=0,6000
    bins_x, bins_y = 40,40

    N=50000
    list1,list2,list3,list4 = load_as_lists(TEST_DATA_DIR,N)
    total_list1 = sums = [tensor.sum(dim=(1)) for tensor in list1]
    total_list2 = sums = [tensor.sum(dim=(1)) for tensor in list2]
    total_list3 = sums = [tensor.sum(dim=(1)) for tensor in list3]
    total_list4 = sums = [tensor.sum(dim=(1)) for tensor in list4]

    total_energy = []
    for i in range(len(list1)):
        total_energy.append(total_list1[i]+total_list2[i]+total_list3[i]+total_list4[i])

    us_ds_list=[]
    for i in range(len(list1)):
        us_ds_list.append(total_list2[i]+total_list3[i]+total_list4[i])


    label_list_en = ["50 GeV","100 GeV","150 GeV","200 GeV","250 GeV","300 GeV",]
    alpha_list=[0.7,0.7,0.7,0.7,0.7,0.7]

    xlabel="SciFi Hit Number"
    name= "hits_scifi"+name

    plot_multiple_hist(total_list1, bins_x , x_min, x_max, xlabel,
                    "SciFi Total Hit Number Histogram", label_list_en, outdir=outdirname,alpha_list=alpha_list, name="hist_total_"+name)

    beam_en_list=[50,100,150,200,250,300]
    plot_1d_beam_energy_graphs(beam_en_list, total_list1, label_list_en , "mean_total_"+name, show_ideal=False, xlabel="Beam Energy [GeV]",ylabel='SciFi Total Hit Number',title="SciFi Total Hit Number vs. Beam Energy", outdir=outdirname)

    first_index_layer_second_index_energy_array=[]
    for i in range(5):
        ith_layer = [wtrenergy_tensor[:,i] for wtrenergy_tensor in list1]
        first_index_layer_second_index_energy_array.append(ith_layer)

    label_list=["Layer 1","Layer 2","Layer 3","Layer 4","Layer 5"]
    plot_1d_compare_2_domains_beam_energy_graphs(beam_en_list, first_index_layer_second_index_energy_array, label_list , "all_layers_hits"+name, show_ideal=False, xlabel="Beam Energy [GeV]",ylabel='SciFi Total Hit Number',title=f"SciFi Hit Number vs. Beam Energy ", outdir=outdirname)

    for i in range(5):
        plot_1d_beam_energy_graphs(beam_en_list, [tensor[:,i] for tensor in list1], label_list_en , str(i+1)+name, show_ideal=False, xlabel="Beam Energy [GeV]",ylabel='SciFi Total Hit Number',title=f"SciFi Hit Number at the Station {i+1} vs. Beam Energy ", outdir=outdirname)

        plot_multiple_hist([tensor[:,i] for tensor in list1], bins_x , x_min, x_max, xlabel,
                    f"SciFi Hit Number at the Station {i+1} Histogram", label_list_en, outdir=outdirname,alpha_list=alpha_list, name="hist_layer_"+str(i+1)+name)

    return first_index_layer_second_index_energy_array


name="small_scifi_ds_2024_electrons_50GeV_run_100933_0"
TEST_DATA_DIR = [
("/eos/user/b/beturk/snd/test_beam/2024/small/scifi_ds_2024_electrons_50GeV_run_100933_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2024/small/scifi_ds_2024_electrons_100GeV_run_100916_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2024/small/scifi_ds_2024_electrons_150GeV_run_100928_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2024/small/scifi_ds_2024_electrons_200GeV_run_100918_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2024/small/scifi_ds_2024_electrons_250GeV_run_100929_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2024/scifi_ds_2024_electrons_300GeV_run_100926_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
]

scifi_first_index_layer_second_index_energy_array = plot_for_single_domain(TEST_DATA_DIR,name)



name="TB_MC_nominal"
tb_mc_dir = "/eos/user/b/beturk/snd/test_beam/MC_24"
TEST_DATA_DIR = [
(f"{tb_mc_dir}/TB_MC_scifi_us_ds_2024_50GeV_11_nominal_entry_points_all_files.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
(f"{tb_mc_dir}/TB_MC_scifi_us_ds_2024_100GeV_11_nominal_entry_points_all_files.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
(f"{tb_mc_dir}/TB_MC_scifi_us_ds_2024_150GeV_11_nominal_entry_points_all_files.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
(f"{tb_mc_dir}/TB_MC_scifi_us_ds_2024_200GeV_11_nominal_entry_points_all_files.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
(f"{tb_mc_dir}/TB_MC_scifi_us_ds_2024_250GeV_11_nominal_entry_points_all_files.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
(f"{tb_mc_dir}/TB_MC_scifi_us_ds_2024_300GeV_11_nominal_entry_points_all_files.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us)
]
MC_scifi_first_index_layer_second_index_energy_array = plot_for_single_domain(TEST_DATA_DIR,name)
label_list=["MC Layer 1","MC Layer 2","MC Layer 3","MC Layer 4","MC Layer 5","Data Layer 1","Data Layer 2","Data Layer 3","Data Layer 4","Data Layer 5"]
skip_some_labels=[1,0,0,0,1,1,0,0,0,1]

beam_en_list=[50,100,150,200,250,300]
plot_1d_compare_3_domains_beam_energy_graphs(beam_en_list, [MC_scifi_first_index_layer_second_index_energy_array,scifi_first_index_layer_second_index_energy_array], label_list , "MC-DATA_COMP_", show_ideal=False, xlabel="Beam Energy [GeV]",ylabel='SciFi Total Hit Number',title=f"SciFi Hit Number vs. Beam Energy ", outdir=outdirname,skip_some=skip_some_labels)
"""