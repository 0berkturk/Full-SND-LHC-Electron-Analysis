import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import os
import numpy
import torch.nn.functional as F

cmap = plt.get_cmap('plasma')
cmap.set_under('white')
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
    frac_layer = torch.mean(sum_in_xy/total_qdc,0)
    frac_std = torch.std(sum_in_xy/total_qdc,0)
    return [frac_layer.item(), frac_std.item()]

def width_of_shower(data,label_str):
    z,h,v = find_shower_max(data)
    batch_size = data.size(0)
    batch_indices = torch.arange(batch_size)
    #data_at_z = data[batch_indices,:,z]
    width = [10, 50, 100,200,300, 400, 600,700,750] # vary this and see frac of qdc in there

    frac_list = np.array([call_cluster_image(data,w, v, h, batch_size) for w in width])
    print("frac list ",frac_list.shape,len(frac_list))
    #print(frac_list)
    means=frac_list[:,0]
    errors=frac_list[:,1]
    errors = np.clip(errors,0,1)
    print("means",means)
    print("errors",errors)
    """upper = np.clip(means + errors, None, 1.0)  # cap at 1
    lower = np.clip(means - errors, 0.0, None) # also avoid negative

    # recompute symmetric errorbars
    yerr = np.vstack([means - lower, upper - means])"""

    # matplotlib expects numpy
    plt.errorbar(width,
                means,
                yerr=errors,
                fmt='o-',
                capsize=4)
    plt.ylim(0, 1.05) 

    plt.xlabel('Cluster Half Widht', fontsize=12)
    plt.ylabel('QDC Fraction in Selected Cluster', fontsize=12)
    plt.title('SciFi QDC Fraction in Selected Cluster vs. Cluster Half Width', fontsize=12)
    plt.grid()
    outdir = "cluster_scifi_dist"
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/{label_str}_cluster_distr.png",dpi=300)
    plt.close()

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
    plt.clf()
    plt.hist(data, bins=bins)
    plt.savefig("hittime.png")

    return bin_centers

def apply_test_beam_cuts(scifi,past_consecutive_time_diff,fname):
    scifi = get_shower_cluster(scifi,cluster_radius)
    hit_number = (scifi!=0).sum((1,2,3))

    hit_cut = ((scifi[:,0]>0).sum((1,2)) > hitx) & ((scifi[:,1]>0).sum((1,2)) > hity) ## 50 gev el: 7,  250gev 25
    if "TB_MC_2024" in fname:
        print("it is mc, passibg consc. time diff cut",fname)
        time_cut = torch.ones_like(hit_cut)
    else:
        time_cut = past_consecutive_time_diff>150   ## plot histogram from scifi_timehits find max bin, reject events smaller than 3ns.
    return hit_cut & time_cut

def pmt_cut_inside_event(signals, times, min_qdc, t_min, t_max):
    # 1. Clean Inputs: Remove noise (low/negative QDC)
    positive_mask = signals > min_qdc
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
            plt.hist(qdc_energy,bins=bins,label=label_str[i],alpha=alpha_list[i])
            plt.xlim(xmin, xmax)
        else:
            plt.hist(qdc_energy , label=label_str[i],alpha=alpha_list[i])
    
        # Log scale
    plt.yscale('log')
    plt.xlabel(x_label)
    plt.ylabel('Counts')
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
        total_qdc_scifi=qdc_energy_list[j]
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

def calculate_qdc_energy(scifi,use_us,use_ds):
    scifi_energy = scifi.sum((1,2,3))*SCIFI_QDC_2_GEV
    if use_us is False:
        us_energy=torch.zeros_like(scifi_energy)
        print("NOT USING US")
    else:
        us_energy = use_us.sum((1,2,3))*US_DS_QDC_2_GEV

    if use_ds is False:
        ds_energy=torch.zeros_like(scifi_energy)
        print("NOT USING DS")
    else:
        ds_horizontal,ds_vertical=use_ds
        ds_energy =  ds_horizontal.sum((1,2,3))*US_DS_QDC_2_GEV+ds_vertical.sum((1,2))*US_DS_QDC_2_GEV
    return scifi_energy, us_energy, ds_energy

def calculate_qdc_energy_1layer(scifi,use_us,use_ds,layer_number):
    scifi_energy = scifi[:,:].sum((1,2,3))*SCIFI_QDC_2_GEV
    if use_us is False:
        us_energy=torch.zeros_like(scifi_energy)
        print("NOT USING US")
    else:
        us_energy = use_us[:,:,layer_number,:].sum((1,2))*US_DS_QDC_2_GEV

    if use_ds is False:
        ds_energy=torch.zeros_like(scifi_energy)
        print("NOT USING DS")
    else:
        ds_horizontal,ds_vertical=use_ds
        ds_energy =  ds_horizontal[:,:,layer_number,:].sum((1,2))*US_DS_QDC_2_GEV+ds_vertical[:,layer_number,:].sum((1))*US_DS_QDC_2_GEV
    return scifi_energy, us_energy, ds_energy

def load_as_lists(file_list,N):
    scifi_energy_list = []
    us_energy_list = []
    ds_energy_list = []

    for fname in file_list:
        fname, use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us = fname
        print(fname)
        first_data = torch.load(fname, map_location="cpu", mmap=True)
        data={}
        for key in first_data:
            data[key] = first_data[key][:N].clone()

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

        if PLOT_MEAN_QDC_ENERGY:
            scifi_energy, us_energy, ds_energy = calculate_qdc_energy(scifi, use_us, use_ds)
            print(scifi_energy, us_energy, ds_energy)
            scifi_energy_list.append(scifi_energy)
            us_energy_list.append(us_energy)
            ds_energy_list.append(ds_energy)

    return scifi_energy_list, us_energy_list, ds_energy_list


SCIFI_QDC_2_GEV=1 ## IN PAPER,0.059
US_DS_QDC_2_GEV=1 # ın paper, 0.0145

min_pmt_qdc_value=0
min_ds_pmt_qdc_value=0

time_window_max=0.5 ## 0.41
time_window_min=2
time_window_ds=3

min_pmt_qdc_value_us=0
time_window_min_us=3
time_window_max_us=3

hitx,hity=15,15
cluster_radius=64

use_us=True
use_ds=True

PLOT_MEAN_QDC_ENERGY=True

N=10000

"""TEST_DATA_DIR = [
("/eos/user/b/beturk/snd/test_beam/2023/scifi_us_ds_2023_pions_140GeV_3wall_run_100673_0.pt", use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2023/scifi_us_ds_2023_pions_140GeV_1wall_run_100661_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2024/scifi_ds_2024_pions_180GeV_run_100948_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2024/TB_MC_2024_electrons_nominal_entry_points_all_files.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2024/scifi_ds_2024_muons_150GeV_run_100892_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2024/scifi_ds_2024_electrons_300GeV_run_100907_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2024/small/scifi_ds_2024_electrons_250GeV_run_100929_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2024/small/scifi_ds_2024_electrons_200GeV_run_100918_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2024/small/scifi_ds_2024_electrons_150GeV_run_100928_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2024/small/scifi_ds_2024_electrons_100GeV_run_100916_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2024/small/scifi_ds_2024_electrons_50GeV_run_100933_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us)
]
label_list=["TB 23 Pions 3 Fe Wall", "TB 23 Pions 1 Fe Wall" ,"TB 24 Data Pions 2 W Wall", "TB 24 MC Electrons", "TB 24 Data Muons",  "TB 24 Data El. 300GeV", "TB 24 Data El. 250GeV", "TB 24 Data El. 200GeV", "TB 24 Data El. 150GeV", "TB 24 Data El. 100GeV", "TB 24 Data El. 50GeV"]
beam_en_list=[140,140,180, 300, 150, 300, 250,200,150,100,50]"""

"""TEST_DATA_DIR = [
("/eos/user/b/beturk/snd/test_beam/2023/scifi_us_ds_2023_pion_180GeV_3Fe_run_100672_0.pt", use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2023/scifi_us_ds_2023_pion_180GeV_2Fe_run_100668_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2023/scifi_us_ds_2023_pion_180GeV_1Fe_run_100660_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2023/scifi_us_ds_2023_pion_180GeV_run_100627_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),


("/eos/user/b/beturk/snd/test_beam/2023/scifi_us_ds_2023_pion_300GeV_3Fe_run_100641_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2023/scifi_us_ds_2023_pion_300GeV_2Fe_run_100650_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2023/scifi_us_ds_2023_pion_300GeV_1Fe_run_100653_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),

("/eos/user/b/beturk/snd/test_beam/2023/scifi_us_ds_2023_pion_140GeV_3Fe_run_100673_0.pt", use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2023/scifi_us_ds_2023_pion_140GeV_2Fe_run_100666_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2023/scifi_us_ds_2023_pion_140GeV_1Fe_run_100661_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2023/scifi_us_ds_2023_pion_140GeV_run_100626_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),



("/eos/user/b/beturk/snd/test_beam/2024/scifi_ds_2024_pions_180GeV_run_100948_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2024/scifi_us_ds_2024_pion_180GeV_2Fe_run_100979_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2024/scifi_us_ds_2024_pion_180GeV_1Fe_run_100962_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),




]
label_list=["TB 23 Pions 3 Fe Wall", "TB 23 Pions 2 Fe Wall","TB 23 Pions 1 Fe Wall","TB 23 Pions ?? Walls??","TB 23 Pions 3 Fe Wall", "TB 23 Pions 2 Fe Wall","TB 23 Pions 1 Fe Wall","TB 23 Pions 3 Fe Wall", "TB 23 Pions 2 Fe Wall","TB 23 Pions 1 Fe Wall","TB 23 Pions ?? Walls??" ,"TB 24 Data Pions 2 W Wall", "TB 24 Data Pions 2 Fe Wall", "TB 24 Data Pions 1 Fe Wall"]
beam_en_list=[180,180,180,180,300,300,300,140,140,140,140,180, 180, 180]"""
"""TEST_DATA_DIR = [
("/eos/user/b/beturk/snd/test_beam/2023/scifi_us_ds_2023_pion_180GeV_3Fe_run_100672_0.pt", use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2023/scifi_us_ds_2023_pion_180GeV_2Fe_run_100668_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2023/scifi_us_ds_2023_pion_180GeV_1Fe_run_100660_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2023/scifi_us_ds_2023_pion_180GeV_run_100627_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),


("/eos/user/b/beturk/snd/test_beam/2024/scifi_ds_2024_pions_180GeV_run_100948_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2024/scifi_us_ds_2024_pion_180GeV_2Fe_run_100979_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2024/scifi_us_ds_2024_pion_180GeV_1Fe_run_100962_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
]
"""

name="250gev_electron_scifi_us_2dhist"
if name=="scifi_us_ds_2023_pion_180GeV_3Fe_run_100672_0":
    TEST_DATA_DIR = [
    ("/eos/user/b/beturk/snd/test_beam/2023/scifi_us_ds_2023_pion_180GeV_3Fe_run_100672_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us)]
    x_min, x_max=0,10000
    y_min, y_max=0,11000
    bins_x, bins_y = 20,20
elif name=="scifi_us_ds_2024_pion_180GeV_2Fe_run_100979_0":
    TEST_DATA_DIR = [
    ("/eos/user/b/beturk/snd/test_beam/2024/scifi_us_ds_2024_pion_180GeV_2Fe_run_100979_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us)]
    x_min, x_max=0,4000
    y_min, y_max=0,4000
    bins_x, bins_y = 20,20

elif name=="scifi_ds_2024_pions_180GeV_run_100948_0":
    TEST_DATA_DIR = [
    ("/eos/user/b/beturk/snd/test_beam/2024/scifi_ds_2024_pions_180GeV_run_100948_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us)]
    x_min, x_max=0,4000
    y_min, y_max=0,8000
    bins_x, bins_y = 30,30

elif name=="300gev_electron_scifi_us_2dhist":
    TEST_DATA_DIR = [
    ("/eos/user/b/beturk/snd/test_beam/2024/scifi_ds_2024_electrons_300GeV_run_100926_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us)]
    x_min, x_max=0,4500
    y_min, y_max=0,11000
    bins_x, bins_y = 40,40

elif name=="250gev_electron_scifi_us_2dhist":
    TEST_DATA_DIR = [
    ("/eos/user/b/beturk/snd/test_beam/2024/small/scifi_ds_2024_electrons_250GeV_run_100929_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us)]
    x_min, x_max=0,4000
    y_min, y_max=0,10000
    bins_x, bins_y = 40,40

elif name=="200gev_electron_scifi_us_2dhist":
    TEST_DATA_DIR = [
    ("/eos/user/b/beturk/snd/test_beam/2024/scifi_ds_2024_electrons_200GeV_run_100924_5.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us)]
    x_min, x_max=0,3500
    y_min, y_max=0,9000
    bins_x, bins_y = 40,40

elif name=="150gev_electron_scifi_us_2dhist":
    TEST_DATA_DIR = [
    ("/eos/user/b/beturk/snd/test_beam/2024/small/scifi_ds_2024_electrons_150GeV_run_100928_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us)]
    x_min, x_max=0,3000
    y_min, y_max=0,8000
    bins_x, bins_y = 40,40

elif name=="100gev_electron_scifi_us_2dhist":
    TEST_DATA_DIR = [
    ("/eos/user/b/beturk/snd/test_beam/2024/small/scifi_ds_2024_electrons_100GeV_run_100916_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us)]
    x_min, x_max=0,2500
    y_min, y_max=0,7000
    bins_x, bins_y = 40,40

elif name=="50gev_electron_scifi_us_2dhist":
    TEST_DATA_DIR = [
    ("/eos/user/b/beturk/snd/test_beam/2024/small/scifi_ds_2024_electrons_50GeV_run_100933_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us)]
    x_min, x_max=0,2000
    y_min, y_max=0,6000
    bins_x, bins_y = 40,40


N=50000
list1,list2,list3 = load_as_lists(TEST_DATA_DIR,N)

total_energy = []
for i in range(len(list1)):
    total_energy.append(list1[i]+list2[i]+list3[i])

us_ds_list=[]
for i in range(len(list1)):
    us_ds_list.append(list2[i]+list3[i])


N=20

label_str = ["2023 TB","2024 TB"]
alpha_list=[0.7,0.7,0.7]

xlabel="DS QDC"
ylabel="SciFi QDC"
name= name+"_longer_intime_hits"
plot_2d_hist(list3, list1,
                 x_min, x_max, y_min, y_max, bins_x, bins_y,
                 name, xlabel,
                 ylabel, title="2D Histogram of SciFi vs. DS QDC ",
                 outdir="tb_electron_plots",
                 )



#plot_1d_graphs(true_en_list, hit_number_list, label_list,bins, out_name ,xlabel="True Energy [GeV]",ylabel='Hit Number',title="Hit Number vs. True Energy",outdir="qdc_comparision")
