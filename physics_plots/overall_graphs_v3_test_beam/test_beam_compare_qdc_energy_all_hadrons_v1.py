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

def pmt_cut_inside_event(scifi, scifi_hit_time, min_pmt_qdc_value, time_window_min, time_window_max):
    mean_time = find_highest_bin(scifi_hit_time, 0, 16, 100)
    mean_time = torch.as_tensor(
        mean_time, device=scifi.device, dtype=scifi_hit_time.dtype
    ).view(-1, 1, 1, 1)

    qdc_mask = scifi > min_pmt_qdc_value

    tmin = torch.clamp_min(mean_time - time_window_min, 0)
    tmax = mean_time + time_window_max

    time_mask = (scifi_hit_time > tmin) & (scifi_hit_time < tmax)

    final_mask = qdc_mask & time_mask

    scifi = scifi.masked_fill(~final_mask, 0)
    scifi_hit_time = scifi_hit_time.masked_fill(~final_mask, 0)

    return scifi, scifi_hit_time


def ds_pmt_cut_inside_event(
    ds_horizontal_qdc,
    ds_horizontal_time,
    ds_vertical_qdc,
    ds_vertical_time):
    # ---------- HORIZONTAL ----------
    mean_time_h = find_highest_bin(ds_horizontal_time, 0, 16, 100)
    mean_time_h = torch.as_tensor(
        mean_time_h,
        device=ds_horizontal_time.device,
        dtype=ds_horizontal_time.dtype,
    ).view(-1, 1, 1, 1)

    qdc_mask_h = ds_horizontal_qdc > min_ds_pmt_qdc_value

    tmin_h = torch.clamp_min(mean_time_h - time_window_ds, 0)
    tmax_h = mean_time_h + time_window_ds

    time_mask_h = (ds_horizontal_time > tmin_h) & (ds_horizontal_time < tmax_h)

    final_mask_h = qdc_mask_h & time_mask_h

    ds_horizontal_qdc = ds_horizontal_qdc.masked_fill(~final_mask_h, 0)
    ds_horizontal_time = ds_horizontal_time.masked_fill(~final_mask_h, 0)

    # ---------- VERTICAL ----------
    mean_time_v = find_highest_bin(ds_vertical_time, 0, 16, 100)
    mean_time_v = torch.as_tensor(
        mean_time_v,
        device=ds_vertical_time.device,
        dtype=ds_vertical_time.dtype,
    ).view(-1, 1, 1)

    qdc_mask_v = ds_vertical_qdc > min_ds_pmt_qdc_value

    tmin_v = torch.clamp_min(mean_time_v - time_window_ds, 0)
    tmax_v = mean_time_v + time_window_ds

    time_mask_v = (ds_vertical_time > tmin_v) & (ds_vertical_time < tmax_v)

    final_mask_v = qdc_mask_v & time_mask_v

    ds_vertical_qdc = ds_vertical_qdc.masked_fill(~final_mask_v, 0)
    ds_vertical_time = ds_vertical_time.masked_fill(~final_mask_v, 0)

    return ds_horizontal_qdc, ds_horizontal_time, ds_vertical_qdc, ds_vertical_time

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
        print(average_scifi_qdc)
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
    scifi_energy = scifi.sum((1,2,3))*SCIFI_QDC_2_GEV
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

def load_as_lists(file_list):
    scifi_energy_list = []
    us_energy_list = []
    ds_energy_list = []

    for fname in file_list:
        fname, use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us = fname
        print(fname)
        data = torch.load(fname, map_location="cpu")

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
        scifi , scifi_hit_time = pmt_cut_inside_event(scifi,scifi_hit_time,min_pmt_qdc_value, time_window_min, time_window_max)
        cut = apply_test_beam_cuts(scifi, past_consecutive_time_diff,fname)
        scifi=scifi[cut]
        scifi_hit_time=scifi_hit_time[cut]
        #next_consecutive_time_diff=next_consecutive_time_diff[cut]
        past_consecutive_time_diff=past_consecutive_time_diff[cut]
        if use_ds:
            ds_horizontal=ds_horizontal[cut]
            ds_horizontal_time=ds_horizontal_time[cut]
            ds_vertical=ds_vertical[cut]
            ds_vertical_time=ds_vertical_time[cut]
            ds_horizontal, ds_horizontal_time, ds_vertical, ds_vertical_time = ds_pmt_cut_inside_event(ds_horizontal, ds_horizontal_time, ds_vertical, ds_vertical_time)
            use_ds = (ds_horizontal,ds_vertical)

        if use_us:
            us_signals=us_signals[cut]
            us_signals_time=us_signals_time[cut]
            use_us, us_signals_time = pmt_cut_inside_event(us_signals,us_signals_time, min_pmt_qdc_value_us, time_window_min_us, time_window_max_us)
            ## use_us is equal to us_signals. it is not mistake. dont delete

        if PLOT_MEAN_QDC_ENERGY:
            scifi_energy, us_energy, ds_energy = calculate_qdc_energy(scifi, use_us, use_ds )
            print(scifi_energy, us_energy, ds_energy)
            scifi_energy_list.append(scifi_energy)
            us_energy_list.append(us_energy)
            ds_energy_list.append(ds_energy)

    return scifi_energy_list, us_energy_list, ds_energy_list


SCIFI_QDC_2_GEV=0.053 ## IN PAPER,0.059
US_DS_QDC_2_GEV=0.0151 # ın paper, 0.0145

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

PLOT_MEAN_QDC_ENERGY=True

N=10000


TEST_DATA_DIR = [
("/eos/user/b/beturk/snd/test_beam/2023/scifi_us_ds_2023_pions_100GeV_1Fe_run_100663_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2023/scifi_us_ds_2023_pions_100GeV_2Fe_run_100664_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2023/scifi_us_ds_2023_pions_100GeV_3Fe_run_100631_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2023/scifi_us_ds_2023_pions_140GeV_1Fe_run_100661_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2023/scifi_us_ds_2023_pions_140GeV_2Fe_run_100666_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2023/scifi_us_ds_2023_pions_140GeV_3Fe_run_100633_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2023/scifi_us_ds_2023_pions_180GeV_1Fe_run_100659_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2023/scifi_us_ds_2023_pions_180GeV_2Fe_run_100668_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2023/scifi_us_ds_2023_pions_180GeV_3Fe_run_100635_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2023/scifi_us_ds_2023_pions_240GeV_1Fe_run_100654_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2023/scifi_us_ds_2023_pions_240GeV_2Fe_run_100649_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2023/scifi_us_ds_2023_pions_240GeV_3Fe_run_100637_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2023/scifi_us_ds_2023_pions_300GeV_1Fe_run_100653_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2023/scifi_us_ds_2023_pions_300GeV_2Fe_run_100650_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/user/b/beturk/snd/test_beam/2023/scifi_us_ds_2023_pions_300GeV_3Fe_run_100638_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us)
]
label_list=["Pions 1Fe 100GeV","Pions 2Fe 100GeV","Pions 3Fe 100GeV","Pions 1Fe 140GeV","Pions 2Fe 140GeV","Pions 3Fe 140GeV","Pions 1Fe 180GeV","Pions 2Fe 180GeV","Pions 3Fe 180GeV","Pions 1Fe 240GeV","Pions 2Fe 240GeV","Pions 3Fe 240GeV","Pions 1Fe 300GeV","Pions 2Fe 300GeV","Pions 3Fe 300GeV"]
beam_en_list=[100,100,100,140,140,140,180,180,180,240,240,240,300,300,300]
list1,list2,list3 = load_as_lists(TEST_DATA_DIR)

total_energy = []
for i in range(len(list1)):
    total_energy.append(list1[i]+list2[i]+list3[i])

us_ds_list=[]
for i in range(len(list1)):
    us_ds_list.append(list2[i]+list3[i])


if PLOT_MEAN_QDC_ENERGY:
    out_name="only_hadrons"
    plot_1d_beam_energy_graphs(beam_en_list, list1,label_list ,out_name,show_ideal=True,xlabel="Beam Energy [GeV]",ylabel='SciFi QDC Energy[GeV]',title="SciFi QDC Energy vs. Beam Energy",outdir="beam_comp")
    plot_1d_beam_energy_graphs(beam_en_list, list2,label_list ,out_name,show_ideal=True,xlabel="Beam Energy [GeV]",ylabel='US QDC Energy[GeV]',title="US QDC Energy vs. Beam Energy",outdir="beam_comp")
    plot_1d_beam_energy_graphs(beam_en_list, list3,label_list ,out_name,show_ideal=True,xlabel="Beam Energy [GeV]",ylabel='DS QDC Energy[GeV]',title="DS QDC Energy vs. Beam Energy",outdir="beam_comp")
    plot_1d_beam_energy_graphs(beam_en_list, total_energy, label_list ,out_name,show_ideal=True,xlabel="Beam Energy [GeV]",ylabel='Total QDC Energy[GeV]',title="Total QDC Energy vs. Beam Energy",outdir="beam_comp")


#plot_1d_graphs(true_en_list, hit_number_list, label_list,bins, out_name ,xlabel="True Energy [GeV]",ylabel='Hit Number',title="Hit Number vs. True Energy",outdir="qdc_comparision")
