import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import os
import numpy
cmap = plt.get_cmap('plasma')
cmap.set_under('white')
def plot_2d_im(scifi_hits, j,out_name,energy):
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

        ax.set_title(f"SciFi Projection {['X', 'Y'][i]} ({energy[j]})")
        ax.set_xlabel("Fiber index")
        if i == 0:
            ax.set_ylabel("Z (plane index)")

    # Move colorbar to the right side of both plots
    cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.04)
    cbar.set_label("Signal")

    plt.tight_layout(rect=[0, 0, 0.95, 1])  # leave space for colorbar
    plt.savefig(out_name+ str(j) + ".png", dpi=200)
    plt.clf()


import numpy as np
import matplotlib.pyplot as plt
import os

def plot_1d_hist(qdc_energy, N, xmin, xmax, x_label, title, label_str,outdir):
    if N!=None:
        bins = np.linspace(xmin, xmax, N)
        counts, bin_edges, _ = plt.hist(qdc_energy, bins=bins)
        plt.xlim(xmin, xmax)

    else:
        counts, bin_edges, _ = plt.hist(qdc_energy)

    # Find maximum bin
    max_bin_index = np.argmax(counts)
    bin_left = bin_edges[max_bin_index]
    bin_right = bin_edges[max_bin_index + 1]
    bin_center = 0.5 * (bin_left + bin_right)

    # Peak line (orange)
    plt.axvline(
        bin_center,
        linestyle='--',
        linewidth=2,
        color='orange',
        label=f'Peak = {bin_center:.2f}'
    )

    # Axis limits exactly match binning

    # Log scale
    plt.yscale('log')

    # Labels and style
    plt.xlabel(x_label)
    plt.ylabel('Counts')
    plt.title(title)
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)

    # Save
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/{label_str}_{title}.png", dpi=300)
    plt.close()
    return counts

def replot_multiple_hist(qdc_energy_list,N,xmin,xmax,x_label, title, label_str, outdir,alpha_list,name="EMPTY"):
    for i in range(len(qdc_energy_list)):
        qdc_energy=qdc_energy_list[i]
        if N!=None:
            bins = np.linspace(xmin, xmax, N)
            plt.hist(bins[:-1],bins=bins,weights=qdc_energy, histtype='stepfilled',label=label_str[i],alpha=alpha_list[i])
            plt.xlim(xmin, xmax)
        else:
            plt.hist(qdc_energy, weights=qdc_energy,histtype='stepfilled',  label=label_str[i],alpha=alpha_list[i])
    
    plt.axvline(
        BEAM_ENERGY,
        linestyle='--',
        linewidth=2,
        color='orange',
        label=f'Peak = {BEAM_ENERGY:.2f}'
    )

        # Log scale
    plt.yscale('log')
    plt.xlabel(x_label)
    plt.ylabel('Counts')
    plt.title(title)
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    # Save
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/{name}_{title}.png", dpi=300)
    plt.close()

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
    plt.savefig(f"{outdir}/{name}_{title}.png", dpi=300)
    plt.close()
        

def plot_qdc_energy(true_en_list, scifi_list,us_list,ds_list,label_list,bins ,out_name):
    plt.figure()
    # Sum over channels, width, and height dimensions (dimensions 1, 2, 3)
    for j in range(len(scifi_list)):
        scifi=scifi_list[j]
        us=us_list[j]
        ds=ds_list[j]
        true_en=true_en_list[j]

        total_qdc_scifi = scifi.sum(dim=(1, 2, 3))*SCIFI_QDC_2_GEV
        total_qdc += total_qdc_scifi 

        if us_list!=None:
            total_qdc_us = us.sum(dim=(1, 2, 3))* something *US_DS_QDC_2_GEV
            total_qdc += total_qdc_us 
            average_us_qdc = []       
            std_us_qdc = []
        
    
        if ds_list!=None:
            total_qdc_ds = ds.sum(dim=(1, 2, 3)) * US_DS_QDC_2_GEV
            total_qdc += total_qdc_ds
            average_ds_qdc = []
            std_ds_qdc = []


        average_qdc = []
        average_scifi_qdc = []


        std_qdc = []
        std_scifi_qdc = []

        for i in range(len(bins)-1):
            en_min = bins[i]
            en_max = bins[i+1]
            index = (true_en>=en_min) & (true_en<en_max)
            average_qdc.append(total_qdc[index].mean().item())
            std_qdc.append(total_qdc[index].std().item())

            average_scifi_qdc.append(total_qdc_scifi[index].mean().item())
            std_scifi_qdc.append(total_qdc_scifi[index].std().item())

            if us_list!=None:
                average_us_qdc.append(total_qdc_us[index].mean().item())
                std_us_qdc.append(total_qdc_us[index].std().item())

            if ds_list!=None:
                std_ds_qdc.append(total_qdc_ds[index].std().item())
                average_ds_qdc.append(total_qdc_ds[index].mean().item())



        plt.errorbar((bins[:-1]+bins[1:])/2, average_qdc, yerr=std_qdc, fmt='o-', label=f'Total QDC ({label_list[j]})',alpha=0.7)
        plt.errorbar((bins[:-1]+bins[1:])/2, average_scifi_qdc, yerr=std_scifi_qdc, fmt='s-', label=f'SciFi QDC({label_list[j]})',alpha=0.7)
        if us_list!=None:
            plt.errorbar((bins[:-1]+bins[1:])/2, average_us_qdc, yerr=std_us_qdc, fmt='^-', label=f'US QDC({label_list[j]})',alpha=0.7)
        if ds_list!=None:
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
    _, h, v = find_shower_max(scifi)

    ver_pos_big,ver_pos_small = call_cluster_size(v,width)
    hor_pos_big, hor_pos_small = call_cluster_size(h,width)
    
    cluster_data=torch.zeros(batch_size,2,5,2*width)
    for i in range(batch_size):
       # print(data[i,1,:, hor_pos_small[i]:hor_pos_big[i] ].shape)
        #print(hor_pos_small[i],hor_pos_big[i])
        cluster_data[i,1,:,:] = data[i,1,:, ver_pos_small[i]:ver_pos_big[i] ]
        cluster_data[i,0,:,:] = data[i,0,:, hor_pos_small[i]:hor_pos_big[i] ]

    return cluster_data

#def plot_time_difference_btw_consecutive_events(next_time_diff,past_time_diff)
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

def apply_test_beam_cuts(scifi,past_consecutive_time_diff):
    scifi = get_shower_cluster(scifi,cluster_radius)
    hit_number = (scifi!=0).sum((1,2,3))
    plot_1d_hist(hit_number,N=40,xmin=0,xmax=100,x_label=f"SciFi Hit Number inside {cluster_radius} SiPM Radius",title=f"Histogram of SciFi Hit Number inside {cluster_radius} SiPM Radius(Loose Cut)" ,label_str=label_str,outdir="1d_hit_number_hist")

    hit_cut = ((scifi[:,0]>0).sum((1,2)) > hitx) & ((scifi[:,1]>0).sum((1,2)) > hity) ## 50 gev el: 7,  250gev 25
    if "TB_MC_2024" in TEST_DATA_DIR[0]:
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


def load_tb_data(TEST_DATA_DIR):
    data=torch.load(TEST_DATA_DIR[0])

    scifi=data["scifi_signals"]
    scifi_hit_time = data["scifi_hit_time"]

    #next_consecutive_time_diff = data["next_consecutive_time_diff"] ## no need, future time diff does not affect current event. ın one event, there could be 2 particles and 2nd particle can leak to next event. but we dont consider it.
    past_consecutive_time_diff = data["past_consecutive_time_diff"]*6.25

    if IS_THERE_DS_IN_DATA:
        ds_horizontal=data["ds_horizontal"]
        ds_horizontal_time=data["ds_horizontal_time"]
        ds_vertical=data["ds_vertical"]
        ds_vertical_time=data["ds_vertical_time"]
    else:
        ds_horizontal=None
        ds_horizontal_time=None
        ds_vertical=None
        ds_vertical_time=None
    
    if IS_THERE_US_IN_DATA:
        us_signals=data["us_signals"]
        us_signals_time=data["us_signals_time"]
    else:
        us_signals=None
        us_signals_time=None

    return scifi, scifi_hit_time, past_consecutive_time_diff, ds_horizontal, ds_horizontal_time, ds_vertical, ds_vertical_time, us_signals, us_signals_time

def plot_and_save(scifi,us=None,ds_horizontal=None,ds_vertical=None,cut_in_title_name="Loose Cut",save=False):
    print("scifi size ",scifi.shape)
    if (not IS_THERE_DS_IN_DATA) or (IS_THERE_DS_IN_DATA and REJECT_DS):
        print("ploting only scifi")
        scifi_qdc_energy = scifi.sum((1,2,3))*SCIFI_QDC_2_GEV
        hist1=plot_1d_hist(scifi_qdc_energy,N=40,xmin=EN_MIN,xmax=EN_MAX,x_label="QDC Energy[GeV]",title=f"Histogram of SciFi QDC Energy({cut_in_title_name})" ,label_str=label_str,outdir="1d_qdc_energy_hist")
        
        hit_number = (scifi!=0).sum((2,3))
        hit_number_ver=hit_number[:,1]
        hit_number_hor=hit_number[:,0]
        plot_multiple_hist([hit_number_hor,hit_number_ver],40,0,HIT_NUMBER,x_label="Hit Number in SciFi",title=f"Histogram of Hit Number in SciFi({cut_in_title_name})" ,label_str=["Horizontal","Vertical"],outdir="1d_hit_number_hist",alpha_list=[0.7,0.7],name=label_str)

    elif IS_THERE_US_IN_DATA and not REJECT_US and IS_THERE_DS_IN_DATA and not REJECT_DS:
        print("plotting all, scifi, us and ds")

        scifi_qdc_energy = scifi.sum((1,2,3))*SCIFI_QDC_2_GEV
        ds_qdc_energy = ds_horizontal.sum((1,2,3))*US_DS_QDC_2_GEV+ds_vertical.sum((1,2))*US_DS_QDC_2_GEV
        us_qdc_energy = us_signals.sum((1,2,3))*US_DS_QDC_2_GEV

        total_qdc_energy = scifi_qdc_energy + ds_qdc_energy + us_qdc_energy
        scifi_us_qdc_energy = scifi_qdc_energy + us_qdc_energy
        

        plot_1d_hist(scifi_qdc_energy,N=40,xmin=EN_MIN,xmax=EN_MAX,x_label="QDC Energy[GeV]",title=f"Histogram of SciFi QDC Energy({cut_in_title_name})" ,label_str=label_str,outdir="1d_qdc_energy_hist")
        plot_1d_hist(ds_qdc_energy ,N=40,xmin=EN_MIN,xmax=EN_MAX,x_label="QDC Energy[GeV]", title =f"Histogram of DS QDC Energy({cut_in_title_name})" ,label_str=label_str,outdir="1d_qdc_energy_hist")
        plot_1d_hist(us_qdc_energy ,N=40,xmin=EN_MIN,xmax=EN_MAX,x_label="QDC Energy[GeV]", title =f"Histogram of US QDC Energy({cut_in_title_name})" ,label_str=label_str,outdir="1d_qdc_energy_hist")
        plot_1d_hist(scifi_us_qdc_energy ,N=40,xmin=EN_MIN,xmax=EN_MAX,x_label="QDC Energy[GeV]", title =f"Histogram of SciFi+US QDC Energy({cut_in_title_name})" ,label_str=label_str,outdir="1d_qdc_energy_hist")

        hist1=plot_1d_hist(total_qdc_energy ,N=40,xmin=EN_MIN,xmax=EN_MAX,x_label="QDC Energy[GeV]", title =f"Histogram of SciFi+US+DS QDC Energy({cut_in_title_name})" ,label_str=label_str,outdir="1d_qdc_energy_hist")

        hit_number = (scifi!=0).sum((2,3))
        hit_number_ver=hit_number[:,1]
        hit_number_hor=hit_number[:,0]
        plot_multiple_hist([hit_number_hor,hit_number_ver],40,0,HIT_NUMBER,x_label="Hit Number in SciFi",title=f"Histogram of Hit Number in SciFi({cut_in_title_name})" ,label_str=["Horizontal","Vertical"],outdir="1d_hit_number_hist",alpha_list=[0.7,0.7],name=label_str)



    elif IS_THERE_DS_IN_DATA and not REJECT_DS:
        print("plotting only scifi and ds")

        scifi_qdc_energy = scifi.sum((1,2,3))*SCIFI_QDC_2_GEV
        ds_qdc_energy = ds_horizontal.sum((1,2,3))*US_DS_QDC_2_GEV+ds_vertical.sum((1,2))*US_DS_QDC_2_GEV
        total_qdc_energy = scifi_qdc_energy + ds_qdc_energy
        plot_1d_hist(scifi_qdc_energy,N=40,xmin=EN_MIN,xmax=EN_MAX,x_label="QDC Energy[GeV]",title=f"Histogram of SciFi QDC Energy({cut_in_title_name})" ,label_str=label_str,outdir="1d_qdc_energy_hist")
        plot_1d_hist(ds_qdc_energy ,N=40,xmin=EN_MIN,xmax=EN_MAX,x_label="QDC Energy[GeV]", title =f"Histogram of DS QDC Energy({cut_in_title_name})" ,label_str=label_str,outdir="1d_qdc_energy_hist")
        hist1=plot_1d_hist( total_qdc_energy ,N=40,xmin=EN_MIN,xmax=EN_MAX,x_label="QDC Energy[GeV]", title =f"Histogram of SciFi+DS QDC Energy({cut_in_title_name})" ,label_str=label_str,outdir="1d_qdc_energy_hist")

        hit_number = (scifi!=0).sum((2,3))
        hit_number_ver=hit_number[:,1]
        hit_number_hor=hit_number[:,0]
        plot_multiple_hist([hit_number_hor,hit_number_ver],40,0,HIT_NUMBER,x_label="Hit Number in SciFi",title=f"Histogram of Hit Number in SciFi({cut_in_title_name})" ,label_str=["Horizontal","Vertical"],outdir="1d_hit_number_hist",alpha_list=[0.7,0.7],name=label_str)

    if save:
        save_dict = {
        "scifi_signals": scifi,
        "en3d": en3d
        }
        torch.save(save_dict, f"{out_name}.pt")
    return hist1

scifi_ds_2024_electrons_50GeV_run_100933=False
scifi_ds_2024_electrons_100GeV_run_100916 = False
scifi_ds_2024_electrons_150GeV_run_100928 = False
scifi_ds_2024_electrons_200GeV_run_100918=False
scifi_ds_2024_electrons_250GeV_run_100929=False
scifi_ds_2024_electrons_300GeV_run_100926 = False

scifi_ds_2024_pions_180GeV_run_100948=False
scifi_ds_2024_muons_150GeV_run_100892=False

TB_MC_2024_electrons_nominal_entry_points_all_files = False

scifi_us_ds_2023_pions_140GeV_3wall_run_100673 = False
scifi_us_ds_2023_pions_140GeV_1wall_run_100661 = True



IS_THERE_DS_IN_DATA = True
REJECT_DS=False ## reject events if there are hits in ds.
IS_THERE_US_IN_DATA=True ## in 2024 tb, there is no us.
REJECT_US=False
i=0

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

if scifi_ds_2024_electrons_50GeV_run_100933:
    TEST_DATA_DIR = [f"/eos/user/b/beturk/snd/test_beam/2024/small/scifi_ds_2024_electrons_50GeV_run_100933_{i}.pt"]
    #["/afs/cern.ch/work/b/beturk/private/snd/test_Beam/electrons_all_50gev_100933_0.pt"]
    label_str =f"scifi_ds_2024_electrons_50GeV_run_100933_{i}"
    EN_MIN,EN_MAX=0,420
    hitx,hity=7,6

    N=20
    HIT_NUMBER=100 ## İN THE HİST
    cluster_radius=64
    BEAM_ENERGY=50
    out_name="/eos/user/b/beturk/snd/test_beam/2024/"+"selected_"+f"scifi_ds_2024_electrons_50GeV_run_100933_{i}"

elif scifi_ds_2024_electrons_300GeV_run_100926:
    TEST_DATA_DIR = [f"/eos/user/b/beturk/snd/test_beam/2024/scifi_ds_2024_electrons_300GeV_run_100907_{i}.pt"]
    #["/afs/cern.ch/work/b/beturk/private/snd/test_Beam/electrons_all_50gev_100933_0.pt"]
    label_str =f"scifi_ds_2024_electrons_300GeV_run_100926_{i}"
    EN_MIN,EN_MAX=0,800
    hitx,hity=32,32
    N=20
    HIT_NUMBER=800 ## İN THE HİST
    cluster_radius=64
    BEAM_ENERGY=300
    out_name="/eos/user/b/beturk/snd/test_beam/2024/"+"selected_"+f"scifi_ds_2024_electrons_300GeV_run_100926_{i}"

elif scifi_ds_2024_electrons_100GeV_run_100916:
    TEST_DATA_DIR = [f"/eos/user/b/beturk/snd/test_beam/2024/small/scifi_ds_2024_electrons_100GeV_run_100916_{i}.pt"]
    #["/afs/cern.ch/work/b/beturk/private/snd/test_Beam/electrons_all_50gev_100933_0.pt"]
    label_str =f"scifi_ds_2024_electrons_100GeV_run_100916_{i}"
    EN_MIN,EN_MAX=0,600
    hitx,hity=10,10
    N=20
    cluster_radius=64
    HIT_NUMBER=400 ## İN THE HİST
    BEAM_ENERGY=100
    out_name="/eos/user/b/beturk/snd/test_beam/2024/"+"selected_"+f"scifi_ds_2024_electrons_100GeV_run_100916_{i}"


elif scifi_ds_2024_electrons_150GeV_run_100928:
    TEST_DATA_DIR = [f"/eos/user/b/beturk/snd/test_beam/2024/small/scifi_ds_2024_electrons_150GeV_run_100928_{i}.pt"]
    #["/afs/cern.ch/work/b/beturk/private/snd/test_Beam/electrons_all_50gev_100933_0.pt"]
    label_str =f"scifi_ds_2024_electrons_150GeV_run_100928_{i}"
    EN_MIN,EN_MAX=0,800
    hitx,hity=15,15
    N=20
    HIT_NUMBER=500 ## İN THE HİST

    cluster_radius=32
    BEAM_ENERGY=150
    out_name="/eos/user/b/beturk/snd/test_beam/2024/"+"selected_"+f"scifi_ds_2024_electrons_150GeV_run_100928_{i}"

elif scifi_ds_2024_electrons_200GeV_run_100918:
    TEST_DATA_DIR = [f"/eos/user/b/beturk/snd/test_beam/2024/small/scifi_ds_2024_electrons_200GeV_run_100918_{i}.pt"]
    #["/afs/cern.ch/work/b/beturk/private/snd/test_Beam/electrons_all_50gev_100933_0.pt"]
    label_str =f"scifi_ds_2024_electrons_200GeV_run_100918_{i}"
    EN_MIN,EN_MAX=0,800
    hitx,hity=15,15
    N=20
    HIT_NUMBER=500 ## İN THE HİST

    cluster_radius=32
    BEAM_ENERGY=200
    out_name="/eos/user/b/beturk/snd/test_beam/2024/"+"selected_"+f"scifi_ds_2024_electrons_200GeV_run_100918_{i}"


elif scifi_ds_2024_electrons_250GeV_run_100929:
    TEST_DATA_DIR = [f"/eos/user/b/beturk/snd/test_beam/2024/small/scifi_ds_2024_electrons_250GeV_run_100929_{i}.pt"]
    #["/afs/cern.ch/work/b/beturk/private/snd/test_Beam/electrons_all_50gev_100933_0.pt"]
    label_str =f"scifi_ds_2024_electrons_250GeV_run_100929_{i}"
    EN_MIN,EN_MAX=0,800
    hitx,hity=15,15
    N=20
    HIT_NUMBER=500 ## İN THE HİST

    cluster_radius=32
    BEAM_ENERGY=250
    out_name="/eos/user/b/beturk/snd/test_beam/2024/"+"selected_"+f"scifi_ds_2024_electrons_250GeV_run_100929_{i}"


elif scifi_ds_2024_pions_180GeV_run_100948:
    TEST_DATA_DIR = [f"/eos/user/b/beturk/snd/test_beam/2024/scifi_ds_2024_pions_180GeV_run_100948_{i}.pt"]
    label_str =f"scifi_ds_2024_pions_180GeV_run_100948_{i}"
    EN_MIN,EN_MAX=0,800
    hitx,hity=5,5
    N=20
    HIT_NUMBER=500 ## İN THE HİST

    cluster_radius=32
    BEAM_ENERGY=180
    out_name="/eos/user/b/beturk/snd/test_beam/2024/"+"selected_"+f"scifi_ds_2024_pions_180GeV_run_100948_{i}"

elif scifi_ds_2024_muons_150GeV_run_100892:
    TEST_DATA_DIR = [f"/eos/user/b/beturk/snd/test_beam/2024/scifi_ds_2024_muons_150GeV_run_100892_{i}.pt"]
    label_str =f"scifi_ds_2024_muons_150GeV_run_100892_{i}"
    EN_MIN,EN_MAX=0,800
    hitx,hity=5,5
    N=20
    HIT_NUMBER=500 ## İN THE HİST

    cluster_radius=32
    BEAM_ENERGY=180
    out_name="/eos/user/b/beturk/snd/test_beam/2024/"+"selected_"+f"scifi_ds_2024_muons_150GeV_run_100892_{i}"

elif TB_MC_2024_electrons_nominal_entry_points_all_files:
    TEST_DATA_DIR = [f"/eos/user/b/beturk/snd/test_beam/2024/TB_MC_2024_electrons_nominal_entry_points_all_files.pt"]
    label_str =f"TB_MC_2024_electrons_nominal_entry_points_all_files"
    EN_MIN,EN_MAX=0,1200
    hitx,hity=32,32
    N=20
    HIT_NUMBER=800 ## İN THE HİST
    cluster_radius=64
    BEAM_ENERGY=300
    out_name="/eos/user/b/beturk/snd/test_beam/2024/"+"selected_"+f"TB_MC_2024_electrons_nominal_entry_points_all_files"


elif scifi_us_ds_2023_pions_140GeV_3wall_run_100673:
    TEST_DATA_DIR = [f"/eos/user/b/beturk/snd/test_beam/2023/scifi_us_ds_2023_pions_140GeV_3wall_run_100673_{i}.pt"]
    label_str =f"scifi_us_ds_2023_pions_140GeV_3wall_run_100673_{i}"
    EN_MIN,EN_MAX=0,400
    hitx,hity=32,32
    BEAM_ENERGY=140
    N=20
    HIT_NUMBER=800 ## İN THE HİST
    cluster_radius=64
    out_name="/eos/user/b/beturk/snd/test_beam/2024/"+"selected_"+f"scifi_us_ds_2023_pions_140GeV_3wall_run_100673_{i}"


elif scifi_us_ds_2023_pions_140GeV_1wall_run_100661:
    TEST_DATA_DIR = [f"/eos/user/b/beturk/snd/test_beam/2023/scifi_us_ds_2023_pions_140GeV_1wall_run_100661_{i}.pt"]
    label_str =f"scifi_us_ds_2023_pions_140GeV_1wall_run_100661_{i}"
    EN_MIN,EN_MAX=0,300
    hitx,hity=16,16
    BEAM_ENERGY=140
    N=20
    HIT_NUMBER=800 ## İN THE HİST
    cluster_radius=64
    out_name="/eos/user/b/beturk/snd/test_beam/2024/"+"selected_"+f"scifi_us_ds_2023_pions_140GeV_1wall_run_100661_{i}"


##### Initial plots
print(TEST_DATA_DIR)
scifi, scifi_hit_time, past_consecutive_time_diff, ds_horizontal, ds_horizontal_time, ds_vertical, ds_vertical_time, us_signals, us_signals_time = load_tb_data(TEST_DATA_DIR)

en3d=BEAM_ENERGY*torch.ones_like(past_consecutive_time_diff)
plot_1d_hist(past_consecutive_time_diff,N=50,xmin=99,xmax=1000, x_label="Consecutive Time Diff.[Time Clock](Loose cut)", title="Histogram of Consecutive Time Difference (Loose cut)" , label_str=label_str, outdir="1d_time_hist")
neg_qdc_loose_cut=plot_and_save(scifi,us=None,ds_horizontal=ds_horizontal,ds_vertical=ds_vertical,cut_in_title_name="Loose Cut with Neg. QDC")

ith_scifi_hit_time = scifi_hit_time[15].reshape(-1)
ith_scifi_hit_time = ith_scifi_hit_time[ith_scifi_hit_time!=0]
plot_1d_hist(ith_scifi_hit_time,N=50,xmin=0,xmax=5,x_label="SciFi Hit Time in a Event[Time Clock]", title="Histogram of SciFi Hit Time in a Event(before scifi hit cut)" ,label_str=label_str,outdir="1d_time_hist")

pos_qdc_loose_cut=plot_and_save(torch.clamp_min(scifi,0),us=None,ds_horizontal=ds_horizontal,ds_vertical=ds_vertical,cut_in_title_name="Loose Cut with Pos. QDC")

##### scifi in time hit cuts
scifi , scifi_hit_time = pmt_cut_inside_event(scifi,scifi_hit_time,min_pmt_qdc_value, time_window_min, time_window_max)
ith_scifi_hit_time = scifi_hit_time[15].reshape(-1)
ith_scifi_hit_time = ith_scifi_hit_time[ith_scifi_hit_time!=0]
plot_1d_hist(ith_scifi_hit_time,N=50,xmin=0,xmax=5,x_label="SciFi Hit Time in a Event[Time Clock]", title="Histogram of SciFi Hit Time in a Event" ,label_str=label_str,outdir="1d_time_hist")
######


print("Initial size ",scifi.shape)
if REJECT_DS and IS_THERE_DS_IN_DATA:
    print("reject ds events")
    cut = reject_events_that_leaked_to_us_ds(us_signals,ds_horizontal,ds_vertical)
    scifi=scifi[cut]
    past_consecutive_time_diff = past_consecutive_time_diff[cut]
    scifi_hit_time = scifi_hit_time[cut]
    print("ds cut",scifi.shape)
    # missing, change when I add us

### CUTS
cut = apply_test_beam_cuts(scifi, past_consecutive_time_diff)
scifi=scifi[cut]
scifi_hit_time=scifi_hit_time[cut]
#next_consecutive_time_diff=next_consecutive_time_diff[cut]
past_consecutive_time_diff=past_consecutive_time_diff[cut]
if IS_THERE_DS_IN_DATA and not REJECT_DS:
    ds_horizontal=ds_horizontal[cut]
    ds_horizontal_time=ds_horizontal_time[cut]
    ds_vertical=ds_vertical[cut]
    ds_vertical_time=ds_vertical_time[cut]
    ds_horizontal, ds_horizontal_time, ds_vertical, ds_vertical_time = ds_pmt_cut_inside_event(ds_horizontal, ds_horizontal_time, ds_vertical, ds_vertical_time)

if IS_THERE_US_IN_DATA and not REJECT_US:
    us_signals=us_signals[cut]
    us_signals_time=us_signals_time[cut]
    us_signals, us_signals_time = pmt_cut_inside_event(us_signals,us_signals_time, min_pmt_qdc_value_us, time_window_min_us, time_window_max_us)


print("after hit cuts",scifi.shape)
## plots after cuts
#plot_1d_hist(next_consecutive_time_diff*6.25,N=50,xmin=100,xmax=120,x_label="Next(dont use stupid, future cannot affect past) Consec. Time Diff.[Time Clock]",label_str=label_str,outdir="1d_time_hist")
plot_1d_hist(past_consecutive_time_diff,N=50,xmin=99,xmax=1000, x_label="Consecutive Time Diff.[Time Clock](Strict cut)", title="Histogram of Consecutive Time Difference (Strict cut)" , label_str=label_str, outdir="1d_time_hist")
qdc_strict_cut = plot_and_save(scifi,us=None,ds_horizontal=ds_horizontal,ds_vertical=ds_vertical,cut_in_title_name="Strict Cut")

ith_ds_horizontal_time = ds_horizontal_time[15].reshape(-1)
ith_ds_horizontal_time = ith_ds_horizontal_time[ith_ds_horizontal_time!=0]
plot_1d_hist(ith_ds_horizontal_time,N=50,xmin=0,xmax=5,x_label="DS Horizontal Hit Time in a Event[Time Clock]", title="Histogram of DS Horizontal Hit Time in a Event" ,label_str=label_str,outdir="1d_time_hist")


ith_ds_vertical_time= ds_vertical_time[15].reshape(-1)
ith_ds_vertical_time= ith_ds_horizontal_time[ith_ds_horizontal_time!=0]
plot_1d_hist(ith_ds_vertical_time,N=50,xmin=0,xmax=5,x_label="DS Vertical Hit Time in a Event[Time Clock]", title="Histogram of DS Vertical Hit Time in a Event" ,label_str=label_str,outdir="1d_time_hist")

qdc_energy_list = [qdc_strict_cut,neg_qdc_loose_cut,pos_qdc_loose_cut]
label_str1 = [ "Pos. QDC(Full Cuts)","Neg. QDC(Loose Cuts)", "Pos. QDC(Loose Cuts)"]
alpha_list = [1,0.5,0.5]
replot_multiple_hist(qdc_energy_list,N=40,xmin=EN_MIN,xmax=EN_MAX,x_label="QDC Energy[GeV]",title=f"Histograms of SciFi+DS QDC Energy" ,label_str=label_str1,outdir="1d_qdc_energy_hist",alpha_list=alpha_list,name=label_str)

N=10
for i in range(0,N):
    plot_2d_im(scifi,i,"images/scifi", scifi.sum((1,2,3))*0.053 )



#plot_1d_hist(scifi,100,"QDC Energy[GeV]",label_str)
#qdc frac yerine qdc/energy tanımla
#scifi_qdc_distr(scifi,label_str)
#width_of_shower(scifi,label_str)
#get_shower_cluster(data,width)
"""cluster_half_width=768
scifi = get_shower_cluster(scifi,cluster_half_width)"""
