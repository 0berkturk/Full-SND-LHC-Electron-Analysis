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
    plt.savefig("signals/"+ out_name+ str(j) + ".png", dpi=200)
    plt.clf()

def plot_qdc_energy_only_scifi(true_en, total_qdc, bins, out_name):

    # SciFi QDC already multiplied by 0.059 BEFORE calling this function
    # Sum QDC over channels, width, height -> shape (N,)

    avg_qdc = []
    std_qdc = []

    # ---------------------------------------------------------
    # Compute average and std in each energy (momentum) bin
    # ---------------------------------------------------------
    for i in range(len(bins) - 1):
        en_min = bins[i]
        en_max = bins[i + 1]

        idx = (true_en >= en_min) & (true_en < en_max)

        if idx.sum() == 0:
            avg_qdc.append(0)
            std_qdc.append(0)
            continue

        avg_qdc.append(total_qdc[idx].mean().item())
        std_qdc.append(total_qdc[idx].std().item())

    # ---------------------------------------------------------
    # Create output directory
    # ---------------------------------------------------------
    out_dir = "Electron_en_qdc"
    os.makedirs(out_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 1) Average QDC × 0.059 vs Generated Momentum
    # ---------------------------------------------------------
    centers = (bins[:-1] + bins[1:]) / 2

    plt.figure()

    # Main curve (what you actually measure)
    plt.errorbar(
        centers,
        avg_qdc,
        yerr=std_qdc,
        fmt='o-',
        alpha=0.7,
        label='Avg QDC × 0.059'
    )

    # Ideal theoretical line: QDC × 0.059 = p_generated
    plt.plot(
        centers,
        centers,
        '--',
        linewidth=2,
        label='Ideal'
    )

    plt.xlabel('Generated Momentum [GeV]')
    plt.ylabel('Average QDC × 0.059 [GeV]')
    plt.title('Average QDC × 0.059 vs Generated Momentum')
    plt.legend()
    plt.grid()

    plt.savefig(f"{out_dir}/{out_name}_average_qdc_energy.png", dpi=300)
    plt.clf()

    # ---------------------------------------------------------
    # 2) 2D Histogram: Total QDC × 0.059 vs Energy
    # ---------------------------------------------------------
    plt.figure()
    cmap = plt.get_cmap('plasma')
    try:
        cmap.set_under('white')
    except:
        pass

    plt.gca().set_facecolor('white')
    plt.plot(
        centers,
        centers,
        '--',
        linewidth=2,
        label='Ideal'
    )
    plt.hist2d(true_en.cpu(), total_qdc.cpu(), bins=50, cmap=cmap, vmin=0.1)
    plt.colorbar(label='Counts')
    plt.xlabel('Generated Momentum [GeV]')
    plt.ylabel('Total QDC × 0.059 [GeV]')
    plt.title('Total QDC × 0.059 vs Generated Momentum')
    plt.grid()

    plt.savefig(f"{out_dir}/{out_name}_qdc_energy.png", dpi=300)
    plt.clf()

    # ---------------------------------------------------------
    # 3) 2D Histogram: SciFi QDC × 0.059 vs Energy
    # ---------------------------------------------------------
    plt.figure()
    plt.plot(
        centers,
        centers,
        '--',
        linewidth=2,
        label='Ideal'
    )
    plt.hist2d(true_en.cpu(), total_qdc.cpu(), bins=np.arange(0,50,2), cmap=cmap, vmin=0.1)
    plt.colorbar(label='Counts')
    plt.xlabel('Generated Momentum [GeV]')
    plt.ylabel('SciFi QDC × 0.059 [GeV]')
    plt.title(f'SciFi QDC × 0.059 vs Generated Momentum ({out_name})')
    plt.grid()

    plt.savefig(f"{out_dir}/{out_name}_scifi_smaller_energy.png", dpi=300)
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



def frac_distribution_in_layers(scifi_data, us,ds,label_str):
    layer_numbers = find_shower_starting_layer(scifi_data, threshold=1)
    total_qdc = scifi_data.sum(dim=(1,2,3)) + us.sum(dim=(1,2,3)) + ds.sum(dim=(1,2,3))
    mean_frac_list_scifi = [ (scifi_data[layer_numbers == i].sum(dim=(1,2,3)) / total_qdc[layer_numbers==i] ).mean().item() for i in range(1, 6)]
    mean_frac_list_us = [ (us[layer_numbers == i].sum(dim=(1,2,3)) / total_qdc[layer_numbers==i] ).mean().item() for i in range(1, 6)]
    mean_frac_list_ds = [ (ds[layer_numbers == i].sum(dim=(1,2,3)) / total_qdc[layer_numbers==i] ).mean().item() for i in range(1, 6)]

    std_frac_list_scifi = [ (scifi_data[layer_numbers == i].sum(dim=(1,2,3)) / total_qdc[layer_numbers==i] ).std().item() for i in range(1, 6)]
    std_frac_list_us = [ (us[layer_numbers == i].sum(dim=(1,2,3)) / total_qdc[layer_numbers==i] ).std().item() for i in range(1, 6)]
    std_frac_list_ds = [ (ds[layer_numbers == i].sum(dim=(1,2,3)) / total_qdc[layer_numbers==i] ).std().item() for i in range(1, 6)]

    x = np.arange(1, 6)
    plt.figure()
    std_frac_list_scifi = np.clip(std_frac_list_scifi, 0, 1)
    std_frac_list_us = np.clip(std_frac_list_us, 0, 1)
    std_frac_list_ds = np.clip(std_frac_list_ds, 0, 1)

    plt.errorbar(x, mean_frac_list_scifi, yerr=std_frac_list_scifi, fmt='o-', label='SciFi Fraction',alpha=0.7)
    plt.errorbar(x, mean_frac_list_us, yerr=std_frac_list_us, fmt='s-', label='US Fraction',alpha=0.7)
    plt.errorbar(x, mean_frac_list_ds, yerr=std_frac_list_ds, fmt='^-', label='DS Fraction',alpha=0.7)

    plt.xlabel('Starting SciFi Layer of Shower')
    plt.ylabel('Average Fraction of Total QDC')
    plt.legend()
    plt.title(f'Average Fraction of Total QDC per Starting SciFi Layer ({label_str})')
    plt.grid()

    outdir = "frac_from_scifi_starting_layer"
    os.makedirs(outdir, exist_ok=True)

    plt.savefig(f"{outdir}/{label_str}_average_frac_scifi_qdc_per_starting_layer.png", dpi=300)
    plt.close()

def frac_distribution_in_layers_wtr_shower_max(scifi_data, us,ds,label_str):
    layer_numbers,h,v = find_shower_max(scifi)
    print("shower max layers",layer_numbers) 
    scifi_sum = scifi_data.sum(dim=(1,2,3))
    us_sum = us.sum(dim=(1,2,3))
    ds_sum = ds.sum(dim=(1,2,3))
    total_qdc = scifi_sum + us_sum + ds_sum


    mean_frac_list_scifi = [ (scifi_sum[(layer_numbers == i)&(scifi_sum > 0)] / total_qdc[(layer_numbers == i)&(scifi_sum > 0)] ).mean().item() for i in range(0, 5)]
    mean_frac_list_us = [ (us_sum[(layer_numbers == i)&(scifi_sum > 0)] / total_qdc[(layer_numbers == i)&(scifi_sum > 0)] ).mean().item() for i in range(0, 5)]
    mean_frac_list_ds = [ (ds_sum[(layer_numbers == i)&(scifi_sum > 0)] / total_qdc[(layer_numbers == i)&(scifi_sum > 0)] ).mean().item() for i in range(0, 5)]

    std_frac_list_scifi = [ (scifi_sum[(layer_numbers == i)&(scifi_sum > 0)]/ total_qdc[(layer_numbers == i)&(scifi_sum > 0)] ).std().item() for i in range(0, 5)]
    std_frac_list_us = [ (us_sum[(layer_numbers == i)&(scifi_sum > 0)]/ total_qdc[(layer_numbers == i)&(scifi_sum > 0)] ).std().item() for i in range(0, 5)]
    std_frac_list_ds = [ (ds_sum[(layer_numbers == i)&(scifi_sum > 0)] / total_qdc[(layer_numbers == i)&(scifi_sum > 0)] ).std().item() for i in range(0, 5)]

    std_frac_list_scifi = np.clip(std_frac_list_scifi, 0, 1)
    std_frac_list_us = np.clip(std_frac_list_us, 0, 1)
    std_frac_list_ds = np.clip(std_frac_list_ds, 0, 1)
    x = np.arange(1, 6)
    plt.figure()
    plt.errorbar(x, mean_frac_list_scifi, yerr=std_frac_list_scifi, fmt='o-', label='SciFi Fraction',alpha=0.7)
    plt.errorbar(x, mean_frac_list_us, yerr=std_frac_list_us, fmt='s-', label='US Fraction',alpha=0.7)
    plt.errorbar(x, mean_frac_list_ds, yerr=std_frac_list_ds, fmt='^-', label='DS Fraction',alpha=0.7)

    plt.xlabel("SciFi Layer at Shower Maximum")
    plt.ylabel("Average Fraction of Total QDC")
    plt.title(f"Average QDC Fraction per Layer at Shower Maximum ({label_str})")
    plt.legend()

    plt.grid()

    outdir = "frac_from_scifi_starting_layer"
    os.makedirs(outdir, exist_ok=True)

    plt.savefig(f"{outdir}/{label_str}_average_frac_scifi_qdc_per_shower_max.png", dpi=300)
    plt.close()



def scifi_qdc_distr(scifi):
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

def plot_1d(data,labels,bins,x_axis,out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for i in range(len(data)):
        plt.hist(data[i],bins,label=labels[i],alpha=0.5)
    plt.yscale("log")
    plt.xlabel(x_axis)
    plt.ylabel("Counts")
    plt.legend()
    plt.title(f"Histogram of {x_axis}")
    plt.savefig(f"{out_dir}/hist_{x_axis}.png",dpi=300)
    plt.clf()

def plot_2d_mean_std_graph(x_data_list, y_data_list,label_list ,bins, x_label, y_label, title, out_dir,error_bar=True ,ideal_line=False):
    """
    Plots mean and std of y_data in bins of x_data.
    Args:
        x_data (torch.Tensor or np.array): variable to bin (e.g., energy or momentum)
        y_data (torch.Tensor or np.array): variable to average in each bin (e.g., QDC)
        bins (array-like): bin edges for x_data
        x_label (str): x-axis label
        y_label (str): y-axis label
        title (str): plot title
        out_dir (str): output directory to save the figure
        out_name (str): file name (without extension)
        ideal_line (bool): if True, draw y=x line for reference
    """
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    for i in range(len(x_data_list)):
        x_data =x_data_list[i]
        y_data = y_data_list[i].float()
        label=label_list[i]

        avg_y = []
        std_y = []

        # Compute mean and std in each x bin
        for i in range(len(bins) - 1):
            x_min = bins[i]
            x_max = bins[i + 1]

            idx = (x_data >= x_min) & (x_data < x_max)

            if idx.sum() == 0:
                avg_y.append(0)
                std_y.append(0)
            else:
                avg_y.append(y_data[idx].mean().item())
                std_y.append(y_data[idx].std().item())

        # Create output directory
        # Bin centers
        centers = (bins[:-1] + bins[1:]) / 2
        # Plot
        
        if error_bar:
            plt.errorbar(
                centers,
                avg_y,
                yerr=std_y,
                fmt='o-',
                alpha=0.7,
                label=label
            )
        else:
            plt.plot(
                centers,
                avg_y,
                alpha=0.7,
                label=label
            )

    if ideal_line:
        plt.plot(
            centers,
            centers,
            '--',
            linewidth=2,
            label='Ideal'
        )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid()

    # Save figure
    plt.savefig(f"{out_dir}/2d_{x_label}_{y_label}.png", dpi=300)
    plt.clf()

import torch

def hit_number_dict(scifi_list, scifi_sum_list):
    """
    Calculates hit-related quantities for a list of SciFi events.

    Args:
        scifi_list: list of torch.Tensor, each shape (N, 2, 5, 1536)
        gen_mom_list: list of torch.Tensor, each shape (N,)
        scifi_sum_list: list of torch.Tensor, total QDC per event, shape (N,)

    Returns:
        Dictionary of lists containing calculated tensors for all datasets.
    """

    # Lists to store results for each dataset
    all_total_hits = []
    all_hits_per_layer = []
    all_hits_per_orientation = []
    all_activated_layers = []
    all_hits_per_active_layer = []
    all_hits_per_total_qdc = []
    all_hits_per_total_qdc_per_act_layer = []

    for i in range(len(scifi_list)):
        scifi = scifi_list[i]
        scifi_sum = scifi_sum_list[i]
        # Boolean mask of hits
        all_hits_mask = (scifi != 0)

        # Total hits per event
        total_hits = all_hits_mask.sum(dim=(1,2,3))  # shape: (N,)

        # Hits per layer (summed over orientation and pixels)
        hits_per_layer = all_hits_mask.sum(dim=(1,3))  # shape: (N,5)

        # Hits per orientation (summed over layers and pixels)
        hits_per_orientation = all_hits_mask.sum(dim=(2,3))  # shape: (N,2)

        # Number of activated layers
        activated_layers = (hits_per_layer != 0).sum(dim=1)  # shape: (N,)

        # Average hits per active layer
        hits_per_active_layer = total_hits / activated_layers.float()

        # Total hits normalized by total QDC per event
        hits_per_total_qdc = total_hits.float() / scifi_sum.float()

        hits_per_total_qdc_per_act_layer = hits_per_total_qdc/activated_layers

        # Append results to lists
        all_total_hits.append(total_hits)
        all_hits_per_layer.append(hits_per_layer)
        all_hits_per_orientation.append(hits_per_orientation)
        all_activated_layers.append(activated_layers)
        all_hits_per_active_layer.append(hits_per_active_layer)
        all_hits_per_total_qdc.append(hits_per_total_qdc)
        all_hits_per_total_qdc_per_act_layer.append(hits_per_total_qdc_per_act_layer)

    # Return all results as a dictionary
    return {
        "total_hits": all_total_hits,
        "hits_per_layer": all_hits_per_layer,
        "hits_per_orientation": all_hits_per_orientation,
        "activated_layers": all_activated_layers,
        "hits_per_active_layer": all_hits_per_active_layer,
        "hits_per_total_qdc": all_hits_per_total_qdc,
        "hits_per_total_qdc_per_act_layer": all_hits_per_total_qdc_per_act_layer
    }



def hit_number_plots(scifi_list,gen_mom_list,scifi_sum_list,label_list,bins_en):

    plot1d=False

    hit_dict=hit_number_dict(scifi_list,scifi_sum_list)
    print( hit_dict["total_hits"])
    print(len(gen_mom_list),len(scifi_list),len(scifi_sum_list))

    plot_2d_mean_std_graph(gen_mom_list[:2], hit_dict["total_hits"][:2] ,label_list[:2] ,bins_en,  "Generated Momentum","Total Hits", "Total Hit vs. Generated Momentum", "hit_plots",True)
    plot_2d_mean_std_graph(scifi_sum_list, hit_dict["total_hits"] ,label_list ,bins_en,  "QDC Energy","Total Hits", "Total Hit vs. QDC Energy", "hit_plots",True)
    
    plot_2d_mean_std_graph(gen_mom_list[:2], hit_dict["hits_per_total_qdc"][:2] ,label_list[:2] ,bins_en,  "Gen. Mom", "Total Hits per QDC Energy", "Total Hits per QDC Energy vs. Gen. Mom", "hit_plots",False)
    plot_2d_mean_std_graph(hit_dict["activated_layers"][:2], hit_dict["hits_per_total_qdc"][:2] ,label_list , np.array([1,2,3,4,5,6])-0.5 ,  "Activated Layer ","Total Hits per QDC Energy", "Total Hits per QDC Energy vs. Activated Layer", "hit_plots",False)

    plot_2d_mean_std_graph(gen_mom_list[:2], hit_dict["hits_per_total_qdc_per_act_layer"][:2] ,label_list[:2] , bins_en ,  "Gen. Mom","Total Hits per QDC Energy per Act. Layer", "Total Hits per QDC Energy per Act. Layer vs. Gen. Mom", "hit_plots",False)


    plot_2d_mean_std_graph(hit_dict["activated_layers"], hit_dict["total_hits"] ,label_list , np.array([1,2,3,4,5,6])-0.5,  "Activated Layer","Total Hits", "Total Hit vs. Activated Layer", "hit_plots",True)

    plot_2d_mean_std_graph(gen_mom_list[:2], hit_dict["activated_layers"][:2], label_list[:2] ,bins_en,  "Generated Momentum", "Activated Layer", "Generated Momentum vs. Activated Layer", "hit_plots",True)
    plot_2d_mean_std_graph(scifi_sum_list, hit_dict["activated_layers"], label_list ,bins_en,  "QDC Energy", "Activated Layer", "QDC Energy vs. Activated Layer", "hit_plots",True)

    plot_2d_mean_std_graph(gen_mom_list, scifi_sum_list, label_list ,bins_en,  "Generated Momentum","QDC Energy", "QDC Energy vs. Generated Momentum", "hit_plots",True)

    if plot1d:
        plot_1d([total_hits_per_event],["Total Hit"], 50, "Hit Number", "hit_plots")
        plot_1d(
            [total_hits_per_event_foreach_layer[:, i] for i in range(5)],
            [f"Hits in Layer {i+1}" for i in range(5)],
            50,
            "Number of Hits for Each Layer",
            "hit_plots"
        )

        # 3) Hits per orientation (2 views)
        plot_1d(
            [total_hits_per_event_per_orientation[:, i] for i in range(2)],
            [f"Hits in Orientation {i+1}" for i in range(2)],
            50,
            "Number of Hits along XZ-YZ Plane",
            "hit_plots"
        )

        # 4) Number of activated layers
        plot_1d(
            [activated_layer],
            ["Activated Layer"],
            [1,2,3,4,5,6],  # 1 to 5 layers
            "Number of Activated Layers",
            "hit_plots"
        )

        # 5) Hits per active layer (normalized)
        plot_1d(
            [total_hits_per_layer],
            ["Hits per Active Layer"],
            50,
            "Average Hits per Active Layer",
            "hit_plots")

def open_data(file,apply_cut=False):

    k=0.059
    data=torch.load(file)
    scifi = k*data["scifi_signals"]
    scifi_sum = scifi.sum(dim=(1, 2, 3))

    if "en3d" in data.keys():
        energy = data["en3d"]
    else:
        energy = scifi_sum
    index = scifi_sum > 0 
    if apply_cut:
        us = data["us_signals"]
        ds_h = data["ds_horizontal"]
        ds_v = data["ds_vertical"]

        us_sum = data["us_signals"].sum(dim=(1, 2, 3))
        ds_sum_v = data["ds_vertical"].sum(dim=(1, 2))
        ds_sum_h = data["ds_horizontal"].sum(dim=(1, 2))

        index = index & (ds_sum_v == 0) & (ds_sum_h==0) & (us_sum == 0)
    energy=energy[index]
    scifi_sum = scifi_sum[index]
    scifi=scifi[index]
    return scifi, energy, scifi_sum


TEST_DATA_DIR = ["/eos/user/b/beturk/snd/MonteCarlo/onlyscifi_withcuts/onlyscifi__combined_kshort_0_100_2025.pt",
#"/eos/user/b/beturk/snd/MonteCarlo/particle_gun_advanced_v1/merge_electron_gun_0_50/test_combined_electron_0_50_2025.pt"
"/eos/user/b/beturk/snd/MonteCarlo/magnetic_moment_mc/magnetic_mom_mc_0.pt",
"/eos/user/b/beturk/snd/MonteCarlo/data/data_2025_309.pt"
]
print("openning ", TEST_DATA_DIR)

scifi_list=[]
energy_list=[]
scifi_sum_list=[]

for file in TEST_DATA_DIR:
    if "magnetic_mom_mc_0" in file:
        apply_cut=True
    else:
        apply_cut=False
    data=open_data(file,apply_cut)
    scifi_list.append(data[0])
    energy_list.append(data[1])
    scifi_sum_list.append(data[2])


bins_en = np.array([0,1,2,3,4,5,6,8,10,14,18,25,30,35,40,45,50,60,70,80,100])
hit_number_plots(scifi_list, energy_list, scifi_sum_list ,["Kaon","Electron","Data"],bins_en)




#plot_2d_mean_std_graph(energy, y_data, bins, x_label, y_label, title, out_name, ideal_line=True)
#frac_distribution_in_layers_wtr_shower_max(scifi, us,ds,label_str)
#scifi_qdc_distr(scifi)
#frac_distribution_in_layers(scifi, us,ds,label_str)
#width_of_shower(scifi,label_str)
