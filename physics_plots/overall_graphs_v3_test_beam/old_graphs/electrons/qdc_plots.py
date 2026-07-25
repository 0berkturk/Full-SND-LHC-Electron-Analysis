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
def plot_qdc_energy(true_en, scifi,us,ds, out_name):
# Sum over channels, width, and height dimensions (dimensions 1, 2, 3)
    total_qdc_scifi = scifi.sum(dim=(1, 2, 3))
    total_qdc_us = us.sum(dim=(1, 2, 3))
    total_qdc_ds = ds.sum(dim=(1, 2, 3))
    total_qdc = total_qdc_scifi + total_qdc_us + total_qdc_ds

    bins=np.linspace(0,2000,5)
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

    plt.figure()

    plt.errorbar((bins[:-1]+bins[1:])/2, average_qdc, yerr=std_qdc, fmt='o-', label='Total QDC',alpha=0.7)
    plt.errorbar((bins[:-1]+bins[1:])/2, average_scifi_qdc, yerr=std_scifi_qdc, fmt='s-', label='SciFi QDC',alpha=0.7)
    plt.errorbar((bins[:-1]+bins[1:])/2, average_us_qdc, yerr=std_us_qdc, fmt='^-', label='US QDC',alpha=0.7)
    plt.errorbar((bins[:-1]+bins[1:])/2, average_ds_qdc, yerr=std_ds_qdc, fmt='d-', label='DS QDC',alpha=0.7)
    plt.xlabel('Electron Energy [GeV]')
    plt.ylabel('Average QDC')
    plt.title(f'Average QDC vs Electron Energy')
    plt.legend()
    plt.grid()
    if os.path.exists("Electron_en_qdc") is False:
        os.mkdir("Electron_en_qdc")
    plt.savefig("Electron_en_qdc/"+out_name+"_average_qdc_energy.png", dpi=300)
    plt.clf()

    cmap = plt.get_cmap('plasma')
    cmap.set_under('white')
    plt.gca().set_facecolor('white')

    plt.figure()
    plt.hist2d(true_en.cpu(), total_qdc.cpu(), bins=50, cmap=cmap, vmin=0.1)
    plt.colorbar(label='Counts')
    plt.xlabel('Electron Energy [GeV]')
    plt.ylabel('Total QDC')
    plt.title('Total QDC vs Electron Energy')
    plt.grid()
    if os.path.exists("Electron_en_qdc") is False:
        os.mkdir("Electron_en_qdc")
    plt.savefig("Electron_en_qdc/"+out_name+"_qdc_energy.png", dpi=300)
    plt.clf()

    plt.figure()
    plt.hist2d(true_en.cpu(), total_qdc_scifi.cpu(), bins=50, cmap=cmap, vmin=0.1)
    plt.colorbar(label='Counts')
    plt.xlabel('True Energy [GeV]')
    plt.ylabel('SciFi QDC')
    plt.title(f'SciFi QDC vs True Energy({out_name})')
    plt.grid()
    plt.savefig("Electron_en_qdc/"+out_name+"_scifi_qdc_energy.png", dpi=300)
    plt.clf()

    plt.figure()
    plt.hist2d(true_en.cpu(), total_qdc_us.cpu(), bins=50, cmap=cmap, vmin=0.1)
    plt.colorbar(label='Counts')
    plt.xlabel('True Energy [GeV]')
    plt.ylabel('US QDC')
    plt.title(f'US QDC vs True Energy({out_name})')
    plt.grid()
    plt.savefig("Electron_en_qdc/"+out_name+"_us_qdc_energy.png", dpi=300)
    plt.clf()

    plt.figure()
    plt.hist2d(true_en.cpu(), total_qdc_ds.cpu(), bins=50, cmap=cmap, vmin=0.1)
    plt.colorbar(label='Counts')
    plt.xlabel('True Energy [GeV]')
    plt.ylabel('DS QDC')
    plt.title(f'DS QDC vs True Energy({out_name})')
    plt.grid()
    plt.savefig("Electron_en_qdc/"+out_name+"_ds_qdc_energy.png", dpi=300)
    plt.close()
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


import h5py

TEST_DATA_DIR = ["/afs/cern.ch/work/b/beturk/private/snd/test_Beam/electrons_all_50gev_100933_0.pt"]
h5f=torch.load(TEST_DATA_DIR[0])
scifi = h5f["scifi_signals"]
us = h5f["us_signals"]
ds_h = h5f["ds_horizontal"]
ds_v = h5f["ds_vertical"]
energy = scifi.sum((1,2,3))*0.059
print(scifi.shape)

label_index = 0 # 0: e, 1: numu, 2: tau, 3: nc
if label_index == 0:
    label_str = "Electron"

ds_h_padded = torch.nn.functional.pad(ds_h, (0, 0, 0, 1))
ds = torch.stack([ds_h_padded, ds_v], dim=1) #2x4x60
#print("ds shape:", ds.shape)  # Debugging line


#qdc frac yerine qdc/energy tanımla
plot_qdc_energy(energy, scifi, us,ds, label_str)
frac_distribution_in_layers_wtr_shower_max(scifi, us,ds,label_str)
scifi_qdc_distr(scifi)
frac_distribution_in_layers(scifi, us,ds,label_str)
#width_of_shower(scifi,label_str)