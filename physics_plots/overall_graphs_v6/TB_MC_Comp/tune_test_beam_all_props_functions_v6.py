import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import os
import numpy
import torch.nn.functional as F

# just use one of them. otherwise, it will call another prop.

#from TB_electron_hadron_comp.QDC.config_TB_QDC import *
from TB_MC_Comp.config import *
from dl_recon_core_sparse.data_loader import *


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
            plt.hist(qdc_energy,bins=20,label=label_str[i],alpha=alpha_list[i],histtype='step',density=True )

    
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



def load_as_lists(file_list,N,funct, name_data, label_name, config_feature_name):
    #torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    datasets = []          # FIX 1: Matched the list name
    label_counts = {}      # FIX 1: Initialized the dictionary
    for k,fname in enumerate(file_list):
        ds = SNDSparseDataset([0,fname], N)  
        if len(ds) > 0:
            datasets.append(ds)

    combined_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(
        combined_dataset, 
        batch_size=batch_size,   # FIX 3: Matched capitalization
        shuffle=is_train,        # FIX 2: Uses the new function argument
        num_workers=4, 
        pin_memory=True)

    for qdc_threshold_mc in [0,5]:
        for qdc_threshold_data in [0,5]
            scifi_list = []

            first_dataset = combined_dataset.datasets[0]
            second_dataset = combined_dataset.datasets[1]
            first_dataset.update_cut(qdc_threshold_value_scifi=qdc_threshold_data, smear_sigma=0)
            second_dataset.update_cut(qdc_threshold_value_scifi=qdc_threshold_mc, smear_sigma=0)

            scifi, us, ds, energy  = next(iter(dataloader))
            # calculate gaussian fit and save it in the rest. 

            if PLOT_MEAN_QDC_ENERGY:
                scifi_prop = torch.sum(scifi,dim=(1,3))

            elif PLOT_MEAN_HIT_NUMBERS:
                scifi_prop = torch.sum(scifi>0,dim=(1,3))


            scifi_list.append(scifi_prop)


            error_in_mean, error_in_std = funct(scifi_list, name_data, label_name, config_feature_name)

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


def plot_for_hdw(TEST_DATA_DIR,name, IS_MC_DATA_TITLE, feature_name):
    x_min, x_max=0,13000
    y_min, y_max=0,60000
    bins_x, bins_y = 40,40

    N=50000
    list1,list2,list3,list4 = load_as_lists(TEST_DATA_DIR,N)
    total_list1 = [tensor.sum(dim=(1)) for tensor in list1]
    total_list2 = list2
    total_list3 = list3
    total_list4 = list4

    alpha_list=[0.7,0.7,0.7,0.7,0.7,0.7]

    xlabel=feature_name
    name= xlabel+"_"+name
    print(len(total_list1))
    print(len(label_list_en))
    print(label_list_en)
    plot_multiple_hist(total_list1, None , None, None, xlabel,
                    f"{feature_name} Distribution (All SciFi Stations)"+IS_MC_DATA_TITLE, label_list_en, outdir=outdirname,alpha_list=[1,1,1,1,1,1], name="hist_total_scifi_"+name)
    plot_multiple_hist(total_list2, None , None, None, xlabel,
                    f"{feature_name} Distribution (Max Station)"+IS_MC_DATA_TITLE, label_list_en, outdir=outdirname,alpha_list=[1,1,1,1,1,1], name="hist_max_"+name)
    plot_multiple_hist(total_list3, None , None, None, xlabel,
                    f"{feature_name} Distribution (Max Hor. Station)"+IS_MC_DATA_TITLE, label_list_en, outdir=outdirname,alpha_list=[1,1,1,1,1,1], name="hist_max_HOR_"+name)
    plot_multiple_hist(total_list4, None , None, None, xlabel,
                    f"{feature_name} Distribution (Max Ver. Station)"+IS_MC_DATA_TITLE, label_list_en, outdir=outdirname,alpha_list=[1,1,1,1,1,1], name="hist_max_VER_"+name)

    beam_xlabel = "Beam Energy [GeV]"
    
    # 1. SciFi (total_list1)
    plot_1d_beam_energy_graphs(beam_en_list, total_list1, label_list_en, 
                               "mean_total_scifi_" + name, show_ideal=False, 
                               xlabel=beam_xlabel, ylabel=f'Mean {feature_name}', 
                               title=f"Mean {feature_name} vs. Beam Energy (All SciFi Stations)" + IS_MC_DATA_TITLE, 
                               outdir=outdirname)

    # 2. US Stations (total_list2)
    plot_1d_beam_energy_graphs(beam_en_list, total_list2, label_list_en, 
                               "mean_max_US_" + name, show_ideal=False, 
                               xlabel=beam_xlabel, ylabel=f'Mean {feature_name}',  
                               title=f"Mean {feature_name} vs. Beam Energy(Max Station)" + IS_MC_DATA_TITLE, 
                               outdir=outdirname)

    # 3. DS Horizontal (total_list3)
    plot_1d_beam_energy_graphs(beam_en_list, total_list3, label_list_en, 
                               "mean_MAX_HOR_" + name, show_ideal=False, 
                               xlabel=beam_xlabel, ylabel=f'Mean {feature_name}',  
                               title=f"Mean {feature_name} vs. Beam Energy(Max Hor.)" + IS_MC_DATA_TITLE, 
                               outdir=outdirname)

    # 4. DS Vertical (total_list4)
    plot_1d_beam_energy_graphs(beam_en_list, total_list4, label_list_en, 
                               "mean_MAX_VER_" + name, show_ideal=False, 
                               xlabel=beam_xlabel, ylabel=f'Mean {feature_name}', 
                               title=f"Mean {feature_name} vs. Beam Energy(Max Ver.)" + IS_MC_DATA_TITLE, 
                               outdir=outdirname)


    first_index_layer_second_index_energy_array=[]
    for i in range(5):
        ith_layer = [wtrenergy_tensor[:,i] for wtrenergy_tensor in list1]
        first_index_layer_second_index_energy_array.append(ith_layer)

    label_list=["Layer 1","Layer 2","Layer 3","Layer 4","Layer 5"]
    plot_1d_compare_2_domains_beam_energy_graphs(beam_en_list, first_index_layer_second_index_energy_array, label_list , "all_layers_hits"+name, show_ideal=False, xlabel="Beam Energy [GeV]",ylabel=f'SciFi {feature_name}',title=f"SciFi {feature_name} vs. Beam Energy"+IS_MC_DATA_TITLE, outdir=outdirname)

    for i in range(5):
        plot_1d_beam_energy_graphs(beam_en_list, [tensor[:,i] for tensor in list1], label_list_en , str(i+1)+name, show_ideal=False, xlabel="Beam Energy [GeV]",ylabel=f'SciFi {feature_name}',title=f"SciFi {feature_name} at the Station {i+1} vs. Beam Energy"+IS_MC_DATA_TITLE, outdir=outdirname)

        plot_multiple_hist([tensor[:,i] for tensor in list1], None , None, None, xlabel,
                    f"SciFi {feature_name} at the Station {i+1} Histogram"+IS_MC_DATA_TITLE, label_list_en, outdir=outdirname,alpha_list=[1,1,1,1,1,1], name="hist_layer_"+str(i+1)+name)

    if feature_name=="QDC":
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
        plot_1d_beam_energy_graphs(beam_en_list, total_qdc_energy_wtr_beam_energy , label_list_en , str(i+1)+name, show_ideal=False, xlabel="Beam Energy [GeV]",ylabel='SciFi QDC Energy[GeV]',title=f"SciFi QDC Energy vs. Beam Energy"+IS_MC_DATA_TITLE, outdir=outdirname)
        return first_index_layer_second_index_energy_array,  [total_qdc_energy_wtr_beam_energy, total_scifi_qdc_energy_wtr_beam_energy, total_us_ds_qdc_energy_wtr_beam_energy]
    else:
        return first_index_layer_second_index_energy_array


def plot_for_everything(scifi_list, name, IS_MC_DATA_TITLE, feature_name):
    x_min, x_max=0,13000
    y_min, y_max=0,60000
    bins_x, bins_y = 40,40
    print("PLOTTİNG FOR EVEVERYTHİNG")
    N=50000
    scifi_total = [tensor.sum(dim=(1)) for tensor in list1]

    alpha_list=[0.7,0.7,0.7,0.7,0.7,0.7]

    xlabel=feature_name
    name= xlabel+"_"+name
    print(len(total_list1))
    print(len(label_list_en))
    print(label_list_en)
    plot_multiple_hist(scifi_total, None , None, None, xlabel,
                    f"{feature_name} Distribution (All SciFi Stations)"+IS_MC_DATA_TITLE, label_list_en, outdir=outdirname,alpha_list=[1,1,1,1,1,1], name="hist_total_scifi_"+name)
  
    beam_xlabel = "Beam Energy [GeV]"
    

    for i in range(5):
        plot_multiple_hist([tensor[:,i] for tensor in list1], None , None, None, xlabel,
                    f"SciFi {feature_name} at the Station {i+1} Histogram"+IS_MC_DATA_TITLE, label_list_en, outdir=outdirname,alpha_list=[1,1,1,1,1,1], name="hist_layer_"+str(i+1)+name)

    #fit gaussian


def plot_for_frac(TEST_DATA_DIR,name, IS_MC_DATA_TITLE, feature_name):
    x_min, x_max=0,13000
    y_min, y_max=0,60000
    bins_x, bins_y = 40,40

    N=50000
    list1,list2,list3,list4 = load_as_lists(TEST_DATA_DIR,N)  

    total_scifi = [i.sum(dim=1) for i in list1]
    total_ds_v = [i.sum(dim=1) for i in list3]
    total_ds_h = [i.sum(dim=1) for i in list4]
    total_ds=[]
    for i in range(len(total_ds_h)):
        total_ds.append(total_ds_v[i]+total_ds_h[i])

    alpha_list=[0.7,0.7,0.7,0.7,0.7,0.7]

    xlabel=feature_name
    name= xlabel+"_"+name

    beam_xlabel = "Beam Energy [GeV]"
    
    ############################################################
    frac_highest_2layer=[]
    frac_highest_2layer_ordered=[]
    K=2
    for ith_energy_list in list1:
        topK_values, topK_indices = torch.topk(ith_energy_list, k=K, dim=1)
        
        # 1. Absolute highest / Second highest
        frac_abs = topK_values[:, 0] / topK_values[:, 1]
        # FIX: Keep only valid, finite numbers
        frac_abs = frac_abs[torch.isfinite(frac_abs)] 
        frac_highest_2layer.append(frac_abs.cpu())

        # 2. Sort indices by detector order (upstream -> downstream)
        topK_indices_sorted, _ = torch.sort(topK_indices, dim=1)

        # 3. Safely extract the energies
        selected_layers = torch.gather(ith_energy_list, dim=1, index=topK_indices_sorted)

        # 4. Compute fraction: Upstream-most / Downstream-most
        frac_ord = selected_layers[:, 0] / selected_layers[:, 1]
        # FIX: Keep only valid, finite numbers
        frac_ord = frac_ord[torch.isfinite(frac_ord)]
        frac_highest_2layer_ordered.append(frac_ord.cpu())

    plot_multiple_hist(frac_highest_2layer, 80 , 0, 250, xlabel,
                    f"{feature_name} Histogram"+IS_MC_DATA_TITLE, label_list_en, outdir=outdirname,alpha_list=[1,1,1,1,1,1], name="hist_"+name)
    plot_multiple_hist(frac_highest_2layer_ordered, 80 , 0, 150, xlabel,
                    f"{feature_name} in Order Histogram"+IS_MC_DATA_TITLE, label_list_en, outdir=outdirname,alpha_list=[1,1,1,1,1,1], name="hist_ordered_"+name)

    plot_multiple_hist(frac_highest_2layer, 80 , 0, 50, xlabel,
                    f"{feature_name} Histogram"+IS_MC_DATA_TITLE, label_list_en, outdir=outdirname,alpha_list=[1,1,1,1,1,1], name="middlehist_"+name)
    plot_multiple_hist(frac_highest_2layer_ordered, 80 , 0, 50, xlabel,
                    f"{feature_name} in Order Histogram"+IS_MC_DATA_TITLE, label_list_en, outdir=outdirname,alpha_list=[1,1,1,1,1,1], name="middlehist_ordered_"+name)


    plot_multiple_hist(frac_highest_2layer, 80 , 0, 50, xlabel,
                    f"{feature_name} Histogram"+IS_MC_DATA_TITLE, label_list_en, outdir=outdirname,alpha_list=[1,1,1,1,1,1], name="closerhist_"+name)
    plot_multiple_hist(frac_highest_2layer_ordered, 80 , 0, 50, xlabel,
                    f"{feature_name} in Order Histogram"+IS_MC_DATA_TITLE, label_list_en, outdir=outdirname,alpha_list=[1,1,1,1,1,1], name="closerhist_ordered_"+name)


    logarithmic = [torch.log(i) for i in frac_highest_2layer ]
    logarithmic_ordered = [torch.log(i) for i in frac_highest_2layer_ordered]

    plot_multiple_hist(logarithmic, 80 , 0, 3, "Log of "+xlabel,
                    f"Log of {feature_name} Histogram"+IS_MC_DATA_TITLE, label_list_en, outdir=outdirname,alpha_list=[1,1,1,1,1,1], name="loghist_"+name)
    plot_multiple_hist(logarithmic_ordered, 80 , -6.5, 3, "Log of "+xlabel,
                    f"Log of {feature_name} in Order Histogram"+IS_MC_DATA_TITLE, label_list_en, outdir=outdirname,alpha_list=[1,1,1,1,1,1], name="loghist_ordered_"+name)

    plot_multiple_hist(logarithmic, 80 , 0, 8, "Log of "+xlabel,
                    f"Log of {feature_name} Histogram"+IS_MC_DATA_TITLE, label_list_en, outdir=outdirname,alpha_list=[1,1,1,1,1,1], name="log_larger_hist_"+name)
    plot_multiple_hist(logarithmic_ordered, 80 ,-8.5, 9, "Log of "+xlabel,
                    f"Log of {feature_name} in Order Histogram"+IS_MC_DATA_TITLE, label_list_en, outdir=outdirname,alpha_list=[1,1,1,1,1,1], name="log_larger_hist_ordered_"+name)



    """plot_multiple_hist(frac_highest_2layer, 40 , 0, 20, xlabel,
                    f"QDC Fraction of Highest 2 Layer Histogram"+IS_MC_DATA_TITLE, label_list_en, outdir=outdirname,alpha_list=[1,1,1,1,1,1], name="frac_hist_"+str(i+1)+name)
    plot_multiple_hist(frac_highest_2layer_ordered, 40 , 0, 20, xlabel,
                    f"QDC Fraction of Highest 2 Layer in Order Histogram"+IS_MC_DATA_TITLE, label_list_en, outdir=outdirname,alpha_list=[1,1,1,1,1,1], name="frac_hist_ordered_"+str(i+1)+name)"""

