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
from TB_MC_DATA_COMP.config import *
from dl_recon_core_sparse.data_loader import *


cmap = plt.get_cmap('plasma')
cmap.set_under('white')

def hdw_all_fast_conv(scifi_qdc, us=None,ds=None):
    """
    scifi_qdc: (N, 2, 5, 1536)
    """
    delta_ch = HDW_CHANNEL
    N = scifi_qdc.shape[0]
    hits = (scifi_qdc !=-999).float()    # (N,2,5,1536)

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

    N=50
    bins = np.linspace(min_en,max_en,N)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    bins_recon = np.linspace(np.min(y_tensor),np.max(y_tensor),N)


def plot_2d_hist(x_data, y_data,
                 bins_x=None, bins_y=None,
                 out_name="qdc_vs_true", xlabel="True Energy [GeV]",
                 ylabel='QDC Energy [GeV]', title="QDC vs True Energy",
                 outdir="qdc_comparison"):
    
    plt.figure(figsize=(8,6))

    if bins_x!=None:
        hist = plt.hist2d(x_data, y_data, bins=[bins_x, bins_y],
                        cmap=cmap, alpha=0.8, vmin=1)  
    else:
        hist = plt.hist2d(x_data, y_data, bins=[40, 40],
                        cmap=cmap, alpha=0.8, vmin=1)  

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
        
        plt.errorbar(true_en, average_scifi_qdc, yerr=std_scifi_qdc, fmt='s-', label=label_list[k],alpha=0.7, color)

    

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


def plot_1d_compare_3_domains_beam_energy_graphs(true_en_list, mc_data_first_index, label_list,out_name ,xlabel="True Energy [GeV]",ylabel='Average QDC Energy[GeV]',title="Average QDC vs True Energy",outdir="qdc_comparision",show_ideal=False,skip_some=list,color_list=color_list):
    plt.figure()
    # Sum over channels, width, and height dimensions (dimensions 1, 2, 3)
    a=0
    for i, first_index_layer_second_index_energy_array in enumerate(mc_data_first_index):
        for k,qdc_energy_list in enumerate(first_index_layer_second_index_energy_array):

            average_scifi_qdc = []
            std_scifi_qdc = []

            for j in range(len(qdc_energy_list)):
                total_qdc_scifi = qdc_energy_list[j].float()
                true_en = true_en_list

                average_scifi_qdc.append(total_qdc_scifi.mean().item())
                std_scifi_qdc.append(total_qdc_scifi.std().item())
            if skip_some[a]==0:
                plt.errorbar(true_en, average_scifi_qdc, yerr=std_scifi_qdc, fmt='s-', label=label_list[a],alpha=0.7,color=color_list[i])
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


import pandas as pd

def apply_scifi2y_calibration(scifi_sig, csv_path="/afs/cern.ch/work/b/beturk/private/snd/overall_graphs_v7/QDC_offset_st2_mat0_ori0.csv", station=1, orientation=0):
    # 1. Load Zhibin's CSV
    df = pd.read_csv(csv_path)
    
    # 2. Create an empty 1D tensor for the 512 channels
    num_channels = scifi_sig.shape[-1]
    offsets = torch.zeros(num_channels, dtype=scifi_sig.dtype, device=scifi_sig.device)

    channels = df['channel'].values
    mpv_vals = df['mpv_data_avg'].values
    offsets[channels] = torch.tensor(mpv_vals, dtype=scifi_sig.dtype, device=scifi_sig.device)
    
    # 4. Isolate the target plane (Station 2, Orientation 0)
    target_plane = scifi_sig[:, orientation, station, :]
    
    # Create a mask to ignore the -999 empty hits
    valid_mask = target_plane != -999

    scifi_sig[:, orientation, station, :] = torch.where(
        valid_mask,
        target_plane - offsets,
        target_plane
    )
    return scifi_sig

def RUN_FINAL_MEGA_COMP_ALL_V8(config,datalist,dict):

    datasets=[]
    for test_data_name in datalist:
        datasets.append(SNDSparseDataset(test_data_name,perc=config.TOTAL_TEST_SIZE))

    for TB_RECALIBRATION_S2Y in dict["TB_RECALIBRATION_S2Y"]:
        for qdc_threshold_value_scifi_data in dict["qdc_threshold_value_scifi_data"]:
            for qdc_threshold_value_scifi_mc in dict["qdc_threshold_value_scifi_mc"]:
                for t_window_data in dict["t_window_data"]:
                    for t_window_mc in dict["t_window_mc"]:


                        cut_dir_data_name=f"S2Ycal{TB_RECALIBRATION_S2Y}_qdcthredata{qdc_threshold_value_scifi_data}_twindata{t_window_data[0]}{t_window_data[1]}"
                        cut_dir_MC_name=f"qdcthremc{qdc_threshold_value_scifi_mc}_twinmc{t_window_mc[0]}{t_window_mc[1]}"

                        common_dir = f"plots/{cut_dir_data_name}_{cut_dir_MC_name}/"
                        os.makedirs(common_dir, exist_ok=True)

                        hit_list_data=[]
                        hit_list_mc=[]
                        qdc_list_data=[]
                        qdc_list_mc=[]

                        for i,test_data_name in enumerate(datalist):
                            ith_dataset=datasets[i]
                            if "MC" in test_data_name[1]:
                                ith_dataset.update_hit_cuts( 
                                t_window_high_mc=t_window_mc[1], t_window_low_mc=t_window_mc[0], qdc_thresh_mc=qdc_threshold_value_scifi_mc)
                            else:
                                ith_dataset.update_hit_cuts( 
                                t_window_high_data=t_window_data[1], t_window_low_data=t_window_data[0], qdc_thresh_data=qdc_threshold_value_scifi_data,
                                TB_RECALIBRATION_S2Y=TB_RECALIBRATION_S2Y)


                            ith_dataset = DataLoader(ith_dataset, batch_size=config.TOTAL_TEST_SIZE, shuffle=False)
                            scifi_sig = next(iter(ith_dataset))[0]
                            print(scifi_sig.shape)
                            cut = scifi_sig!=0
                            hits_per_layers = torch.sum(cut,dim=(1,3))
                            qdc_per_layers = torch.sum(scifi_sig,dim=(1,3))

                            if i <= config.DATA_MC_INDEX_SEPARATION:  
                                hit_list_data.append(hits_per_layers)
                                qdc_list_data.append(qdc_per_layers)
                            else:
                                hit_list_mc.append(hits_per_layers)
                                qdc_list_mc.append(qdc_per_layers)

                    
                        feat_name="Hit Number"
                        data_first_index_layer_second_index_energy_array = plot_for_everything(hit_list_data, NAME, IS_MC_DATA_TITLE, feat_name, common_dir+feat_name, label_list_en[:DATA_MC_INDEX_SEPARATION+1] )

                        if DATA_MC_INDEX_SEPARATION<len(data_first_index_layer_second_index_energy_array): ## if it is very large number, it means no mc data comparision.
                            mc_first_index_layer_second_index_energy_array = plot_for_everything(hit_list_mc, NAME, "MC", feat_name ,common_dir+feat_name, label_list_en[DATA_MC_INDEX_SEPARATION+1,:] )
                            plot_1d_compare_3_domains_beam_energy_graphs(true_en_list, [data_first_index_layer_second_index_energy_array, mc_first_index_layer_second_index_energy_array], label_list_en, feat_name, xlabel="True Energy [GeV]", ylabel=feat_name , title=feat_name+" vs True Energy", outdir=common_dir+"MC_DATA_DOMAIN_COMP", show_ideal=False,skip_some=skip_somelist,color_list=color_list)

                        feat_name="QDC"
                        data_first_index_layer_second_index_energy_array,total_scifi_qdc_energy_wtr_beam_energy = plot_for_everything(qdc_list_data, NAME, IS_MC_DATA_TITLE, feat_name, common_dir+feat_name, label_list_en[:DATA_MC_INDEX_SEPARATION+1] )

                        if DATA_MC_INDEX_SEPARATION<len(data_first_index_layer_second_index_energy_array): ## if it is very large number, it means no mc data comparision.
                            mc_first_index_layer_second_index_energy_array,total_scifi_qdc_energy_wtr_beam_energy = plot_for_everything(qdc_list_mc, NAME, "MC", feat_name ,common_dir+feat_name, label_list_en[DATA_MC_INDEX_SEPARATION+1,:] )
                            plot_1d_compare_3_domains_beam_energy_graphs(true_en_list, [data_first_index_layer_second_index_energy_array, mc_first_index_layer_second_index_energy_array], label_list_en, feat_name, xlabel="True Energy [GeV]", ylabel=feat_name , title=feat_name+" vs True Energy", outdir=common_dir+"MC_DATA_DOMAIN_COMP", show_ideal=False,skip_some=skip_somelist,color_list=color_list)


def plot_for_everything(list1,name, IS_MC_DATA_TITLE, feature_name,outdirname,label_list_en):
    x_min, x_max=0,13000
    y_min, y_max=0,60000
    bins_x, bins_y = 40,40
    print("PLOTTİNG FOR EVEVERYTHING")

    N=50000
    total_list1 = [tensor.sum(dim=(1)) for tensor in list1]


    xlabel=feature_name
    name= xlabel+"_"+name
    plot_multiple_hist(total_list1, None , None, None, xlabel,
                    f"{feature_name} Distribution (All SciFi Stations)"+IS_MC_DATA_TITLE, label_list_en, outdir=outdirname,alpha_list=[1,1,1,1,1,1], name="hist_total_scifi_"+name)

    beam_xlabel = "Beam Energy [GeV]"
    
    # 1. SciFi (total_list1)
    plot_1d_beam_energy_graphs(beam_en_list, total_list1, label_list_en, 
                               "mean_total_scifi_" + name, show_ideal=False, 
                               xlabel=beam_xlabel, ylabel=f'Mean {feature_name}', 
                               title=f"Mean {feature_name} vs. Beam Energy (All SciFi Stations)" + IS_MC_DATA_TITLE, 
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
        
        plot_1d_beam_energy_graphs(beam_en_list, total_scifi_qdc_energy_wtr_beam_energy , label_list_en , str(i+1)+name, show_ideal=False, xlabel="Beam Energy [GeV]",ylabel='SciFi QDC Energy[GeV]',title=f"SciFi QDC Energy vs. Beam Energy"+IS_MC_DATA_TITLE, outdir=outdirname)
        
        return first_index_layer_second_index_energy_array, total_scifi_qdc_energy_wtr_beam_energy
    else:
        return first_index_layer_second_index_energy_array


def LOAD_PLOT_ALL_2D_COMBINATIONS(outdir,file_list, N, layers=[0, 1, 2], planes=[0, 1],thresholds=[-3], time_window_max_mc=[1],time_window_min_mc=[-1], time_window_max_data=[0.5],time_window_min_data=[-0.5]):
    for k, fname in enumerate(file_list):
        # Load the file ONCE
        dataloader, _, _ = data_loader([[0, fname]], N, N, "cpu", is_train=False)
        scifi_sig, scifi_hittime_diff = next(iter(dataloader))
              
        label = "MC" if "MC" in fname else "Data"
        if "MC" in fname:
            time_window_min = time_window_min_mc
            time_window_max = time_window_max_mc
        else:
            time_window_min = time_window_min_data
            time_window_max = time_window_max_data
            print("runing calibration")
            scifi_sig = apply_scifi2y_calibration(scifi_sig, station=1, orientation=0)  
        
        # Loop through all combinations of plane, layer, and threshold
        for plane in planes:
            for layer in layers:
                for threshold in thresholds:
                    for i in range(len(time_window_max)):
                        hist_qdc = scifi_sig[:, plane, layer, :]
                        hist_time = scifi_hittime_diff[: , plane, layer, : ]

                        cut = (hist_qdc != -999) & (hist_qdc > threshold) & (hist_time > time_window_min[i]) & (hist_time < time_window_max[i])
                        
                        # Skip plotting if the cut leaves no hits
                        if not cut.any():
                            continue 
                            
                        filtered_qdc = hist_qdc[cut]
                        filtered_time_diff = hist_time[cut]

                        coords = cut.nonzero()
                        hist_sipm_indices = coords[:, -1]

                        timing_name = f"{time_window_max[i]}_{time_window_min[i]}"
                        
                        plot_2d_hist(
                            hist_sipm_indices, 
                            filtered_qdc,
                            bins_x=None, 
                            bins_y=None,
                            out_name=f"qdc_vs_channel_L{layer}_P{plane}_Thresh{threshold}_{timing_name}_{label}", 
                            xlabel="Channel",
                            ylabel="QDC", 
                            title=f"QDC vs Channel (TB {label} | L{layer} O{plane})",
                            outdir=outdir
                        )
                        plot_2d_hist(
                            filtered_time_diff, 
                            filtered_qdc,
                            bins_x=None, 
                            bins_y=None,
                            out_name=f"qdc_vs_time_L{layer}_P{plane}_Thresh{threshold}_{timing_name}_{label}", 
                            xlabel="Hit Time Diff from the Highest Bin[Clock Cyc.]",
                            ylabel="QDC", 
                            title=f"QDC vs Hit Time Diff (TB {label} | L{layer} O{plane})",
                            outdir=outdir
                        )
                        plot_2d_hist(
                            hist_sipm_indices, 
                            filtered_time_diff,
                            bins_x=None, 
                            bins_y=None,
                            out_name=f"time_vs_channel_L{layer}_P{plane}_Thresh{threshold}_{timing_name}_{label}", 
                            xlabel="Channel",
                            ylabel="Hit Time Diff from the Highest Bin[Clock Cyc.]", 
                            title=f"Hit Time Diff. vs Channel (TB {label} | L{layer} O{plane})",
                            outdir=outdir
                        )





def get_event_stats(scifi_sig, scifi_hittime_diff, plane, layer, threshold, t_min, t_max):
    hist_qdc = scifi_sig[:, plane, layer, :]
    hist_time = scifi_hittime_diff[:, plane, layer, :]

    cut = (hist_qdc != -999) & (hist_qdc > threshold) & (hist_time > t_min) & (hist_time < t_max)
    
    if not cut.any():
        return None, None

    hit_numbers = torch.sum(cut, dim=1)

    clean_qdc = torch.where(cut, hist_qdc, torch.tensor(0.0, device=hist_qdc.device))
    total_qdc_per_event = torch.sum(clean_qdc, dim=1)

    return hit_numbers, total_qdc_per_event

def LOAD_PLOT_TUNING(outdir,file_list, N, layers=[0,1,2], planes=[0,1], thresholds=[-5,-3,-1,0,2], 
                       t_max_mc=[2.2], t_min_mc=[-0.5], 
                       t_max_data=[2.2], t_min_data=[-0.5]):
    
    # Load Data (Index 0)
    dl_data, _, _ = data_loader([[0, file_list[0]]], N, N, "cpu", is_train=False)
    sig_data, time_data = next(iter(dl_data))
    sig_data = apply_scifi2y_calibration(scifi_sig, station=1, orientation=0)  

    # Load MC (Index 1)
    dl_mc, _, _ = data_loader([[0, file_list[1]]], N, N, "cpu", is_train=False)
    sig_mc, time_mc = next(iter(dl_mc))
    
    for plane in planes:
        for layer in layers:
            for threshold in thresholds:
                for i in range(len(t_max_mc)):
                    
                    hits_d, qdc_d = get_event_stats(sig_data, time_data, plane, layer, 
                                                    threshold, t_min_data[i], t_max_data[i])
                    
                    hits_m, qdc_m = get_event_stats(sig_mc, time_mc, plane, layer, 
                                                    threshold, t_min_mc[i], t_max_mc[i])
                    
                    if hits_d is None or hits_m is None:
                        continue

                    timing_name = f"Tdata_{t_min_data[i]}_to_{t_max_data[i]}_andTMC_{t_min_mc[i]}_to_{t_max_mc[i]}"
                    base_name = f"L{layer}_O{plane}_Th{threshold}_{timing_name}"
                    base_id=f"L{layer} O{plane}"

                    plot_multiple_hist(
                        [hits_d, hits_m], None, None, None, 
                        x_label="Number of Hits per Event",
                        title=f"Hit Multiplicity ({base_id})", 
                        label_str=config.label_list_en, 
                        outdir="1d_hists", 
                        alpha_list=[0.7, 0.7], 
                        name="hits_" + base_name
                    )
                    plot_multiple_hist(
                        [qdc_d, qdc_m], None, None, None, 
                        x_label="Total QDC per Event",
                        title=f"Total QDC Distribution ({base_id})", 
                        label_str=config.label_list_en, 
                        outdir=outdir, 
                        alpha_list=[0.7, 0.7], 
                        name="qdc_sum_" + base_name
                    )

def LOAD_PLOT_ALL_COMP(outdir,file_list, N, layers=[0,1,2], planes=[0,1], thresholds=[-5,-3,-1,0,2], 
                       t_max_mc=1, t_min_mc=-1, 
                       t_max_data=2.2, t_min_data=-0.5):
    
    all_sigs = []
    all_times = []
    
    # 1. Load all datasets into memory
    for path in file_list:
        dl, _, _ = data_loader([[0, path]], N, N, "cpu", is_train=False)
        sig, time = next(iter(dl))
        all_sigs.append(sig)
        all_times.append(time)

    for plane in planes:
        for layer in layers:
            for threshold in thresholds:
                
                hits_to_compare = []
                qdc_sum_to_compare = []
                individual_qdc_to_compare = []
                
                # 2. Process each loaded dataset for this config
                for k in range(len(all_sigs)):
                    # Grab the correct signal/time for this dataset
                    current_sig = all_sigs[k]
                    current_time = all_times[k]
                    
                    # Logic for MC vs Data windows
                    if "MC" in file_list[k]:
                        t_min, t_max = t_min_mc, t_max_mc
                    else:
                        t_min, t_max = t_min_data, t_max_data
                        current_sig = apply_scifi2y_calibration(scifi_sig, station=1, orientation=0)  

                    # Get event-level stats (Multiplicity and Total Sum)
                    h, q_sum = get_event_stats(current_sig, current_time, plane, layer, 
                                               threshold, t_min, t_max)
                    hits_to_compare.append(h)
                    qdc_sum_to_compare.append(q_sum)

                    # 3. Individual Hit QDC Distribution (The fix is here)
                    # We slice the specific dataset 'current_sig'
                    hist_qdc = current_sig[:, plane, layer, :]
                    hist_time = current_time[:, plane, layer, :]
                    
                    # Apply the mask
                    cut = (hist_qdc != -999) & (hist_qdc > threshold) & \
                          (hist_time > t_min) & (hist_time < t_max)
                    
                    individual_qdc_to_compare.append(hist_qdc[cut])

                # Formatting strings
                timing_name = f"D_{t_min_data}_{t_max_data}_M_{t_min_mc}_{t_max_mc}"
                base_name = f"L{layer}_O{plane}_Th{threshold}_{timing_name}"
                base_id = f"L{layer} O{plane}"

                # 4. Generate the three comparison plots
                plot_configs = [
                    (individual_qdc_to_compare, "QDC of Each Hit", "single_qdc_"),
                    (hits_to_compare, "Number of Hits per Event", "hits_"),
                    (qdc_sum_to_compare, "Total QDC per Event", "qdc_sum_")
                ]

                for data_to_plot, x_label, prefix in plot_configs:
                    plot_multiple_hist(
                        data_to_plot, None, None, None, 
                        x_label=x_label,
                        title=f"{x_label} ({base_id})", 
                        label_str=config.label_list_en, 
                        outdir=outdir, 
                        alpha_list=[0.6] * len(file_list), 
                        name=prefix + base_name
                    )

def LOAD_PLOT_ALL_COMP_SINGLE_QDC(outdir, file_list, N, layers=[0,1,2], planes=[0,1], thresholds=[-5,-3,-1,0,2], 
                       t_max_mc=1, t_min_mc=-1, 
                       t_max_data=2.2, t_min_data=-0.5):
    
    all_sigs = []
    all_times = []
    
    # 1. Load all datasets into memory
    for path in file_list:
        dl, _, _ = data_loader([[0, path]], N, N, "cpu", is_train=False)
        sig, time = next(iter(dl))
        all_sigs.append(sig)
        all_times.append(time)
        if "MC" not in path:
            print(f"Applying calibration to Data: {path}")
            sig = apply_scifi2y_calibration(sig, station=1, orientation=0)

    for plane in planes:
        for layer in layers:
            for threshold in thresholds:

                individual_qdc_to_compare = []
                
                # 2. Process each loaded dataset for this config
                for k in range(len(all_sigs)):
                    # Grab the correct signal/time for this dataset
                    current_sig = all_sigs[k]
                    current_time = all_times[k]
                    
                    # Logic for MC vs Data windows
                    if "MC" in file_list[k]:
                        t_min, t_max = t_min_mc, t_max_mc
                    else:
                        t_min, t_max = t_min_data, t_max_data
                        #print("applied calibration")
                        #current_sig = apply_scifi2y_calibration(current_sig, station=1, orientation=0)  

 
                    hist_qdc = current_sig[:, plane, layer, :]
                    hist_time = current_time[:, plane, layer, :]
                    
                    # Apply the mask
                    cut = (hist_qdc != -999) & (hist_qdc > threshold) & \
                          (hist_time > t_min) & (hist_time < t_max)
                    
                    individual_qdc_to_compare.append(hist_qdc[cut])

                # Formatting strings
                timing_name = f"D_{t_min_data}_{t_max_data}_M_{t_min_mc}_{t_max_mc}"
                base_name = f"L{layer}_O{plane}_Th{threshold}_{timing_name}"
                base_id = f"L{layer} O{plane}"

                # 4. Generate the three comparison plots
                plot_configs = [
                    (individual_qdc_to_compare, "QDC of Each Hit", "single_qdc_"),
                ]

                for data_to_plot, x_label, prefix in plot_configs:
                    plot_multiple_hist(
                        data_to_plot, None, None, None, 
                        x_label=x_label,
                        title=f"{x_label} ({base_id})", 
                        label_str=config.label_list_en, 
                        outdir=outdir, 
                        alpha_list=[0.6] * len(file_list), 
                        name=prefix + base_name
                    )


def LOAD_PLOT_ALL_COMP_SINGLE_QDC_PLANE_COMP(outdir, file_list, N, layers=[0,1,2], planes=[0,1], thresholds=[-5,-3,-1,0,2], 
                       t_max_mc=1, t_min_mc=-1, 
                       t_max_data=2.2, t_min_data=-0.5):
    
    all_sigs = []
    all_times = []
    
    # 1. Load all datasets into memory
    for path in file_list:
        dl, _, _ = data_loader([[0, path]], N, N, "cpu", is_train=False)
        sig, time = next(iter(dl))
        all_sigs.append(sig)
        all_times.append(time)
        if "MC" not in path:
            print(f"Applying calibration to Data: {path}")
            sig = apply_scifi2y_calibration(sig, station=1, orientation=0)
    
    for layer in layers:
        individual_qdc_to_compare = []
        for plane in planes:
            for threshold in thresholds:
                # 2. Process each loaded dataset for this config
                for k in range(len(all_sigs)):
                    # Grab the correct signal/time for this dataset
                    current_sig = all_sigs[k]
                    current_time = all_times[k]
                    
                    # Logic for MC vs Data windows
                    if "MC" in file_list[k]:
                        t_min, t_max = t_min_mc, t_max_mc
                    else:
                        t_min, t_max = t_min_data, t_max_data
                        #print("applied calibration")
                        #current_sig = apply_scifi2y_calibration(current_sig, station=1, orientation=0)  
                    hist_qdc = current_sig[:, plane, layer, :]
                    hist_time = current_time[:, plane, layer, :]
                    
                    # Apply the mask
                    cut = (hist_qdc != -999) & (hist_qdc > threshold) & \
                          (hist_time > t_min) & (hist_time < t_max)
                    
                    individual_qdc_to_compare.append(hist_qdc[cut])
                    print(len(individual_qdc_to_compare))

                # Formatting strings
            timing_name = f"D_{t_min_data}_{t_max_data}_M_{t_min_mc}_{t_max_mc}"
            base_name = f"L{layer}_O{plane}_Th{threshold}_{timing_name}"
            base_id = f"L{layer} O{plane}"

            # 4. Generate the three comparison plots
            plot_configs = [
                (individual_qdc_to_compare, "QDC of Each Hit", "single_qdc_"),
            ]

            for data_to_plot, x_label, prefix in plot_configs:
                plot_multiple_hist(
                    data_to_plot, None, None, None, 
                    x_label=x_label,
                    title=f"{x_label} ({base_id})", 
                    label_str=config.label_list_en, 
                    outdir=outdir, 
                    alpha_list=[0.6] * len(data_to_plot), 
                    name=prefix + base_name
                )



def load_as_lists(file_list,N):
    scifi_list = []
    us_list = []
    ds_hor_list = []
    ds_ver_list = []
    #torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    for k,fname in enumerate(file_list):
        # apply every cut in here, hit number,  in time hits etc.
        dataloader, _, _= data_loader([[0, fname]], N, N, "cpu", is_train=False)#istrain makes shuffle true or false
        scifi, scifi_hit_time_diff  = next(iter(dataloader))
        # don't put into batches. return dataloader first. then change the cuts and call batch which apply cut. do this with for loop and fit with gaussian.

        print(torch.max(scifi))

        if PLOT_MEAN_QDC_ENERGY:
            scifi_prop = torch.sum(scifi,dim=(1,3))
            #us_prop = torch.sum(us,dim=(1,3))
            #ds_prop = torch.sum(ds,dim=(1,3))
        
        elif PLOT_MEAN_HIT_NUMBERS:
            scifi_prop = torch.sum(scifi!=-999,dim=(1,3))
            #us_prop = torch.sum(us!=-999,dim=(1,3))
            #ds_prop = torch.sum(ds!=-999,dim=(1,3))
        #print(scifi_prop.shape, us_prop.shape, ds_prop.shape)

        scifi_list.append(scifi_prop)
        #us_list.append(us_prop)
        #ds_hor_list.append(ds_prop)

    return scifi_list#, us_list, ds_hor_list,ds_ver_list

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
