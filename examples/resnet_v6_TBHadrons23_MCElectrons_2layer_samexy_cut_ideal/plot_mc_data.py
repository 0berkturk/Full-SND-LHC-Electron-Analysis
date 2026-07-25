from dl_recon_core.mc_data_comp import *
import torch

def calculate_energy(name, scifi_data,threshold):
    data = torch.load(name)
    probs=data["new_model"]
    
    scifi_data = torch.load(scifi_data)["scifi_signals"]

    index=probs>threshold

    qdc_energy=scifi_data[index].sum((1,2,3))*0.059

    qdc_energy_v2 = data["en3d"][index]

    print(qdc_energy)
    print(qdc_energy_v2)

    return qdc_energy, probs[index]



qdc_energy,_ = calculate_energy("/eos/user/b/beturk/snd/dl_general/dl_cls_v4/resnet_v6_data_el_bce/probs_test_data_2022_model_ecal_20251210-0909_epoch-9.pt", 
"/eos/user/b/beturk/snd/MonteCarlo/data_no_beam_cuts/test_data_2022.pt", 2)


key="new_model"
key2="en3d"

combined_my_hadrons, energy_my_hadrons = load_and_combine(
    ["/eos/user/b/beturk/snd/dl_general/dl_cls_v4/resnet_v6_data_el_bce/probs_onlyscifi__combined_kshort_0_100_2025_model_ecal_20251210-0909_epoch-5.pt"], key,key2)

combined_data_2022, energy_data_2022 = load_and_combine(
    ["/eos/user/b/beturk/snd/dl_general/dl_cls_v4/resnet_v6_data_el_bce/probs_test_data_2022_model_ecal_20251210-0909_epoch-9.pt"], key,key2)

combined_data_2022_all, energy_data_2022_all = load_and_combine(
    ["/eos/user/b/beturk/snd/dl_general/dl_cls_v4/resnet_v6_data_el_bce/probs_test_data_2022_model_ecal_20251210-0909_epoch-9.pt",
    "/eos/user/b/beturk/snd/dl_general/dl_cls_v4/resnet_v6_data_el_bce/probs_train_data_2022_model_ecal_20251210-0909_epoch-9.pt",
    "/eos/user/b/beturk/snd/dl_general/dl_cls_v4/resnet_v6_data_el_bce/probs_val_data_2022_model_ecal_20251210-0909_epoch-9.pt"], key,key2)


combined_data_all, energy_data_all = load_and_combine(
    ["/eos/user/b/beturk/snd/dl_general/dl_cls_v4/resnet_v6_data_el_bce/probs_test_data_2022_model_ecal_20251210-0909_epoch-9.pt",
    "/eos/user/b/beturk/snd/dl_general/dl_cls_v4/resnet_v6_data_el_bce/probs_test_data_2023_model_ecal_20251210-0909_epoch-9.pt",
    "/eos/user/b/beturk/snd/dl_general/dl_cls_v4/resnet_v6_data_el_bce/probs_test_data_2024_model_ecal_20251210-0909_epoch-9.pt",
    "/eos/user/b/beturk/snd/dl_general/dl_cls_v4/resnet_v6_data_el_bce/probs_test_data_2025_model_ecal_20251210-0909_epoch-9.pt"], key,key2)

combined_mc_electrons, energy_mc_electrons = load_and_combine(
    [
    "/eos/user/b/beturk/snd/dl_general/dl_cls_v4/resnet_v6_data_el_bce/probs_test_combined_electron_0_50_2025_model_ecal_20251210-0909_epoch-9.pt"], key,key2)

combined_tb_electrons,energy_tb_electrons = load_and_combine(
    [
    "/eos/user/b/beturk/snd/dl_general/dl_cls_v4/resnet_v6_data_el_bce/probs_selected_electrons_all_50gev_100933_0_model_ecal_20251210-0909_epoch-9.pt"], key,key2)


labels=[ "MC Electrons(<50 GeV)","Data(2022 %20)", "TB Electron(50 GeV)"]

plot_all_histograms([combined_data_2022_all,combined_tb_electrons], ["Data(2022 All)", "TB Electron(50 GeV)"], out_dir="/eos/user/b/beturk/snd/dl_general/dl_cls_v4/resnet_v6_data_el_bce/2022data_all_tb_electrons" ,x_axis="ResNet's Logits",y_axis="Count",title="ResNet's Logits Histogram")
plot_all_histograms([combined_mc_electrons,combined_data_2022_all,combined_tb_electrons], [ "MC Electrons(<50 GeV)","Data(2022 %20)", "TB Electron(50 GeV)"], out_dir="/eos/user/b/beturk/snd/dl_general/dl_cls_v4/resnet_v6_data_el_bce/general2022data_all_tb_mc_electrons" ,x_axis="ResNet's Logits",y_axis="Count",title="ResNet's Logits Histogram")



plot_all_histograms([combined_data_2022,combined_tb_electrons], ["Data(2022 %20)", "TB Electron(50 GeV)"], out_dir="/eos/user/b/beturk/snd/dl_general/dl_cls_v4/resnet_v6_data_el_bce/2022data_tb_electrons" ,x_axis="ResNet's Logits",y_axis="Count",title="ResNet's Logits Histogram")
plot_all_histograms([combined_data_all,combined_tb_electrons], [ "Data(All Years %20)", "TB Electron(50 GeV)"], out_dir="/eos/user/b/beturk/snd/dl_general/dl_cls_v4/resnet_v6_data_el_bce/data_all_years_tb_electrons" ,x_axis="ResNet's Logits",y_axis="Count",title="ResNet's Logits Histogram")

plot_all_histograms([combined_my_hadrons,combined_data_2022,combined_tb_electrons], [ "MC combined_my_hadrons(<100 GeV)","Data(2022 %20)", "TB Electron(50 GeV)"], out_dir="/eos/user/b/beturk/snd/dl_general/dl_cls_v4/resnet_v6_data_el_bce/compare_myhadronsepch5" ,x_axis="ResNet's Logits",y_axis="Count",title="ResNet's Logits Histogram")

plot_all_histograms([combined_mc_electrons,combined_data_2022,combined_tb_electrons], [ "MC Electrons(<50 GeV)","Data(2022 %20)", "TB Electron(50 GeV)"], out_dir="/eos/user/b/beturk/snd/dl_general/dl_cls_v4/resnet_v6_data_el_bce/compare" ,x_axis="ResNet's Logits",y_axis="Count",title="ResNet's Logits Histogram")




import matplotlib.pyplot as plt

index = combined_data_2022_all>1
plt.hist(energy_data_2022_all[index], bins=np.arange(0, 20, 0.5))
plt.yscale("log")
plt.title("Energy Histogram of Data 2022 Electrons")
plt.xlabel("Energy[GeV]")
plt.ylabel("Counts")
plt.savefig("/afs/cern.ch/work/b/beturk/private/snd/dl_general/dl_cls_v4/resnet_v6_data_el_bce/img.png")
