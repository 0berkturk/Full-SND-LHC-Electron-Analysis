import torch


def load_data(path1,en1,en2):
    data1 = torch.load(path1)
    print(data1["y"].shape,data1["y"])
    label = data1["y"]
    energy = data1["en3d"]
    cut = (energy<en2 ) & (energy>en1) #& (label==3)

    energy = energy[cut]
    total_particles = len(energy)
    print("total",total_particles)
    #scifi = data1["scifi_signals"][cut]
    us = data1["us_signals"][cut]
    ds_h = data1["ds_horizontal"][cut]
    ds_v = data1["ds_vertical"][cut]
    
    ds_h_padded = torch.nn.functional.pad(ds_h, (0, 0, 0, 1))
    ds = torch.stack([ds_h_padded, ds_v], dim=1) #2x4x60

    #scifi_sum = scifi_data.sum(dim=(1,2,3))
    us_sum = us.sum(dim=(1,2,3))
    ds_sum = ds.sum(dim=(1,2,3))

    cut_us = us_sum<0
    cut_ds = ds_sum<0
    n_ds_zero = cut_ds.sum()
    n_us_zero = cut_us.sum()
    print("ds negative sum number", n_ds_zero.item()," percentage " ,(n_ds_zero/total_particles).item())
    print("us negative sum number", n_us_zero.item()," percentage " ,(n_us_zero/total_particles).item(),"\n")


    cut_us = us_sum == 0
    cut_ds = ds_sum == 0
    n_ds_zero = cut_ds.sum()
    n_us_zero = cut_us.sum()
    print("ds zero sum number", n_ds_zero.item()," percentage " ,(n_ds_zero/total_particles).item())
    print("us zero sum  number", n_us_zero.item()," percentage " ,(n_us_zero/total_particles).item(),"\n")
    

    cut_us = us_sum>20
    cut_ds = ds_sum>20
    n_ds_zero = cut_ds.sum()
    n_us_zero = cut_us.sum()
    print("ds positive sum number", n_ds_zero.item()," percentage " ,(n_ds_zero/total_particles).item())
    print("us positive sum number", n_us_zero.item()," percentage " ,(n_us_zero/total_particles).item(),"\n")


    return n_ds_zero, n_us_zero




path2="/eos/user/b/beturk/snd/MonteCarlo/particle_gun_advanced_v1/merge_electron_gun_0_50/test_combined_electron_0_50_2025.pt"

path="/eos/user/b/beturk/snd/MonteCarlo/particle_gun_advanced_v1/merge_kshort_gun_0_100/test_combined_kshort_0_100_2025.pt"

#path="/eos/user/b/beturk/snd/MonteCarlo/particle_gun_advanced_v1/merge_muon_gun_0_10/test_combined_muon_0_10_2025.pt"

#path = "/eos/user/b/beturk/snd/MonteCarlo/particle_gun_advanced_v1/merge_muon_gun_0_1000/test_combined_muon_0_1000_2025.pt"

#path = "/eos/user/b/beturk/snd/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/test_sim_neutrino_2024_new.pt"
load_data(path2,0,50)



