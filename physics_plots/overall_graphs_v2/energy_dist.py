el_50gev="/eos/user/b/beturk/snd/MonteCarlo/particle_gun_advanced_v1/merge_electron_gun_0_50/test_combined_electron_0_50_2025.pt"
el_2tev="/eos/user/b/beturk/snd/MonteCarlo/particle_gun_advanced_v1/merge_electron_gun_0_2000/test_combined_electron_0_2000_2025_v2.pt"


import matplotlib.pyplot as plt
import torch
import numpy as numpy
print("passed")

file = torch.load(el_50gev)

print(file["en3d"].shape)

us_sum = file["us_signals"].sum(dim=(1, 2, 3))
ds_sum_v = file["ds_vertical"].sum(dim=(1, 2))
ds_sum_h = file["ds_horizontal"].sum(dim=(1, 2))
index = (ds_sum_v == 0) & (ds_sum_h==0) & (us_sum == 0)
print(file["en3d"][index].shape)


plt.hist(file["en3d"],100,label="Before US+DS Cuts")
plt.hist(file["en3d"][index],100,label="After US+DS Cuts")
plt.title("Generated Momentum Histogram of Electrons")
plt.xlabel("Generated Momentum[GeV]")
plt.ylabel("Counts")
plt.yscale("log")
plt.legend()
plt.savefig("el2tev.png",dpi=300)
