import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import os
import numpy
i=0
#TEST_DATA_DIR = [f"/eos/user/b/beturk/snd/test_beam/2024/scifi_ds_2024_electrons_300GeV_run_100907_{i}.pt"]
TEST_DATA_DIR = [f"/eos/user/b/beturk/snd/test_beam/2023/scifi_us_ds_2023_pions_140GeV_3wall_run_100673_{i}.pt"]
data = torch.load(TEST_DATA_DIR[0])
# Clamp
scifi_pos = torch.clamp_min(data["scifi_signals"], 0)
us_pos    = torch.clamp_min(data["us_signals"], 0)
ds_h_pos  = torch.clamp_min(data["ds_horizontal"], 0)
ds_v_pos  = torch.clamp_min(data["ds_vertical"], 0)

# Enerjiler
scifi_E = scifi_pos.sum((1,2,3)) * 0.059
us_E    = us_pos.sum((1,2,3))    * 0.0145
ds_h_E  = ds_h_pos.sum((1,2,3))  * 0.0145
ds_v_E  = ds_v_pos.sum((1,2))  * 0.0145

# Toplam enerji
total_E = scifi_E + us_E + ds_h_E + ds_v_E

# Plot
plt.figure(figsize=(8,6))

plt.hist(scifi_E.numpy(), bins=50, range=(0,400),
         histtype="step", linewidth=2, label="SciFi")

plt.hist(us_E.numpy(), bins=50, range=(0,400),
         histtype="step", linewidth=2, label="US")

plt.hist(ds_h_E.numpy(), bins=50, range=(0,400),
         histtype="step", linewidth=2, label="DS Horizontal")

plt.hist(ds_v_E.numpy(), bins=50, range=(0,400),
         histtype="step", linewidth=2, label="DS Vertical")

plt.hist(total_E.numpy(), bins=50, range=(0,400),
         histtype="stepfilled", alpha=0.25, label="Total Energy")

plt.xlabel("Energy [arb. units]")
plt.ylabel("Events")
plt.legend()
plt.tight_layout()

plt.savefig("energy_histograms.png")
plt.close()
