import torch

data=torch.load("/eos/experiment/sndlhc/users/beturk/TB/TB_MC_New/Sparse_Datasets_2024/MCEB_TB_MC_2024_electron_150GeV_0.pt",weights_only=False)
print(data["en3d"])
