import torch

file_list=["/eos/experiment/sndlhc/users/beturk/TB/TB_Data/2024/scifi_us_ds_2024_pions_180GeV_W_run_100947_10.pt"]
new_data={}
for file in file_list:
    data= torch.load(file,weights_only=False)
    for key in data:
        new_data[key]=data[key][:5000]
        print(key)

torch.save(new_data,"/eos/experiment/sndlhc/users/beturk/TB/TB_Small/small_scifi_us_ds_2024_pions_180GeV_W_run_100947_10.pt")