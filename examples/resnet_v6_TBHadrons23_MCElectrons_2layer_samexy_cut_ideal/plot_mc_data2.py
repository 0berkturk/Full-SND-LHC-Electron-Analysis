from dl_recon_core.mc_data_comp import *
import torch

name="/eos/user/b/beturk/snd/MonteCarlo/data_no_beam_cuts/data_2022.pt"
data=torch.load(name)
for key in data:
    print(key)
run_name_list = data["run_id"]
print(run_name_list)

event_number_list=data["event_number"]
print(event_number_list)

event=136401256
run=4724

index =  (run_name_list==run) 

print(event_number_list[index])




