import torch
import os
import glob

def save_first_2000_from_dict_pt(file_list, output_dir, n_samples=10000):
    os.makedirs(output_dir, exist_ok=True)

    for fname in file_list:
        print(fname)
        out_path = os.path.join(output_dir, os.path.basename(fname))
        print(out_path)
        data = torch.load(fname, map_location="cpu")  # dict

        out_data = {}
        for k, v in data.items():
            # only slice tensors that have sample dimension
            if torch.is_tensor(v) and v.size(0) >= n_samples:
                out_data[k] = v[:n_samples].clone()
            else:
                # keep as-is (metadata, scalars, etc.)
                out_data[k] = v
        
        torch.save(out_data, out_path)


file_list = glob.glob("/eos/experiment/sndlhc/users/beturk/TB/TB_Data/2023/*0.pt")


save_first_2000_from_dict_pt(
    file_list,
    output_dir="/eos/experiment/sndlhc/users/beturk/TB/TB_Data/2023/small"
)
