import torch
import numpy as np
import matplotlib.pyplot as plt

def load_and_combine(file_list, key, key2):
    data_list = []

    data_list2=[]

    for f in file_list:
        print("Loading:", f)

        data = torch.load(f)
        for key in data:
            print(key)
        
        data_list.append(data[key])
        data_list2.append(data[key2])

    return torch.cat(data_list, dim=0), torch.cat(data_list2, dim=0)
 

def plot_all_histograms(logits, labels, out_dir ,x_axis="Energy Resolution",y_axis="Generated Momentum",title="Energy Resolution vs Generated Momentum"):
    bins = [np.linspace(-25,20,50) for _ in range(len(logits))] 
    bins2 = [np.linspace(0,1,50) for _ in range(len(logits))]
    soft = torch.nn.Sigmoid()
    plt.figure()
    for i in range(len(logits)):
        plt.hist(logits[i],bins[i],label=labels[i],alpha=0.5)
        mask = logits[i] > 1
        n_pass = mask.sum().item()
        n_total = logits[i].numel()

        eff = n_pass / n_total

        print(f"Eff and passing number of {labels[i]} are", n_pass, eff)

    plt.title(title)
    plt.ylabel(y_axis)
    plt.xlabel(x_axis)
    plt.grid(True)
    plt.legend()
    plt.yscale("log")
    plt.savefig(out_dir+".png",dpi=300)
    plt.close()

    plt.figure()
    for i in range(len(logits)):
        plt.hist(soft(logits[i]),bins2[i],label=labels[i],alpha=0.5)
    plt.title(title)
    plt.ylabel(y_axis)
    plt.xlabel(x_axis)
    plt.grid(True)
    plt.legend()
    plt.yscale("log")
    plt.savefig(out_dir+"norm.png",dpi=300)
    plt.close()

    






