import torch
import numpy as np
import os.path
from torch.utils.data import DataLoader, TensorDataset
import config
from dl_recon_core_sparse.aploss import *
import pandas as pd
import torch.nn as nn

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        print(torch.version.cuda)
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def train(feature_extractor, classifier, train_loader,lossfun_train,optimizer_train,device,dir):
    print("training function")

    loss_data_sum = 0

    for k, data in enumerate(train_loader):
        if config.USE_ONLY_SCIFI:
            scifi, energy, labels = data
            scifi, energy, labels = scifi.to(device), energy.to(device), labels.to(device)
            x = (scifi)
        elif config.USE_SCIFI_US:
            scifi, us, energy, labels = data
            scifi, us, energy, labels = scifi.to(device),us.to(device), energy.to(device), labels.to(device)
            x = (scifi,us)
        elif config.USE_SCIFI_US_DS:
            scifi, us, ds, energy, labels = data
            scifi, us,ds, energy, labels = scifi.to(device),us.to(device),ds.to(device) ,energy.to(device), labels.to(device)
            x = (scifi,us,ds)
            """        elif config.USE_TB_24:
            scifi, ds, energy, labels = data
            scifi ,ds, energy, labels = scifi.to(device),ds.to(device) ,energy.to(device), labels.to(device)
            x = (scifi,ds)"""

        optimizer_train.zero_grad()
        if config.IS_SINGLE_NETWORK:
            output = feature_extractor(x)
        else:
            output = classifier(feature_extractor(x))
        #print(k,labels,output)

        if config.IS_BINARY:
            labels=labels.float()

        if getattr(config, "IS_LOG_NORM_OUTPUT", False):
            energy = torch.log(torch.clamp_min(energy, 1))
            response = output
            target = energy
            # assume 

        elif config.train_loss_response:
            response = output / energy   # ratio
            target = torch.ones_like(response).to(device)  # want response ~ 1

        elif config.train_loss_rel_bias:
            rel_bias = torch.abs(output-energy)/energy
            target = torch.zeros_like(rel_bias).to(device)

        elif config.train_loss_rel_bias_square:
            rel_bias = torch.abs(output-energy)/energy**2
            target = torch.zeros_like(rel_bias).to(device)

        elif config.TRAIN_MY_LOSS_FUNC_V1:
            response = output
            target = energy

        else:
            response = output
            target = energy
        
        loss = lossfun_train(response, target)
        loss.backward()
        optimizer_train.step()
        loss_data_sum = loss_data_sum + loss
        #print(energy,torch.sum(scifi,(1,2,3)), torch.sum(us,(1,2,3)), torch.sum(ds,(1,2,3) ))
    #print(k, "output", output)
    loss_mean = loss_data_sum/k
    print("Loss of Training Data = ", loss_mean)
    # save loss
    c = (loss_mean).detach().cpu().numpy()
    if os.path.isfile(f"{dir}/results/training_loss.npy"):
        y=np.load(f"{dir}/results/training_loss.npy")
    else:
        y=[]
    np.save(f"{dir}/results/training_loss.npy", np.append(y, c))

def validation(val_dataset, feature_extractor, classifier, lossfun_val, device, dir):
    print("calculate_validation function")
    loss_sum_val = 0
    for k, data in enumerate(val_dataset):
        if config.USE_ONLY_SCIFI:
            scifi, energy, labels = data
            scifi, energy, labels = scifi.to(device), energy.to(device), labels.to(device)
            x = (scifi)
        elif config.USE_SCIFI_US:
            scifi, us, energy, labels = data
            scifi, us, energy, labels = scifi.to(device),us.to(device), energy.to(device), labels.to(device)
            x = (scifi,us)
        elif config.USE_SCIFI_US_DS:
            scifi, us, ds, energy, labels = data
            scifi, us,ds, energy, labels = scifi.to(device),us.to(device),ds.to(device) ,energy.to(device), labels.to(device)
            x = (scifi,us,ds)
        """elif config.USE_TB_24:
            scifi, ds, energy, labels = data
            scifi ,ds, energy, labels = scifi.to(device),ds.to(device) ,energy.to(device), labels.to(device)
            x = (scifi,ds)"""

        with torch.no_grad():
            if config.IS_SINGLE_NETWORK:
                output = feature_extractor(x)
            else:
                output = classifier(feature_extractor(x))
                    # assume 

            if getattr(config, "IS_LOG_NORM_OUTPUT", False):
                energy = torch.log(torch.clamp_min(energy, 1))
                response = output
                target = energy

            elif config.val_loss_response:
                response = output / energy   # ratio
                target = torch.ones_like(response).to(device)  # want response ~ 1
            elif config.val_loss_rel_bias:
                rel_bias = (output-energy)/energy
                target = torch.zeros_like(rel_bias).to(device)
            
            elif config.val_loss_rel_bias_square:
                rel_bias = torch.abs(output-energy)/energy**2
                target = torch.zeros_like(rel_bias).to(device)

            elif config.VAL_MY_LOSS_FUNC_V1:
                response = output
                target = energy

            else:
                response = output
                target = energy
            
            loss = lossfun_val(response, target)

            loss_sum_val+=loss

    mean_loss = loss_sum_val/k
    c = (mean_loss).detach().cpu().numpy()

    y = np.load(f"{dir}/results/v_mean_loss_val.npy") if os.path.isfile(f"{dir}/results/v_mean_loss_val.npy") else []
    np.save(f"{dir}/results/v_mean_loss_val.npy", np.append(y, c))

#y = np.load(f"{dir}/results/v_acc.npy") if os.path.isfile(f"{dir}/results/v_acc.npy") else []
   # np.save(f"{dir}/results/v_acc.npy", np.append(y, accuracy))
    return mean_loss#, accuracy


def training_discriminator(feature_extractor,discriminator,label0_train_domain_adapt_dataset,label1_train_domain_adapt_dataset,lossfun_discriminator,optimizer_discriminator,optimizer_model,device,BATCH_SIZE=128):
    print("training Discriminator")
    feature_extractor.eval()
    loss_data_sum = 0
    ##label0 is target
    ## since target data is smaller and I want to use all of it, don't combine them before this.
    ## this method uses all target data.

    ## shuffle data in here. Otherwise, It wont use all of the all data during different epoch.
    label1_train_domain_adapt_dataset = DataLoader(label1_train_domain_adapt_dataset, batch_size=BATCH_SIZE,shuffle=True)

    data_zip = enumerate(zip(label0_train_domain_adapt_dataset, label1_train_domain_adapt_dataset))   ### it stops when small one finishes.
    for k, ((x_0, y_0), (x_1, y_1)) in data_zip:
        x_0, y_0 = x_0.to(device), y_0.to(device)
        x_1, y_1 = x_1.to(device), y_1.to(device)

        x = torch.cat([x_0, x_1])
        y = torch.cat((y_0, y_1))

        optimizer_discriminator.zero_grad()
        out=feature_extractor(x)
        output = discriminator(out)
        loss = lossfun_discriminator(output.float(), y.float())
        loss.backward(retain_graph=True)
        optimizer_discriminator.step()

        y = torch.cat((y_1, y_0)) ## maximize
        optimizer_model.zero_grad()
        output = discriminator(out)
        loss = lossfun_discriminator(output.float(), y.float())
        loss.backward()
        optimizer_model.step()

        loss_data_sum = loss_data_sum + loss

    loss_mean = loss_data_sum / k
    print("Loss of Training Data = ", loss_mean)
    # save loss
    c = (loss_mean).detach().cpu().numpy()
    if os.path.isfile("results/domain_training_loss.npy"):
        y=np.load("results/domain_training_loss.npy")
    else:
        y=[]
    np.save("results/domain_training_loss.npy", np.append(y, c))


def test(test_dataset,cvt_target,classifier,device):
    print("calculate_test function")
    real_label_list = []
    energy_list = []
    mom_list = []
    cvt_list =np.array([])
    bdt_list = []
    lhd_list = []
    for batch in test_dataset:
        with torch.no_grad():
            images,real_label,bdt,lhd,en3d = batch
            max_prob = classifier(cvt_target(images.to(device)))
            max_prob = torch.sigmoid(max_prob).cpu().detach().numpy()

        cvt_list=np.append(cvt_list, max_prob)
        real_label_list=np.append(real_label_list, real_label)
        bdt_list=np.append(bdt_list, bdt)
        lhd_list=np.append(lhd_list, lhd)
        energy_list=np.append(energy_list, en3d)

    np.savez("result_test/test_results",cvt=cvt_list,y=real_label_list,bdt=bdt_list,ecal_lhd=lhd_list,en3d=energy_list)
    print("test finished")
    return cvt_list




def save_all_cpu(val_loader, model):  ####wrong
    print("calculate_test function")
    real_label_list = []
    energy_list = []
    mom_list = []
    cvt_list =np.array([])
    bdt_list = []
    lhd_list = []
    for batch in val_loader:
        with torch.no_grad():
            images,real_label,bdt,lhd,en3d,mom = batch
            max_prob = model(images)
            max_prob = torch.sigmoid(max_prob)

        cvt_list=np.append(cvt_list, max_prob)

        real_label_list=np.append(real_label_list, real_label)
        bdt_list=np.append(bdt_list, bdt)
        lhd_list=np.append(lhd_list, lhd)
        energy_list=np.append(energy_list, en3d)
        mom_list=np.append(mom_list, mom)



    np.save("result_test/cvt_probs.npy",cvt_list)
    np.save("result_test/real_label.npy", real_label_list)
    np.save("result_test/bdt.npy", bdt_list)

    np.save("result_test/lhd.npy", lhd_list)
    np.save("result_test/en3d.npy", energy_list)
    np.save("result_test/mom.npy", mom_list)
    print("test finished")
