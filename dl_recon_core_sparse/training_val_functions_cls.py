import torch
import numpy as np
import os.path
from torch.utils.data import DataLoader, TensorDataset
import config
from dl_recon_core_sparse.aploss import *
import pandas as pd

def get_accuracy(logits, labels, multiple_2_binary=False,signal_index=0):
    """
    logits: tensor of shape (batch_size, num_classes) for multiclass
            or (batch_size,) / (batch_size, 1) for binary
    labels: tensor of shape (batch_size,) with class indices (0..C-1)
    is_binary: whether this is binary classification
    """
    if config.IS_BINARY:
        # if logits is (batch_size, 1), squeeze it
        if logits.dim() > 1 and logits.size(1) == 1:
            logits = logits.squeeze(1)

        # apply sigmoid and threshold
        preds = (torch.sigmoid(logits) > 0.5).long()
    elif multiple_2_binary:
        labels = (labels == signal_index).float()
        preds = torch.argmax(logits, dim=1)
        preds = (preds == 0)
        print("convert multiple class to electron signal/background binary, acc is;")

    else:
        # pick the class with highest logit
        preds = torch.argmax(logits, dim=1)

    correct = (preds == labels).sum().item()
    acc = correct / labels.size(0)
    print("acc is",acc)
    return acc


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

        optimizer_train.zero_grad()
        if config.IS_SINGLE_NETWORK:
            output = feature_extractor(x)
        else:
            output = classifier(feature_extractor(x))

        if config.IS_BINARY:
            labels=labels.float()
            output=output.reshape(-1)


        loss = lossfun_train(output, labels).mean()
        loss.backward()
        optimizer_train.step()
        loss_data_sum = loss_data_sum + loss
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

def calculate_misclassified_protons(logits,targets):
    index_pos_samples=targets==1
    pos_logits = logits[index_pos_samples]

    index_neg_samples=targets==0
    neg_logits =logits[index_neg_samples]
    total_misclassified_protons=0
    for i in pos_logits:
        difference=-i+neg_logits
        total_misclassified_protons += torch.sum(
            torch.heaviside(difference, torch.tensor([1], dtype=difference.dtype, device=difference.device))
        )

    return total_misclassified_protons

def find_prej_at_eff(thresholds,cvt_p_Rej,cvt_e_eff, eff):
    xy1 = pd.DataFrame({'x': thresholds, 'y': cvt_e_eff,'z':cvt_p_Rej})
    pd.set_option('display.precision', 32)

    xy1['abs_diff'] = abs(xy1['y'] - eff)
    sorted_xy = xy1.sort_values(by='abs_diff')
    closest_value = sorted_xy.iloc[0]

    e_eff = closest_value['y']

    sorted_xy = sorted_xy.loc[sorted_xy['y'] == e_eff].sort_values(by='x')
    closest_value = sorted_xy.iloc[-1]

    p_rej = closest_value['z']

    print("eff: ",e_eff, " \nprej:",p_rej)
    return int(p_rej)

def p_rejection_e_eff(prob, y_test_loader):
    list_p_rej=[]
    list_e_eff=[]
    thresholds=np.load(config.thresholds_file)
    for threshold in thresholds:
        index = y_test_loader == 1
        e_prob=prob[index]
        total_electron=len(e_prob)

        index = e_prob > threshold
        e_prob_true = e_prob[index]
        no_correct_electron = len(e_prob_true)

        index = y_test_loader == 0
        p_prob = prob[index]
        total_proton = len(p_prob)

        index = p_prob <= threshold
        p_prob_true = p_prob[index]
        no_correct_proton = len(p_prob_true)

        no_misidentified_proton = total_proton - no_correct_proton

        if (no_misidentified_proton == 0):
            no_misidentified_proton = 1

        electron_eff = no_correct_electron / total_electron
        proton_rej = total_proton / no_misidentified_proton

        list_p_rej = np.append(list_p_rej, proton_rej)
        list_e_eff = np.append(list_e_eff, electron_eff)
    loss=0
    for e_eff in config.e_eff_lists:
        loss-=find_prej_at_eff(thresholds,list_p_rej, list_e_eff, e_eff) ## I've added minus sign becasue I want to minimize it.
    return loss

def validation(val_dataset, feature_extractor, classifier, lossfun_val, device, dir):
    print("calculate_validation function")
    loss_sum_val = 0
    logits = torch.tensor([])
    targets = []  ## labels


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

        with torch.no_grad():
            if config.IS_SINGLE_NETWORK:
                output = feature_extractor(x)
            else:
                output = classifier(feature_extractor(x))

            if config.IS_BINARY:
                labels=labels.float()
                output=output.reshape(-1)

            loss = lossfun_val(output, labels).mean()
            loss_sum_val+=loss

        if logits.numel() == 0:
            logits = output  # Direct assignment for the first entry
            targets = labels
        else:
            logits = torch.cat((logits, output), dim=0)
            targets = torch.cat((targets, labels), dim=0)

    accuracy = get_accuracy(logits, targets)
    print("binary accuracy ", get_accuracy(logits, targets,True))


    if config.VAL_PREJ:
        print("USING REJECTION, UPDATE THIS LINE 137")
        logits = logits.reshape(-1)
        targets = targets.reshape(-1)

        n_misclass_protons = calculate_misclassified_protons(logits, targets)
        p_rej_loss = p_rejection_e_eff(logits, targets)

        #loss = lossfun_val(logits.float(), targets.float()).mean() no need this
        #loss_sum_val += loss
        print("N Misclassified Protons ", n_misclass_protons)
        print("p_rej_loss ", p_rej_loss)
    print('Percent correct: ', accuracy)

    if config.VAL_PREJ:
        mean_loss = p_rej_loss
        c = mean_loss

    else:
        mean_loss = loss_sum_val/k
        c = (mean_loss).detach().cpu().numpy()

    print(" ")
    print("Loss of Validation Data     = ", mean_loss)


    y = np.load(f"{dir}/results/v_mean_loss_val.npy") if os.path.isfile(f"{dir}/results/v_mean_loss_val.npy") else []
    np.save(f"{dir}/results/v_mean_loss_val.npy", np.append(y, c))

    y = np.load(f"{dir}/results/v_acc.npy") if os.path.isfile(f"{dir}/results/v_acc.npy") else []
    np.save(f"{dir}/results/v_acc.npy", np.append(y, accuracy))

    if config.VAL_PREJ:
        c=(n_misclass_protons).cpu().numpy()
        y = np.load(f"{dir}/results/v_n_miscl_pr.npy") if os.path.isfile(f"{dir}/results/v_n_miscl_pr.npy") else []
        np.save(f"{dir}/results/v_n_miscl_pr.npy", np.append(y, c))

        if config.focal_loss:
            c = (loss_sum_val / k).cpu().numpy()
            y = np.load(f"{dir}/results/v_fcloss.npy") if os.path.isfile(f"{dir}/results/v_fcloss.npy") else []
            np.save(f"{dir}/results/v_fcloss.npy", np.append(y, c))
        else:
            c = (loss_sum_val/k).cpu().numpy()
            y = np.load(f"{dir}/results/v_bceloss.npy") if os.path.isfile(f"{dir}/results/v_bceloss.npy") else []
            np.save(f"{dir}/results/v_bceloss.npy", np.append(y, c))
    print("accuracy 260:",accuracy)
    return mean_loss, accuracy


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

