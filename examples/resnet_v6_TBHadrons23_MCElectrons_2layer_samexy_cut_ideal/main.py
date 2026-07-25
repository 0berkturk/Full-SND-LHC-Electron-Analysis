
import config
from dl_recon_core_sparse.data_loader import *
from dl_recon_core_sparse.test_functions_cls import *
from dl_recon_core_sparse.training_val_functions_cls import *
from dl_recon_core_sparse.test_function_for_main import *

import torch
import time
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os.path
import torch.nn as nn
import sys
from matplotlib import pyplot as plt
import os
import glob

dir = sys.argv[1]
eos = sys.argv[2]

if not os.path.exists(f"{eos}"):
    # If it doesn't exist, create it
    os.makedirs(f"{eos}")

if not os.path.exists(f"{eos}/checkpoints"):
    # If it doesn't exist, create it
    os.makedirs(f"{eos}/checkpoints")

if not os.path.exists(f"{eos}/tests"):
    # If it doesn't exist, create it
    os.makedirs(f"{eos}/tests")
    
if not os.path.exists(f"{eos}/tests_neutrinos"):
    # If it doesn't exist, create it
    os.makedirs(f"{eos}/tests_neutrinos")


if not os.path.exists(f"{dir}/results"):
    # If it doesn't exist, create it
    os.makedirs(f"{dir}/results")



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



BATCH_SIZE = config.BATCH_SIZE
model_name = "model_ecal"
device = get_default_device()
print(device)
print(torch.version.cuda)

timestamp = time.strftime("%Y%m%d-%H%M")

limit_patience = config.PATIENCE
lr=config.LEARNING_RATE


best_loss=10000000
patience = 0

best_acc=0
epochs=1000
if config.RUN_TRAINING:
    
    model_ecal=config.MODEL
    #model_ecal=CoAtNet(config.N_BLOCKS, config.N_CHANNELS)
    classifier = config.CLASSIFIER_MLP
    print("Trainable parameters:", count_parameters(model_ecal), count_parameters(classifier))

    name_trained_model=config.TRAINED_CLS_NAME  #"model_ecal_20220929-1053_epoch-33.pt" write the name of model. start training from here
    if (os.path.isfile(name_trained_model)):
        checkpoint = torch.load(name_trained_model) if torch.cuda.is_available() else torch.load(name_trained_model, map_location=device)
        classifier.load_state_dict(checkpoint['state_dict']) if os.path.isfile(name_trained_model) else print(" ")
        print("classifier trained model is uploaded")

    name_trained_model=config.TRAINED_MODEL_NAME #"model_ecal_20220929-1053_epoch-33.pt" write the name of model. start training from here
    if (os.path.isfile(name_trained_model)):
        checkpoint = torch.load(name_trained_model) if torch.cuda.is_available() else torch.load(name_trained_model, map_location=device)
        model_ecal.load_state_dict(checkpoint['state_dict']) if os.path.isfile(name_trained_model) else print(" ")
        print("model_ecal source trained model is uploaded")

    model_ecal.to(device)
    classifier.to(device)

    if torch.cuda.device_count() > 1:
        print("Number of GPUs being used:", torch.cuda.device_count())
        model_ecal = nn.DataParallel(model_ecal)  #2
        classifier = nn.DataParallel(classifier)


    if config.IS_SINGLE_NETWORK:
        optimizer_cls_featext = torch.optim.Adam(model_ecal.parameters(), lr=lr)
    else:
        optimizer_cls_featext = torch.optim.Adam(list(model_ecal.parameters()) + list(classifier.parameters()), lr)

    train_dataset, val_dataset, label0_train_domain_adapt_dataset, label1_train_domain_adapt_dataset, lossfun_train, lossfun_val, lossfun_domain = \
        data_loader_loss_train_val_target(device)
    ### PreTrain
    for epoch in range(epochs):
        print("epoch in pretrain",epoch)
        start_time=time.time()
        print(" ")

        print("Train Classifier")
        model_ecal.train()
        classifier.train()
        train(model_ecal, classifier, train_dataset,lossfun_train,optimizer_cls_featext,device,dir)

        print("Validation ")
        classifier.eval()
        model_ecal.eval()
        loss, acc = validation(val_dataset,model_ecal, classifier, lossfun_val,device,dir)
        if config.VAL_PREJ:
            loss = loss
        else:
            loss = loss.cpu().detach().numpy()

        if acc>best_acc:
            best_acc=acc
            print("Best Acc",best_acc)

        else:
            print("Best Acc didn't change, best acc is",best_acc)

        if config.VAL_ACC_BEST_EPOCH:
            print("using accuracy to determine best epoch")
            loss=-acc

        if (loss < best_loss):
            patience = 0
            best_loss = loss
            print("best loss in prevalidation :",loss)
            print("Model kaydedilmiyor, maini düzelt")

            try:
                state_dict = model_ecal.module.state_dict()  # To unwrap DataParallel model.
            except AttributeError:
                state_dict = model_ecal.state_dict()
            state = {
                'epoch': epoch,
                'state_dict': state_dict,
                'optimizer': optimizer_cls_featext.state_dict()
            }
            torch.save(state, f"{eos}/checkpoints/" + "model_ecal" + "_" + timestamp +
                       "_epoch-{}.pt".format(epoch + 1))
            if config.IS_SINGLE_NETWORK==False:
                try:
                    state_dict = classifier.module.state_dict()  # To unwrap DataParallel model.
                except AttributeError:
                    state_dict = classifier.state_dict()
                state = {
                    'epoch': epoch,
                    'state_dict': state_dict,
                    'optimizer': optimizer_cls_featext.state_dict()
                }
                torch.save(state, f"{eos}/checkpoints/" + "classifier" + "_" + timestamp +
                           "_epoch-{}.pt".format(epoch + 1))
        else:
            patience = patience + 1
            print("best loss in prevalidation did not change, best loss is :",loss)
            print(best_loss,patience)
            if (patience == limit_patience):
                print("Model kaydedilmiyor, maini düzelt")
                break





import re

model_ecal=config.MODEL
#model_ecal=CoAtNet(config.N_BLOCKS, config.N_CHANNELS)
classifier = config.CLASSIFIER_MLP
model_ecal.to(device)
classifier.to(device)


initial_directory = os.path.join(eos, "checkpoints")
# --- Single network case ---
# --- Classifier ---
# --- Classifier (Updated) ---
if not config.IS_SINGLE_NETWORK:
    clas_files = glob.glob(os.path.join(initial_directory, "clas*.pt"))

    if clas_files:
        # Extract epoch numbers similar to ECAL block
        def extract_epoch(filename):
            match = re.search(r"epoch-(\d+)\.pt", os.path.basename(filename))
            return int(match.group(1)) if match else -1

        latest_clas = max(clas_files, key=extract_epoch)
        print("Latest classifier checkpoint:", latest_clas)

        if os.path.isfile(latest_clas):
            checkpoint = torch.load(latest_clas, map_location=device)

            # If using your custom loader
            classifier = load_checkpoint(classifier, checkpoint)

            # OR if you want direct state_dict loading:
            # classifier.load_state_dict(checkpoint['state_dict'])

        else:
            print(f"Classifier file not found: {latest_clas}")
    else:
        print("No classifier checkpoint files found.")


# --- ECAL model ---
ecal_files = glob.glob(os.path.join(initial_directory, "model_ecal_*.pt"))
if ecal_files:
    # Extract epoch number from each filename
    def extract_epoch(filename):
        match = re.search(r"epoch-(\d+)\.pt", os.path.basename(filename))
        return int(match.group(1)) if match else -1

    # Find the file with the largest epoch number
    latest_ecal = max(ecal_files, key=extract_epoch)
    print("Latest ecal file:", latest_ecal)

    if os.path.isfile(latest_ecal):
        checkpoint = torch.load(latest_ecal) if torch.cuda.is_available() \
            else torch.load(latest_ecal, map_location=device)
        model_ecal.load_state_dict(checkpoint['state_dict'])
    else:
        print(f"File not found: {latest_ecal}")
else:
    print("No ecal model files found.")


if torch.cuda.device_count() > 1:
    print("Number of GPUs being used:", torch.cuda.device_count())
    model_ecal = nn.DataParallel(model_ecal)  #2
    classifier = nn.DataParallel(classifier)
filename = os.path.basename(latest_ecal)
name_only = os.path.splitext(filename)[0]

dict_cut={
        "TB_RECALIBRATION_S2Y": [False,True], # if it is false, it uses same xy planes.

        "qdc_threshold_value_scifi_data": [0],
        "qdc_threshold_value_scifi_mc": [0],

        "t_window_data": [(2.3,1.5),(2.3,0.5),(0.5,0.5),(5,5)], #contains list of (max,min)

        "t_window_mc": [(1,1),(5,5)]
    }
run_test_for_cls(config, eos, model_ecal, classifier, device, name_only,
config.TEST_FILE_2024_W_hadrons_50gev_electron,
dict=dict_cut,
test_name="TEST_FILE_2024_W_hadrons_50gev_electron",
bins=config.bins,
signal_index = 1, 
is_neutrino=False) 

run_test_for_cls(config, eos, model_ecal, classifier, device, name_only,
config.TEST_FILE_2023_1fe_hadrons_50gev_electron,
dict=dict_cut,

test_name="TEST_FILE_2023_1fe_hadrons_50gev_electron",
bins=config.bins,
signal_index = 1, 
is_neutrino=False) 

run_test_for_cls(config, eos, model_ecal, classifier, device, name_only,
config.TEST_FILE_2023_2fe_hadrons_50gev_electron,
dict=dict_cut,
test_name="TEST_FILE_2023_2fe_hadrons_50gev_electron",
bins=config.bins,
signal_index = 1, 
is_neutrino=False)

run_test_for_cls(config, eos, model_ecal, classifier, device, name_only,
config.TEST_FILE_2023_3fe_hadrons_50gev_electron,
dict=dict_cut,
test_name="TEST_FILE_2023_3fe_hadrons_50gev_electron",
bins=config.bins,
signal_index = 1, 
is_neutrino=False)
