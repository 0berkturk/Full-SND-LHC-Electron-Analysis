#import libs
def plot_1dhist():
    pass

def plot_onebyone_1d_hists(data_dict,keys_dict):
    for key in keys_dict:
        title, x_axis, y_axis, scale, x_min, x_max, y_min,y_max =keys_dict.get(key)
        plot_variables=data_dict[key]
        plot_1dhist()

def calculate_properties():
    #calulcate properties and adds to dict
    pass

def load_and_match_data(data_number_list):
    original_data_path ="/eos/experiment/sndlhc/users/beturk/Data/data_2023"
    dl_data_path = "/eos/experiment/sndlhc/users/beturk/Data_DL_processed/data_2023"
    for ith in data_number_list:
        clsdata = torch.load(f"{dl_data_path}_{ith}_CLS_resnet_v6_TBHadrons23_MCElectrons_2layer_samexy_cut_ideal__model_ecal_20260503-2332_epoch-13.pt")
        recon_100gev = torch.load(f"{dl_data_path}_{ith}_MC_electrons_ResNets_SciFi_2layers_R256_100gev_-0.5_s2_q13_ft0_layer5__model_ecal_20260515-1842_epoch-12.pt")
        recon_log400gev = torch.load(f"{dl_data_path}_{ith}_energy_recon_log_input_more_data_MC_electrons_ResNets_SciFi_R128_400gev_0_s2_q13_ft0_layer2__model_ecal_20260517-0957_epoch-62.pt")

        #check idxs
        clsdata["idx"] == recon_100gev["idx"]
        recon_log400gev["idx"] ==recon_100gev["idx"]
        if not (clsdata["idx"] == recon_100gev["idx"])==(recon_log400gev["idx"] ==recon_100gev["idx"]):
            exit()
        
        ith_dataset=SNDSparseDataset([0,f"{original_data_path}_{ith}.pt"],EN_MIN=0,EN_MAX=20000,is_lhcdata=True)
        if len(ith_dataset)==0:
            print("zero data, passing this")
            continue


        update cuts...

        dataloader = DataLoader(
            ith_dataset, 
            batch_size=ith_dataset.size,   # FIX 3: Matched capitalization
            shuffle=False,        # FIX 2: Uses the new function argument
            num_workers=4, 
            pin_memory=True
        )

        save info....




    


keys_dict={
    "recon1":"Xasis,","sdsdgsdg"
}

#change the cuts from config.
def main():
    data_list=[108]#range(0,339)
    data_dict = load_and_match_data()
    data_dict = calculate_properties()
    plot_onebyone_1d_hists()




# import cuts from config.
# apply cuts through snd sparse dataset. it'll give idx. wtr idx in

#plot total energy deposition. and 