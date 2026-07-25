import os
import torch
import matplotlib.pyplot as plt
from dl_recon_core_sparse.test_functions_cls import *
from dl_recon_core_sparse.test_functions_energy import *
from dl_recon_core_sparse.data_loader import *
def save_all_probs(test_data_name, feature_extractor, classifier, out_name, device):
    feature_extractor.eval()
    classifier.eval()
    
    print("\nRUNNING save_all_probs")

    all_tensors = SNDSparseDataset(test_data_name)
    test_loader = DataLoader(all_tensors, batch_size=config.BATCH_SIZE_TEST, shuffle=False)

    logits = []
    targets = []
    energy_list_check = []

    # --- 3. INFERENCE LOOP ---
    print("Starting inference loop...")
    with torch.no_grad():
        for k, data in enumerate(test_loader):
            # Unpack data based on config
            # Note: Ensure these match the order in config.KEYS_FOR_DATA_LOADER
            if config.USE_ONLY_SCIFI:
                scifi, energy, labels = data
                x = scifi.to(device)
                energy, labels = energy.to(device), labels.to(device)
                
            elif config.USE_SCIFI_US:
                scifi, us, energy, labels = data
                scifi, us = scifi.to(device), us.to(device)
                energy, labels = energy.to(device), labels.to(device)
                x = (scifi, us)
                
            elif config.USE_SCIFI_US_DS:
                scifi, us, ds, energy, labels = data
                scifi, us, ds = scifi.to(device), us.to(device), ds.to(device)
                energy, labels = energy.to(device), labels.to(device)
                x = (scifi, us, ds)

            if config.IS_SINGLE_NETWORK:
                output = feature_extractor(x)
            else:
                output = classifier(feature_extractor(x))

        # Collect results
        # Move to CPU immediately to save GPU memory
            output = output.cpu()
            labels = labels.cpu()
            energy = energy.cpu()

            logits.append(output)
            targets.append(labels)
            energy_list_check.append(energy)
        

    logits = torch.cat(logits, dim=0)
    targets = torch.cat(targets, dim=0)
    energy_list_check = torch.cat(energy_list_check, dim=0)
    # --- 4. SAVE AND VERIFY ---
    dataset_energies = all_tensors.energies.cpu()
    dataset_labels = all_tensors.labels.cpu()

    # Verification: Check if the order was preserved
    is_energy_ok = torch.equal(energy_list_check, dataset_energies)
    is_target_ok = torch.equal(targets, dataset_labels)

    save_tensor ={}
    save_tensor['new_model'] = logits
    save_tensor["en3d"] = energy_list_check
    save_tensor["y"] = targets 

    if is_energy_ok and is_target_ok:
        print("Saving prob data...")
        torch.save(save_tensor, out_name)
        print(f"Saved to {out_name}")
        print("Logits shape:", logits.shape)
        
        # Debug prints
        for key in save_tensor:
            if hasattr(save_tensor[key], 'shape'):
                print("key in saved file:", key, save_tensor[key].shape)
        
        return save_tensor
    else:
        print("\nCRITICAL ERROR: Data mismatch!")
        print(f"Energy match: {is_energy_ok}")
        print(f"Target match: {is_target_ok}")
        print("Expected shapes:", dataset_energies.shape, dataset_labels.shape)
        print("Actual shapes:  ", energy_list_check.shape, targets.shape)
        print("Exiting to prevent corrupt data save.")
        exit()


def run_test_for_energy(config, eos, model_ecal, classifier, device, name_only,datalist,test_name="define particle type name",signal_index = None, is_neutrino=False,particle_name=" ",BEAM_OR_TRUE_ENERGY="True"):

    common_dir = f"{eos}/tests_{test_name}"
    os.makedirs(common_dir, exist_ok=True)
    common_out_name = f"{common_dir}/{test_name}_{name_only}_"

    test_data = {}
    if BEAM_OR_TRUE_ENERGY=="Beam":
        beam_energy_list=[]
        recon_en_list=[]
    # === Load or create probability data for each test dataset ===
    for test_data_name in datalist:
        test_data_name1 = os.path.splitext(os.path.basename(test_data_name[1]))[0]

        out_name = f"{eos}/energy_{test_data_name1}_{name_only}.pt"

        if os.path.isfile(out_name):
            print("found data",out_name)
            ithtest_data = torch.load(out_name)
            for key in ithtest_data:
                print(key, ithtest_data[key])
        else:
            print("No saved probs found, creating one:", out_name)
            ithtest_data = save_all_probs(test_data_name, model_ecal, classifier, out_name, device)
        os.makedirs(common_out_name+test_data_name1+f"/{test_data_name1}", exist_ok=True)
        plot_res_energy(ithtest_data["new_model"],ithtest_data["en3d"],common_out_name+test_data_name1+f"/{test_data_name1}",particle_name, BEAM_OR_TRUE_ENERGY)
        #test_model_params_hist(ithtest_data["scifi_sig"], ithtest_data["new_model"], ithtest_data["en3d"], test_data_name1 ,eos,common_out_name)
        if BEAM_OR_TRUE_ENERGY=="Beam":
            beam_energy_list.append(ithtest_data["en3d"][0])
            recon_en_list.append(ithtest_data["new_model"])

        # Concatenate across datasets
        for key in ithtest_data:
            if key in test_data:
                test_data[key] = torch.cat((test_data[key], ithtest_data[key]))
            else:
                test_data[key] = ithtest_data[key]

        print("Added to test_data dict.\n")

    plot_1d_beam_energy_graphs(beam_energy_list, recon_en_list,"MEAN_STD_OF_RECON_ENERGY" ,xlabel="Beam Energy [GeV]",ylabel='Average Recon. Energy[GeV]',title="Average Recon. vs True Energy",outdir=common_dir,show_ideal=True)


    # === Extract data ===
    if signal_index!=None:
        index = test_data["y"]==signal_index

        res_array = plot_res_energy(test_data["new_model"][index],test_data["en3d"][index],common_out_name+"all",particle_name, BEAM_OR_TRUE_ENERGY)
        #test_model_params_hist(test_data["scifi_sig"][index], test_data["new_model"][index], test_data["en3d"][index], "all_"+test_name ,eos,common_out_name)
    else:
        res_array = plot_res_energy(test_data["new_model"],test_data["en3d"],common_out_name+"all",particle_name, BEAM_OR_TRUE_ENERGY)
        #test_model_params_hist(test_data["scifi_sig"], test_data["new_model"], test_data["en3d"],  "all_"+test_name ,eos,common_out_name)
    return res_array


def run_test_for_cls(config, eos, model_ecal, classifier, device, name_only,datalist,bins,test_name="define particle type name",
    ,cuts, signal_index = 4, is_neutrino=False):

    cut_dir_name="makeitusing cuts"

    common_dir = f"{eos}/{cut_dir_name}/tests_{test_name}"
    os.makedirs(common_dir, exist_ok=True)
    common_out_name = f"{common_dir}/{test_name}_{name_only}_"

    test_data = {}

    # === Load or create probability data for each test dataset ===
    for test_data_name in datalist:

        test_data_name1 = os.path.splitext(os.path.basename(test_data_name[1]))[0]

        out_name = f"{eos}/{cut_dir_name}/probs_{test_data_name1}_{name_only}.pt"

        if os.path.isfile(out_name):
            ithtest_data = torch.load(out_name)
                
            print("Loaded test data, found as", out_name)

        else:
            print("No saved probs found, creating one:", out_name)
            ithtest_data = save_all_probs(test_data_name, model_ecal, classifier, out_name, device)
        # Concatenate across datasets

        for key in ithtest_data:
            print(key, ithtest_data[key].shape)
            if key in test_data:
                test_data[key] = torch.cat((test_data[key], ithtest_data[key]))
            else:
                test_data[key] = ithtest_data[key]

        print("Added to test_data dict.\n")

    #index = test_data["y"]==0

    # === Extract data ===
    logits = test_data["new_model"]#[index]
    targets = test_data["y"]#[index]
    energies = test_data["en3d"]#[index]

    print("Loaded all data.")
    with open(common_out_name+"tf_metrics.txt", "w") as f:
        for i in range(10):
            print(i, torch.sum(targets==i))
            precision, recall, f1, accuracy = binary_metrics(targets,logits,i)
        
            f.write(f"for signal index i, {i}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n \n")
    print(common_out_name)
    # === Convert to binary probabilities ===
    print(logits)
    print(targets)
    signal_probs, background_probs, background_en3d = convert_2_binary_probs(logits, targets, energies, signal_index)

    if config.IS_BINARY:

        plt.hist(signal_probs.cpu().numpy(), bins=30, alpha=0.5, label="Signal")
        plt.hist(background_probs.cpu().numpy(), bins=30, alpha=0.5, label="Background")
        plt.xlabel("Model's Logit Value")
        plt.ylabel("Counts")
        plt.title("Signal vs Background Logit Distribution")
        plt.legend()
        plt.yscale("log")
        plt.savefig(common_out_name + "norm.png", dpi=300)
        plt.clf()
        print("shape of model", background_probs.shape,signal_probs.shape)
        print("max and min of logits",background_probs.max().item(), signal_probs.min().item())
        # return 0
    else:
        plot_confusion_matrix(logits, targets,
                              out_name=common_out_name + "conf_matrix.png",
                              class_names=config.CLASS_NAMES_CONF_MATRIX,GET_PARTICLES_WITH_INDEX=config.GET_PARTICLES_WITH_INDEX)
        plot_confusion_matrix(logits, targets,
                              out_name=common_out_name + "conf_matrix_larger.png",
                              class_names=config.CLASS_NAMES_CONF_MATRIX_LARGER,GET_PARTICLES_WITH_INDEX=config.GET_PARTICLES_WITH_INDEX_LARGER)

    # === Global efficiency and rejection ===
    thresholds, signal_eff, background_rej = s_eff_b_rej(signal_probs, background_probs,
                                                         common_out_name + str(signal_index))
    required_eff = config.REQUIRED_EFF
    threshold, _, _ = find_threshold(thresholds, signal_eff, background_rej, required_eff)
    p_rej_energy(threshold, background_probs, background_en3d,
                 common_out_name + str(signal_index), str(int(required_eff * 100)),bins ,True)

    if is_neutrino:
        calc_signal_background(logits,targets,threshold,common_out_name)
        calc_signal_background_def(logits,targets,common_out_name)
        plot_purities(logits,targets,common_out_name)


    # === Energy binning ===
    better_rej_at_given_eff = []

    for i in range(len(bins) - 1):
        en1, en2 = bins[i], bins[i + 1]
        print(f"\nEnergy range: {en1}-{en2} GeV")

        common_dir = f"{eos}/tests_{test_name}/energy_bins/en_{en1}_{en2}/"
        os.makedirs(common_dir, exist_ok=True)
        common_out_name = f"{common_dir}/{test_name}_{name_only}_"

        cut = (energies > en1) & (energies < en2)
        title2 = f"({en1}-{en2} GeV)"

        print("Electron number:", torch.sum(targets[cut] == 4))
        print("Hadron number:", torch.sum(targets[cut] == 5), "\n")
        print("NC number:", torch.sum(targets[cut] == 3), "\n")

        signal_probs, background_probs, _ = convert_2_binary_probs(
            logits[cut], targets[cut], energies[cut], signal_index
        )

        if (len(signal_probs) > 5) and (len(background_probs) > 5):
            if config.IS_BINARY:
                plt.hist(signal_probs.cpu().numpy(), bins=np.linspace(0, 1, 20), alpha=0.5, label="Signal")
                plt.hist(background_probs.cpu().numpy(), bins=np.linspace(0, 1, 20), alpha=0.5, label="Background")
                plt.xlabel("Model's Logit Value")
                plt.ylabel("Counts")
                plt.title("Signal vs Background Logit Distribution")
                plt.legend()
                plt.yscale("log")
                plt.savefig(common_out_name + ".png", dpi=300)
                plt.clf()
            else:
                plot_confusion_matrix(logits[cut], targets[cut],
                                    out_name=common_out_name + "conf_matrix.png",
                                    class_names=config.CLASS_NAMES_CONF_MATRIX,
                                    title2=title2)

            thresholds, signal_eff, background_rej = s_eff_b_rej(
                signal_probs, background_probs,
                common_out_name + str(signal_index), title2
            )
            threshold, _, rejection_at_given_eff = find_threshold(
                thresholds, signal_eff, background_rej, required_eff
            )
            better_rej_at_given_eff.append(rejection_at_given_eff)

            if is_neutrino:
                calc_signal_background(logits,targets,threshold,common_out_name)
                calc_signal_background_def(logits,targets,common_out_name)
                plot_purities(logits,targets,common_out_name)
        else:
            better_rej_at_given_eff.append(0)

    # === Final combined rejection plot ===
    common_dir = f"{eos}/tests_{test_name}"
    common_out_name = f"{eos}/tests_{test_name}/{test_name}_{name_only}_"
    single_better_p_rej_energy(better_rej_at_given_eff, background_en3d,
                               common_out_name + str(signal_index),
                               str(int(required_eff * 100)), bins)

    print("All-particle testing completed successfully.")
