# main.py
import config_TB_QDC
import sys
import os
sys.path.append(os.path.abspath(".."))
from test_beam_all_props_functions_v5 import *
import glob
def main():
    TEST_DATA_DIR_DATA=[]
    # do comparison energy by energy for electrons and hadrons
    # for each energy, open dir.
    # plot hadron and electron histograms


    TEST_DATA_DIR_DATA=[]
    #TEST_DATA_DIR_DATA.append(glob.glob(f"/eos/experiment/sndlhc/users/beturk/combined_new_MC_sparse_datasets/PG_MC_electron_1_50GeV*5.pt")[0])
    TEST_DATA_DIR_DATA.append("/eos/experiment/sndlhc/users/beturk/TB/TB_Data/2024/scifi_us_ds_2024_electrons_50GeV_run_100933_0.pt")
    TEST_DATA_DIR_DATA.append("/eos/experiment/sndlhc/users/beturk/combined_new_MC_sparse_datasets/TB_MC_2024_electron_50GeV__0.pt")
    TEST_DATA_DIR_DATA.append("/eos/experiment/sndlhc/users/beturk/TB/TB_MC/2024/OLD_TB_MC_scifi_us_ds_2024_50GeV_11_nominal_entry_points_0.pt")
    
    plot_dispatch = {
    "Mean_QDC_Energy": plot_for_everything,
    "Mean_Hit_Numbers": plot_for_everything,
    "Fraction_Shower_Width": plot_for_frac_width_shower,
    "HDW": plot_for_hdw,
    "QDC_Frac":plot_for_frac,
    "HIT_Frac": plot_for_frac,
    }
    func_plot = plot_dispatch.get(plot_prefix)
    print("PLOTTING",plot_prefix)

    if plot_prefix=="Mean_QDC_Energy":
        scifi_first_index_layer_second_index_energy_array, detector_first_index_qdc_energy_of_detector_secondindex_TBenergy = func_plot(TEST_DATA_DIR_DATA, name_data, f"(TB Data)", config_feature_name)
    elif plot_prefix==("QDC_Frac" or "HIT_Frac"):
        func_plot(TEST_DATA_DIR_DATA, name_data, f"(TB Data)", config_feature_name)
    else:
        scifi_first_index_layer_second_index_energy_array = func_plot(TEST_DATA_DIR_DATA, name_data, f"(TB Data)", config_feature_name)


if __name__ == "__main__":
    main()