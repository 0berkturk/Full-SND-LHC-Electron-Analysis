# main.py
import config_TB_QDC
import sys
import os
sys.path.append(os.path.abspath("../.."))
from test_beam_all_props_functions_v5 import *
import glob
def main():
    TEST_DATA_DIR_DATA=[]
    # do comparison energy by energy for electrons and hadrons
    # for each energy, open dir.
    # plot hadron and electron histograms


    TEST_DATA_DIR_DATA=[]
    TEST_DATA_DIR_DATA.append(glob.glob(f"/eos/experiment/sndlhc/users/beturk/TB/TB_Data/2024/*pions_180GeV_W*.pt")[0])
    TEST_DATA_DIR_DATA.append(glob.glob(f"/eos/experiment/sndlhc/users/beturk/TB/TB_Data/2024/*electrons_50GeV*.pt")[0])
    TEST_DATA_DIR_DATA.append(glob.glob(f"/eos/experiment/sndlhc/users/beturk/TB/TB_Data/2024/*electrons_150GeV*.pt")[0])
    TEST_DATA_DIR_DATA.append(glob.glob(f"/eos/experiment/sndlhc/users/beturk/TB/TB_Data/2024/*electrons_300GeV*.pt")[0])

    
    plot_dispatch = {
    "Mean_QDC_Energy": plot_for_everything,
    "Mean_Hit_Numbers": plot_for_everything,
    "Fraction_Shower_Width": plot_for_frac_width_shower,
    "HDW": plot_for_hdw,
    "QDC_Frac":plot_for_frac,
    "HIT_Frac": plot_for_frac,
    }
    func_plot = plot_dispatch.get(plot_prefix)

    if plot_prefix=="Mean_QDC_Energy":
        scifi_first_index_layer_second_index_energy_array, detector_first_index_qdc_energy_of_detector_secondindex_TBenergy = func_plot(TEST_DATA_DIR_DATA, name_data, f"(TB Data)", config_feature_name)
    elif plot_prefix==("QDC_Frac" or "HIT_Frac"):
        func_plot(TEST_DATA_DIR_DATA, name_data, f"(TB Data)", config_feature_name)
    else:
        scifi_first_index_layer_second_index_energy_array = func_plot(TEST_DATA_DIR_DATA, name_data, f"(TB Data)", config_feature_name)


if __name__ == "__main__":
    main()