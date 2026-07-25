# main.py
import config
import sys
import os
sys.path.append(os.path.abspath(".."))
from test_beam_all_props_functions_v8 import *

import glob
def main():
    # do comparison energy by energy for electrons and hadrons
    # for each energy, open dir.
    # plot hadron and electron histograms


    TEST_DATA_DIR_DATA=[]
    #TEST_DATA_DIR_DATA.append("/eos/experiment/sndlhc/users/beturk/TB/TB_MC_New/Sparse_Datasets_2024/MCEB_TB_MC_2024_pions_180GeV_0.pt")
    #TEST_DATA_DIR_DATA.append("/eos/experiment/sndlhc/users/beturk/TB/TB_Data/2024/scifi_us_ds_2024_pions_180GeV_W_run_100947_10.pt")
    #TEST_DATA_DIR_DATA.append("/eos/experiment/sndlhc/users/beturk/TB/TB_MC_New/Sparse_Datasets_2024/MCEB_TB_MC_2024_electron_100GeV_0.pt")
    #TEST_DATA_DIR_DATA.append("/eos/experiment/sndlhc/users/beturk/TB/TB_Small/small_scifi_us_ds_2024_electrons_150GeV_run_100928_0.pt")
    #TEST_DATA_DIR_DATA.append("/eos/experiment/sndlhc/users/beturk/TB/TB_MC/2024/OLD_TB_MC_scifi_us_ds_2024_50GeV_11_nominal_entry_points_0.pt")
    TEST_DATA_DIR_DATA.append([1,glob.glob("/eos/experiment/sndlhc/users/beturk/TB/TB_Small/small_scifi_us_ds_2024_electrons_50GeV_*")[0]])
    TEST_DATA_DIR_DATA.append([1,glob.glob("/eos/experiment/sndlhc/users/beturk/TB/TB_Small/small_scifi_us_ds_2024_electrons_100GeV_*")[0]])

    TEST_DATA_DIR_DATA.append([1,glob.glob("/eos/experiment/sndlhc/users/beturk/TB/TB_Small/small_MCEB_TB_MC_2024_electron_50GeV*")[0]])
    TEST_DATA_DIR_DATA.append([1,glob.glob("/eos/experiment/sndlhc/users/beturk/TB/TB_Small/small_MCEB_TB_MC_2024_electron_100GeV*")[0]])


    dict_cut={
        "TB_RECALIBRATION_S2Y": [True,False], # if it is false, it uses same xy planes.

        "qdc_threshold_value_scifi_data": [-210,-10,0],
        "qdc_threshold_value_scifi_mc": [0,-100],

        "t_window_data": [(2.3,0.5),(2.3,1.5),(5,5),(0.5,0.5) ], #contains list of (max,min)

        "t_window_mc": [(5,5),(1,1)]
    }

    RUN_FINAL_MEGA_COMP_ALL_V8(config,TEST_DATA_DIR_DATA,dict_cut)


if __name__ == "__main__":
    main()