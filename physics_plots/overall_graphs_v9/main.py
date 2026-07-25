import config
import sys
import os
import glob

# NEW IMPORT HERE
from test_beam_plotting import RUN_FINAL_MEGA_COMP_ALL_V8

def main_only_electrons():
    TEST_DATA_DIR_DATA=[]
    for energy in [50,100,150,200,250,300]:
        TEST_DATA_DIR_DATA.append([1,glob.glob(f"/eos/experiment/sndlhc/users/beturk/TB/TB_Small/small_scifi_us_ds_2024_electrons_{energy}GeV_*")[0]])
    
    for energy in [50,100,150,200,250,300]:
        TEST_DATA_DIR_DATA.append([1,glob.glob(f"/eos/experiment/sndlhc/users/beturk/TB/TB_MC_New/Sparse_Datasets_2024/shuffled_smaller_MCEB_TB_MC_2024_electron_{energy}GeV*")[0]])
    custom_labels=None
    print(TEST_DATA_DIR_DATA)
    print(len(TEST_DATA_DIR_DATA))
    # ... (Your existing glob path appends stay exactly the same) ...
    
    dict_cut={
        "TB_RECALIBRATION_S2Y": [True,False],
        "qdc_threshold_value_scifi_data": [-500,0],
        "qdc_threshold_value_scifi_mc": [-500,0],
        "t_window_data": [(2.3,0.5),(0.5,0.5)],
        "t_window_mc": [(1,1),(5,5)],
    }

    # This passes smoothly into the new framework
    RUN_FINAL_MEGA_COMP_ALL_V8(config, TEST_DATA_DIR_DATA, dict_cut, custom_labels)

def main_electron_pion_comparision():
    TEST_DATA_DIR_DATA=[]
    for energy in [50,150,300]:
        TEST_DATA_DIR_DATA.append([1,glob.glob(f"/eos/experiment/sndlhc/users/beturk/TB/TB_Small/small_scifi_us_ds_2024_electrons_{energy}GeV_*")[0]])
    
    """TEST_DATA_DIR_DATA.append([1,"/eos/experiment/sndlhc/users/beturk/TB/TB_Small/small_scifi_us_ds_2024_pions_180GeV_W_run_100947_10.pt"])
    TEST_DATA_DIR_DATA.append([1,"/eos/experiment/sndlhc/users/beturk/TB/TB_Small/small_scifi_us_ds_2023_pions_100GeV_3Fe_run_100631_0.pt"])
    """
    custom_labels=["180GeV Pion(2W)","100GeV Pion(3Fe)"]
    print(TEST_DATA_DIR_DATA)
    print(len(TEST_DATA_DIR_DATA))


    dict_cut={
        "TB_RECALIBRATION_S2Y": [True,False],
        "qdc_threshold_value_scifi_data": [0,-500],
        "t_window_data": [(2.3,0.5),(0.5,0.5)]
    }

    # This passes smoothly into the new framework
    RUN_FINAL_MEGA_COMP_ALL_V8(config, TEST_DATA_DIR_DATA, dict_cut, custom_labels)

def main_only_pion_comparision():
    TEST_DATA_DIR_DATA=[]
    for energy in [100,140,180,240,300]:
        for wall in ["1Fe","2Fe","3Fe"]:
            print(energy,wall)
            TEST_DATA_DIR_DATA.append([1,glob.glob(f"/eos/experiment/sndlhc/users/beturk/TB/TB_Small/small_scifi_us_ds_2023_pions_{energy}GeV*{wall}*")[0]])
    TEST_DATA_DIR_DATA.append([1,"/eos/experiment/sndlhc/users/beturk/TB/TB_Small/small_scifi_us_ds_2024_pions_180GeV_W_run_100947_10.pt"])

    """TEST_DATA_DIR_DATA.append([1,"/eos/experiment/sndlhc/users/beturk/TB/TB_Small/small_scifi_us_ds_2024_pions_180GeV_W_run_100947_10.pt"])
    TEST_DATA_DIR_DATA.append([1,"/eos/experiment/sndlhc/users/beturk/TB/TB_Small/small_scifi_us_ds_2023_pions_100GeV_3Fe_run_100631_0.pt"])
    """
    custom_labels=None#["180GeV Pion(2W)","100GeV Pion(3Fe)"]
    print(TEST_DATA_DIR_DATA)
    print(len(TEST_DATA_DIR_DATA))


    dict_cut={
        "TB_RECALIBRATION_S2Y": [True,False],
        "qdc_threshold_value_scifi_data": [0,-500],
        "t_window_data": [(2.3,0.5),(0.5,0.5)]
    }

    # This passes smoothly into the new framework
    RUN_FINAL_MEGA_COMP_ALL_V8(config, TEST_DATA_DIR_DATA, dict_cut, custom_labels)


def main_tune():
    TEST_DATA_DIR_DATA=[]
    for energy in [50,100,150,200,250,300]:
        TEST_DATA_DIR_DATA.append([1,glob.glob(f"/eos/experiment/sndlhc/users/beturk/TB/TB_Small/small_scifi_us_ds_2024_electrons_{energy}GeV_*")[0]])
    
    for energy in [50,100,150,200,250,300]:
        TEST_DATA_DIR_DATA.append([1,glob.glob(f"/eos/experiment/sndlhc/users/beturk/TB/TB_MC_New/Sparse_Datasets_2024/shuffled_smaller_MCEB_TB_MC_2024_electron_{energy}GeV*")[0]])
    custom_labels=None
    print(TEST_DATA_DIR_DATA)
    print(len(TEST_DATA_DIR_DATA))
    # ... (Your existing glob path appends stay exactly the same) ...
    
    dict_cut={
        "TB_RECALIBRATION_S2Y": [False],
        "qdc_threshold_value_scifi_data": [0],
        "qdc_threshold_value_scifi_mc": [-1],
        "t_window_data": [(2.3,0.5)],
        "t_window_mc": [(1,1)],
        "noise_sigma":[2],
        "q_max":[19]
    }

    # This passes smoothly into the new framework
    RUN_FINAL_MEGA_COMP_ALL_V8(config, TEST_DATA_DIR_DATA, dict_cut,custom_labels)

if __name__ == "__main__":
    main_tune()