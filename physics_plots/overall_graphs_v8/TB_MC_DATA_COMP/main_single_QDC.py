# main.py
import config
import sys
import os
sys.path.append(os.path.abspath(".."))
from test_beam_all_props_functions_v7 import *

import glob
def main():
    # do comparison energy by energy for electrons and hadrons
    # for each energy, open dir.
    # plot hadron and electron histograms


    TEST_DATA_DIR_DATA=[]
    """TEST_DATA_DIR_DATA.append("/eos/experiment/sndlhc/users/beturk/TB/TB_Small/small_scifi_us_ds_2024_pions_180GeV_W_run_100947_10.pt")
    TEST_DATA_DIR_DATA.append("/eos/experiment/sndlhc/users/beturk/TB/TB_Small/small_scifi_us_ds_2024_electrons_3000GeV_run_100926_0.pt")
    TEST_DATA_DIR_DATA.append("/eos/experiment/sndlhc/users/beturk/TB/TB_Small/small_scifi_us_ds_2024_electrons_150GeV_run_100928_0.pt")
    TEST_DATA_DIR_DATA.append("/eos/experiment/sndlhc/users/beturk/TB/TB_Small/small_scifi_us_ds_2024_electrons_50GeV_run_100933_0.pt")
    outdir="""
    """TEST_DATA_DIR_DATA.append("/eos/experiment/sndlhc/users/beturk/TB/TB_Small/small_scifi_us_ds_2024_pions_180GeV_W_run_100947_10.pt")
    TEST_DATA_DIR_DATA.append("/eos/experiment/sndlhc/users/beturk/TB/TB_Data/2023/scifi_us_ds_2023_pions_180GeV_3Fe_run_100635_16.pt")
    TEST_DATA_DIR_DATA.append("/eos/experiment/sndlhc/users/beturk/TB/TB_Data/2023/scifi_us_ds_2023_pions_180GeV_2Fe_run_100668_5.pt")
    #TEST_DATA_DIR_DATA.append("/eos/experiment/sndlhc/users/beturk/TB/TB_Data/2023/scifi_us_ds_2023_pions_180GeV_1Fe_run_100659_2.pt")
    outdir="1d_hists_single_qdc_180gev_pion_wall_comp"""

    TEST_DATA_DIR_DATA.append("/eos/experiment/sndlhc/users/beturk/TB/TB_Small/small_scifi_us_ds_2024_pions_180GeV_W_run_100947_10.pt")
    TEST_DATA_DIR_DATA.append("/eos/experiment/sndlhc/users/beturk/TB/TB_Data/2024/scifi_us_ds_2024_pions_100GeV_1Fe_run_100957_0.pt")
    outdir="1d_hists_single_qdc_180gev_pion_wall_comp_2024"


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
    
    LOAD_PLOT_ALL_COMP_SINGLE_QDC(outdir, TEST_DATA_DIR_DATA, N, layers=[0,1,2,3,4], planes=[0,1], thresholds=[-500], 
                       t_max_mc=1, t_min_mc=-1, 
                       t_max_data=2.2, t_min_data=-0.5)
    #LOAD_PLOT_ALL_HIST(TEST_DATA_DIR_DATA,N)
    #LOAD_PLOT_ALL_2D_COMBINATIONS(TEST_DATA_DIR_DATA,N)
    


if __name__ == "__main__":
    main()