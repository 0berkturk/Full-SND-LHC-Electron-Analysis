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
    #TEST_DATA_DIR_DATA.append("/eos/experiment/sndlhc/users/beturk/TB/TB_MC_New/Sparse_Datasets_2024/MCEB_TB_MC_2024_pions_180GeV_0.pt")
    #TEST_DATA_DIR_DATA.append("/eos/experiment/sndlhc/users/beturk/TB/TB_Data/2024/scifi_us_ds_2024_pions_180GeV_W_run_100947_10.pt")

    #TEST_DATA_DIR_DATA.append("/eos/experiment/sndlhc/users/beturk/TB/TB_MC_New/Sparse_Datasets_2024/MCEB_TB_MC_2024_electron_100GeV_0.pt")
    #TEST_DATA_DIR_DATA.append("/eos/experiment/sndlhc/users/beturk/TB/TB_Small/small_scifi_us_ds_2024_electrons_150GeV_run_100928_0.pt")


    #TEST_DATA_DIR_DATA.append("/eos/experiment/sndlhc/users/beturk/TB/TB_MC/2024/OLD_TB_MC_scifi_us_ds_2024_50GeV_11_nominal_entry_points_0.pt")
    TEST_DATA_DIR_DATA.append("/eos/experiment/sndlhc/users/beturk/TB/TB_Small/MCEB_TB_MC_2024_electron_50GeV__0.pt")
    TEST_DATA_DIR_DATA.append("/eos/experiment/sndlhc/users/beturk/TB/TB_Small/small_scifi_us_ds_2024_electrons_50GeV_run_100933_0.pt")
    outdir="1d_50gev"

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
    

    #LOAD_PLOT_ALL_COMP(outdir,TEST_DATA_DIR_DATA, N, layers=[0,1,2], planes=[0,1], thresholds=[-5,-3,-1,0,2], t_max_mc=1, t_min_mc=-1, t_max_data=2.2, t_min_data=-0.5)
    #LOAD_PLOT_ALL_HIST(outdir,TEST_DATA_DIR_DATA,N)
    #LOAD_PLOT_ALL_2D_COMBINATIONS(outdir,TEST_DATA_DIR_DATA,N)
    #LOAD_PLOT_ALL_COMP_SINGLE_QDC(outdir, TEST_DATA_DIR_DATA, N, layers=[0,1,2], planes=[0,1], thresholds=[-500])

    LOAD_PLOT_ALL_COMP_SINGLE_QDC_PLANE_COMP(outdir, TEST_DATA_DIR_DATA, N, layers=[0,1,2], planes=[0,1], thresholds=[-500])
    


if __name__ == "__main__":
    main()