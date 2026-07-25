# main.py
import config_TB_HDW_electrons
import sys
import os
sys.path.append(os.path.abspath(".."))
from test_beam_all_props_functions_v4 import *

def main():
    
    scifi_first_index_layer_second_index_energy_array = plot_for_HDW(TEST_DATA_DIR_DATA,name_data,"(TB Data)")

    MC_scifi_first_index_layer_second_index_energy_array = plot_for_HDW(TEST_DATA_DIR_MC,name_MC,"(TB MC)")


    label_list=["MC Layer 1","MC Layer 2","MC Layer 3","MC Layer 4","MC Layer 5","Data Layer 1","Data Layer 2","Data Layer 3","Data Layer 4","Data Layer 5"]
    skip_some_labels=[1,0,0,0,1,1,0,0,0,1]
    beam_en_list=[50,100,150,200,250,300]
    plot_1d_compare_3_domains_beam_energy_graphs(beam_en_list, [MC_scifi_first_index_layer_second_index_energy_array,scifi_first_index_layer_second_index_energy_array], label_list , "MC-DATA_COMP_", show_ideal=False, xlabel="Beam Energy [GeV]",ylabel='HDW',title=f"SciFi HDW vs. Beam Energy ", outdir=outdirname,skip_some=skip_some_labels)


if __name__ == "__main__":
    main()