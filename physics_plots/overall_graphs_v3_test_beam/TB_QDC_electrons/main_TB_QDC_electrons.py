# main.py
import config_TB_QDC_electrons
import sys
import os
sys.path.append(os.path.abspath(".."))
from test_beam_all_props_functions_v4 import *

def main():
    
    scifi_first_index_layer_second_index_energy_array, detector_first_index_qdc_energy_of_detector_secondindex_TBenergy = plot_for_QDC(TEST_DATA_DIR_DATA,name_data,"(TB Data)")

    MC_scifi_first_index_layer_second_index_energy_array, MC_detector_first_index_qdc_energy_of_detector_secondindex_TBenergy = plot_for_QDC(TEST_DATA_DIR_MC,name_MC,"(TB MC)")
    
    label_list=["Data Total(SciFi+1DS)","Data SciFi Total","Data 1DS","MC Total(SciFi+1DS)","MC SciFi Total","MC 1DS"]
    skip_some_labels=[1,0,0, 1,0,0]
    
    plot_1d_compare_3_domains_beam_energy_graphs(beam_en_list, [detector_first_index_qdc_energy_of_detector_secondindex_TBenergy,MC_detector_first_index_qdc_energy_of_detector_secondindex_TBenergy], label_list , "MC-DATA_COMP_ENERGY_", show_ideal=True, xlabel="Beam Energy [GeV]",ylabel='QDC Energy',title=f"QDC Energy vs. Beam Energy ", outdir=outdirname,skip_some=skip_some_labels)


    label_list=["MC Layer 1","MC Layer 2","MC Layer 3","MC Layer 4","MC Layer 5","Data Layer 1","Data Layer 2","Data Layer 3","Data Layer 4","Data Layer 5"]
    skip_some_labels=[1,0,0,0,1,1,0,0,0,1]
    plot_1d_compare_3_domains_beam_energy_graphs(beam_en_list, [MC_scifi_first_index_layer_second_index_energy_array,scifi_first_index_layer_second_index_energy_array], label_list , "MC-DATA_COMP_", show_ideal=False, xlabel="Beam Energy [GeV]",ylabel='SciFi QDC',title=f"SciFi QDC vs. Beam Energy ", outdir=outdirname,skip_some=skip_some_labels)


if __name__ == "__main__":
    main()