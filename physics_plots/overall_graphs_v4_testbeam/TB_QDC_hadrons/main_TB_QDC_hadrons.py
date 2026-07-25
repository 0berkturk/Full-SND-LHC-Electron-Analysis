# main.py
import config_TB_QDC_hadrons
import sys
import os
sys.path.append(os.path.abspath(".."))
from test_beam_all_props_functions_v4 import *
import glob
def main():
    TEST_DATA_DIR_DATA=[]
    for energy in beam_en_list:
        print(energy)
        TEST_DATA_DIR_DATA.append(glob.glob(f"/eos/experiment/sndlhc/users/beturk/TB_slight_cuts/TB_Data/2024/*{energy}GeV*.pt")[0])

    print(TEST_DATA_DIR_DATA)
    name_data = f"electrons"
    scifi_first_index_layer_second_index_energy_array, detector_first_index_qdc_energy_of_detector_secondindex_TBenergy = plot_for_QDC(TEST_DATA_DIR_DATA, name_data, "(TB Data)")
    label_list=["Data Total(SciFi+1DS)","Data SciFi Total","Data 1DS","MC Total(SciFi+1DS)","MC SciFi Total","MC 1DS"]
    skip_some_labels=[1,0,0, 1,1,1]
    plot_1d_compare_3_domains_beam_energy_graphs(beam_en_list, [detector_first_index_qdc_energy_of_detector_secondindex_TBenergy], label_list , "only_DATA_COMP_ENERGY_", show_ideal=True, xlabel="Beam Energy [GeV]",ylabel='QDC Energy',title=f"QDC Energy vs. Beam Energy ", outdir=outdirname,skip_some=skip_some_labels)


    exit()
    tb_data_24_dir="/eos/experiment/sndlhc/users/beturk/TB/TB_Data/2023"

    wall_list=["2Fe","1Fe","3Fe"]

    TEST_DATA_DIR_DATA=[]
    for wall in wall_list:
        for energy in beam_en_list:
            print(energy,wall)
            TEST_DATA_DIR_DATA.append(glob.glob(f"{tb_data_24_dir}/*{energy}GeV_{wall}*.pt")[0])

        print(TEST_DATA_DIR_DATA)
        name_data = f"hadron_{wall}"
        scifi_first_index_layer_second_index_energy_array, detector_first_index_qdc_energy_of_detector_secondindex_TBenergy = plot_for_QDC(TEST_DATA_DIR_DATA, name_data, "(TB Data)")



if __name__ == "__main__":
    main()