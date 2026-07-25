# main.py
import config_TB_frac_width_shower_electrons
import sys
import os
sys.path.append(os.path.abspath(".."))
from test_beam_all_props_functions_v4 import *

def main():
    
    plot_for_frac_width_shower(TEST_DATA_DIR_DATA,"mc_data_comp","")

if __name__ == "__main__":
    main()
    