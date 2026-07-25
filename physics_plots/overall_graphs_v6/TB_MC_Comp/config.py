from cuts import *
import numpy as np

SHOWER_WIDTH = None               # Crops channel width to 2*SHOWER_WIDTH (512)
USE_HIGHEST_N_LAYER = None         # How many Z-planes to keep

IS_MC_TUNING=False ## If so, empty channels become -999 rather than 0. In this way, I calculate hit numbers correctly. because sipm can also produce 0 too.

if IS_MC_TUNING:
    KEYS_FOR_DATA_LOADER = ["scifi_signals", "scifi_times"]
else:
    KEYS_FOR_DATA_LOADER = ["scifi_signals", "us_signals", "ds" , "en3d"]
print(KEYS_FOR_DATA_LOADER)
# DO NOT TOUCH BELOW, JUST CHANGE MODEL NAME.
USE_US = False 
USE_DS = False 
if "us_signals" in KEYS_FOR_DATA_LOADER:
    USE_US = True 

if "ds" in KEYS_FOR_DATA_LOADER:
    USE_DS = True 

if USE_US and USE_DS:
    USE_SCIFI_US_DS = True
    USE_SCIFI_US = False
elif USE_US:
    USE_SCIFI_US = True
    USE_SCIFI_US_DS = False
else:
    USE_ONLY_SCIFI = True 
    USE_SCIFI_US = False
    USE_SCIFI_US_DS = False


SCIFI_QDC_2_GEV=1 ## IN PAPER,0.059
US_DS_QDC_2_GEV=1 # ın paper, 0.0145

scifi_cluster_radius=None # this is for the plots, this does not apply cut.


PLOT_MEAN_QDC_ENERGY = False
PLOT_MEAN_HIT_NUMBERS = True
PLOT_FRACTION_SHOWER_WIDTH = False
PLOT_HDW = False
PLOT_QDC_FRAC=False
PLOT_HIT_FRAC=False

# 1. Determine the prefix based on the active plot type
if PLOT_MEAN_QDC_ENERGY:
    plot_prefix = "Mean_QDC_Energy"
    config_feature_name="QDC"
elif PLOT_MEAN_HIT_NUMBERS:
    plot_prefix = "Mean_Hit_Numbers"
    config_feature_name="Total Hit Number"
elif PLOT_FRACTION_SHOWER_WIDTH:
    plot_prefix = "Fraction_Shower_Width"
elif PLOT_HDW:
    plot_prefix = "HDW"
    config_feature_name="Total HDW"
elif PLOT_QDC_FRAC:
    plot_prefix="QDC_Frac"
    config_feature_name="QDC Fraction of Highest 2 Layer"
elif PLOT_HIT_FRAC:
    plot_prefix="HIT_Frac"
    config_feature_name="Hit Number Fraction of Highest 2 Layer"
else:
    plot_prefix = "General"


# 3. Construct the base directory name using f-strings
outdirname = f"{plot_prefix}_plots"


N=5000 # both batchsize and total number

name_data = f"pg_electron_comp"
label_list_en = ["TB Data","TB New MC"]#["Pions(2W)","Pions(2Fe)","Pions(1Fe)"]
beam_en_list = [50,50]

"""label_list_en = ["100 GeV","140 GeV","180 GeV","240 GeV","300 GeV"]
beam_en_list = [100,140,180,240,300]"""