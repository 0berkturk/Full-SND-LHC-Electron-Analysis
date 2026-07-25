from cuts import *
import numpy as np

SHOWER_WIDTH = None               # Crops channel width to 2*SHOWER_WIDTH (512)
USE_HIGHEST_N_LAYER = None         # How many Z-planes to keep
LOGARITHMIC_SCALING=False
IS_MC_TUNING=False ## If so, empty channels become -999 rather than 0. In this way, I calculate hit numbers correctly. because sipm can also produce 0 too.

if IS_MC_TUNING:
    KEYS_FOR_DATA_LOADER = ["scifi_signals", "scifi_time_diff"]
else:
    KEYS_FOR_DATA_LOADER = ["scifi_signals"]
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




N=5000 
TOTAL_TEST_SIZE=N

#label_list_en = ["TB Pion MC 180GeV","TB Data Pion 180GeV","TB MC Electron 50GeV","TB Data Electron 50GeV"]#["Pions(2W)","Pions(2Fe)","Pions(1Fe)"]
#label_list_en = ["TB Data Pion 180GeV","TB Data Electron 300GeV","TB Data Electron 150GeV","TB Data Electron 50GeV"]#["Pions(2W)","Pions(2Fe)","Pions(1Fe)"]
#label_list_en = ["TB MC Electron 50GeV(Y)", "TB Data Electron 50GeV(Y)","TB MC Electron 50GeV(X)", "TB Data Electron 50GeV(X)"]#["TB Data 24 Pion 180GeV-2W","TB Data 24 Pion 180GeV-1Fe","TB Data 24 Pion 180GeV-2Fe","TB Data 24 Pion 180GeV-1Fe"]#["Pions(2W)","Pions(2Fe)","Pions(1Fe)"]
label_list_en = ["TB Data Electron 50GeV","TB Data Electron 100GeV",  "TB MC Electron 50GeV", "TB MC Electron 100GeV"]#["Pions(2W)","Pions(2Fe)","Pions(1Fe)"]

beam_en_list = [50,100]
color_list=["r","b"]
NAME="Electron" ##naming of plots 
IS_MC_DATA_TITLE=" " #"No need 
skip_somelist=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
DATA_MC_INDEX_SEPARATION=500

"""label_list_en = ["100 GeV","140 GeV","180 GeV","240 GeV","300 GeV"]
beam_en_list = [100,140,180,240,300]"""