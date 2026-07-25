SCIFI_QDC_2_GEV=1 ## IN PAPER,0.059
US_DS_QDC_2_GEV=1 # ın paper, 0.0145

scifi_cluster_radius=None # this is for the plots, this does not apply cut.
INCLUDE_NEG_QDC=False

min_pmt_qdc_value=0
min_ds_pmt_qdc_value=0

time_window_min=0.5
time_window_max=0.5 ## 0.41

time_window_ds=3

min_pmt_qdc_value_us=0
time_window_min_us=3
time_window_max_us=3

hitx,hity=15,15
cluster_radius=64

use_us=True
use_ds=True

PLOT_MEAN_QDC_ENERGY = False
PLOT_MEAN_HIT_NUMBERS = False
PLOT_FRACTION_SHOWER_WIDTH = False
PLOT_HDW = False
PLOT_QDC_FRAC=True
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


# 2. Determine the QDC string
if INCLUDE_NEG_QDC:
    pos_and_neg_qdc_name = "_pos_and_neg_qdc"
else:
    pos_and_neg_qdc_name = ""

# 3. Construct the base directory name using f-strings
outdirname = f"radius15cut_{plot_prefix}_plots{pos_and_neg_qdc_name}_{time_window_min}_{time_window_max}"

# 4. Append the radius if it is defined
if scifi_cluster_radius is not None:
    outdirname += f"_Radius_{scifi_cluster_radius}"



N=10000

name_data = f"electron_hadron_comp"
label_list_en = ["Pions(180GeV)","Electrons(50GeV)","Electrons(150GeV)","Electrons(300GeV)"]#["Pions(2W)","Pions(2Fe)","Pions(1Fe)"]
beam_en_list = [180,50,150,300]
"""label_list_en = ["100 GeV","140 GeV","180 GeV","240 GeV","300 GeV"]
beam_en_list = [100,140,180,240,300]"""