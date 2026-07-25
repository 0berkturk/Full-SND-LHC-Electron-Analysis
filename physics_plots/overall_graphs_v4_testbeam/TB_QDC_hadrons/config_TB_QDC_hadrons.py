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

if INCLUDE_NEG_QDC:
     pos_and_neg_qdc_name = "_pos_and_neg_qdc"
     outdirname="QDC_plots"+"_pos_and_neg_qdc"+str(time_window_min)+"_"+ str(time_window_max)
else:
    pos_and_neg_qdc_name=""
    outdirname="QDC_plots_beam_energy_"+str(time_window_min)+"_"+ str(time_window_max)
if scifi_cluster_radius!=None:
    outdirname=outdirname+"_Radius_"+str(scifi_cluster_radius)


PLOT_MEAN_QDC_ENERGY=True
PLOT_MEAN_HIT_NUMBERS= False
PLOT_FRACTION_SHOWER_WIDTH=False
PLOT_HDW=False

N=10000



#name_data="small_scifi_ds_2024_electrons"
name_data="qdc"+str(time_window_min)+"_"+ str(time_window_max)+"_scifi_ds_2024_electrons"

label_list_en = ["50 GeV","100 GeV","150","200","250","300"]
beam_en_list = [50,100,150,200,250,300]
"""label_list_en = ["100 GeV","140 GeV","180 GeV","240 GeV","300 GeV"]
beam_en_list = [100,140,180,240,300]"""