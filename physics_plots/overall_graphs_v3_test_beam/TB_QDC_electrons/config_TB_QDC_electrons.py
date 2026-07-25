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

label_list_en = ["50 GeV","100 GeV","150 GeV","200 GeV","250 GeV","300 GeV"]
beam_en_list = [50,100,150,200,250,300]

#name_data="small_scifi_ds_2024_electrons"
name_data="50_300_intime"+str(time_window_min)+"_"+ str(time_window_max)+"_scifi_ds_2024_electrons"

tb_data_24_dir="/eos/experiment/sndlhc/users/beturk/TB_tensor_datasets/TB_Data/2024"
TEST_DATA_DIR_DATA = [
(f"{tb_data_24_dir}/small/scifi_us_ds_2024_electrons_50GeV_run_100933_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
(f"{tb_data_24_dir}/small/scifi_us_ds_2024_electrons_100GeV_run_100916_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
(f"{tb_data_24_dir}/small/scifi_us_ds_2024_electrons_150GeV_run_100928_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
(f"{tb_data_24_dir}/small/scifi_us_ds_2024_electrons_200GeV_run_100924_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
(f"{tb_data_24_dir}/small/scifi_us_ds_2024_electrons_250GeV_run_100929_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
(f"{tb_data_24_dir}/small/scifi_us_ds_2024_electrons_300GeV_run_100926_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
]

name_MC="TB_MC_nominal"
tb_mc_dir = "/eos/user/b/beturk/snd/test_beam/MC_24"
TEST_DATA_DIR_MC = [
(f"{tb_mc_dir}/TB_MC_scifi_us_ds_2024_50GeV_11_nominal_entry_points_all_files.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
(f"{tb_mc_dir}/TB_MC_scifi_us_ds_2024_100GeV_11_nominal_entry_points_all_files.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
(f"{tb_mc_dir}/TB_MC_scifi_us_ds_2024_150GeV_11_nominal_entry_points_all_files.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
(f"{tb_mc_dir}/TB_MC_scifi_us_ds_2024_200GeV_11_nominal_entry_points_all_files.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
(f"{tb_mc_dir}/TB_MC_scifi_us_ds_2024_250GeV_11_nominal_entry_points_all_files.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
(f"{tb_mc_dir}/TB_MC_scifi_us_ds_2024_300GeV_11_nominal_entry_points_all_files.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us)
]

