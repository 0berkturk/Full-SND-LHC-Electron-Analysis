SCIFI_QDC_2_GEV=1 ## IN PAPER,0.059
US_DS_QDC_2_GEV=1 # ın paper, 0.0145

scifi_cluster_radius=None # this is for the plots, this does not apply cut.


INCLUDE_NEG_QDC=False
if INCLUDE_NEG_QDC:
     pos_and_neg_qdc_name = "_pos_and_neg_qdc"
     outdirname="new_hit_numbers_pos_and_neg_qdc"
else:
    pos_and_neg_qdc_name=""
    outdirname="new_hit_numbers"
if scifi_cluster_radius!=None:
    outdirname=outdirname+"_Radius_"+str(scifi_cluster_radius)


min_pmt_qdc_value=0
min_ds_pmt_qdc_value=0

time_window_max=0.5 ## 0.41
time_window_min=0.5
time_window_ds=3

min_pmt_qdc_value_us=0
time_window_min_us=3
time_window_max_us=3

hitx,hity=15,15
cluster_radius=64


use_us=True
use_ds=True

PLOT_MEAN_QDC_ENERGY=False
PLOT_MEAN_HIT_NUMBERS= True
PLOT_FRACTION_SHOWER_WIDTH=False
PLOT_HDW=False
N=10000

name_data="small_TB_DATA_2023_hadrons"
TEST_DATA_DIR_DATA = [
("/eos/experiment/sndlhc/users/beturk/2023/small/scifi_us_ds_2023_pions_100GeV_1Fe_run_100663_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/experiment/sndlhc/users/beturk/2023/small/scifi_us_ds_2023_pions_100GeV_2Fe_run_100664_1.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/experiment/sndlhc/users/beturk/2023/small/scifi_us_ds_2023_pions_100GeV_3Fe_run_100631_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/experiment/sndlhc/users/beturk/2023/small/scifi_us_ds_2023_pions_140GeV_1Fe_run_100661_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/experiment/sndlhc/users/beturk/2023/small/scifi_us_ds_2023_pions_140GeV_2Fe_run_100666_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/experiment/sndlhc/users/beturk/2023/small/scifi_us_ds_2023_pions_140GeV_3Fe_run_100633_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/experiment/sndlhc/users/beturk/2023/small/scifi_us_ds_2023_pions_180GeV_1Fe_run_100659_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/experiment/sndlhc/users/beturk/2023/small/scifi_us_ds_2023_pions_180GeV_2Fe_run_100668_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/experiment/sndlhc/users/beturk/2023/small/scifi_us_ds_2023_pions_180GeV_3Fe_run_100635_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/experiment/sndlhc/users/beturk/2023/small/scifi_us_ds_2023_pions_240GeV_1Fe_run_100654_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/experiment/sndlhc/users/beturk/2023/small/scifi_us_ds_2023_pions_240GeV_2Fe_run_100649_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/experiment/sndlhc/users/beturk/2023/small/scifi_us_ds_2023_pions_240GeV_3Fe_run_100637_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/experiment/sndlhc/users/beturk/2023/small/scifi_us_ds_2023_pions_300GeV_1Fe_run_100653_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/experiment/sndlhc/users/beturk/2023/small/scifi_us_ds_2023_pions_300GeV_2Fe_run_100650_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us),
("/eos/experiment/sndlhc/users/beturk/2023/small/scifi_us_ds_2023_pions_300GeV_3Fe_run_100638_0.pt",use_us,use_ds, min_pmt_qdc_value, time_window_min, time_window_max,  min_pmt_qdc_value_us, time_window_min_us, time_window_max_us)
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

