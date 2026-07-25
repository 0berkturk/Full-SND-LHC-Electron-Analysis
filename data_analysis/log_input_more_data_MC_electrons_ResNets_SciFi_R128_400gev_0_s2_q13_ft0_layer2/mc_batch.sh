#!/bin/sh -f

# Print system information
uname -a
source /afs/cern.ch/work/b/beturk/private/snd/source_conda.sh
# Activate the conda environment
conda activate myenv
cd /afs/cern.ch/work/b/beturk/private/snd/data_analysis/log_input_more_data_MC_electrons_ResNets_SciFi_R128_400gev_0_s2_q13_ft0_layer2

python main_analysis.py
