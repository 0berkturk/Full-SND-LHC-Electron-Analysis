#!/bin/sh -f

# Print system information
uname -a
source /afs/cern.ch/work/b/beturk/private/snd/source_conda.sh
# Activate the conda environment
conda activate myenv
cd /afs/cern.ch/work/b/beturk/private/snd/data_analysis/MC_electrons_ResNets_SciFi_2layers_R256_100gev_-0.5_s2_q13_ft0_layer5

python main_analysis.py
