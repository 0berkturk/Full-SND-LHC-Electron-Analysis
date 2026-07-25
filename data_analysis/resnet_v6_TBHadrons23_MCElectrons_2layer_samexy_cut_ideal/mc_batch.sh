#!/bin/sh -f

# Print system information
uname -a
source /afs/cern.ch/work/b/beturk/private/snd/source_conda.sh
# Activate the conda environment
conda activate myenv
cd /afs/cern.ch/work/b/beturk/private/snd/data_analysis/resnet_v6_TBHadrons23_MCElectrons_2layer_samexy_cut_ideal

python main_analysis.py
