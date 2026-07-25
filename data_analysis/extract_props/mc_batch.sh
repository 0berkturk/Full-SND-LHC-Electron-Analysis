#!/bin/sh -f

# Print system information
uname -a
source /afs/cern.ch/work/b/beturk/private/snd/source_conda.sh
# Activate the conda environment
conda activate myenv
cd /afs/cern.ch/work/b/beturk/private/snd/data_analysis/extract_props
python extract_props.py
