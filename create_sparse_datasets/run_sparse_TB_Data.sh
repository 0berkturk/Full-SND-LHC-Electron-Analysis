#!/bin/bash

# Print system information
uname -a

# Go to your working directory
#cd /afs/cern.ch/work/b/beturk/private/snd || exit 1
#source snd.sh
cd /afs/cern.ch/work/b/beturk/private/snd/sparse_datasets

run_number=$1
particle_name_beam_energy=$2
tb_year=$3
# Run the Python script with the variables
python get_sparse_hit_info_v11.py "$run_number" "$particle_name_beam_energy" "$tb_year"

# List files in current directory for verification
ls -lh

# Exit the script
#exit 0
