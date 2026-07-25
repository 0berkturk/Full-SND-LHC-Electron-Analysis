#!/bin/bash

# Print system information
uname -a

# Go to your working directory


batch_number=$1
divide="1"

eos_dir="/eos/experiment/sndlhc/users/beturk/TB/TB_MC_New"
afs_dir="/afs/cern.ch/work/b/beturk/private/snd/sparse_datasets"

txt_file="/afs/cern.ch/work/b/beturk/private/snd/sparse_datasets/TB_New_Electrons.txt"

cd $afs_dir
mkdir signals
# Get command-line argument
#i=$1

# Output directory or name
mkdir "$eos_dir"
out_name="$eos_dir/MCEB_TB_MC_2024_electron_150GeV"  ### change this

#echo "i = $i"
# Run the Python script with the variables


python get_sparse_hit_info_v11.py "$out_name" "$batch_number" "$divide" "$txt_file"
 
