#!/bin/bash

# Print system information
uname -a

# Go to your working directory


batch_number=$1
divide=$2
year=$3

eos_dir="/eos/experiment/sndlhc/users/beturk/Data"
afs_dir="/afs/cern.ch/work/b/beturk/private/snd/sparse_datasets/clean_dir"

txt_file="/afs/cern.ch/work/b/beturk/private/snd/sparse_datasets/clean_dir/${year}list.txt"

cd $afs_dir
mkdir signals
# Get command-line argument
#i=$1

# Output directory or name
mkdir "$eos_dir"
out_name="$eos_dir/PT/Data_${year}/batched_data_${year}"  ### change this

#echo "i = $i"
# Run the Python script with the variables


python get_sparse_hit_info_v11.py "$out_name" "$batch_number" "$divide" "$txt_file"

# List files in current directory for verification
