#!/bin/sh -f

# Print system information
uname -a

# Activate the conda environment
source /afs/cern.ch/work/b/beturk/private/snd/source_conda.sh

# Assign the experiment name to a variable
afs="/afs/cern.ch/work/b/beturk/private/snd/dl_general/dl_cls_v9"
eos="/eos/user/b/beturk/snd/dl_general/dl_cls_v9"
name="resnet_v6_TBHadrons23_MCElectrons_2layer_samexy_cut_ideal"

# Execute the Python script, passing the variable name as an argument#
cp "$afs/$name"/* "$eos/$name"/
python "$afs/$name/main.py" "$afs/$name" "$eos/$name"

# List files in the current directory with detailed info
#ls -lh

# Exit the script
#exit 0
