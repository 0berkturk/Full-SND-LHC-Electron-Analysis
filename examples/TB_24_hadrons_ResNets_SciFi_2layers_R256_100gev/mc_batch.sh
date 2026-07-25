#!/bin/sh -f

# Print system information
uname -a
source /afs/cern.ch/work/b/beturk/private/snd/source_conda.sh
# Activate the conda environment
conda activate myenv
afs="/afs/cern.ch/work/b/beturk/private/snd/dl_general/dl_energy_v6"
eos="/eos/user/b/beturk/snd/dl_general/dl_energy_v6"
name="TB_24_hadrons_ResNets_SciFi_2layers_R256_100gev"

#cp "/afs/cern.ch/work/b/beturk/private/dl/ams/all_test_coatnet/run_all_ISS_test.sh" "$afs/$name"
#cp "/afs/cern.ch/work/b/beturk/private/dl/ams/all_test_coatnet/run_single_ISS_test.sh" "$afs/$name"
#cp "/afs/cern.ch/work/b/beturk/private/dl/ams/all_test_coatnet/run_all_MC_test.sh" "$afs/$name"
#cp "/afs/cern.ch/work/b/beturk/private/dl/ams/all_test_coatnet/run_single_MC_test.sh" "$afs/$name"
#cp "/afs/cern.ch/work/b/beturk/private/dl/ams/all_test_coatnet/plot_loss.py" "$afs/$name"

# Execute the Python script, passing the variable name as an argument#
python "$afs/$name/main.py" "$afs/$name" "$eos/$name"
#cd "$afs/$name"
#ls
#chmod +x run_all_ISS_test.sh
#./run_all_ISS_test.sh "$eos/$name"


# List files in the current directory with detailed info
#ls -lh

# Exit the script
