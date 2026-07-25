#!/bin/bash

# === 1. Environment Setup ===
#conda deactivate
source /cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/setUp.sh
export ALIBUILD_WORK_DIR=/afs/cern.ch/work/b/beturk/private/snd/sw
cd /afs/cern.ch/work/b/beturk/private/snd

#alienv enter sndsw/latest
# CRITICAL FIX: Use eval for Condor batch jobs, NOT alienv enter
eval $(alienv load sndsw/latest --no-refresh)

# === 2. Parameters ===
X=-45
Y=15
Z="$1"
PGrunID=1
PID=11
Estart=1
Eend=50
POS_X="$X"
POS_Y="$Y"
POS_Z="$Z"
NEVENTS=40

# === 3. Base output directory ===
BASE_OUTDIR="/eos/experiment/sndlhc/users/beturk/new_MC_sparse_datasets"
DIRNAME="single_PID${PID}_E${Estart}-${Eend}_Pos${POS_X}_${POS_Y}_${POS_Z}_N${NEVENTS}"
OUTDIR="${BASE_OUTDIR}/${DIRNAME}"
mkdir -p "$OUTDIR"

# === 4. Run Generation ===
python /afs/cern.ch/work/b/beturk/private/snd/particle_gun_new_mc_v2/PGsingle_run_simSND.py \
    --PG \
    --multiplePGSources \
    --PGrunID $PGrunID \
    --pID $PID \
    --Dx 35 \
    --Dy 37.5 \
    --Estart $Estart \
    --Eend $Eend \
    --EVx $POS_X \
    --EVy $POS_Y \
    --EVz $POS_Z \
    -n $NEVENTS \
    -o "$OUTDIR" 

# === 5. Run Digitization ===
echo "Running digitization..."
cd "$OUTDIR" || exit 1

# Find the actual files
SIMFILE=$(find . -maxdepth 1 -name "snd*.root" | head -n1)
GEOFILE=$(find . -maxdepth 1 -name "geo*.root" | head -n1)

if [ -z "$SIMFILE" ] || [ -z "$GEOFILE" ]; then
  echo "❌ Could not find input files in $OUTDIR"
  ls -la "$OUTDIR/"
  exit 1
fi

echo "✅ Using simulation file: $(realpath $SIMFILE)"
echo "✅ Using geometry file:   $(realpath $GEOFILE)"

# Run digitizer AND only delete the raw file if digitization succeeds
python $SNDSW_ROOT/shipLHC/run_digiSND.py \
    -f "$SIMFILE" \
    -g "$GEOFILE"

rm "$SIMFILE" && echo "✅ Raw simulation file deleted to save space."

cd /afs/cern.ch/work/b/beturk/private/snd/particle_gun_new_mc_v2/electron_gun_0_50