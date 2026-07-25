#!/bin/bash

# Configuration
ENERGY="300"
BASE_DIR="/eos/experiment/sndlhc/users/beturk/TB/TB_MC_New/2024/${ENERGY}GeV_11_W"

cd "$BASE_DIR" || { echo "❌ Cannot enter $BASE_DIR"; exit 1; }

GRAND_TOTAL_SIM=0
GRAND_TOTAL_MCEB=0

echo "🚀 Starting Merging and Counting in $BASE_DIR"
echo "--------------------------------------------------"

for x_dir in X_neg*/; do
    X_DIR_CLEAN=${x_dir%/}
    echo "📂 Processing Coordinate: $X_DIR_CLEAN"
    
    # Enter the directory to perform tasks safely
    cd "$X_DIR_CLEAN" || continue

    # 0. Copy one Geofile from subdirectories if not already present
    if [ ! -f "geofile_full.PG_11-TGeant4.root" ]; then
        # Find the first geofile in any subfolder (0, 1, etc.)
        GEO_SRC=$(find . -name "geofile_full.PG_11-TGeant4.root" | head -n 1)
        if [ -n "$GEO_SRC" ]; then
            echo "  📦 Copying Geofile from $GEO_SRC"
            cp "$GEO_SRC" .
        else
            echo "  ⚠️ No Geofile found in subdirectories!"
        fi
    fi

    PARAM_SRC=$(find . -name "ship.params.PG_11-TGeant4.root" | head -n 1)
    cp "$PARAM_SRC" .

    # 1. Merge the Simulation files

    

    echo "  🔗 Merging SIM files..."
    hadd -f sndLHC.PG_11-TGeant4.root */sndLHC.PG_11-TGeant4.root
    

    # 2. Merge the MCEB (Event Builder) files

    echo "  🔗 Merging MCEB files..."
    hadd -f sndLHC.PG_11-TGeant4_MCEB.root */sndLHC.PG_11-TGeant4_MCEB.root 


    # 3. Function to count events using ROOT
    count_events() {
        root -l -b -q -e "TFile *f = TFile::Open(\"$1\"); if(!f || f->IsZombie()){cout<<0<<endl; exit(1);} TTree *t = (TTree*)f->Get(\"cbmsim\"); cout << (t ? t->GetEntries() : 0) << endl; f->Close();" | tail -n 1
    }

    # 4. Count for both merged files
    SIM_COUNT=$(count_events "sndLHC.PG_11-TGeant4.root")
    MCEB_COUNT=$(count_events "sndLHC.PG_11-TGeant4_MCEB.root")

    echo "  📊 Results for $X_DIR_CLEAN:"
    echo "     - SIM Events:  $SIM_COUNT"
    echo "     - MCEB Events: $MCEB_COUNT"

    # Add to Grand Totals
    GRAND_TOTAL_SIM=$((GRAND_TOTAL_SIM + SIM_COUNT))
    GRAND_TOTAL_MCEB=$((GRAND_TOTAL_MCEB + MCEB_COUNT))

    # Go back up to the base directory for the next loop
    cd ..
done
cd /afs/cern.ch/work/b/beturk/private/snd/particle_gun_new_mc_v2
echo "--------------------------------------------------"
echo "🏆 FINAL GLOBAL SUMMARY"
echo "✅ TOTAL SIM PARTICLES:  $GRAND_TOTAL_SIM"
echo "✅ TOTAL MCEB PARTICLES: $GRAND_TOTAL_MCEB"
echo "--------------------------------------------------"