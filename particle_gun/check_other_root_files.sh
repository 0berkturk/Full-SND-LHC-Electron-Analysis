#!/bin/bash

ENERGY="180"
BASE_DIR="/eos/experiment/sndlhc/users/beturk/TB/TB_MC_New/2024/${ENERGY}GeV_211_W"

cd "$BASE_DIR" || { echo "❌ Cannot enter $BASE_DIR"; exit 1; }

TOTAL_EVENTS=0

echo "🚀 Starting sub-directory scan in $BASE_DIR"
echo "--------------------------------------------------"

# Loop through every batch folder directly (e.g., X_neg.../0/, X_neg.../part2_0/)
for sub_dir in X_neg*/*/; do
    # Remove trailing slash
    SUB_CLEAN=${sub_dir%/}
    
    DIGI_FILE="${SUB_CLEAN}/sndLHC.PG_11-TGeant4_MCEB_digCPP.root"
    SIM_FILE="${SUB_CLEAN}/sndLHC.PG_11-TGeant4.root"

    # 1. Check if the Digitized file exists
    if [ ! -f "$DIGI_FILE" ]; then
        echo "🗑️  Missing Digi: Deleting ONLY batch directory -> $SUB_CLEAN"
        rm -rf "$SUB_CLEAN"
        continue
    fi

    # 2. If it exists, count events in the SIM file
    if [ -f "$SIM_FILE" ]; then
        # Run ROOT to get entries
        EVENT_COUNT=$(root -l -b -q -e "TFile *f = TFile::Open(\"$SIM_FILE\"); if(!f || f->IsZombie()){cout<<0<<endl; exit(1);} TTree *t = (TTree*)f->Get(\"cbmsim\"); cout << (t ? t->GetEntries() : 0) << endl; f->Close();" | tail -n 1)
        
        # 3. Sum them up
        if [[ "$EVENT_COUNT" =~ ^[0-9]+$ ]]; then
            TOTAL_EVENTS=$((TOTAL_EVENTS + EVENT_COUNT))
            echo "✅ $SUB_CLEAN -> Events: $EVENT_COUNT"
        else
            echo "⚠️  $SUB_CLEAN -> Could not read $SIM_FILE"
        fi
    fi
done

cd /afs/cern.ch/work/b/beturk/private/snd/particle_gun_new_mc_v2
echo "--------------------------------------------------"
echo "📊 FINAL SUMMARY"
echo "✅ TOTAL VALID EVENTS FOR ${ENERGY}GeV: $TOTAL_EVENTS"
echo "--------------------------------------------------"