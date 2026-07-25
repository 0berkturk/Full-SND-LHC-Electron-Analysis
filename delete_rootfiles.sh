#!/bin/bash

# Define the target parent directory (change this to your actual path)
TARGET_DIR="$1"

# Check if the parent directory actually exists before entering
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' does not exist."
    exit 1
fi

cd "$TARGET_DIR" || exit 1

# Loop through everything inside the directory
for item in *; do
    # Check if the item is a directory
    if [ -d "$item" ]; then
        echo "Cleaning inside directory: $item"
        
        # Remove the specific root files if they exist inside the directory
        rm -f "$item/sndLHC.PG_11-TGeant4_MCEB_dig.root"
        rm -f "$item/sndLHC.PG_11-TGeant4_MCEB.root"
    fi
done
cd /afs/cern.ch/work/b/beturk/private/snd
echo "Cleanup complete!"