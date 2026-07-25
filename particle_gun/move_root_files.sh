#!/bin/bash

# 1. Check if the user provided exactly 2 arguments
if [ "$#" -ne 2 ]; then
    echo "❌ Error: Missing arguments."
    echo "Usage: $0 <pattern> <directory_path>"
    echo "Example: $0 part4 50GeV_11_W/X_neg-38.95_Y_43.12_Z_315"
    exit 1
fi

PATTERN=$1
# Strip any trailing slash the user might accidentally add to keep paths clean
DIR_PATH=${2%/} 

# 2. Create the target extras directory (-p creates parent directories as needed)
echo "📁 Creating: extras/${DIR_PATH}/"
mkdir -p "/eos/experiment/sndlhc/users/beturk/TB/TB_MC_New/2024/extras/${DIR_PATH}/"

# 3. Copy the files/directories matching the pattern
# The * is outside the quotes so Bash can expand the wildcard
echo "⏳ Copying: ${DIR_PATH}/${PATTERN}* -> extras/${DIR_PATH}/"
mv "/eos/experiment/sndlhc/users/beturk/TB/TB_MC_New/2024/${DIR_PATH}/${PATTERN}"* "/eos/experiment/sndlhc/users/beturk/TB/TB_MC_New/2024/extras/${DIR_PATH}/"

# Check if the copy was successful
if [ $? -eq 0 ]; then
    echo "✅ Successfully copied files!"
else
    echo "❌ Error during copy. Check if the files exist."
fi