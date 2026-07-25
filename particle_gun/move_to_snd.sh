PASTE_DEST="/eos/experiment/sndlhc/MonteCarlo/testbeam2024/2mmRangeCut/300GeV_11/nominal_entry_points"
COPY_SRC="/eos/experiment/sndlhc/users/beturk/TB/TB_MC_New/2024/300GeV_11_W"  # Change "smt" to your actual source directory path

for dir_path in "$COPY_SRC"/X_neg*/; do
    # Extract the original folder name
    folder_name=$(basename "$dir_path")
    
    # Remove all hyphens (-) from the folder name
    clean_folder_name="${folder_name//-/}"
    
    echo "Processing original: $folder_name -> Target: $clean_folder_name"
    
    # Create the new folder WITHOUT the hyphen
    mkdir -p "$PASTE_DEST/$clean_folder_name"
    
    # Copy the .root files into the new, clean folder
    cp "$dir_path"/*TGeant4.root "$PASTE_DEST/$clean_folder_name/"
    cp "$dir_path"/*TGeant4_MCEB.root "$PASTE_DEST/$clean_folder_name/"
done

echo "✅ All files copied and renamed successfully!"