import ROOT
import glob
import SndlhcGeo  # This is the magic line that loads the SND dictionaries!

# 1. Grab all the digitized files from the subfolders
input_files = glob.glob("/eos/experiment/sndlhc/users/beturk/TB/TB_MC_New/2024/50GeV_11_W/X_neg-37.93_Y_41.74_Z_315/*/sndLHC.PG_11-TGeant4_MCEB_digCPP.root")
output_file = "/eos/experiment/sndlhc/users/beturk/TB/TB_MC_New/2024/50GeV_11_W/X_neg-37.93_Y_41.74_Z_315/merged_50GeV_electrons.root"

if len(input_files) == 0:
    print("❌ No files found! Check your path.")
    exit()

print(f"Found {len(input_files)} files. Teaching ROOT the dictionaries and merging...")

# 2. Use TFileMerger
merger = ROOT.TFileMerger(False)
merger.OutputFile(output_file)

for f in input_files:
    print(f)
    merger.AddFile(f)

# 3. Execute the merge (this automatically saves to the output_file)
status = merger.Merge()

if status:
    print(f"✅ Successfully merged into {output_file}")
    
    # 4. Open the saved file and check the event count
    check_file = ROOT.TFile.Open(output_file, "READ")
    tree = check_file.Get("cbmsim")
    
    if tree:
        n_events = tree.GetEntries()
        print(f"📊 Total number of events in merged file: {n_events}")
    else:
        print("⚠️ Warning: Could not find the 'cbmsim' tree to count events.")
        
    check_file.Close()
else:
    print("❌ Merge failed.")