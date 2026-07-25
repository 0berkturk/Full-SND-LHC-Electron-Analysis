import ROOT
import glob
import SndlhcGeo # Load dictionaries just in case

pair=4840

number = "1000"

energy="250"

files = glob.glob(f"/eos/experiment/sndlhc/users/beturk/TB/TB_MC_New/2024/{energy}GeV_11_W/*sndLHC.PG_11-TGeant4_MCEB_digCPP.root")
print(f"🔍 Scanning {len(files)} files for empty lists...\n")

total=0
for f_name in files:
    f = ROOT.TFile.Open(f_name, "READ")
    
    # Catch completely broken files
    if not f or f.IsZombie():
        #print(f"💀 CORRUPT FILE FOUND: {f_name}")
        continue

    # Loop through every object inside the ROOT file
    for key in f.GetListOfKeys():
        obj = key.ReadObj()
        
        # Check if the object is a list or collection (TList, TObjArray, etc.)
        if obj.InheritsFrom("TCollection"):
            if obj.GetEntries() == 0:
                pass
                #print(f"⚠️ Empty list named '{key.GetName()}' found in -> {f_name}")
    #print("f",f)
    tree = f.Get("cbmsim")
    N=tree.GetEntries()
    #print(N)
    total+=N
    f.Close()
    print("total",total)
    print("✅ Scan complete.\n")