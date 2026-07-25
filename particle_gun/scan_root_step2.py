import ROOT
import glob
import SndlhcGeo # Load dictionaries just in case


x_dict={
    "x_360":[-37.93, -37.93, -39.97 , -39.97],
    "x_600":[-37.93, -38.95, -38.95 , -39.97 ],
    "x_1000":[-38.95]}
y_dict={
    "y_360":[41.74 , 44.50, 41.74,  44.50 ],
    "y_600":[43.12, 41.74, 44.50 , 43.12 ],
    "y_1000":[ 43.12]}

number = "360"

energy="250"

x=x_dict.get(f"x_{number}")
y=y_dict.get(f"y_{number}")

for i in range(len(x)):
    print(f"/eos/experiment/sndlhc/users/beturk/TB/TB_MC_New/2024/{energy}GeV_11_W/X_neg{x[i]:.2f}_Y_{y[i]:.2f}_Z_315/sndLHC.PG_11-TGeant4_MCEB_digCPP.root")
    files = glob.glob(f"/eos/experiment/sndlhc/users/beturk/TB/TB_MC_New/2024/{energy}GeV_11_W/X_neg{x[i]:.2f}_Y_{y[i]:.2f}_Z_315/sndLHC.PG_11-TGeant4_MCEB_digCPP.root")
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