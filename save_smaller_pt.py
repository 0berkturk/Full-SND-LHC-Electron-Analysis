import uproot

# Replace with the path to your ROOT file
file_path = "/afs/cern.ch/work/b/beturk/private/snd/sndLHC.Ntuple-TGeant4_digCPP.root"

# Open the ROOT file
with uproot.open(file_path) as file:

    print("=== ROOT file contents ===",file.keys())
    obj= file["TimeBasedBranchList;1"]

    # List all keys (objects) at the top level
    for key in file.keys():
        obj = file[key]
        print(f"\nObject: {key}")
        #print(f"Type: {obj.classname}")



        # If the object is a TTree, print its branches
        if isinstance(obj, uproot.models.TTree.Model_TTree):
            print("Branches:")
            for branch_name in obj.keys():
                print(f"  {branch_name}")
