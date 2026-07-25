import ROOT
file_name = "/eos/experiment/sndlhc/MonteCarlo/NeutralHadrons/QGSP_BERT_HP_PEN/kaons/K_100_150/Ntuples/1/sndLHC.PG_130-TGeant4_digCPP.root"
file_mc = ROOT.TFile(file_name)#20240126_
tree_mc = file_mc.cbmsim
tree_mc.GetEntries()

#tree_mc.Print()
tree_mc.GetEvent(1) # get event at index 0
#tree_mc.EventHeader.GetEventNumber() # get its event number


# make the output easier to grasp - decoration is important!
from decorators import *
# Loop over the first 100 entries and dump them

# Print only first 10 tracks
mc_tracks = tree_mc.MCTrack
n_to_show = 20

#mc_tracks.Dump()
for i in range(min(n_to_show, mc_tracks.GetEntriesFast())):
    mc_tracks.At(i).Dump()
