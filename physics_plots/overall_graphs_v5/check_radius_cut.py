import sys
import os
sys.path.append(os.path.abspath("../.."))
from test_beam_all_props_functions_v5 import *
import glob

def passes_shower_density_cut(idx_list, sig_list, radius=64, min_hits=15):
    """
    Finds the shower core and checks if there are strictly more than 
    'min_hits' inside the core +/- 'radius' for BOTH horizontal and vertical planes.
    Maintains a strict 2*radius window size even at the detector edges.
    """
    if len(idx_list) <= (min_hits * 2):
        return False

    idx_arr = np.array(idx_list, dtype=int)
    sig_arr = np.array(sig_list, dtype=np.float32)

    dense = np.zeros((2, 5, 1536), dtype=np.float32)
    dense[idx_arr[:, 0], idx_arr[:, 1], idx_arr[:, 2]] = sig_arr

    max_layer = np.argmax(np.sum(dense, axis=(0, 2)))
    max_hor = np.argmax(dense[0, max_layer, :])
    max_ver = np.argmax(dense[1, max_layer, :])

    # 1. Initial boundaries clamped to detector limits
    hor_start = max(0, max_hor - radius)
    hor_end   = min(1536, max_hor + radius)
    ver_start = max(0, max_ver - radius)
    ver_end   = min(1536, max_ver + radius)

    # 2. SHIFT LOGIC: Guarantee the window is exactly 2*radius wide
    if hor_start == 0:
        hor_end = 2 * radius
    elif hor_end == 1536:
        hor_start = 1536 - (2 * radius)

    if ver_start == 0:
        ver_end = 2 * radius
    elif ver_end == 1536:
        ver_start = 1536 - (2 * radius)

    # 3. Fast NumPy masking for both orientations
    hor_mask = (idx_arr[:, 0] == 0) & (idx_arr[:, 2] >= hor_start) & (idx_arr[:, 2] < hor_end)
    ver_mask = (idx_arr[:, 0] == 1) & (idx_arr[:, 2] >= ver_start) & (idx_arr[:, 2] < ver_end)
    
    hits_in_radius_h = np.sum(hor_mask)
    hits_in_radius_v = np.sum(ver_mask)
    print(hits_in_radius_h,hits_in_radius_v)
    return (hits_in_radius_h > min_hits) and (hits_in_radius_v > min_hits)

def load_as_lists(file_list):
    scifi_list = []
    us_list = []
    ds_hor_list = []
    ds_ver_list = []

    for fname in file_list:
        print(fname)
        data = torch.load(fname)
        scifi_idx = data["scifi_indices"][:1000]
        scifi_sig = data["scifi_signals"][:1000]
        for i in range(1000):
            print(passes_shower_density_cut(scifi_idx[i],scifi_sig[i]))

TEST_DATA_DIR_DATA=[]
TEST_DATA_DIR_DATA.append(glob.glob(f"/eos/experiment/sndlhc/users/beturk/TB_slight_cuts/TB_Data/2024/*electrons_50GeV*.pt")[0])
load_as_lists(TEST_DATA_DIR_DATA)