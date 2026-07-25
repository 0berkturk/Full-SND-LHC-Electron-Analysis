import glob
import os

def check_number_of_files(TB_BASE_DIR,particle=None, energy=None, year=None, label=None, num_files=None, fe_walls=None):
    if energy==None and year==None:
        pattern = TB_BASE_DIR
    else:
        if fe_walls is not None:
            pattern = f"{TB_BASE_DIR}/{year}/scifi_us_ds_{year}_{particle}_{energy}GeV_{fe_walls}_run_*.pt"
        else:
            pattern = f"{TB_BASE_DIR}/{year}/scifi_us_ds_{year}_{particle}_{energy}GeV_run_*.pt"            
    all_files = glob.glob(pattern)
    all_files.sort()
    print(len(all_files))

USED_FILES = set()

def register_manual_files(dataset_list):
    for item in dataset_list:
        label, filepath = item[0], item[1]
        USED_FILES.add(filepath)

def get_tb_files(TB_BASE_DIR,particle=None, energy=None, year=None, label=None, num_files=None, fe_walls=None):
    if energy==None and year==None:
        pattern = TB_BASE_DIR

    else:
        if fe_walls is not None:
            pattern = f"{TB_BASE_DIR}/{year}/scifi_us_ds_{year}_{particle}_{energy}GeV_{fe_walls}_run_*.pt"


        else:
            pattern = f"{TB_BASE_DIR}/{year}/scifi_us_ds_{year}_{particle}_{energy}GeV_run_*.pt"
            
    all_files = glob.glob(pattern)
    all_files.sort()
    
    clean_files = [f for f in all_files if f not in USED_FILES]
    
    if num_files is not None:
        selected_files = clean_files[:num_files]
    else:
        selected_files = clean_files
        
    if num_files is not None and len(selected_files) < num_files:
        print(f"WARNING: Requested {num_files} files for {particle} {energy}GeV (Fe: {fe_walls}), but only {len(selected_files)} clean files were left!")

    return [[label, f] for f in selected_files]


"""for energy in in [50,100,150,200,250,300]:
    TB_ELECTRONS_TEST = += get_tb_files("electrons", energy, 2024, label=1, fe_walls=None, num_files=1)
# Register Test files immediately!
register_manual_files(TB_ELECTRONS_TEST)"""

TB_BASE_DIR = "/eos/experiment/sndlhc/users/beturk/TB/TB_Data"

##### TEST FILES
TEST_FILE_2024_W_hadrons_50gev_electron=[]
TEST_FILE_2024_W_hadrons_50gev_electron += get_tb_files(TB_BASE_DIR,"electrons", 50, 2024, label=1, fe_walls=None, num_files=1)
for energy in [180]:
    TEST_FILE_2024_W_hadrons_50gev_electron += get_tb_files(TB_BASE_DIR,"pions", energy, 2024, label=0, fe_walls="W", num_files=1)
#print(TEST_FILE_2024_W_hadrons_50gev_electron)

TEST_FILE_2023_1fe_hadrons_50gev_electron = []
TEST_FILE_2023_1fe_hadrons_50gev_electron += get_tb_files(TB_BASE_DIR,"electrons", 50, 2024, label=1, fe_walls=None, num_files=1)
for energy in [100,140,180,240,300]:
    TEST_FILE_2023_1fe_hadrons_50gev_electron += get_tb_files(TB_BASE_DIR,"pions", energy, 2023, label=0, fe_walls="1Fe", num_files=1)

TEST_FILE_2023_2fe_hadrons_50gev_electron = []
TEST_FILE_2023_2fe_hadrons_50gev_electron  += get_tb_files(TB_BASE_DIR,"electrons", 50, 2024, label=1, fe_walls=None, num_files=1)
for energy in [100,140,180,240,300]:
    TEST_FILE_2023_2fe_hadrons_50gev_electron += get_tb_files(TB_BASE_DIR,"pions", energy, 2023, label=0, fe_walls="2Fe", num_files=1)

TEST_FILE_2023_3fe_hadrons_50gev_electron = []
TEST_FILE_2023_3fe_hadrons_50gev_electron  += get_tb_files(TB_BASE_DIR,"electrons", 50, 2024, label=1, fe_walls=None, num_files=1)
for energy in [100,140,180,240,300]:
    TEST_FILE_2023_3fe_hadrons_50gev_electron += get_tb_files(TB_BASE_DIR,"pions", energy, 2023, label=0, fe_walls="3Fe", num_files=1)


register_manual_files(TEST_FILE_2024_W_hadrons_50gev_electron)
register_manual_files(TEST_FILE_2023_1fe_hadrons_50gev_electron)
register_manual_files(TEST_FILE_2023_2fe_hadrons_50gev_electron)
register_manual_files(TEST_FILE_2023_3fe_hadrons_50gev_electron)
### END OF TEST FILES

# --- VALIDATION SET ---

VALIDATION_FILE = []
VALIDATION_FILE += get_tb_files("/eos/experiment/sndlhc/users/beturk/combined_new_MC_sparse_datasets/PG_MC_electron_1_50GeV__*.pt",label=1,num_files=2)
VALIDATION_FILE += get_tb_files(TB_BASE_DIR,"pions", 100, 2023, label=0, fe_walls="3Fe", num_files=1)
VALIDATION_FILE += get_tb_files(TB_BASE_DIR,"pions", 180, 2024, label=0, fe_walls="W", num_files=2)
register_manual_files(VALIDATION_FILE) # Register the manual MC files


# --- TRAINING SET ---
TRAINING_FILE = []
TRAINING_FILE += get_tb_files("/eos/experiment/sndlhc/users/beturk/combined_new_MC_sparse_datasets/PG_MC_electron_1_50GeV__*.pt",label=1,num_files=7)
"""for energy in [180]:
    TRAINING_FILE += get_tb_files(TB_BASE_DIR, "pions", energy, 2024, label=0, fe_walls="W", num_files=6)
    print("\ntungsten"," energy", energy)
    check_number_of_files(TB_BASE_DIR, "pions", energy, 2024, label=0, fe_walls="W", num_files=10000)
"""
for fe in ["1Fe","2Fe", "3Fe"]:
    for energy in [100,140,180,240, 300]:
        TRAINING_FILE += get_tb_files(TB_BASE_DIR,"pions", energy, 2023, label=0, fe_walls=fe, num_files=1)
        print("\nwall",fe," energy",energy)
        #check_number_of_files(TB_BASE_DIR,"pions", energy, 2023, label=0, fe_walls=fe, num_files=1000)


print("Training",TRAINING_FILE,"\n")
print("Validation",VALIDATION_FILE,"\n")
print("Test",TEST_FILE_2023_1fe_hadrons_50gev_electron,"\n")
print("Test",TEST_FILE_2023_2fe_hadrons_50gev_electron,"\n")
print("Test",TEST_FILE_2023_3fe_hadrons_50gev_electron,"\n")

