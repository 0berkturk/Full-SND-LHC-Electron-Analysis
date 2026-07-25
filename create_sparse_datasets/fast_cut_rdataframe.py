import ROOT
import math
import sys

# Dışarıdan argümanları al
YEAR = sys.argv[1]
BATCH_INDEX = int(sys.argv[2])
TOTAL_BATCHES = int(sys.argv[3])

tree_name = "rawConv" 
base_out_name = f"/eos/experiment/sndlhc/users/beturk/Data/ROOT/DATA_{YEAR}_noveto_nous_nods_scifi30"

with open(f"/afs/cern.ch/work/b/beturk/private/snd/sparse_datasets/filelist_{YEAR}.txt", "r") as f:
    file_list = [line.strip() for line in f if line.strip()]


total_files = len(file_list)

# Çoklu çekirdek aktif
ROOT.ROOT.EnableImplicitMT()

batch_size = math.ceil(total_files / TOTAL_BATCHES)

# Sadece bu job'a (BATCH_INDEX) ait olan başlangıç ve bitiş sınırlarını hesapla
start_idx = BATCH_INDEX * batch_size
end_idx = min((BATCH_INDEX + 1) * batch_size, total_files)

batch_files = file_list[start_idx:end_idx]
print(batch_files)
exit()
# Eğer bu batch'e dosya kalmadıysa işlemi sessizce bitir
if not batch_files:
    sys.exit(0)

out_name = f"{base_out_name}_batch_{BATCH_INDEX}.root"


df = ROOT.RDataFrame(tree_name, batch_files)

# Filtreleri uyguluyoruz
df_filtered = df.Filter("Digi_ScifiHits.GetEntries() > 30") \
                .Filter("Digi_MuFilterHits.GetEntries() == 0")

df_filtered.Snapshot(tree_name, out_name)

print(f"saved: {out_name}")