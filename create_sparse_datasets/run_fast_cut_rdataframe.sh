#!/bin/bash

# Argümanları al
YEAR=$1
BATCH_INDEX=$2
TOTAL_BATCHES=$3 # 3. argüman girilmezse varsayılan olarak 10 kabul eder

echo "Starting job for YEAR: $YEAR, BATCH: $BATCH_INDEX"
eval $(alienv load sndsw/latest --no-refresh)

# Condor node'larında ROOT ve ilgili kütüphanelerin çalışması için ortamı yükle
# (Kendi kullandığın SND@LHC setup yolunu buraya eklemelisin, örnek olarak:)
# eval "$(/cvmfs/sndlhc.cern.ch/SNDLHC-2024/Feb14/setup.sh)"

# Python scriptini argümanlarla çalıştır
python /afs/cern.ch/work/b/beturk/private/snd/sparse_datasets/fast_cut_rdataframe_copy.py $YEAR $BATCH_INDEX $TOTAL_BATCHES

echo "Job finished."