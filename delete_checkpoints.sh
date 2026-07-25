#!/bin/bash

initial_directory="$1"

for dir in "$initial_directory"/*/; do
    ckpt="$dir/checkpoints"

    # ---- CLAS ----
    latest_clas=$(ls "$ckpt"/clas*.pt 2>/dev/null | sort -V | tail -n 1)
    if [[ -n "$latest_clas" ]]; then
        find "$ckpt" -maxdepth 1 -type f -name "clas*.pt" ! -path "$latest_clas" -delete
    fi

    # ---- ECAL ----
    latest_ecal=$(ls "$ckpt"/model_ecal_*.pt 2>/dev/null | sort -V | tail -n 1)
    if [[ -n "$latest_ecal" ]]; then
        find "$ckpt" -maxdepth 1 -type f -name "model_ecal_*.pt" ! -path "$latest_ecal" -delete
    fi

    echo "Cleaned: $ls"
done
