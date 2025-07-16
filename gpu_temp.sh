#!/bin/bash

# Fichier de sortie
outfile="gpu_temp_log.csv"

# En-tête du fichier (écrit une fois)
echo "timestamp,temperature_C,used_VRAM" > "$outfile"

# Boucle infinie avec une mesure toutes les 5 secondes (modifiable)
while true; do
    timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -n 1)
    vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1)
    echo "$timestamp,$temp,$vram" >> "$outfile"
    sleep 1
done
