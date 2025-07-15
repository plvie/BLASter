#!/usr/bin/env bash
set -euo pipefail

# Paramètres fixes
dim=512
halfdim=$(( dim / 2 ))
logq=30
beta=80
num_seeds=1

# Variants à tester pour -S
variants=(64 96 128 160 192)

# Deux modes : avec et sans --pnj
modes=(
  "--hkz --hkz-prog"           # mode 1 : classique
  "--hkz --hkz-prog --pnj"     # mode 2 : avec pnj
)

echo "=== Benchmarks: dim=$dim logq=$logq beta=$beta ==="
for flags in "${modes[@]}"; do
  # Label pour le nom du fichier
  if [[ "$flags" == *"--pnj"* ]]; then
    label="hkzprog_pnj"
  else
    label="hkzprog"
  fi

  echo "--- Mode: $label ---"
  for S in "${variants[@]}"; do
    echo ">>> Blocksize S=$S"
    for seed in $(seq 0 $(( num_seeds - 1 ))); do
      outfile="out_dim${dim}_logq${logq}_b${beta}_bs${S}_${label}_s${seed}.log"
      echo "-> seed=$seed  S=$S  flags=($flags)"
      latticegen -randseed "$seed" q "$dim" "$halfdim" "$logq" p \
        | python src/blaster/app.py -pq -b "$beta" -t 1 -P 2 $flags -S "$S" -l "$outfile"
    done
  done
done
