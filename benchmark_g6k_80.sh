#!/usr/bin/env bash
set -euo pipefail

dim=512
halfdim=$(( dim / 2 ))
logq=30
beta=70 #for the moment i see that 74-76 crash at some point on 512
num_seeds=1

# a restart a partir d'ici
##>>> Blocksize S=128
##-> seed=0  S=128  flags=(--hkz --hkz-prog)

variants=(64 96 128 160 192)

modes=(
  "--hkz --hkz-prog"
  "--hkz --hkz-prog --pnj"
)

mkdir -p logs_errors

echo "=== Benchmarks: dim=$dim logq=$logq beta=$beta ==="
for flags in "${modes[@]}"; do
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
      outfilestdout="stdout_dim${dim}_logq${logq}_b${beta}_bs${S}_${label}_s${seed}.log"
      errfile="logs_errors/err_${outfile}.txt"
      echo "-> seed=$seed  S=$S  flags=($flags)"

    {
      latticegen -randseed "$seed" q "$dim" "$halfdim" "$logq" p \
        | python src/blaster/app.py -pq -b "$beta" -t 1 -P 2 $flags -S "$S" -l "$outfile"
    } > >(tee -a "$outfilestdout") 2> >(tee -a "$errfile" >&2) || {
      echo "$(date '+%Y-%m-%d %H:%M:%S') Crash at seed=$seed S=$S label=$label" | tee -a "$errfile"
    }
    done
  done
done
