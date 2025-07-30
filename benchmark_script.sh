#!/usr/bin/env bash
set -euo pipefail

# --- Configuration générale ---
dims=(256 512 1024 2048 4096)
logqs=(20 20 30 30 30)
beta=70
blocksize=78
num_seeds=10

run_benchmarks() {
  local -n _dims=$1
  local -n _logqs=$2

  # on génère les seeds 0,1,...,num_seeds-1
  local seeds=($(seq 0 $(( num_seeds - 1 ))))

  for i in "${!_dims[@]}"; do
    dim=${_dims[$i]}
    logq=${_logqs[$i]}
    halfdim=$(( dim / 2 ))

    for seed in "${seeds[@]}"; do

      # 1) LLL
      outfile="out_dim${dim}_logq${logq}_lll_hkz_off_s${seed}.log"
      echo "-> seed=$seed  dim=$dim  logq=$logq  algo=LLL  HKZ=off"
      latticegen -randseed "$seed" q "$dim" "$halfdim" "$logq" p \
        | python src/blaster/app.py -pq -t 1 -l "$outfile"

      # 2) G6K (--hkz)
      if [[ "$dim" -eq "${_dims[0]}" ]]; then
      outfile="out_dim${dim}_logq${logq}_lll_hkz_on_beta_s${seed}.log"
      echo "-> seed=$seed  dim=$dim  logq=$logq  algo=LLL  HKZ=on_beta"
      latticegen -randseed "$seed" q "$dim" "$halfdim" "$logq" p \
        | python src/blaster/app.py -b "$beta" --hkz -pq -t 1 -l "$outfile"
      fi
      # 3) deepLLL-4 (–d 4)
      outfile="out_dim${dim}_logq${logq}_deep4_hkz_off_s${seed}.log"
      echo "-> seed=$seed  dim=$dim  logq=$logq  algo=deepLLL-4  HKZ=off"
      latticegen -randseed "$seed" q "$dim" "$halfdim" "$logq" p \
        | python src/blaster/app.py -d 4 -pq -t 1 -l "$outfile"

      if [[ "$dim" -eq "${_dims[0]}" ]]; then
      # 4) BKZ (–b beta) avec blocksize –S
      outfile="out_dim${dim}_logq${logq}_bkz_b${beta}_bs${blocksize}_s${seed}.log"
      echo "-> seed=$seed  dim=$dim  logq=$logq  algo=BKZ  β=$beta  bs=$blocksize"
      latticegen -randseed "$seed" q "$dim" "$halfdim" "$logq" p \
        | python src/blaster/app.py -b "$beta" -pq -t 1 -S "$blocksize" -l "$outfile" -P 2
      fi
    done
  done
}

echo "=== Lancement des 4 algos :"
echo "   1) LLL (HKZ off)"
echo "   2) LLL (HKZ on_beta)"
echo "   3) deepLLL-4"
echo "   4) BKZ (β=${beta}, bs=${blocksize})"
run_benchmarks dims logqs
