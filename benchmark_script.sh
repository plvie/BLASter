#!/usr/bin/env bash
set -euo pipefail

# config : dim, logq, beta, blocksize
dims=(128 256 512)
logqs=(10 20 30)
betas=(70 70 70)
blocks=(78 78 78)

# after with beta=70 and blocksize=72

# dims2=(128 256 512 1024)
# logqs2=(10 20 30 30)
# betas2=(70 70 70 70)
# blocks2=(72 72 72 72)


run_benchmarks() {
  local -n _dims=$1
  local -n _logqs=$2
  local -n _betas=$3
  local -n _blocks=$4
  local num_seeds=$5
  local -a seeds
  seeds=($(seq 0 $(( num_seeds - 1 ))))

  for i in "${!_dims[@]}"; do
    dim=${_dims[$i]}
    logq=${_logqs[$i]}
    beta=${_betas[$i]}
    block=${_blocks[$i]}
    halfdim=$(( dim / 2 ))

    for mode in "off" "on_beta" "on_norm"; do
      case $mode in
        off)
          hkz_flag=""
          hkz_label="off"
          block_for_run=$block
          ;;
        on_beta)
          hkz_flag="--hkz"
          hkz_label="on_beta"
          block_for_run=$beta
          ;;
        on_norm)
          hkz_flag="--hkz"
          hkz_label="on_norm"
          block_for_run=$block
          ;;
      esac

      for seed in "${seeds[@]}"; do
        outfile="out_dim${dim}_logq${logq}_b${beta}_bs${block_for_run}_hkz${hkz_label}_s${seed}.log"
        echo "-> seed=$seed  dim=$dim  logq=$logq  β=$beta  bs=$block_for_run  HKZ=$hkz_label"
        latticegen -randseed "$seed" q "$dim" "$halfdim" "$logq" p \
          | python src/blaster/app.py -b "$beta" $hkz_flag -pq -t 1 -S "$block_for_run" -l "$outfile"
      done
    done
  done
}


echo "=== beta=70, blocksize=72 ==="
run_benchmarks dims logqs betas blocks 10