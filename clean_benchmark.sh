#!/usr/bin/env bash
set -euo pipefail

SPARSE_DIR="cleaned"

OUT_ROOT="clean"
modes=(off on_beta on_norm)

mkdir -p "cleaned"

for m in "${modes[@]}"; do
  mkdir -p "${OUT_ROOT}/hkz_${m}"
done

for infile in out_dim*_*_*.log; do
  echo "Sparsifying $infile …"

  touch cleaned/"$infile"

  python slope-graphs-blaster/code/sparsify_csv.py "$infile"

  base=$(basename "$infile" .log)
  raw="${SPARSE_DIR}/${base}.log"
  csv="${SPARSE_DIR}/${base}.csv"

  # renomme en .csv
  if [[ -f "$raw" ]]; then
    mv "$raw" "$csv"
  else
    echo "pas de $raw" >&2
    continue
  fi
  raw_mode=${base#*_hkz}
  mode=${raw_mode%_s*}
  target_dir="${OUT_ROOT}/hkz_${mode}"

  target="${target_dir}/${base}.csv"
  mv "$csv" "$target"
  echo "déplacé -> $target"
done

echo
echo "all the CSV are in ${OUT_ROOT}/hkz_off/, hkz_on_beta/ and hkz_on_norm/."
