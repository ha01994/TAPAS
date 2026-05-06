#!/usr/bin/env bash
set -euo pipefail

SRC="/shared/ha01994/alphafast_output_vdjdbiedb"
DST="/shared/ha01994/alphafast_output_vdjdbiedb_iptm_filtered"
CSV="iptm_filtered_vdjdbiedb.csv"

mkdir -p "$DST"

# 헤더가 있으면 사용(없으면 아래 블록의 tail -n +2를 cat으로 바꾸세요)
tail -n +2 "$CSV" | cut -d',' -f1 | sed 's/\r$//' | sort -u | while read -r id; do
  [[ -z "$id" ]] && continue
  if [[ -d "$SRC/$id" ]]; then
    echo "copy: $id"
    cp -ar "$SRC/$id" "$DST/"
  else
    echo "missing: $id" >&2
  fi
done
