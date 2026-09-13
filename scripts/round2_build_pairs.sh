#!/usr/bin/env bash
# Round-2 pair building (run once per dataset, then FREEZE).
#
#   dl21 / dl22 : official scoreddocs top-50 ∩ NIST qrels, stratified cap 30/query
#   antique     : qrels-only (grades 1-4 raw), stratified cap 10/query (~2k pairs)
#
# Prints a SHA256 freeze fingerprint per file — record them in PREDICTIONS.md /
# the internal progress log. Re-running with the same seed is deterministic.
set -euo pipefail
: "${DATA_DIR:?set DATA_DIR first}"
mkdir -p "$DATA_DIR/inputs"

python -m src.build_pairs --dataset dl21 --source scoreddocs --top-k 50 \
    --max-per-query 30 --seed 42 --out "$DATA_DIR/inputs/pairs_dl21.jsonl"

python -m src.build_pairs --dataset dl22 --source scoreddocs --top-k 50 \
    --max-per-query 30 --seed 42 --out "$DATA_DIR/inputs/pairs_dl22.jsonl"

python -m src.build_pairs --dataset antique --source qrels \
    --max-per-query 10 --seed 42 --out "$DATA_DIR/inputs/pairs_antique.jsonl"

echo "== Round-2 pairs built. Freeze the three SHA256 fingerprints above. =="
