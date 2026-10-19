#!/usr/bin/env bash
# B7 ABLATION (exploratory): evaluate the ablation students on dl22 (same triple gate as the registered
# students), run the paired compares, and write the summary. Usage on Colab: bash scripts/b7_ablation_eval.sh
set -u
cd "$(dirname "$0")/.."
: "${DATA_DIR:?DATA_DIR not set}"
MODELS=${MODELS:-/content/batch7_models}
for ARM in minilm-raw msmarco-ce; do
 for SLUG in deepseek__deepseek-v4-flash google__gemini-3.5-flash; do
  E="$DATA_DIR/derived/batch7/eval/$ARM/$SLUG"; mkdir -p "$E"
  for SET in SHIFT_FULL SHIFT_n1000 DIAL_ONLY MAP_ONLY; do for SEED in 1 2 3; do
   M=$MODELS/$ARM/$SLUG/${SET}_s$SEED; O="$E/${SET}_s${SEED}_dl22.json"
   [ -d "$M" ] || { echo "NOTE: model absent $M"; continue; }
   [ -f "$O" ] && continue
   python scripts/batch7_distill.py evaluate --model-dir "$M" --dataset dl22 --unseal-dl22 --out "$O" > /tmp/b7_eval_last.txt 2>&1 || { tail -20 /tmp/b7_eval_last.txt; exit 1; }
  done; done
  s() { echo "$E/$1_s1_dl22.json,$E/$1_s2_dl22.json,$E/$1_s3_dl22.json"; }
  c() { [ -f "$E/$1_s1_dl22.json" ] && [ -f "$E/$2_s1_dl22.json" ] || { echo "NOTE: $1 or $2 absent"; return; }
        python scripts/batch7_distill.py compare --a "$(s $1)" --b "$(s $2)" --out "$E/compare_$1_vs_$2.json" > "$E/compare_$1_vs_$2.txt" 2>&1 || tail -10 "$E/compare_$1_vs_$2.txt"; }
  for A in SHIFT_FULL SHIFT_n1000 DIAL_ONLY MAP_ONLY; do c $A RAW; c $A HUMAN; done
  c SHIFT_FULL CORR_FROZEN; c DIAL_ONLY CORR_FROZEN; c MAP_ONLY CORR_FROZEN; c DIAL_ONLY MAP_ONLY
  echo "$(date +%H:%M) evaluated+compared $ARM/$SLUG"
 done
done
python scripts/batch7_summary.py --eval-root "$DATA_DIR/derived/batch7/eval" --csv-dir "$DATA_DIR/derived/batch7" > "$DATA_DIR/derived/batch7/b7_summary_ablation.txt" 2>&1; echo SUMMARY_EXIT=$?
sed -n '/ABLATION/,$p' "$DATA_DIR/derived/batch7/b7_summary_ablation.txt"
echo ABL_EVAL_DONE
