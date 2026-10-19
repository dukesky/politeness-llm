#!/usr/bin/env bash
# B7 HEAD-BIAS check (EXPLORATORY; evaluator 2026-10-19 sec. 5 item 2): is the label-level NDCG cost an
# artifact of the regression head-bias init (1.5)?  Retrains v4-flash minilm-raw SHIFT_FULL and HUMAN with
# head bias = training-label mean (tag "hb" -> label_set <SET>@hb), 3 seeds, registered recipe otherwise;
# evaluates on dl22, runs the paired compares and the summary.  Usage on Colab: bash scripts/b7_headbias.sh
set -u
cd "$(dirname "$0")/.."
: "${DATA_DIR:?DATA_DIR not set}"
MODELS=${MODELS:-/content/batch7_models}
ARM=${ARM:-minilm-raw}; TAG=${TAG:-hb}; SLUGS=${SLUGS:-deepseek__deepseek-v4-flash}; HB_SETS=${HB_SETS:-"SHIFT_FULL HUMAN"}
EP=${EP:-5}; LR=${LR:-3e-5}
for SLUG in $SLUGS; do
E="$DATA_DIR/derived/batch7/eval/$ARM/$SLUG"; T="$DATA_DIR/derived/batch7/train_logs/$ARM/$SLUG"
mkdir -p "$E" "$T"
for SET in $HB_SETS; do
 L="$DATA_DIR/derived/batch7/$SLUG/labels_$SET.jsonl"
 [ -f "$L" ] || { echo "ERROR: $L absent"; exit 1; }
 for SEED in 1 2 3; do
  N="${SET}@${TAG}_s$SEED"; M="$MODELS/$ARM/$SLUG/$N"; O="$E/${N}_dl22.json"
  if [ ! -f "$M/train_log.json" ]; then
   python scripts/batch7_distill.py train --arm $ARM --labels "$L" --out "$M" --seed $SEED --epochs $EP --lr $LR \
     --bs 32 --max-len 256 --objective regression --head-bias label_mean --tag $TAG \
     > /tmp/b7_hb_train_last.txt 2>&1 || { tail -30 /tmp/b7_hb_train_last.txt; exit 1; }
   grep "head bias =" /tmp/b7_hb_train_last.txt
  fi
  cp "$M/train_log.json" "$T/$N.json"
  [ -f "$O" ] || python scripts/batch7_distill.py evaluate --model-dir "$M" --dataset dl22 --unseal-dl22 --out "$O" \
     > /tmp/b7_eval_last.txt 2>&1 || { tail -20 /tmp/b7_eval_last.txt; exit 1; }
  echo "$(date +%H:%M) done $N"
 done
done
s() { echo "$E/$1_s1_dl22.json,$E/$1_s2_dl22.json,$E/$1_s3_dl22.json"; }
c() { [ -f "$E/$1_s1_dl22.json" ] && [ -f "$E/$2_s1_dl22.json" ] || { echo "NOTE: $1 or $2 absent"; return; }
      python scripts/batch7_distill.py compare --a "$(s $1)" --b "$(s $2)" --out "$E/compare_$1_vs_$2.json" > "$E/compare_$1_vs_$2.txt" 2>&1 || tail -10 "$E/compare_$1_vs_$2.txt"; }
for SET in $HB_SETS; do c $SET@$TAG $SET; c $SET@$TAG RAW; [ "$SET" = HUMAN ] || c $SET@$TAG HUMAN; done
done
python scripts/batch7_summary.py --eval-root "$DATA_DIR/derived/batch7/eval" --csv-dir "$DATA_DIR/derived/batch7" > "$DATA_DIR/derived/batch7/b7_summary_${TAG}.txt" 2>&1; echo SUMMARY_EXIT=$?
sed -n '/ABLATION/,$p' "$DATA_DIR/derived/batch7/b7_summary_${TAG}.txt"
echo HB_DONE
