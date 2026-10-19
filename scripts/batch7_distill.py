"""Batch 7 / experiment B — DISTILLATION: does a student inherit the label set's
operating point, and does operating-point correction fix that without hurting
ranking?

Practitioner scenario
---------------------
No budget for human labels -> label a training pool with an LLM judge -> fine-
tune a cheap cross-encoder on those labels -> ship it.  The batch-4 framing
carries over unchanged: the correction removes a MEASUREMENT ARTIFACT (the
judge's operating point vs. the reference annotators'), it does not make the
judge "better".  The question is whether that artifact is INHERITED by a
student trained on the labels (the RAW student's score scale sits high/low as a
whole) and whether training on corrected labels pulls the student back onto
the human scale WITHOUT hurting its ranking.

Label sets (build-labels; one pair universe shared by every set)
----------------------------------------------------------------
  HUMAN             qrels grade — upper-bound reference.
  RAW               judge L3, paraphrase-mean then round-half-up.
  CORR_FROZEN       the frozen family rule from correction_rules.json (dial
                    level + monotone map, map applied per record BEFORE the
                    mean).  NOTE: that rule was fitted on dl19+dl20+dl21 human
                    labels, i.e. it is IN-SAMPLE on this training pool — it is
                    not an "imported" rule here.
  CORR_REFIT        FULL-BUDGET UPPER REFERENCE: query-disjoint 2-fold cross-
                    fit of dial + map within the training pool (batch-4 recipe:
                    md5 folds seed 42, best_level, 35 monotone maps); each
                    pair's label comes from the rule fitted on the OTHER fold.
                    Uses every training human label, so it is not a budget-
                    matched competitor of MIX.
  MIX_<n>           n human-labelled pairs (whole queries, seeded random order)
                    + RAW for the rest.  MIX_200 ⊂ MIX_1000 (nested).
  CORR_REFIT_n<n>   EQUAL-BUDGET correction: dial + map fitted ONLY on the human
                    queries MIX_<n> uses, then applied to ALL pairs (the MIX
                    queries included) -> 100% LLM-derived labels at the same
                    annotation budget as MIX_<n>.
The universe = human-judged pairs of the pairs files that carry a usable judge
label at EVERY politeness level L1-L5 (any rule may pick any dial level), so
students differ in their labels only.  A set whose n_h >= universe is skipped
with a note.

Arms (base models) — every train/eval output path carries an <arm>/ directory
------------------------------------------------------------------------------
  minilm-raw   nreimers/MiniLM-L6-H384-uncased    no ranking fine-tuning;
               defaults epochs=5 lr=3e-5
  msmarco-ce   cross-encoder/ms-marco-MiniLM-L6-v2 already an MS MARCO ranker;
               defaults epochs=2 lr=2e-5 (the old ``-L-6-`` id is an alias)
For BOTH arms and BOTH objectives the output head is re-initialised under the
seed (W ~ N(0, 0.02); bias 1.5 for regression, 0 for ordinal), so neither arm
starts from a pre-trained score scale.  The registered epochs/lr are whatever
`sweep` (HUMAN set, training-pool validation only) picks; train_log echoes them.

Subcommands
-----------
  build-labels  judgments.parquet -> labels_<SET>.jsonl + manifest.json
  build-ablation  EXPLORATORY (evaluator-requested, not pre-registered): reads
                the registered manifest + label files (never rewrites them)
                and adds, on the same universe, SHIFT_FULL / SHIFT_n1000
                (clip(RAW - Δ̂, 0, 3), continuous) and DIAL_ONLY (frozen dial,
                identity map) / MAP_ONLY (L3, frozen map per record) under
                manifest["ablation_sets"].  train/evaluate/compare take them
                like any set (labels_<SET>.jsonl; no set-name whitelist).
                Colab, after build-labels:
                  for J in <judges>; do python scripts/batch7_distill.py \
                    build-ablation --judge "$J"; done
                  then the cell-3/4/5 loops with
                  SETS="SHIFT_FULL SHIFT_n1000 DIAL_ONLY MAP_ONLY" and e.g.
                  c SHIFT_FULL RAW; c SHIFT_FULL CORR_FROZEN; c DIAL_ONLY RAW;
                  c MAP_ONLY RAW; c CORR_FROZEN DIAL_ONLY; c CORR_FROZEN MAP_ONLY
  sweep         per-arm hyper-parameter grid on the HUMAN set only (train split
                -> validation loss + val_pred_mean); never touches dl22
  train         fine-tune one student (plain PyTorch loop; AMP fp16 on CUDA;
                deterministic seeding; fixed epochs, default --select last;
                10% query-disjoint validation split from the TRAINING pool)
  evaluate      score every judged test pair in fp32 and report ranking + abs-
                olute endpoints with query-clustered bootstrap 90% CIs.  dl22
                is gated: the heading "# B7 预注册 — 蒸馏训练" must be in a
                clean, committed paper/PREDICTIONS.md AND --unseal-dl22 given.
  compare       paired-query bootstrap A - B (90% CI) for every endpoint + TOST
                statement for NDCG@10 (margin 0.01).  A side may list several
                seed files (side metric = mean over them; same resampled
                queries).  Base model / arm must match within and across sides
                (unless --allow-cross-arm); duplicate files are refused.
  selftest      CPU-only, offline, synthetic stub DATA_DIR + in-code tiny BERT.

Ranking convention (evaluate) — NOT the batch-5 condensed convention
---------------------------------------------------------------------
Per query, ALL judged pairs of the test universe (pairs file ∩ qrels) are
ranked by the student's UNCLIPPED score (descending; ties by docid).  NDCG@10:
gain 2^rel - 1 with the HUMAN grade, discount log2(rank + 1), ideal DCG from the
human grades of the same judged pool.  MAP: binary at human >= 2, R = number of
human-relevant pairs in the pool.  Queries with IDCG = 0 (R = 0) are excluded
from NDCG (MAP) and counted.  A RE-RANKING evaluation of a fixed judged pool —
not batch 5's condensed lists, not trec_eval on a full run.

Controls and noise
------------------
gemini-3.8-flash's frozen rule is identity@L3, so its CORR_FROZEN labels are
bit-identical to RAW and, with deterministic training, the two students should
be bit-identical: a PIPELINE-INVARIANCE check (a difference flags GPU non-
determinism or a bug), NOT a noise floor.  The noise floor is the across-seed
sd that `compare` prints per side.

Colab cell sequence (L4 GPU runtime; repo root as cwd)
------------------------------------------------------
Runtime figures below are ESTIMATES for an L4 (not measured).

    # -- cell 0: setup (new runtime) -------------------------------- ~3 min
    from google.colab import drive; drive.mount('/content/drive')
    %cd /content/politeness-llm
    !git pull
    !pip install -q -r requirements.txt -r requirements-batch7.txt
    import os
    os.environ["DATA_DIR"] = "/content/drive/MyDrive/Projects/research/politeness-llm"
    !python scripts/batch7_distill.py selftest                       # ~1 min

    # -- cell 1: build labels (CPU work, blind: no dl22) ------ ~1-3 min/judge
    # gemini-3.8-flash is a round-2 judge: if it has no dl19/dl20 rows,
    # build-labels refuses; re-run it with --train-datasets dl21 and record the
    # smaller pool in the B7 registration.
    %%bash
    for J in deepseek/deepseek-v4-flash google/gemini-3.5-flash google/gemini-3.8-flash; do
      python scripts/batch7_distill.py build-labels --judge "$J" \
        --train-datasets dl19,dl20,dl21 \
        --rules "$DATA_DIR/derived/correction_rules.json"
    done

    # -- cell 2: per-arm hyper-parameter sweep (HUMAN set only) --------------
    # 9 configs/arm; ~15-25 min (msmarco-ce) / ~20-30 min (minilm-raw)
    %%bash
    H="$DATA_DIR/derived/batch7/deepseek__deepseek-v4-flash/labels_HUMAN.jsonl"
    for ARM in minilm-raw msmarco-ce; do
      python scripts/batch7_distill.py sweep --arm $ARM --labels "$H" \
        --grid "epochs=3,5,8;lr=2e-5,3e-5,5e-5" \
        --out "$DATA_DIR/derived/batch7/sweep/$ARM/sweep.json"
    done
    # -> put the label means (cell 1) and the picked epochs/lr per arm into
    #    the "# B7 预注册 — 蒸馏训练" block, COMMIT, then continue.

    # -- cell 3: train 2 arms x 3 judges x 8 sets x 3 seeds = 144 students --
    # ~1-2 min/student (msmarco-ce, 2 ep) and ~3-4 min (minilm-raw, 5 ep) on
    # ~5.5k pairs -> roughly 5-7 h in total; resumable, run one arm per session
    # if needed.  Models stay on local disk (~90 MB each); logs go to Drive.
    %%bash
    declare -A EP=( [minilm-raw]=5 [msmarco-ce]=2 )        # <- registered sweep picks
    declare -A LR=( [minilm-raw]=3e-5 [msmarco-ce]=2e-5 )  # <- registered sweep picks
    SETS="HUMAN RAW CORR_FROZEN CORR_REFIT MIX_200 MIX_1000 CORR_REFIT_n200 CORR_REFIT_n1000"
    for ARM in minilm-raw msmarco-ce; do
     for SLUG in deepseek__deepseek-v4-flash google__gemini-3.5-flash google__gemini-3.8-flash; do
      for SET in $SETS; do
       L="$DATA_DIR/derived/batch7/$SLUG/labels_$SET.jsonl"
       [ -f "$L" ] || { echo "NOTE: $L absent (skipped at build-labels)"; continue; }
       for SEED in 1 2 3; do
        OUT=/content/batch7_models/$ARM/$SLUG/${SET}_s$SEED
        if [ -f $OUT/train_log.json ]; then echo "exists, not retrained: $OUT"; continue; fi
        python scripts/batch7_distill.py train --arm $ARM --labels "$L" --out $OUT \
          --seed $SEED --epochs ${EP[$ARM]} --lr ${LR[$ARM]} --bs 32 --max-len 256 \
          --objective regression || exit 1
        mkdir -p "$DATA_DIR/derived/batch7/train_logs/$ARM/$SLUG"
        cp $OUT/train_log.json "$DATA_DIR/derived/batch7/train_logs/$ARM/$SLUG/${SET}_s$SEED.json"
       done
      done
     done
    done
    # (ordinal-threshold variant: same loop with --objective ordinal-threshold
    #  and OUT=..._ord_s$SEED, for the sets the registration names.)

    # -- cell 4: evaluate on dl22 (ONLY after the B7 registration commit) ----
    # fp32 scoring, ~10-20 s/student -> ~30-45 min for 144
    %%bash
    for ARM in minilm-raw msmarco-ce; do
     for SLUG in deepseek__deepseek-v4-flash google__gemini-3.5-flash google__gemini-3.8-flash; do
      for M in /content/batch7_models/$ARM/$SLUG/*_s[123]; do
        python scripts/batch7_distill.py evaluate --model-dir $M \
          --dataset dl22 --unseal-dl22 \
          --out "$DATA_DIR/derived/batch7/eval/$ARM/$SLUG/$(basename $M)_dl22.json" || exit 1
      done
     done
    done

    # -- cell 5: paired comparisons, always WITHIN one arm (seconds each) ----
    %%bash
    for ARM in minilm-raw msmarco-ce; do
     for SLUG in deepseek__deepseek-v4-flash google__gemini-3.5-flash google__gemini-3.8-flash; do
      E="$DATA_DIR/derived/batch7/eval/$ARM/$SLUG"
      s() { echo "$E/$1_s1_dl22.json,$E/$1_s2_dl22.json,$E/$1_s3_dl22.json"; }
      c() { [ -f "$E/$1_s1_dl22.json" ] && [ -f "$E/$2_s1_dl22.json" ] || { echo "NOTE: $1 or $2 absent"; return; }
            python scripts/batch7_distill.py compare --a "$(s $1)" --b "$(s $2)" \
              --out "$E/compare_$1_vs_$2.json"; }
      for A in CORR_FROZEN CORR_REFIT MIX_200 MIX_1000 CORR_REFIT_n200 CORR_REFIT_n1000 HUMAN; do
        c $A RAW
      done
      c CORR_REFIT_n200 MIX_200          # equal-budget head-to-head
      c CORR_REFIT_n1000 MIX_1000
      c CORR_REFIT HUMAN
     done
    done

Requires: numpy, pandas, pyarrow (requirements.txt); torch + transformers for
sweep/train/evaluate scoring (requirements-batch7.txt).  build-labels and
compare need no torch.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

# Deterministic cuBLAS must be configured before torch initialises CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.qrels import (DATASETS, GRADE_OFFSET, SEALED,            # noqa: E402
                       assert_not_sealed, load_qrels, parse_datasets)
from batch5_downstream import judge_labels, round_half_up         # noqa: E402
import batch4_correction as b4                                     # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

B7_MARKER = "B7 预注册 — 蒸馏训练"
B7_HEADING = re.compile(r"^#{1,6}[ \t]*" + re.escape(B7_MARKER), re.M)
PREDICTIONS_PATH = REPO / "paper" / "PREDICTIONS.md"   # selftest patches this

ARMS: dict[str, dict] = {
    "minilm-raw": {"model": "nreimers/MiniLM-L6-H384-uncased",
                   "epochs": 5, "lr": 3e-5},
    "msmarco-ce": {"model": "cross-encoder/ms-marco-MiniLM-L6-v2",
                   "epochs": 2, "lr": 2e-5},
}
MODEL_ALIASES = {"cross-encoder/ms-marco-MiniLM-L-6-v2":
                 "cross-encoder/ms-marco-MiniLM-L6-v2"}
MODEL_TO_ARM = {v["model"]: k for k, v in ARMS.items()}

FAMILY_OF_DATASET = {"dl19": "msmarco", "dl20": "msmarco", "dl21": "msmarco",
                     "dl22": "msmarco", "antique": "antique"}
TRAIN_ALLOWED = ("dl19", "dl20", "dl21", "antique")   # dl22 NEVER trains
LEVELS = (1, 2, 3, 4, 5)
IDENTITY_MAP = (0, 1, 2, 3)
REL_THRESHOLD = 2          # TREC DL binary convention (human grade >= 2)
NDCG_K = 10
VAL_MOD = 10               # 1/10 of training queries -> validation
VAL_SALT = "b7val"
FOLD_SEED = 42             # batch-4 SEED; CORR_REFIT folds = batch-4 folds
MIX_SEED = 42
BOOT_B = 1000
BOOT_SEED = 42
TOST_MARGIN = 0.01
COVERAGE_WARN = 0.95       # universe / human-judged pairs
DELTA_TOL = 0.02           # |recomputed Delta - rule delta_hat|
CALIB_WARN = 0.25          # |val_pred_mean - val_label_mean| at selection
HEAD_STD = 0.02
HEAD_BIAS_REGRESSION = 1.5
HEAD_BIAS_DEFAULT = "1.5"      # train --head-bias default: the registered recipe
TAG_RE = re.compile(r"^[A-Za-z0-9-]+$")   # train --tag (EXPLORATORY variants)

SET_ROLES = {
    "HUMAN": "upper-bound reference: qrels grade",
    "RAW": "judge L3, paraphrase-mean, round-half-up",
    "CORR_FROZEN": ("frozen family rule (dial+map) from correction_rules.json; "
                    "IN-SAMPLE on this training pool (fitted on its human "
                    "labels), not an imported rule"),
    "CORR_REFIT": ("FULL-BUDGET UPPER REFERENCE: 2-fold query-disjoint cross-"
                   "fit of dial+map using every training human label"),
    "MIX": "n_h human pairs (whole queries, seeded) + RAW for the rest",
    "CORR_REFIT_n": ("EQUAL-BUDGET: dial+map fitted ONLY on MIX_<n>'s human "
                     "queries, applied to ALL pairs (100% LLM-derived)"),
}


# ═══════════════════════════════════════════════════════════════════════════
# Small helpers
# ═══════════════════════════════════════════════════════════════════════════

def resolve_data_dir(arg: str | None) -> Path:
    d = arg or os.environ.get("DATA_DIR")
    if not d:
        sys.exit("ERROR: pass --data-dir or set $DATA_DIR.")
    return Path(d)


def judge_slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, cwd=str(REPO),
            stderr=subprocess.DEVNULL).strip()
    except Exception:                                          # noqa: BLE001
        return "unknown"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def jsonl_bytes(rows) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n"
                   for r in rows).encode("utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(jsonl_bytes(rows))


def val_fold(dataset: str, qid: str) -> bool:
    """Query-disjoint validation membership — fixed across sets and seeds."""
    h = hashlib.md5(f"{VAL_SALT}|{dataset}|{qid}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) % VAL_MOD == 0


def require_arm_in_path(path, arm: str, what: str) -> None:
    if arm not in Path(path).parts:
        sys.exit(f"ERROR: {what} path {path} has no '{arm}/' directory "
                 f"component. Every train/eval output lives under its arm so "
                 f"arms can never be mixed silently.")


def resolve_arm(arm: str | None, model: str | None):
    """-> (arm, model). Arm defaults from the base model and vice versa."""
    if model:
        model = MODEL_ALIASES.get(model, model)
    if arm is None and model is None:
        sys.exit("ERROR: pass --arm (one of %s) or --model." % sorted(ARMS))
    if arm is None:
        arm = MODEL_TO_ARM.get(model)
        if arm is None:
            sys.exit(f"ERROR: base model {model} is not a registered arm "
                     f"{sorted(ARMS)}; pass --arm <name> explicitly.")
    if model is None:
        if arm not in ARMS:
            sys.exit(f"ERROR: unknown arm {arm}; pass --model too.")
        model = ARMS[arm]["model"]
    if arm in ARMS and model != ARMS[arm]["model"]:
        sys.exit(f"ERROR: arm {arm} is defined as {ARMS[arm]['model']}, got "
                 f"--model {model}.")
    return arm, model


def arm_hparam(arm: str, name: str, given):
    """Explicit value wins; else the arm default; unknown arm -> required."""
    if given is not None:
        return given, False
    if arm in ARMS:
        return ARMS[arm][name], True
    sys.exit(f"ERROR: --{name} is required for the non-registered arm {arm}.")


def load_pairs_texts(path: Path, dataset: str) -> dict:
    """pairs_<ds>.jsonl -> {(ds, qid, docid): {query, passage, rel|None}}.

    The dataset comes from the CLI, not the file: round-1 pairs files carry no
    ``dataset`` field and bare integer qids collide across collections.
    """
    if not path.exists():
        sys.exit(f"ERROR: pairs file not found: {path}")
    out = {}
    for p in read_jsonl(path):
        if "dataset" in p and str(p["dataset"]) != dataset:
            sys.exit(f"ERROR: {path} has dataset={p['dataset']!r}, expected "
                     f"{dataset!r}.")
        out[(dataset, str(p["qid"]), str(p["docid"]))] = {
            "query": p["query"], "passage": p["passage"], "rel": p.get("rel")}
    return out


def qrels_from_pairs_rel(texts: dict, dataset: str) -> dict:
    """Human grades from the pairs file's raw ``rel`` field (build_pairs.py
    stores the RAW qrels grade), shifted with src.qrels.GRADE_OFFSET exactly as
    load_qrels does.  Round-1 pairs files have no ``rel`` -> abort."""
    off = GRADE_OFFSET.get(dataset, 0)
    out = {}
    for k, v in texts.items():
        if v["rel"] is None:
            sys.exit(f"ERROR: --qrels-source pairs-rel but the {dataset} pairs "
                     f"file has no 'rel' field (round-1 files don't). Use "
                     f"--qrels-source ir_datasets.")
        g = int(v["rel"]) + off
        if not 0 <= g <= 3:
            sys.exit(f"ERROR: {dataset} rel {v['rel']} -> {g} outside 0-3.")
        out[k] = g
    return out


def load_grades(datasets, texts: dict, source: str) -> dict:
    if source == "ir_datasets":
        return load_qrels(list(datasets))
    out = {}
    for ds in datasets:
        out.update(qrels_from_pairs_rel(
            {k: v for k, v in texts.items() if k[0] == ds}, ds))
    return out


def rel_disagreements(texts: dict, grades: dict, source: str) -> dict:
    """Per dataset: pairs whose pairs-file ``rel`` (offset-shifted) disagrees
    with the grade actually used.  Only informative for ir_datasets grades."""
    out: dict = {}
    for k, v in texts.items():
        if v["rel"] is None or k not in grades:
            continue
        e = out.setdefault(k[0], {"n_compared": 0, "n_disagree": 0})
        e["n_compared"] += 1
        if int(v["rel"]) + GRADE_OFFSET.get(k[0], 0) != int(grades[k]):
            e["n_disagree"] += 1
    return {"grade_source": source, "per_dataset": out}


# ═══════════════════════════════════════════════════════════════════════════
# Gates
# ═══════════════════════════════════════════════════════════════════════════

def file_committed(p: Path) -> tuple[bool, str]:
    p = Path(p).resolve()
    try:
        subprocess.run(["git", "-C", str(p.parent), "ls-files",
                        "--error-unmatch", p.name], check=True,
                       capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False, "is not tracked by git"
    st = subprocess.run(["git", "-C", str(p.parent), "status", "--porcelain",
                         "--", p.name], capture_output=True,
                        text=True).stdout.strip()
    if st:
        return False, f"has uncommitted changes ({st})"
    return True, ""


def require_registration(path: Path | None = None) -> None:
    p = Path(path) if path is not None else PREDICTIONS_PATH
    text = p.read_text(encoding="utf-8") if p.exists() else ""
    if not B7_HEADING.search(text):
        sys.exit(
            f"ERROR: {p} has no heading line '# {B7_MARKER}'.\nCommit the B7 "
            f"registration block FIRST — evaluating a student on the sealed "
            f"dl22 must never precede its registration.")
    ok, why = file_committed(p)
    if not ok:
        sys.exit(f"ERROR: {p} {why}. The registration must be COMMITTED (its "
                 f"commit timestamp is the freeze time) before dl22 is opened.")


def gate_test_dataset(dataset: str, unseal: bool) -> None:
    """dl22: heading in a committed file AND the flag, checked before reads."""
    if dataset in SEALED:
        require_registration()
        assert_not_sealed([dataset], unseal=unseal)


def guard_train_datasets(names: list[str]) -> None:
    bad = [n for n in names if n not in TRAIN_ALLOWED]
    if bad:
        sys.exit(f"ERROR: training datasets must be within {TRAIN_ALLOWED}; got "
                 f"{bad}. dl22 is the held-out test set and never trains, "
                 f"with or without --unseal-dl22.")


def guard_rules_file(rules_all: dict) -> None:
    used = [str(x) for x in rules_all.get("datasets_used", [])]
    bad = sorted(set(used) & set(SEALED))
    for model, fams in (rules_all.get("rules") or {}).items():
        for fam, r in (fams or {}).items():
            bad += [f"{model}/{fam}:{d}" for d in r.get("datasets", [])
                    if d in SEALED]
    if bad:
        sys.exit(f"ERROR: the correction rules were fitted on sealed data "
                 f"{bad}; refusing to build training labels from them.")


# ═══════════════════════════════════════════════════════════════════════════
# build-labels — core (injectable; selftest drives it)
# ═══════════════════════════════════════════════════════════════════════════

def qualified_labels(d: pd.DataFrame, datasets, level: int, mono=None) -> dict:
    """batch5.judge_labels per dataset, keys qualified as (ds, qid, docid)."""
    out = {}
    for ds in datasets:
        dd = d[d.dataset.astype(str) == ds]
        if dd.empty:
            continue
        for (q, doc), v in judge_labels(dd, level, mono=mono).items():
            out[(ds, q, doc)] = v
    return out


def independent_level_labels(d: pd.DataFrame, level: int) -> dict:
    """Independent pandas/numpy re-derivation of judge_labels (no mono):
    groupby-mean of parse_ok scores, then floor(x + 0.5) clipped to 0..3."""
    ok = d[(d.politeness_level == level) & d.parse_ok.astype(bool)
           & d.score.notna()]
    g = ok.groupby([ok.dataset.astype(str), ok.qid.astype(str),
                    ok.docid.astype(str)]).score.mean()
    return {k: int(np.clip(np.floor(v + 0.5), 0, 3)) for k, v in g.items()}


def is_noop_rule(rule: dict) -> bool:
    return (int(rule["dial_level"]) == 3
            and tuple(int(x) for x in rule["monotone_map"]) == IDENTITY_MAP)


def assert_noop_invariant(model: str, fam: str, rule: dict, raw: dict,
                          corr: dict) -> None:
    """identity@L3 MUST reproduce RAW bit-exactly; anything else is a bug."""
    if not is_noop_rule(rule):
        return
    bad = [k for k in set(raw) | set(corr) if raw.get(k) != corr.get(k)]
    if bad:
        sys.exit(
            f"ERROR: no-op invariant VIOLATED for {model} ({fam}): the frozen "
            f"rule is identity@L3, so CORR_FROZEN must equal RAW bit-exactly, "
            f"but {len(bad)} pair(s) differ (e.g. {sorted(bad)[:3]}). This is a "
            f"BUG in label construction, not a result. No labels written.")


def delta_check(d: pd.DataFrame, grades: dict, rule: dict, datasets) -> dict:
    """Independent numpy Delta = mean(L3 ok scores) - mean(human over matched
    rows) vs the rule's delta_hat (batch-4 Step-A convention)."""
    hum = np.array([grades.get((str(a), str(b), str(c)), np.nan)
                    for a, b, c in zip(d.dataset, d.qid, d.docid)], dtype=float)
    m = np.isfinite(hum)
    ok = (m & d.parse_ok.astype(bool).to_numpy()
          & d.score.notna().to_numpy()
          & (d.politeness_level.to_numpy() == 3))
    delta = float(d.score.to_numpy(dtype=float)[ok].mean() - hum[m].mean())
    out = {"delta_recomputed": delta, "delta_hat_rule": rule.get("delta_hat"),
           "rule_datasets": rule.get("datasets"), "train_datasets": list(datasets)}
    if rule.get("delta_hat") is None:
        out["status"] = "rule has no delta_hat — not checked"
    else:
        diff = abs(delta - float(rule["delta_hat"]))
        out["abs_diff"] = diff
        same_pool = sorted(rule.get("datasets") or []) == sorted(datasets)
        if diff <= DELTA_TOL:
            out["status"] = "ok"
        elif same_pool:
            out["status"] = (f"WARNING: |Delta - delta_hat| = {diff:.4f} > "
                             f"{DELTA_TOL} on the SAME pool the rule was "
                             f"fitted on — check the parquet / qrels")
        else:
            out["status"] = (f"note: |Delta - delta_hat| = {diff:.4f}; pools "
                             f"differ, a gap is expected")
    return out


def pack_train(d: pd.DataFrame, grades: dict, model: str, fold_seed: int):
    """batch4.pack + the (ds, qid) of each packed row (same row order: pack
    copies, attaches human and dropna's without reordering)."""
    hum = [grades.get((str(a), str(b), str(c)))
           for a, b, c in zip(d.dataset, d.qid, d.docid)]
    keep = np.array([h is not None for h in hum])
    P = b4.pack(d, grades, model, "b7-train", fold_seed)
    rows_q = [(str(a), str(b)) for a, b in
              zip(d.dataset[keep], d.qid[keep])]
    if len(rows_q) != len(P.h):
        sys.exit("ERROR: internal row alignment with batch4.pack failed.")
    return P, rows_q


def fit_rule(P, idx: np.ndarray, what: str) -> dict:
    est = b4.delta_D(P, idx, strict=False)
    if est is None:
        sys.exit(f"ERROR: {what}: the fitting rows lack a politeness level "
                 f"L1-L5; cannot fit the dial.")
    delta, D, _m, _q = est
    dial = b4.best_level(delta, D)
    mono, kfit = b4.fit_monotone_map(P, idx, dial)
    return {"dial_level": int(dial), "monotone_map": [int(x) for x in mono],
            "delta_hat": float(delta),
            "kappa_fit": (float(kfit) if np.isfinite(kfit) else None)}


def refit_corr_labels(d: pd.DataFrame, grades: dict, model: str, datasets,
                      fold_seed: int = FOLD_SEED, packed=None):
    """CORR_REFIT: query-disjoint 2-fold cross-fit of dial + monotone map.

    Batch-4 recipe verbatim (pack -> md5 folds -> delta_D -> best_level ->
    fit_monotone_map at the dial level).  The rule fitted on fold A labels
    fold B's pairs and vice versa, so no pair's label used its own human grade.
    """
    dd = d[d.dataset.astype(str).isin(list(datasets))]
    P, _rq = packed if packed is not None else pack_train(dd, grades, model,
                                                         fold_seed)
    rules, labels = [], {}
    for f in (0, 1):
        fit_idx = np.where(P.fold != f)[0]
        if fit_idx.size == 0:
            sys.exit(f"ERROR: {model}: refit fold {1 - f} is empty — too few "
                     f"training queries to cross-fit.")
        r = fit_rule(P, fit_idx, f"{model} refit fold {1 - f}")
        r.update({"labels_fold": f, "fitted_on_fold": 1 - f})
        rules.append(r)
        mask = np.fromiter(
            (b4.fold_of(ds, str(q), fold_seed) == f
             for ds, q in zip(dd.dataset.astype(str), dd.qid.astype(str))),
            dtype=bool, count=len(dd))
        labels.update(qualified_labels(dd[mask], datasets, r["dial_level"],
                                       mono=tuple(r["monotone_map"])))
    return labels, rules


def draw_mix(universe_keys, n_h: int, seed: int):
    """Whole queries, in seeded random order, until >= n_h human pairs.

    Query-disjoint by construction (a query is all-human or all-RAW); the
    actual count overshoots n_h by < one query's pairs and is reported.  The
    order depends only on (universe queries, seed), so MIX_200 ⊂ MIX_1000.
    Returns (None, None) when n_h >= universe (the set is skipped).
    """
    if n_h >= len(universe_keys):
        return None, None
    by_q = defaultdict(list)
    for k in universe_keys:
        by_q[(k[0], k[1])].append(k)
    queries = sorted(by_q)
    order = np.random.default_rng(seed).permutation(len(queries))
    chosen, picked_q = set(), []
    for i in order:
        q = queries[int(i)]
        chosen.update(by_q[q])
        picked_q.append(q)
        if len(chosen) >= n_h:
            break
    return chosen, picked_q


def build_label_sets(d: pd.DataFrame, grades: dict, texts: dict, model: str,
                     rules_for_model: dict, datasets, mix_sizes,
                     mix_seed: int = MIX_SEED,
                     fold_seed: int = FOLD_SEED) -> tuple[dict, dict]:
    """Returns ({set: {key: (label, source)}}, info)."""
    d = d[d.dataset.astype(str).isin(list(datasets))]
    lvl = {l: qualified_labels(d, datasets, l) for l in LEVELS}
    raw = lvl[3]

    # independent re-derivation of judge_labels at every level (hard-fail)
    for l in LEVELS:
        ind = independent_level_labels(d, l)
        if ind != lvl[l]:
            bad = [k for k in set(ind) | set(lvl[l])
                   if ind.get(k) != lvl[l].get(k)]
            sys.exit(f"ERROR: independent recomputation of L{l} labels "
                     f"disagrees with batch5.judge_labels on {len(bad)} pairs "
                     f"(e.g. {sorted(bad)[:3]}). Bug, not a result.")

    frozen, frozen_info, dchecks = {}, {}, {}
    for fam in sorted({FAMILY_OF_DATASET[ds] for ds in datasets}):
        rule = rules_for_model.get(fam)
        if rule is None:
            sys.exit(f"ERROR: no frozen '{fam}' rule for {model} in the rules "
                     f"file.")
        fam_ds = [ds for ds in datasets if FAMILY_OF_DATASET[ds] == fam]
        lab = qualified_labels(d, fam_ds, int(rule["dial_level"]),
                               mono=tuple(int(x) for x in rule["monotone_map"]))
        raw_f = {k: v for k, v in raw.items() if k[0] in fam_ds}
        assert_noop_invariant(model, fam, rule, raw_f, lab)
        frozen.update(lab)
        frozen_info[fam] = {"dial_level": int(rule["dial_level"]),
                            "monotone_map": [int(x) for x in rule["monotone_map"]],
                            "fitted_on": rule.get("datasets"),
                            "is_noop": is_noop_rule(rule)}
        dchecks[fam] = delta_check(d[d.dataset.astype(str).isin(fam_ds)],
                                   grades, rule, fam_ds)

    packed = pack_train(d, grades, model, fold_seed)
    refit, refit_rules = refit_corr_labels(d, grades, model, datasets,
                                           fold_seed, packed=packed)

    human = {k: g for k, g in grades.items() if k[0] in datasets}
    judged = set(human) & set(texts)              # human-judged pairs pool
    all_lvl = set.intersection(*[set(v) for v in lvl.values()])
    universe = sorted(judged & all_lvl)
    if not universe:
        sys.exit(f"ERROR: {model}: empty training universe.")
    if not (set(universe) <= set(frozen) and set(universe) <= set(refit)):
        sys.exit("ERROR: internal: a universe pair lacks a CORR label.")
    coverage = len(universe) / max(len(judged), 1)
    drops = {
        "judged_pairs": len(judged),
        "judged_missing_some_level": len(judged - all_lvl),
        "judge_labelled_without_human_or_text": len(all_lvl - judged),
    }

    sets = {
        "HUMAN": {k: (float(human[k]), "human") for k in universe},
        "RAW": {k: (float(raw[k]), "raw") for k in universe},
        "CORR_FROZEN": {k: (float(frozen[k]), "corr_frozen") for k in universe},
        "CORR_REFIT": {k: (float(refit[k]), "corr_refit") for k in universe},
    }
    P, rows_q = packed
    mix_info, eq_info, notes = {}, {}, []
    for n_h in mix_sizes:
        n_h = int(n_h)
        chosen, picked_q = draw_mix(universe, n_h, mix_seed)
        if chosen is None:
            notes.append(f"MIX_{n_h} and CORR_REFIT_n{n_h} SKIPPED: n_h={n_h} "
                         f">= universe {len(universe)}")
            continue
        sets[f"MIX_{n_h}"] = {
            k: ((float(human[k]), "human") if k in chosen
                else (float(raw[k]), "raw")) for k in universe}
        qset = set(picked_q)
        mix_info[f"MIX_{n_h}"] = {
            "n_h_requested": n_h, "n_h_actual": len(chosen),
            "n_queries_human": len(picked_q), "seed": mix_seed,
            "human_queries": [f"{a}|{b}" for a, b in picked_q]}
        # equal-budget correction: fit ONLY on the MIX human queries
        fit_idx = np.array([i for i, q in enumerate(rows_q) if q in qset],
                           dtype=np.int64)
        r = fit_rule(P, fit_idx, f"{model} CORR_REFIT_n{n_h}")
        lab = qualified_labels(d, datasets, r["dial_level"],
                               mono=tuple(r["monotone_map"]))
        missing = [k for k in universe if k not in lab]
        if missing:
            sys.exit(f"ERROR: internal: CORR_REFIT_n{n_h} lacks {len(missing)} "
                     f"universe labels.")
        sets[f"CORR_REFIT_n{n_h}"] = {
            k: (float(lab[k]), f"corr_refit_n{n_h}") for k in universe}
        r.update({"fit_queries": [f"{a}|{b}" for a, b in sorted(qset)],
                  "n_fit_rows": int(fit_idx.size)})
        eq_info[f"CORR_REFIT_n{n_h}"] = r

    noop_all = all(v["is_noop"] for v in frozen_info.values())
    if noop_all:
        a = [sets["RAW"][k][0] for k in universe]
        b = [sets["CORR_FROZEN"][k][0] for k in universe]
        if a != b:
            sys.exit(f"ERROR: no-op invariant VIOLATED on the universe for "
                     f"{model}: CORR_FROZEN != RAW. Bug, not a result.")

    per_ds = defaultdict(int)
    for k in universe:
        per_ds[k[0]] += 1
    info = {
        "model": model,
        "datasets": list(datasets),
        "set_roles": SET_ROLES,
        "n_universe": len(universe),
        "n_queries": len({(k[0], k[1]) for k in universe}),
        "per_dataset_pairs": dict(per_ds),
        "coverage_of_human_judged": coverage,
        "coverage_warning": (coverage < COVERAGE_WARN),
        "drops": drops,
        "frozen_rule": frozen_info,
        "frozen_is_noop": noop_all,
        "delta_check": dchecks,
        "refit_rules": refit_rules,
        "equal_budget_rules": eq_info,
        "fold_seed": fold_seed,
        "mix": mix_info,
        "notes": notes,
    }
    return sets, info


def _summ(y: np.ndarray, h: np.ndarray) -> dict:
    return {"n": int(y.size), "mean": float(y.mean()),
            "offset_vs_human": float(y.mean() - h.mean()),
            "rel_rate_ge2": float((y >= REL_THRESHOLD).mean()),
            "hist": {str(g): int((np.floor(y + 0.5) == g).sum())
                     for g in range(4)}}


def set_summary(sets: dict) -> dict:
    out = {}
    keys = sorted(sets["HUMAN"])
    h = np.array([sets["HUMAN"][k][0] for k in keys])
    dsv = np.array([k[0] for k in keys])
    for name, lab in sets.items():
        y = np.array([lab[k][0] for k in keys])
        s = _summ(y, h)
        s["per_dataset"] = {ds: _summ(y[dsv == ds], h[dsv == ds])
                            for ds in sorted(set(dsv))}
        out[name] = s
    return out


def write_label_sets(out_dir: Path, sets: dict, texts: dict, info: dict,
                     extra: dict, force: bool = False) -> dict:
    blobs = {}
    for name, lab in sets.items():
        rows = []
        for k in sorted(lab):
            ds, q, doc = k
            t = texts[k]
            rows.append({"dataset": ds, "qid": q, "docid": doc,
                         "query": t["query"], "passage": t["passage"],
                         "label": lab[k][0], "source": lab[k][1], "set": name})
        blobs[name] = jsonl_bytes(rows)
    shas = {n: sha256_bytes(b) for n, b in blobs.items()}
    mpath = out_dir / "manifest.json"
    if mpath.exists() and not force:
        old = json.loads(mpath.read_text(encoding="utf-8"))
        old_shas = {n: f["sha256"] for n, f in old.get("files", {}).items()}
        if old_shas != shas:
            diff = sorted(n for n in set(old_shas) | set(shas)
                          if old_shas.get(n) != shas.get(n))
            sys.exit(f"ERROR: {mpath} exists and these label sets would "
                     f"CHANGE: {diff}. Students may already be trained on the "
                     f"old files; pass --force to overwrite deliberately.")
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, b in blobs.items():
        p = out_dir / f"labels_{name}.jsonl"
        p.write_bytes(b)
        files[name] = {"path": str(p), "sha256": shas[name]}
    manifest = dict(info)
    manifest.update(extra)
    manifest.update({
        "summary": set_summary(sets),
        "files": files,
        "code_git": git_hash(),
        "timestamp_utc": datetime.datetime.utcnow().isoformat(
            timespec="seconds") + "Z",
    })
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                     encoding="utf-8")
    return manifest


def print_manifest(m: dict) -> None:
    print(f"  universe: {m['n_universe']} pairs / {m['n_queries']} queries  "
          f"{m['per_dataset_pairs']}  coverage of human-judged "
          f"{m['coverage_of_human_judged']:.3f}")
    if m["coverage_warning"]:
        print(f"  WARNING: universe covers < {COVERAGE_WARN:.0%} of the human-"
              f"judged pairs — the student pool is materially smaller than "
              f"the judged pool.")
    print(f"  drops: {m['drops']}")
    rd = m.get("rel_disagreements", {})
    if rd:
        print(f"  pairs-file rel vs grades ({rd['grade_source']}): "
              f"{rd['per_dataset']}")
    for fam, r in m["frozen_rule"].items():
        print(f"  frozen {fam} rule: dial=L{r['dial_level']} "
              f"map={r['monotone_map']}{'  [NO-OP]' if r['is_noop'] else ''}"
              f"  (fitted on {r['fitted_on']}: IN-SAMPLE here)")
        dc = m["delta_check"][fam]
        print(f"    Delta recomputed {dc['delta_recomputed']:+.4f} vs rule "
              f"delta_hat {dc['delta_hat_rule']}: {dc['status']}")
    for r in m["refit_rules"]:
        print(f"  CORR_REFIT (labels fold {r['labels_fold']}, fit on fold "
              f"{r['fitted_on_fold']}): dial=L{r['dial_level']} "
              f"map={r['monotone_map']}  Delta_fit={r['delta_hat']:+.4f}")
    for k, v in m["mix"].items():
        print(f"  {k}: {v['n_h_actual']} human pairs from "
              f"{v['n_queries_human']} whole queries (seed {v['seed']})")
    for k, r in m["equal_budget_rules"].items():
        print(f"  {k}: fitted on {len(r['fit_queries'])} MIX queries "
              f"({r['n_fit_rows']} rows): dial=L{r['dial_level']} "
              f"map={r['monotone_map']}")
    for n in m["notes"]:
        print(f"  NOTE: {n}")
    print()
    print(f"  {'set':<17} {'n':>6} {'mean':>8} {'Δ vs HUMAN':>11} "
          f"{'rate>=2':>8}   hist 0/1/2/3   per-dataset Δ")
    for name, s in m["summary"].items():
        hist = "/".join(str(s["hist"][str(g)]) for g in range(4))
        pds = " ".join(f"{ds}:{v['offset_vs_human']:+.3f}"
                       for ds, v in s["per_dataset"].items())
        print(f"  {name:<17} {s['n']:>6} {s['mean']:>8.4f} "
              f"{s['offset_vs_human']:>+11.4f} {s['rel_rate_ge2']:>8.4f}   "
              f"{hist:<13}  {pds}")
    print("  (label mean = the operating point a student trained on that set "
          "inherits)")


def load_judge_frame(data_dir: Path, judge: str, datasets) -> pd.DataFrame:
    """One judge's judgments.parquet rows on ``datasets`` (flagship-scoped as
    in batch 4).  Shared by build-labels and build-ablation so both see the
    exact same records."""
    df = pd.read_parquet(data_dir / "derived" / "judgments.parquet")
    df = df[(df.model_id == judge)
            & df.dataset.astype(str).isin(datasets)]
    present = sorted(df.dataset.astype(str).unique())
    missing = [ds for ds in datasets if ds not in present]
    if missing:
        sys.exit(f"ERROR: {judge} has NO rows on {missing}. The training "
                 f"pool would silently differ from the requested one; re-run "
                 f"with --train-datasets {','.join(present) or '(none)'} and "
                 f"record that pool in the registration.")
    try:
        flagships = b4.flagship_model_ids(REPO / "config" / "models.yaml")
    except Exception:                                          # noqa: BLE001
        flagships = set()
    if judge in flagships:
        frozen_pairs, _p = b4.flagship_pairs_for(data_dir, datasets)
        df = b4.scope_frame(df, judge, datasets, frozen_pairs, True,
                            ",".join(datasets))
    return df


def cmd_build_labels(args) -> None:
    data_dir = resolve_data_dir(args.data_dir)
    datasets = parse_datasets(args.train_datasets)
    guard_train_datasets(datasets)
    rules_path = Path(args.rules) if args.rules else \
        data_dir / "derived" / "correction_rules.json"
    rules_all = json.loads(rules_path.read_text(encoding="utf-8"))
    guard_rules_file(rules_all)
    rules_for_model = (rules_all.get("rules", {}) or {}).get(args.judge)
    if not rules_for_model:
        sys.exit(f"ERROR: no rules for {args.judge} in {rules_path}.")

    df = load_judge_frame(data_dir, args.judge, datasets)

    pairs_dir = Path(args.pairs_dir) if args.pairs_dir else data_dir / "inputs"
    texts = {}
    for ds in datasets:
        texts.update(load_pairs_texts(pairs_dir / f"pairs_{ds}.jsonl", ds))
    grades = load_grades(datasets, texts, args.qrels_source)

    mix_sizes = [int(x) for x in str(args.mix_sizes).split(",") if x.strip()]
    sets, info = build_label_sets(df, grades, texts, args.judge,
                                  rules_for_model, datasets, mix_sizes,
                                  mix_seed=args.mix_seed,
                                  fold_seed=args.fold_seed)
    out_root = Path(args.out_dir) if args.out_dir else \
        data_dir / "derived" / "batch7"
    out_dir = out_root / judge_slug(args.judge)
    extra = {"rules_file": {"path": str(rules_path),
                            "sha256": sha256_file(rules_path),
                            "git_hash": rules_all.get("git_hash"),
                            "datasets_used": rules_all.get("datasets_used")},
             "rel_disagreements": rel_disagreements(texts, grades,
                                                    args.qrels_source)}
    m = write_label_sets(out_dir, sets, texts, info, extra, force=args.force)
    print("=" * 78)
    print(f"  BATCH 7 build-labels — {args.judge}   (train: {datasets})")
    print("=" * 78)
    print_manifest(m)
    print(f"\n  wrote {len(sets)} label files + manifest.json -> {out_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# build-ablation — EXPLORATORY (evaluator-requested, NOT pre-registered)
# ═══════════════════════════════════════════════════════════════════════════
# Four extra label sets on the SAME universe as the registered sets, built
# from the registered files + manifest (+ judgments.parquet for the per-record
# dial/map sets).  The registered files and manifest entries are never
# rewritten; the new sets live under manifest["ablation_sets"] /
# manifest["ablation_meta"] only.

ABLATION_ROLES = {
    "SHIFT_FULL": ("EXPLORATORY: clip(RAW - Δ̂_full, 0, 3), CONTINUOUS (not "
                   "rounded); Δ̂_full = mean(RAW) - mean(HUMAN) over the whole "
                   "training universe (label scale). Locus test: a continuous "
                   "constant shift on the labels, no rule granularity"),
    "SHIFT_n1000": ("EXPLORATORY: clip(RAW - delta_hat, 0, 3), CONTINUOUS; "
                    "delta_hat = the registered CORR_REFIT_n1000 rule's "
                    "delta_hat (record-level, fitted on MIX_1000's human "
                    "queries) — equal budget to CORR_REFIT_n1000"),
    "DIAL_ONLY": ("EXPLORATORY: the frozen rule's dial level with the IDENTITY "
                  "map; CORR_FROZEN code path (per family, paraphrase-mean, "
                  "round-half-up) with only the map replaced"),
    "MAP_ONLY": ("EXPLORATORY: L3 (default tone) records with the frozen map "
                 "applied PER RECORD before the mean (as CORR_FROZEN), then "
                 "round-half-up; CORR_FROZEN code path with only the dial "
                 "replaced by L3"),
}
ABLATION_SETS = tuple(ABLATION_ROLES)
ABLATION_SOURCE = {"SHIFT_FULL": "shift_full", "SHIFT_n1000": "shift_n1000",
                   "DIAL_ONLY": "dial_only", "MAP_ONLY": "map_only"}
ABLATION_DELTA_TOL = 1e-6      # recomputed Δ̂_full vs manifest summary


def frozen_family_labels(d: pd.DataFrame, datasets, frozen_rule: dict,
                         dial: int | None = None, mono=None) -> dict:
    """The CORR_FROZEN construction of build_label_sets (per family:
    qualified_labels at the rule's dial with the rule's map, map applied per
    record before the mean) with the dial and/or the map overridable.  With
    no override it must reproduce labels_CORR_FROZEN exactly (checked)."""
    out = {}
    for fam in sorted({FAMILY_OF_DATASET[ds] for ds in datasets}):
        r = frozen_rule.get(fam)
        if r is None:
            sys.exit(f"ERROR: manifest has no frozen '{fam}' rule.")
        fam_ds = [ds for ds in datasets if FAMILY_OF_DATASET[ds] == fam]
        lv = int(r["dial_level"]) if dial is None else int(dial)
        mp = (tuple(int(x) for x in r["monotone_map"]) if mono is None
              else tuple(int(x) for x in mono))
        out.update(qualified_labels(d, fam_ds, lv, mono=mp))
    return out


def ablation_diag(y: np.ndarray, h: np.ndarray, raw: np.ndarray,
                  dsv: np.ndarray, y_unclipped: np.ndarray | None = None
                  ) -> dict:
    """Label diagnostics the evaluator used: level, Δ vs HUMAN, rounded-grade
    histogram, share of RAW grade-1 pairs routed to (rounded) grade 0, label-
    human Pearson."""
    s = _summ(y, h)
    yr = np.floor(y + 0.5)
    r1 = raw == 1
    s["raw1_routed_to_0"] = (float((yr[r1] == 0).mean()) if r1.any()
                             else None)
    s["n_raw1"] = int(r1.sum())
    s["pearson_vs_human"] = (float(np.corrcoef(y, h)[0, 1])
                             if y.std() > 0 and h.std() > 0 else None)
    s["n_fractional"] = int((y != np.round(y)).sum())
    if y_unclipped is not None:
        s["mean_unclipped"] = float(y_unclipped.mean())
        s["offset_vs_human_unclipped"] = float(y_unclipped.mean() - h.mean())
        s["n_clipped"] = int((y != y_unclipped).sum())
    s["per_dataset_offset"] = {ds: float(y[dsv == ds].mean()
                                         - h[dsv == ds].mean())
                               for ds in sorted(set(dsv))}
    return s


def build_ablation_sets(d: pd.DataFrame, manifest: dict, reg: dict
                        ) -> tuple[dict, dict]:
    """reg = {registered SET: {key: label}} read from the label files.
    Returns ({ablation SET: {key: (label, source)}}, per-set info)."""
    model = manifest["model"]
    datasets = list(manifest["datasets"])
    frozen_rule = manifest["frozen_rule"]
    d = d[d.dataset.astype(str).isin(datasets)]
    universe = sorted(reg["HUMAN"])
    for n, lab in reg.items():
        if sorted(lab) != universe:
            sys.exit(f"ERROR: registered set {n} is not on the HUMAN universe.")

    # integrity: the parquet + CORR_FROZEN code path reproduce the files
    raw_re = qualified_labels(d, datasets, 3)
    fro_re = frozen_family_labels(d, datasets, frozen_rule)
    for nm, re_, f in (("RAW", raw_re, reg["RAW"]),
                       ("CORR_FROZEN", fro_re, reg["CORR_FROZEN"])):
        bad = [k for k in universe if k not in re_ or float(re_[k]) != f[k]]
        if bad:
            sys.exit(f"ERROR: {model}: recomputing {nm} from judgments.parquet "
                     f"disagrees with labels_{nm}.jsonl on {len(bad)} pairs "
                     f"(e.g. {bad[:3]}); the parquet or the code changed since "
                     f"build-labels. No ablation labels written.")

    h = np.array([reg["HUMAN"][k] for k in universe])
    raw = np.array([reg["RAW"][k] for k in universe])
    delta_full = float(raw.mean() - h.mean())
    man_off = float(manifest["summary"]["RAW"]["offset_vs_human"])
    if abs(delta_full - man_off) > ABLATION_DELTA_TOL:
        sys.exit(f"ERROR: Δ̂_full recomputed from the files {delta_full:+.8f} "
                 f"!= manifest RAW offset_vs_human {man_off:+.8f}.")

    sets, info, notes = {}, {}, []
    shifts = {"SHIFT_FULL": (delta_full, "mean(RAW) - mean(HUMAN), full "
                             "universe, label scale (manifest summary "
                             f"{man_off:+.8f})")}
    eb = (manifest.get("equal_budget_rules") or {}).get("CORR_REFIT_n1000")
    if eb is None or eb.get("delta_hat") is None:
        notes.append("SHIFT_n1000 SKIPPED: no registered CORR_REFIT_n1000 "
                     "rule with a delta_hat in the manifest")
    else:
        shifts["SHIFT_n1000"] = (float(eb["delta_hat"]),
                                 "registered CORR_REFIT_n1000 rule delta_hat "
                                 "(record level, MIX_1000 human queries)")
    unclipped = {}
    for name, (dh, why) in shifts.items():
        yu = raw - dh
        y = np.clip(yu, 0.0, 3.0)
        unclipped[name] = yu
        sets[name] = {k: (float(v), ABLATION_SOURCE[name])
                      for k, v in zip(universe, y)}
        info[name] = {"delta_hat": dh, "delta_source": why}

    for name, kw in (("DIAL_ONLY", {"mono": IDENTITY_MAP}),
                     ("MAP_ONLY", {"dial": 3})):
        lab = frozen_family_labels(d, datasets, frozen_rule, **kw)
        miss = [k for k in universe if k not in lab]
        if miss:
            sys.exit(f"ERROR: internal: {name} lacks {len(miss)} universe "
                     f"labels.")
        sets[name] = {k: (float(lab[k]), ABLATION_SOURCE[name])
                      for k in universe}
        info[name] = {"per_family": {
            fam: {"dial_level": (3 if name == "MAP_ONLY"
                                 else int(r["dial_level"])),
                  "monotone_map": (list(IDENTITY_MAP) if name == "DIAL_ONLY"
                                   else [int(x) for x in r["monotone_map"]])}
            for fam, r in frozen_rule.items()}}

    dsv = np.array([k[0] for k in universe])
    for name, lab in sets.items():
        y = np.array([lab[k][0] for k in universe])
        equals = [n for n in ("RAW", "CORR_FROZEN", "HUMAN")
                  if all(lab[k][0] == reg[n][k] for k in universe)]
        info[name].update({
            "role": ABLATION_ROLES[name],
            "summary": ablation_diag(y, h, raw, dsv, unclipped.get(name)),
            "equals_registered": equals,
            "degenerate": bool(equals),
        })
    return sets, {"sets": info, "notes": notes, "delta_full": delta_full}


def print_ablation_table(rows: list[tuple[str, dict, str]]) -> None:
    print(f"  {'set':<17} {'n':>6} {'mean':>8} {'Δ vs HUMAN':>11} "
          f"{'Δ unclip':>9} {'clip':>5} {'hist 0/1/2/3 (rounded)':<24}"
          f"{'RAW1→0':>7} {'r(h)':>6}  flag")
    for name, s, flag in rows:
        hist = "/".join(str(s["hist"][str(g)]) for g in range(4))
        un = s.get("offset_vs_human_unclipped")
        r10, pr = s.get("raw1_routed_to_0"), s.get("pearson_vs_human")
        print(f"  {name:<17} {s['n']:>6} {s['mean']:>8.4f} "
              f"{s['offset_vs_human']:>+11.4f} "
              f"{(f'{un:+.4f}' if un is not None else '-'):>9} "
              f"{s.get('n_clipped', '-')!s:>5} {hist:<24}"
              f"{(f'{r10:.3f}' if r10 is not None else '-'):>7} "
              f"{(f'{pr:.3f}' if pr is not None else '-'):>6}  {flag}")


def cmd_build_ablation(args) -> None:
    data_dir = resolve_data_dir(args.data_dir)
    out_root = Path(args.out_dir) if args.out_dir else \
        data_dir / "derived" / "batch7"
    jd = out_root / judge_slug(args.judge)
    mpath = jd / "manifest.json"
    if not mpath.exists():
        sys.exit(f"ERROR: {mpath} missing — run build-labels first.")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    if manifest.get("model") != args.judge:
        sys.exit(f"ERROR: {mpath} is for {manifest.get('model')}, not "
                 f"{args.judge}.")
    datasets = list(manifest["datasets"])
    guard_train_datasets(datasets)

    # registered files: verify against the manifest, load, remember bytes
    reg_files = manifest["files"]
    before = {}
    for n, f in reg_files.items():
        p = jd / f"labels_{n}.jsonl"
        before[n] = sha256_file(p)
        if before[n] != f["sha256"]:
            sys.exit(f"ERROR: {p} no longer matches its manifest sha256; "
                     f"refusing to build ablations on changed files.")
    reg_recs = {n: read_jsonl(jd / f"labels_{n}.jsonl")
                for n in ("HUMAN", "RAW", "CORR_FROZEN")}
    reg = {n: {(r["dataset"], r["qid"], r["docid"]): float(r["label"])
               for r in recs} for n, recs in reg_recs.items()}
    texts = {(r["dataset"], r["qid"], r["docid"]):
             {"query": r["query"], "passage": r["passage"]}
             for r in reg_recs["RAW"]}

    df = load_judge_frame(data_dir, args.judge, datasets)
    sets, ainfo = build_ablation_sets(df, manifest, reg)

    blobs = {}
    for name, lab in sets.items():
        rows = []
        for k in sorted(lab):
            ds, q, doc = k
            rows.append({"dataset": ds, "qid": q, "docid": doc,
                         "query": texts[k]["query"],
                         "passage": texts[k]["passage"],
                         "label": lab[k][0], "source": lab[k][1], "set": name})
        blobs[name] = jsonl_bytes(rows)
    shas = {n: sha256_bytes(b) for n, b in blobs.items()}
    old = manifest.get("ablation_sets") or {}
    changed = sorted(n for n in old if n in shas
                     and old[n].get("sha256") != shas[n])
    if changed and not args.force:
        sys.exit(f"ERROR: ablation sets {changed} would CHANGE; students may "
                 f"already be trained on them; pass --force to overwrite.")
    entries = {}
    for name, b in blobs.items():
        p = jd / f"labels_{name}.jsonl"
        p.write_bytes(b)
        entries[name] = dict(ainfo["sets"][name], path=str(p),
                             sha256=shas[name], source=ABLATION_SOURCE[name])
    after = {n: sha256_file(jd / f"labels_{n}.jsonl") for n in reg_files}
    if after != before:
        sys.exit("ERROR: internal: a registered label file changed.")

    keys = sorted(reg["HUMAN"])
    h = np.array([reg["HUMAN"][k] for k in keys])
    raw = np.array([reg["RAW"][k] for k in keys])
    dsv = np.array([k[0] for k in keys])
    ref = {}
    for n in reg_files:
        recs = reg_recs.get(n) or read_jsonl(jd / f"labels_{n}.jsonl")
        by_k = {(r["dataset"], r["qid"], r["docid"]): float(r["label"])
                for r in recs}
        y = np.array([by_k[k] for k in keys])
        ref[n] = ablation_diag(y, h, raw, dsv)
    fro = manifest["frozen_rule"]
    manifest["ablation_sets"] = entries          # existing keys untouched
    manifest["ablation_meta"] = {
        "status": "EXPLORATORY (evaluator-requested 2026-10-12, NOT "
                  "pre-registered); registered sets unchanged",
        "delta_full": ainfo["delta_full"],
        "frozen_rule_is_identity_at_L3": all(v["is_noop"]
                                             for v in fro.values()),
        "notes": ainfo["notes"],
        "reference_diagnostics": ref,
        "registered_sha256_verified": before,
        "code_git": git_hash(),
        "timestamp_utc": datetime.datetime.utcnow().isoformat(
            timespec="seconds") + "Z",
    }
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                     encoding="utf-8")

    print("=" * 78)
    print(f"  BATCH 7 build-ablation (EXPLORATORY) — {args.judge}   "
          f"(universe {len(keys)} pairs, train {datasets})")
    print("=" * 78)
    for fam, r in fro.items():
        print(f"  frozen {fam} rule: dial=L{r['dial_level']} "
              f"map={r['monotone_map']}{'  [NO-OP]' if r['is_noop'] else ''}")
    for name in ("SHIFT_FULL", "SHIFT_n1000"):
        if name in entries:
            print(f"  {name}: Δ̂ = {entries[name]['delta_hat']:+.6f} "
                  f"({entries[name]['delta_source']})")
    for n in ainfo["notes"]:
        print(f"  NOTE: {n}")
    for name, e in entries.items():
        if e["degenerate"]:
            print(f"  WARNING: {name} is DEGENERATE — bit-identical to "
                  f"{e['equals_registered']} (manifest flag degenerate=true); "
                  f"its students only re-measure that set.")
    print()
    rows = [(n, ref[n], "registered") for n in reg_files]
    rows += [(n, e["summary"],
              "DEGENERATE=" + "/".join(e["equals_registered"])
              if e["degenerate"] else "ablation") for n, e in entries.items()]
    print_ablation_table(rows)
    print("  (RAW1→0 = share of RAW grade-1 pairs whose rounded label is 0; "
          "r(h) = label-human Pearson; clip = labels moved by the [0,3] clip)")
    print(f"\n  wrote {len(entries)} ablation label files; manifest "
          f"'ablation_sets' updated -> {jd}  (registered files verified "
          f"byte-identical)")


# ═══════════════════════════════════════════════════════════════════════════
# torch side (imported lazily so build-labels / compare need no torch)
# ═══════════════════════════════════════════════════════════════════════════

def _torch():
    try:
        import torch
        import transformers  # noqa: F401
    except ImportError:
        sys.exit("ERROR: torch + transformers required: "
                 "pip install -r requirements-batch7.txt")
    return torch


def set_determinism(seed: int) -> None:
    import random
    torch = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def pick_device(spec: str):
    torch = _torch()
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def n_outputs(objective: str) -> int:
    return 1 if objective == "regression" else 3


def objective_loss(logits, y, objective: str):
    import torch.nn.functional as F
    if objective == "regression":
        return F.mse_loss(logits[:, 0], y)
    # ordinal-threshold: 3 sigmoid heads for P(grade > k), k = 0,1,2.
    # Target for threshold k is clip(y - k, 0, 1): exactly 1[y > k] for
    # integer labels, a soft target for fractional ones.
    ks = logits.new_tensor([0.0, 1.0, 2.0])
    t = (y[:, None] - ks[None, :]).clamp(0.0, 1.0)
    return F.binary_cross_entropy_with_logits(logits[:, :3], t)


def logits_to_score(logits, objective: str):
    torch = _torch()
    if objective == "regression":
        return logits[:, 0]
    return torch.sigmoid(logits[:, :3]).sum(dim=1)   # expected grade in [0,3]


def output_head(model, k: int):
    """The final Linear producing the k logits (classifier / score / out_proj
    preferred; works for BERT-style and ELECTRA-style heads)."""
    import torch.nn as nn
    cands = [(n, m) for n, m in model.named_modules()
             if isinstance(m, nn.Linear) and m.out_features == k]
    pref = [c for c in cands
            if any(t in c[0] for t in ("classifier", "score", "out_proj"))]
    pick = pref or cands
    if not pick:
        sys.exit(f"ERROR: cannot locate a {k}-output head in the model.")
    return pick[-1]


def reset_head(model, objective: str, seed: int,
               bias: float | None = None) -> str:
    """Deterministic head re-init for EVERY arm and objective: W ~ N(0, 0.02)
    from a seed-derived generator; bias 1.5 (regression, mid-scale) or 0
    (ordinal: sum of three sigmoid(0) = 1.5, the same starting point)."""
    torch = _torch()
    name, lin = output_head(model, n_outputs(objective))
    g = torch.Generator().manual_seed(1_000_003 * int(seed) + 17)
    with torch.no_grad():
        w = torch.empty(tuple(lin.weight.shape)).normal_(0.0, HEAD_STD,
                                                         generator=g)
        lin.weight.copy_(w)
        if lin.bias is not None:
            if bias is None:
                bias = (HEAD_BIAS_REGRESSION if objective == "regression"
                        else 0.0)
            lin.bias.fill_(bias)
    return name


def resolve_head_bias(spec: str, objective: str, y_train: np.ndarray):
    """train --head-bias -> (mode, value).  '1.5' (default) is the registered
    recipe, bit-identical to HEAD_BIAS_REGRESSION; 'label_mean' = mean of the
    training-split labels actually fitted (EXPLORATORY); else a float."""
    spec = str(spec)
    if spec == HEAD_BIAS_DEFAULT:
        assert float(spec) == HEAD_BIAS_REGRESSION, "default head bias drifted"
        return "default", (HEAD_BIAS_REGRESSION if objective == "regression"
                           else 0.0)
    if objective != "regression":
        sys.exit("ERROR: --head-bias applies to the regression objective only.")
    if spec == "label_mean":
        v = float(np.asarray(y_train, dtype=np.float32).mean())
        print(f"  head bias = training-label mean {v:.6f} "
              f"(n={len(y_train)} train-split labels)")
        return "label_mean", v
    try:
        return "float", float(spec)
    except ValueError:
        sys.exit(f"ERROR: --head-bias must be 1.5, label_mean or a float; "
                 f"got {spec!r}.")


def tagged_set_name(set_name: str, tag: str | None) -> str:
    """train --tag: '<SET>@<tag>' marks an EXPLORATORY recipe variant as a
    distinct label set for evaluate / compare / summary."""
    if tag is None:
        return set_name
    if not TAG_RE.match(str(tag)):
        sys.exit(f"ERROR: --tag must match {TAG_RE.pattern}; got {tag!r}.")
    return f"{set_name}@{tag}"


def encode(tokenizer, queries, passages, max_len: int) -> list[dict]:
    enc = tokenizer(list(queries), list(passages), truncation="longest_first",
                    max_length=max_len)
    keys = list(enc.keys())
    return [{k: enc[k][i] for k in keys} for i in range(len(queries))]


def make_batches(tokenizer, feats: list[dict], idx, device):
    batch = tokenizer.pad([feats[i] for i in idx], return_tensors="pt")
    return {k: v.to(device) for k, v in batch.items()}


def predict(model, tokenizer, feats, objective, device,
            bs: int = 128) -> np.ndarray:
    """fp32 scoring, always (no autocast at evaluation)."""
    torch = _torch()
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(feats), bs):
            batch = make_batches(tokenizer, feats,
                                 range(s, min(s + bs, len(feats))), device)
            logits = model(**batch).logits.float()
            out.append(logits_to_score(logits, objective).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def load_student(name_or_dir, num_labels: int | None = None):
    torch = _torch()  # noqa: F841
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name_or_dir)
    kw = ({"num_labels": num_labels, "ignore_mismatched_sizes": True}
          if num_labels is not None else {})
    model = AutoModelForSequenceClassification.from_pretrained(name_or_dir,
                                                               **kw)
    model.float()                      # fp32 master weights regardless of ckpt
    return model, tok


def run_training(recs: list[dict], *, model_name: str, objective: str,
                 epochs: int, lr: float, bs: int, max_len: int, seed: int,
                 device, fp16: bool, weight_decay: float, warmup: float,
                 select: str, log_every: int,
                 head_bias: str = HEAD_BIAS_DEFAULT):
    """Train one student; returns (model, tokenizer, partial log dict)."""
    torch = _torch()
    from transformers import get_linear_schedule_with_warmup

    is_val = np.array([val_fold(str(r["dataset"]), str(r["qid"]))
                       for r in recs])
    tr_idx = np.where(~is_val)[0]
    va_idx = np.where(is_val)[0]
    if tr_idx.size == 0:
        sys.exit("ERROR: validation split took every query.")
    tr_q = {(recs[i]["dataset"], recs[i]["qid"]) for i in tr_idx}
    va_q = {(recs[i]["dataset"], recs[i]["qid"]) for i in va_idx}
    y_all = np.array([float(r["label"]) for r in recs], dtype=np.float32)
    if y_all.min() < 0 or y_all.max() > 3:
        sys.exit("ERROR: labels outside [0,3].")
    v_lab_mean = float(y_all[va_idx].mean()) if va_idx.size else None

    set_determinism(seed)
    model, tok = load_student(model_name, n_outputs(objective))
    hb_mode, hb_val = resolve_head_bias(head_bias, objective, y_all[tr_idx])
    head = reset_head(model, objective, seed,
                      None if hb_mode == "default" else hb_val)
    model.to(device)
    feats = encode(tok, [r["query"] for r in recs],
                   [r["passage"] for r in recs], max_len)
    y_t = torch.tensor(y_all)

    def val_pass():
        if not va_idx.size:
            return float("nan"), None
        model.eval()
        vl, vp = 0.0, []
        with torch.no_grad():
            for s in range(0, va_idx.size, 128):
                bidx = va_idx[s:s + 128]
                lg = model(**make_batches(tok, feats, bidx, device)).logits.float()
                yb = y_t[bidx].to(device)
                vl += float(objective_loss(lg, yb, objective)) * len(bidx)
                vp.append(logits_to_score(lg, objective).cpu().numpy())
        return vl / va_idx.size, float(np.concatenate(vp).mean())

    v0, p0 = val_pass()
    val_hist = [{"epoch": 0, "train_loss_epoch": None, "val_loss": v0,
                 "val_pred_mean": p0, "val_label_mean": v_lab_mean}]

    steps_per_epoch = math.ceil(tr_idx.size / bs)
    total = steps_per_epoch * epochs
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
    sched = get_linear_schedule_with_warmup(opt, int(warmup * total), total)
    scaler = torch.amp.GradScaler(device.type, enabled=fp16)
    gen = torch.Generator().manual_seed(seed)

    curve = []
    best_loss, best_epoch, best_state = float("inf"), None, None
    step = 0
    win_loss, win_n = 0.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss, ep_n = 0.0, 0
        perm = tr_idx[torch.randperm(tr_idx.size, generator=gen).numpy()]
        for s in range(0, perm.size, bs):
            bidx = perm[s:s + bs]
            batch = make_batches(tok, feats, bidx, device)
            yb = y_t[bidx].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=fp16):
                logits = model(**batch).logits
            loss = objective_loss(logits.float(), yb, objective)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            lv = float(loss.detach()) * len(bidx)
            win_loss += lv
            win_n += len(bidx)
            ep_loss += lv
            ep_n += len(bidx)
            if step % log_every == 0 or step == total:
                curve.append({"step": step, "epoch": epoch,
                              "loss": win_loss / max(win_n, 1),
                              "lr": float(sched.get_last_lr()[0])})
                win_loss, win_n = 0.0, 0
        vloss, vpm = val_pass()
        tl = ep_loss / max(ep_n, 1)
        val_hist.append({"epoch": epoch, "train_loss_epoch": tl,
                         "val_loss": vloss, "val_pred_mean": vpm,
                         "val_label_mean": v_lab_mean})
        print(f"  epoch {epoch}: train loss={tl:.4f}  val loss={vloss:.4f}  "
              f"val pred mean={vpm if vpm is None else round(vpm, 4)} "
              f"(label mean {v_lab_mean if v_lab_mean is None else round(v_lab_mean, 4)})")
        if select == "best" and np.isfinite(vloss) and vloss <= best_loss:
            best_loss, best_epoch = vloss, epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    if select == "best" and best_state is not None:
        model.load_state_dict(best_state)
        sel_epoch = best_epoch
    else:
        sel_epoch = epochs                 # 'last', or 'best' without val
    sel = val_hist[sel_epoch]
    warnings = []
    if sel["val_pred_mean"] is not None and v_lab_mean is not None:
        gap = sel["val_pred_mean"] - v_lab_mean
        if abs(gap) > CALIB_WARN:
            warnings.append(
                f"selected epoch {sel_epoch}: |val_pred_mean - val_label_mean|"
                f" = {abs(gap):.3f} > {CALIB_WARN} — the student has not "
                f"reached its label set's operating point; absolute endpoints "
                f"would reflect under-training, not inheritance")
    vl_seq = [v["val_loss"] for v in val_hist[1:]]
    rose = any(b > a for a, b in zip(vl_seq, vl_seq[1:])
               if np.isfinite(a) and np.isfinite(b))
    log = {
        "head_reset": {"module": head, "w_std": HEAD_STD, "bias": hb_val},
        "head_bias_mode": hb_mode, "head_bias_value": hb_val,
        "n_train": int(tr_idx.size), "n_val": int(va_idx.size),
        "n_train_queries": len(tr_q), "n_val_queries": len(va_q),
        "val_query_disjoint": len(tr_q & va_q) == 0,
        "label_mean_all": float(y_all.mean()),
        "label_mean_train": float(y_all[tr_idx].mean()),
        "label_mean_val": v_lab_mean,
        "steps_per_epoch": steps_per_epoch, "total_steps": total,
        "loss_curve": curve, "val": val_hist,
        "select_rule": select, "selected_epoch": sel_epoch,
        "val_loss_rose": bool(rose), "warnings": warnings,
    }
    for w in warnings:
        print(f"  WARNING: {w}")
    return model, tok, log


def labels_meta(labels_path: Path, recs: list[dict]):
    datasets = sorted({str(r["dataset"]) for r in recs})
    guard_train_datasets(datasets)
    set_name = recs[0].get("set") or labels_path.stem.replace("labels_", "")
    mp = labels_path.parent / "manifest.json"
    judge = (json.loads(mp.read_text(encoding="utf-8")).get("model")
             if mp.exists() else None)
    return datasets, set_name, judge


def cmd_train(args) -> None:
    torch = _torch()
    import transformers
    t0 = time.time()
    arm, model_name = resolve_arm(args.arm, args.model)
    epochs, ep_def = arm_hparam(arm, "epochs", args.epochs)
    lr, lr_def = arm_hparam(arm, "lr", args.lr)
    require_arm_in_path(args.out, arm, "--out")
    labels_path = Path(args.labels)
    recs = read_jsonl(labels_path)
    if not recs:
        sys.exit(f"ERROR: {labels_path} is empty.")
    datasets, base_set, judge = labels_meta(labels_path, recs)
    tag = getattr(args, "tag", None)
    set_name = tagged_set_name(base_set, tag)

    device = pick_device(args.device)
    fp16 = bool(device.type == "cuda" and not args.no_fp16)
    hp = {"epochs": epochs, "lr": lr, "bs": args.bs, "max_len": args.max_len,
          "objective": args.objective, "warmup": args.warmup,
          "weight_decay": args.weight_decay, "select": args.select,
          "seed": args.seed, "fp16_train": fp16}
    model, tok, log = run_training(
        recs, model_name=model_name, objective=args.objective, epochs=epochs,
        lr=lr, bs=args.bs, max_len=args.max_len, seed=args.seed,
        device=device, fp16=fp16, weight_decay=args.weight_decay,
        warmup=args.warmup, select=args.select, log_every=args.log_every,
        head_bias=getattr(args, "head_bias", HEAD_BIAS_DEFAULT))
    hp["head_bias_mode"] = log.pop("head_bias_mode")
    hp["head_bias_value"] = log.pop("head_bias_value")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    cfg = {"objective": args.objective, "num_labels": n_outputs(args.objective),
           "max_len": args.max_len, "base_model": model_name, "arm": arm}
    (out / "batch7_config.json").write_text(json.dumps(cfg, indent=2))
    log.update({
        "labels": str(labels_path), "labels_sha256": sha256_file(labels_path),
        "label_set": set_name, "label_set_base": base_set, "tag": tag,
        "judge": judge, "train_datasets": datasets, "arm": arm, "base_model": model_name,
        "hyperparams": hp,
        "hyperparams_from_arm_defaults": {"epochs": ep_def, "lr": lr_def},
        "objective": args.objective, "seed": args.seed,
        "device": str(device), "fp16": fp16,
        "runtime_s": round(time.time() - t0, 1),
        "versions": {"torch": torch.__version__,
                     "transformers": transformers.__version__},
        "code_git": git_hash(),
    })
    (out / "train_log.json").write_text(json.dumps(log, indent=2,
                                                   ensure_ascii=False))
    print(f"  [{arm}/{set_name}] seed {args.seed}: epochs={epochs} lr={lr} "
          f"n_train={log['n_train']} n_val={log['n_val']} label mean="
          f"{log['label_mean_all']:.4f} selected epoch={log['selected_epoch']}"
          f" -> {out}  ({log['runtime_s']}s)")


def parse_grid(spec: str) -> dict:
    casts = {"epochs": int, "lr": float, "bs": int}
    grid = {}
    for part in str(spec).split(";"):
        if not part.strip():
            continue
        k, _, vals = part.partition("=")
        k = k.strip()
        if k not in casts:
            sys.exit(f"ERROR: sweep grid key {k!r} not in {sorted(casts)}.")
        grid[k] = [casts[k](v) for v in vals.split(",") if v.strip()]
    if "epochs" not in grid or "lr" not in grid:
        sys.exit("ERROR: sweep grid needs epochs=... and lr=...")
    return grid


def cmd_sweep(args) -> None:
    arm, model_name = resolve_arm(args.arm, args.model)
    require_arm_in_path(args.out, arm, "--out")
    labels_path = Path(args.labels)
    recs = read_jsonl(labels_path)
    if not recs or any(r.get("set") != "HUMAN" or r.get("source") != "human"
                       for r in recs):
        sys.exit("ERROR: sweep runs on the HUMAN label set only "
                 "(labels_HUMAN.jsonl).")
    datasets, _set, judge = labels_meta(labels_path, recs)
    grid = parse_grid(args.grid)
    device = pick_device(args.device)
    fp16 = bool(device.type == "cuda" and not args.no_fp16)
    keys = sorted(grid)
    rows = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        c = dict(zip(keys, combo))
        bs = c.get("bs", args.bs)
        print(f"-- sweep [{arm}] epochs={c['epochs']} lr={c['lr']} bs={bs}")
        _m, _t, log = run_training(
            recs, model_name=model_name, objective=args.objective,
            epochs=c["epochs"], lr=c["lr"], bs=bs, max_len=args.max_len,
            seed=args.seed, device=device, fp16=fp16,
            weight_decay=args.weight_decay, warmup=args.warmup,
            select="last", log_every=10 ** 9)
        fin = log["val"][-1]
        rows.append({"epochs": c["epochs"], "lr": c["lr"], "bs": bs,
                     "final_val_loss": fin["val_loss"],
                     "val_pred_mean": fin["val_pred_mean"],
                     "val_label_mean": fin["val_label_mean"],
                     "val_curve": log["val"], "warnings": log["warnings"]})
        del _m
    ok = [r for r in rows if np.isfinite(r["final_val_loss"])]
    if not ok:
        sys.exit("ERROR: no config produced a finite validation loss.")
    pick = min(ok, key=lambda r: (r["final_val_loss"], r["epochs"], r["lr"]))
    print("=" * 78)
    print(f"  BATCH 7 sweep — arm {arm} ({model_name}) on HUMAN "
          f"({judge}, {datasets}); select=last; validation = training-pool "
          f"10% queries; no test data")
    print("=" * 78)
    print(f"  {'epochs':>6} {'lr':>8} {'bs':>4} {'val loss':>9} "
          f"{'val pred':>9} {'val label':>9}")
    for r in sorted(rows, key=lambda r: (r["epochs"], r["lr"])):
        star = "  <- pick" if r is pick else ""
        print(f"  {r['epochs']:>6} {r['lr']:>8.1e} {r['bs']:>4} "
              f"{r['final_val_loss']:>9.4f} {r['val_pred_mean']:>9.4f} "
              f"{r['val_label_mean']:>9.4f}{star}")
    print(f"\n  pick (lowest final val loss; ties -> fewer epochs, lower lr): "
          f"epochs={pick['epochs']} lr={pick['lr']}")
    out = {"arm": arm, "base_model": model_name, "labels": str(labels_path),
           "labels_sha256": sha256_file(labels_path), "judge": judge,
           "train_datasets": datasets, "objective": args.objective,
           "seed": args.seed, "grid": grid, "rows": rows,
           "pick": {"epochs": pick["epochs"], "lr": pick["lr"],
                    "bs": pick["bs"]},
           "rule": "lowest final validation loss (select=last); ties -> "
                   "fewer epochs, then lower lr",
           "code_git": git_hash()}
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  -> {op}")


# ═══════════════════════════════════════════════════════════════════════════
# evaluate — metrics (pure numpy; selftest checks them directly)
# ═══════════════════════════════════════════════════════════════════════════

def _dcg(g) -> float:
    g = np.asarray(g, dtype=float)
    return float(((2.0 ** g - 1.0) / np.log2(np.arange(2, g.size + 2))).sum())


def query_metrics(docids, pred, human, k: int = NDCG_K) -> tuple[float, float]:
    """(NDCG@k, AP) of one query, standard (non-condensed) convention."""
    order = sorted(range(len(docids)), key=lambda i: (-pred[i], docids[i]))
    ranked = [int(human[i]) for i in order]
    idcg = _dcg(sorted((int(h) for h in human), reverse=True)[:k])
    ndcg = _dcg(ranked[:k]) / idcg if idcg > 0 else float("nan")
    rel = np.array([g >= REL_THRESHOLD for g in ranked], dtype=float)
    R = rel.sum()
    if R == 0:
        ap = float("nan")
    else:
        prec = np.cumsum(rel) / np.arange(1, rel.size + 1)
        ap = float((prec * rel).sum() / R)
    return ndcg, ap


STAT_COLS = ("n", "sum_pc", "sum_pr", "sum_h", "sum_abs", "tp", "fp", "fn",
             "ndcg", "ndcg_ok", "ap", "ap_ok",
             "b0_n", "b1_n", "b2_n", "b3_n", "b0_p", "b1_p", "b2_p", "b3_p",
             "b0_h", "b1_h", "b2_h", "b3_h")


def per_query_stats(pairs) -> tuple[list, np.ndarray]:
    """pairs: iterable of (qid, docid, pred_raw, human) -> (qids, S[Q, cols])."""
    byq = defaultdict(list)
    for q, doc, p, h in pairs:
        byq[str(q)].append((str(doc), float(p), int(h)))
    qids = sorted(byq)
    S = np.zeros((len(qids), len(STAT_COLS)))
    c = {name: i for i, name in enumerate(STAT_COLS)}
    for qi, q in enumerate(qids):
        docs = [x[0] for x in byq[q]]
        pr = np.array([x[1] for x in byq[q]])
        h = np.array([x[2] for x in byq[q]])
        pc = np.clip(pr, 0.0, 3.0)
        rb = np.array([round_half_up(v) for v in pc])     # rounded prediction
        pred_rel = rb >= REL_THRESHOLD
        hum_rel = h >= REL_THRESHOLD
        nd, ap = query_metrics(docs, pr, h)
        row = {"n": h.size, "sum_pc": pc.sum(), "sum_pr": pr.sum(),
               "sum_h": h.sum(), "sum_abs": np.abs(pc - h).sum(),
               "tp": (pred_rel & hum_rel).sum(),
               "fp": (pred_rel & ~hum_rel).sum(),
               "fn": (~pred_rel & hum_rel).sum(),
               "ndcg": 0.0 if math.isnan(nd) else nd,
               "ndcg_ok": 0.0 if math.isnan(nd) else 1.0,
               "ap": 0.0 if math.isnan(ap) else ap,
               "ap_ok": 0.0 if math.isnan(ap) else 1.0}
        for b in range(4):
            m = rb == b
            row[f"b{b}_n"] = m.sum()
            row[f"b{b}_p"] = pc[m].sum()
            row[f"b{b}_h"] = h[m].sum()
        for name, v in row.items():
            S[qi, c[name]] = float(v)
    return qids, S


def _div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1.0), np.nan)


def aggregate(W: np.ndarray, S: np.ndarray) -> dict:
    """Endpoints for query-weight rows W[R, Q] (R=1 -> point estimate)."""
    T = W @ S                                           # (R, cols)
    c = {name: T[:, i] for i, name in enumerate(STAT_COLS)}
    N = c["n"]
    mean_p = _div(c["sum_pc"], N)
    mean_h = _div(c["sum_h"], N)
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    out = {
        "ndcg10": _div(c["ndcg"], c["ndcg_ok"]),
        "map": _div(c["ap"], c["ap_ok"]),
        "mean_pred": mean_p,
        "mean_human": mean_h,
        "offset": mean_p - mean_h,
        "offset_unclipped": _div(c["sum_pr"], N) - mean_h,
        "mae": _div(c["sum_abs"], N),
        "precision": _div(tp, tp + fp),
        "recall": _div(tp, tp + fn),
        "f1": _div(2 * tp, 2 * tp + fp + fn),
        "rel_rate_ratio": _div(tp + fp, tp + fn),
        "ece4": _div(sum(np.abs(c[f"b{b}_p"] - c[f"b{b}_h"]) for b in range(4)),
                     N),
    }
    out["abs_offset"] = np.abs(out["offset"])
    out["abs_ratio_dev"] = np.abs(out["rel_rate_ratio"] - 1.0)
    for b in range(4):
        out[f"calib_bin{b}_n"] = c[f"b{b}_n"]
        out[f"calib_bin{b}_mean_pred"] = _div(c[f"b{b}_p"], c[f"b{b}_n"])
        out[f"calib_bin{b}_mean_human"] = _div(c[f"b{b}_h"], c[f"b{b}_n"])
    return out


ENDPOINTS = ("ndcg10", "map", "offset", "abs_offset", "offset_unclipped",
             "mae", "precision", "recall", "f1", "rel_rate_ratio",
             "abs_ratio_dev", "ece4", "mean_pred", "mean_human")


def boot_weights(n_q: int, b: int = BOOT_B, seed: int = BOOT_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_q, size=(b, n_q))
    W = np.zeros((b, n_q))
    for r in range(b):
        W[r] = np.bincount(idx[r], minlength=n_q)
    return W


def ci90(x: np.ndarray) -> list:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(x, 5.0)), float(np.percentile(x, 95.0))]


def evaluate_pairs(pairs, b: int = BOOT_B, seed: int = BOOT_SEED) -> dict:
    qids, S = per_query_stats(pairs)
    point = aggregate(np.ones((1, len(qids))), S)
    boot = aggregate(boot_weights(len(qids), b, seed), S)
    res = {"n_pairs": int(S[:, 0].sum()), "n_queries": len(qids),
           "n_queries_ndcg": int(S[:, STAT_COLS.index("ndcg_ok")].sum()),
           "n_queries_map": int(S[:, STAT_COLS.index("ap_ok")].sum()),
           "point": {k: float(v[0]) for k, v in point.items()},
           "ci90": {k: ci90(boot[k]) for k in ENDPOINTS}}
    return res


def universe_sha(pairs) -> str:
    keys = sorted(f"{q}\t{d}" for q, d, _p, _h in pairs)
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()


def cmd_evaluate(args) -> None:
    # GATE FIRST — nothing is read before the seal check passes.
    if args.dataset not in DATASETS:
        sys.exit(f"ERROR: unknown dataset {args.dataset}.")
    gate_test_dataset(args.dataset, args.unseal_dl22)
    if args.dataset in SEALED:
        print(f"[B7] {args.dataset} opened for student evaluation: heading "
              f"'# {B7_MARKER}' found in committed PREDICTIONS.md and "
              f"--unseal-dl22 given.")

    model_dir = Path(args.model_dir)
    tlog = json.loads((model_dir / "train_log.json").read_text(encoding="utf-8"))
    cfg = json.loads((model_dir / "batch7_config.json").read_text())
    arm = tlog.get("arm") or cfg.get("arm")
    require_arm_in_path(args.out, arm, "--out")
    if args.dataset in tlog.get("train_datasets", []):
        sys.exit(f"ERROR: {args.dataset} is one of this student's training "
                 f"datasets {tlog['train_datasets']}; evaluating on it is "
                 f"in-sample.")
    data_dir = resolve_data_dir(args.data_dir)
    pairs_path = Path(args.pairs) if args.pairs else \
        data_dir / "inputs" / f"pairs_{args.dataset}.jsonl"
    texts = load_pairs_texts(pairs_path, args.dataset)
    grades = load_grades([args.dataset], texts, args.qrels_source)
    keys = sorted(k for k in texts if k in grades)
    if not keys:
        sys.exit("ERROR: no judged pairs in the test universe.")

    device = pick_device(args.device)
    model, tok = load_student(model_dir)
    model.to(device)
    feats = encode(tok, [texts[k]["query"] for k in keys],
                   [texts[k]["passage"] for k in keys], int(cfg["max_len"]))
    pred = predict(model, tok, feats, cfg["objective"], device)   # fp32
    pairs = [(k[1], k[2], float(p), int(grades[k])) for k, p in zip(keys, pred)]
    res = evaluate_pairs(pairs, args.b, args.seed)
    out = {
        "dataset": args.dataset, "model_dir": str(model_dir),
        "arm": arm, "base_model": tlog.get("base_model") or cfg.get("base_model"),
        "label_set": tlog.get("label_set"), "judge": tlog.get("judge"),
        "labels_sha256": tlog.get("labels_sha256"),
        "seed": tlog.get("seed"), "objective": cfg["objective"],
        "hyperparams": tlog.get("hyperparams"),
        "selected_epoch": tlog.get("selected_epoch"),
        "train_warnings": tlog.get("warnings", []),
        "train_label_mean": tlog.get("label_mean_all"),
        "train_datasets": tlog.get("train_datasets"),
        "scoring": "fp32",
        "convention": ("standard re-ranking of the judged pool "
                       "(pairs file ∩ qrels): rank by unclipped score, ties by "
                       "docid; NDCG@10 gain 2^rel-1, ideal from human grades; "
                       "MAP at human>=2; NOT the batch-5 condensed convention"),
        "absolute_convention": ("predictions clipped to [0,3] for offset/MAE/"
                                "calibration; binary relevance = "
                                "round_half_up(clipped pred) >= 2 vs human >= 2"),
        "bootstrap": {"B": args.b, "seed": args.seed,
                      "unit": "query (clustered)", "ci": "90% percentile"},
        "rel_disagreements": rel_disagreements(texts, grades,
                                               args.qrels_source),
        "universe_sha256": universe_sha(pairs),
        **res,
        "pairs": [[q, d, p, h] for q, d, p, h in pairs],
        "code_git": git_hash(),
    }
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print_eval(out)
    print(f"  -> {op}")


def print_eval(r: dict) -> None:
    p, c = r["point"], r["ci90"]
    print("=" * 78)
    print(f"  BATCH 7 evaluate — [{r['arm']}] {r['judge']} / {r['label_set']} "
          f"/ seed {r['seed']}  on {r['dataset']}  (train label mean "
          f"{r['train_label_mean']}, selected epoch {r['selected_epoch']})")
    print("=" * 78)
    for w in r.get("train_warnings", []):
        print(f"  TRAIN WARNING: {w}")
    print(f"  pairs {r['n_pairs']}  queries {r['n_queries']} (NDCG over "
          f"{r['n_queries_ndcg']}, MAP over {r['n_queries_map']})")
    for k in ENDPOINTS:
        print(f"  {k:<18} {p[k]:>+9.4f}   90% CI [{c[k][0]:+.4f}, "
              f"{c[k][1]:+.4f}]")
    print("  calibration (bin = rounded prediction):")
    for b in range(4):
        print(f"    bin {b}: n={int(p[f'calib_bin{b}_n']):>5}  mean pred "
              f"{p[f'calib_bin{b}_mean_pred']:.3f}  mean human "
              f"{p[f'calib_bin{b}_mean_human']:.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# compare — paired-query bootstrap
# ═══════════════════════════════════════════════════════════════════════════

def check_sides(side_a: list[dict], side_b: list[dict],
                allow_cross_arm: bool = False,
                allow_self: bool = False) -> None:
    for side, nm in ((side_a, "A"), (side_b, "B")):
        if not side:
            sys.exit(f"ERROR: side {nm} is empty.")
        dirs = [r.get("model_dir") for r in side]
        if len(set(dirs)) != len(dirs):
            sys.exit(f"ERROR: side {nm} lists the same student twice: {dirs}.")
        seeds = [r.get("seed") for r in side]
        if len(set(seeds)) != len(seeds):
            sys.exit(f"ERROR: side {nm} repeats a seed: {seeds}.")
        for fld in ("label_set", "judge", "objective", "arm", "base_model"):
            vals = {str(r.get(fld)) for r in side}
            if len(vals) != 1:
                sys.exit(f"ERROR: side {nm} mixes {fld} values {sorted(vals)}; "
                         f"a side is one student config over seeds.")
    if not allow_self:
        both = ({r.get("model_dir") for r in side_a}
                & {r.get("model_dir") for r in side_b})
        if both:
            sys.exit(f"ERROR: the same student appears on both sides: {both}.")
    arms = {(str(r.get("arm")), str(r.get("base_model")))
            for r in side_a + side_b}
    if len(arms) != 1 and not allow_cross_arm:
        sys.exit(f"ERROR: sides use different arms/base models {sorted(arms)}; "
                 f"pass --allow-cross-arm to compare across arms deliberately.")


def compare_results(side_a: list[dict], side_b: list[dict],
                    b: int = BOOT_B, seed: int = BOOT_SEED,
                    margin: float = TOST_MARGIN, allow_cross_arm: bool = False,
                    allow_self: bool = False) -> dict:
    check_sides(side_a, side_b, allow_cross_arm, allow_self)
    allr = side_a + side_b
    ds = {r["dataset"] for r in allr}
    us = {r["universe_sha256"] for r in allr}
    if len(ds) != 1 or len(us) != 1:
        sys.exit(f"ERROR: compare needs one dataset and one pair universe; got "
                 f"datasets {ds}, {len(us)} distinct universes.")
    stats = [per_query_stats(r["pairs"]) for r in allr]
    qids0 = stats[0][0]
    if any(s[0] != qids0 for s in stats):
        sys.exit("ERROR: query sets differ between result files.")
    n_q = len(qids0)
    W = np.vstack([np.ones((1, n_q)), boot_weights(n_q, b, seed)])
    aggs = [aggregate(W, s[1]) for s in stats]
    na = len(side_a)

    def side(lo, hi, k):
        return np.mean([aggs[i][k] for i in range(lo, hi)], axis=0)

    def desc(sd):
        r = sd[0]
        return {"arm": r.get("arm"), "base_model": r.get("base_model"),
                "label_set": r.get("label_set"), "judge": r.get("judge"),
                "objective": r.get("objective"),
                "seeds": [x.get("seed") for x in sd],
                "model_dirs": [x.get("model_dir") for x in sd]}

    out = {"dataset": ds.pop(), "n_queries": n_q, "B": b, "seed": seed,
           "a": desc(side_a), "b": desc(side_b), "endpoints": {}}
    for k in ENDPOINTS:
        a_k, b_k = side(0, na, k), side(na, len(allr), k)
        d = a_k - b_k
        out["endpoints"][k] = {
            "a": float(a_k[0]), "b": float(b_k[0]), "diff": float(d[0]),
            "ci90": ci90(d[1:]),
            "a_per_file": [float(aggs[i][k][0]) for i in range(na)],
            "b_per_file": [float(aggs[i][k][0]) for i in range(na, len(allr))],
        }
    lo, hi = out["endpoints"]["ndcg10"]["ci90"]
    eq = bool(np.isfinite(lo) and np.isfinite(hi) and -margin < lo
              and hi < margin)
    out["tost_ndcg10"] = {
        "margin": margin, "ci90": [lo, hi], "equivalent": eq,
        "statement": (
            f"NDCG@10 difference {out['endpoints']['ndcg10']['diff']:+.4f}, "
            f"90% CI [{lo:+.4f}, {hi:+.4f}]: "
            + (f"EQUIVALENT within ±{margin} (TOST, alpha=0.05 per side: the "
               f"90% CI lies inside the margin)" if eq else
               f"equivalence within ±{margin} NOT shown (90% CI not inside "
               f"the margin)"))}
    return out


def _sd(x: list) -> float:
    return float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")


def cmd_compare(args) -> None:
    pa = [p for p in str(args.a).split(",") if p.strip()]
    pb = [p for p in str(args.b).split(",") if p.strip()]
    res_paths = [str(Path(p).resolve()) for p in pa + pb]
    if len(set(res_paths)) != len(res_paths):
        sys.exit(f"ERROR: duplicate result files in --a/--b: {pa + pb}.")
    A = [json.loads(Path(p).read_text(encoding="utf-8")) for p in pa]
    B = [json.loads(Path(p).read_text(encoding="utf-8")) for p in pb]
    res = compare_results(A, B, args.b_boot, args.seed, args.margin,
                          allow_cross_arm=args.allow_cross_arm)
    print("=" * 78)
    print(f"  BATCH 7 compare on {res['dataset']} [arm {res['a']['arm']}"
          f"{'' if res['a']['arm'] == res['b']['arm'] else ' vs ' + str(res['b']['arm'])}]: "
          f"A={res['a']['label_set']} seeds {res['a']['seeds']}  vs  "
          f"B={res['b']['label_set']} seeds {res['b']['seeds']}  "
          f"(judge {res['a']['judge']})")
    print(f"  paired query bootstrap B={res['B']} seed={res['seed']}, "
          f"{res['n_queries']} queries; side metric = mean over its seeds")
    print("=" * 78)
    print(f"  {'endpoint':<18} {'A':>9} {'B':>9} {'A-B':>9}   90% CI"
          f"                 seed sd A / B")
    for k, e in res["endpoints"].items():
        print(f"  {k:<18} {e['a']:>+9.4f} {e['b']:>+9.4f} {e['diff']:>+9.4f}"
              f"   [{e['ci90'][0]:+.4f}, {e['ci90'][1]:+.4f}]"
              f"   {_sd(e['a_per_file']):.4f} / {_sd(e['b_per_file']):.4f}")
    print("  (seed sd = the run-to-run noise floor of each side)")
    print(f"\n  {res['tost_ndcg10']['statement']}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"  -> {args.out}")


# ═══════════════════════════════════════════════════════════════════════════
# selftest
# ═══════════════════════════════════════════════════════════════════════════

STUB_JUDGES = {
    "stub/lenient": {"dial_level": 2, "monotone_map": [0, 1, 2, 3]},
    "stub/strict": {"dial_level": 4, "monotone_map": [0, 1, 2, 3]},
    "stub/identity": {"dial_level": 3, "monotone_map": [0, 1, 2, 3]},
}
TINY_ARM = "tiny-test"


def _stub_rows(model, ds, grades, rng):
    """5 levels x 3 paraphrases x 1 run. lenient: L2 = h, other levels h+1
    (clip).  strict: L4 = h, others h-1 (clip).  identity: every level = h
    with paraphrase noise (fractional means -> round-half-up exercised)."""
    rows = []
    for (d_, q, doc), h in grades.items():
        if d_ != ds:
            continue
        for lvl in LEVELS:
            for para in "abc":
                if model == "stub/lenient":
                    s = h if lvl == 2 else min(h + 1, 3)
                elif model == "stub/strict":
                    s = h if lvl == 4 else max(h - 1, 0)
                else:
                    s = int(np.clip(h + rng.choice([-1, 0, 0, 1]), 0, 3))
                rows.append({"model_id": model, "prompt_id": f"L{lvl}_{para}",
                             "politeness_level": lvl, "dataset": ds,
                             "qid": q, "docid": doc, "run": 1,
                             "score": float(s), "parse_ok": True})
    return rows


def _make_stub_data_dir(root: Path, n_q: int = 30, n_d: int = 20) -> dict:
    rng = np.random.default_rng(7)
    words = ["alpha", "bravo", "delta", "echo", "golf", "hotel", "kilo",
             "lima", "oscar", "papa", "romeo", "tango"]
    grades = {}
    (root / "inputs").mkdir(parents=True)
    (root / "derived").mkdir(parents=True)
    for ds in ("dl19", "dl20", "dl22"):
        rows = []
        for qi in range(n_q if ds != "dl22" else 12):
            q = str(1000 * (1 + ["dl19", "dl20", "dl22"].index(ds)) + qi)
            qw = words[qi % len(words)]
            for di in range(n_d if ds != "dl22" else 10):
                doc = f"p{ds}_{qi}_{di}"
                h = int(rng.integers(0, 4))
                grades[(ds, q, doc)] = h
                passage = " ".join([qw] * h + [words[(qi + 3) % 12]] * (4 - h))
                rows.append({"dataset": ds, "qid": q, "docid": doc,
                             "query": f"what is {qw}", "passage": passage,
                             "rel": h})
        write_jsonl(root / "inputs" / f"pairs_{ds}.jsonl", rows)
    rows = []
    for m in STUB_JUDGES:
        for ds in ("dl19", "dl20", "dl22"):
            rows += _stub_rows(m, ds, grades, rng)
    pd.DataFrame(rows).to_parquet(root / "derived" / "judgments.parquet",
                                  index=False)
    rules = {"git_hash": "stub", "datasets_used": ["dl19", "dl20"], "rules": {
        m: {"msmarco": dict(r, datasets=["dl19", "dl20"])}
        for m, r in STUB_JUDGES.items()}}
    # delta_hat: exact for lenient (check must say ok), deliberately wrong for
    # strict (same pool -> WARNING), absent for identity (not checked).
    tr = [h for k, h in grades.items() if k[0] in ("dl19", "dl20")]
    rules["rules"]["stub/lenient"]["msmarco"]["delta_hat"] = float(
        np.mean([min(h + 1, 3) for h in tr]) - np.mean(tr))
    rules["rules"]["stub/strict"]["msmarco"]["delta_hat"] = 0.5
    (root / "derived" / "correction_rules.json").write_text(json.dumps(rules))
    return grades


def _make_tiny_model(path: Path, num_labels: int = 1) -> None:
    # Built from the `tokenizers` primitives rather than BertTokenizerFast(
    # vocab_file=...), whose constructor changed across transformers 4.x/5.x
    # (5.x silently maps every token to [UNK] with that call).
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
    from tokenizers.processors import TemplateProcessing
    from transformers import (BertConfig, BertForSequenceClassification,
                              PreTrainedTokenizerFast)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "what", "is",
             "alpha", "bravo", "delta", "echo", "golf", "hotel", "kilo",
             "lima", "oscar", "papa", "romeo", "tango"]
    path.mkdir(parents=True, exist_ok=True)
    tk = Tokenizer(models.WordPiece({w: i for i, w in enumerate(vocab)},
                                    unk_token="[UNK]"))
    tk.normalizer = normalizers.BertNormalizer(lowercase=True)
    tk.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tk.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]",
        cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
        model_input_names=["input_ids", "token_type_ids", "attention_mask"])
    _torch().manual_seed(0)
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=16,
                     num_hidden_layers=1, num_attention_heads=2,
                     intermediate_size=32, max_position_embeddings=64,
                     num_labels=num_labels)
    BertForSequenceClassification(cfg).save_pretrained(path)
    tok.save_pretrained(path)


def _git(repo: Path, *a) -> None:
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=b7selftest",
                    "-c", "user.email=b7@selftest.invalid",
                    "-c", "commit.gpgsign=false", *a],
                   check=True, capture_output=True)


def cmd_selftest(args) -> None:
    global PREDICTIONS_PATH
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print("=" * 78)
    print("  BATCH 7 SELF-TEST (synthetic; CPU; offline)")
    print("=" * 78)
    failures = []

    def check(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  — {detail}" if detail else ""))
        if not cond:
            failures.append(name)

    def exits(fn) -> bool:
        try:
            fn()
        except SystemExit:
            return True
        return False

    tmp = Path(tempfile.mkdtemp(prefix="b7selftest_"))
    root = tmp / "DATA"
    grades = _make_stub_data_dir(root)
    ALL = ("HUMAN", "RAW", "CORR_FROZEN", "CORR_REFIT", "MIX_200", "MIX_1000",
           "CORR_REFIT_n200", "CORR_REFIT_n1000")

    def bl_ns(judge, **kw):
        ns = dict(data_dir=str(root), train_datasets="dl19,dl20", judge=judge,
                  rules=None, pairs_dir=None, qrels_source="pairs-rel",
                  mix_sizes="200,1000", mix_seed=MIX_SEED,
                  fold_seed=FOLD_SEED, out_dir=None, force=False)
        ns.update(kw)
        return argparse.Namespace(**ns)

    # -- 1. build-labels via the CLI path ----------------------------------
    means = {}
    for judge in STUB_JUDGES:
        cmd_build_labels(bl_ns(judge))
        jd = root / "derived" / "batch7" / judge_slug(judge)
        check(f"{judge}: 8 label files written",
              all((jd / f"labels_{n}.jsonl").exists() for n in ALL))
        labs = {n: read_jsonl(jd / f"labels_{n}.jsonl") for n in ALL}
        keysets = {n: [(r["dataset"], r["qid"], r["docid"]) for r in v]
                   for n, v in labs.items()}
        check(f"{judge}: all sets share one pair universe",
              all(keysets[n] == keysets["HUMAN"] for n in ALL))
        rec = labs["RAW"][0]
        check(f"{judge}: record schema",
              all(f in rec for f in ("qid", "docid", "query", "passage",
                                     "label", "source")))
        m = {n: float(np.mean([r["label"] for r in v])) for n, v in labs.items()}
        means[judge] = m
        check(f"{judge}: HUMAN labels == stub qrels",
              [r["label"] for r in labs["HUMAN"]]
              == [float(grades[k]) for k in keysets["HUMAN"]])
        mf = json.loads((jd / "manifest.json").read_text())
        hq = {}
        for n in ("200", "1000"):
            mix = labs[f"MIX_{n}"]
            src_by_q = defaultdict(set)
            for r in mix:
                src_by_q[(r["dataset"], r["qid"])].add(r["source"])
            hq[n] = {f"{a}|{b}" for (a, b), s in src_by_q.items()
                     if s == {"human"}}
            check(f"{judge}: MIX_{n} human part is query-disjoint, >= n_h",
                  all(len(s) == 1 for s in src_by_q.values())
                  and sum(r["source"] == "human" for r in mix) >= int(n))
            eq = labs[f"CORR_REFIT_n{n}"]
            check(f"{judge}: CORR_REFIT_n{n} is 100% LLM-derived",
                  all(r["source"] == f"corr_refit_n{n}" for r in eq))
            fitq = set(mf["equal_budget_rules"][f"CORR_REFIT_n{n}"]
                       ["fit_queries"])
            check(f"{judge}: CORR_REFIT_n{n} fitted exactly on MIX_{n}'s "
                  f"human queries", fitq == hq[n] ==
                  set(mf["mix"][f"MIX_{n}"]["human_queries"]))
            if judge != "stub/identity":
                lo, hi = sorted((m["HUMAN"], m["RAW"]))
                check(f"{judge}: MIX_{n} mean strictly between HUMAN and RAW",
                      lo < m[f"MIX_{n}"] < hi)
        check(f"{judge}: MIX nesting (MIX_200 queries ⊂ MIX_1000 queries)",
              hq["200"] < hq["1000"])
        check(f"{judge}: manifest has 2 refit fold rules, roles, rules sha, "
              f"per-dataset summary",
              len(mf["refit_rules"]) == 2 and "CORR_REFIT_n" in mf["set_roles"]
              and len(mf["rules_file"]["sha256"]) == 64
              and set(mf["summary"]["RAW"]["per_dataset"]) == {"dl19", "dl20"})
        want = {"stub/lenient": "ok", "stub/strict": "WARNING",
                "stub/identity": "rule has no delta_hat"}[judge]
        check(f"{judge}: Delta vs delta_hat check -> {want}",
              mf["delta_check"]["msmarco"]["status"].startswith(want),
              mf["delta_check"]["msmarco"]["status"][:60])
        check(f"{judge}: full coverage, no warning",
              mf["coverage_of_human_judged"] == 1.0
              and not mf["coverage_warning"])
    ml, ms = means["stub/lenient"], means["stub/strict"]
    check("lenient: RAW mean > HUMAN mean", ml["RAW"] > ml["HUMAN"])
    for n in ("CORR_FROZEN", "CORR_REFIT", "CORR_REFIT_n200",
              "CORR_REFIT_n1000"):
        check(f"lenient: {n} recovers HUMAN mean exactly",
              abs(ml[n] - ml["HUMAN"]) < 1e-12)
    check("lenient: MIX_1000 closer to HUMAN than MIX_200",
          abs(ml["MIX_1000"] - ml["HUMAN"]) < abs(ml["MIX_200"] - ml["HUMAN"]))
    check("strict: RAW mean < HUMAN mean", ms["RAW"] < ms["HUMAN"])
    check("strict: CORR_REFIT and CORR_REFIT_n200 recover HUMAN mean",
          abs(ms["CORR_REFIT"] - ms["HUMAN"]) < 1e-12
          and abs(ms["CORR_REFIT_n200"] - ms["HUMAN"]) < 1e-12)

    jd = root / "derived" / "batch7" / judge_slug("stub/identity")
    cf = read_jsonl(jd / "labels_CORR_FROZEN.jsonl")
    rw = read_jsonl(jd / "labels_RAW.jsonl")
    check("no-op judge: CORR_FROZEN labels == RAW labels bit-exactly",
          [repr(r["label"]) for r in cf] == [repr(r["label"]) for r in rw])
    raw_t = {("dl19", "1", "a"): 2, ("dl19", "1", "b"): 1}
    check("no-op invariant hard-fails on a tampered CORR set",
          exits(lambda: assert_noop_invariant(
              "x", "msmarco", {"dial_level": 3, "monotone_map": [0, 1, 2, 3]},
              raw_t, {("dl19", "1", "a"): 2, ("dl19", "1", "b"): 2})))
    check("non-no-op rule is not checked against RAW",
          not exits(lambda: assert_noop_invariant(
              "x", "msmarco", {"dial_level": 1, "monotone_map": [0, 0, 2, 3]},
              raw_t, {("dl19", "1", "a"): 3, ("dl19", "1", "b"): 0})))
    check("build-labels refuses dl22 as a training dataset",
          exits(lambda: guard_train_datasets(["dl19", "dl22"])))
    check("rules fitted on dl22 are refused",
          exits(lambda: guard_rules_file({"datasets_used": ["dl19", "dl22"],
                                          "rules": {}})))
    check("rerun with identical inputs is idempotent (no --force needed)",
          not exits(lambda: cmd_build_labels(bl_ns("stub/lenient"))))
    check("changed labels refuse to overwrite the manifest without --force",
          exits(lambda: cmd_build_labels(bl_ns("stub/lenient", mix_seed=7))))
    check("... and overwrite with --force",
          not exits(lambda: cmd_build_labels(bl_ns("stub/lenient",
                                                   mix_seed=7, force=True))))
    cmd_build_labels(bl_ns("stub/lenient", force=True))     # restore seed 42
    bad_pairs = tmp / "pairs_bad.jsonl"
    write_jsonl(bad_pairs, [{"dataset": "dl20", "qid": "1", "docid": "x",
                             "query": "q", "passage": "p"}])
    check("pairs file with a different dataset field is refused",
          exits(lambda: load_pairs_texts(bad_pairs, "dl19")))
    tx = {("dl19", "1", "a"): {"rel": 2}, ("dl19", "1", "b"): {"rel": 1}}
    rd = rel_disagreements(tx, {("dl19", "1", "a"): 2, ("dl19", "1", "b"): 3},
                           "ir_datasets")
    check("qrels-vs-pairs rel disagreement counted",
          rd["per_dataset"]["dl19"] == {"n_compared": 2, "n_disagree": 1})

    # -- 1b. leakage probes (core functions, perturbed human grades) ---------
    df = pd.read_parquet(root / "derived" / "judgments.parquet")
    dl = df[(df.model_id == "stub/lenient")
            & df.dataset.isin(["dl19", "dl20"])]
    tr_grades = {k: g for k, g in grades.items() if k[0] in ("dl19", "dl20")}
    base, _r = refit_corr_labels(dl, tr_grades, "stub/lenient",
                                 ["dl19", "dl20"])
    pert = {k: (3 - g if b4.fold_of(k[0], k[1], FOLD_SEED) == 0 else g)
            for k, g in tr_grades.items()}
    probe, _r2 = refit_corr_labels(dl, pert, "stub/lenient", ["dl19", "dl20"])
    f0 = [k for k in base if b4.fold_of(k[0], k[1], FOLD_SEED) == 0]
    f1 = [k for k in base if b4.fold_of(k[0], k[1], FOLD_SEED) == 1]
    check("refit no-leak: perturbing fold-0 human grades leaves fold-0 "
          "CORR_REFIT labels unchanged",
          all(base[k] == probe[k] for k in f0) and len(f0) > 0)
    check("refit probe is live: fold-1 labels DO change",
          any(base[k] != probe[k] for k in f1))
    texts = {}
    for ds in ("dl19", "dl20"):
        texts.update(load_pairs_texts(root / "inputs" / f"pairs_{ds}.jsonl",
                                      ds))
    rule = {"msmarco": dict(STUB_JUDGES["stub/lenient"],
                            datasets=["dl19", "dl20"])}
    s0, i0 = build_label_sets(dl, tr_grades, texts, "stub/lenient", rule,
                              ["dl19", "dl20"], [200])
    mixq = {tuple(x.split("|")) for x in i0["mix"]["MIX_200"]["human_queries"]}
    pert2 = {k: (g if (k[0], k[1]) in mixq else 3 - g)
             for k, g in tr_grades.items()}
    s1, i1 = build_label_sets(dl, pert2, texts, "stub/lenient", rule,
                              ["dl19", "dl20"], [200])
    check("equal-budget: perturbing non-MIX human grades leaves "
          "CORR_REFIT_n200 unchanged (fit uses MIX queries only)",
          s0["CORR_REFIT_n200"] == s1["CORR_REFIT_n200"])
    check("equal-budget probe is live: full CORR_REFIT does change",
          s0["CORR_REFIT"] != s1["CORR_REFIT"])
    _s, i_big = build_label_sets(dl, tr_grades, texts, "stub/lenient", rule,
                                 ["dl19", "dl20"], [200, 5000])
    check("MIX with n_h >= universe is skipped with a note (no exit)",
          "MIX_5000" not in _s and "CORR_REFIT_n5000" not in _s
          and any("MIX_5000" in n for n in i_big["notes"]))
    check("independent L3 recomputation agrees with judge_labels",
          independent_level_labels(dl, 3)
          == qualified_labels(dl, ["dl19", "dl20"], 3))

    # -- 1c. build-ablation (EXPLORATORY sets) --------------------------------
    def ab_ns(judge, **kw):
        ns = dict(data_dir=str(root), judge=judge, out_dir=None, force=False)
        ns.update(kw)
        return argparse.Namespace(**ns)

    def snap(jdir):
        return {n: (jdir / f"labels_{n}.jsonl").read_bytes() for n in ALL}

    def labmap(path):
        return {(r["dataset"], r["qid"], r["docid"]): r["label"]
                for r in read_jsonl(path)}

    def indep_mapped(dd, level, mono):
        ok = dd[(dd.politeness_level == level) & dd.parse_ok.astype(bool)
                & dd.score.notna()]
        s = np.asarray(mono, dtype=float)[ok.score.to_numpy().astype(int)]
        g = pd.Series(s).groupby([ok.dataset.astype(str).to_numpy(),
                                  ok.qid.astype(str).to_numpy(),
                                  ok.docid.astype(str).to_numpy()]).mean()
        return {k: float(np.clip(np.floor(v + 0.5), 0, 3)) for k, v in g.items()}

    # (i) non-trivial frozen rule: the noisy stub judge with dial L4 and map
    # [0,0,2,3] (own rules file + own labels root; the defaults stay intact)
    ab_map = [0, 0, 2, 3]
    rules_ab = json.loads((root / "derived" / "correction_rules.json")
                          .read_text())
    rules_ab["rules"]["stub/identity"]["msmarco"].update(
        dial_level=4, monotone_map=ab_map)
    rp_ab = tmp / "rules_ab.json"
    rp_ab.write_text(json.dumps(rules_ab))
    ab_root = tmp / "abl_out"
    cmd_build_labels(bl_ns("stub/identity", rules=str(rp_ab),
                           out_dir=str(ab_root)))
    ja = ab_root / judge_slug("stub/identity")
    reg_before = snap(ja)
    mf_before = json.loads((ja / "manifest.json").read_text())
    cmd_build_ablation(ab_ns("stub/identity", out_dir=str(ab_root)))
    mf_after = json.loads((ja / "manifest.json").read_text())
    check("ablation: registered label files byte-identical before/after",
          snap(ja) == reg_before)
    check("ablation: existing manifest keys unchanged; new keys only "
          "'ablation_sets' / 'ablation_meta'",
          {k: v for k, v in mf_after.items()
           if k not in ("ablation_sets", "ablation_meta")} == mf_before
          and set(mf_after["ablation_sets"]) == set(ABLATION_SETS))
    A = {n: read_jsonl(ja / f"labels_{n}.jsonl") for n in ABLATION_SETS}
    hum_recs = read_jsonl(ja / "labels_HUMAN.jsonl")
    check("ablation: 4 sets on the registered universe (same order), same "
          "record schema",
          all([(r["dataset"], r["qid"], r["docid"]) for r in v]
              == [(r["dataset"], r["qid"], r["docid"]) for r in hum_recs]
              and set(v[0]) == set(hum_recs[0]) for v in A.values()))
    dfa = pd.read_parquet(root / "derived" / "judgments.parquet")
    di = dfa[(dfa.model_id == "stub/identity")
             & dfa.dataset.isin(["dl19", "dl20"])]
    keys_u = [(r["dataset"], r["qid"], r["docid"]) for r in hum_recs]
    L = {n: labmap(ja / f"labels_{n}.jsonl")
         for n in ("RAW", "CORR_FROZEN") + ABLATION_SETS}
    ind4 = independent_level_labels(di, 4)
    ind3 = independent_level_labels(di, 3)
    m4, m3 = indep_mapped(di, 4, ab_map), indep_mapped(di, 3, ab_map)
    check("DIAL_ONLY = frozen dial (L4) without the map; CORR_FROZEN = the "
          "same L4 records WITH the map (independent pandas recomputation)",
          all(L["DIAL_ONLY"][k] == ind4[k] and L["CORR_FROZEN"][k] == m4[k]
              for k in keys_u))
    check("MAP_ONLY = L3 records with the frozen map per record; RAW = the "
          "same L3 records without it (independent pandas recomputation)",
          all(L["MAP_ONLY"][k] == m3[k] and L["RAW"][k] == ind3[k]
              for k in keys_u))
    ea = mf_after["ablation_sets"]
    check("non-trivial rule: DIAL_ONLY != CORR_FROZEN, MAP_ONLY != RAW, "
          "neither flagged degenerate",
          L["DIAL_ONLY"] != L["CORR_FROZEN"] and L["MAP_ONLY"] != L["RAW"]
          and not ea["DIAL_ONLY"]["degenerate"]
          and not ea["MAP_ONLY"]["degenerate"])
    raw_v = np.array([L["RAW"][k] for k in keys_u])
    mo_v = np.array([L["MAP_ONLY"][k] for k in keys_u])
    check("manifest RAW1→0 share for MAP_ONLY matches a recomputation",
          abs(ea["MAP_ONLY"]["summary"]["raw1_routed_to_0"]
              - float((np.floor(mo_v[raw_v == 1] + 0.5) == 0).mean()))
          < 1e-12)
    tampered = ja / "labels_RAW.jsonl"
    tampered.write_bytes(reg_before["RAW"] + b"\n")
    check("build-ablation refuses when a registered file no longer matches "
          "its manifest sha256",
          exits(lambda: cmd_build_ablation(ab_ns("stub/identity",
                                                 out_dir=str(ab_root)))))
    tampered.write_bytes(reg_before["RAW"])

    # (ii) SHIFT sets on the lenient judge (RAW offset +0.73)
    jl = root / "derived" / "batch7" / judge_slug("stub/lenient")
    reg_l = snap(jl)
    cmd_build_ablation(ab_ns("stub/lenient"))
    check("ablation (lenient): registered files byte-identical",
          snap(jl) == reg_l)
    mfl = json.loads((jl / "manifest.json").read_text())
    el = mfl["ablation_sets"]
    hl = labmap(jl / "labels_HUMAN.jsonl")
    rl = labmap(jl / "labels_RAW.jsonl")
    kl = sorted(hl)
    hv = np.array([hl[k] for k in kl])
    rv = np.array([rl[k] for k in kl])
    dfull = float(rv.mean() - hv.mean())
    sf = labmap(jl / "labels_SHIFT_FULL.jsonl")
    sv = np.array([sf[k] for k in kl])
    check("SHIFT_FULL: Δ̂_full = mean(RAW)-mean(HUMAN) = manifest summary "
          "(1e-6); mean(RAW-Δ̂)-mean(HUMAN) ≈ 0 before clipping (1e-6)",
          abs(el["SHIFT_FULL"]["delta_hat"] - dfull) < 1e-6
          and abs(dfull - mfl["summary"]["RAW"]["offset_vs_human"]) < 1e-6
          and abs(float((rv - dfull).mean() - hv.mean())) < 1e-6
          and abs(el["SHIFT_FULL"]["summary"]["offset_vs_human_unclipped"])
          < 1e-6)
    check("SHIFT_FULL: labels = clip(RAW-Δ̂,0,3) exactly, continuous (not "
          "rounded), offset after clipping small",
          np.array_equal(sv, np.clip(rv - dfull, 0, 3))
          and (sv != np.round(sv)).any() and sv.min() >= 0 and sv.max() <= 3
          and abs(float(sv.mean() - hv.mean())) < 0.05,
          f"clipped offset {float(sv.mean() - hv.mean()):+.6f}")
    dn = mfl["equal_budget_rules"]["CORR_REFIT_n1000"]["delta_hat"]
    s1k = labmap(jl / "labels_SHIFT_n1000.jsonl")
    s1v = np.array([s1k[k] for k in kl])
    check("SHIFT_n1000: Δ̂ = registered CORR_REFIT_n1000 delta_hat; labels "
          "= clip(RAW-Δ̂,0,3); unclipped offset = Δ̂_full - Δ̂_n1000",
          el["SHIFT_n1000"]["delta_hat"] == dn
          and np.array_equal(s1v, np.clip(rv - dn, 0, 3))
          and abs(el["SHIFT_n1000"]["summary"]["offset_vs_human_unclipped"]
                  - (dfull - dn)) < 1e-9)
    check("lenient (dial L2, identity map): DIAL_ONLY == CORR_FROZEN and "
          "MAP_ONLY == RAW, both flagged degenerate",
          el["DIAL_ONLY"]["equals_registered"][:1] == ["CORR_FROZEN"]
          and el["MAP_ONLY"]["equals_registered"][:1] == ["RAW"]
          and el["DIAL_ONLY"]["degenerate"] and el["MAP_ONLY"]["degenerate"])
    check("build-ablation rerun is idempotent (no --force needed)",
          not exits(lambda: cmd_build_ablation(ab_ns("stub/lenient"))))
    mfl["ablation_sets"]["SHIFT_FULL"]["sha256"] = "0" * 64
    (jl / "manifest.json").write_text(json.dumps(mfl, indent=2))
    check("changed ablation set refuses to overwrite without --force",
          exits(lambda: cmd_build_ablation(ab_ns("stub/lenient"))))
    check("... and overwrites with --force (registered files untouched)",
          not exits(lambda: cmd_build_ablation(ab_ns("stub/lenient",
                                                     force=True)))
          and snap(jl) == reg_l)

    # (iii) identity@L3 judge: DIAL_ONLY / MAP_ONLY == RAW, degenerate
    jdi = root / "derived" / "batch7" / judge_slug("stub/identity")
    reg_i = snap(jdi)
    cmd_build_ablation(ab_ns("stub/identity"))
    ei = json.loads((jdi / "manifest.json").read_text())["ablation_sets"]
    ri = [r["label"] for r in read_jsonl(jdi / "labels_RAW.jsonl")]
    check("identity@L3 judge: DIAL_ONLY and MAP_ONLY == RAW bit-exactly, "
          "manifest degenerate=true, registered files byte-identical",
          all([r["label"] for r in read_jsonl(jdi / f"labels_{n}.jsonl")]
              == ri and ei[n]["degenerate"]
              and "RAW" in ei[n]["equals_registered"]
              for n in ("DIAL_ONLY", "MAP_ONLY"))
          and snap(jdi) == reg_i
          and json.loads((jdi / "manifest.json").read_text())
          ["ablation_meta"]["frozen_rule_is_identity_at_L3"])

    # -- 2. metric unit checks ---------------------------------------------
    nd, ap = query_metrics(["a", "b", "c"], [3.0, 2.0, 1.0], [3, 1, 2])
    want = (7 + 1 / math.log2(3) + 3 / 2) / (7 + 3 / math.log2(3) + 1 / 2)
    check("standard NDCG hand example", abs(nd - want) < 1e-12)
    check("AP hand example (rel>=2 at ranks 1,3)",
          abs(ap - (1.0 + 2 / 3) / 2) < 1e-12)
    rng = np.random.default_rng(3)
    perfect = [(f"q{i}", f"d{j}", float(h), h) for i in range(8)
               for j, h in enumerate(rng.integers(0, 4, 10))]
    rp = evaluate_pairs(perfect, b=200)["point"]
    check("pred == human: NDCG 1, MAE 0, F1 1, ratio 1, offset 0, ECE 0",
          abs(rp["ndcg10"] - 1) < 1e-12 and rp["mae"] == 0
          and rp["f1"] == 1 and rp["rel_rate_ratio"] == 1
          and rp["offset"] == 0 and rp["ece4"] == 0)
    rs = evaluate_pairs([(q, d, p + 1.0, h) for q, d, p, h in perfect],
                        b=200)["point"]
    check("pred = human+1 (unclipped): NDCG unchanged, offset>0, ratio>1, "
          "recall 1, precision<1",
          abs(rs["ndcg10"] - 1) < 1e-12 and rs["offset"] > 0
          and rs["rel_rate_ratio"] > 1 and rs["recall"] == 1
          and rs["precision"] < 1)

    # -- 3. gate (real git repo in the scratch dir) --------------------------
    repo = tmp / "gitrepo"
    repo.mkdir()
    saved = PREDICTIONS_PATH
    try:
        _git(repo, "init", "-q")
        P = repo / "PREDICTIONS.md"
        PREDICTIONS_PATH = P
        P.write_text("# nothing registered\n")
        _git(repo, "add", "PREDICTIONS.md")
        _git(repo, "commit", "-q", "-m", "init")
        check("dl22 gate: refuses without flag and heading",
              exits(lambda: gate_test_dataset("dl22", False)))
        check("dl22 gate: refuses with flag but without heading",
              exits(lambda: gate_test_dataset("dl22", True)))
        P.write_text(f"some text mentioning {B7_MARKER} in a body line\n")
        _git(repo, "commit", "-q", "-am", "body only")
        check("dl22 gate: marker in body text (not a heading) is refused",
              exits(lambda: gate_test_dataset("dl22", True)))
        P.write_text(f"# {B7_MARKER}（预测）\n\npredictions...\n")
        check("dl22 gate: heading present but UNCOMMITTED is refused",
              exits(lambda: gate_test_dataset("dl22", True)))
        _git(repo, "commit", "-q", "-am", "register")
        check("dl22 gate: committed heading but no flag is refused",
              exits(lambda: gate_test_dataset("dl22", False)))
        check("dl22 gate: committed heading AND flag passes",
              not exits(lambda: gate_test_dataset("dl22", True)))
        check("non-sealed dataset needs no gate",
              not exits(lambda: gate_test_dataset("dl21", False)))
        gate_ok = True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        check("git available for the gate checks", False, repr(e))
        gate_ok = False
    finally:
        PREDICTIONS_PATH = saved

    # -- 3b. --tag / --head-bias plumbing (no torch) ------------------------
    check("--tag: label_set becomes <SET>@<tag>; no tag = unchanged; bad tag "
          "refused",
          tagged_set_name("SHIFT_FULL", "hb") == "SHIFT_FULL@hb"
          and tagged_set_name("HUMAN", None) == "HUMAN"
          and exits(lambda: tagged_set_name("HUMAN", "a_s1"))
          and exits(lambda: tagged_set_name("HUMAN", "")))
    yy = np.array([0.0, 1.0, 2.0, 0.5], dtype=np.float32)
    check("--head-bias: default '1.5' is the registered bias; label_mean = "
          "mean of the given labels; ordinal refuses a non-default bias",
          resolve_head_bias(HEAD_BIAS_DEFAULT, "regression", yy)
          == ("default", HEAD_BIAS_REGRESSION)
          and resolve_head_bias(HEAD_BIAS_DEFAULT, "ordinal-threshold", yy)
          == ("default", 0.0)
          and resolve_head_bias("label_mean", "regression", yy)
          == ("label_mean", 0.875)
          and resolve_head_bias("0.9", "regression", yy) == ("float", 0.9)
          and exits(lambda: resolve_head_bias("label_mean",
                                              "ordinal-threshold", yy))
          and exits(lambda: resolve_head_bias("x", "regression", yy)))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "b7summary", Path(__file__).resolve().parent / "batch7_summary.py")
    b7s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b7s)
    sroot = tmp / "sum_eval"
    sd_ = sroot / "minilm-raw" / "deepseek__deepseek-v4-flash"
    sd_.mkdir(parents=True)
    for nm in ("SHIFT_FULL@hb_s1", "SHIFT_FULL@hb_s2", "SHIFT_FULL_s1",
               "SHIFT_n1000_s3"):
        (sd_ / f"{nm}_dl22.json").write_text("{}")
    sev = b7s.load_evals(sroot)["minilm-raw"]["deepseek__deepseek-v4-flash"]
    check("summary parser: '<SET>@<tag>_s<seed>' parses to a distinct set "
          "and is listed as an extra (ablation) set",
          set(sev) == {"SHIFT_FULL@hb", "SHIFT_FULL", "SHIFT_n1000"}
          and sorted(sev["SHIFT_FULL@hb"]) == [1, 2]
          and b7s.extra_sets(sev) == ("SHIFT_FULL@hb",))

    # -- 4. torch side -----------------------------------------------------
    try:
        import torch
        import transformers  # noqa: F401
        have_torch = True
    except ImportError:
        have_torch = False
    if not have_torch:
        check("torch + transformers importable (train/evaluate checks)", False,
              "pip install -r requirements-batch7.txt")
    else:
        t = torch.tensor([[2.0, -1.0, 0.5]])
        y = torch.tensor([1.5])
        tg = torch.tensor([1.0, 0.5, 0.0])
        sg = torch.sigmoid(t[0])
        manual = float((-(tg * torch.log(sg) + (1 - tg) * torch.log(1 - sg)))
                       .mean())
        check("ordinal target math: BCE vs clip(y-k,0,1) targets",
              abs(float(objective_loss(t, y, "ordinal-threshold")) - manual)
              < 1e-6)
        check("ordinal score = sum of sigmoids",
              abs(float(logits_to_score(t, "ordinal-threshold")[0])
                  - float(sg.sum())) < 1e-6)
        check("regression loss = MSE",
              abs(float(objective_loss(torch.tensor([[2.0]]), y,
                                       "regression")) - 0.25) < 1e-9)

        tiny = tmp / "tiny_bert"
        _make_tiny_model(tiny)
        for obj in ("regression", "ordinal-threshold"):
            mA, _ = load_student(tiny, n_outputs(obj))
            mB, _ = load_student(tiny, n_outputs(obj))
            reset_head(mA, obj, 5)
            reset_head(mB, obj, 5)
            _n, la = output_head(mA, n_outputs(obj))
            _n, lb = output_head(mB, n_outputs(obj))
            bias_ok = bool(torch.all(la.bias == (1.5 if obj == "regression"
                                                 else 0.0)))
            check(f"head reset [{obj}]: deterministic, bias "
                  f"{'1.5' if obj == 'regression' else '0'}, small weights",
                  torch.equal(la.weight, lb.weight) and bias_ok
                  and float(la.weight.detach().abs().max()) < 0.1)

        jd = root / "derived" / "batch7" / judge_slug("stub/lenient")
        allr = [r for r in read_jsonl(jd / "labels_RAW.jsonl")
                if r["dataset"] == "dl19"]
        va = [r for r in allr if val_fold(r["dataset"], r["qid"])][:16]
        tr = [r for r in allr if not val_fold(r["dataset"], r["qid"])]
        sub = va + tr[:64 - len(va)]
        sub_p = jd / "labels_RAW64.jsonl"
        write_jsonl(sub_p, sub)

        def train_ns(obj, out, seed=1, **kw):
            ns = dict(labels=str(sub_p), arm=TINY_ARM, model=str(tiny),
                      out=str(out), seed=seed, epochs=1, lr=1e-3, bs=16,
                      max_len=32, objective=obj, device="cpu", no_fp16=True,
                      weight_decay=0.01, warmup=0.1, log_every=100,
                      select="last")
            ns.update(kw)
            return argparse.Namespace(**ns)

        mroot = tmp / "models" / TINY_ARM
        outs = {"reg_s1": mroot / "RAW_s1", "reg_s1b": mroot / "RAW_s1_rerun",
                "reg_s2": mroot / "RAW_s2", "ord_s1": mroot / "RAW_ord_s1"}
        cmd_train(train_ns("regression", outs["reg_s1"], epochs=2))
        cmd_train(train_ns("regression", outs["reg_s1b"], epochs=2))
        cmd_train(train_ns("regression", outs["reg_s2"], seed=2, epochs=2))
        cmd_train(train_ns("ordinal-threshold", outs["ord_s1"]))
        for key, out in outs.items():
            tl = json.loads((out / "train_log.json").read_text())
            check(f"train[{key}]: outputs, finite loss, label mean, arm, "
                  f"hyperparams logged",
                  (out / "batch7_config.json").exists()
                  and len(tl["loss_curve"]) > 0
                  and all(np.isfinite(x["loss"]) for x in tl["loss_curve"])
                  and abs(tl["label_mean_all"]
                          - np.mean([r["label"] for r in sub])) < 1e-6
                  and tl["arm"] == TINY_ARM and tl["hyperparams"]["lr"] == 1e-3)
            check(f"train[{key}]: query-disjoint non-empty validation; epoch-0 "
                  f"val pred mean ≈ 1.5 after head reset; select=last",
                  tl["val_query_disjoint"] and tl["n_val"] > 0
                  and tl["val"][0]["epoch"] == 0
                  and abs(tl["val"][0]["val_pred_mean"] - 1.5) < 0.5
                  and tl["selected_epoch"] == tl["hyperparams"]["epochs"]
                  and all(v["train_loss_epoch"] is not None
                          for v in tl["val"][1:]),
                  f"epoch-0 pred mean {tl['val'][0]['val_pred_mean']:.3f}")
        sh_all = [r for r in read_jsonl(jd / "labels_SHIFT_FULL.jsonl")
                  if r["dataset"] == "dl19"]
        sh_keys = {(r["qid"], r["docid"]) for r in sub}
        sh_sub = [r for r in sh_all if (r["qid"], r["docid"]) in sh_keys]
        sh_p = jd / "labels_SHIFT64.jsonl"
        write_jsonl(sh_p, sh_sub)
        out_sh = mroot / "SHIFT_FULL_s1"
        cmd_train(train_ns("regression", out_sh, labels=str(sh_p)))
        tl = json.loads((out_sh / "train_log.json").read_text())
        check("train accepts continuous (SHIFT_FULL) float targets: label "
              "mean logged exactly, finite loss, set name recorded",
              any(r["label"] != round(r["label"]) for r in sh_sub)
              and abs(tl["label_mean_all"]
                      - np.mean([r["label"] for r in sh_sub])) < 1e-6
              and all(np.isfinite(x["loss"]) for x in tl["loss_curve"])
              and tl["label_set"] == "SHIFT_FULL"
              and tl["hyperparams"]["head_bias_mode"] == "default"
              and tl["hyperparams"]["head_bias_value"] == HEAD_BIAS_REGRESSION
              and tl["head_reset"]["bias"] == HEAD_BIAS_REGRESSION)
        out_hb = mroot / "SHIFT_FULL@hb_s1"
        cmd_train(train_ns("regression", out_hb, labels=str(sh_p),
                           head_bias="label_mean", tag="hb"))
        tlh = json.loads((out_hb / "train_log.json").read_text())
        check("train --head-bias label_mean --tag hb: bias = train-split label "
              "mean (logged), label_set SHIFT_FULL@hb, same labels sha",
              tlh["hyperparams"]["head_bias_mode"] == "label_mean"
              and abs(tlh["hyperparams"]["head_bias_value"]
                      - tlh["label_mean_train"]) < 1e-6
              and tlh["head_reset"]["bias"]
              == tlh["hyperparams"]["head_bias_value"]
              and abs(tlh["val"][0]["val_pred_mean"]
                      - tlh["label_mean_train"]) < 0.5
              and tlh["label_set"] == "SHIFT_FULL@hb"
              and tlh["label_set_base"] == "SHIFT_FULL"
              and tlh["labels_sha256"] == tl["labels_sha256"])
        from safetensors.torch import load_file
        wa = load_file(str(outs["reg_s1"] / "model.safetensors"))
        wb = load_file(str(outs["reg_s1b"] / "model.safetensors"))
        check("same-seed determinism: two CPU runs give identical weights",
              wa.keys() == wb.keys()
              and all(torch.equal(wa[k], wb[k]) for k in wa))
        check("train refuses an --out without the arm directory",
              exits(lambda: cmd_train(train_ns("regression",
                                               tmp / "noarm" / "x"))))
        check("arm/base-model mismatch is refused",
              exits(lambda: resolve_arm("msmarco-ce",
                                        "nreimers/MiniLM-L6-H384-uncased")))
        check("arm defaults: minilm-raw 5 ep / 3e-5, msmarco-ce 2 ep / 2e-5, "
              "legacy id aliased",
              resolve_arm(None, "cross-encoder/ms-marco-MiniLM-L-6-v2")[0]
              == "msmarco-ce"
              and arm_hparam("minilm-raw", "epochs", None) == (5, True)
              and arm_hparam("msmarco-ce", "lr", None) == (2e-5, True))

        sw = tmp / "sweep" / TINY_ARM / "sweep.json"
        cmd_sweep(argparse.Namespace(
            labels=str(jd / "labels_HUMAN.jsonl"), arm=TINY_ARM,
            model=str(tiny), grid="epochs=1,2;lr=1e-3,2e-3", out=str(sw),
            seed=1, bs=64, max_len=32, objective="regression",
            weight_decay=0.01, warmup=0.1, device="cpu", no_fp16=True))
        swj = json.loads(sw.read_text())
        check("sweep: 4 configs, pick recorded, HUMAN only",
              len(swj["rows"]) == 4 and "epochs" in swj["pick"])
        check("sweep refuses a non-HUMAN label set",
              exits(lambda: cmd_sweep(argparse.Namespace(
                  labels=str(sub_p), arm=TINY_ARM, model=str(tiny),
                  grid="epochs=1;lr=1e-3", out=str(sw), seed=1, bs=64,
                  max_len=32, objective="regression", weight_decay=0.01,
                  warmup=0.1, device="cpu", no_fp16=True))))

        evs = {}
        if gate_ok:
            saved = PREDICTIONS_PATH
            try:
                PREDICTIONS_PATH = repo / "PREDICTIONS.md"
                for key, mdir in outs.items():
                    ev = tmp / "eval" / TINY_ARM / f"{key}_dl22.json"
                    cmd_evaluate(argparse.Namespace(
                        model_dir=str(mdir), dataset="dl22", unseal_dl22=True,
                        data_dir=str(root), pairs=None,
                        qrels_source="pairs-rel", device="cpu", b=200,
                        seed=BOOT_SEED, out=str(ev)))
                    evs[key] = json.loads(ev.read_text())
                r = evs["reg_s1"]
                check("evaluate: end-to-end on stub dl22, fp32, provenance "
                      "fields recorded",
                      r["n_pairs"] == 120 and r["n_queries"] == 12
                      and 0 <= r["point"]["ndcg10"] <= 1
                      and r["scoring"] == "fp32" and r["arm"] == TINY_ARM
                      and len(r["labels_sha256"]) == 64
                      and r["selected_epoch"] == 2
                      and r["hyperparams"]["epochs"] == 2)
                check("evaluate: predictions vary across pairs",
                      len({round(x[2], 6) for x in r["pairs"]}) > 1)
                check("evaluate: same-seed reruns give identical predictions",
                      [x[2] for x in r["pairs"]]
                      == [x[2] for x in evs["reg_s1b"]["pairs"]])
                check("evaluate refuses an in-sample (training) dataset",
                      exits(lambda: cmd_evaluate(argparse.Namespace(
                          model_dir=str(outs["reg_s1"]), dataset="dl19",
                          unseal_dl22=False, data_dir=str(root), pairs=None,
                          qrels_source="pairs-rel", device="cpu", b=10,
                          seed=1, out=str(tmp / "eval" / TINY_ARM / "x.json")))))
                check("evaluate refuses an --out without the arm directory",
                      exits(lambda: cmd_evaluate(argparse.Namespace(
                          model_dir=str(outs["reg_s1"]), dataset="dl22",
                          unseal_dl22=True, data_dir=str(root), pairs=None,
                          qrels_source="pairs-rel", device="cpu", b=10,
                          seed=1, out=str(tmp / "x.json")))))
                P = repo / "PREDICTIONS.md"
                P.write_text(P.read_text() + "\nedited after commit\n")
                check("evaluate refuses dl22 when PREDICTIONS.md is dirty",
                      exits(lambda: cmd_evaluate(argparse.Namespace(
                          model_dir=str(outs["reg_s1"]), dataset="dl22",
                          unseal_dl22=True, data_dir=str(root), pairs=None,
                          qrels_source="pairs-rel", device="cpu", b=10,
                          seed=1, out=str(tmp / "eval" / TINY_ARM / "y.json")))))
            finally:
                PREDICTIONS_PATH = saved

        if evs:
            r = evs["reg_s1"]
            same = compare_results([r], [r], b=200, allow_self=True)
            check("compare(A, A): every diff and CI bound exactly 0",
                  all(e["diff"] == 0.0 and e["ci90"] == [0.0, 0.0]
                      for e in same["endpoints"].values()
                      if np.isfinite(e["a"])))
            check("compare(A, A): NDCG declared equivalent",
                  same["tost_ndcg10"]["equivalent"])
            check("compare refuses the same student on both sides",
                  exits(lambda: compare_results([r], [r], b=10)))
            check("compare refuses a duplicate within a side",
                  exits(lambda: compare_results([r, r], [evs["ord_s1"]], b=10)))
            mixed_obj = dict(evs["reg_s2"], objective="ordinal-threshold")
            check("compare refuses a side mixing objectives",
                  exits(lambda: compare_results([r, mixed_obj],
                                                [evs["ord_s1"]], b=10)))
            other = dict(evs["ord_s1"], arm="other-arm")
            check("compare refuses cross-arm without --allow-cross-arm",
                  exits(lambda: compare_results([r], [other], b=10)))
            check("... and allows it with the flag",
                  not exits(lambda: compare_results([r], [other], b=10,
                                                    allow_cross_arm=True)))
            bad_u = dict(evs["ord_s1"], universe_sha256="0" * 64)
            check("compare refuses a pair-universe mismatch",
                  exits(lambda: compare_results([r], [bad_u], b=10)))
            two = compare_results([r, evs["reg_s2"]], [evs["ord_s1"]], b=200)
            check("compare with seed lists (A = seeds 1,2) gives finite CIs",
                  all(np.isfinite(two["endpoints"][k]["ci90"]).all()
                      for k in ("ndcg10", "mae", "offset")))
            e1 = tmp / "eval" / TINY_ARM / "reg_s1_dl22.json"
            check("compare CLI refuses the same file twice",
                  exits(lambda: cmd_compare(argparse.Namespace(
                      a=f"{e1},{e1}", b=str(tmp / "eval" / TINY_ARM /
                                            "ord_s1_dl22.json"),
                      b_boot=10, seed=1, margin=0.01, allow_cross_arm=False,
                      out=None))))
            cmd_compare(argparse.Namespace(
                a=f"{e1},{tmp / 'eval' / TINY_ARM / 'reg_s2_dl22.json'}",
                b=str(tmp / "eval" / TINY_ARM / "ord_s1_dl22.json"),
                b_boot=100, seed=BOOT_SEED, margin=TOST_MARGIN,
                allow_cross_arm=False, out=str(tmp / "cmp.json")))
            check("compare CLI path writes its JSON",
                  (tmp / "cmp.json").exists())

    print()
    if failures:
        sys.exit(f"SELF-TEST FAILED: {len(failures)} check(s): {failures}\n"
                 f"  (scratch dir kept for inspection: {tmp})")
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print("  SELF-TEST PASSED")


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def _add_train_common(p) -> None:
    p.add_argument("--arm", default=None,
                   help=f"one of {sorted(ARMS)} (default: derived from --model)")
    p.add_argument("--model", default=None,
                   help="base model (default: the arm's model)")
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--objective", choices=["regression", "ordinal-threshold"],
                   default="regression")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.1,
                   help="linear warmup fraction of total steps")
    p.add_argument("--device", default="auto")
    p.add_argument("--no-fp16", action="store_true",
                   help="disable AMP fp16 during TRAINING (eval is always fp32)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build-labels", help="training label sets for one judge")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--train-datasets", default="dl19,dl20,dl21")
    p.add_argument("--judge", required=True, help="model_id")
    p.add_argument("--rules", default=None,
                   help="correction_rules.json (default "
                        "$DATA_DIR/derived/correction_rules.json)")
    p.add_argument("--pairs-dir", default=None,
                   help="dir with pairs_<ds>.jsonl (default $DATA_DIR/inputs)")
    p.add_argument("--qrels-source", choices=["ir_datasets", "pairs-rel"],
                   default="ir_datasets",
                   help="human grades: src.qrels.load_qrels (default) or the "
                        "pairs file's raw 'rel' field (round-2 files only)")
    p.add_argument("--mix-sizes", default="200,1000")
    p.add_argument("--mix-seed", type=int, default=MIX_SEED)
    p.add_argument("--fold-seed", type=int, default=FOLD_SEED,
                   help="CORR_REFIT fold seed (batch-4 default 42)")
    p.add_argument("--out-dir", default=None,
                   help="default $DATA_DIR/derived/batch7 (a <judge_slug>/ "
                        "subdir is created)")
    p.add_argument("--force", action="store_true",
                   help="overwrite label files that would change")
    p.set_defaults(func=cmd_build_labels)

    p = sub.add_parser("build-ablation",
                       help="EXPLORATORY: SHIFT_FULL / SHIFT_n1000 / DIAL_ONLY "
                            "/ MAP_ONLY on the registered universe")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--judge", required=True, help="model_id")
    p.add_argument("--out-dir", default=None,
                   help="labels root used by build-labels (default "
                        "$DATA_DIR/derived/batch7)")
    p.add_argument("--force", action="store_true",
                   help="overwrite ablation label files that would change")
    p.set_defaults(func=cmd_build_ablation)

    p = sub.add_parser("sweep", help="per-arm hyper-parameter grid on HUMAN")
    p.add_argument("--labels", required=True, help="labels_HUMAN.jsonl")
    p.add_argument("--grid", default="epochs=3,5,8;lr=2e-5,3e-5,5e-5")
    p.add_argument("--out", required=True, help="JSON; path must contain <arm>/")
    p.add_argument("--seed", type=int, default=1)
    _add_train_common(p)
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("train", help="fine-tune a cross-encoder on one label set")
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True, help="model dir; must contain <arm>/")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=int, default=None,
                   help="default: the arm's (minilm-raw 5, msmarco-ce 2)")
    p.add_argument("--lr", type=float, default=None,
                   help="default: the arm's (minilm-raw 3e-5, msmarco-ce 2e-5)")
    p.add_argument("--select", choices=["last", "best"], default="last",
                   help="'last' (default, fixed epochs) or 'best' (lowest "
                        "validation loss; robustness option)")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--head-bias", default=HEAD_BIAS_DEFAULT,
                   help="regression head bias init: 1.5 (registered default), "
                        "label_mean (train-split label mean; EXPLORATORY) or "
                        "a float")
    p.add_argument("--tag", default=None,
                   help="EXPLORATORY variant tag: label_set becomes <SET>@<tag>")
    _add_train_common(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="score a test set vs human qrels (fp32)")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--unseal-dl22", action="store_true")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--pairs", default=None,
                   help="default $DATA_DIR/inputs/pairs_<dataset>.jsonl")
    p.add_argument("--qrels-source", choices=["ir_datasets", "pairs-rel"],
                   default="ir_datasets")
    p.add_argument("--device", default="auto")
    p.add_argument("--b", type=int, default=BOOT_B)
    p.add_argument("--seed", type=int, default=BOOT_SEED)
    p.add_argument("--out", required=True, help="JSON; path must contain <arm>/")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("compare", help="paired-query bootstrap A - B")
    p.add_argument("--a", required=True, help="eval JSON(s), comma-separated")
    p.add_argument("--b", required=True, help="eval JSON(s), comma-separated")
    p.add_argument("--b-boot", type=int, default=BOOT_B)
    p.add_argument("--seed", type=int, default=BOOT_SEED)
    p.add_argument("--margin", type=float, default=TOST_MARGIN)
    p.add_argument("--allow-cross-arm", action="store_true")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("selftest", help="synthetic offline CPU self-test")
    p.set_defaults(func=cmd_selftest)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
