"""Batch 6 / A — LABEL-POOL EXTENSION with an LLM judge, on the human scale.

Practitioner scenario
---------------------
You hold a small set of human relevance labels and use an LLM judge to extend
the labeled pool.  The extended labels are then USED ON THE HUMAN SCALE: a
"relevant" threshold, a reported relevant-rate, training targets.  This script
asks whether the operating-point correction (tone dial + monotone output map,
scripts/batch4_correction.py) makes those labels usable on the human scale.

Framing (unchanged from batch 4/5): any gain is the removal of a MEASUREMENT
ARTIFACT (operating-point misalignment with the reference annotators), never a
judgment-quality improvement.

Label sets (per judge x dataset, on ONE pair universe)
------------------------------------------------------
Pair universe = human-judged pairs  AND  pairs the judge has a usable label for
(``parse_ok`` & numeric score) at EVERY politeness level L1..L5.  All five
levels are required because CORR-REFIT may pick any level as its dial; a
universe that depended on the fitted dial would make the label sets
non-comparable.  (Batch-5 top-up pairs, collected at L1+L3 only, therefore
never enter the universe.)

  HUMAN        qrels grade, 0-3 (ANTIQUE arrives already shifted 1-4 -> 0-3 by
               src/qrels.py).
  RAW          judge label at L3: mean over ALL records of the level
               (paraphrases x runs; identical to the paraphrase-mean when the
               design is balanced), round-half-up
               (scripts/batch5_downstream.judge_labels).
  CORR-FROZEN  frozen family rule from derived/correction_rules.json (msmarco
               family for dl21/dl22, antique family for antique): labels taken
               at the rule's dial level with its monotone map applied PER
               RECORD, BEFORE the mean and the rounding (batch 5 order).
  CORR-REFIT   in-domain, query-disjoint 2-fold cross-fit on the SAME dataset
               (folds = md5("seed|dataset|qid") as in batch 4): on the train
               fold fit the dial (argmin |Delta + D(l)|) and the best of the 35
               monotone maps (max paraphrase-mean linear kappa at the dial),
               both with batch 4's own estimators; apply to the test fold;
               concatenate the two test folds.  Primary fold seed = first of
               --fold-seeds (default 42); the other seeds are a robustness
               report of FIT variability (mean +- sd across seeds).

In-sample flags: the msmarco family rule was fitted on dl19+dl20+dl21, so on
dl21 CORR-FROZEN is PARTIALLY in-sample; the antique rule was fitted on antique
itself, so on antique CORR-FROZEN is FULLY in-sample.  dl22 is the only
held-out test of the frozen rule.

Endpoints (each label set vs HUMAN)
-----------------------------------
  1. binary relevance at the human threshold >=2 (primary) and >=1
     (secondary): precision, recall, F1, relevant-rate ratio
     |LLM-rel| / |HUMAN-rel| and |ratio - 1|;
  2. grade MAE, mean offset of the rounded labels (LLM mean - human mean),
     the CONTINUOUS Delta (record-mean score - human mean, the batch-4
     estimand) per label set, and a 4-bin calibration table;
  3. linear-weighted kappa on the PAIR labels (fast path, hard cross-checked
     against sklearn), plus the authoritative paraphrase-mean kappa at L3 on
     the universe as a bridge number.
Uncertainty: query-clustered bootstrap (B=1000, seed 42), 90% percentile CI
for every scalar endpoint and for the PAIRED differences FROZEN-RAW,
REFIT-RAW and REFIT-FROZEN.  The bootstrap is EVALUATION-ONLY (the REFIT rule
is fitted once per fold seed); fit variability is shown by the fold-seed
spread and by --sample-efficiency.

Sample-efficiency mode (--sample-efficiency)
--------------------------------------------
For n in {50,100,200,400} human-labeled pairs (B=1000 draws): queries are
drawn WITHOUT replacement and taken whole until n pairs are reached; the last
query is truncated to make exactly n.  Its left-over pairs are dropped from
BOTH sides, so the calibration set and the extension pool are query-disjoint
(checked on the rows actually fed to the fit).  Dial + map are refitted on the
n pairs and applied to the extension pool.  Reported: mean +- sd of P@2, R@2,
F1@2 and the relevant-rate ratio for RAW, CORR-FROZEN and CORR-REFIT.

Registration, gating and adjudication
-------------------------------------
* ``paste`` prints the B6 registration block WITHOUT loading qrels.
* ``run`` on ANY dataset requires the registered block (heading containing
  "B6 预注册 — 标注池扩充") unless ``--non-blind`` is given, which stamps
  NON-BLIND into the header, the JSON meta and the tex comment.
* dl22 (src.qrels.SEALED) additionally requires ``--unseal-dl22``, a clean and
  committed paper/PREDICTIONS.md, and a B6 block with no "____" left;
  ``--non-blind`` never relaxes the dl22 gate.
* The block records "rules_sha256: <hex>"; ``run`` aborts if it does not
  match derived/correction_rules.json.
* Registered predictions are parsed from the block and scored by the rules in
  ADJUDICATION_RULES (printed verbatim in PASTE and in BACKFILL).

No-op invariant: for any judge whose frozen rule is dial=L3 + identity map,
CORR-FROZEN must equal RAW BIT-EXACTLY on every endpoint, CI and paired
difference; otherwise the script exits before writing anything.

Follow-up (``followup``; EXPLORATORY — not pre-registered)
---------------------------------------------------------
Evaluator follow-up on the SAME universe as ``run`` (recomputed from the
judgments and hard-checked against derived/batch6_labels_<ds>), fold seed =
first --fold-seeds (42).  No registration gate; dl22 still needs
--unseal-dl22.  All fits/evaluations use per-query sufficient statistics, so
bootstrap and sub-sampling replicates are matrix products.
  (A) 2x2 factorial dial {L3, chosen} x map {identity, chosen} for FROZEN
      (frozen choices) and REFIT (per-fold cross-fit choices; the MAP arm
      applies the map fitted at the fitted dial) + a supplementary MAP@L3
      arm (map refitted at L3).  Attribution: dial = DIAL-RAW, map = MAP-RAW,
      interaction = (FULL-DIAL)-(MAP-RAW); eval-only query-bootstrap CIs.
  (B) operating curve: L3 vote sum v = K*mean (K = modal L3 records/pair),
      positive iff v >= t, t = 1..3K; overlaid with the in-sample
      (L1..L5) x 35-map grid; FROZEN / REFIT / F1-REFIT compared with the
      curve's F1@2 at matched ratio@2 (interpolated in (n_pos, TP)).
  (C) F1-objective refit: cross-fit grid argmax of train pair-label F1@2
      (ties -> train pair-label kappa -> RAW-first order); eval-only CIs plus
      a NESTED bootstrap (cross-fit redone in each query-resampled replicate)
      for fit-aware CIs of dF1@2 vs RAW, both objectives.
  (E) E6': calibrate on q whole queries (q in 2,4,8,16,32; simple random
      sample, B=1000), evaluate dF1@2 (REFIT-RAW) on the remaining queries:
      mean, median, fraction < -0.05; ">=80% of the q=32 gain by q=8" stated
      only if the q=32 mean gain > TIE_F1.

Follow-up 2 (``followup2``; EXPLORATORY — not pre-registered)
-------------------------------------------------------------
The evaluator's re-run requests on ``followup`` (which is left unchanged and
re-used, via followup_cell(), for every shared section; same dl22 gate plus
the B6 registration heading):
  (R) RETHRESH baseline: query-disjoint 2-fold cross-fit (same folds / fold
      seed as REFIT) of ONE threshold t on the L3 vote sum, t chosen by train
      binary kappa [RETHRESH_K] or train F1@2 [RETHRESH_F1] (ties -> RAW t,
      then lower t); held-out F1/P/R/ratio@2 and dF1@2 REFIT-RETHRESH_K,
      F1REFIT-RETHRESH_F1, FROZEN-RETHRESH_{K,F1} with eval-only and
      fit-aware (nested; every fitted arm re-fitted per replicate) CIs.
  (B2) curve verdicts for every arm incl. FROZEN:DIAL / FROZEN:MAP; the grid
      "ceiling" is reported as an optimistic in-sample reference (NOT an
      upper bound) beside the in-sample best L3 re-threshold, with flags.
  (S) fold-seed robustness (--fold-seeds, default 42 only): REFIT 2x2
      attribution, curve verdicts and C/R paired dF1@2 point estimates per
      seed, with sign flips vs the primary seed.
  (E) E6' plus the RAW-fallback rate; q/n_queries finite-population flags.
  Header: multiplicity / non-independence, in-sample status, flagship
  subsample, replicates per nested-bootstrap tail (warning if < 50).

Subcommands
-----------
    paste     --datasets dl22,dl21,antique
    validate  --datasets dl21,antique          (counts only, no agreement)
    run       --datasets dl21,antique [--sample-efficiency] [--non-blind]
    run       --datasets dl22 --unseal-dl22
    followup  --datasets dl22,dl21,antique --unseal-dl22 [--judges ...]
              [--b 1000] [--nested-b 200] [--seed 42] [--e6p-b 1000]
    followup2 --datasets dl22,dl21,antique --unseal-dl22 [--judges ...]
              [--fold-seeds 42[,43,44,45,46]] [--nested-b 200|1000] [--b 1000]
              [--e6p-b 1000] [--q-sizes 2,4,8,16,32]
    selftest  [--keep-dir DIR]

Outputs (under DATA_DIR — this repo carries no data artefacts):
    $DATA_DIR/derived/batch6_poolext_<ds>.json
    $DATA_DIR/derived/batch6_labels_<ds>.parquet   (.pkl if no parquet engine)
    $DATA_DIR/derived/tex/batch6_poolext_<ds>.tex
    $DATA_DIR/derived/tex/batch6_poolext_<ds>_sampleff.tex  (--sample-efficiency)
  followup:
    $DATA_DIR/derived/batch6_followup_<ds>.json
    $DATA_DIR/derived/batch6_opcurve_<ds>_<judge_slug>.csv
    $DATA_DIR/derived/tex/batch6_followup_<ds>_{factorial,attrib,opcurve,
                                               f1refit,e6p}.tex
  followup2:
    $DATA_DIR/derived/batch6_followup2_<ds>.json
    $DATA_DIR/derived/tex/batch6_followup2_<ds>_{rethresh,opcurve,seeds,
                                                e6p}.tex
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.qrels import assert_not_sealed, load_qrels, parse_datasets  # noqa: E402
from batch1r_stats import esc, kappa_from_cm, write_tex  # noqa: E402
from batch4_correction import (  # noqa: E402
    IDENTITY_MAP,
    LEVELS,
    MIN_N_KAPPA,
    MONOTONE_MAPS,
    best_level,
    delta_D,
    fit_monotone_map,
    fold_of,
    git_hash,
    level_kappas,
    pack,
)
from batch5_downstream import judge_labels, resolve_data_dir  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

# Explicit allow-list. dl22 is allowed ONLY through the gate.
ALLOWED: tuple[str, ...] = ("dl21", "dl22", "antique")
FAMILY_OF: dict[str, str] = {"dl21": "msmarco", "dl22": "msmarco",
                             "antique": "antique"}

JUDGES = [
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v4.1-flash",
    "google/gemini-3.5-flash",
    "google/gemini-3.8-flash",
    "anthropic/claude-opus-5",
    "openai/gpt-5.6-sol",
]
FLAGSHIPS = frozenset({"anthropic/claude-opus-5", "openai/gpt-5.6-sol"})

B6_MARKER = "B6 预注册 — 标注池扩充"

SEED = 42
FOLD_SEEDS = (42, 43, 44, 45, 46)
B_BOOT = 1000
B_SE = 1000
SE_SIZES = (50, 100, 200, 400)
SE_MIN_POOL_PAIRS = 50
SE_MIN_POOL_QUERIES = 2
SE_FAIL_WARN = 0.05
E6_N_LO, E6_N_HI, E6_FRAC = 200, 400, 0.8
THRESHOLDS = (2, 1)              # primary, secondary human relevance threshold
CI_LO, CI_HI = 5.0, 95.0         # 90% percentile interval
LABEL_SETS = ("RAW", "FROZEN", "REFIT")
DIFFS = (("FROZEN", "RAW"), ("REFIT", "RAW"), ("REFIT", "FROZEN"))
DELTA_HAT_TOL = 1e-9

# Adjudication tie bands — HARD-CODED (registered verbatim; not adjustable).
TIE_F1 = 0.005
TIE_RATIO = 0.02
REFIT_CI_NOTE = ("evaluation-only bootstrap; fit variability shown by "
                 "fold-seed spread")

ADJUDICATION_RULES = [
    "1. 方向一律按点估计判定。tie 带为代码硬编码常量：TIE_F1 = 0.005（作用于 "
    "E3/E4 的 ΔF1@2、E5 的 ΔP@2 或 ΔR@2、E6 的差额）；TIE_RATIO = 0.02（作用于 "
    "E1 的 |ratio−1| 与 E2 的 |ratio−1| 变化量）。",
    "2. tie 既不计命中也不计未中（与此前各轮相同）。",
    "3. \"≥0\" 型预测（E3/E4/E5/E6）：预测\"是\"时，Δ > +TIE 记命中，Δ < −TIE "
    "记未中，否则 tie；预测\"否\"时方向相反。",
    "4. E1：RAW ratio@≥2 点估计落在 [1−TIE_RATIO, 1+TIE_RATIO] 内一律记 tie，"
    "不看 CI；否则点估计方向与预测（>1 / <1）一致记命中，相反记未中。",
    "5. E2：变化量 = |r_FROZEN−1| − |r_RAW−1|。预测\"是\"（收缩）时 < −TIE_RATIO "
    "记命中，> +TIE_RATIO 记未中，否则 tie；预测\"否\"时方向相反。",
    "6. 90% CI 排除 0（E1 为排除 1）单独报告为\"显著\"，永不改变命中/未中判定。",
    "7. 合取单元 = (judge, 数据集)：单元内每个适用且已登记的终点均命中 → 单元"
    "合取命中；任一未中 → 单元未中；无未中但有已登记终点无法判定（如 E6 未运行）"
    " → 单元 not adjudicated；否则有 tie → 单元 tie。被跳过/报错的单元记 "
    "not adjudicated，不计分。",
    "8. E5：冻结规则 delta_hat < 0 的单元（严域）判定 R@2(REFIT) − R@2(RAW) ≥ 0；"
    "delta_hat ≥ 0 的单元（宽域）判定 P@2(REFIT) − P@2(RAW) ≥ 0。",
    "9. E6：Δ = mean dF1@2(REFIT−RAW)@n=200 − 0.8 × mean dF1@2(REFIT−RAW)@n=400"
    "（扩充池上的配对差），按第 3 条判定；仅当 n=200 与 n=400 均有成功抽样时"
    "判定，否则 not adjudicated。",
    "10. E3/E4/E5 使用主 fold seed（42）的 CORR-REFIT；其余 fold seeds 只作稳健性"
    f"报告，不计分。REFIT 的 CI：{REFIT_CI_NOTE}。",
]

_G = np.arange(4, dtype=float)
_W = np.abs(np.subtract.outer(_G, _G))

MAIN_EPS = []
for _t in THRESHOLDS:
    MAIN_EPS += [f"P@{_t}", f"R@{_t}", f"F1@{_t}", f"ratio@{_t}",
                 f"absdev@{_t}", f"rate@{_t}"]
MAIN_EPS += ["MAE", "offset", "mean_llm", "kappa"]
CAL_EPS = [f"cal_mh_g{g}" for g in range(4)] + [f"cal_n_g{g}" for g in range(4)]
HUMAN_EPS = [f"rate_h@{t}" for t in THRESHOLDS] + ["mean_human"]
ALL_EPS = MAIN_EPS + CAL_EPS + HUMAN_EPS
DIFF_EPS = MAIN_EPS        # paired differences: not for cal cells / human side
SEED_KEYS = ("dF1@2 REFIT-RAW", "dF1@2 REFIT-FROZEN", "dP@2 REFIT-RAW",
             "dR@2 REFIT-RAW")


# ═══════════════════════════════════════════════════════════════════════════
# git helpers
# ═══════════════════════════════════════════════════════════════════════════

def _git(*args) -> tuple[int, str]:
    try:
        p = subprocess.run(["git", *args], cwd=str(REPO), capture_output=True,
                           text=True)
        return p.returncode, (p.stdout.strip() or p.stderr.strip())
    except Exception as e:                                   # noqa: BLE001
        return 1, str(e)


def predictions_path() -> Path:
    return REPO / "paper" / "PREDICTIONS.md"


def predictions_clean(path: Path) -> tuple[bool, str]:
    """(clean, why). Untracked, modified or un-git-able all count as dirty."""
    rc, out = _git("status", "--porcelain", "--", str(path))
    if rc != 0:
        return False, f"git status failed: {out}"
    if out:
        return False, out
    return True, ""


def registration_commit(path: Path | None = None) -> str | None:
    rc, out = _git("log", "-1", "--format=%H,%cI", "--",
                   str(path or predictions_path()))
    return out if rc == 0 and out else None


def code_dirty() -> bool | None:
    rc, out = _git("status", "--porcelain", "--", "scripts", "src")
    return bool(out) if rc == 0 else None


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════
# Registration block: locate, parse, gate
# ═══════════════════════════════════════════════════════════════════════════

def b6_block(text: str) -> str | None:
    """The B6 block: from its '#' heading to the next top-level '# ' heading."""
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines)
                  if ln.lstrip().startswith("#") and B6_MARKER in ln), None)
    if start is None:
        return None
    end = next((j for j in range(start + 1, len(lines))
                if re.match(r"^#\s", lines[j])), len(lines))
    return "\n".join(lines[start:end])


def read_block(path: Path | None = None) -> str | None:
    p = path or predictions_path()
    return b6_block(p.read_text(encoding="utf-8")) if p.exists() else None


def has_registration(path: Path | None = None) -> bool:
    return read_block(path) is not None


_SEC = re.compile(r"^###\s+(\S+)\s+—\s+(dl21|dl22|antique)(?![A-Za-z0-9])")
_ELINE = re.compile(r"^-\s*B6-(E[0-6])(?![0-9])(.*)$")
_PRED = re.compile(r"预测[:：]\s*(\S*)")
_TOK = re.compile(r"^(>1|<1|是|否)")
_SHA = re.compile(r"rules_sha256:\s*([0-9a-f]{64})")


def parse_registration(block: str) -> dict:
    sha = _SHA.search(block)
    preds: dict = {}
    invalid = []
    cur = None
    for line in block.splitlines():
        m = _SEC.match(line)
        if m:
            cur = (m.group(1), m.group(2))
            preds.setdefault(cur, {})
            continue
        m = _ELINE.match(line)
        if not m or cur is None:
            continue
        E, rest = m.group(1), m.group(2)
        if E == "E0":
            preds[cur]["E0"] = "structural"
            continue
        mp = _PRED.search(rest)
        if not mp:
            continue                                  # N/A / ≡ lines
        mt = _TOK.match(mp.group(1))
        ok = {">1", "<1"} if E == "E1" else {"是", "否"}
        if mt and mt.group(1) in ok:
            preds[cur][E] = mt.group(1)
        else:
            invalid.append((cur, E, mp.group(1)))
    return {"rules_sha256": sha.group(1) if sha else None, "preds": preds,
            "invalid": invalid}


def require_registration(path: Path | None = None, strict: bool = False,
                         clean_fn=predictions_clean) -> str:
    """Marker must exist.  strict (dl22): dirty/untracked or '____' -> exit."""
    p = path or predictions_path()
    block = read_block(p)
    if block is None:
        sys.exit(
            "ERROR: paper/PREDICTIONS.md does not contain the B6 registration "
            f"block (a heading with '{B6_MARKER}').\nPrint it with `paste`, "
            "fill it, commit it FIRST — or pass --non-blind for a non-dl22 "
            "exploratory run.")
    problems = []
    ok, why = clean_fn(p)
    if not ok:
        problems.append(f"paper/PREDICTIONS.md is not clean/committed: {why}")
    if "____" in block:
        problems.append("the B6 block still contains unfilled '____'")
    if problems:
        if strict:
            sys.exit("ERROR (dl22 gate): " + "; ".join(problems)
                     + "\n  The registration counts only once it is complete "
                       "and committed.")
        for pr in problems:
            print(f"[gate] WARNING: {pr}", file=sys.stderr)
    return block


def gate(datasets, unseal: bool, non_blind: bool = False,
         path: Path | None = None, clean_fn=predictions_clean) -> str:
    """Returns the blind status. dl22: marker+clean+filled+flag, always."""
    if "dl22" in datasets:
        require_registration(path, strict=True, clean_fn=clean_fn)
        assert_not_sealed(datasets, unseal=unseal)
        return "NON-BLIND" if non_blind else "registered"
    if has_registration(path):
        require_registration(path, strict=False, clean_fn=clean_fn)
        return "NON-BLIND" if non_blind else "registered"
    if not non_blind:
        sys.exit("ERROR: `run` requires the committed B6 registration block "
                 f"('{B6_MARKER}') in paper/PREDICTIONS.md.\n  Use `paste` to "
                 "print it, or pass --non-blind for an explicitly NON-BLIND "
                 "exploratory run (stamped in header, JSON and tex).")
    print("[gate] NON-BLIND run: no B6 registration — every number printed is "
          "non-blind for any later registration.", file=sys.stderr)
    return "NON-BLIND"


def check_rules_sha(reg: dict, sha: str) -> None:
    if reg["rules_sha256"] is None:
        sys.exit("ERROR: the registered B6 block has no 'rules_sha256: <hex>' "
                 "line — the frozen rules cannot be verified.")
    if reg["rules_sha256"] != sha:
        sys.exit(f"ERROR: correction_rules.json sha256 {sha} does not match "
                 f"the registered rules_sha256 {reg['rules_sha256']}.\n  The "
                 "rules changed after registration; refusing to run.")


def resolve_datasets(spec: str) -> list[str]:
    names = parse_datasets(spec)
    bad = [n for n in names if n not in ALLOWED]
    if bad:
        sys.exit(f"ERROR: dataset(s) {bad} not allowed in batch 6. "
                 f"ALLOWED={list(ALLOWED)} (dl22 only via the gate).")
    return names


# ═══════════════════════════════════════════════════════════════════════════
# Loading
# ═══════════════════════════════════════════════════════════════════════════

REQUIRED_COLS = ("model_id", "prompt_id", "politeness_level", "dataset", "qid",
                 "docid", "score", "parse_ok")


def load_judgments(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"ERROR: {path} not found. Run src/parse.py first.")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".pkl":
        print(f"[TEST] judgments read from pickle {path} — stub data, NOT the "
              f"production parquet.")
        df = pd.read_pickle(path)
    else:
        sys.exit(f"ERROR: unsupported judgments file {path} (.parquet / .pkl).")
    miss = [c for c in REQUIRED_COLS if c not in df.columns]
    if miss:
        sys.exit(f"ERROR: judgments frame lacks columns {miss}.")
    return df


def load_human(datasets, qrels_json: str | None = None, quiet=False) -> dict:
    """(dataset, qid, docid) -> 0-3 grade.  ``qrels_json`` is a TEST hook."""
    if not qrels_json:
        return load_qrels(list(datasets), quiet=quiet)
    print(f"[TEST] qrels read from stub file {qrels_json} — NOT the real "
          f"qrels (grades taken as already on the 0-3 scale).")
    out = {}
    for ds, q, d, g in json.loads(Path(qrels_json).read_text()):
        if ds not in datasets:
            continue
        g = int(g)
        if not 0 <= g <= 3:
            sys.exit(f"ERROR: stub qrels grade {g} outside 0-3.")
        out[(str(ds), str(q), str(d))] = g
    return out


def load_flagship_pairs(data_dir: Path, ds: str) -> set:
    """Frozen 40% pairs for ONE dataset (2-tuple keys are safe: per-dataset)."""
    fp = data_dir / "inputs" / f"pairs_{ds}_flagship40.jsonl"
    if not fp.exists():
        sys.exit(f"ERROR: flagship pairs file {fp} missing; flagship judges "
                 f"must be restricted to their frozen 40% sample.")
    out = set()
    with open(fp) as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                out.add((str(p["qid"]), str(p["docid"])))
    return out


def load_rules(data_dir: Path, rules: str | None) -> tuple[dict, Path, str]:
    path = Path(rules) if rules else data_dir / "derived" / "correction_rules.json"
    if not path.exists():
        sys.exit(f"ERROR: correction rules not found at {path}. "
                 f"Run scripts/batch4_correction.py first.")
    return json.loads(path.read_text()), path, sha256_file(path)


def frozen_rule(rules_all: dict, model: str, ds: str) -> dict | None:
    rule = (rules_all.get("rules", {}).get(model, {}) or {}).get(FAMILY_OF[ds])
    if rule is not None and "datasets" not in rule:
        sys.exit(f"ERROR: frozen {FAMILY_OF[ds]} rule for '{model}' has no "
                 f"'datasets' key — its in-sample status is unknowable.")
    return rule


def is_noop_rule(rule: dict) -> bool:
    return (int(rule["dial_level"]) == 3
            and tuple(int(x) for x in rule["monotone_map"]) == IDENTITY_MAP)


def domain_of(rule: dict, model: str, ds: str) -> str:
    """E5 domain, fixed at registration from the frozen rule's delta_hat."""
    if rule.get("delta_hat") is None:
        sys.exit(f"ERROR: frozen rule for '{model}' @ {ds} has no delta_hat; "
                 f"the E5 domain (strict/lenient) cannot be fixed blindly.")
    return "strict" if float(rule["delta_hat"]) < 0 else "lenient"


def in_sample_note(rule: dict, ds: str) -> str:
    fitted = list(rule["datasets"])
    if ds not in fitted:
        return "held-out (frozen rule never fitted on this dataset)"
    if fitted == [ds]:
        return f"FULLY IN-SAMPLE (frozen rule fitted on {ds} itself)"
    return (f"PARTIALLY IN-SAMPLE (frozen rule fitted on "
            f"{'+'.join(fitted)}, which includes {ds})")


def scope_rows(df: pd.DataFrame, model: str, ds: str, flagships,
               data_dir: Path | None, frozen_pairs: dict) -> tuple:
    d = df[(df.model_id == model) & (df.dataset.astype(str) == ds)]
    d = d.assign(score=pd.to_numeric(d.score, errors="coerce"))
    restricted = False
    if model in flagships and not d.empty:
        if ds not in frozen_pairs:
            frozen_pairs[ds] = load_flagship_pairs(data_dir, ds)
        fz = frozen_pairs[ds]
        keys = zip(d.qid.astype(str), d.docid.astype(str))
        mask = np.fromiter((k in fz for k in keys), dtype=bool, count=len(d))
        d = d[mask]
        restricted = True
        if d.empty:
            sys.exit(f"ERROR: flagship pairs filter left ZERO rows for "
                     f"'{model}' on {ds}.")
    return d, restricted


# ═══════════════════════════════════════════════════════════════════════════
# Cell preparation: universe, per-level score counts, batch-4 Packed
# ═══════════════════════════════════════════════════════════════════════════

def prepare_cell(model: str, ds: str, d: pd.DataFrame, qrels: dict,
                 seed: int) -> dict:
    """Universe + count tensor + Packed for one (judge, dataset)."""
    d = d.assign(score=pd.to_numeric(d.score, errors="coerce"))
    human_ds = {(q, doc): g for (dds, q, doc), g in qrels.items() if dds == ds}
    qid = d.qid.astype(str).to_numpy()
    doc = d.docid.astype(str).to_numpy()
    lvl = d.politeness_level.astype(int).to_numpy()
    sc = d.score.to_numpy(dtype=float)
    ok = d.parse_ok.to_numpy().astype(bool) & np.isfinite(sc)
    if ok.any() and not np.all(np.isin(sc[ok], [0.0, 1.0, 2.0, 3.0])):
        bad = sorted(set(sc[ok].tolist()) - {0.0, 1.0, 2.0, 3.0})[:5]
        sys.exit(f"ERROR: '{model}' @ {ds}: parse_ok rows carry scores outside "
                 f"0..3: {bad}. Scores are never clipped — fix the parquet.")
    if ok.any() and not np.all(np.isin(lvl[ok], LEVELS)):
        bad = sorted(set(lvl[ok].tolist()) - set(LEVELS))[:5]
        sys.exit(f"ERROR: '{model}' @ {ds}: politeness_level outside 1..5 on "
                 f"usable rows: {bad}.")
    prompts_per_level = {
        f"L{l}": int(d.prompt_id[lvl == l].nunique()) for l in LEVELS}

    keys = list(zip(qid, doc))
    collected = set(keys)
    usable = {l: set(k for k, o, lv in zip(keys, ok, lvl) if o and lv == l)
              for l in LEVELS}
    judged = collected & set(human_ds)
    universe = sorted(set(human_ds).intersection(*usable.values()))
    info = {
        "n_rows": int(len(d)),
        "prompts_per_level": prompts_per_level,
        "n_collected_pairs": len(collected),
        "n_collected_human_judged": len(judged),
        "n_llm_only": len(collected - set(human_ds)),
        "n_usable_by_level": {f"L{l}": len(usable[l] & set(human_ds))
                              for l in LEVELS},
        "n_dropped_unusable": len(judged) - len(universe),
        "n_pairs": len(universe),
    }
    if not universe:
        return {"error": "empty pair universe", "info": info}

    uidx = {k: i for i, k in enumerate(universe)}
    row_u = np.fromiter((uidx.get(k, -1) for k in keys), dtype=np.int64,
                        count=len(keys))
    sel = ok & (row_u >= 0)
    cnt = np.zeros((len(LEVELS), len(universe), 4), dtype=np.int64)
    np.add.at(cnt, (lvl[sel] - 1, row_u[sel], sc[sel].astype(np.int64)), 1)
    if not (cnt.sum(axis=2) > 0).all():
        sys.exit(f"ERROR: '{model}' @ {ds}: universe pair without a usable "
                 f"label at some level — usability predicate drifted.")

    h = np.array([human_ds[k] for k in universe], dtype=np.int64)
    pair_qid = [k[0] for k in universe]
    qnames = sorted(set(pair_qid))
    qcode = {q: i for i, q in enumerate(qnames)}
    pair_q = np.array([qcode[q] for q in pair_qid], dtype=np.int64)

    # batch-4 Packed on the universe rows (parse failures kept, as batch 4
    # does for the human mean in Delta).
    du = d[row_u >= 0]
    P = pack(du, qrels, model, ds, seed)
    if len(P.h) != len(du):
        sys.exit(f"ERROR: '{model}' @ {ds}: batch-4 pack dropped rows that "
                 f"are in the universe.")
    ru = row_u[row_u >= 0]
    code2u = np.full(P.n_pairs, -1, dtype=np.int64)
    code2u[P.pair] = ru
    if not np.array_equal(code2u[P.pair], ru) or (code2u < 0).any():
        sys.exit(f"ERROR: '{model}' @ {ds}: pair-code mapping inconsistent.")
    u2code = np.empty(len(universe), dtype=np.int64)
    u2code[code2u] = np.arange(P.n_pairs)

    info["n_queries"] = len(qnames)
    pf = np.array([fold_of(ds, q, seed) for q in pair_qid])
    info["fold_pairs"] = [int((pf == k).sum()) for k in (0, 1)]
    info["fold_queries"] = [len(set(pair_q[pf == k].tolist())) for k in (0, 1)]
    return {"model": model, "dataset": ds, "universe": universe, "h": h,
            "cnt": cnt, "pair_q": pair_q, "pair_qid": pair_qid,
            "n_queries": len(qnames), "P": P, "pack_seed": seed,
            "u2code": u2code, "row_u_P": ru, "d": du, "d_full": d,
            "info": info}


def labels_from_counts(cnt_l: np.ndarray, mono) -> np.ndarray:
    """Per-record map, mean over records, round-half-up — vectorised.

    Same arithmetic as batch5_downstream.judge_labels: the mapped scores are
    small integers, so the sum is exact and the single division is correctly
    rounded in both paths; floor(x + 0.5) is round-half-up.
    """
    n = cnt_l.sum(axis=1)
    mean = (cnt_l @ np.asarray(mono, dtype=float)) / n
    return np.clip(np.floor(mean + 0.5), 0, 3).astype(np.int64)


def fit_rule(P, rows: np.ndarray):
    """Batch-4 dial + monotone-map estimators on rows ``rows``."""
    est = delta_D(P, rows, strict=False)
    if est is None:
        return None
    delta, D, _m, _q = est
    dial = best_level(delta, D)
    mono, kfit = fit_monotone_map(P, rows, dial)
    return {"dial_level": int(dial), "monotone_map": [int(x) for x in mono],
            "delta_fit": float(delta),
            "kappa_fit": float(kfit) if np.isfinite(kfit) else None}


def crossfit(cell: dict, fold_seed: int):
    """Query-disjoint 2-fold cross-fit with fold seed ``fold_seed``."""
    P, cnt, ds = cell["P"], cell["cnt"], cell["dataset"]
    pair_fold = np.array([fold_of(ds, q, fold_seed) for q in cell["pair_qid"]],
                         dtype=np.int64)
    row_fold = pair_fold[cell["row_u_P"]]
    if fold_seed == cell["pack_seed"] and not np.array_equal(row_fold, P.fold):
        sys.exit(f"ERROR: '{cell['model']}' @ {ds}: fold assignment differs "
                 f"from batch-4 pack() for the same seed.")
    if (pair_fold == 0).sum() == 0 or (pair_fold == 1).sum() == 0:
        return None, None, None, "query-hash split left one fold empty"
    lab = np.full(len(pair_fold), -1, dtype=np.int64)
    rules = []
    num = den = 0.0
    for k in (0, 1):
        r = fit_rule(P, np.where(row_fold != k)[0])
        if r is None:
            return None, None, None, f"train fold for test fold {k} lacks a level"
        test = pair_fold == k
        c = cnt[r["dial_level"] - 1][test]
        lab[test] = labels_from_counts(c, r["monotone_map"])
        num += float((c @ np.asarray(r["monotone_map"], float)).sum())
        den += float(c.sum())
        r.update({"test_fold": k, "n_test_pairs": int(test.sum()),
                  "n_train_pairs": int((~test).sum())})
        rules.append(r)
    return lab, rules, pair_fold, num / den


# ═══════════════════════════════════════════════════════════════════════════
# Endpoints (vectorised over leading axes of a [human, llm] confusion tensor)
# ═══════════════════════════════════════════════════════════════════════════

def endpoints(cm) -> dict:
    cm = np.asarray(cm, dtype=float)
    n = cm.sum(axis=(-2, -1))
    rowm = cm.sum(axis=-1)                    # human marginal
    colm = cm.sum(axis=-2)                    # llm marginal
    out = {"n": n}
    with np.errstate(divide="ignore", invalid="ignore"):
        for t in THRESHOLDS:
            hr = rowm[..., t:].sum(axis=-1)
            lr = colm[..., t:].sum(axis=-1)
            tp = cm[..., t:, t:].sum(axis=(-2, -1))
            ratio = lr / hr
            out[f"P@{t}"] = tp / lr
            out[f"R@{t}"] = tp / hr
            out[f"F1@{t}"] = 2.0 * tp / (lr + hr)
            out[f"ratio@{t}"] = ratio
            out[f"absdev@{t}"] = np.abs(ratio - 1.0)
            out[f"rate@{t}"] = lr / n
            out[f"rate_h@{t}"] = hr / n
        mh = (rowm * _G).sum(axis=-1) / n
        ml = (colm * _G).sum(axis=-1) / n
        out["mean_human"] = mh
        out["mean_llm"] = ml
        out["offset"] = ml - mh
        out["MAE"] = (cm * _W).sum(axis=(-2, -1)) / n
        nn = n[..., None, None]
        o = cm / nn
        e = (rowm[..., :, None] * colm[..., None, :]) / (nn * nn)
        po = (o * _W).sum(axis=(-2, -1))
        pe = (e * _W).sum(axis=(-2, -1))
        out["kappa"] = np.where(pe == 0, np.nan, 1.0 - po / pe)
        for g in range(4):
            cg = colm[..., g]
            out[f"cal_n_g{g}"] = cg
            out[f"cal_mh_g{g}"] = (cm[..., :, g] * _G).sum(axis=-1) / cg
    return out


def cm_by_query(h, lab, pair_q, nq) -> np.ndarray:
    keys = pair_q * 16 + h * 4 + lab
    return np.bincount(keys, minlength=nq * 16).reshape(nq, 4, 4)


def crosscheck(h: np.ndarray, lab: np.ndarray, pt: dict, tag: str) -> float:
    """Fast-path point estimates vs independent sklearn / vector paths."""
    from sklearn.metrics import cohen_kappa_score, precision_score, recall_score
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k_sk = float(cohen_kappa_score(h, lab, weights="linear"))
    k_cm = kappa_from_cm(np.bincount(h * 4 + lab, minlength=16)
                         .reshape(4, 4).astype(float))
    worst = 0.0
    for name, a, b in (("kappa/sklearn", pt["kappa"], k_sk),
                       ("kappa/batch1r", pt["kappa"], k_cm)):
        fa, fb = np.isfinite(a), np.isfinite(b)
        if fa != fb:
            sys.exit(f"ERROR: {tag}: {name} defined on one path only "
                     f"({a} vs {b}).")
        if fa:
            worst = max(worst, abs(float(a) - float(b)))
    checks = []
    for t in THRESHOLDS:
        yt, yp = (h >= t).astype(int), (lab >= t).astype(int)
        if yp.sum():
            checks.append((f"P@{t}", pt[f"P@{t}"], precision_score(yt, yp)))
        if yt.sum():
            checks.append((f"R@{t}", pt[f"R@{t}"], recall_score(yt, yp)))
        if yt.sum() + yp.sum():
            checks.append((f"F1@{t}", pt[f"F1@{t}"],
                           2 * (yt & yp).sum() / (yt.sum() + yp.sum())))
    for g in range(4):
        m = lab == g
        checks.append((f"cal_n_g{g}", pt[f"cal_n_g{g}"], float(m.sum())))
        if m.any():
            checks.append((f"cal_mh_g{g}", pt[f"cal_mh_g{g}"], h[m].mean()))
    checks += [("MAE", pt["MAE"], np.abs(h - lab).mean()),
               ("offset", pt["offset"], lab.mean() - h.mean())]
    for name, a, b in checks:
        d = abs(float(a) - float(b))
        if not np.isfinite(d) or d > 1e-9:
            sys.exit(f"ERROR: {tag}: fast endpoint {name} = {a} disagrees with "
                     f"the sklearn/vector path ({b}).")
        worst = max(worst, d)
    if worst > 1e-9:
        sys.exit(f"ERROR: {tag}: fast kappa disagrees with sklearn "
                 f"(max |diff| {worst:.2e}).")
    return worst


def boot_indices(nq: int, B: int, seed: int) -> np.ndarray:
    """(B, nq) resampled query indices — the single source of resampling."""
    rng = np.random.default_rng(seed)
    return np.stack([rng.integers(0, nq, size=nq) for _ in range(B)])


def bootstrap(cms: dict, B: int, seed: int, return_reps: bool = False) -> dict:
    """Query-clustered bootstrap; one resampled query set shared by all sets."""
    nq = next(iter(cms.values())).shape[0]
    idx = boot_indices(nq, B, seed)
    Wt = np.stack([np.bincount(r, minlength=nq) for r in idx]).astype(float)
    reps = {ls: endpoints(np.einsum("bq,qij->bij", Wt, cm.astype(float)))
            for ls, cm in cms.items()}

    def ci(x):
        x = np.asarray(x, dtype=float)
        f = x[np.isfinite(x)]
        if f.size == 0:
            return [None, None, 0]
        return [float(np.percentile(f, CI_LO)), float(np.percentile(f, CI_HI)),
                int(f.size)]

    out = {"B": B, "seed": seed,
           "ci90": {ls: {ep: ci(reps[ls][ep]) for ep in ALL_EPS}
                    for ls in cms},
           "diff": {}}
    for a, b in DIFFS:
        if a not in reps or b not in reps:
            continue
        dd = {}
        for ep in DIFF_EPS:
            x = reps[a][ep] - reps[b][ep]
            f = x[np.isfinite(x)]
            lo, hi, nok = ci(x)
            dd[ep] = {"ci90": [lo, hi], "n_ok": nok,
                      "p_gt0": float((f > 0).mean()) if f.size else None,
                      "p_ge0": float((f >= 0).mean()) if f.size else None}
        out["diff"][f"{a}-{b}"] = dd
    if return_reps:
        out["reps"] = reps
    return out


def fnum(x):
    """JSON-safe float (NaN/inf -> None)."""
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def check_delta_hat(cell: dict, rule: dict, qrels: dict) -> dict:
    """S1: full-data continuous Delta(L3) vs the rule's delta_hat.

    Enforced (|diff| <= 1e-9, else exit) only where the rule was fitted on
    exactly this dataset — the only case where the estimand is identical
    (same rows, same batch-4 delta_D).  Elsewhere it is printed only.
    """
    ds, model = cell["dataset"], cell["model"]
    applicable = list(rule["datasets"]) == [ds]
    Pf = pack(cell["d_full"], qrels, model, ds, cell["pack_seed"])
    est = delta_D(Pf, np.arange(len(Pf.h)), strict=False)
    full = float(est[0]) if est is not None else None
    hat = rule.get("delta_hat")
    out = {"applicable": applicable, "delta_full_L3": full,
           "delta_hat": fnum(hat)}
    if applicable and hat is not None and full is not None:
        diff = abs(full - float(hat))
        out["abs_diff"] = diff
        if diff > DELTA_HAT_TOL:
            sys.exit(f"ERROR: '{model}' @ {ds}: continuous Delta(L3) on the "
                     f"full matched data = {full!r} but the frozen rule (fitted "
                     f"on {ds} alone) records delta_hat = {hat!r} "
                     f"(|diff| {diff:.2e} > {DELTA_HAT_TOL}).\n  The data the "
                     f"rule was fitted on has changed, or the estimand drifted.")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Analysis of one (judge, dataset)
# ═══════════════════════════════════════════════════════════════════════════

def analyze_cell(cell: dict, rule: dict, B: int, seed: int,
                 se_sizes=None, se_B: int = B_SE,
                 fold_seeds=FOLD_SEEDS, qrels: dict | None = None) -> dict:
    model, ds = cell["model"], cell["dataset"]
    h, cnt, P = cell["h"], cell["cnt"], cell["P"]
    nP, nq = len(h), cell["n_queries"]
    dial_f = int(rule["dial_level"])
    mono_f = tuple(int(x) for x in rule["monotone_map"])

    labels = {"RAW": labels_from_counts(cnt[2], IDENTITY_MAP),
              "FROZEN": labels_from_counts(cnt[dial_f - 1], mono_f)}
    # Cross-check the vectorised labels against batch 5's judge_labels.
    for ls, lev, mono in (("RAW", 3, None), ("FROZEN", dial_f, mono_f)):
        ref = judge_labels(cell["d"], lev, mono=mono)
        want = np.array([ref[k] for k in cell["universe"]], dtype=np.int64)
        if not np.array_equal(want, labels[ls]):
            sys.exit(f"ERROR: '{model}' @ {ds}: vectorised {ls} labels differ "
                     f"from batch5.judge_labels on "
                     f"{int((want != labels[ls]).sum())} pairs.")

    # CORR-REFIT, primary fold seed + robustness seeds.
    fold_seeds = list(fold_seeds)
    primary = fold_seeds[0]
    lab, refit_rules, pair_fold, rec_mean_refit = crossfit(cell, primary)
    if lab is None:
        return {"model": model, "dataset": ds, "info": cell["info"],
                "error": f"fold seed {primary}: {rec_mean_refit}"}
    labels["REFIT"] = lab

    cms = {ls: cm_by_query(h, labels[ls], cell["pair_q"], nq)
           for ls in LABEL_SETS}
    points = {ls: {ep: float(v) for ep, v in endpoints(cms[ls].sum(0)).items()}
              for ls in LABEL_SETS}
    worst = max(crosscheck(h, labels[ls], points[ls], f"{model}@{ds}/{ls}")
                for ls in LABEL_SETS)
    boot = bootstrap(cms, B, seed)
    pdiff = {f"{a}-{b}": {ep: points[a][ep] - points[b][ep] for ep in DIFF_EPS}
             for a, b in DIFFS}

    # S1: continuous Delta (record mean - human mean, batch-4 convention:
    # human mean over all universe rows incl. parse failures).
    qmean = float(P.h.mean())
    rec = {
        "RAW": float((cnt[2] @ _G).sum() / cnt[2].sum()),
        "FROZEN": float((cnt[dial_f - 1] @ np.asarray(mono_f, float)).sum()
                        / cnt[dial_f - 1].sum()),
        "REFIT": float(rec_mean_refit),
    }
    delta_cont = {ls: rec[ls] - qmean for ls in LABEL_SETS}
    est = delta_D(P, np.arange(len(P.h)), strict=True)
    if abs(est[0] - delta_cont["RAW"]) > 1e-9:
        sys.exit(f"ERROR: '{model}' @ {ds}: continuous Delta(RAW) "
                 f"{delta_cont['RAW']!r} != batch-4 delta_D {est[0]!r}.")
    kappa_auth = level_kappas(P, np.arange(len(P.h)), levels=[3])[3]

    # fold-seed robustness
    seed_rows, seed_labels = {}, {}
    for s in fold_seeds:
        if s == primary:
            lab_s, rules_s = lab, refit_rules
        else:
            lab_s, rules_s, _pf, why = crossfit(cell, s)
            if lab_s is None:
                seed_rows[str(s)] = {"error": why}
                continue
        seed_labels[s] = lab_s
        ps = endpoints(np.bincount(h * 4 + lab_s, minlength=16).reshape(4, 4))
        seed_rows[str(s)] = {
            "rules": [{"test_fold": r["test_fold"], "dial_level": r["dial_level"],
                       "monotone_map": r["monotone_map"]} for r in rules_s],
            "points": {ep: fnum(ps[ep]) for ep in MAIN_EPS},
            "dF1@2 REFIT-RAW": float(ps["F1@2"] - points["RAW"]["F1@2"]),
            "dF1@2 REFIT-FROZEN": float(ps["F1@2"] - points["FROZEN"]["F1@2"]),
            "dP@2 REFIT-RAW": float(ps["P@2"] - points["RAW"]["P@2"]),
            "dR@2 REFIT-RAW": float(ps["R@2"] - points["RAW"]["R@2"]),
        }
    ok_rows = [v for v in seed_rows.values() if "error" not in v]

    def msd(vals):
        v = np.asarray([x for x in vals if x is not None and np.isfinite(x)],
                       dtype=float)
        if not v.size:
            return None
        return {"mean": float(v.mean()),
                "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                "min": float(v.min()), "max": float(v.max()), "n": int(v.size)}

    seed_summary = {ep: msd([r["points"][ep] for r in ok_rows])
                    for ep in MAIN_EPS}
    seed_summary.update({k: msd([r[k] for r in ok_rows]) for k in SEED_KEYS})

    res = {
        "model": model, "dataset": ds, "info": cell["info"],
        "rule_frozen": {"dial_level": dial_f, "monotone_map": list(mono_f),
                        "fitted_on": list(rule["datasets"]),
                        "delta_hat": fnum(rule.get("delta_hat")),
                        "in_sample": in_sample_note(rule, ds)},
        "e5_domain": domain_of(rule, model, ds),
        "noop_frozen": is_noop_rule(rule),
        "refit_rules": refit_rules,
        "fold_seeds": fold_seeds,
        "refit_seeds": seed_rows, "refit_seed_summary": seed_summary,
        "points": points, "point_diff": pdiff, "boot": boot,
        "delta_cont": delta_cont, "kappa_auth_L3_universe": fnum(kappa_auth),
        "labels": labels, "seed_labels": seed_labels, "pair_fold": pair_fold,
        "universe": cell["universe"], "h": h,
        "kappa_crosscheck_max_absdiff": worst,
        "n_pairs": nP, "n_queries": nq,
    }
    if qrels is not None:
        res["delta_hat_check"] = check_delta_hat(cell, rule, qrels)
    if se_sizes:
        res["sample_eff"] = sample_efficiency(cell, labels, se_sizes, se_B,
                                              seed)
    return res


def sample_efficiency(cell: dict, labels: dict, sizes, B: int,
                      seed: int) -> dict:
    h, cnt, P, u2code = cell["h"], cell["cnt"], cell["P"], cell["u2code"]
    pair_q, nq = cell["pair_q"], cell["n_queries"]
    pair_qid = np.asarray(cell["pair_qid"])
    code2u = np.empty(P.n_pairs, dtype=np.int64)
    code2u[u2code] = np.arange(len(u2code))
    q_pairs = [np.where(pair_q == qi)[0] for qi in range(nq)]
    rng = np.random.default_rng(seed)
    eps = ("P@2", "R@2", "F1@2", "ratio@2")
    out = {}
    for n in sizes:
        if n > len(h) - SE_MIN_POOL_PAIRS:
            out[str(n)] = {"skipped": f"n={n} leaves < {SE_MIN_POOL_PAIRS} "
                                      f"pool pairs (universe {len(h)})"}
            continue
        acc = {ls: {ep: [] for ep in eps} for ls in LABEL_SETS}
        d_f1, pool_sizes, dials, n_fail = [], [], {}, 0
        for _ in range(B):
            chosen, touched = [], []
            for qi in rng.permutation(nq):
                need = n - len(chosen)
                ps = q_pairs[qi]
                touched.append(int(qi))
                if len(ps) <= need:
                    chosen.extend(ps.tolist())
                else:
                    chosen.extend(rng.permutation(ps)[:need].tolist())
                if len(chosen) == n:
                    break
            chosen = np.asarray(chosen, dtype=np.int64)
            tset = np.zeros(nq, dtype=bool)
            tset[touched] = True
            pool = ~tset[pair_q]
            if (len(chosen) != n or len(np.unique(chosen)) != n
                    or pool.sum() < SE_MIN_POOL_PAIRS
                    or len(set(pair_q[pool].tolist())) < SE_MIN_POOL_QUERIES):
                n_fail += 1
                continue
            rows = np.concatenate([P.pair_rows[c] for c in u2code[chosen]])
            # Real disjointness check on the rows actually fed to the fit,
            # via the qid STRINGS (independent of the integer query codes).
            fit_q = set(pair_qid[code2u[P.pair[rows]]].tolist())
            pool_q = set(pair_qid[pool].tolist())
            if fit_q & pool_q or len(rows) == 0:
                sys.exit(f"ERROR: '{cell['model']}' @ {cell['dataset']}: "
                         f"calibration rows share queries with the pool "
                         f"({sorted(fit_q & pool_q)[:3]}).")
            r = fit_rule(P, rows)
            if r is None:
                n_fail += 1
                continue
            dials[r["dial_level"]] = dials.get(r["dial_level"], 0) + 1
            lab_pool = {
                "RAW": labels["RAW"][pool],
                "FROZEN": labels["FROZEN"][pool],
                "REFIT": labels_from_counts(cnt[r["dial_level"] - 1][pool],
                                            r["monotone_map"]),
            }
            hp = h[pool]
            f1s = {}
            for ls in LABEL_SETS:
                cm = np.bincount(hp * 4 + lab_pool[ls], minlength=16)\
                    .reshape(4, 4)
                ep = endpoints(cm)
                for k in eps:
                    acc[ls][k].append(float(ep[k]))
                f1s[ls] = float(ep["F1@2"])
            d_f1.append(f1s["REFIT"] - f1s["RAW"])
            pool_sizes.append(int(pool.sum()))

        def ms(x):
            x = np.asarray(x, dtype=float)
            x = x[np.isfinite(x)]
            return ([float(x.mean()), float(x.std(ddof=1)) if x.size > 1
                     else 0.0] if x.size else [None, None])

        if not pool_sizes:
            out[str(n)] = {"skipped": "no successful draws",
                           "n_draws_failed": n_fail}
            continue
        dd = np.asarray(d_f1, dtype=float)
        out[str(n)] = {
            "n_draws_ok": len(pool_sizes), "n_draws_failed": n_fail,
            "fail_frac": n_fail / B,
            "pool_pairs_mean": float(np.mean(pool_sizes)),
            "dial_freq": {f"L{k}": v for k, v in sorted(dials.items())},
            **{f"{ls}_{ep}": ms(acc[ls][ep]) for ls in LABEL_SETS
               for ep in eps},
            "REFIT-RAW_F1@2": ms(dd),
            "frac_REFIT_F1_ge_RAW": float((dd >= 0).mean()),
        }
    return out


# ═══════════════════════════════════════════════════════════════════════════
# No-op invariant
# ═══════════════════════════════════════════════════════════════════════════

def _same(a, b) -> bool:
    if a is None or b is None:
        return a is b
    a, b = float(a), float(b)
    return (np.isnan(a) and np.isnan(b)) or a == b


def noop_violations(res: dict) -> list[str]:
    """CORR-FROZEN must equal RAW bit-exactly when the rule is identity@L3."""
    if "error" in res or not res["noop_frozen"]:
        return []
    bad = []
    if not np.array_equal(res["labels"]["FROZEN"], res["labels"]["RAW"]):
        bad.append("labels")
    for ep in ALL_EPS:
        if not _same(res["points"]["FROZEN"][ep], res["points"]["RAW"][ep]):
            bad.append(f"point:{ep}")
        cf, cr = res["boot"]["ci90"]["FROZEN"][ep], res["boot"]["ci90"]["RAW"][ep]
        if not all(_same(x, y) for x, y in zip(cf, cr)):
            bad.append(f"ci90:{ep}")
    for ep in DIFF_EPS:
        v = res["point_diff"]["FROZEN-RAW"][ep]
        if np.isfinite(v) and v != 0.0:
            bad.append(f"diff:{ep}")
        lo, hi = res["boot"]["diff"]["FROZEN-RAW"][ep]["ci90"]
        if (lo is not None and lo != 0.0) or (hi is not None and hi != 0.0):
            bad.append(f"diffci:{ep}")
    if not _same(res["delta_cont"]["FROZEN"], res["delta_cont"]["RAW"]):
        bad.append("delta_cont")
    return bad


def enforce_noop(results: list[dict]) -> None:
    viol = {(r["model"], r["dataset"]): noop_violations(r) for r in results}
    viol = {k: v for k, v in viol.items() if v}
    n_noop = sum(1 for r in results if "error" not in r and r["noop_frozen"])
    print(f"\n  no-op control: {n_noop} (judge, dataset) cell(s) with frozen "
          f"rule identity@L3 — CORR-FROZEN == RAW bit-exactly: "
          f"{'OK' if not viol else 'VIOLATED'}")
    if viol:
        sys.exit("\nERROR: no-op invariant VIOLATED — identity@L3 changed "
                 "labels/endpoints for\n"
                 + "".join(f"    {m} @ {ds}: {v[:8]}{' ...' if len(v) > 8 else ''}\n"
                           for (m, ds), v in viol.items())
                 + "  This is a BUG in label construction or endpoints, not a "
                   "result. Nothing was written.")


# ═══════════════════════════════════════════════════════════════════════════
# Adjudication
# ═══════════════════════════════════════════════════════════════════════════

def _signed(delta, tie: float, predict_positive: bool) -> str:
    if delta is None or not np.isfinite(delta):
        return "not adjudicated"
    if abs(delta) <= tie:
        return "tie"
    return "hit" if (delta > 0) == predict_positive else "miss"


def score_endpoint(E: str, value, pred: str | None) -> str:
    if pred is None:
        return "unregistered"
    if E == "E0":
        return "hit"                       # invariant passed or we exited
    if value is None or not np.isfinite(value):
        return "not adjudicated"
    if E == "E1":
        if 1.0 - TIE_RATIO <= value <= 1.0 + TIE_RATIO:   # rule 4, closed
            return "tie"
        return "hit" if (value > 1.0) == (pred == ">1") else "miss"
    if E == "E2":                          # 是 = shrink = negative change
        return _signed(value, TIE_RATIO, predict_positive=(pred == "否"))
    return _signed(value, TIE_F1, predict_positive=(pred == "是"))


def conjunction(verdicts: dict) -> str:
    reg = [v for v in verdicts.values() if v != "unregistered"]
    if not reg:
        return "not adjudicated (no registered endpoint)"
    if "miss" in reg:
        return "miss"
    if "not adjudicated" in reg:
        return "not adjudicated (incomplete)"
    if "tie" in reg:
        return "tie"
    return "hit"


def endpoint_values(res: dict) -> dict:
    """E -> {value, ci, sig, desc} for one analysed cell."""
    pt, b, pdf = res["points"], res["boot"], res["point_diff"]
    out = {}

    def sig0(ci):
        return None if ci[0] is None else bool(ci[0] > 0 or ci[1] < 0)

    if res["noop_frozen"]:
        out["E0"] = {"value": None, "ci": None, "sig": None,
                     "desc": "CORR-FROZEN ≡ RAW bit-exactly (invariant passed)"}
    rc = b["ci90"]["RAW"]["ratio@2"]
    out["E1"] = {"value": pt["RAW"]["ratio@2"], "ci": rc[:2],
                 "sig": None if rc[0] is None else bool(rc[0] > 1 or rc[1] < 1),
                 "desc": "RAW ratio@>=2"}
    if not res["noop_frozen"]:
        c = b["diff"]["FROZEN-RAW"]["absdev@2"]["ci90"]
        out["E2"] = {"value": pdf["FROZEN-RAW"]["absdev@2"], "ci": c,
                     "sig": sig0(c), "desc": "|r_FROZEN-1| - |r_RAW-1|"}
    c = b["diff"]["REFIT-RAW"]["F1@2"]["ci90"]
    out["E3"] = {"value": pdf["REFIT-RAW"]["F1@2"], "ci": c, "sig": sig0(c),
                 "desc": "F1@2 REFIT-RAW"}
    if not res["noop_frozen"]:
        c = b["diff"]["REFIT-FROZEN"]["F1@2"]["ci90"]
        out["E4"] = {"value": pdf["REFIT-FROZEN"]["F1@2"], "ci": c,
                     "sig": sig0(c), "desc": "F1@2 REFIT-FROZEN"}
    ep5 = "R@2" if res["e5_domain"] == "strict" else "P@2"
    c = b["diff"]["REFIT-RAW"][ep5]["ci90"]
    out["E5"] = {"value": pdf["REFIT-RAW"][ep5], "ci": c, "sig": sig0(c),
                 "desc": f"{ep5} REFIT-RAW ({res['e5_domain']} domain)"}
    se = res.get("sample_eff") or {}
    lo, hi = se.get(str(E6_N_LO), {}), se.get(str(E6_N_HI), {})
    v6 = None
    if ("skipped" not in lo and "skipped" not in hi and lo and hi
            and lo["REFIT-RAW_F1@2"][0] is not None
            and hi["REFIT-RAW_F1@2"][0] is not None):
        v6 = lo["REFIT-RAW_F1@2"][0] - E6_FRAC * hi["REFIT-RAW_F1@2"][0]
    out["E6"] = {"value": v6, "ci": None, "sig": None,
                 "desc": f"dF1@2(REFIT-RAW)@{E6_N_LO} - {E6_FRAC} x "
                         f"@{E6_N_HI}" + ("" if v6 is not None else
                                          " [sizes not both run]")}
    return out


def adjudicate(res: dict, preds: dict | None) -> dict:
    if "error" in res:
        return {"conjunction": "not adjudicated", "reason": res["error"]}
    cell_preds = (preds or {}).get((res["model"], res["dataset"]), {})
    vals = endpoint_values(res)
    eps = {}
    for E, v in vals.items():
        p = cell_preds.get(E)
        eps[E] = {**v, "pred": p, "verdict": score_endpoint(E, v["value"], p)}
    # registered endpoints that this cell cannot carry (e.g. E2 for a no-op)
    for E, p in cell_preds.items():
        if E not in eps:
            eps[E] = {"value": None, "ci": None, "sig": None, "pred": p,
                      "desc": "not applicable to this cell",
                      "verdict": "not adjudicated"}
    return {"endpoints": eps,
            "conjunction": conjunction({E: e["verdict"] for E, e in eps.items()})}


# ═══════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════

def _f(x, w=7, p=4, sign=False):
    if x is None or not np.isfinite(x):
        return f"{'nan':>{w}}"
    return f"{x:>+{w}.{p}f}" if sign else f"{x:>{w}.{p}f}"


def _ci(c, p=3):
    if c is None or c[0] is None:
        return "[  n/a  ]"
    return f"[{c[0]:+.{p}f},{c[1]:+.{p}f}]"


def print_cell(res: dict) -> None:
    print()
    print("-" * 96)
    print(f"  JUDGE: {res['model']}   dataset: {res['dataset']}")
    print("-" * 96)
    info = res["info"]
    if "error" in res:
        print(f"  SKIPPED: {res['error']}   info={info}")
        return
    rf = res["rule_frozen"]
    print(f"  CORR-FROZEN rule : dial=L{rf['dial_level']} map={rf['monotone_map']}"
          f"  delta_hat={_f(rf['delta_hat'], 0, 4, True).strip()}"
          f"  [{rf['in_sample']}]"
          + ("  <- identity@L3: NO-OP control" if res["noop_frozen"] else ""))
    print(f"  E5 domain        : {res['e5_domain']} (sign of frozen delta_hat)")
    for r in res["refit_rules"]:
        print(f"  CORR-REFIT seed {res['fold_seeds'][0]} fold {r['test_fold']} "
              f"(test {r['n_test_pairs']} pairs; fit on {r['n_train_pairs']}): "
              f"dial=L{r['dial_level']} map={r['monotone_map']}  "
              f"Delta_fit={r['delta_fit']:+.4f}")
    print(f"  universe         : {info['n_pairs']} pairs / {info['n_queries']} "
          f"queries{'  [flagship 40% sample]' if info.get('flagship') else ''}")
    print(f"    collected {info['n_collected_pairs']} pairs; human-judged "
          f"{info['n_collected_human_judged']}; llm-only (no qrels) "
          f"{info['n_llm_only']}; dropped (no usable label at some level) "
          f"{info['n_dropped_unusable']}")
    print(f"    distinct prompt_id per level: {info['prompts_per_level']}")
    pt, ci = res["points"], res["boot"]["ci90"]
    hum = pt["RAW"]
    print(f"  HUMAN            : rel@>=2 rate {hum['rate_h@2']:.4f}, rel@>=1 rate "
          f"{hum['rate_h@1']:.4f}, mean grade {hum['mean_human']:.4f}")
    print()
    print(f"  {'endpoint':<10}" + "".join(f"{ls:>29}" for ls in LABEL_SETS)
          + "      (point [90% CI]; REFIT CI: " + REFIT_CI_NOTE + ")")
    for ep in MAIN_EPS:
        line = f"  {ep:<10}"
        for ls in LABEL_SETS:
            line += f"{_f(pt[ls][ep], 8, 4)} {_ci(ci[ls][ep]):>20}"
        print(line)
    dc = res["delta_cont"]
    print(f"  {'Delta_cont':<10}" + "".join(
        f"{_f(dc[ls], 8, 4, True)} {'(point only)':>20}" for ls in LABEL_SETS)
        + "   record mean - human mean")
    dh = res.get("delta_hat_check")
    if dh:
        tag = ("ENFORCED, |diff| %.1e" % dh["abs_diff"]
               if dh.get("abs_diff") is not None else
               "print only (rule not fitted on this dataset alone)")
        print(f"  Delta(L3) full matched data = {_f(dh['delta_full_L3'], 0, 6, True).strip()}"
              f"  vs rule delta_hat {_f(dh['delta_hat'], 0, 6, True).strip()}  [{tag}]")
    print(f"  authoritative paraphrase-mean kappa(L3) on the universe (bridge): "
          f"{_f(res['kappa_auth_L3_universe'], 0, 4).strip()}")
    print()
    print("  calibration: mean HUMAN grade among pairs the LLM labeled g (n)")
    for ls in LABEL_SETS:
        cells = "  ".join(
            f"g{g}: {_f(pt[ls][f'cal_mh_g{g}'], 5, 3)} (n={int(pt[ls][f'cal_n_g{g}'])})"
            for g in range(4))
        print(f"    {ls:<7} {cells}")
    print()
    print("  paired differences (same resampled queries): point [90% CI] P(>0)")
    print(f"  {'endpoint':<10}" + "".join(f"{k:>32}" for k in
                                         ("FROZEN-RAW", "REFIT-RAW",
                                          "REFIT-FROZEN")))
    for ep in DIFF_EPS:
        line = f"  {ep:<10}"
        for a, b in DIFFS:
            key = f"{a}-{b}"
            dd = res["boot"]["diff"][key][ep]
            p = dd["p_gt0"]
            line += (f"{_f(res['point_diff'][key][ep], 8, 4, True)} "
                     f"{_ci(dd['ci90']):>17} {('n/a' if p is None else f'{p:.2f}'):>5}")
        print(line)
    print()
    ss = res["refit_seed_summary"]
    print(f"  CORR-REFIT across fold seeds {res['fold_seeds']} "
          f"(fit variability; mean +- sd [min, max])")
    for k in ("F1@2", "ratio@2", "P@2", "R@2", "MAE", "kappa") + SEED_KEYS:
        v = ss.get(k)
        if v is None:
            print(f"    {k:<20} n/a")
            continue
        print(f"    {k:<20} {v['mean']:+.4f} +- {v['sd']:.4f}  "
              f"[{v['min']:+.4f}, {v['max']:+.4f}]  (n={v['n']})")
    for s, v in res["refit_seeds"].items():
        if "error" in v:
            print(f"    seed {s}: ERROR {v['error']}")
    se = res.get("sample_eff")
    if se:
        print()
        print("  sample efficiency (refit on n human pairs, evaluate on the "
              "query-disjoint extension pool; mean +- sd over draws)")
        print(f"  {'n':>5} {'ok':>5} {'fail':>5} {'pool':>7}  "
              f"{'F1@2 RAW':>15} {'F1@2 FROZEN':>15} {'F1@2 REFIT':>15}  "
              f"{'P@2 RAW':>15} {'P@2 REFIT':>15} {'R@2 RAW':>15} "
              f"{'R@2 REFIT':>15}  {'ratio RAW':>15} {'ratio REFIT':>15}  "
              f"{'dF1 REFIT-RAW':>15} {'P(>=0)':>6}  dial")

        def m(x):
            return ("      n/a      " if x is None or x[0] is None
                    else f"{x[0]:.4f}+-{x[1]:.4f}")
        for n, v in se.items():
            if "skipped" in v:
                print(f"  {n:>5}  SKIPPED: {v['skipped']}"
                      + (f" (failed draws {v['n_draws_failed']})"
                         if "n_draws_failed" in v else ""))
                continue
            print(f"  {n:>5} {v['n_draws_ok']:>5} {v['n_draws_failed']:>5} "
                  f"{v['pool_pairs_mean']:>7.1f}  "
                  f"{m(v['RAW_F1@2']):>15} {m(v['FROZEN_F1@2']):>15} "
                  f"{m(v['REFIT_F1@2']):>15}  {m(v['RAW_P@2']):>15} "
                  f"{m(v['REFIT_P@2']):>15} {m(v['RAW_R@2']):>15} "
                  f"{m(v['REFIT_R@2']):>15}  {m(v['RAW_ratio@2']):>15} "
                  f"{m(v['REFIT_ratio@2']):>15}  {m(v['REFIT-RAW_F1@2']):>15} "
                  f"{v['frac_REFIT_F1_ge_RAW']:>6.3f}  {v['dial_freq']}")
            if v["fail_frac"] > SE_FAIL_WARN:
                print(f"        WARNING: {v['fail_frac']:.1%} of draws failed "
                      f"(> {SE_FAIL_WARN:.0%}) — the n={n} summary is "
                      f"conditional on the successful draws.")


def print_rules_block(prefix: str = "  ") -> None:
    print(f"{prefix}判定纪律（逐字，与 PASTE 块相同）：")
    for line in ADJUDICATION_RULES:
        print(f"{prefix}{line}")
    print(f"{prefix}constants: TIE_F1={TIE_F1}  TIE_RATIO={TIE_RATIO}  "
          f"E6: n={E6_N_LO} vs {E6_FRAC} x n={E6_N_HI}")


def print_backfill(results: list[dict], ds: str, blind: str) -> dict:
    print()
    print("=" * 96)
    print(f"  BACKFILL — B6 adjudicable endpoints, {ds}  [{blind}]")
    print("=" * 96)
    print_rules_block()
    tally = {"cells": {}, "endpoints": {}}
    for r in results:
        adj = r.get("adjudication") or {}
        print()
        if "error" in r:
            print(f"  {r['model']} — {ds}: NOT ADJUDICATED ({r['error']})")
            tally["cells"]["not adjudicated"] = \
                tally["cells"].get("not adjudicated", 0) + 1
            continue
        print(f"  {r['model']} — {ds}")
        for E in sorted(adj["endpoints"]):
            e = adj["endpoints"][E]
            val = ("" if e["value"] is None else
                   f"= {e['value']:+.4f} " if E != "E1" else f"= {e['value']:.4f} ")
            sig = ("" if e["sig"] is None else
                   ("  significant (90% CI excludes " + ("1" if E == "E1" else "0")
                    + ")") if e["sig"] else "  not significant")
            note = f"  ({REFIT_CI_NOTE})" if E in ("E3", "E4", "E5") else ""
            print(f"    B6-{E} {e['desc']} {val}{_ci(e['ci'], 4) if e['ci'] else ''}"
                  f"  pred={e['pred'] or '—'} -> {e['verdict'].upper()}{sig}{note}")
            if e["verdict"] != "unregistered":
                tally["endpoints"][e["verdict"]] = \
                    tally["endpoints"].get(e["verdict"], 0) + 1
        ss = r["refit_seed_summary"]
        parts = []
        for k in ("dF1@2 REFIT-RAW", "dF1@2 REFIT-FROZEN",
                  "dR@2 REFIT-RAW" if r["e5_domain"] == "strict"
                  else "dP@2 REFIT-RAW"):
            v = ss.get(k)
            if v:
                parts.append(f"{k} {v['mean']:+.4f}+-{v['sd']:.4f} "
                             f"[{v['min']:+.4f},{v['max']:+.4f}]")
        print(f"    fold-seed robustness {r['fold_seeds']} (not scored): "
              + "; ".join(parts))
        print(f"    => cell conjunction: {adj['conjunction'].upper()}")
        tally["cells"][adj["conjunction"]] = \
            tally["cells"].get(adj["conjunction"], 0) + 1
    print()
    print(f"  {ds} tally — cells: {tally['cells']}   registered endpoints: "
          f"{tally['endpoints']}")
    return tally


def _t(x, sign=False) -> str:
    if x is None or not np.isfinite(x):
        return "--"
    return f"${x:+.3f}$" if sign else f"{x:.3f}"


def _tci(c) -> str:
    if c[0] is None:
        return ""
    return f"{{\\scriptsize[{c[0]:.3f}, {c[1]:.3f}]}}"


def _tms(x) -> str:
    return "--" if x is None or x[0] is None else \
        f"{x[0]:.3f} $\\pm$ {x[1]:.3f}"


def write_outputs(results: list[dict], ds: str, data_dir: Path, meta: dict,
                  se: bool) -> None:
    tex_dir = data_dir / "derived" / "tex"
    stamp = "[NON-BLIND] " if meta.get("blind_status") == "NON-BLIND" else ""
    rows = []
    for r in results:
        if "error" in r:
            continue
        pt, ci = r["points"], r["boot"]["ci90"]
        for ls in LABEL_SETS:
            name = {"RAW": "RAW", "FROZEN": "CORR-frozen",
                    "REFIT": "CORR-refit"}[ls]
            if ls == "FROZEN" and "IN-SAMPLE" in r["rule_frozen"]["in_sample"]:
                name += r"$^\dagger$"
            p = pt[ls]
            rows.append(
                f"{esc(r['model'])} & {name} & {r['n_pairs']} & "
                f"{_t(p['P@2'])} & {_t(p['R@2'])} & "
                f"{_t(p['F1@2'])} {_tci(ci[ls]['F1@2'])} & "
                f"{_t(p['ratio@2'])} {_tci(ci[ls]['ratio@2'])} & "
                f"{_t(p['MAE'])} & {_t(p['offset'], sign=True)} & "
                f"{_t(p['kappa'])} \\\\")
    write_tex(
        tex_dir, f"batch6_poolext_{ds}.tex", rows,
        f"{stamp}Batch 6 / A: label-pool extension on {ds}. Pair universe = "
        f"human-judged pairs with a usable judge label at every politeness "
        f"level. Relevance at human grade >= 2. Columns: judge, label set, "
        f"n pairs, precision, recall, F1 [90\\% CI], relevant-rate ratio "
        f"|LLM-rel|/|HUMAN-rel| [90\\% CI], grade MAE, mean offset (LLM - "
        f"human), pair-label linear kappa. CIs: query-clustered bootstrap "
        f"(B={meta['B']}, seed {meta['seed']}); CORR-refit CIs are "
        f"{REFIT_CI_NOTE}. CORR-frozen = frozen {FAMILY_OF[ds]}-family rule "
        f"(rules sha256 {str(meta.get('rules_sha256'))[:12]}); $\\dagger$ = "
        f"rule fitted on data including {ds} (in-sample). CORR-refit = "
        f"query-disjoint 2-fold in-domain cross-fit, fold seed "
        f"{meta['fold_seeds'][0]}. Gains are operating-point artifact "
        f"removal, not judgment quality.")
    if se:
        srows = []
        for r in results:
            if "error" in r or "sample_eff" not in r:
                continue
            for n, v in r["sample_eff"].items():
                if "skipped" in v:
                    continue
                srows.append(
                    f"{esc(r['model'])} & {n} & {v['pool_pairs_mean']:.0f} & "
                    f"{_tms(v['RAW_F1@2'])} & {_tms(v['REFIT_F1@2'])} & "
                    f"{_tms(v['RAW_P@2'])} & {_tms(v['REFIT_P@2'])} & "
                    f"{_tms(v['RAW_R@2'])} & {_tms(v['REFIT_R@2'])} & "
                    f"{_tms(v['RAW_ratio@2'])} & {_tms(v['REFIT_ratio@2'])} \\\\")
        write_tex(
            tex_dir, f"batch6_poolext_{ds}_sampleff.tex", srows,
            f"{stamp}Batch 6 / A sample efficiency on {ds}: dial + map "
            f"refitted on n human-labeled pairs (query-clustered draws, "
            f"B={meta['se_B']}), evaluated on the query-disjoint extension "
            f"pool. Columns: judge, n, mean pool size, F1@2 RAW, F1@2 "
            f"CORR-refit, P@2 RAW, P@2 CORR-refit, R@2 RAW, R@2 CORR-refit, "
            f"ratio RAW, ratio CORR-refit (mean $\\pm$ sd over draws).")

    drop = ("labels", "seed_labels", "pair_fold", "universe", "h")

    def clean(r):
        if "error" in r:
            return {"model": r["model"], "error": r["error"], "info": r["info"],
                    "adjudication": r.get("adjudication")}
        out = {k: v for k, v in r.items() if k not in drop}
        out["points"] = {ls: {ep: fnum(v) for ep, v in d.items()}
                         for ls, d in r["points"].items()}
        out["point_diff"] = {k: {ep: fnum(v) for ep, v in d.items()}
                             for k, d in r["point_diff"].items()}
        out["delta_cont"] = {k: fnum(v) for k, v in r["delta_cont"].items()}
        if out.get("adjudication"):
            out["adjudication"] = {
                "conjunction": r["adjudication"]["conjunction"],
                "endpoints": {E: {k: (fnum(v) if k == "value" else v)
                                  for k, v in e.items()}
                              for E, e in r["adjudication"]["endpoints"].items()}}
        return out

    payload = {"meta": {**meta, "dataset": ds},
               "judges": {r["model"]: clean(r) for r in results}}
    path = data_dir / "derived" / f"batch6_poolext_{ds}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                               allow_nan=False, default=_json_default),
                    encoding="utf-8")
    print(f"[json] wrote {path}")
    write_labels(results, ds, data_dir)


def write_labels(results: list[dict], ds: str, data_dir: Path) -> Path | None:
    """S7: pair-level labels (HUMAN, RAW, FROZEN, REFIT per fold seed)."""
    frames = []
    for r in results:
        if "error" in r:
            continue
        f = pd.DataFrame({
            "dataset": ds, "model_id": r["model"],
            "qid": [k[0] for k in r["universe"]],
            "docid": [k[1] for k in r["universe"]],
            "human": r["h"], "RAW": r["labels"]["RAW"],
            "FROZEN": r["labels"]["FROZEN"], "REFIT": r["labels"]["REFIT"],
            "fold": r["pair_fold"],
        })
        for s, lab in r["seed_labels"].items():
            f[f"REFIT_seed{s}"] = lab
        frames.append(f)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    path = data_dir / "derived" / f"batch6_labels_{ds}.parquet"
    try:
        df.to_parquet(path, index=False)
    except ImportError:
        path = path.with_suffix(".pkl")
        df.to_pickle(path)
        print(f"[labels] no parquet engine — fell back to pickle")
    print(f"[labels] wrote {path} ({len(df)} rows)")
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return fnum(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def finalize(res_by_ds: dict, datasets, data_dir: Path, meta: dict,
             se: bool) -> None:
    """No-op invariant FIRST, then (and only then) every write."""
    enforce_noop([r for ds in datasets for r in res_by_ds[ds]])
    for ds in datasets:
        write_outputs(res_by_ds[ds], ds, data_dir, meta, se)


# ═══════════════════════════════════════════════════════════════════════════
# Driver shared by `run` and `selftest`
# ═══════════════════════════════════════════════════════════════════════════

def analyze_datasets(df, qrels, rules_all, datasets, judges, flagships,
                     data_dir, B, seed, se_sizes, se_B,
                     fold_seeds=FOLD_SEEDS) -> dict:
    out: dict = {}
    frozen_pairs: dict = {}
    for ds in datasets:
        res_ds = []
        for model in judges:
            d, restricted = scope_rows(df, model, ds, flagships, data_dir,
                                       frozen_pairs)
            if d.empty:
                res_ds.append({"model": model, "dataset": ds, "info": {},
                               "error": f"no {ds} rows"})
                continue
            rule = frozen_rule(rules_all, model, ds)
            if rule is None:
                res_ds.append({"model": model, "dataset": ds, "info": {},
                               "error": f"no frozen {FAMILY_OF[ds]} rule"})
                continue
            cell = prepare_cell(model, ds, d, qrels, seed)
            cell["info"]["flagship"] = restricted
            if "error" in cell:
                res_ds.append({"model": model, "dataset": ds,
                               "info": cell["info"], "error": cell["error"]})
                continue
            res_ds.append(analyze_cell(cell, rule, B, seed, se_sizes, se_B,
                                       fold_seeds, qrels))
        out[ds] = res_ds
    return out


def print_dataset_header(ds, meta):
    print()
    print("=" * 96)
    print(f"  BATCH 6 / A — LABEL-POOL EXTENSION   dataset={ds}  "
          f"(frozen family: {FAMILY_OF[ds]})   [{meta['blind_status']}]")
    print("=" * 96)
    print(f"  rules: {meta['rules_path']}  (git {meta['rules_git_hash']}, "
          f"{meta['rules_timestamp']})")
    print(f"  rules_sha256: {meta['rules_sha256']}")
    print(f"  code git {meta['git_hash']}  code dirty (scripts/, src/): "
          f"{meta['code_dirty']}   PREDICTIONS.md registration commit: "
          f"{meta['registration_commit']}  (clean: {meta['predictions_clean']})")
    print(f"  bootstrap: query-clustered B={meta['B']} seed={meta['seed']}, 90% "
          f"percentile CI ({REFIT_CI_NOTE} for REFIT)")
    print(f"  fold seeds: {meta['fold_seeds']} (primary {meta['fold_seeds'][0]})"
          f"   tie bands: TIE_F1={TIE_F1} TIE_RATIO={TIE_RATIO}")
    if ds == "dl22":
        print("  STATUS: held-out — the frozen msmarco rule never saw dl22.")
    elif ds == "dl21":
        print("  STATUS: CORR-FROZEN PARTIALLY IN-SAMPLE (msmarco rule fitted "
              "on dl19+dl20+dl21).")
    elif ds == "antique":
        print("  STATUS: CORR-FROZEN FULLY IN-SAMPLE (antique rule fitted on "
              "antique itself) — a reference, not a transfer test.")
    if meta["blind_status"] == "NON-BLIND":
        print("  *** NON-BLIND RUN — not a registered unboxing ***")
    print("  FRAMING: human-scale usability of extended labels; gains are "
          "operating-point artifact")
    print("  removal against THESE annotators, never judgment quality.")


# ═══════════════════════════════════════════════════════════════════════════
# Subcommands
# ═══════════════════════════════════════════════════════════════════════════

def _judges(spec):
    return [s.strip() for s in spec.split(",") if s.strip()] if spec else JUDGES


def _flagships(spec):
    return (frozenset(s.strip() for s in spec.split(",") if s.strip())
            if spec is not None else FLAGSHIPS)


def _fold_seeds(spec) -> list[int]:
    seeds = [int(x) for x in str(spec).split(",") if x.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        sys.exit("ERROR: --fold-seeds must be a non-empty list of distinct ints.")
    return seeds


def cmd_run(args) -> None:
    datasets = resolve_datasets(args.datasets)
    blind = gate(datasets, args.unseal_dl22, args.non_blind)   # BEFORE data
    data_dir = resolve_data_dir(args.data_dir)
    rules_all, rules_path, sha = load_rules(data_dir, args.rules)
    block = read_block()
    reg = parse_registration(block) if block else None
    if reg is not None:
        check_rules_sha(reg, sha)                              # BEFORE data
        for cur, E, tok in reg["invalid"]:
            print(f"[registration] WARNING: {cur} B6-{E}: unparseable "
                  f"prediction {tok!r} — treated as unregistered.",
                  file=sys.stderr)
    fold_seeds = _fold_seeds(args.fold_seeds)
    jpath = Path(args.judgments) if args.judgments else \
        data_dir / "derived" / "judgments.parquet"
    df = load_judgments(jpath)
    df = df[df.dataset.astype(str).isin(datasets)]
    qrels = load_human(datasets, args.qrels_json)
    judges, flagships = _judges(args.judges), _flagships(args.flagships)
    sizes = ([int(x) for x in args.sizes.split(",") if x.strip()]
             if args.sample_efficiency else None)
    pclean = predictions_clean(predictions_path())[0]
    meta = {"script": "scripts/batch6_poolext.py", "git_hash": git_hash(),
            "code_dirty": code_dirty(),
            "blind_status": blind,
            "registration_commit": registration_commit(),
            "predictions_clean": pclean,
            "rules_path": str(rules_path), "rules_sha256": sha,
            "rules_git_hash": rules_all.get("git_hash"),
            "rules_timestamp": rules_all.get("timestamp_utc"),
            "B": args.b, "seed": args.seed, "ci": "90% percentile",
            "fold_seeds": fold_seeds, "tie_f1": TIE_F1, "tie_ratio": TIE_RATIO,
            "e6": {"n_lo": E6_N_LO, "n_hi": E6_N_HI, "frac": E6_FRAC},
            "adjudication_rules": ADJUDICATION_RULES,
            "se_B": args.se_b if args.sample_efficiency else None,
            "se_sizes": sizes, "qrels_source": args.qrels_json or "src.qrels",
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc)
            .isoformat(timespec="seconds")}

    res = analyze_datasets(df, qrels, rules_all, datasets, judges, flagships,
                           data_dir, args.b, args.seed, sizes, args.se_b,
                           fold_seeds)
    for ds in datasets:
        print_dataset_header(ds, meta)
        for r in res[ds]:
            print_cell(r)
    allres = [r for ds in datasets for r in res[ds]]
    if not any("error" not in r for r in allres):
        sys.exit("\nERROR: no (judge, dataset) cell produced a result.")
    worst = max(r["kappa_crosscheck_max_absdiff"] for r in allres
                if "error" not in r)
    print(f"\n  fast endpoints vs sklearn/vector paths (kappa, P/R@1,2, F1, "
          f"calibration, MAE, offset): max |diff| = {worst:.2e}")
    enforce_noop(allres)                     # before backfill and any write
    preds = reg["preds"] if reg else None
    if preds is None:
        print("\n  [no registration] verdicts below are 'unregistered'.")
    for r in allres:
        r["adjudication"] = adjudicate(r, preds)
        if (preds and not args.sample_efficiency
                and "E6" in preds.get((r["model"], r["dataset"]), {})):
            print(f"  WARNING: {r['model']} @ {r['dataset']}: E6 is registered "
                  f"but --sample-efficiency was not run -> not adjudicated.")
    meta["tally"] = {ds: print_backfill(res[ds], ds, blind) for ds in datasets}
    print()
    finalize(res, datasets, data_dir, meta, bool(sizes))
    print("\nREMINDER: report gains as operating-point artifact removal, "
          "never as judgment quality.")


def cmd_validate(args) -> None:
    """Inputs + coverage, counts only. Computes NO agreement metric."""
    datasets = resolve_datasets(args.datasets)
    data_dir = resolve_data_dir(args.data_dir)
    jpath = Path(args.judgments) if args.judgments else \
        data_dir / "derived" / "judgments.parquet"
    df = load_judgments(jpath)
    df = df[df.dataset.astype(str).isin(datasets)]
    rules_all, rules_path, sha = load_rules(data_dir, args.rules)
    qrels = load_human(datasets, args.qrels_json, quiet=False)
    judges, flagships = _judges(args.judges), _flagships(args.flagships)
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    block = read_block()
    print("=" * 96)
    print("  BATCH 6 VALIDATE — inputs and coverage only (no kappa, no "
          "judge-vs-human agreement)")
    print("=" * 96)
    print(f"  judgments: {jpath}  ({len(df)} rows on {datasets})")
    print(f"  rules    : {rules_path}  (git {rules_all.get('git_hash')})")
    print(f"  rules_sha256: {sha}")
    print(f"  B6 registration block in PREDICTIONS.md: "
          f"{'present' if block else 'ABSENT'}"
          + ("" if not block else
             f"; registered rules_sha256 "
             f"{'MATCHES' if parse_registration(block)['rules_sha256'] == sha else 'DOES NOT MATCH'}"))
    problems = []
    frozen_pairs: dict = {}
    for ds in datasets:
        print()
        print(f"  [{ds}]  human-judged pairs in qrels: "
              f"{sum(1 for k in qrels if k[0] == ds)}")
        print(f"    {'judge':<30} {'rows':>7} {'pairs':>6} {'judged':>6} "
              f"{'univ':>6} {'drop':>5} {'nq':>4} {'fold q':>8} "
              f"{'fold pairs':>11}  rule")
        for model in judges:
            d, restricted = scope_rows(df, model, ds, flagships, data_dir,
                                       frozen_pairs)
            rule = frozen_rule(rules_all, model, ds)
            if d.empty:
                print(f"    {model:<30} NO ROWS")
                problems.append(f"{model}@{ds}: no rows")
                continue
            if rule is None:
                problems.append(f"{model}@{ds}: no frozen {FAMILY_OF[ds]} rule")
            elif rule.get("delta_hat") is None:
                problems.append(f"{model}@{ds}: rule has no delta_hat (E5)")
            cell = prepare_cell(model, ds, d, qrels, args.seed)
            i = cell["info"]
            if "error" in cell:
                print(f"    {model:<30} {i['n_rows']:>7} ERROR {cell['error']}")
                problems.append(f"{model}@{ds}: {cell['error']}")
                continue
            if min(i["fold_pairs"]) == 0:
                problems.append(f"{model}@{ds}: empty cross-fit fold")
            rtxt = ("MISSING" if rule is None else
                    f"L{rule['dial_level']} {rule['monotone_map']}"
                    + (" no-op" if is_noop_rule(rule) else "")
                    + (" in-sample" if ds in rule["datasets"] else ""))
            print(f"    {model[:30]:<30} {i['n_rows']:>7} "
                  f"{i['n_collected_pairs']:>6} {i['n_collected_human_judged']:>6} "
                  f"{i['n_pairs']:>6} {i['n_dropped_unusable']:>5} "
                  f"{i['n_queries']:>4} {str(i['fold_queries']):>8} "
                  f"{str(i['fold_pairs']):>11}  {rtxt}"
                  f"{'  [flagship 40%]' if restricted else ''}")
            ppl = i["prompts_per_level"]
            print(f"      prompt_ids per level: {ppl}"
                  + ("" if set(ppl.values()) == {3} else "  <- expected 3"))
            infeas = [n for n in sizes if n > i["n_pairs"] - SE_MIN_POOL_PAIRS]
            if infeas:
                print(f"      sample-efficiency sizes {infeas} infeasible "
                      f"(universe {i['n_pairs']}) — will be SKIPPED"
                      + (" (E6 then not adjudicated)"
                         if E6_N_HI in infeas or E6_N_LO in infeas else ""))
    print()
    if problems:
        print("  PROBLEMS:")
        for p in problems:
            print(f"    - {p}")
        sys.exit(1)
    print("  VALIDATE OK — no agreement metric was computed.")


def paste_block(rules_all: dict, rules_path: Path, sha: str, datasets,
                judges, flagships, fold_seeds=FOLD_SEEDS) -> list[str]:
    head = git_hash()
    rc, dirty = _git("status", "--porcelain", "--", "scripts", "src")
    fs = list(fold_seeds)
    L = []
    L.append(f"# {B6_MARKER}（batch6，scripts/batch6_poolext.py）")
    L.append("")
    L.append(f"登记时间：____。分析代码 git hash：{head}"
             + ("（⚠ scripts/ 或 src/ 有未提交改动 — 先 commit 代码再登记）"
                if dirty else "") + "。")
    L.append(f"rules_sha256: {sha}")
    L.append(f"冻结规则文件：{rules_path.name}（git {rules_all.get('git_hash')}，"
             f"时间戳 {rules_all.get('timestamp_utc')}）；run 时 sha256 不符即中止。")
    L.append("分析：`run` 子命令。pair universe = human-judged ∩ judge 在 L1–L5 "
             "均有可用标签（parse_ok & 数值 score）；")
    L.append("标签 = 该档全部记录（paraphrase×run）的均值后 round-half-up"
             "（文中\"paraphrase-mean\"即此记录均值；设计平衡时与先按 paraphrase "
             "取均值再平均完全相同）。RAW = L3；CORR-FROZEN = 家族冻结规则"
             "（dl21/dl22 用 msmarco，antique 用 antique；dial 档 + monotone map "
             "逐记录施加于取均值之前）；")
    L.append("CORR-REFIT = 同数据集 query-disjoint 2 折交叉拟合（md5(seed|dataset|qid) "
             f"分折；主 fold seed {fs[0]}，稳健性 seeds {fs[1:]} 仅报告不计分；"
             "dial = argmin|Δ+D(ℓ)|，map = 35 个单调映射中训练折 paraphrase-mean "
             "线性 κ 最大者，均为 batch4 估计量），测试折拼接。")
    L.append(f"不确定性：query-clustered bootstrap B={B_BOOT} seed {SEED}，90% "
             "百分位 CI；差值为同一重采样 query 集上的配对差。"
             f"REFIT 的 CI：{REFIT_CI_NOTE}。")
    L.append(f"样本效率（E6）：n ∈ {list(SE_SIZES)}，B={B_SE} 次 query 不放回抽样，"
             "校准集与扩充池 query-disjoint。")
    L.append("判定纪律（逐字；代码在 BACKFILL 中同文打印）：")
    L.extend(ADJUDICATION_RULES)
    L.append("填写格式：把每条终点行末\"预测：\"之后的下划线占位替换为 >1 或 <1"
             "（E1），是 或 否（E2–E6）；同一 token 后不得紧接其他文字（如需说明，"
             "空格后写）。块内任何下划线占位未填，dl22 门控即拒绝运行。")
    L.append("盲态声明：____")
    n_end = 0
    for ds in datasets:
        L.append("")
        status = {"dl22": "held-out（冻结 msmarco 规则未见 dl22）",
                  "dl21": "冻结规则部分 in-sample（msmarco 规则拟合于 "
                          "dl19+dl20+dl21）",
                  "antique": "冻结规则完全 in-sample（antique 规则拟合于 "
                             "antique 本身）"}[ds]
        L.append(f"## {ds} — {status}")
        for model in judges:
            rule = frozen_rule(rules_all, model, ds)
            fl = "（40% 冻结子样本）" if model in flagships else ""
            if rule is None:
                L.append(f"### {model} — {ds}{fl} — 无冻结 {FAMILY_OF[ds]} 规则，"
                         "不入登记")
                continue
            noop = is_noop_rule(rule)
            dom = domain_of(rule, model, ds)
            L.append(f"### {model} — {ds}{fl} — 冻结规则 dial=L{rule['dial_level']} "
                     f"map={list(rule['monotone_map'])}，delta_hat="
                     f"{float(rule['delta_hat']):+.4f}（E5 {'严域' if dom == 'strict' else '宽域'}）"
                     + ("（identity@L3 → no-op 对照）" if noop else ""))
            if noop:
                L.append("- B6-E0 [no-op] CORR-FROZEN ≡ RAW 全终点逐位相同"
                         "（结构性不变量，代码 hard-fail 强制）")
                n_end += 1
            L.append("- B6-E1 [ratio direction] RAW 相关率比 @≥2（|LLM-rel|/"
                     "|HUMAN-rel|），取值 >1 或 <1｜预测：____")
            n_end += 1
            if noop:
                L.append("- B6-E2 [ratio convergence] N/A（结构性：FROZEN ≡ RAW）")
            else:
                L.append("- B6-E2 [ratio convergence] CORR-FROZEN 使 "
                         "|ratio@≥2 − 1| 相对 RAW 收缩，取值 是/否｜预测：____")
                n_end += 1
            L.append(f"- B6-E3 [F1 refit≥raw] F1@≥2(CORR-REFIT, fold seed {fs[0]}) − "
                     f"F1@≥2(RAW) ≥ 0，取值 是/否（{REFIT_CI_NOTE}）｜预测：____")
            n_end += 1
            if noop:
                L.append("- B6-E4 [F1 refit≥frozen] ≡ B6-E3（FROZEN ≡ RAW），"
                         "不单独计分")
            else:
                L.append(f"- B6-E4 [F1 refit≥frozen] F1@≥2(CORR-REFIT, fold seed "
                         f"{fs[0]}) − F1@≥2(CORR-FROZEN) ≥ 0，取值 是/否"
                         f"（{REFIT_CI_NOTE}）｜预测：____")
                n_end += 1
            ep5 = "R@2（recall）" if dom == "strict" else "P@2（precision）"
            L.append(f"- B6-E5 [recall direction @≥2] {'严域' if dom == 'strict' else '宽域'}"
                     f"：{ep5}(CORR-REFIT) − {ep5}(RAW) ≥ 0，取值 是/否"
                     f"（{REFIT_CI_NOTE}）｜预测：____")
            n_end += 1
            L.append(f"- B6-E6 [sample efficiency] mean dF1@2(REFIT−RAW)@n={E6_N_LO} − "
                     f"{E6_FRAC} × mean dF1@2(REFIT−RAW)@n={E6_N_HI} ≥ 0，取值 是/否"
                     "（需 --sample-efficiency；两档未都运行则 not adjudicated）"
                     "｜预测：____")
            n_end += 1
            L.append("- 开箱结果（run 后回填）：")
    L.append("")
    L.append(f"合计可判定终点：{n_end}（含结构性 no-op 断言 E0）。")
    return L


def cmd_paste(args) -> None:
    """Registration block. Reads ONLY correction_rules.json — blind."""
    datasets = resolve_datasets(args.datasets)
    data_dir = resolve_data_dir(args.data_dir)
    rules_all, rules_path, sha = load_rules(data_dir, args.rules)
    lines = paste_block(rules_all, rules_path, sha, datasets,
                        _judges(args.judges), _flagships(args.flagships),
                        _fold_seeds(args.fold_seeds))
    print("PASTE INTO paper/PREDICTIONS.md  (fill every ____, commit, THEN run;"
          " nothing below was computed from data)")
    print("-" * 96)
    print("\n".join(lines))
    print("-" * 96)


# ═══════════════════════════════════════════════════════════════════════════
# followup — EXPLORATORY evaluator follow-up (not pre-registered)
#   (A) 2x2 dial x map factorial      (B) operating curve vs the (dial,map) grid
#   (C) F1-objective refit + nested (fit-aware) bootstrap
#   (E) E6' sample efficiency in calibration QUERIES
# ═══════════════════════════════════════════════════════════════════════════

EXPLORATORY = "EXPLORATORY — not pre-registered"
NESTED_B = 200
E6P_SIZES = (2, 4, 8, 16, 32)
E6P_B = 1000
E6P_LO, E6P_HI, E6P_FRAC = 8, 32, 0.8
CATASTROPHIC_DF1 = -0.05
FX_EPS = ("ratio@2", "absdev@2", "P@2", "R@2", "F1@2", "MAE", "delta_cont")
FX_ARMS = ("RAW", "DIAL", "MAP", "FULL")
KAPPA_CHUNK = 200

NMAP = len(MONOTONE_MAPS)
MAP_INDEX = {tuple(int(x) for x in m): i for i, m in enumerate(MONOTONE_MAPS)}
MAP_ID_IDX = MAP_INDEX[IDENTITY_MAP]
# combo c = (dial - 1) * NMAP + map index — the full (L1..L5) x 35-map grid
COMBOS = [(int(l), tuple(int(x) for x in m)) for l in LEVELS
          for m in MONOTONE_MAPS]
COMBO_DIAL = np.array([c[0] for c in COMBOS], dtype=np.int64)
_MAPM = np.zeros((NMAP, 4, 4))
for _i, _m in enumerate(MONOTONE_MAPS):
    _MAPM[_i, np.arange(4), list(_m)] = 1.0
# F1-objective tie-break order (after F1, then pair-label kappa): RAW first,
# then dials by distance to L3 (L2, L4, L1, L5), identity map first.
_DIAL_PREF = (3, 2, 4, 1, 5)
_MAP_PREF = [MAP_ID_IDX] + [i for i in range(NMAP) if i != MAP_ID_IDX]
PREF_ORDER = np.array([(l - 1) * NMAP + mi for l in _DIAL_PREF
                       for mi in _MAP_PREF], dtype=np.int64)


def combo_of(dial, mono) -> int:
    return (int(dial) - 1) * NMAP + MAP_INDEX[tuple(int(x) for x in mono)]


RAW_C = combo_of(3, IDENTITY_MAP)


def combo_name(c: int) -> str:
    l, m = COMBOS[int(c)]
    return f"L{l}/{''.join(map(str, m))}"


def judge_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9.\-]+", "_", model).strip("_")


def _ci90(x) -> list:
    x = np.asarray(x, dtype=float)
    f = x[np.isfinite(x)]
    if f.size == 0:
        return [None, None, 0]
    return [float(np.percentile(f, CI_LO)), float(np.percentile(f, CI_HI)),
            int(f.size)]


def kappa_vec(cm) -> np.ndarray:
    """Linear-weighted kappa over the trailing 4x4 axes (kappa_from_cm math)."""
    cm = np.asarray(cm, dtype=float)
    tot = cm.sum(axis=(-2, -1))
    with np.errstate(divide="ignore", invalid="ignore"):
        o = cm / tot[..., None, None]
        e = o.sum(axis=-1)[..., :, None] * o.sum(axis=-2)[..., None, :]
        po = (_W * o).sum(axis=(-2, -1))
        pe = (_W * e).sum(axis=(-2, -1))
        k = 1.0 - po / pe
    return np.where((tot > 0) & (pe != 0) & np.isfinite(pe), k, np.nan)


def followup_cache(cell: dict, fold_seed: int) -> dict:
    """Per-QUERY sufficient statistics: every fit / evaluation below is a
    weighted sum over queries, so bootstrap and sub-sampling replicates are
    matrix products (query multiplicities as weights) instead of row loops."""
    P, cnt, h = cell["P"], cell["cnt"], cell["h"]
    nq, pair_q, nP = cell["n_queries"], cell["pair_q"], len(cell["h"])
    row_q = pair_q[cell["row_u_P"]]
    qname = np.empty(nq, dtype=object)
    qname[pair_q] = np.asarray(cell["pair_qid"], dtype=object)
    qfold = np.array([fold_of(cell["dataset"], q, fold_seed) for q in qname],
                     dtype=np.int64)
    ok, lv = P.ok, P.lvl
    # batch-4 delta_D inputs: human mean over ALL matched rows, score means
    # over the scorable rows of each level
    n_all = np.bincount(row_q, minlength=nq).astype(float)
    h_all = np.bincount(row_q, weights=P.h.astype(float), minlength=nq)
    nok = np.zeros((nq, len(LEVELS)))
    ssum = np.zeros((nq, len(LEVELS)))
    for i, l in enumerate(LEVELS):
        m = ok & (lv == l)
        nok[:, i] = np.bincount(row_q[m], minlength=nq)
        ssum[:, i] = np.bincount(row_q[m], weights=P.s[m], minlength=nq)
    # batch-4 level_kappas inputs: per-variant confusion on scorable rows
    V = P.n_variants
    keys = (((row_q[ok] * V + P.v[ok]) * 4 + P.h[ok]) * 4
            + P.s[ok].astype(np.int64))
    vcm = np.bincount(keys, minlength=nq * V * 16).reshape(nq, V, 4, 4)\
        .astype(float)
    vlevel = np.array([P.vlevel[v] for v in range(V)], dtype=np.int64)
    # every grid combo: pair labels, per-query confusion, record-mean sums
    C = len(COMBOS)
    labs = np.empty((C, nP), dtype=np.int64)
    ccm = np.empty((C, nq, 4, 4))
    num = np.empty((C, nq))
    for c, (l, m) in enumerate(COMBOS):
        lab = labels_from_counts(cnt[l - 1], m)
        labs[c] = lab
        ccm[c] = cm_by_query(h, lab, pair_q, nq)
        num[c] = np.bincount(pair_q, weights=cnt[l - 1] @ np.asarray(m, float),
                             minlength=nq)
    den = np.stack([np.bincount(pair_q, weights=cnt[l - 1].sum(1).astype(float),
                                minlength=nq) for l in LEVELS])
    return {
        "nq": nq, "nP": nP, "pair_q": pair_q, "qfold": qfold, "qname": qname,
        "n_all": n_all, "h_all": h_all, "nok": nok, "ssum": ssum,
        "vcm": vcm, "vlevel": vlevel, "labs": labs, "ccm": ccm, "num": num,
        "den": den,
        "tp": ccm[:, :, 2:, 2:].sum(axis=(-2, -1)),
        "lr": ccm[:, :, :, 2:].sum(axis=(-2, -1)),
        "hr": np.bincount(pair_q, weights=(h >= 2).astype(float), minlength=nq),
        "ppq": np.bincount(pair_q, minlength=nq).astype(float),
        "row_q": row_q,
        # flattened [nq, X*16] views: weighted sums over queries become one
        # BLAS matmul (integer-valued floats, so exact in any order)
        "vcm_flat": vcm.reshape(nq, V * 16),
        "ccm_flat": ccm.transpose(1, 0, 2, 3).reshape(nq, C * 16),
    }


def fit_kappa_vec(fc: dict, Wt) -> tuple[np.ndarray, np.ndarray]:
    """batch-4 fit_rule for many query-weight vectors at once.

    dial = argmin_l |Delta + D(l)| (ties -> lower level, as best_level);
    map  = first maximiser, in MONOTONE_MAPS order, of the paraphrase-mean
    linear kappa at the dial over variants with n > MIN_N_KAPPA (identity if
    none is finite), as fit_monotone_map.  Integer weights == duplicated rows.
    Returns (combo index [B], ok [B]); ok=False where delta_D would be None.
    """
    Wt = np.atleast_2d(np.asarray(Wt, dtype=float))
    B = Wt.shape[0]
    N, H = Wt @ fc["n_all"], Wt @ fc["h_all"]
    NOK, S = Wt @ fc["nok"], Wt @ fc["ssum"]
    ok = (N > 0) & (NOK > 0).all(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        means = S / NOK
        delta = means[:, 2] - H / N
        D = means - means[:, 2:3]
        dist = np.abs(delta[:, None] + D)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    dial = np.argmin(dist, axis=1) + 1
    out = np.full(B, RAW_C, dtype=np.int64)
    vl = fc["vlevel"]
    for s0 in range(0, B, KAPPA_CHUNK):
        sl = slice(s0, min(B, s0 + KAPPA_CHUNK))
        VC = (Wt[sl] @ fc["vcm_flat"]).reshape(-1, fc["vcm"].shape[1], 4, 4)
        tot = VC.sum(axis=(-2, -1))                               # [b, V]
        K = kappa_vec(VC[:, None] @ _MAPM[None, :, None])         # [b, M, V]
        use = (vl[None, :] == dial[sl, None]) & (tot > MIN_N_KAPPA)
        acc = np.zeros(K.shape[:2])
        nv = np.zeros(K.shape[:2])
        for v in range(K.shape[2]):
            good = use[:, v][:, None] & np.isfinite(K[:, :, v])
            acc = acc + np.where(good, K[:, :, v], 0.0)
            nv = nv + good
        with np.errstate(divide="ignore", invalid="ignore"):
            km = np.where(nv > 0, acc / nv, -np.inf)
        mi = np.argmax(km, axis=1)
        mi = np.where(np.isfinite(km).any(axis=1), mi, MAP_ID_IDX)
        out[sl] = (dial[sl] - 1) * NMAP + mi
    out[~ok] = RAW_C
    return out, ok


def grid_f1(fc: dict, Wt) -> np.ndarray:
    """Pair-label F1@2 of every grid combo on query weights Wt -> [B, C]."""
    Wt = np.atleast_2d(np.asarray(Wt, dtype=float))
    TP, LR, HR = Wt @ fc["tp"].T, Wt @ fc["lr"].T, Wt @ fc["hr"]
    den = LR + HR[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, 2.0 * TP / den, np.nan)


def fit_f1_vec(fc: dict, Wt) -> tuple[np.ndarray, np.ndarray]:
    """F1-objective selection over the full (dial, map) grid: argmax of the
    train pair-label F1@2; ties -> higher train pair-label linear kappa ->
    PREF_ORDER (RAW first).  Never fails on a non-empty train set."""
    Wt = np.atleast_2d(np.asarray(Wt, dtype=float))
    F1 = grid_f1(fc, Wt)
    F1 = np.where(np.isfinite(F1), F1, -np.inf)[:, PREF_ORDER]
    K = kappa_vec((Wt @ fc["ccm_flat"]).reshape(Wt.shape[0], len(COMBOS), 4,
                                                4))[:, PREF_ORDER]
    cand = F1 == F1.max(axis=1, keepdims=True)
    key = np.where(cand, np.where(np.isfinite(K), K, -1e300), -np.inf)
    out = PREF_ORDER[np.argmax(key, axis=1)]
    return out, Wt.sum(axis=1) > 0


def crossfit_vec(fc: dict, Wq, fitter):
    """2-fold query-disjoint cross-fit (fold of a query fixed by its md5 hash,
    so a resampled query keeps its fold) for many query-multiplicity vectors.
    Returns (combos [B,2] (test fold k -> column k), ok [B], test cm [B,4,4])."""
    Wq = np.atleast_2d(np.asarray(Wq, dtype=float))
    B = Wq.shape[0]
    combos = np.full((B, 2), RAW_C, dtype=np.int64)
    ok = np.ones(B, dtype=bool)
    cm = np.zeros((B, 4, 4))
    for k in (0, 1):
        infold = fc["qfold"] == k
        tr, te = Wq * (~infold), Wq * infold
        c, okk = fitter(fc, tr)
        ok &= okk & (tr.sum(axis=1) > 0) & (te.sum(axis=1) > 0)
        c = np.where(ok, c, RAW_C)
        combos[:, k] = c
        cm += np.einsum("bq,bqij->bij", te, fc["ccm"][c])
    return combos, ok, cm


def arm_query_combos(fc: dict, fold_combos) -> np.ndarray:
    """Per-query combo for an arm defined by (combo for test fold 0, fold 1)."""
    return np.where(fc["qfold"] == 0, int(fold_combos[0]), int(fold_combos[1]))


def arm_eval(fc: dict, arms: dict, Wt: np.ndarray) -> tuple[dict, dict, dict]:
    """Point endpoints, bootstrap replicates and labels for per-query-combo arms."""
    nq, ar = fc["nq"], np.arange(fc["nq"])
    pts, reps, labs = {}, {}, {}
    hden = Wt @ fc["n_all"]
    hnum = Wt @ fc["h_all"]
    qmean = fc["h_all"].sum() / fc["n_all"].sum()
    for name, cq in arms.items():
        cmq = fc["ccm"][cq, ar]
        numq = fc["num"][cq, ar]
        denq = fc["den"][COMBO_DIAL[cq] - 1, ar]
        p = {ep: float(v) for ep, v in endpoints(cmq.sum(axis=0)).items()}
        p["delta_cont"] = float(numq.sum() / denq.sum() - qmean)
        pts[name] = p
        r = endpoints(np.einsum("bq,qij->bij", Wt, cmq))
        with np.errstate(divide="ignore", invalid="ignore"):
            r["delta_cont"] = (Wt @ numq) / (Wt @ denq) - hnum / hden
        reps[name] = r
        labs[name] = fc["labs"][cq[fc["pair_q"]], np.arange(fc["nP"])]
    return pts, reps, labs


def _diff(pts, reps, a, b, eps=FX_EPS) -> dict:
    return {ep: {"point": pts[a][ep] - pts[b][ep],
                 "ci90": _ci90(reps[a][ep] - reps[b][ep])} for ep in eps}


def _attrib(pts, reps, fam: str) -> dict:
    """total = FULL-RAW; dial = DIAL-RAW; map = MAP-RAW;
    interaction = (FULL-DIAL) - (MAP-RAW)  (exactly 0 when a choice is L3 /
    identity, since the coinciding arms are bit-identical)."""
    a = {x: f"{fam}:{x}" for x in FX_ARMS}
    out = {}
    for ep in FX_EPS:
        P_ = {x: pts[a[x]][ep] for x in FX_ARMS}
        R_ = {x: reps[a[x]][ep] for x in FX_ARMS}
        comp = {
            "total": (P_["FULL"] - P_["RAW"], R_["FULL"] - R_["RAW"]),
            "dial": (P_["DIAL"] - P_["RAW"], R_["DIAL"] - R_["RAW"]),
            "map": (P_["MAP"] - P_["RAW"], R_["MAP"] - R_["RAW"]),
            "interaction": ((P_["FULL"] - P_["DIAL"]) - (P_["MAP"] - P_["RAW"]),
                            (R_["FULL"] - R_["DIAL"]) - (R_["MAP"] - R_["RAW"])),
        }
        out[ep] = {k: {"point": v[0], "ci90": _ci90(v[1])}
                   for k, v in comp.items()}
    return out


def coincidence_checks(fc, fam_choices: dict, labs: dict, pts: dict,
                       reps: dict) -> list[dict]:
    """Where a fitted/frozen choice is L3 (or identity) in a test fold, the
    DIAL (MAP) arm must equal RAW and the MAP (DIAL) arm must equal FULL on
    that fold's pairs; when it holds in BOTH folds the endpoints and bootstrap
    replicates must be bit-identical.  Violations hard-fail."""
    out = []
    pf = fc["qfold"][fc["pair_q"]]
    for fam, ch in fam_choices.items():
        for kind, same_as_raw, same_as_full, pred in (
                ("dial==L3", "DIAL", "MAP", lambda k: ch["dial"][k] == 3),
                ("map==identity", "MAP", "DIAL",
                 lambda k: tuple(ch["map"][k]) == IDENTITY_MAP)):
            folds = [k for k in (0, 1) if pred(k)]
            if not folds:
                continue
            m = np.isin(pf, folds)
            bad = []
            if not np.array_equal(labs[f"{fam}:{same_as_raw}"][m],
                                  labs[f"{fam}:RAW"][m]):
                bad.append(f"{same_as_raw}!=RAW labels")
            if not np.array_equal(labs[f"{fam}:{same_as_full}"][m],
                                  labs[f"{fam}:FULL"][m]):
                bad.append(f"{same_as_full}!=FULL labels")
            if len(folds) == 2:
                for x, y in ((same_as_raw, "RAW"), (same_as_full, "FULL")):
                    for ep in FX_EPS:
                        if not _same(pts[f"{fam}:{x}"][ep], pts[f"{fam}:{y}"][ep]):
                            bad.append(f"point {x}!={y} {ep}")
                        if not np.array_equal(reps[f"{fam}:{x}"][ep],
                                              reps[f"{fam}:{y}"][ep],
                                              equal_nan=True):
                            bad.append(f"boot {x}!={y} {ep}")
            out.append({"family": fam, "kind": kind, "folds": folds,
                        "violations": bad})
            if bad:
                sys.exit(f"ERROR: factorial coincidence violated ({fam}, {kind},"
                         f" folds {folds}): {bad[:6]}")
    return out


def opcurve(fc: dict, cell: dict, marks: dict, fold_marks: dict):
    """(B) L3 vote-sum threshold curve + in-sample (dial, map) grid.

    Curve variable: v = K * (mean L3 record score), K = modal L3 records per
    pair (v == the plain vote sum for pairs with K records); positive iff
    v >= t, compared exactly as sum*K >= t*n.  t = 1..3K, plus the trivial
    anchors t = 0 (all positive) and t = 3K+1 (none) so a matched-ratio
    comparison is always defined.  Points between thresholds are linearly
    interpolated in (n_pos, TP) — achievable by randomising between them.
    """
    h, cnt3 = cell["h"], cell["cnt"][2]
    nrec = cnt3.sum(axis=1)
    vsum = cnt3 @ np.arange(4)
    vals, freq = np.unique(nrec, return_counts=True)
    K = int(vals[np.argmax(freq)])
    yt = h >= 2
    hr, N = int(yt.sum()), len(h)

    def stats(tp, lr):
        with np.errstate(divide="ignore", invalid="ignore"):
            return {"ratio@2": lr / hr if hr else np.nan,
                    "P@2": tp / lr if lr else np.nan,
                    "R@2": tp / hr if hr else np.nan,
                    "F1@2": 2.0 * tp / (lr + hr) if lr + hr else np.nan,
                    "n_pos": int(lr), "tp": int(tp)}

    raw_bin = fc["labs"][RAW_C] >= 2
    curve, raw_t = [], []
    for t in range(0, 3 * K + 2):
        yp = vsum * K >= t * nrec
        s = stats(int((yt & yp).sum()), int(yp.sum()))
        s.update({"t": t, "anchor": t in (0, 3 * K + 1)})
        curve.append(s)
        if np.array_equal(yp, raw_bin) and not s["anchor"]:
            raw_t.append(t)
    xs = {}
    for c in curve:
        xs[c["n_pos"]] = max(xs.get(c["n_pos"], -1), c["tp"])
    lr_pts = np.array(sorted(xs), dtype=float)
    tp_pts = np.array([xs[k] for k in sorted(xs)], dtype=float)

    def curve_f1_at(ratio):
        if ratio is None or not np.isfinite(ratio) or not hr:
            return np.nan
        lr = ratio * hr
        tp = float(np.interp(lr, lr_pts, tp_pts))
        return 2.0 * tp / (lr + hr) if lr + hr else np.nan

    def versus(f1, ratio):
        cf = curve_f1_at(ratio)
        d = f1 - cf if np.isfinite(f1) and np.isfinite(cf) else np.nan
        if not np.isfinite(d):
            v = "n/a"
        elif d > TIE_F1:
            v = "above"
        elif d < -TIE_F1:
            v = "below"
        else:
            v = "on"
        return cf, d, v

    rows = []
    for c in curve:
        rows.append({"kind": "anchor" if c["anchor"] else "threshold_L3",
                     "t": c["t"], "dial": 3, "map": "", "tag": "",
                     **{k: c[k] for k in ("ratio@2", "P@2", "R@2", "F1@2",
                                          "n_pos")},
                     "curve_F1_at_ratio": np.nan, "dF1_vs_curve": np.nan})
    tp_g, lr_g = fc["tp"].sum(axis=1), fc["lr"].sum(axis=1)
    grid = []
    for ci_, (l, m) in enumerate(COMBOS):
        s = stats(int(tp_g[ci_]), int(lr_g[ci_]))
        cf, d, _v = versus(s["F1@2"], s["ratio@2"])
        tags = [nm for nm, cc in marks.items() if cc == ci_]
        for nm, ccs in fold_marks.items():
            tags += [f"{nm}_fold{k}" for k, cc in enumerate(ccs) if cc == ci_]
        g = {"kind": "grid", "t": np.nan, "dial": l, "map": "".join(map(str, m)),
             "tag": "+".join(tags),
             **{k: s[k] for k in ("ratio@2", "P@2", "R@2", "F1@2", "n_pos")},
             "curve_F1_at_ratio": cf, "dF1_vs_curve": d}
        grid.append(g)
        rows.append(g)
    f1g = np.array([g["F1@2"] for g in grid], dtype=float)
    gb = int(np.nanargmax(f1g))
    thr = [c for c in curve if not c["anchor"]]
    f1c = np.array([c["F1@2"] for c in thr], dtype=float)
    cb = thr[int(np.nanargmax(f1c))]
    summary = {
        "K_records_mode": K, "frac_pairs_with_K": float((nrec == K).mean()),
        "t_range": [1, 3 * K], "raw_equivalent_t": raw_t,
        "curve_best": {"t": cb["t"], "F1@2": cb["F1@2"],
                       "ratio@2": cb["ratio@2"], "P@2": cb["P@2"],
                       "R@2": cb["R@2"]},
        "grid_ceiling": {"combo": combo_name(gb), "F1@2": grid[gb]["F1@2"],
                         "ratio@2": grid[gb]["ratio@2"],
                         "P@2": grid[gb]["P@2"], "R@2": grid[gb]["R@2"],
                         "note": "in-sample best-of-grid (optimistic ceiling)"},
        "curve": [{k: c[k] for k in ("t", "ratio@2", "P@2", "R@2", "F1@2",
                                     "n_pos")} for c in thr],
        "points": {},
    }
    return rows, summary, versus


def nested_bootstrap(fc: dict, B: int, seed: int,
                     return_reps: bool = False) -> dict:
    """Fit-aware CI: resample queries, redo the 2-fold cross-fit (kappa- and
    F1-objective) inside every replicate, evaluate on the replicate.
    ``return_reps`` (followup2 only) adds the per-replicate query weights and
    held-out F1@2 under key "reps"; the summary is unchanged."""
    idx = boot_indices(fc["nq"], B, seed)
    Wq = np.stack([np.bincount(r, minlength=fc["nq"]) for r in idx])\
        .astype(float)
    f1_raw = endpoints(np.einsum("bq,qij->bij", Wq, fc["ccm"][RAW_C]))["F1@2"]
    d, oks, dials, f1s = {}, {}, {}, {}
    for name, fitter in (("REFIT", fit_kappa_vec), ("F1REFIT", fit_f1_vec)):
        combos, ok, cm = crossfit_vec(fc, Wq, fitter)
        f1s[name] = endpoints(cm)["F1@2"]
        x = f1s[name] - f1_raw
        x[~ok] = np.nan
        d[name], oks[name] = x, ok
        vals, cts = np.unique(COMBO_DIAL[combos[ok]].ravel(), return_counts=True)
        dials[name] = {f"L{v}": int(c) for v, c in zip(vals, cts)}
    both = oks["REFIT"] & oks["F1REFIT"]
    dd = np.where(both, d["F1REFIT"] - d["REFIT"], np.nan)
    out = {"B": B, "seed": seed,
            "n_ok": {k: int(v.sum()) for k, v in oks.items()},
            "dial_freq_fold_fits": dials,
            "REFIT-RAW F1@2": {"ci90": _ci90(d["REFIT"]),
                               "mean": fnum(np.nanmean(d["REFIT"]))},
            "F1REFIT-RAW F1@2": {"ci90": _ci90(d["F1REFIT"]),
                                 "mean": fnum(np.nanmean(d["F1REFIT"]))},
            "F1REFIT-REFIT F1@2": {"ci90": _ci90(dd),
                                   "mean": fnum(np.nanmean(dd))}}
    if return_reps:
        out["reps"] = {"Wq": Wq, "RAW": f1_raw, "ok": oks, **f1s}
    return out


def _dist(x) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"n": 0}
    return {"n": int(x.size), "mean": float(x.mean()),
            "median": float(np.median(x)),
            "sd": float(x.std(ddof=1)) if x.size > 1 else 0.0,
            "min": float(x.min()), "max": float(x.max())}


def e6_prime(fc: dict, sizes, B: int, seed: int,
             fallback: bool = False) -> dict:
    """(E) E6': calibrate on q WHOLE queries (simple random sample without
    replacement), refit, evaluate dF1@2 (REFIT - RAW) on the query-disjoint
    extension pool (all other queries).  ``fallback`` (followup2 only) adds
    the RAW-fallback rate: share of evaluated draws whose fitted rule is
    RAW (L3 / identity map)."""
    nq, ppq = fc["nq"], fc["ppq"]
    out = {"ppq": _dist(ppq), "B": B, "seed": seed, "sizes": {}}
    for q in sizes:
        key = str(q)
        if q > nq - SE_MIN_POOL_QUERIES:
            out["sizes"][key] = {"skipped": f"q={q} leaves < "
                                            f"{SE_MIN_POOL_QUERIES} pool queries "
                                            f"({nq} queries)"}
            continue
        rng = np.random.default_rng([seed, q])
        chosen = np.argsort(rng.random((B, nq)), axis=1)[:, :q]
        cal = np.zeros((B, nq))
        np.put_along_axis(cal, chosen, 1.0, axis=1)
        pool = 1.0 - cal
        if (cal * pool).any():
            sys.exit("ERROR: E6' calibration and pool share a query.")
        pool_pairs = pool @ ppq
        F1 = grid_f1(fc, pool)
        base_ok = (pool_pairs >= SE_MIN_POOL_PAIRS) & np.isfinite(F1[:, RAW_C])
        row = {"cal_pairs": _dist(cal @ ppq), "pool_pairs": _dist(pool_pairs),
               "RAW_F1@2_pool": _dist(F1[base_ok, RAW_C])}
        for name, fitter in (("REFIT", fit_kappa_vec), ("F1REFIT", fit_f1_vec)):
            c, okf = fitter(fc, cal)
            okd = base_ok & okf
            d = F1[np.arange(B), c] - F1[:, RAW_C]
            d = d[okd & np.isfinite(d)]
            st = _dist(d)
            st.update({"n_failed": int(B - d.size),
                       "frac_catastrophic": (float((d < CATASTROPHIC_DF1).mean())
                                             if d.size else None),
                       "frac_ge0": float((d >= 0).mean()) if d.size else None})
            vals, cts = np.unique(COMBO_DIAL[c[okd]], return_counts=True)
            st["dial_freq"] = {f"L{v}": int(n) for v, n in zip(vals, cts)}
            if fallback:
                ce = c[okd & np.isfinite(F1[np.arange(B), c] - F1[:, RAW_C])]
                st["raw_fallback_rate"] = (float((ce == RAW_C).mean())
                                           if ce.size else None)
                st["n_raw_fallback"] = int((ce == RAW_C).sum())
            row[name] = st
        out["sizes"][key] = row
    out["verdict"] = {name: e6p_verdict(out["sizes"], name)
                      for name in ("REFIT", "F1REFIT")}
    return out


def e6p_verdict(sizes: dict, name: str) -> str:
    hi, lo = sizes.get(str(E6P_HI)), sizes.get(str(E6P_LO))
    if not hi or "skipped" in hi or hi[name].get("n", 0) == 0:
        return f"not applicable (q={E6P_HI} not evaluable)"
    g_hi = hi[name]["mean"]
    if not g_hi > TIE_F1:
        return "not applicable (no gain to capture)"
    if not lo or "skipped" in lo or lo[name].get("n", 0) == 0:
        return f"not applicable (q={E6P_LO} not evaluable)"
    frac = lo[name]["mean"] / g_hi
    yes = frac >= E6P_FRAC
    return (f"{'YES' if yes else 'NO'} — q={E6P_LO} captures {frac:.0%} of the "
            f"q={E6P_HI} mean gain ({lo[name]['mean']:+.4f} vs {g_hi:+.4f}; "
            f"criterion >= {E6P_FRAC:.0%})")


def load_stored_labels(data_dir: Path, ds: str):
    base = data_dir / "derived" / f"batch6_labels_{ds}"
    for suf in (".parquet", ".pkl"):
        p = base.with_suffix(suf)
        if p.exists():
            try:
                df = pd.read_parquet(p) if suf == ".parquet" else pd.read_pickle(p)
            except ImportError:
                print(f"[labels] WARNING: no parquet engine to read {p}; "
                      f"same-universe check skipped.", file=sys.stderr)
                return None, p
            return df, p
    return None, base.with_suffix(".parquet")


def check_stored(stored, model: str, cell: dict, labels: dict) -> str:
    """Same universe and labels as `run` (batch6_labels_<ds>)."""
    if stored is None:
        return "not checked (no batch6_labels file)"
    s = stored[stored.model_id == model]
    if s.empty:
        return "not checked (judge absent from batch6_labels file)"
    keys = list(zip(s.qid.astype(str), s.docid.astype(str)))
    if keys != list(cell["universe"]):
        sys.exit(f"ERROR: '{model}' @ {cell['dataset']}: pair universe differs "
                 f"from the stored batch6_labels file ({len(keys)} vs "
                 f"{len(cell['universe'])} pairs) — not the SAME universe as "
                 f"`run`.")
    bad = [c for c in ("human", "RAW", "FROZEN", "REFIT")
           if not np.array_equal(s[c].to_numpy(dtype=np.int64),
                                 cell["h"] if c == "human" else labels[c])]
    if bad:
        sys.exit(f"ERROR: '{model}' @ {cell['dataset']}: recomputed {bad} "
                 f"differ from the stored batch6_labels file.")
    return "identical (universe, human, RAW, FROZEN, REFIT)"


def followup_cell(cell: dict, rule: dict, B: int, nested_b: int, seed: int,
                  fold_seed: int, e6p_sizes, e6p_B: int, stored=None) -> dict:
    model, ds = cell["model"], cell["dataset"]
    h, cnt, P = cell["h"], cell["cnt"], cell["P"]
    nq = cell["n_queries"]
    fc = followup_cache(cell, fold_seed)
    dial_f = int(rule["dial_level"])
    mono_f = tuple(int(x) for x in rule["monotone_map"])
    labels = {"RAW": labels_from_counts(cnt[2], IDENTITY_MAP),
              "FROZEN": labels_from_counts(cnt[dial_f - 1], mono_f)}
    lab, rules_r, pair_fold, why = crossfit(cell, fold_seed)
    if lab is None:
        return {"model": model, "dataset": ds, "info": cell["info"],
                "error": f"fold seed {fold_seed}: {why}"}
    labels["REFIT"] = lab
    if not np.array_equal(pair_fold, fc["qfold"][cell["pair_q"]]):
        sys.exit(f"ERROR: '{model}' @ {ds}: query-level folds differ from the "
                 f"cross-fit's pair folds.")
    rules_r = sorted(rules_r, key=lambda r: r["test_fold"])
    want = [combo_of(r["dial_level"], r["monotone_map"]) for r in rules_r]
    ones = np.ones((1, nq))
    ck, okk, _cm = crossfit_vec(fc, ones, fit_kappa_vec)
    if not okk[0] or [int(x) for x in ck[0]] != want:
        sys.exit(f"ERROR: '{model}' @ {ds}: vectorised kappa-objective fitter "
                 f"chose {[combo_name(x) for x in ck[0]]} but batch-4 fit_rule "
                 f"chose {[combo_name(x) for x in want]}.")
    stored_note = check_stored(stored, model, cell, labels)

    # batch-4 map fitted at L3 on each train fold (supplementary MAP@L3 arm)
    row_fold = pair_fold[cell["row_u_P"]]
    m3 = [fit_monotone_map(P, np.where(row_fold != k)[0], 3)[0] for k in (0, 1)]
    cF, okF, _cmF = crossfit_vec(fc, ones, fit_f1_vec)
    cF = [int(x) for x in cF[0]]

    fam = {"FROZEN": {"dial": [dial_f, dial_f], "map": [mono_f, mono_f]},
           "REFIT": {"dial": [r["dial_level"] for r in rules_r],
                     "map": [tuple(r["monotone_map"]) for r in rules_r]}}
    arms = {}
    for f, ch in fam.items():
        d, m = ch["dial"], ch["map"]
        arms[f"{f}:RAW"] = arm_query_combos(fc, [RAW_C, RAW_C])
        arms[f"{f}:DIAL"] = arm_query_combos(
            fc, [combo_of(d[k], IDENTITY_MAP) for k in (0, 1)])
        arms[f"{f}:MAP"] = arm_query_combos(fc, [combo_of(3, m[k]) for k in (0, 1)])
        arms[f"{f}:FULL"] = arm_query_combos(fc, [combo_of(d[k], m[k])
                                                  for k in (0, 1)])
    arms["REFIT:MAP@L3"] = arm_query_combos(fc, [combo_of(3, m3[k])
                                                 for k in (0, 1)])
    arms["F1REFIT"] = arm_query_combos(fc, cF)
    idx = boot_indices(nq, B, seed)
    Wt = np.stack([np.bincount(r, minlength=nq) for r in idx]).astype(float)
    pts, reps, labs = arm_eval(fc, arms, Wt)

    # arms reproduce RAW / FROZEN / REFIT label sets exactly
    for a, ls in (("FROZEN:RAW", "RAW"), ("REFIT:RAW", "RAW"),
                  ("FROZEN:FULL", "FROZEN"), ("REFIT:FULL", "REFIT")):
        if not np.array_equal(labs[a], labels[ls]):
            sys.exit(f"ERROR: '{model}' @ {ds}: factorial arm {a} does not "
                     f"reproduce the {ls} labels.")
    # ... and experiment A's evaluation-only bootstrap, replicate for replicate
    cms = {ls: cm_by_query(h, labels[ls], cell["pair_q"], nq)
           for ls in LABEL_SETS}
    ref = bootstrap(cms, B, seed)["ci90"]
    for a, ls in (("FROZEN:RAW", "RAW"), ("FROZEN:FULL", "FROZEN"),
                  ("REFIT:FULL", "REFIT")):
        for ep in ("F1@2", "ratio@2", "P@2", "R@2", "MAE"):
            if _ci90(reps[a][ep]) != ref[ls][ep]:
                sys.exit(f"ERROR: '{model}' @ {ds}: arm {a} {ep} CI differs "
                         f"from experiment A's bootstrap of {ls}.")
    coinc = coincidence_checks(fc, fam, labs, pts, reps)

    factorial = {}
    for f, ch in fam.items():
        names = [f"{f}:{x}" for x in FX_ARMS] + (["REFIT:MAP@L3"]
                                                  if f == "REFIT" else [])
        factorial[f] = {
            "choices": {"dial": [int(x) for x in ch["dial"]],
                        "map": [list(m) for m in ch["map"]]},
            "arms": {n.split(":", 1)[1]: {
                "point": {ep: pts[n][ep] for ep in FX_EPS},
                "ci90": {ep: _ci90(reps[n][ep]) for ep in FX_EPS},
                "diff_vs_RAW": _diff(pts, reps, n, f"{f}:RAW")}
                for n in names},
            "attribution": _attrib(pts, reps, f),
        }
    factorial["REFIT"]["map_at_L3_refit"] = [list(m) for m in m3]

    # (C) F1-objective refit
    Wtr = [(np.where(fc["qfold"] != k, 1.0, 0.0))[None, :] for k in (0, 1)]
    train_f1 = {k: {"REFIT": float(grid_f1(fc, Wtr[k])[0, want[k]]),
                    "F1REFIT": float(grid_f1(fc, Wtr[k])[0, cF[k]])}
                for k in (0, 1)}
    pts["REFIT"], reps["REFIT"] = pts["REFIT:FULL"], reps["REFIT:FULL"]
    pts["RAW"], reps["RAW"] = pts["REFIT:RAW"], reps["REFIT:RAW"]
    f1ref = {
        "rules": [{"test_fold": k, "combo": combo_name(cF[k]),
                   "dial_level": COMBOS[cF[k]][0],
                   "monotone_map": list(COMBOS[cF[k]][1]),
                   "train_F1@2": train_f1[k]["F1REFIT"],
                   "train_F1@2_kappa_refit": train_f1[k]["REFIT"]}
                  for k in (0, 1)],
        "point": {ep: pts["F1REFIT"][ep] for ep in FX_EPS},
        "ci90": {ep: _ci90(reps["F1REFIT"][ep]) for ep in FX_EPS},
        "diff": {"F1REFIT-RAW": _diff(pts, reps, "F1REFIT", "RAW"),
                 "F1REFIT-REFIT": _diff(pts, reps, "F1REFIT", "REFIT"),
                 "REFIT-RAW": _diff(pts, reps, "REFIT", "RAW")},
        "nested": nested_bootstrap(fc, nested_b, seed) if nested_b else None,
        "ci_note": "diff CIs: evaluation-only query bootstrap (fit held fixed);"
                   " 'nested': fit-aware (cross-fit redone per replicate)",
    }

    # (B) operating curve
    marks = {"RAW": RAW_C, "FROZEN": combo_of(dial_f, mono_f)}
    fold_marks = {"REFIT": want, "F1REFIT": cF}
    rows, oc, versus = opcurve(fc, cell, marks, fold_marks)
    for nm, a in (("RAW", "RAW"), ("FROZEN", "FROZEN:FULL"),
                  ("REFIT", "REFIT"), ("F1REFIT", "F1REFIT"),
                  ("REFIT:DIAL", "REFIT:DIAL"), ("REFIT:MAP", "REFIT:MAP")):
        p = pts[a]
        cf, dlt, v = versus(p["F1@2"], p["ratio@2"])
        oc["points"][nm] = {"ratio@2": p["ratio@2"], "P@2": p["P@2"],
                            "R@2": p["R@2"], "F1@2": p["F1@2"],
                            "curve_F1_at_ratio": cf, "dF1_vs_curve": dlt,
                            "vs_curve": v}
        rows.append({"kind": "point", "t": np.nan,
                     "dial": "", "map": "", "tag": nm,
                     **{k: p[k] for k in ("ratio@2", "P@2", "R@2", "F1@2")},
                     "n_pos": int(round(p["rate@2"] * len(h))),
                     "curve_F1_at_ratio": cf, "dF1_vs_curve": dlt})

    return {
        "model": model, "dataset": ds, "info": cell["info"],
        "n_pairs": len(h), "n_queries": nq, "fold_seed": fold_seed,
        "rule_frozen": {"dial_level": dial_f, "monotone_map": list(mono_f),
                        "in_sample": in_sample_note(rule, ds)},
        "stored_labels_check": stored_note,
        "factorial": factorial, "coincidences": coinc,
        "f1_refit": f1ref, "opcurve": oc, "opcurve_rows": rows,
        "e6_prime": (e6_prime(fc, e6p_sizes, e6p_B, seed)
                     if e6p_sizes else None),
    }


# ───────────────────────────── printing ─────────────────────────────────────

def _pm(d) -> str:
    return f"{_f(d['point'], 7, 4, True)} {_ci(d['ci90'])}"


def print_followup(res: dict) -> None:
    print()
    print("-" * 96)
    print(f"  JUDGE: {res['model']}   dataset: {res['dataset']}   [{EXPLORATORY}]")
    if "error" in res:
        print(f"  ERROR: {res['error']}")
        return
    print(f"  pairs {res['n_pairs']}  queries {res['n_queries']}  fold seed "
          f"{res['fold_seed']}  frozen rule L{res['rule_frozen']['dial_level']}/"
          f"{''.join(map(str, res['rule_frozen']['monotone_map']))} "
          f"({res['rule_frozen']['in_sample']})")
    print(f"  same universe as `run`: {res['stored_labels_check']}")

    print("\n  (A) 2x2 FACTORIAL  dial {L3, fitted} x map {identity, fitted}  "
          "(eval-only query bootstrap, 90% CI)")
    hdr = (f"  {'family:arm':<14}{'choice':<17}{'ratio@2':>8}{'P@2':>8}{'R@2':>8}"
           f"{'F1@2':>8}{'MAE':>8}{'Δcont':>8}   dF1@2 vs RAW [90% CI]")
    for f, fx in res["factorial"].items():
        ch = fx["choices"]
        print(f"\n  {f}: dial {ch['dial']}  map "
              f"{[''.join(map(str, m)) for m in ch['map']]}  (test fold 0, 1)")
        print(hdr)
        for arm, a in fx["arms"].items():
            if arm == "RAW":
                desc = "L3/0123"
            elif arm == "DIAL":
                desc = "dial/0123"
            elif arm == "MAP":
                desc = "L3/map"
            elif arm == "FULL":
                desc = "dial/map"
            else:
                desc = "L3/map refit@L3"
            p = a["point"]
            print(f"  {f + ':' + arm:<14}{desc:<17}{_f(p['ratio@2'], 8, 3)}"
                  f"{_f(p['P@2'], 8, 3)}{_f(p['R@2'], 8, 3)}{_f(p['F1@2'], 8, 3)}"
                  f"{_f(p['MAE'], 8, 3)}{_f(p['delta_cont'], 8, 3, True)}   "
                  f"{_pm(a['diff_vs_RAW']['F1@2'])}")
        print(f"  attribution of the {f} effect (total = dial + map + "
              f"interaction):")
        for ep in ("F1@2", "absdev@2", "P@2", "R@2", "MAE", "delta_cont"):
            at = fx["attribution"][ep]
            print(f"    {ep:<10} total {_pm(at['total'])}  dial "
                  f"{_pm(at['dial'])}  map {_pm(at['map'])}  inter "
                  f"{_pm(at['interaction'])}")
    if res["coincidences"]:
        print("  coincidence checks (arm == RAW/FULL where the choice is L3 / "
              "identity): " + "; ".join(
                  f"{c['family']} {c['kind']} folds {c['folds']} OK"
                  for c in res["coincidences"]))

    oc = res["opcurve"]
    print(f"\n  (B) OPERATING CURVE  L3 vote sum v = K*mean, K={oc['K_records_mode']}"
          f" (share of pairs with K records {oc['frac_pairs_with_K']:.3f}), "
          f"t={oc['t_range'][0]}..{oc['t_range'][1]}; RAW == t="
          f"{oc['raw_equivalent_t'] or 'none (records per pair vary)'}")
    print(f"  {'t':>3}{'ratio@2':>9}{'P@2':>8}{'R@2':>8}{'F1@2':>8}")
    for c in oc["curve"]:
        print(f"  {c['t']:>3}{_f(c['ratio@2'], 9, 3)}{_f(c['P@2'], 8, 3)}"
              f"{_f(c['R@2'], 8, 3)}{_f(c['F1@2'], 8, 3)}")
    cb, gc = oc["curve_best"], oc["grid_ceiling"]
    print(f"  L3 threshold-curve best: t={cb['t']} F1@2={cb['F1@2']:.4f} at "
          f"ratio@2={cb['ratio@2']:.3f}")
    print(f"  in-sample best-of-grid F1 ceiling: {gc['combo']} F1@2="
          f"{gc['F1@2']:.4f}; F1-optimal ratio@2 = {gc['ratio@2']:.3f}  "
          f"(optimistic: chosen on the evaluation data)")
    for nm, p in oc["points"].items():
        print(f"  {nm:<11} ratio {_f(p['ratio@2'], 6, 3)}  F1 "
              f"{_f(p['F1@2'], 6, 4)}  curve@ratio {_f(p['curve_F1_at_ratio'], 6, 4)}"
              f"  d {_f(p['dF1_vs_curve'], 7, 4, True)}  -> {p['vs_curve']}"
              f"{'  (lost discrimination vs re-thresholding L3)' if p['vs_curve'] == 'below' else ''}")

    fr = res["f1_refit"]
    print("\n  (C) F1-OBJECTIVE REFIT (grid argmax of train F1@2; ties -> "
          "train kappa -> RAW-first order)")
    for r in fr["rules"]:
        print(f"    test fold {r['test_fold']}: {r['combo']}  train F1@2 "
              f"{r['train_F1@2']:.4f} (kappa-refit rule on same train fold "
              f"{r['train_F1@2_kappa_refit']:.4f})")
    p = fr["point"]
    print(f"    held-out: F1@2 {p['F1@2']:.4f}  P@2 {p['P@2']:.4f}  R@2 "
          f"{p['R@2']:.4f}  ratio@2 {p['ratio@2']:.3f}")
    nb = fr["nested"]
    for k, lbl in (("F1REFIT-RAW", "F1REFIT-RAW"), ("F1REFIT-REFIT",
                                                    "F1REFIT-REFIT(kappa)"),
                   ("REFIT-RAW", "REFIT(kappa)-RAW")):
        line = f"    dF1@2 {lbl:<21} {_pm(fr['diff'][k]['F1@2'])} eval-only"
        if nb:
            nk = {"F1REFIT-RAW": "F1REFIT-RAW F1@2",
                  "F1REFIT-REFIT": "F1REFIT-REFIT F1@2",
                  "REFIT-RAW": "REFIT-RAW F1@2"}[k]
            line += f"   {_ci(nb[nk]['ci90'])} fit-aware (nested B={nb['B']})"
        print(line)
    if nb:
        print(f"    nested replicates ok: {nb['n_ok']}")

    e6 = res["e6_prime"]
    if e6:
        pp = e6["ppq"]
        print(f"\n  (E) E6' SAMPLE EFFICIENCY in calibration QUERIES (B={e6['B']};"
              f" pairs/query mean {pp['mean']:.1f}, median {pp['median']:.0f}, "
              f"range {pp['min']:.0f}-{pp['max']:.0f})")
        print(f"    {'q':>3} {'cal pairs':>10} {'pool':>6}  objective   "
              f"{'mean dF1':>9} {'median':>8} {'P(dF1<-.05)':>12} {'P(>=0)':>7}"
              f" {'n':>5}")
        for q, row in e6["sizes"].items():
            if "skipped" in row:
                print(f"    {q:>3}  skipped: {row['skipped']}")
                continue
            for nm in ("REFIT", "F1REFIT"):
                st = row[nm]
                if not st.get("n"):
                    print(f"    {q:>3}  {nm}: no successful draws")
                    continue
                print(f"    {q:>3} {row['cal_pairs']['mean']:>10.1f} "
                      f"{row['pool_pairs']['mean']:>6.0f}  "
                      f"{'kappa' if nm == 'REFIT' else 'F1':<10}"
                      f"{st['mean']:>+9.4f} {st['median']:>+8.4f} "
                      f"{st['frac_catastrophic']:>12.3f} {st['frac_ge0']:>7.3f}"
                      f" {st['n']:>5}")
        for nm, v in e6["verdict"].items():
            print(f"    E6' ({'kappa' if nm == 'REFIT' else 'F1'}-objective): "
                  f"captures >=80% of the q={E6P_HI} gain by q={E6P_LO}? {v}")


# ───────────────────────────── outputs ──────────────────────────────────────

def _jclean(o):
    if isinstance(o, dict):
        return {str(k): _jclean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jclean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_jclean(v) for v in o.tolist()]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return fnum(o)
    return o


def write_followup(results: list[dict], ds: str, data_dir: Path,
                   meta: dict) -> None:
    der = data_dir / "derived"
    tex_dir = der / "tex"
    der.mkdir(parents=True, exist_ok=True)
    ok = [r for r in results if "error" not in r]
    for r in ok:
        p = der / f"batch6_opcurve_{ds}_{judge_slug(r['model'])}.csv"
        pd.DataFrame(r["opcurve_rows"]).to_csv(p, index=False)
        print(f"[csv] wrote {p}")
    stamp = f"[{EXPLORATORY}] "
    rows = []
    for r in ok:
        for f, fx in r["factorial"].items():
            for arm, a in fx["arms"].items():
                p, c = a["point"], a["ci90"]
                d = a["diff_vs_RAW"]["F1@2"]
                rows.append(
                    f"{esc(r['model'])} & {f} & {esc(arm)} & "
                    f"{_t(p['ratio@2'])} & {_t(p['P@2'])} & {_t(p['R@2'])} & "
                    f"{_t(p['F1@2'])} {_tci(c['F1@2'])} & {_t(p['MAE'])} & "
                    f"{_t(p['delta_cont'], sign=True)} & "
                    f"{_t(d['point'], sign=True)} {_tci(d['ci90'])} \\\\")
    write_tex(tex_dir, f"batch6_followup_{ds}_factorial.tex", rows,
              f"{stamp}Batch 6 follow-up (A) on {ds}: 2x2 factorial dial "
              f"{{L3, chosen}} x map {{identity, chosen}} for the frozen rule "
              f"and the query-disjoint 2-fold cross-fit (fold seed "
              f"{meta['fold_seed']}); MAP@L3 = map refitted at L3. Columns: "
              f"judge, family, arm, ratio@2, P@2, R@2, F1@2 [90\\% CI], MAE, "
              f"continuous Delta, dF1@2 vs RAW [90\\% CI]. Query bootstrap "
              f"B={meta['B']}, seed {meta['seed']}, evaluation-only.")
    rows = []
    for r in ok:
        for f, fx in r["factorial"].items():
            for ep in ("F1@2", "absdev@2", "MAE", "delta_cont"):
                at = fx["attribution"][ep]
                rows.append(
                    f"{esc(r['model'])} & {f} & {esc(ep)} & " + " & ".join(
                        f"{_t(at[k]['point'], sign=True)} {_tci(at[k]['ci90'])}"
                        for k in ("total", "dial", "map", "interaction"))
                    + " \\\\")
    write_tex(tex_dir, f"batch6_followup_{ds}_attrib.tex", rows,
              f"{stamp}Batch 6 follow-up (A) on {ds}: attribution of the "
              f"FROZEN / REFIT effect vs RAW into dial-only, map-only and "
              f"interaction = (FULL-DIAL)-(MAP-RAW). Columns: judge, family, "
              f"endpoint, total, dial, map, interaction [90\\% CI].")
    rows = []
    for r in ok:
        oc = r["opcurve"]
        pt = oc["points"]
        rows.append(
            f"{esc(r['model'])} & {_t(oc['curve_best']['F1@2'])} "
            f"({oc['curve_best']['t']}; {_t(oc['curve_best']['ratio@2'])}) & "
            f"{_t(oc['grid_ceiling']['F1@2'])} "
            f"({_t(oc['grid_ceiling']['ratio@2'])}) & "
            + " & ".join(f"{_t(pt[k]['dF1_vs_curve'], sign=True)}"
                         for k in ("RAW", "FROZEN", "REFIT", "F1REFIT"))
            + " \\\\")
    write_tex(tex_dir, f"batch6_followup_{ds}_opcurve.tex", rows,
              f"{stamp}Batch 6 follow-up (B) on {ds}: L3 vote-sum threshold "
              f"curve vs the in-sample (dial L1-L5) x (35 monotone maps) grid. "
              f"Columns: judge, best curve F1@2 (t; ratio@2), in-sample "
              f"best-of-grid F1@2 ceiling (its ratio@2), and F1@2 minus the "
              f"L3 curve's F1@2 at matched ratio for RAW, CORR-frozen, "
              f"CORR-refit (kappa objective), CORR-refit (F1 objective); "
              f"negative = discrimination lost vs re-thresholding L3.")
    rows = []
    for r in ok:
        fr = r["f1_refit"]
        nb = fr["nested"] or {}
        rows.append(
            f"{esc(r['model'])} & {_t(fr['point']['F1@2'])} & "
            f"{_t(fr['point']['P@2'])} & {_t(fr['point']['R@2'])} & "
            f"{_t(fr['point']['ratio@2'])} & "
            + " & ".join(
                f"{_t(fr['diff'][k]['F1@2']['point'], sign=True)} "
                f"{_tci(fr['diff'][k]['F1@2']['ci90'])}"
                + (f" {_tci(nb[nk]['ci90'])}" if nb else "")
                for k, nk in (("REFIT-RAW", "REFIT-RAW F1@2"),
                              ("F1REFIT-RAW", "F1REFIT-RAW F1@2"),
                              ("F1REFIT-REFIT", "F1REFIT-REFIT F1@2")))
            + " \\\\")
    write_tex(tex_dir, f"batch6_followup_{ds}_f1refit.tex", rows,
              f"{stamp}Batch 6 follow-up (C) on {ds}: F1-objective refit "
              f"(query-disjoint 2-fold cross-fit, grid argmax of train F1@2). "
              f"Columns: judge, held-out F1@2, P@2, R@2, ratio@2; dF1@2 "
              f"REFIT(kappa)-RAW, F1REFIT-RAW, F1REFIT-REFIT(kappa), each with "
              f"the evaluation-only 90\\% CI (B={meta['B']}) then the fit-aware "
              f"nested-bootstrap 90\\% CI (B={meta['nested_b']}).")
    rows = []
    for r in ok:
        e6 = r["e6_prime"]
        if not e6:
            continue
        for q, row in e6["sizes"].items():
            if "skipped" in row:
                continue
            for nm in ("REFIT", "F1REFIT"):
                st = row[nm]
                if not st.get("n"):
                    continue
                rows.append(
                    f"{esc(r['model'])} & {'kappa' if nm == 'REFIT' else 'F1'} &"
                    f" {q} & {row['cal_pairs']['mean']:.0f} & "
                    f"{row['pool_pairs']['mean']:.0f} & "
                    f"{_t(st['mean'], sign=True)} & {_t(st['median'], sign=True)}"
                    f" & {st['frac_catastrophic']:.3f} \\\\")
    write_tex(tex_dir, f"batch6_followup_{ds}_e6p.tex", rows,
              f"{stamp}Batch 6 follow-up (E) on {ds}: E6' sample efficiency in "
              f"calibration queries q (whole queries, simple random sample, "
              f"B={meta['e6p_B']}); refit on q queries, evaluated on the "
              f"query-disjoint extension pool. Columns: judge, objective, q, "
              f"mean calibration pairs, mean pool pairs, mean and median "
              f"dF1@2 (REFIT-RAW), fraction of draws with dF1@2 < "
              f"{CATASTROPHIC_DF1}.")
    payload = {"meta": {**meta, "dataset": ds},
               "judges": {r["model"]: {k: v for k, v in r.items()
                                       if k != "opcurve_rows"}
                          for r in results}}
    path = der / f"batch6_followup_{ds}.json"
    path.write_text(json.dumps(_jclean(payload), indent=2, ensure_ascii=False,
                               allow_nan=False), encoding="utf-8")
    print(f"[json] wrote {path}")


def followup_datasets(df, qrels, rules_all, datasets, judges, flagships,
                      data_dir, B, nested_b, seed, fold_seed, e6p_sizes,
                      e6p_B, check_labels: bool = True) -> dict:
    out: dict = {}
    frozen_pairs: dict = {}
    for ds in datasets:
        stored = None
        if check_labels:
            stored, sp = load_stored_labels(data_dir, ds)
            if stored is None:
                print(f"[labels] WARNING: {sp} not found — cannot verify the "
                      f"universe equals `run`'s.", file=sys.stderr)
        res_ds = []
        for model in judges:
            d, restricted = scope_rows(df, model, ds, flagships, data_dir,
                                       frozen_pairs)
            rule = frozen_rule(rules_all, model, ds) if not d.empty else None
            if d.empty or rule is None:
                res_ds.append({"model": model, "dataset": ds, "info": {},
                               "error": (f"no {ds} rows" if d.empty else
                                         f"no frozen {FAMILY_OF[ds]} rule")})
                continue
            cell = prepare_cell(model, ds, d, qrels, seed)
            cell["info"]["flagship"] = restricted
            if "error" in cell:
                res_ds.append({"model": model, "dataset": ds,
                               "info": cell["info"], "error": cell["error"]})
                continue
            res_ds.append(followup_cell(cell, rule, B, nested_b, seed,
                                        fold_seed, e6p_sizes, e6p_B, stored))
        out[ds] = res_ds
    return out


def cmd_followup(args) -> None:
    import time
    t0 = time.time()
    datasets = resolve_datasets(args.datasets)
    assert_not_sealed(datasets, unseal=args.unseal_dl22)     # BEFORE data
    data_dir = resolve_data_dir(args.data_dir)
    rules_all, rules_path, sha = load_rules(data_dir, args.rules)
    fold_seed = _fold_seeds(args.fold_seeds)[0]
    jpath = Path(args.judgments) if args.judgments else \
        data_dir / "derived" / "judgments.parquet"
    df = load_judgments(jpath)
    df = df[df.dataset.astype(str).isin(datasets)]
    qrels = load_human(datasets, args.qrels_json)
    judges, flagships = _judges(args.judges), _flagships(args.flagships)
    qsizes = [int(x) for x in args.q_sizes.split(",") if x.strip()]
    meta = {"script": "scripts/batch6_poolext.py followup",
            "status": EXPLORATORY, "exploratory": True,
            "pre_registered": False, "git_hash": git_hash(),
            "code_dirty": code_dirty(), "rules_path": str(rules_path),
            "rules_sha256": sha, "B": args.b, "nested_b": args.nested_b,
            "seed": args.seed, "fold_seed": fold_seed, "ci": "90% percentile",
            "e6p_B": args.e6p_b, "e6p_q_sizes": qsizes,
            "catastrophic_dF1": CATASTROPHIC_DF1, "tie_f1": TIE_F1,
            "qrels_source": args.qrels_json or "src.qrels",
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc)
            .isoformat(timespec="seconds")}
    res = followup_datasets(df, qrels, rules_all, datasets, judges, flagships,
                            data_dir, args.b, args.nested_b, args.seed,
                            fold_seed, qsizes, args.e6p_b)
    for ds in datasets:
        print()
        print("=" * 96)
        print(f"  BATCH 6 FOLLOW-UP   dataset={ds}   *** {EXPLORATORY} ***")
        print("=" * 96)
        print(f"  code git {meta['git_hash']} (dirty: {meta['code_dirty']})  "
              f"rules_sha256 {sha[:12]}  fold seed {fold_seed}  B={args.b}  "
              f"nested B={args.nested_b}  E6' B={args.e6p_b} q={qsizes}")
        print("  FRAMING: operating-point artifact removal against THESE "
              "annotators, never judgment quality.")
        for r in res[ds]:
            print_followup(r)
    if not any("error" not in r for ds in datasets for r in res[ds]):
        sys.exit("\nERROR: no (judge, dataset) cell produced a result.")
    print()
    for ds in datasets:
        write_followup(res[ds], ds, data_dir, meta)
    print(f"\n[{EXPLORATORY}] done in {time.time() - t0:.1f}s")


# ═══════════════════════════════════════════════════════════════════════════
# followup2 — EXPLORATORY: the evaluator's re-run requests on `followup`
#   (R) RETHRESH baseline: cross-fitted ONE-parameter re-threshold of the L3
#       vote sum (same 2 folds / fold seed as REFIT), t by train binary kappa
#       [RETHRESH_K] or train F1@2 [RETHRESH_F1]; paired dF1@2 vs REFIT,
#       F1REFIT, FROZEN with eval-only AND fit-aware (nested) CIs
#   (B2) FROZEN:DIAL / FROZEN:MAP (+ RETHRESH) operating-curve verdicts;
#       grid "ceiling" renamed to an optimistic in-sample reference and the
#       in-sample best L3 re-threshold printed beside it
#   (S) fold-seed robustness (--fold-seeds; default 42 only)
#   (C) --nested-b with replicates-per-tail warning
#   (E) E6' RAW-fallback rate + confounding / finite-population caveats
#   header: multiplicity / non-independence / in-sample status
# `followup` itself is untouched; followup2 re-uses followup_cell() for every
# section the two share (so those numbers are the same code path).
# ═══════════════════════════════════════════════════════════════════════════

FU2_FOLD_SEEDS = (42,)
RT_ARMS = ("RETHRESH_K", "RETHRESH_F1")
RT_OBJECTIVE = {"RETHRESH_K": "kappa", "RETHRESH_F1": "F1"}
RT_CONTRASTS = (("REFIT", "RETHRESH_K"), ("F1REFIT", "RETHRESH_F1"),
                ("FROZEN", "RETHRESH_K"), ("FROZEN", "RETHRESH_F1"))
RT_EXTRA = (("RETHRESH_K", "RAW"), ("RETHRESH_F1", "RAW"))
GRID_REF_NOTE = ("in-sample best-of-grid F1 (optimistic reference, NOT an "
                 "upper bound)")
NESTED_TAIL_MIN = 50
CURVE_ARMS2 = ("RAW", "FROZEN", "FROZEN:DIAL", "FROZEN:MAP", "REFIT",
               "REFIT:DIAL", "REFIT:MAP", "F1REFIT", "RETHRESH_K",
               "RETHRESH_F1")
SEED_CURVE_ARMS = ("REFIT", "F1REFIT", "RETHRESH_K", "RETHRESH_F1")
SEED_C_DIFFS = (("F1REFIT", "RAW"), ("F1REFIT", "REFIT"),
                ("REFIT", "RAW")) + RT_CONTRASTS
HEAVY_OVERLAP = 0.25
RT_NOTE = ("RETHRESH = query-disjoint 2-fold cross-fit (same folds as REFIT) "
           "of ONE threshold t on the L3 vote sum v = K*mean (positive iff "
           "v >= t, t = 1..3K); train objective = Cohen's kappa of the binary "
           "decision (v >= t vs human >= 2; linear == unweighted for 2 "
           "classes) [RETHRESH_K] or train F1@2 [RETHRESH_F1]; ties -> the "
           "RAW-equivalent t first, then lower t")


def rethresh_cache(fc: dict, cell: dict) -> dict:
    """Per-query TP / n_pos of every L3 vote-sum threshold (the opcurve()
    variable: positive iff vsum*K >= t*n_records, t = 1..3K)."""
    h, cnt3 = cell["h"], cell["cnt"][2]
    nrec = cnt3.sum(axis=1)
    vsum = cnt3 @ np.arange(4)
    vals, freq = np.unique(nrec, return_counts=True)
    K = int(vals[np.argmax(freq)])
    ts = np.arange(1, 3 * K + 1, dtype=np.int64)
    yp = vsum[None, :] * K >= ts[:, None] * nrec[None, :]            # [T, nP]
    yt = h >= 2
    pq, nq = fc["pair_q"], fc["nq"]
    tp = np.stack([np.bincount(pq, weights=(y & yt).astype(float),
                               minlength=nq) for y in yp])           # [T, nq]
    lr = np.stack([np.bincount(pq, weights=y.astype(float), minlength=nq)
                   for y in yp])
    raw_bin = fc["labs"][RAW_C] >= 2
    raw_t = [int(t) for t, y in zip(ts, yp) if np.array_equal(y, raw_bin)]
    t_raw = raw_t[0] if raw_t else int(np.ceil(1.5 * K))
    pref = np.array([t_raw] + [int(t) for t in ts if t != t_raw],
                    dtype=np.int64) - 1
    return {"K": K, "ts": ts, "tp": tp, "lr": lr, "raw_t": raw_t,
            "t_raw": t_raw, "t_raw_exact": bool(raw_t), "pref": pref,
            "yp": yp}


def _bin_eps(tp, lr, hr) -> dict:
    tp, lr, hr = (np.asarray(x, dtype=float) for x in (tp, lr, hr))
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"F1@2": 2.0 * tp / (lr + hr), "P@2": tp / lr,
                "R@2": tp / hr, "ratio@2": lr / hr}


def _bin_kappa(TP, LR, HR, N) -> np.ndarray:
    """Cohen's kappa of a 2x2 table from (TP, n_pred_pos, n_true_pos, N)."""
    TP, LR, HR, N = (np.asarray(x, dtype=float) for x in (TP, LR, HR, N))
    with np.errstate(divide="ignore", invalid="ignore"):
        po = (N - LR - HR + 2.0 * TP) / N
        pe = (LR * HR + (N - LR) * (N - HR)) / (N * N)
        k = (po - pe) / (1.0 - pe)
    return np.where((N > 0) & (pe != 1.0) & np.isfinite(k), k, np.nan)


def rt_scores(rc: dict, fc: dict, Wt, objective: str) -> np.ndarray:
    """Train score of every threshold t on query weights Wt -> [B, T]."""
    Wt = np.atleast_2d(np.asarray(Wt, dtype=float))
    TP, LR = Wt @ rc["tp"].T, Wt @ rc["lr"].T
    HR, N = (Wt @ fc["hr"])[:, None], (Wt @ fc["ppq"])[:, None]
    if objective == "kappa":
        return _bin_kappa(TP, LR, HR, N)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(LR + HR > 0, 2.0 * TP / (LR + HR), np.nan)


def rt_pick(rc: dict, fc: dict, Wt, objective: str):
    """argmax of the train objective; ties (and an all-undefined objective)
    -> RAW-equivalent t first, then lower t.  Returns (t index [B], ok [B])."""
    Wt = np.atleast_2d(np.asarray(Wt, dtype=float))
    S = rt_scores(rc, fc, Wt, objective)
    S = np.where(np.isfinite(S), S, -np.inf)[:, rc["pref"]]
    return rc["pref"][np.argmax(S, axis=1)], Wt.sum(axis=1) > 0


def rt_crossfit(rc: dict, fc: dict, qfold, Wq, objective: str):
    """2-fold query-disjoint cross-fit of t (fold of a query fixed, as in
    crossfit_vec).  Returns (t index [B,2], ok [B], test sums tp/lr/hr [B])."""
    Wq = np.atleast_2d(np.asarray(Wq, dtype=float))
    B = Wq.shape[0]
    tsel = np.full((B, 2), rc["t_raw"] - 1, dtype=np.int64)
    ok = np.ones(B, dtype=bool)
    TP, LR, HR = np.zeros(B), np.zeros(B), np.zeros(B)
    ar = np.arange(B)
    for k in (0, 1):
        infold = qfold == k
        tr, te = Wq * (~infold), Wq * infold
        j, okk = rt_pick(rc, fc, tr, objective)
        ok &= okk & (te.sum(axis=1) > 0)
        j = np.where(ok, j, rc["t_raw"] - 1)
        tsel[:, k] = j
        TP += (te @ rc["tp"].T)[ar, j]
        LR += (te @ rc["lr"].T)[ar, j]
        HR += te @ fc["hr"]
    return tsel, ok, {"tp": TP, "lr": LR, "hr": HR}


def _sign3(x, tie: float) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return "+" if x > tie else "-" if x < -tie else "0"


def _beats(x) -> str:
    s = _sign3(x, TIE_F1)
    return {"+": "beats", "-": "below", "0": "within"}.get(s, "n/a")


def nested_tail_note(B: int) -> dict:
    per_tail = 0.05 * B
    return {"B": B, "replicates_per_tail": per_tail,
            "warn": bool(B and per_tail < NESTED_TAIL_MIN)}


def seed_robustness(cell: dict, fc: dict, rc: dict, cZ: int, fold_seeds,
                    versus, primary: dict) -> dict:
    """Point estimates of the REFIT 2x2 attribution, curve verdicts and the
    C-section / RETHRESH paired dF1@2 at every fold seed (no nested CIs).
    The primary-seed row is hard-checked against ``primary``."""
    ds, nq = cell["dataset"], fc["nq"]
    ones = np.ones((1, nq))
    per: dict = {}
    for s in fold_seeds:
        fcs = dict(fc)
        fcs["qfold"] = np.array([fold_of(ds, q, s) for q in fc["qname"]],
                                dtype=np.int64)
        lab, rules_r, _pf, why = crossfit(cell, s)
        if lab is None:
            per[str(s)] = {"error": why}
            continue
        want = [combo_of(r["dial_level"], r["monotone_map"])
                for r in sorted(rules_r, key=lambda r: r["test_fold"])]
        ck, okk, _cm = crossfit_vec(fcs, ones, fit_kappa_vec)
        if not okk[0] or [int(x) for x in ck[0]] != want:
            sys.exit(f"ERROR: '{cell['model']}' @ {ds} fold seed {s}: "
                     f"vectorised kappa fitter chose "
                     f"{[combo_name(x) for x in ck[0]]} but batch-4 fit_rule "
                     f"chose {[combo_name(x) for x in want]}.")
        cF, _okF, _cmF = crossfit_vec(fcs, ones, fit_f1_vec)
        cF = [int(x) for x in cF[0]]
        arms = {"RAW": [RAW_C, RAW_C],
                "DIAL": [combo_of(COMBOS[c][0], IDENTITY_MAP) for c in want],
                "MAP": [combo_of(3, COMBOS[c][1]) for c in want],
                "FULL": want, "F1REFIT": cF, "FROZEN": [cZ, cZ]}
        pts, _r, _l = arm_eval(fcs, {nm: arm_query_combos(fcs, cc)
                                     for nm, cc in arms.items()}, ones)
        att = {}
        for ep in ("F1@2", "absdev@2"):
            p = {x: pts[x][ep] for x in ("RAW", "DIAL", "MAP", "FULL")}
            att[ep] = {"total": p["FULL"] - p["RAW"], "dial": p["DIAL"] - p["RAW"],
                       "map": p["MAP"] - p["RAW"],
                       "interaction": (p["FULL"] - p["DIAL"])
                       - (p["MAP"] - p["RAW"])}
        f1 = {"RAW": pts["RAW"]["F1@2"], "FROZEN": pts["FROZEN"]["F1@2"],
              "REFIT": pts["FULL"]["F1@2"], "F1REFIT": pts["F1REFIT"]["F1@2"]}
        ratio = {"REFIT": pts["FULL"]["ratio@2"],
                 "F1REFIT": pts["F1REFIT"]["ratio@2"]}
        rts = {}
        for nm in RT_ARMS:
            tsel, okr, sm = rt_crossfit(rc, fc, fcs["qfold"], ones,
                                        RT_OBJECTIVE[nm])
            e = _bin_eps(sm["tp"], sm["lr"], sm["hr"])
            f1[nm] = float(e["F1@2"][0]) if okr[0] else np.nan
            ratio[nm] = float(e["ratio@2"][0]) if okr[0] else np.nan
            rts[nm] = [int(rc["ts"][j]) for j in tsel[0]]
        curve = {}
        for nm in SEED_CURVE_ARMS:
            _cf, dlt, v = versus(f1[nm], ratio[nm])
            curve[nm] = {"dF1_vs_curve": dlt, "vs_curve": v}
        per[str(s)] = {
            "REFIT_rules": [combo_name(c) for c in want],
            "F1REFIT_rules": [combo_name(c) for c in cF],
            "RETHRESH_t": rts,
            "attribution": att, "curve": curve,
            "dF1@2": {f"{a}-{b}": f1[a] - f1[b] for a, b in SEED_C_DIFFS},
        }
    # primary-seed row == the primary analysis (same code path, bit-exact)
    p0 = per.get(str(fold_seeds[0]), {})
    if "error" not in p0:
        bad = []
        for ep in ("F1@2", "absdev@2"):
            for k in ("total", "dial", "map", "interaction"):
                if not _same(p0["attribution"][ep][k],
                             primary["attribution"][ep][k]):
                    bad.append(f"attr {ep} {k}")
        for k, v in primary["dF1@2"].items():
            if not _same(p0["dF1@2"][k], v):
                bad.append(f"dF1 {k}")
        for nm in SEED_CURVE_ARMS:
            if p0["curve"][nm]["vs_curve"] != primary["curve"][nm]:
                bad.append(f"curve {nm}")
        if bad:
            sys.exit(f"ERROR: '{cell['model']}' @ {ds}: fold-seed robustness "
                     f"row for the primary seed differs from the primary "
                     f"analysis: {bad[:6]}")
    # per-metric table + sign flips relative to the primary seed
    ok_seeds = [str(s) for s in fold_seeds if "error" not in per[str(s)]]
    table = {}

    def add(key, vals, tie):
        if not vals:
            return
        s0 = _sign3(vals[0], tie)
        sg = [_sign3(v, tie) for v in vals]
        table[key] = {"values": vals, "signs": sg,
                      "flips": sum(x != s0 for x in sg[1:]),
                      "hard_flips": sum({x, s0} == {"+", "-"} for x in sg[1:]),
                      "tie": tie}
    for ep, tie in (("F1@2", TIE_F1), ("absdev@2", TIE_RATIO)):
        for k in ("total", "dial", "map", "interaction"):
            add(f"REFIT attr {ep} {k}",
                [per[s]["attribution"][ep][k] for s in ok_seeds], tie)
    for a, b in SEED_C_DIFFS:
        add(f"dF1@2 {a}-{b}", [per[s]["dF1@2"][f"{a}-{b}"] for s in ok_seeds],
            TIE_F1)
    for nm in SEED_CURVE_ARMS:
        vs = [per[s]["curve"][nm]["vs_curve"] for s in ok_seeds]
        if vs:
            table[f"curve {nm}"] = {
                "values": [per[s]["curve"][nm]["dF1_vs_curve"] for s in ok_seeds],
                "signs": vs, "flips": sum(v != vs[0] for v in vs[1:]),
                "hard_flips": sum({v, vs[0]} == {"above", "below"}
                                  for v in vs[1:]), "tie": TIE_F1}
    return {"seeds": [int(s) for s in fold_seeds], "ok_seeds": ok_seeds,
            "per_seed": per, "table": table,
            "note": "point estimates only; the nested (fit-aware) bootstrap is "
                    "run for the primary seed only; FROZEN is seed-invariant"}


def followup2_cell(cell: dict, rule: dict, B: int, nested_b: int, seed: int,
                   fold_seeds, e6p_sizes, e6p_B: int, stored=None) -> dict:
    fs0 = int(fold_seeds[0])
    # every section shared with `followup` comes from followup_cell itself
    # (nested bootstrap and E6' are re-run below with their followup2 extras)
    res = followup_cell(cell, rule, B, 0, seed, fs0, None, e6p_B, stored)
    if "error" in res:
        return res
    model, ds = cell["model"], cell["dataset"]
    nq = cell["n_queries"]
    ar = np.arange(nq)
    fc = followup_cache(cell, fs0)
    rc = rethresh_cache(fc, cell)
    oc = res["opcurve"]
    if rc["K"] != oc["K_records_mode"] or rc["raw_t"] != oc["raw_equivalent_t"]:
        sys.exit(f"ERROR: '{model}' @ {ds}: RETHRESH curve variable differs "
                 f"from opcurve() (K {rc['K']} vs {oc['K_records_mode']}, RAW t "
                 f"{rc['raw_t']} vs {oc['raw_equivalent_t']}).")
    fx = res["factorial"]
    cZ = combo_of(rule["dial_level"], rule["monotone_map"])
    want = [combo_of(fx["REFIT"]["choices"]["dial"][k],
                     fx["REFIT"]["choices"]["map"][k]) for k in (0, 1)]
    cF = [combo_of(r["dial_level"], r["monotone_map"])
          for r in sorted(res["f1_refit"]["rules"], key=lambda r: r["test_fold"])]
    res["rule_frozen"]["fitted_on"] = list(rule["datasets"])
    res["rule_frozen"]["n_pairs_fit"] = rule.get("n_pairs")

    # ── (R) RETHRESH: full-data cross-fit, per-query statistics ───────────
    q_tp, q_lr = {}, {}
    for nm, cc in (("RAW", [RAW_C, RAW_C]), ("FROZEN", [cZ, cZ]),
                   ("REFIT", want), ("F1REFIT", cF)):
        cq = arm_query_combos(fc, cc)
        q_tp[nm], q_lr[nm] = fc["tp"][cq, ar], fc["lr"][cq, ar]
    ones = np.ones((1, nq))
    rt = {}
    for nm in RT_ARMS:
        obj = RT_OBJECTIVE[nm]
        tsel, okr, _sm = rt_crossfit(rc, fc, fc["qfold"], ones, obj)
        if not okr[0]:
            sys.exit(f"ERROR: '{model}' @ {ds}: RETHRESH cross-fit failed on "
                     f"the full data although REFIT succeeded.")
        j = np.where(fc["qfold"] == 0, tsel[0, 0], tsel[0, 1])
        q_tp[nm], q_lr[nm] = rc["tp"][j, ar], rc["lr"][j, ar]
        folds = []
        for k in (0, 1):
            tr = (fc["qfold"] != k).astype(float)[None, :]
            S = rt_scores(rc, fc, tr, obj)[0]
            folds.append({"test_fold": k, "t": int(rc["ts"][tsel[0, k]]),
                          f"train_{obj}": fnum(S[tsel[0, k]]),
                          f"train_{obj}_at_RAW_t": fnum(S[rc["t_raw"] - 1])})
        rt[nm] = {"objective": obj, "folds": folds}
    hrq = fc["hr"]
    pts = {nm: {ep: float(v) for ep, v in
                _bin_eps(q_tp[nm].sum(), q_lr[nm].sum(), hrq.sum()).items()}
           for nm in q_tp}
    for nm, src in (("RAW", fx["REFIT"]["arms"]["RAW"]["point"]),
                    ("FROZEN", fx["FROZEN"]["arms"]["FULL"]["point"]),
                    ("REFIT", fx["REFIT"]["arms"]["FULL"]["point"]),
                    ("F1REFIT", res["f1_refit"]["point"])):
        for ep in ("F1@2", "P@2", "R@2", "ratio@2"):
            if not _same(pts[nm][ep], src[ep]):
                sys.exit(f"ERROR: '{model}' @ {ds}: followup2 {nm} {ep} "
                         f"{pts[nm][ep]} differs from followup's {src[ep]}.")
    idx = boot_indices(nq, B, seed)
    Wt = np.stack([np.bincount(r, minlength=nq) for r in idx]).astype(float)
    HRb = Wt @ hrq
    reps = {nm: _bin_eps(Wt @ q_tp[nm], Wt @ q_lr[nm], HRb) for nm in q_tp}
    if _ci90(reps["REFIT"]["F1@2"] - reps["RAW"]["F1@2"]) != \
            res["f1_refit"]["diff"]["REFIT-RAW"]["F1@2"]["ci90"]:
        sys.exit(f"ERROR: '{model}' @ {ds}: followup2 eval-only bootstrap "
                 f"differs from followup's (REFIT-RAW F1@2).")
    for nm in RT_ARMS:
        rt[nm]["point"] = pts[nm]
        rt[nm]["ci90"] = {ep: _ci90(reps[nm][ep]) for ep in pts[nm]}

    # ── (C) nested (fit-aware) bootstrap: followup's + RETHRESH, paired ───
    nb, nested = None, {}
    if nested_b:
        nb = nested_bootstrap(fc, nested_b, seed, return_reps=True)
        nr = nb.pop("reps")
        Wq = nr["Wq"]
        HRn = Wq @ hrq
        f1n = {"RAW": nr["RAW"],
               "REFIT": np.where(nr["ok"]["REFIT"], nr["REFIT"], np.nan),
               "F1REFIT": np.where(nr["ok"]["F1REFIT"], nr["F1REFIT"], np.nan),
               "FROZEN": _bin_eps(Wq @ q_tp["FROZEN"], Wq @ q_lr["FROZEN"],
                                  HRn)["F1@2"]}
        tfreq = {}
        for nm in RT_ARMS:
            tsel, okr, sm = rt_crossfit(rc, fc, fc["qfold"], Wq,
                                        RT_OBJECTIVE[nm])
            f1n[nm] = np.where(okr, _bin_eps(sm["tp"], sm["lr"],
                                             sm["hr"])["F1@2"], np.nan)
            v, c = np.unique(rc["ts"][tsel[okr]].ravel(), return_counts=True)
            tfreq[nm] = {f"t{int(a)}": int(b) for a, b in zip(v, c)}
        if _ci90(f1n["REFIT"] - f1n["RAW"]) != nb["REFIT-RAW F1@2"]["ci90"]:
            sys.exit(f"ERROR: '{model}' @ {ds}: nested replicates do not "
                     f"reproduce nested_bootstrap's REFIT-RAW CI.")
        for a, b in RT_CONTRASTS + RT_EXTRA:
            x = f1n[a] - f1n[b]
            nested[f"{a}-{b}"] = {"ci90": _ci90(x),
                                  "mean": fnum(np.nanmean(x))
                                  if np.isfinite(x).any() else None}
        nested["t_freq_fold_fits"] = tfreq
    res["f1_refit"]["nested"] = nb
    tail = nested_tail_note(nested_b)
    contrasts = {}
    for a, b in RT_CONTRASTS + RT_EXTRA:
        k = f"{a}-{b}"
        pnt = pts[a]["F1@2"] - pts[b]["F1@2"]
        contrasts[k] = {"point": pnt,
                        "ci90": _ci90(reps[a]["F1@2"] - reps[b]["F1@2"]),
                        "nested_ci90": nested.get(k, {}).get("ci90"),
                        "nested_mean": nested.get(k, {}).get("mean"),
                        "verdict": _beats(pnt)}
    res["rethresh"] = {
        "note": RT_NOTE, "K": rc["K"], "t_range": [1, 3 * rc["K"]],
        "t_raw": rc["t_raw"], "t_raw_exact": rc["t_raw_exact"],
        "arms": rt, "contrasts": contrasts,
        "nested": {"B": nested_b, "seed": seed, "tail": tail,
                   "t_freq_fold_fits": nested.get("t_freq_fold_fits")},
        "ci_note": "ci90: evaluation-only query bootstrap (fits held fixed); "
                   "nested_ci90: fit-aware (REFIT / F1REFIT / RETHRESH all "
                   "re-fitted per replicate; FROZEN and RAW need no fit)",
    }

    # ── (B2) operating-curve verdicts incl. FROZEN:DIAL / FROZEN:MAP ──────
    marks = {"RAW": RAW_C, "FROZEN": cZ}
    _rows, _oc, versus = opcurve(fc, cell, marks, {"REFIT": want,
                                                   "F1REFIT": cF})
    src = {"RAW": pts["RAW"], "FROZEN": pts["FROZEN"],
           "FROZEN:DIAL": fx["FROZEN"]["arms"]["DIAL"]["point"],
           "FROZEN:MAP": fx["FROZEN"]["arms"]["MAP"]["point"],
           "REFIT": pts["REFIT"],
           "REFIT:DIAL": fx["REFIT"]["arms"]["DIAL"]["point"],
           "REFIT:MAP": fx["REFIT"]["arms"]["MAP"]["point"],
           "F1REFIT": pts["F1REFIT"], "RETHRESH_K": pts["RETHRESH_K"],
           "RETHRESH_F1": pts["RETHRESH_F1"]}
    points2 = {}
    for nm in CURVE_ARMS2:
        p = src[nm]
        cf, dlt, v = versus(p["F1@2"], p["ratio@2"])
        points2[nm] = {"ratio@2": p["ratio@2"], "F1@2": p["F1@2"],
                       "curve_F1_at_ratio": cf, "dF1_vs_curve": dlt,
                       "vs_curve": v}
        if nm in oc["points"] and oc["points"][nm]["vs_curve"] != v:
            sys.exit(f"ERROR: '{model}' @ {ds}: curve verdict for {nm} differs "
                     f"from followup's.")
    l3 = oc["curve_best"]
    gref = dict(oc.pop("grid_ceiling"), note=GRID_REF_NOTE)
    oc["grid_best_insample"] = gref
    res["opcurve2"] = {
        "points": points2, "l3_best_insample": l3, "grid_best_insample": gref,
        "grid_minus_l3_best": gref["F1@2"] - l3["F1@2"],
        "grid_below_l3_best": bool(gref["F1@2"] < l3["F1@2"] - 1e-12),
        "grid_exceeded_by": [nm for nm in CURVE_ARMS2
                             if points2[nm]["F1@2"] > gref["F1@2"] + 1e-12],
        "above_l3_best_by_tie": [nm for nm in CURVE_ARMS2 if points2[nm]["F1@2"]
                                 > l3["F1@2"] + TIE_F1],
        "note": "the 175-combo grid does not contain every L3 re-threshold and "
                "is chosen on the evaluation data; cross-fit arms (a mixture of "
                "per-fold rules) can exceed it",
    }

    # ── (S) fold-seed robustness ──────────────────────────────────────────
    primary = {"attribution": {ep: {k: fx["REFIT"]["attribution"][ep][k]["point"]
                                    for k in ("total", "dial", "map",
                                              "interaction")}
                               for ep in ("F1@2", "absdev@2")},
               "dF1@2": {f"{a}-{b}": pts[a]["F1@2"] - pts[b]["F1@2"]
                         for a, b in SEED_C_DIFFS},
               "curve": {nm: points2[nm]["vs_curve"] for nm in SEED_CURVE_ARMS}}
    res["fold_seed_robustness"] = seed_robustness(cell, fc, rc, cZ, fold_seeds,
                                                  versus, primary)

    # ── (E) E6' with the RAW-fallback rate ────────────────────────────────
    if e6p_sizes:
        e6 = e6_prime(fc, e6p_sizes, e6p_B, seed, fallback=True)
        e6["caveats"] = {
            "n_queries": nq, "pairs_per_query_mean": e6["ppq"].get("mean"),
            "q_over_n_queries": {str(q): q / nq for q in e6p_sizes},
            "heavy_overlap_q": [int(q) for q in e6p_sizes
                                if q / nq >= HEAVY_OVERLAP
                                and q <= nq - SE_MIN_POOL_QUERIES],
        }
        res["e6_prime"] = e6
    return res


# ───────────────────────────── printing ─────────────────────────────────────

def _grid_lines(o2: dict) -> list[str]:
    l3, g = o2["l3_best_insample"], o2["grid_best_insample"]
    out = [f"  in-sample best L3 re-threshold (one parameter, chosen on the "
           f"evaluation data): t={l3['t']} F1@2={l3['F1@2']:.4f} at "
           f"ratio@2={l3['ratio@2']:.3f}",
           f"  {GRID_REF_NOTE}: {g['combo']} F1@2={g['F1@2']:.4f} at "
           f"ratio@2={g['ratio@2']:.3f}"]
    if o2["grid_below_l3_best"]:
        out.append(f"  ** FLAG: the grid reference is BELOW the in-sample best "
                   f"L3 re-threshold by {-o2['grid_minus_l3_best']:.4f} (the "
                   f"grid does not contain every L3 threshold) **")
    if o2["grid_exceeded_by"]:
        out.append(f"  ** FLAG: exceeded by {', '.join(o2['grid_exceeded_by'])}"
                   f" (cross-fit mixtures of per-fold rules) — not a ceiling **")
    return out


def print_followup2(res: dict) -> None:
    import contextlib
    import io
    if "error" in res:
        print_followup(res)
        return
    # the shared sections, printed by followup's own printer; only the old
    # "ceiling" line is replaced (hard-fail if it cannot be found exactly once)
    view = {**res, "opcurve": {**res["opcurve"],
                               "grid_ceiling": res["opcurve"]["grid_best_insample"]}}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_followup(view)
    lines = buf.getvalue().rstrip("\n").split("\n")
    o2 = res["opcurve2"]
    tail = res["rethresh"]["nested"]["tail"]
    tail_line = (f"    nested B={tail['B']}: {tail['replicates_per_tail']:.0f} "
                 f"replicates per 5% tail"
                 + (f"  ** WARNING: < {NESTED_TAIL_MIN} per tail — percentile "
                    f"bounds are Monte-Carlo noisy; re-run borderline cells "
                    f"with --nested-b 1000 **" if tail["warn"] else ""))
    out = []
    n_grid = n_best = n_nest = 0
    for ln in lines:
        if ln.startswith("  in-sample best-of-grid F1 ceiling:"):
            out += _grid_lines(o2)
            n_grid += 1
        elif ln.startswith("  L3 threshold-curve best:"):
            n_best += 1               # re-printed by _grid_lines
        elif ln.startswith("    nested replicates ok:"):
            out += [ln, tail_line]
            n_nest += 1
        else:
            out.append(ln)
    if (n_grid, n_best, n_nest) != (1, 1, int(bool(res["f1_refit"]["nested"]))):
        sys.exit("ERROR: followup2 printer could not locate followup's grid / "
                 "curve-best / nested lines exactly once.")
    print("\n".join(out))

    print("\n  (B2) OPERATING-CURVE VERDICTS at matched ratio@2 (all arms; "
          f"band TIE_F1={TIE_F1}, a fixed band, not a test)")
    for nm, p in o2["points"].items():
        print(f"    {nm:<12} ratio {_f(p['ratio@2'], 6, 3)}  F1 "
              f"{_f(p['F1@2'], 6, 4)}  curve@ratio "
              f"{_f(p['curve_F1_at_ratio'], 6, 4)}  d "
              f"{_f(p['dF1_vs_curve'], 7, 4, True)}  -> {p['vs_curve']}")
    if o2["above_l3_best_by_tie"]:
        print(f"    arms above the in-sample best L3 re-threshold by > TIE_F1: "
              f"{', '.join(o2['above_l3_best_by_tie'])}")

    rr = res["rethresh"]
    print(f"\n  (R) RETHRESH BASELINE  K={rr['K']}, t={rr['t_range'][0]}.."
          f"{rr['t_range'][1]}, RAW-equivalent t={rr['t_raw']}"
          f"{'' if rr['t_raw_exact'] else ' (approx.: no t reproduces RAW exactly)'}")
    print(f"    {RT_NOTE}")
    for nm, a in rr["arms"].items():
        p, c = a["point"], a["ci90"]
        fl = "; ".join(f"fold {f['test_fold']}: t={f['t']} (train "
                       f"{a['objective']} {_f(f['train_' + a['objective']], 7, 4)}"
                       f" vs RAW t {_f(f['train_' + a['objective'] + '_at_RAW_t'], 7, 4)})"
                       for f in a["folds"])
        print(f"    {nm:<12} {fl}")
        print(f"    {'':<12} held-out F1@2 {p['F1@2']:.4f} {_ci(c['F1@2'])}  "
              f"P@2 {p['P@2']:.4f}  R@2 {p['R@2']:.4f}  ratio@2 "
              f"{p['ratio@2']:.3f}")
    for k, c in rr["contrasts"].items():
        line = (f"    dF1@2 {k:<24} {_f(c['point'], 7, 4, True)} "
                f"{_ci(c['ci90'])} eval-only")
        if c["nested_ci90"] is not None:
            line += f"   {_ci(c['nested_ci90'])} fit-aware (nested B={tail['B']})"
        if k in {f"{a}-{b}" for a, b in RT_CONTRASTS}:
            line += f"   -> {c['verdict']}"
        print(line)
    tf = rr["nested"]["t_freq_fold_fits"]
    if tf:
        print("    nested fold-fit t frequencies: " + "; ".join(
            f"{nm} {d}" for nm, d in tf.items()))

    sr = res["fold_seed_robustness"]
    if len(sr["seeds"]) > 1:
        print(f"\n  (S) FOLD-SEED ROBUSTNESS seeds {sr['seeds']} (point "
              f"estimates; nested CIs only for the primary seed "
              f"{sr['seeds'][0]}; FROZEN is seed-invariant)")
        errs = {s: v["error"] for s, v in sr["per_seed"].items() if "error" in v}
        if errs:
            print(f"    seeds skipped: {errs}")
        hdr = "".join(f"{'s' + s:>9}" for s in sr["ok_seeds"])
        print(f"    {'metric':<34}{hdr}   flips (band / hard)")
        for key, t in sr["table"].items():
            if key.startswith("curve"):
                vals = "".join(f"{v:>9}" for v in t["signs"])
            else:
                vals = "".join(_f(v, 9, 4, True) for v in t["values"])
            print(f"    {key:<34}{vals}   {t['flips']} / {t['hard_flips']}")
        for s in sr["ok_seeds"]:
            ps = sr["per_seed"][s]
            print(f"    seed {s}: REFIT {ps['REFIT_rules']}  F1REFIT "
                  f"{ps['F1REFIT_rules']}  RETHRESH_K t={ps['RETHRESH_t']['RETHRESH_K']}"
                  f"  RETHRESH_F1 t={ps['RETHRESH_t']['RETHRESH_F1']}")
    else:
        print(f"\n  (S) fold-seed robustness: primary seed {sr['seeds'][0]} only "
              f"(pass --fold-seeds 42,43,44,45,46 for the robustness table)")

    e6 = res.get("e6_prime")
    if e6:
        cv = e6["caveats"]
        print(f"\n  (E+) E6' RAW-FALLBACK RATE (share of evaluated draws whose "
              f"fitted rule is RAW = L3/identity); {cv['n_queries']} queries, "
              f"{cv['pairs_per_query_mean']:.1f} pairs/query")
        print(f"    {'q':>3} {'q/n_q':>6}  objective   {'mean dF1':>9} "
              f"{'median':>8} {'P(cat)':>7} {'RAW-fallback':>13} {'n':>5}")
        for q, row in e6["sizes"].items():
            if "skipped" in row:
                print(f"    {q:>3}  skipped: {row['skipped']}")
                continue
            for nm in ("REFIT", "F1REFIT"):
                st = row[nm]
                if not st.get("n"):
                    print(f"    {q:>3}  {nm}: no successful draws")
                    continue
                print(f"    {q:>3} {cv['q_over_n_queries'][q]:>6.2f}  "
                      f"{'kappa' if nm == 'REFIT' else 'F1':<10}"
                      f"{st['mean']:>+9.4f} {st['median']:>+8.4f} "
                      f"{st['frac_catastrophic']:>7.3f} "
                      f"{st['raw_fallback_rate']:>13.3f} {st['n']:>5}")
        if cv["heavy_overlap_q"]:
            print(f"    finite population: q in {cv['heavy_overlap_q']} samples "
                  f">= {HEAVY_OVERLAP:.0%} of the {cv['n_queries']} queries — "
                  f"calibration draws overlap heavily, so their spread (and "
                  f"P(cat)) is understated")


def rethresh_tally(results) -> dict:
    ok = [r for r in results if "error" not in r]
    out = {}
    for a, b in RT_CONTRASTS:
        k = f"{a}-{b}"
        v = [r["rethresh"]["contrasts"][k]["verdict"] for r in ok]
        out[k] = {"beats": v.count("beats"), "within": v.count("within"),
                  "below": v.count("below"), "n": len(v)}
    return out


def print_rethresh_tally(tally: dict, label: str) -> None:
    def part(k):
        t = tally[k]
        a, b = k.split("-")
        return (f"{a} vs {b}: beats {t['beats']} / within {t['within']} / "
                f"below {t['below']}")
    n = tally[f"{RT_CONTRASTS[0][0]}-{RT_CONTRASTS[0][1]}"]["n"]
    print(f"  RETHRESH tally {label} ({n} cells; dF1@2 point estimate, band "
          f"TIE_F1={TIE_F1}; descriptive, no test):")
    print("    kappa-objective: " + part("REFIT-RETHRESH_K") + ";  "
          + part("FROZEN-RETHRESH_K"))
    print("    F1-objective:    " + part("F1REFIT-RETHRESH_F1") + ";  "
          + part("FROZEN-RETHRESH_F1"))


def _vendor(model: str) -> str:
    return model.split("/", 1)[0] if "/" in model else model


def followup2_header(res: dict, datasets, meta: dict) -> dict:
    """Run-level caveats, read from the results (judges, datasets, rules)."""
    ok = [r for ds in datasets for r in res[ds] if "error" not in r]
    judges = sorted({r["model"] for r in ok})
    fam: dict = {}
    for j in judges:
        fam.setdefault(_vendor(j), []).append(j)
    same = {k: v for k, v in fam.items() if len(v) > 1}
    return {"n_cells": len(ok), "n_judges": len(judges),
            "datasets": [ds for ds in datasets
                         if any("error" not in r for r in res[ds])],
            "same_vendor_judges": same,
            "multiplicity": "no multiplicity correction; cells share queries "
                            "and human labels within a dataset and same-vendor "
                            "judges are not independent; every count/tally is "
                            "descriptive"}


def print_followup2_header(h: dict, meta: dict) -> None:
    print()
    print("=" * 96)
    print(f"  BATCH 6 FOLLOW-UP 2   *** {EXPLORATORY} ***")
    print("=" * 96)
    print(f"  code git {meta['git_hash']} (dirty: {meta['code_dirty']})  "
          f"rules_sha256 {meta['rules_sha256'][:12]}  fold seeds "
          f"{meta['fold_seeds']} (primary {meta['fold_seeds'][0]})  B={meta['B']}"
          f"  nested B={meta['nested_b']}  E6' B={meta['e6p_B']} "
          f"q={meta['e6p_q_sizes']}")
    print("  DESIGN: designed AFTER unboxing at the evaluator's request; "
          "'held-out' for dl22 means only that the frozen rule was never "
          "fitted there.")
    same = "; ".join(f"{k}: {len(v)} judges" for k, v in
                     h["same_vendor_judges"].items()) or "none"
    print(f"  MULTIPLICITY / NON-INDEPENDENCE: {h['n_cells']} cells "
          f"({h['n_judges']} judges x {len(h['datasets'])} datasets "
          f"{h['datasets']}); many contrasts per cell; same-vendor (same-"
          f"family) judges: {same}.")
    print(f"    {h['multiplicity']}.")
    nt = nested_tail_note(meta["nested_b"])
    if meta["nested_b"]:
        print(f"  NESTED BOOTSTRAP: B={nt['B']} -> {nt['replicates_per_tail']:.0f}"
              f" replicates per 5% tail"
              + (f"  ** WARNING: < {NESTED_TAIL_MIN}; use --nested-b 1000 for "
                 f"borderline cells **" if nt["warn"] else ""))
    print("  FRAMING: operating-point artifact removal against THESE "
          "annotators, never judgment quality.")


def print_dataset_header2(ds: str, results: list[dict], meta: dict) -> None:
    ok = [r for r in results if "error" not in r]
    print()
    print("=" * 96)
    print(f"  FOLLOW-UP 2   dataset={ds}   *** {EXPLORATORY} ***")
    print("=" * 96)
    notes: dict = {}
    for r in ok:
        notes.setdefault(r["rule_frozen"]["in_sample"], []).append(r["model"])
    for n, js in notes.items():
        print(f"  IN-SAMPLE STATUS (frozen rule, from correction_rules.json "
              f"'datasets'): {n}" + ("" if len(notes) == 1 else
                                     f"  [{', '.join(js)}]"))
    print("  REFIT / F1REFIT / RETHRESH are query-disjoint 2-fold cross-fits "
          "on this dataset (held-out within the dataset).")
    fl = [r for r in ok if r["info"].get("flagship")]
    for r in fl:
        print(f"  SUBSAMPLE: {r['model']} is restricted to inputs/pairs_{ds}_"
              f"flagship40.jsonl — its frozen rule was fitted on the flagship "
              f"subsample (rule n_pairs={r['rule_frozen'].get('n_pairs_fit') or 'n/a'}, "
              f"fitted on {'+'.join(r['rule_frozen'].get('fitted_on', []))}) and "
              f"is evaluated on {r['n_pairs']} pairs / {r['n_queries']} queries "
              f"({r['n_pairs'] / r['n_queries']:.1f} pairs/query)")
    e6 = [r for r in ok if r.get("e6_prime")]
    if e6:
        ppq = ", ".join(f"{judge_slug(r['model'])} "
                        f"{r['e6_prime']['caveats']['pairs_per_query_mean']:.1f}"
                        for r in e6)
        print(f"  E6' CAVEATS: q (calibration QUERIES) is confounded with "
              f"pairs-per-query (mean pairs/query: {ppq}).")
        for nq in sorted({r["n_queries"] for r in e6}):
            fr = ", ".join(f"q={q}: {q / nq:.2f}"
                           + (" HEAVY" if q / nq >= HEAVY_OVERLAP else "")
                           for q in meta["e6p_q_sizes"]
                           if q <= nq - SE_MIN_POOL_QUERIES)
            print(f"    finite population ({nq} queries): q/n_queries = "
                  f"expected overlap of two draws — {fr}")


# ───────────────────────────── outputs ──────────────────────────────────────

def write_followup2(results: list[dict], ds: str, data_dir: Path,
                    meta: dict, header: dict) -> None:
    der = data_dir / "derived"
    tex_dir = der / "tex"
    der.mkdir(parents=True, exist_ok=True)
    ok = [r for r in results if "error" not in r]
    stamp = f"[{EXPLORATORY}] "

    def nci(c):
        return "" if c is None else f" {_tci(c)}"
    rows = []
    for r in ok:
        rr = r["rethresh"]
        cells = []
        for nm in RT_ARMS:
            p = rr["arms"][nm]["point"]
            ts = ",".join(str(f["t"]) for f in rr["arms"][nm]["folds"])
            cells.append(f"{_t(p['F1@2'])} {_tci(rr['arms'][nm]['ci90']['F1@2'])}"
                         f" & {_t(p['P@2'])} & {_t(p['R@2'])} & "
                         f"{_t(p['ratio@2'])} ({ts})")
        for a, b in RT_CONTRASTS:
            c = rr["contrasts"][f"{a}-{b}"]
            cells.append(f"{_t(c['point'], sign=True)} {_tci(c['ci90'])}"
                         f"{nci(c['nested_ci90'])}")
        rows.append(f"{esc(r['model'])} & " + " & ".join(cells) + " \\\\")
    write_tex(tex_dir, f"batch6_followup2_{ds}_rethresh.tex", rows,
              f"{stamp}Batch 6 follow-up 2 (R) on {ds}: RETHRESH baseline = "
              f"query-disjoint 2-fold cross-fit (fold seed "
              f"{meta['fold_seeds'][0]}) of one threshold t on the L3 vote sum,"
              f" t chosen by train binary kappa (RETHRESH\\_K) or train F1@2 "
              f"(RETHRESH\\_F1). Columns: judge; for RETHRESH\\_K then "
              f"RETHRESH\\_F1: held-out F1@2 [90\\% CI], P@2, R@2, ratio@2 "
              f"(t per test fold); dF1@2 REFIT-RETHRESH\\_K, F1REFIT-"
              f"RETHRESH\\_F1, FROZEN-RETHRESH\\_K, FROZEN-RETHRESH\\_F1, each "
              f"with the evaluation-only 90\\% CI (B={meta['B']}) then the "
              f"fit-aware nested 90\\% CI (B={meta['nested_b']}).")
    rows = []
    dag = "$^\\dagger$"
    for r in ok:
        o2 = r["opcurve2"]
        rows.append(
            f"{esc(r['model'])} & {_t(o2['l3_best_insample']['F1@2'])} "
            f"({o2['l3_best_insample']['t']}) & "
            f"{_t(o2['grid_best_insample']['F1@2'])}"
            f"{dag if o2['grid_below_l3_best'] else ''} & "
            + " & ".join(f"{_t(o2['points'][k]['dF1_vs_curve'], sign=True)}"
                         for k in CURVE_ARMS2) + " \\\\")
    write_tex(tex_dir, f"batch6_followup2_{ds}_opcurve.tex", rows,
              f"{stamp}Batch 6 follow-up 2 (B2) on {ds}: F1@2 minus the L3 "
              f"vote-sum curve's F1@2 at matched ratio@2. Columns: judge, "
              f"in-sample best L3 re-threshold F1@2 (t), {GRID_REF_NOTE} "
              f"($\\dagger$ = below the L3 best), then "
              f"{', '.join(esc(k) for k in CURVE_ARMS2)}.")
    rows = []
    for r in ok:
        sr = r["fold_seed_robustness"]
        for key, t in sr["table"].items():
            vals = (" & ".join(t["signs"]) if key.startswith("curve") else
                    " & ".join(_t(v, sign=True) for v in t["values"]))
            rows.append(f"{esc(r['model'])} & {esc(key)} & {vals} & "
                        f"{t['flips']} & {t['hard_flips']} \\\\")
    write_tex(tex_dir, f"batch6_followup2_{ds}_seeds.tex", rows,
              f"{stamp}Batch 6 follow-up 2 (S) on {ds}: point estimates by "
              f"fold seed {meta['fold_seeds']} (first = primary). Columns: "
              f"judge, metric, one value per seed, flips vs the primary seed "
              f"(tie band TIE\\_F1 / TIE\\_RATIO), hard flips (+ vs -).")
    rows = []
    for r in ok:
        e6 = r.get("e6_prime")
        if not e6:
            continue
        for q, row in e6["sizes"].items():
            if "skipped" in row:
                continue
            for nm in ("REFIT", "F1REFIT"):
                st = row[nm]
                if not st.get("n"):
                    continue
                rows.append(
                    f"{esc(r['model'])} & {'kappa' if nm == 'REFIT' else 'F1'} &"
                    f" {q} & {e6['caveats']['q_over_n_queries'][q]:.2f} & "
                    f"{_t(st['mean'], sign=True)} & {_t(st['median'], sign=True)}"
                    f" & {st['frac_catastrophic']:.3f} & "
                    f"{st['raw_fallback_rate']:.3f} \\\\")
    write_tex(tex_dir, f"batch6_followup2_{ds}_e6p.tex", rows,
              f"{stamp}Batch 6 follow-up 2 (E) on {ds}: E6' with the RAW-"
              f"fallback rate. Columns: judge, objective, q, q/n\\_queries "
              f"(expected overlap of two calibration draws), mean and median "
              f"dF1@2, fraction < {CATASTROPHIC_DF1}, share of draws whose "
              f"fitted rule is RAW. q is confounded with pairs per query.")
    payload = {"meta": {**meta, "dataset": ds, "header": header},
               "rethresh_tally": rethresh_tally(results),
               "judges": {r["model"]: {k: v for k, v in r.items()
                                       if k != "opcurve_rows"}
                          for r in results}}
    path = der / f"batch6_followup2_{ds}.json"
    path.write_text(json.dumps(_jclean(payload), indent=2, ensure_ascii=False,
                               allow_nan=False), encoding="utf-8")
    print(f"[json] wrote {path}")


def followup2_datasets(df, qrels, rules_all, datasets, judges, flagships,
                       data_dir, B, nested_b, seed, fold_seeds, e6p_sizes,
                       e6p_B, check_labels: bool = True) -> dict:
    out: dict = {}
    frozen_pairs: dict = {}
    for ds in datasets:
        stored = None
        if check_labels:
            stored, sp = load_stored_labels(data_dir, ds)
            if stored is None:
                print(f"[labels] WARNING: {sp} not found — cannot verify the "
                      f"universe equals `run`'s.", file=sys.stderr)
        res_ds = []
        for model in judges:
            d, restricted = scope_rows(df, model, ds, flagships, data_dir,
                                       frozen_pairs)
            rule = frozen_rule(rules_all, model, ds) if not d.empty else None
            if d.empty or rule is None:
                res_ds.append({"model": model, "dataset": ds, "info": {},
                               "error": (f"no {ds} rows" if d.empty else
                                         f"no frozen {FAMILY_OF[ds]} rule")})
                continue
            cell = prepare_cell(model, ds, d, qrels, seed)
            cell["info"]["flagship"] = restricted
            if "error" in cell:
                res_ds.append({"model": model, "dataset": ds,
                               "info": cell["info"], "error": cell["error"]})
                continue
            res_ds.append(followup2_cell(cell, rule, B, nested_b, seed,
                                         fold_seeds, e6p_sizes, e6p_B, stored))
        out[ds] = res_ds
    return out


def followup2_gate(datasets, unseal: bool, path: Path | None = None) -> None:
    """dl22: --unseal-dl22 (src.qrels seal) AND the B6 registration heading in
    paper/PREDICTIONS.md.  Checked before any data is loaded."""
    assert_not_sealed(datasets, unseal=unseal)
    if "dl22" in datasets and not has_registration(path):
        sys.exit("ERROR (dl22 gate): followup2 on dl22 requires the B6 "
                 f"registration heading ('{B6_MARKER}') in "
                 "paper/PREDICTIONS.md.")


def cmd_followup2(args) -> None:
    import time
    t0 = time.time()
    datasets = resolve_datasets(args.datasets)
    followup2_gate(datasets, args.unseal_dl22)               # BEFORE data
    data_dir = resolve_data_dir(args.data_dir)
    rules_all, rules_path, sha = load_rules(data_dir, args.rules)
    fold_seeds = _fold_seeds(args.fold_seeds)
    if args.nested_b < 0:
        sys.exit("ERROR: --nested-b must be >= 0.")
    jpath = Path(args.judgments) if args.judgments else \
        data_dir / "derived" / "judgments.parquet"
    df = load_judgments(jpath)
    df = df[df.dataset.astype(str).isin(datasets)]
    qrels = load_human(datasets, args.qrels_json)
    judges, flagships = _judges(args.judges), _flagships(args.flagships)
    qsizes = [int(x) for x in args.q_sizes.split(",") if x.strip()]
    meta = {"script": "scripts/batch6_poolext.py followup2",
            "status": EXPLORATORY, "exploratory": True,
            "pre_registered": False, "git_hash": git_hash(),
            "code_dirty": code_dirty(), "rules_path": str(rules_path),
            "rules_sha256": sha, "B": args.b, "nested_b": args.nested_b,
            "nested_tail": nested_tail_note(args.nested_b),
            "seed": args.seed, "fold_seeds": fold_seeds,
            "fold_seed": fold_seeds[0], "ci": "90% percentile",
            "e6p_B": args.e6p_b, "e6p_q_sizes": qsizes,
            "catastrophic_dF1": CATASTROPHIC_DF1, "tie_f1": TIE_F1,
            "tie_ratio": TIE_RATIO, "rethresh_note": RT_NOTE,
            "grid_note": GRID_REF_NOTE,
            "qrels_source": args.qrels_json or "src.qrels",
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc)
            .isoformat(timespec="seconds")}
    res = followup2_datasets(df, qrels, rules_all, datasets, judges, flagships,
                             data_dir, args.b, args.nested_b, args.seed,
                             fold_seeds, qsizes, args.e6p_b)
    report_followup2(res, datasets, meta)
    if not any("error" not in r for ds in datasets for r in res[ds]):
        sys.exit("\nERROR: no (judge, dataset) cell produced a result.")
    print()
    header = followup2_header(res, datasets, meta)
    for ds in datasets:
        write_followup2(res[ds], ds, data_dir, meta, header)
    print(f"\n[{EXPLORATORY}] done in {time.time() - t0:.1f}s")


def report_followup2(res: dict, datasets, meta: dict) -> None:
    header = followup2_header(res, datasets, meta)
    print_followup2_header(header, meta)
    for ds in datasets:
        print_dataset_header2(ds, res[ds], meta)
        for r in res[ds]:
            print_followup2(r)
        if any("error" not in r for r in res[ds]):
            print()
            print_rethresh_tally(rethresh_tally(res[ds]), f"[{ds}]")
    allr = [r for ds in datasets for r in res[ds]]
    if len(datasets) > 1 and any("error" not in r for r in allr):
        print()
        print_rethresh_tally(rethresh_tally(allr), "[all datasets]")
    flips = [(r["dataset"], r["model"], k, t["flips"], t["hard_flips"])
             for r in allr if "error" not in r
             for k, t in r["fold_seed_robustness"]["table"].items()]
    if len(meta["fold_seeds"]) > 1 and flips:
        print(f"  fold-seed flips (seeds {meta['fold_seeds']}): "
              f"{sum(1 for f in flips if f[3])} of {len(flips)} cell-metrics "
              f"change sign/verdict vs seed {meta['fold_seeds'][0]} on some "
              f"seed; {sum(1 for f in flips if f[4])} flip hard (+ vs -).")


# ═══════════════════════════════════════════════════════════════════════════
# selftest — synthetic stub, offline
# ═══════════════════════════════════════════════════════════════════════════

# judge -> dataset -> {level: offset added to the human grade}; clipped to 0..3
STUB_JUDGES = {
    # MS MARCO-like lenient judge; L1 removes the leniency.
    "stub/lenient": {"dl21": {1: 0, 2: 1, 3: 1, 4: 1, 5: 1}},
    # ANTIQUE-like strict judge; L5 removes the strictness.
    "stub/strict": {"antique": {1: -1, 2: -1, 3: -1, 4: -1, 5: 0}},
    # Well-calibrated judge with identity@L3 frozen rule on both families.
    "stub/noop": {"dl21": {l: 0 for l in LEVELS},
                  "antique": {l: 0 for l in LEVELS}},
    # Lenient judge whose frozen rule was fitted on a different distribution.
    "stub/drift": {"dl21": {l: 1 for l in LEVELS}},
    # Flagship-like: restricted to its frozen pairs file.
    "stub/flag": {"dl21": {1: 0, 2: 1, 3: 1, 4: 1, 5: 1}},
    # identity msmarco rule, NON-identity antique rule.
    "stub/mixed": {"dl21": {l: 0 for l in LEVELS},
                   "antique": {1: -1, 2: -1, 3: -1, 4: -1, 5: -1}},
}
# (dial, map, datasets, delta_hat); delta_hat=None -> filled from the stub
# data for rules fitted on a single dataset (so the S1 check is exercised).
STUB_RULES = {
    "stub/lenient": {"msmarco": (3, [0, 0, 1, 2], ["dl19", "dl20", "dl21"], 0.9)},
    "stub/strict": {"antique": (3, [1, 2, 3, 3], ["antique"], None)},
    "stub/noop": {"msmarco": (3, [0, 1, 2, 3], ["dl19", "dl20", "dl21"], 0.01),
                  "antique": (3, [0, 1, 2, 3], ["antique"], None)},
    "stub/drift": {"msmarco": (3, [0, 0, 2, 3], ["dl19", "dl20", "dl21"], 0.9)},
    "stub/flag": {"msmarco": (1, [0, 1, 2, 3], ["dl19", "dl20", "dl21"], 0.9)},
    "stub/mixed": {"msmarco": (3, [0, 1, 2, 3], ["dl19", "dl20", "dl21"], 0.0),
                   "antique": (3, [1, 2, 3, 3], ["antique"], None)},
}
STUB_SHAPE = {"dl21": (40, 25, [0.45, 0.30, 0.15, 0.10]),
              "antique": (30, 20, [0.10, 0.25, 0.35, 0.30]),
              "dl22": (20, 20, [0.50, 0.30, 0.15, 0.05])}


def build_stub(seed: int = 7):
    rng = np.random.default_rng(seed)
    qrels, rows = {}, []
    for ds, (nq, nd, p) in STUB_SHAPE.items():
        for qi in range(nq):
            for di in range(nd):
                qrels[(ds, f"{100 + qi}", f"{ds}-d{qi}-{di}")] = \
                    int(rng.choice(4, p=p))
    for model, per_ds in STUB_JUDGES.items():
        for ds, offs in per_ds.items():
            for (dds, q, doc), g in qrels.items():
                if dds != ds:
                    continue
                for l in LEVELS:
                    for para in "abc":
                        for run in (1, 2):
                            s = min(3, max(0, g + offs[l]))
                            u = rng.random()
                            if u < 0.08:
                                s = min(3, max(0, s + rng.choice([-1, 1])))
                            fail = u > 0.995    # ~0.5% parse failures
                            # one pair loses EVERY record at L4 -> must
                            # leave the universe (usable-at-all-levels rule)
                            if (model == "stub/lenient" and q == "101"
                                    and doc.endswith("-d1-0") and l == 4):
                                fail = True
                            rows.append({
                                "model_id": model,
                                "prompt_id": f"L{l}_{para}",
                                "politeness_level": l, "dataset": ds,
                                "qid": q, "docid": doc, "run": run,
                                "score": (np.nan if fail else float(s)),
                                "parse_ok": not fail,
                            })
    # one judged pair the stub judges never collected
    qrels[("dl21", "100", "dl21-extra")] = 2
    df = pd.DataFrame(rows)
    rules = {"git_hash": "stub", "timestamp_utc": "stub", "rules": {}}
    for m, fams in STUB_RULES.items():
        for fam, (dl, mp, dss, dh) in fams.items():
            if dh is None:                    # fitted on one dataset: compute
                ds = dss[0]
                d = df[(df.model_id == m) & (df.dataset == ds)]
                P = pack(d, qrels, m, ds, SEED)
                dh = float(delta_D(P, np.arange(len(P.h)))[0])
            rules["rules"].setdefault(m, {})[fam] = {
                "dial_level": dl, "monotone_map": mp, "datasets": dss,
                "delta_hat": dh}
    flag_pairs = sorted({(q, doc) for (ds, q, doc) in qrels
                         if ds == "dl21" and int(q) % 5 < 2})
    return df, qrels, rules, flag_pairs


def write_stub_dir(root: Path, seed: int = 7) -> None:
    df, qrels, rules, flag_pairs = build_stub(seed)
    (root / "derived").mkdir(parents=True, exist_ok=True)
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    df.to_pickle(root / "derived" / "judgments.stub.pkl")
    (root / "derived" / "correction_rules.json").write_text(
        json.dumps(rules, indent=2))
    (root / "qrels_stub.json").write_text(
        json.dumps([[a, b, c, g] for (a, b, c), g in qrels.items()]))
    with open(root / "inputs" / "pairs_dl21_flagship40.jsonl", "w") as f:
        for q, doc in flag_pairs:
            f.write(json.dumps({"qid": q, "docid": doc}) + "\n")


def _fill_stub_registration(lines: list[str]) -> str:
    """Fill a stub PASTE block: E1 '>1' except strict domains '<1'; rest 是."""
    out, cur_strict = [], False
    for ln in lines:
        if ln.startswith("### "):
            cur_strict = "严域" in ln
        if "预测：____" in ln and _ELINE.match(ln):
            E = _ELINE.match(ln).group(1)
            tok = ("<1" if cur_strict else ">1") if E == "E1" else "是"
            ln = ln.replace("预测：____", f"预测：{tok}")
        out.append(ln.replace("____", "2026-10-11"))
    return "\n".join(out)


def _noisy_stub(qrels: dict, seed: int = 11) -> pd.DataFrame:
    """Lenient dl21 judge with heavy record noise and ONE run (K=3 records at
    L3, as in production) — exercises near-ties in the fitters, a non-trivial
    nested bootstrap and the K=3 vote-sum curve."""
    rng = np.random.default_rng(seed)
    offs = {1: 0, 2: 1, 3: 1, 4: 1, 5: 1}
    rows = []
    for (ds, q, doc), g in qrels.items():
        if ds != "dl21" or doc == "dl21-extra":
            continue
        for l in LEVELS:
            for para in "abc":
                s = min(3, max(0, g + offs[l]))
                if rng.random() < 0.40:
                    s = min(3, max(0, s + int(rng.choice([-1, 1]))))
                fail = rng.random() < 0.01
                rows.append({"model_id": "stub/noisy", "prompt_id": f"L{l}_{para}",
                             "politeness_level": l, "dataset": ds, "qid": q,
                             "docid": doc, "run": 1,
                             "score": np.nan if fail else float(s),
                             "parse_ok": not fail})
    return pd.DataFrame(rows)


def selftest_followup(check, df, qrels, rules, root: Path, R: dict) -> None:
    import contextlib
    import io
    import shutil
    import time
    print("\n  -- follow-up subcommand (EXPLORATORY) --")
    t0 = time.time()
    R = {k: v for k, v in R.items() if "error" not in v}
    out = root / "out"
    shutil.copytree(root / "inputs", out / "inputs", dirs_exist_ok=True)
    df2 = pd.concat([df, _noisy_stub(qrels)], ignore_index=True)
    rules2 = json.loads(json.dumps(rules))
    rules2["rules"]["stub/noisy"] = {"msmarco": {
        "dial_level": 3, "monotone_map": [0, 0, 1, 2],
        "datasets": ["dl19", "dl20", "dl21"], "delta_hat": 0.9}}
    judges = list(STUB_JUDGES) + ["stub/noisy"]
    flag = frozenset({"stub/flag"})
    fu = followup_datasets(df2, qrels, rules2, ["dl21", "antique"], judges,
                           flag, out, 200, 60, SEED, FOLD_SEEDS[0], E6P_SIZES,
                           150)
    F = {(r["model"], r["dataset"]): r for ds in fu for r in fu[ds]
         if "error" not in r}
    print(f"  ({len(F)} follow-up cells in {time.time() - t0:.1f}s; vectorised "
          f"kappa fitter == batch-4 fit_rule on every primary cross-fit, arms "
          f"== RAW/FROZEN/REFIT labels and experiment-A bootstrap CIs — "
          f"asserted inside followup_cell)")
    check("follow-up: every stub cell analysed (incl. noisy K=3 judge)",
          len(F) == len(R) + 1 and ("stub/noisy", "dl21") in F, f"{len(F)}")
    check("follow-up: same universe/labels as `run` (stored labels file)",
          all(F[k]["stored_labels_check"].startswith("identical") for k in R),
          F[("stub/noisy", "dl21")]["stored_labels_check"])

    # (A) factorial reproduces experiment A and the coincidence identities
    bad = []
    for k, r in R.items():
        fx = F[k]["factorial"]
        for fam, arm, ls in (("FROZEN", "RAW", "RAW"), ("FROZEN", "FULL", "FROZEN"),
                             ("REFIT", "FULL", "REFIT")):
            for ep in ("F1@2", "ratio@2", "P@2", "R@2", "MAE", "absdev@2"):
                if not _same(fx[fam]["arms"][arm]["point"][ep], r["points"][ls][ep]):
                    bad.append((k, fam, arm, ep))
            if not _same(fx[fam]["arms"][arm]["point"]["delta_cont"],
                         r["delta_cont"][ls]):
                bad.append((k, fam, arm, "delta_cont"))
    check("factorial: (L3,id) / frozen / fitted arms == RAW / FROZEN / REFIT of "
          "experiment A bit-exactly (points + continuous Delta)", not bad,
          str(bad[:3]))
    kinds = {(c["family"], c["kind"]) for r in F.values()
             for c in r["coincidences"] if len(c["folds"]) == 2}
    check("factorial: coincidence identities exercised for dial==L3 and "
          "map==identity (both folds) and all held",
          {("FROZEN", "dial==L3"), ("FROZEN", "map==identity"),
           ("REFIT", "map==identity")} <= kinds, str(sorted(kinds)))
    fx = F[("stub/lenient", "dl21")]["factorial"]["FROZEN"]
    at = fx["attribution"]["F1@2"]
    check("factorial: lenient frozen L3/0012 -> DIAL==RAW, MAP==FULL, dial and "
          "interaction exactly 0",
          fx["arms"]["DIAL"]["point"] == fx["arms"]["RAW"]["point"]
          and fx["arms"]["MAP"]["point"] == fx["arms"]["FULL"]["point"]
          and at["dial"]["point"] == 0.0 and at["interaction"]["point"] == 0.0
          and at["map"]["point"] == at["total"]["point"] > 0.2,
          f"total {at['total']['point']:+.4f}")
    fx = F[("stub/flag", "dl21")]["factorial"]["FROZEN"]
    check("factorial: flagship frozen L1/identity -> MAP==RAW, DIAL==FULL",
          fx["arms"]["MAP"]["point"] == fx["arms"]["RAW"]["point"]
          and fx["arms"]["DIAL"]["point"] == fx["arms"]["FULL"]["point"])
    rn = F[("stub/noisy", "dl21")]
    ch = rn["factorial"]["REFIT"]["choices"]
    cell_n = prepare_cell("stub/noisy", "dl21",
                          scope_rows(df2, "stub/noisy", "dl21", flag, out,
                                     {})[0], qrels, SEED)
    fc_n = followup_cache(cell_n, FOLD_SEEDS[0])
    arms = {f"X:{a}": arm_query_combos(fc_n, [combo_of(d, m) for d, m in cc])
            for a, cc in (("RAW", [(3, IDENTITY_MAP)] * 2),
                          ("DIAL", [(3, IDENTITY_MAP)] * 2),   # doctored
                          ("MAP", [(3, IDENTITY_MAP)] * 2),
                          ("FULL", [(1, (0, 1, 2, 3))] * 2))}
    Wt1 = np.ones((3, fc_n["nq"]))
    p_, r_, l_ = arm_eval(fc_n, arms, Wt1)
    try:
        coincidence_checks(fc_n, {"X": {"dial": [3, 3], "map": [(0, 1, 2, 3)] * 2}},
                           l_, p_, r_)
        check("factorial: coincidence check hard-fails on a doctored arm", False)
    except SystemExit:
        check("factorial: coincidence check hard-fails on a doctored arm", True)

    # vectorised kappa fitter == batch-4 fit_rule on explicitly resampled rows
    worst = []
    for key in (("stub/noisy", "dl21"), ("stub/strict", "antique")):
        m_, ds_ = key
        cell = prepare_cell(m_, ds_, scope_rows(df2, m_, ds_, flag, out, {})[0],
                            qrels, SEED)
        fc = followup_cache(cell, FOLD_SEEDS[0])
        q_rows = [np.where(fc["row_q"] == q)[0] for q in range(fc["nq"])]
        idx = boot_indices(fc["nq"], 12, SEED + 5)
        for b in range(12):
            w = np.bincount(idx[b], minlength=fc["nq"])
            for k in (0, 1):
                wk = w * (fc["qfold"] != k)
                rows = np.concatenate([np.repeat(q_rows[q], wk[q])
                                       for q in range(fc["nq"]) if wk[q]] or
                                      [np.array([], dtype=np.int64)])
                r = fit_rule(cell["P"], rows)
                cv, okv = fit_kappa_vec(fc, wk[None, :])
                want = (None if r is None else
                        combo_of(r["dial_level"], r["monotone_map"]))
                got = int(cv[0]) if okv[0] else None
                if want != got:
                    worst.append((key, b, k, want, got))
    check("kappa fitter (vectorised, query weights) == batch-4 fit_rule on "
          "explicitly duplicated rows (2 cells x 12 replicates x 2 folds)",
          not worst, str(worst[:3]))

    # (B) operating curve
    mono_bad, p_bad = [], []
    for k, r in F.items():
        cv = r["opcurve"]["curve"]
        for a, b in zip(cv, cv[1:]):
            for ep in ("ratio@2", "R@2"):
                if not b[ep] <= a[ep] + 1e-12:
                    mono_bad.append((k, ep, a["t"]))
            if (np.isfinite(a["P@2"]) and np.isfinite(b["P@2"])
                    and not b["P@2"] >= a["P@2"] - 1e-12):
                p_bad.append((k, a["t"], a["P@2"], b["P@2"]))
    check("opcurve: ratio@2 and R@2 non-increasing in t (invariant, all cells)",
          not mono_bad, str(mono_bad[:3]))
    check("opcurve: P@2 non-decreasing in t (monotone stub judges)", not p_bad,
          str(p_bad[:3]))
    oc_n = rn["opcurve"]
    check("opcurve: K=6 stub cells reproduce RAW at t=9; noisy judge K=3",
          all(9 in F[k]["opcurve"]["raw_equivalent_t"]
              for k in F if k[0] != "stub/noisy")
          and oc_n["K_records_mode"] == 3 and oc_n["t_range"] == [1, 9],
          f"noisy RAW-equivalent t: {oc_n['raw_equivalent_t']}")
    check("opcurve: in-sample grid ceiling >= RAW and FROZEN F1@2; RAW on the "
          "L3 curve",
          all(F[k]["opcurve"]["grid_ceiling"]["F1@2"] + 1e-12
              >= max(F[k]["opcurve"]["points"]["RAW"]["F1@2"],
                     F[k]["opcurve"]["points"]["FROZEN"]["F1@2"])
              and abs(F[k]["opcurve"]["points"]["RAW"]["dF1_vs_curve"]) < 1e-12
              for k in F if k[0] != "stub/noisy"))
    print("  noisy judge points vs L3 curve: " + "; ".join(
        f"{nm} {p['vs_curve']} ({p['dF1_vs_curve']:+.4f})"
        for nm, p in oc_n["points"].items()))

    # (C) F1-objective refit
    bad = [(k, x["test_fold"], x["train_F1@2"], x["train_F1@2_kappa_refit"])
           for k, r in F.items() for x in r["f1_refit"]["rules"]
           if not x["train_F1@2"] >= x["train_F1@2_kappa_refit"]]
    check("F1-objective refit >= kappa-objective refit in train-fold F1@2 "
          "(every cell, both folds)", not bad, str(bad[:3]))
    check("F1-objective refit strictly better on some train fold (noisy judge)",
          any(x["train_F1@2"] > x["train_F1@2_kappa_refit"]
              for x in rn["f1_refit"]["rules"]),
          str([(x["combo"], round(x["train_F1@2"], 4),
                round(x["train_F1@2_kappa_refit"], 4))
               for x in rn["f1_refit"]["rules"]]))
    bad = []
    for k, r in F.items():
        fr = r["f1_refit"]
        for dk, nk in (("REFIT-RAW", "REFIT-RAW F1@2"),
                       ("F1REFIT-RAW", "F1REFIT-RAW F1@2")):
            v = fr["diff"][dk]["F1@2"]["point"]
            lo, hi, _n = fr["nested"][nk]["ci90"]
            if lo is None or not lo - 1e-12 <= v <= hi + 1e-12:
                bad.append((k, dk, v, lo, hi))
    nb = rn["f1_refit"]["nested"]
    check("nested (fit-aware) bootstrap 90% CI contains the point estimate "
          "(all cells, both objectives)", not bad,
          f"noisy REFIT-RAW {_ci(nb['REFIT-RAW F1@2']['ci90'])} vs eval-only "
          f"{_ci(rn['f1_refit']['diff']['REFIT-RAW']['F1@2']['ci90'])}; "
          f"violations {bad[:2]}")

    # (E) E6'
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_followup(F[("stub/noop", "dl21")])
    txt = buf.getvalue()
    check("E6': no-gain stub (noop) prints 'not applicable (no gain to capture)'",
          txt.count("not applicable (no gain to capture)") == 2,
          str(F[("stub/noop", "dl21")]["e6_prime"]["verdict"]))
    e6 = F[("stub/lenient", "dl21")]["e6_prime"]
    check("E6': lenient stub states a verdict; antique (30 q) q=32 skipped",
          all(v.startswith(("YES", "NO")) for v in e6["verdict"].values())
          and "skipped" in F[("stub/strict", "antique")]["e6_prime"]["sizes"]["32"]
          and F[("stub/strict", "antique")]["e6_prime"]["verdict"]["REFIT"]
          .startswith("not applicable (q=32"), str(e6["verdict"]))
    fake = {"8": {"REFIT": {"n": 5, "mean": 0.01}},
            "32": {"REFIT": {"n": 5, "mean": TIE_F1}}}
    fake2 = {"8": {"REFIT": {"n": 5, "mean": 0.07}},
             "32": {"REFIT": {"n": 5, "mean": 0.10}}}
    check("E6' verdict rule: q=32 gain <= TIE_F1 -> not applicable; 70% -> NO",
          e6p_verdict(fake, "REFIT") == "not applicable (no gain to capture)"
          and e6p_verdict(fake2, "REFIT").startswith("NO"))
    sz = e6["sizes"]["8"]
    check("E6': whole queries (cal pairs == q x pairs/query), catastrophic "
          "fraction reported",
          abs(sz["cal_pairs"]["mean"] - 8 * e6["ppq"]["mean"]) < 1.0
          and sz["REFIT"]["frac_catastrophic"] is not None
          and sz["F1REFIT"]["n"] > 0, f"cal pairs {sz['cal_pairs']['mean']:.1f}")

    # writers + gate
    meta = {"status": EXPLORATORY, "exploratory": True, "pre_registered": False,
            "B": 200, "nested_b": 60, "seed": SEED, "fold_seed": 42,
            "e6p_B": 150}
    with contextlib.redirect_stdout(io.StringIO()):
        write_followup(fu["dl21"], "dl21", out, meta)
    js = json.loads((out / "derived" / "batch6_followup_dl21.json").read_text())
    texs = sorted((out / "derived" / "tex").glob("batch6_followup_dl21_*.tex"))
    csv = pd.read_csv(out / "derived" / "batch6_opcurve_dl21_stub_noisy.csv")
    check("follow-up writers: JSON (EXPLORATORY meta), 5 tex tables, opcurve CSV",
          js["meta"]["status"] == EXPLORATORY and js["meta"]["pre_registered"]
          is False and len(texs) == 5
          and all(t.read_text().startswith(f"% [{EXPLORATORY}]") for t in texs)
          and len(csv) == (3 * 3 + 2) + len(COMBOS) + 6
          and csv["tag"].fillna("").str.contains("FROZEN").any(),
          f"{len(csv)} csv rows")
    ns = argparse.Namespace(datasets="dl22", unseal_dl22=False,
                            data_dir=str(root / "nowhere"))
    try:
        cmd_followup(ns)
        check("follow-up: dl22 refused without --unseal-dl22", False)
    except SystemExit as e:
        check("follow-up: dl22 refused without --unseal-dl22",
              "SEALED" in str(e.code))
    return F


def selftest_followup2(check, df, qrels, rules, root: Path, F: dict) -> None:
    """followup2 (EXPLORATORY): every new code path on the stub, plus the
    shared sections == `followup` (F) bit-exactly."""
    import contextlib
    import io
    import time
    from sklearn.metrics import cohen_kappa_score
    print("\n  -- follow-up 2 subcommand (EXPLORATORY) --")
    t0 = time.time()
    out = root / "out"
    df2 = pd.concat([df, _noisy_stub(qrels)], ignore_index=True)
    rules2 = json.loads(json.dumps(rules))
    rules2["rules"]["stub/noisy"] = {"msmarco": {
        "dial_level": 3, "monotone_map": [0, 0, 1, 2],
        "datasets": ["dl19", "dl20", "dl21"], "delta_hat": 0.9}}
    judges = list(STUB_JUDGES) + ["stub/noisy"]
    flag = frozenset({"stub/flag"})
    seeds = (FOLD_SEEDS[0], FOLD_SEEDS[1])
    fu = followup2_datasets(df2, qrels, rules2, ["dl21", "antique"], judges,
                            flag, out, 200, 60, SEED, seeds, E6P_SIZES, 150)
    G = {(r["model"], r["dataset"]): r for ds in fu for r in fu[ds]
         if "error" not in r}
    print(f"  ({len(G)} follow-up-2 cells in {time.time() - t0:.1f}s; RETHRESH "
          f"points / eval-only CIs == followup's arms, nested replicates == "
          f"nested_bootstrap, primary-seed robustness row == primary analysis "
          f"— asserted inside followup2_cell)")
    check("followup2: same cells as followup", set(G) == set(F), f"{len(G)}")

    # shared sections are followup's, unchanged
    bad = []
    for k, r in G.items():
        f = F[k]
        for sec in ("factorial", "coincidences", "stored_labels_check"):
            if json.dumps(_jclean(r[sec])) != json.dumps(_jclean(f[sec])):
                bad.append((k, sec))
        a, b = dict(r["f1_refit"]), dict(f["f1_refit"])
        if json.dumps(_jclean(a)) != json.dumps(_jclean(b)):
            bad.append((k, "f1_refit (incl. nested)"))
        oa = {x: y for x, y in r["opcurve"].items() if x != "grid_best_insample"}
        ob = {x: y for x, y in f["opcurve"].items() if x != "grid_ceiling"}
        if json.dumps(_jclean(oa)) != json.dumps(_jclean(ob)):
            bad.append((k, "opcurve"))
        for q, row in f["e6_prime"]["sizes"].items():
            r2 = r["e6_prime"]["sizes"][q]
            for nm in ("REFIT", "F1REFIT"):
                if "skipped" in row:
                    continue
                x = {kk: vv for kk, vv in r2[nm].items()
                     if kk not in ("raw_fallback_rate", "n_raw_fallback")}
                if json.dumps(_jclean(x)) != json.dumps(_jclean(row[nm])):
                    bad.append((k, f"e6p {q} {nm}"))
    check("followup2: factorial / opcurve / F1-refit + nested / E6' == "
          "followup bit-exactly (only additions + the grid rename)", not bad,
          str(bad[:3]))
    check("followup2: grid key renamed (grid_ceiling -> grid_best_insample, "
          "'NOT an upper bound')",
          all("grid_ceiling" not in r["opcurve"]
              and r["opcurve"]["grid_best_insample"]["note"] == GRID_REF_NOTE
              for r in G.values()))

    # nested_bootstrap(return_reps=True) leaves the summary unchanged
    cell_n = prepare_cell("stub/noisy", "dl21",
                          scope_rows(df2, "stub/noisy", "dl21", flag, out,
                                     {})[0], qrels, SEED)
    fc_n = followup_cache(cell_n, FOLD_SEEDS[0])
    a = nested_bootstrap(fc_n, 30, SEED)
    b = nested_bootstrap(fc_n, 30, SEED, return_reps=True)
    reps = b.pop("reps")
    check("nested_bootstrap(return_reps=True): identical summary + "
          "per-replicate weights / F1",
          json.dumps(_jclean(a)) == json.dumps(_jclean(b))
          and reps["Wq"].shape == (30, fc_n["nq"]) and reps["REFIT"].shape == (30,))

    # (R) RETHRESH == brute force on explicit pair rows (sklearn kappa)
    worst, kbad = [], 0.0
    for key in (("stub/noisy", "dl21"), ("stub/lenient", "dl21"),
                ("stub/strict", "antique"), ("stub/flag", "dl21")):
        m_, ds_ = key
        cell = prepare_cell(m_, ds_, scope_rows(df2, m_, ds_, flag, out, {})[0],
                            qrels, SEED)
        cell["info"]["flagship"] = m_ in flag
        cnt3, h = cell["cnt"][2], cell["h"]
        nrec, vsum = cnt3.sum(1), cnt3 @ np.arange(4)
        K = G[key]["rethresh"]["K"]
        t_raw = G[key]["rethresh"]["t_raw"]
        pf = np.array([fold_of(ds_, q, FOLD_SEEDS[0]) for q in cell["pair_qid"]])
        yt = (h >= 2).astype(int)
        order = [t_raw] + [t for t in range(1, 3 * K + 1) if t != t_raw]
        for nm in RT_ARMS:
            lab = np.zeros(len(h), dtype=int)
            for k in (0, 1):
                tr = pf != k
                best, bt = -np.inf, None
                for t in order:
                    yp = (vsum[tr] * K >= t * nrec[tr]).astype(int)
                    if nm == "RETHRESH_K":
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            sc = cohen_kappa_score(yt[tr], yp)
                        Ns = tr.sum()
                        mine = _bin_kappa((yt[tr] & yp).sum(), yp.sum(),
                                          yt[tr].sum(), Ns)
                        if np.isfinite(sc) and np.isfinite(mine):
                            kbad = max(kbad, abs(float(sc) - float(mine)))
                    else:
                        den = yp.sum() + yt[tr].sum()
                        sc = 2 * (yt[tr] & yp).sum() / den if den else np.nan
                    if np.isfinite(sc) and sc > best:
                        best, bt = sc, t
                bt = t_raw if bt is None else bt
                got = G[key]["rethresh"]["arms"][nm]["folds"][k]["t"]
                if got != bt:
                    worst.append((key, nm, k, bt, got))
                te = pf == k
                lab[te] = (vsum[te] * K >= bt * nrec[te]).astype(int)
            f1 = 2 * (yt & lab).sum() / (yt.sum() + lab.sum())
            if abs(f1 - G[key]["rethresh"]["arms"][nm]["point"]["F1@2"]) > 1e-12:
                worst.append((key, nm, "F1", f1))
    check("RETHRESH: fold t (ties -> RAW t, then lower t) and held-out F1@2 == "
          "brute force on explicit pair rows (4 cells x 2 objectives)",
          not worst, str(worst[:3]))
    check("RETHRESH: binary kappa == sklearn cohen_kappa_score", kbad < 1e-12,
          f"max |diff| {kbad:.1e}")
    rc_fake = {"tp": np.array([[5.0], [5.0], [5.0], [5.0]]),
               "lr": np.array([[10.0], [10.0], [10.0], [10.0]]),
               "pref": np.array([2, 0, 1, 3]), "t_raw": 3,
               "ts": np.arange(1, 5)}
    fc_fake = {"hr": np.array([10.0]), "ppq": np.array([30.0])}
    j_all, _ = rt_pick(rc_fake, fc_fake, np.ones((1, 1)), "F1")
    rc_fake2 = dict(rc_fake, tp=np.array([[6.0], [5.0], [5.0], [6.0]]))
    j_low, _ = rt_pick(rc_fake2, fc_fake, np.ones((1, 1)), "F1")
    check("RETHRESH tie-break: all tied -> RAW t; tie not incl. RAW -> lower t",
          int(j_all[0]) == 2 and int(j_low[0]) == 0,
          f"{int(j_all[0]) + 1}, {int(j_low[0]) + 1}")
    bad = [(k, c) for k, r in G.items()
           for c, v in r["rethresh"]["contrasts"].items()
           if v["nested_ci90"] is None or v["nested_ci90"][2] != 60
           or v["verdict"] not in ("beats", "within", "below")]
    check("RETHRESH: every contrast has eval-only + nested (60/60 replicates) "
          "CIs and a verdict", not bad, str(bad[:3]))
    rl = G[("stub/lenient", "dl21")]["rethresh"]
    check("RETHRESH: lenient stub re-thresholds away from RAW t (both "
          "objectives) and recovers most of the F1",
          all(f["t"] > rl["t_raw"] for nm in RT_ARMS
              for f in rl["arms"][nm]["folds"])
          and rl["contrasts"]["RETHRESH_K-RAW"]["point"] > 0.2,
          str({nm: [f["t"] for f in rl["arms"][nm]["folds"]] for nm in RT_ARMS}))

    # (B2) curve verdicts incl. FROZEN:DIAL / FROZEN:MAP; grid flag
    o_l = G[("stub/lenient", "dl21")]["opcurve2"]["points"]
    o_f = G[("stub/flag", "dl21")]["opcurve2"]["points"]
    check("curve: FROZEN:DIAL/MAP verdicts — lenient (L3/0012): DIAL==RAW, "
          "MAP==FROZEN; flagship (L1/id): MAP==RAW, DIAL==FROZEN",
          o_l["FROZEN:DIAL"] == o_l["RAW"] and o_l["FROZEN:MAP"] == o_l["FROZEN"]
          and o_f["FROZEN:MAP"] == o_f["RAW"] and o_f["FROZEN:DIAL"] == o_f["FROZEN"]
          and all(set(CURVE_ARMS2) == set(r["opcurve2"]["points"])
                  for r in G.values()))
    check("curve: grid_below_l3_best flag == (grid F1 < L3-best F1)",
          all(r["opcurve2"]["grid_below_l3_best"]
              == (r["opcurve2"]["grid_best_insample"]["F1@2"]
                  < r["opcurve2"]["l3_best_insample"]["F1@2"] - 1e-12)
              for r in G.values()))
    fake = {"l3_best_insample": {"t": 12, "F1@2": 0.74, "ratio@2": 1.1},
            "grid_best_insample": {"combo": "L3/0123", "F1@2": 0.73,
                                   "ratio@2": 1.0},
            "grid_below_l3_best": True, "grid_minus_l3_best": -0.01,
            "grid_exceeded_by": ["REFIT"]}
    gl = "\n".join(_grid_lines(fake))
    check("curve: printed grid reference renamed and both flags printed",
          "NOT an upper bound" in gl and "ceiling:" not in gl
          and "BELOW the in-sample best L3" in gl and "exceeded by REFIT" in gl)

    # (S) fold-seed robustness
    bad = []
    for k, r in G.items():
        sr = r["fold_seed_robustness"]
        if sr["seeds"] != list(seeds) or len(sr["table"]) != 19:
            bad.append((k, "shape"))
            continue
        ok_s = sr["ok_seeds"]
        for key, t in sr["table"].items():
            if len(t["values"]) != len(ok_s) or not isinstance(t["flips"], int):
                bad.append((k, key))
        if len(ok_s) < 2:
            continue
        # the robustness seed's REFIT == batch-4 crossfit labels at that seed
        hh = prepare_cell(k[0], k[1], scope_rows(df2, k[0], k[1], flag, out,
                                                 {})[0], qrels, SEED)
        lab, _rr, _pf, _w = crossfit(hh, int(ok_s[1]))
        e = endpoints(np.bincount(hh["h"] * 4 + lab, minlength=16)
                      .reshape(4, 4))
        raw = endpoints(np.bincount(
            hh["h"] * 4 + labels_from_counts(hh["cnt"][2], IDENTITY_MAP),
            minlength=16).reshape(4, 4))
        v = sr["per_seed"][ok_s[1]]["dF1@2"]["REFIT-RAW"]
        if abs(v - float(e["F1@2"] - raw["F1@2"])) > 1e-12:
            bad.append((k, ok_s[1], v))
    check("fold-seed robustness: 19 metrics x seeds (42, 43), flips counted; "
          "seed-43 REFIT-RAW == batch-4 crossfit labels", not bad, str(bad[:3]))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_followup2(fu, ["dl21", "antique"],
                         {"git_hash": "stub", "code_dirty": None,
                          "rules_sha256": "0" * 64, "fold_seeds": list(seeds),
                          "B": 200, "nested_b": 60, "e6p_B": 150,
                          "e6p_q_sizes": list(E6P_SIZES)})
    txt = buf.getvalue()
    check("report: header caveats (multiplicity, in-sample, subsample, E6' "
          "confounding/finite population), RETHRESH 2-line tally, seed table, "
          "nested-tail warning",
          "MULTIPLICITY / NON-INDEPENDENCE" in txt
          and "no multiplicity correction" in txt
          and "PARTIALLY IN-SAMPLE" in txt and "FULLY IN-SAMPLE" in txt
          and "SUBSAMPLE: stub/flag" in txt
          and "confounded with pairs-per-query" in txt and "HEAVY" in txt
          and txt.count("kappa-objective: REFIT vs RETHRESH_K") == 3
          and txt.count("F1-objective:    F1REFIT vs RETHRESH_F1") == 3
          and "(S) FOLD-SEED ROBUSTNESS" in txt and "fold-seed flips" in txt
          and "3 replicates per 5% tail" in txt and "WARNING: < 50" in txt
          and "in-sample best-of-grid F1 ceiling" not in txt
          and "L3 threshold-curve best" not in txt
          and txt.count("NOT an upper bound") >= len(G)
          and "(B2) OPERATING-CURVE VERDICTS" in txt
          and "FROZEN:DIAL" in txt and "RAW-FALLBACK" in txt)

    # (E) RAW-fallback rate
    bad = []
    for k, r in G.items():
        for q, row in r["e6_prime"]["sizes"].items():
            if "skipped" in row:
                continue
            for nm in ("REFIT", "F1REFIT"):
                st = row[nm]
                rate = st.get("raw_fallback_rate")
                if st.get("n") and (rate is None or not 0 <= rate <= 1
                                    or abs(rate * st["n"] - st["n_raw_fallback"])
                                    > 1e-9):
                    bad.append((k, q, nm))
    fb = G[("stub/noop", "dl21")]["e6_prime"]["sizes"]["8"]["REFIT"]
    fl = G[("stub/lenient", "dl21")]["e6_prime"]["sizes"]["8"]["REFIT"]
    check("E6' RAW-fallback: rate in [0,1] == n_raw_fallback/n everywhere; "
          "calibrated (noop) stub falls back more often than lenient",
          not bad and fb["raw_fallback_rate"] > fl["raw_fallback_rate"],
          f"noop {fb['raw_fallback_rate']:.3f} vs lenient "
          f"{fl['raw_fallback_rate']:.3f}; violations {bad[:2]}")

    # writers + gate
    meta = {"status": EXPLORATORY, "exploratory": True, "pre_registered": False,
            "B": 200, "nested_b": 60, "seed": SEED, "fold_seeds": list(seeds),
            "fold_seed": seeds[0], "e6p_B": 150}
    hdr = followup2_header(fu, ["dl21", "antique"], meta)
    with contextlib.redirect_stdout(io.StringIO()):
        write_followup2(fu["dl21"], "dl21", out, meta, hdr)
    js = json.loads((out / "derived" / "batch6_followup2_dl21.json").read_text())
    texs = sorted((out / "derived" / "tex").glob("batch6_followup2_dl21_*.tex"))
    check("followup2 writers: JSON (EXPLORATORY meta, RETHRESH tally, seed "
          "table), 4 tex tables with the EXPLORATORY stamp",
          js["meta"]["status"] == EXPLORATORY
          and js["meta"]["pre_registered"] is False
          and js["meta"]["header"]["n_cells"] == len(G)
          and "REFIT-RETHRESH_K" in js["rethresh_tally"]
          and "fold_seed_robustness" in js["judges"]["stub/noisy"]
          and len(texs) == 4
          and all(t.read_text().startswith(f"% [{EXPLORATORY}]") for t in texs),
          f"{[t.name for t in texs]}")
    nomark = root / "PRED_nomark.md"
    mark = root / "PRED_mark.md"
    outcome = []
    for dsl, unseal, path in ((["dl22"], False, mark), (["dl22"], True, nomark),
                              (["dl22"], True, mark), (["dl21"], False, nomark)):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                followup2_gate(dsl, unseal, path)
            outcome.append("pass")
        except SystemExit as e:
            outcome.append("SEALED" if "SEALED" in str(e.code) else "heading"
                           if B6_MARKER in str(e.code) else "other")
    check("followup2 gate: dl22 needs --unseal-dl22 AND the B6 heading; "
          "dl21 needs neither",
          outcome == ["SEALED", "heading", "pass", "pass"], str(outcome))
    ns = argparse.Namespace(datasets="dl22", unseal_dl22=False,
                            data_dir=str(root / "nowhere"))
    try:
        cmd_followup2(ns)
        check("followup2: dl22 refused without --unseal-dl22 (CLI path)", False)
    except SystemExit as e:
        check("followup2: dl22 refused without --unseal-dl22 (CLI path)",
              "SEALED" in str(e.code))


def cmd_selftest(args) -> None:
    print("=" * 96)
    print("  BATCH 6 SELF-TEST (synthetic stub; no Drive, no network)")
    print("=" * 96)
    failures = []

    def check(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  — {detail}" if detail else ""))
        if not cond:
            failures.append(name)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_stub_dir(root)
        df, qrels, rules, flag_pairs = build_stub()
        judges = list(STUB_JUDGES)
        res = analyze_datasets(df, qrels, rules, ["dl21", "antique"], judges,
                               frozenset({"stub/flag"}), root, 1000, SEED,
                               (E6_N_LO, E6_N_HI), 20, FOLD_SEEDS)
        R = {(r["model"], r["dataset"]): r for ds in res for r in res[ds]}
        print("  (labels: vectorised path == batch5.judge_labels for RAW and "
              "FROZEN; fast endpoints == sklearn/vector paths incl. P/R@1,2 "
              "and calibration cells — asserted inside analyze_cell)")

        # 1. lenient: ratio > 1 under RAW, -> 1 under correction
        r = R[("stub/lenient", "dl21")]
        pr, pf, pre = (r["points"][x]["ratio@2"] for x in LABEL_SETS)
        check("lenient: RAW ratio@2 > 1", pr > 1.2, f"{pr:.3f}")
        check("lenient: FROZEN ratio@2 -> 1", abs(pf - 1) < 0.1
              and abs(pf - 1) < abs(pr - 1), f"{pf:.3f}")
        check("lenient: REFIT ratio@2 -> 1", abs(pre - 1) < 0.1, f"{pre:.3f}")
        check("lenient: REFIT picks dial L1 in both folds",
              all(x["dial_level"] == 1 for x in r["refit_rules"]),
              str([(x["dial_level"], x["monotone_map"]) for x in r["refit_rules"]]))
        check("lenient: Delta_cont RAW > 0, REFIT |Delta_cont| smaller",
              r["delta_cont"]["RAW"] > 0.3 and
              abs(r["delta_cont"]["REFIT"]) < r["delta_cont"]["RAW"],
              str({k: round(v, 4) for k, v in r["delta_cont"].items()}))
        check("dl21 msmarco rule flagged PARTIALLY in-sample; delta_hat check "
              "print-only", r["rule_frozen"]["in_sample"].startswith("PARTIALLY")
              and not r["delta_hat_check"]["applicable"])
        check("fold seeds: 5 robustness refits with mean+-sd",
              r["refit_seed_summary"]["dF1@2 REFIT-RAW"]["n"] == 5,
              str(r["refit_seed_summary"]["dF1@2 REFIT-RAW"]))

        # 2. strict: reversed
        r = R[("stub/strict", "antique")]
        pr, pf = r["points"]["RAW"]["ratio@2"], r["points"]["FROZEN"]["ratio@2"]
        check("strict: RAW ratio@2 < 1", pr < 0.8, f"{pr:.3f}")
        check("strict: FROZEN ratio@2 -> 1", abs(pf - 1) < abs(pr - 1)
              and abs(pf - 1) < 0.1, f"{pf:.3f}")
        check("strict: correction raises recall@2",
              r["points"]["FROZEN"]["R@2"] > r["points"]["RAW"]["R@2"],
              f"{r['points']['RAW']['R@2']:.3f} -> "
              f"{r['points']['FROZEN']['R@2']:.3f}")
        check("strict: E5 domain = strict (delta_hat < 0)",
              r["e5_domain"] == "strict")
        check("antique rule FULLY in-sample; Delta(L3) == delta_hat ENFORCED",
              r["rule_frozen"]["in_sample"].startswith("FULLY")
              and r["delta_hat_check"]["applicable"]
              and r["delta_hat_check"]["abs_diff"] <= DELTA_HAT_TOL,
              f"|diff|={r['delta_hat_check'].get('abs_diff')}")
        bad_rule = dict(rules["rules"]["stub/strict"]["antique"])
        bad_rule["delta_hat"] += 1e-6
        cell_s = prepare_cell("stub/strict", "antique",
                              scope_rows(df, "stub/strict", "antique",
                                         frozenset(), root, {})[0], qrels, SEED)
        try:
            check_delta_hat(cell_s, bad_rule, qrels)
            check("delta_hat mismatch (1e-6) hard-fails", False)
        except SystemExit:
            check("delta_hat mismatch (1e-6) hard-fails", True)

        # 3. no-op judge identical; mixed judge; invariant catches violations
        for key in (("stub/noop", "dl21"), ("stub/noop", "antique"),
                    ("stub/mixed", "dl21")):
            r = R[key]
            check(f"no-op {key}: flagged and bit-exact",
                  r["noop_frozen"] and noop_violations(r) == [])
        r = R[("stub/mixed", "antique")]
        check("mixed judge: antique rule NOT no-op and FROZEN != RAW",
              not r["noop_frozen"] and not np.array_equal(
                  r["labels"]["FROZEN"], r["labels"]["RAW"]))
        r = R[("stub/noop", "dl21")]
        doctored = dict(r)
        doctored["points"] = {k: dict(v) for k, v in r["points"].items()}
        doctored["points"]["FROZEN"]["F1@2"] += 1e-15
        check("no-op invariant detects a 1e-15 perturbation",
              "point:F1@2" in noop_violations(doctored))
        fresh = root / "fresh_out"
        fresh.mkdir()
        meta_probe = {"B": 10, "seed": SEED, "se_B": None, "fold_seeds": [42],
                      "blind_status": "registered", "rules_sha256": "x"}
        try:
            finalize({"dl21": [R[("stub/lenient", "dl21")], doctored]},
                     ["dl21"], fresh, meta_probe, False)
            check("finalize hard-fails on violation", False)
        except SystemExit:
            written = [p for p in fresh.rglob("*") if p.is_file()]
            check("finalize hard-fails on violation and writes NOTHING",
                  not written, f"files written: {written}")

        # 4. refit beats frozen when target distribution differs
        r = R[("stub/drift", "dl21")]
        f1 = {x: r["points"][x]["F1@2"] for x in LABEL_SETS}
        check("drift: F1@2 REFIT > FROZEN", f1["REFIT"] > f1["FROZEN"] + 0.05,
              f"{ {k: round(v, 3) for k, v in f1.items()} }")
        check("drift: F1@2 REFIT >= RAW", f1["REFIT"] >= f1["RAW"])
        check("drift: REFIT-FROZEN F1 CI excludes 0",
              r["boot"]["diff"]["REFIT-FROZEN"]["F1@2"]["ci90"][0] > 0)

        # 5. bootstrap CIs bracket point estimates (all cells, all endpoints)
        n_chk, bad = 0, []
        for (m, ds), rr in R.items():
            if "error" in rr:
                continue
            for ls in LABEL_SETS:
                for ep in ALL_EPS:
                    v = rr["points"][ls][ep]
                    lo, hi, _n = rr["boot"]["ci90"][ls][ep]
                    if lo is None or not np.isfinite(v):
                        continue
                    n_chk += 1
                    if not lo - 1e-12 <= v <= hi + 1e-12:
                        bad.append((m, ds, ls, ep, v, lo, hi))
            for key in rr["point_diff"]:
                for ep in DIFF_EPS:
                    v = rr["point_diff"][key][ep]
                    lo, hi = rr["boot"]["diff"][key][ep]["ci90"]
                    if lo is None or not np.isfinite(v):
                        continue
                    n_chk += 1
                    if not lo - 1e-12 <= v <= hi + 1e-12:
                        bad.append((m, ds, key, ep, v, lo, hi))
        check("bootstrap 90% CIs bracket point estimates", not bad,
              f"{n_chk} intervals checked; violations: {bad[:3]}")

        # 6. explicit-resampling replicate check
        from sklearn.metrics import cohen_kappa_score
        cell_l = prepare_cell("stub/lenient", "dl21",
                              scope_rows(df, "stub/lenient", "dl21",
                                         frozenset(), root, {})[0], qrels, SEED)
        rl = R[("stub/lenient", "dl21")]
        cms = {ls: cm_by_query(cell_l["h"], rl["labels"][ls], cell_l["pair_q"],
                               cell_l["n_queries"]) for ls in LABEL_SETS}
        bt = bootstrap(cms, 40, SEED, return_reps=True)
        idx = boot_indices(cell_l["n_queries"], 40, SEED)
        worst = 0.0
        for b in range(8):
            sel = np.concatenate([np.where(cell_l["pair_q"] == q)[0]
                                  for q in idx[b]])
            hb = cell_l["h"][sel]
            for ls in LABEL_SETS:
                lb = rl["labels"][ls][sel]
                yt, yp = hb >= 2, lb >= 2
                direct = {
                    "F1@2": 2 * (yt & yp).sum() / (yt.sum() + yp.sum()),
                    "ratio@2": yp.sum() / yt.sum(),
                    "MAE": np.abs(hb - lb).mean(),
                    "kappa": cohen_kappa_score(hb, lb, weights="linear"),
                }
                for ep, v in direct.items():
                    worst = max(worst, abs(float(bt["reps"][ls][ep][b]) - v))
        check("bootstrap replicates == explicit query resampling", worst < 1e-9,
              f"8 replicates x 3 sets x 4 endpoints, max |diff| {worst:.1e}")

        # 7. kappa cross-check ran; bridge kappa present
        worst = max(rr["kappa_crosscheck_max_absdiff"] for rr in R.values()
                    if "error" not in rr)
        check("fast kappa == sklearn (pair labels)", worst < 1e-9,
              f"max |diff| {worst:.1e}")
        check("authoritative paraphrase-mean kappa(L3) bridge reported",
              all(rr["kappa_auth_L3_universe"] is not None
                  for rr in R.values() if "error" not in rr))

        # 8. flagship restriction + universe accounting
        r = R[("stub/flag", "dl21")]
        fz = set(flag_pairs)
        check("flagship universe restricted to frozen pairs",
              r["info"]["flagship"] and r["n_pairs"] <= len(fz)
              < R[("stub/lenient", "dl21")]["n_pairs"],
              f"{r['n_pairs']} pairs (frozen {len(fz)})")
        rl = R[("stub/lenient", "dl21")]
        check("universe excludes parse-failed and never-collected pairs",
              rl["info"]["n_dropped_unusable"] >= 1 and
              ("101", "dl21-d1-0") not in set(rl["universe"]) and
              rl["n_pairs"] < sum(1 for k in qrels if k[0] == "dl21"),
              f"universe {rl['n_pairs']}, dropped unusable "
              f"{rl['info']['n_dropped_unusable']}")
        check("prompt ids per level counted",
              rl["info"]["prompts_per_level"] == {f"L{l}": 3 for l in LEVELS})

        # 9. sample-efficiency mode
        rr = analyze_cell(cell_l, rules["rules"]["stub/lenient"]["msmarco"],
                          100, SEED, se_sizes=(50, 100, 200, 400, 990),
                          se_B=30, fold_seeds=(42,))
        se = rr["sample_eff"]
        check("sample-eff: n=50..400 evaluated, oversize n skipped",
              all("skipped" not in se[str(n)] for n in (50, 100, 200, 400))
              and "skipped" in se["990"])
        check("sample-eff: P@2/R@2 tracked; REFIT F1 > RAW F1 (n=100)",
              se["100"]["REFIT_F1@2"][0] > se["100"]["RAW_F1@2"][0]
              and se["100"]["REFIT_P@2"][0] is not None
              and se["100"]["RAW_R@2"][0] is not None,
              f"{se['100']['RAW_F1@2'][0]:.3f} -> "
              f"{se['100']['REFIT_F1@2'][0]:.3f}")
        check("sample-eff: pool shrinks as n grows; failures counted",
              se["50"]["pool_pairs_mean"] > se["400"]["pool_pairs_mean"]
              and se["100"]["n_draws_failed"] == 0)

        # 10. gate (dl22 strict; non-dl22 needs marker or --non-blind)
        nomark = root / "PRED_nomark.md"
        nomark.write_text("# nothing registered\n", encoding="utf-8")
        mark_unfilled = root / "PRED_unfilled.md"
        mark_unfilled.write_text(f"# {B6_MARKER}\n登记时间：____\n", encoding="utf-8")
        mark = root / "PRED_mark.md"
        mark.write_text(f"# {B6_MARKER}\n登记时间：2026-10-11\n# next\n____\n",
                        encoding="utf-8")
        clean = lambda p: (True, "")                     # noqa: E731
        dirty = lambda p: (False, " M paper/PREDICTIONS.md")  # noqa: E731
        cases = (
            ("dl22: no marker -> refuse", ["dl22"], nomark, True, False, clean, True),
            ("dl22: marker, no --unseal-dl22 -> refuse", ["dl22"], mark, False, False, clean, True),
            ("dl22: marker+flag but DIRTY -> refuse", ["dl22"], mark, True, False, dirty, True),
            ("dl22: marker+flag but '____' in block -> refuse", ["dl22"], mark_unfilled, True, False, clean, True),
            ("dl22: --non-blind does not relax the gate", ["dl22"], nomark, True, True, clean, True),
            ("dl22: clean, filled, marker+flag -> pass ('____' after the block ignored)", ["dl22"], mark, True, False, clean, False),
            ("dl21: no marker, no --non-blind -> refuse", ["dl21"], nomark, False, False, clean, True),
            ("dl21: no marker + --non-blind -> pass", ["dl21"], nomark, False, True, clean, False),
            ("dl21: marker, dirty -> warn only", ["dl21"], mark, False, False, dirty, False),
        )
        for name, dsl, path, flag, nb, cfn, want_exit in cases:
            try:
                status = gate(dsl, flag, nb, path, cfn)
                exited = False
            except SystemExit:
                exited, status = True, None
            check(f"gate {name}", exited == want_exit, f"status={status}")
        try:
            resolve_datasets("dl19")
            check("dl19 refused (not in ALLOWED)", False)
        except SystemExit:
            check("dl19 refused (not in ALLOWED)", True)

        # 11. PASTE -> fill -> parse -> sha check -> adjudication
        stub_sha = sha256_file(root / "derived" / "correction_rules.json")
        lines = paste_block(rules, Path("correction_rules.json"), stub_sha,
                            ["dl21", "antique"], judges, frozenset({"stub/flag"}))
        txt = "\n".join(lines)
        check("paste block carries marker, sha line and verbatim rules",
              B6_MARKER in txt and f"rules_sha256: {stub_sha}" in txt
              and all(r_ in txt for r_ in ADJUDICATION_RULES))
        check("paste block: no-op judge gets E0, no E2/E4 scoring",
              "B6-E0" in txt and "N/A（结构性" in txt)
        check("paste block: REFIT CI note on E3/E4/E5 lines",
              all(REFIT_CI_NOTE in ln for ln in lines
                  if re.match(r"^- B6-E[345] ", ln) and "≡" not in ln))
        filled = _fill_stub_registration(lines)
        blk = b6_block(filled + "\n# Next section\n")
        reg = parse_registration(blk)
        check("registration parses sha and predictions",
              reg["rules_sha256"] == stub_sha and not reg["invalid"]
              and reg["preds"][("stub/strict", "antique")]["E1"] == "<1"
              and reg["preds"][("stub/noop", "dl21")]["E0"] == "structural"
              and "E2" not in reg["preds"][("stub/noop", "dl21")],
              f"{len(reg['preds'])} cells")
        try:
            check_rules_sha(reg, stub_sha)
            check("rules sha match passes", True)
        except SystemExit:
            check("rules sha match passes", False)
        for name, rg in (("sha mismatch aborts", {**reg, "rules_sha256": "0" * 64}),
                         ("missing sha line aborts", {**reg, "rules_sha256": None})):
            try:
                check_rules_sha(rg, stub_sha)
                check(name, False)
            except SystemExit:
                check(name, True)
        adj = {k: adjudicate(v, reg["preds"]) for k, v in R.items()}
        a = adj[("stub/drift", "dl21")]
        check("adjudication drift: all E1-E6 hit -> cell HIT",
              a["conjunction"] == "hit",
              str({E: e["verdict"] for E, e in a["endpoints"].items()}))
        a = adj[("stub/lenient", "dl21")]
        check("adjudication lenient: E4 tie (REFIT==FROZEN) -> cell TIE",
              a["endpoints"]["E4"]["verdict"] == "tie"
              and a["conjunction"] == "tie",
              str({E: e["verdict"] for E, e in a["endpoints"].items()}))
        a = adj[("stub/flag", "dl21")]
        check("adjudication flagship: E6 sizes not run -> NOT ADJUDICATED",
              a["endpoints"]["E6"]["verdict"] == "not adjudicated"
              and a["conjunction"].startswith("not adjudicated"),
              a["conjunction"])
        a = adj[("stub/strict", "antique")]
        check("adjudication strict: E1 '<1' hit, E5 uses recall",
              a["endpoints"]["E1"]["verdict"] == "hit"
              and a["endpoints"]["E5"]["desc"].startswith("R@2"))
        check("score rules: tie bands and signs",
              score_endpoint("E3", 0.005, "是") == "tie"
              and score_endpoint("E3", 0.0051, "是") == "hit"
              and score_endpoint("E3", -0.0051, "是") == "miss"
              and score_endpoint("E3", -0.0051, "否") == "hit"
              and score_endpoint("E1", 1.02, ">1") == "tie"
              and score_endpoint("E1", 0.97, ">1") == "miss"
              and score_endpoint("E2", -0.021, "是") == "hit"
              and score_endpoint("E2", 0.019, "是") == "tie"
              and score_endpoint("E3", 0.3, None) == "unregistered"
              and conjunction({"E1": "hit", "E3": "tie", "E4": "miss"}) == "miss"
              and conjunction({"E1": "hit", "E3": "tie"}) == "tie"
              and conjunction({"E1": "hit", "E6": "not adjudicated",
                               "E3": "tie"}).startswith("not adjudicated"))

        # 12. end-to-end writers (incl. pair-level labels)
        for rr in (r_ for ds in res for r_ in res[ds]):
            rr["adjudication"] = adjudicate(rr, reg["preds"])
        meta = {"B": 1000, "seed": SEED, "se_B": 20, "fold_seeds": list(FOLD_SEEDS),
                "blind_status": "NON-BLIND", "rules_sha256": stub_sha}
        out_dir = root / "out"
        finalize(res, ["dl21", "antique"], out_dir, meta, True)
        js = json.loads((out_dir / "derived" / "batch6_poolext_dl21.json")
                        .read_text())
        tex = (out_dir / "derived" / "tex" / "batch6_poolext_dl21.tex").read_text()
        labs = list((out_dir / "derived").glob("batch6_labels_dl21.*"))
        lab_df = (pd.read_parquet(labs[0]) if labs and labs[0].suffix == ".parquet"
                  else pd.read_pickle(labs[0]) if labs else None)
        check("json + tex + labels written; NON-BLIND stamped",
              "stub/lenient" in js["judges"] and tex.startswith("% [NON-BLIND]")
              and lab_df is not None and "REFIT_seed46" in lab_df.columns
              and js["judges"]["stub/drift"]["adjudication"]["conjunction"] == "hit",
              f"labels file {labs[0].name if labs else None}, "
              f"{0 if lab_df is None else len(lab_df)} rows")

        # 13. follow-up subcommand (exploratory)
        F_fu = selftest_followup(check, df, qrels, rules, root, R)

        # 14. follow-up 2 subcommand (exploratory)
        selftest_followup2(check, df, qrels, rules, root, F_fu)

    if args.keep_dir:
        kd = Path(args.keep_dir)
        write_stub_dir(kd)
        print(f"\n  stub DATA_DIR written to {kd}  (judgments: "
              f"derived/judgments.stub.pkl; qrels: qrels_stub.json; judges: "
              f"{','.join(STUB_JUDGES)}; flagship: stub/flag)")
    print()
    if failures:
        sys.exit(f"SELF-TEST FAILED: {len(failures)} check(s): {failures}")
    print("  SELF-TEST PASSED")


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, default_ds):
        p.add_argument("--data-dir", default=None,
                       help="DATA_DIR (default: $DATA_DIR)")
        p.add_argument("--datasets", default=default_ds,
                       help=f"comma list from {list(ALLOWED)}")
        p.add_argument("--judges", default=None,
                       help="comma list of judge model_ids (default: the six "
                            "round-2 judges)")
        p.add_argument("--flagships", default=None,
                       help="comma list of judges restricted to "
                            "inputs/pairs_<ds>_flagship40.jsonl "
                            f"(default: {sorted(FLAGSHIPS)})")
        p.add_argument("--rules", default=None,
                       help="correction_rules.json (default: "
                            "$DATA_DIR/derived/correction_rules.json)")
        p.add_argument("--fold-seeds", default=",".join(map(str, FOLD_SEEDS)),
                       help="REFIT cross-fit fold seeds; the FIRST is primary "
                            "(scored), the rest are a robustness report")

    def data_opts(p):
        p.add_argument("--judgments", default=None,
                       help="judgments file (default: $DATA_DIR/derived/"
                            "judgments.parquet; .pkl accepted for stub tests)")
        p.add_argument("--qrels-json", default=None,
                       help="TEST ONLY: stub qrels [[ds,qid,docid,grade],...] "
                            "instead of src.qrels")
        p.add_argument("--seed", type=int, default=SEED,
                       help="bootstrap / sample-efficiency seed")
        p.add_argument("--sizes", default=",".join(map(str, SE_SIZES)),
                       help="sample-efficiency n_human sizes")

    p = sub.add_parser("run", help="the analysis (registration-gated)")
    common(p, "dl21,antique")
    data_opts(p)
    p.add_argument("--unseal-dl22", action="store_true",
                   help="required (with a clean, filled, committed B6 block) "
                        "for dl22")
    p.add_argument("--non-blind", action="store_true",
                   help="run dl21/antique WITHOUT a B6 registration; stamps "
                        "NON-BLIND everywhere. Never relaxes the dl22 gate.")
    p.add_argument("--b", type=int, default=B_BOOT, help="bootstrap replicates")
    p.add_argument("--sample-efficiency", action="store_true")
    p.add_argument("--se-b", type=int, default=B_SE,
                   help="sample-efficiency draws per n")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("validate", help="inputs + coverage, no agreement metric")
    common(p, "dl21,antique")
    data_opts(p)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("paste", help="print the B6 registration block (blind)")
    common(p, "dl22,dl21,antique")
    p.set_defaults(func=cmd_paste)

    p = sub.add_parser("followup", help=f"{EXPLORATORY}: factorial, "
                       "operating curve, F1-objective refit + nested "
                       "bootstrap, E6' query-count sample efficiency")
    common(p, "dl21,antique")
    data_opts(p)
    p.add_argument("--unseal-dl22", action="store_true",
                   help="required for dl22 (sealed holdout)")
    p.add_argument("--b", type=int, default=B_BOOT,
                   help="evaluation-only query-bootstrap replicates")
    p.add_argument("--nested-b", type=int, default=NESTED_B,
                   help="nested (fit-aware) bootstrap replicates; 0 = skip")
    p.add_argument("--e6p-b", type=int, default=E6P_B,
                   help="E6' draws per calibration-query count")
    p.add_argument("--q-sizes", default=",".join(map(str, E6P_SIZES)),
                   help="E6' calibration query counts ('' = skip E6')")
    p.set_defaults(func=cmd_followup)

    p = sub.add_parser("followup2", help=f"{EXPLORATORY}: evaluator re-run "
                       "requests — RETHRESH baseline, FROZEN dial/map curve "
                       "verdicts, fold-seed robustness, E6' RAW-fallback")
    common(p, "dl21,antique")
    data_opts(p)
    for act in p._actions:
        if act.dest == "fold_seeds":
            act.default = ",".join(map(str, FU2_FOLD_SEEDS))
            act.help = ("cross-fit fold seeds; the FIRST is primary (eval-only "
                        "+ nested CIs), the rest give point-estimate "
                        "robustness tables (default: 42 only; e.g. "
                        "42,43,44,45,46)")
    p.add_argument("--unseal-dl22", action="store_true",
                   help="required for dl22 (sealed holdout); dl22 also needs "
                        "the B6 registration heading in paper/PREDICTIONS.md")
    p.add_argument("--b", type=int, default=B_BOOT,
                   help="evaluation-only query-bootstrap replicates")
    p.add_argument("--nested-b", type=int, default=NESTED_B,
                   help="nested (fit-aware) bootstrap replicates; 0 = skip; "
                        "use 1000 for borderline cells")
    p.add_argument("--e6p-b", type=int, default=E6P_B,
                   help="E6' draws per calibration-query count")
    p.add_argument("--q-sizes", default=",".join(map(str, E6P_SIZES)),
                   help="E6' calibration query counts ('' = skip E6')")
    p.set_defaults(func=cmd_followup2)

    p = sub.add_parser("selftest", help="synthetic stub drive (offline)")
    p.add_argument("--keep-dir", default=None,
                   help="also write the stub DATA_DIR here for CLI demos")
    p.set_defaults(func=cmd_selftest)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
