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

Subcommands
-----------
    paste     --datasets dl22,dl21,antique
    validate  --datasets dl21,antique          (counts only, no agreement)
    run       --datasets dl21,antique [--sample-efficiency] [--non-blind]
    run       --datasets dl22 --unseal-dl22
    selftest  [--keep-dir DIR]

Outputs (under DATA_DIR — this repo carries no data artefacts):
    $DATA_DIR/derived/batch6_poolext_<ds>.json
    $DATA_DIR/derived/batch6_labels_<ds>.parquet   (.pkl if no parquet engine)
    $DATA_DIR/derived/tex/batch6_poolext_<ds>.tex
    $DATA_DIR/derived/tex/batch6_poolext_<ds>_sampleff.tex  (--sample-efficiency)
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

    p = sub.add_parser("selftest", help="synthetic stub drive (offline)")
    p.add_argument("--keep-dir", default=None,
                   help="also write the stub DATA_DIR here for CLI demos")
    p.set_defaults(func=cmd_selftest)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
