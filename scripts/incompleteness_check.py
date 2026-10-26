#!/usr/bin/env python3
"""IC — pre-registered LABEL-INCOMPLETENESS check (analysis only, no API).

Reviewer objection answered
---------------------------
"LLM judges look lenient on MS MARCO / TREC DL only because the human qrels
missed relevant passages; the offset Δ is a dataset artifact, not judge
miscalibration."  The paper's framing is *alignment to THESE assessors'
operating point*; this script asks, from existing data only, whether the
measured offset behaves like a SCALE SHIFT of the judge or like MISSING HUMAN
POSITIVES.  Neither outcome turns the correction into a quality claim.

Data (nothing new is collected)
-------------------------------
* judge records: $DATA_DIR/derived/judgments.parquet (src/parse.py; columns
  model_id, prompt_id, politeness_level 1..5, dataset, qid, docid, run,
  score 0..3, parse_ok); tone level = ``politeness_level`` (L3 = neutral
  default prompt).
* human grade: src.qrels.load_qrels (ANTIQUE 1-4 -> 0-3), through
  batch6.load_human.
* judges: the paper roster config/models.yaml (12 judges); flagships
  (tier == flagship) restricted to their frozen 40% sample
  inputs/pairs_<ds>_flagship40.jsonl (batch6.scope_rows) — marked "[40%]".
* pair universe per (judge, dataset): batch6.prepare_cell VERBATIM — the
  human-judged pairs the judge has a usable label for (parse_ok & numeric
  score) at EVERY level L1..L5 (the B6 / DE universe; it drops parse failures
  and the B5 L1/L3-only top-up pairs, so the tone comparison D is on one
  identical pair set).  Where derived/batch6_labels_<ds> exists the universe
  (paste: keys; run: keys + human + RAW) is hard-checked against it.
* RAW = the L3 record mean (mean over ALL usable L3 records: paraphrases x
  runs), continuous — the Δ estimand of Part I / batch 4.  RAW_lab =
  round-half-up(RAW) (batch5.judge_labels = batch6 RAW label) is reported
  beside it and is the label used for "relevant" (>= 2) in E.
  Offsets are pair-weighted (one human grade per pair).

Analyses (per judge x dataset unless stated)
--------------------------------------------
  A  conditional offset by human grade h in {0,1,2,3}: n, mean RAW,
     off(h) = mean(RAW - h) [query bootstrap 90% CI], the same for RAW_lab;
     overall Δ; share of the offset mass Σ(RAW - h) carried by h = 0 pairs
     vs the share of pairs that are h = 0.  Criteria A1 / A2 (CRITERIA).
  B  offset vs judging depth: per query depth = # universe pairs (B) and
     # qrels judgments of the query (Bq, uncapped); o_q = mean(RAW - h);
     Spearman ρ(depth, o_q) with a query-bootstrap 90% CI (B = 1000, seed 42);
     median split (depth <= median vs > median; if more than half the queries
     sit at the cap, depth < median vs >= median; threshold fixed on the full
     sample), mean o_q per side and the difference shallow - deep.
  C  mean offset per dataset per judge (Δ; NON-BLIND — already published).
  D  offset at each tone level L1..L5 on the identical universe, D(l) =
     off(L_l) - off(L3) (NON-BLIND — already published): a fixed label
     incompleteness cannot move with the prompt tone.
  E  EXPLORATORY.  Per dataset, the cheap judges, pairs in EVERY cheap
     judge's universe with human grade 0 (h = 1 printed as a reference
     stratum); R_j = RAW_lab >= 2; c = Σ_j R_j.  Share of pairs with c >= k
     among pairs with c >= 1, k = 1..m, observed vs the Poisson-binomial
     expectation under independence with the judges' marginal rates on the
     same pairs; lift_k = observed / independent; mean pairwise lift
     P(R_i & R_j) / (p_i p_j).

Subcommands
-----------
  paste     prints the "# IC 预注册 — 标注不完整性检查" registration block:
            unit counts per judge x dataset (pairs, queries, universe and
            qrels depth quantiles, pairs per human grade), the input sha256s
            (judgments file, qrels per dataset as loaded — canonical sorted
            "qid<TAB>docid<TAB>grade" lines —, flagship 40% files) and the
            prediction placeholders.  Loads NO agreement numbers: no judge
            score enters any printed value (selftest checks that the block is
            invariant to the judge scores).
  run       gated like scripts/decision_errors.py: the heading must exist in
            a clean, committed paper/PREDICTIONS.md with no '____' left; dl22
            needs --unseal-dl22; every input sha256 must be registered
            ("sha256[<name>]: <hex>") and match.  Computes A-E, adjudicates
            the registered prediction lines, writes the outputs.
  selftest  synthetic, offline (scale-shift judges vs missing-positive
            judges, gate in a temp git repo, sha checks, paste invariance,
            end-to-end run); must print "ALL SELFTESTS PASSED".

Registered prediction lines
---------------------------
    - IC[<criterion>|<family>] <text>｜预测：≥k/N   (or ≤k/N)
criterion ∈ A1, A2, B, Bq, E; family ∈ msmarco (dl19-dl22), antique.
"≥k/N" = at least k of the N cells (judge x dataset; datasets for E) whose
criterion is DEFINED pass it.  HIT / MISS on the point estimates; a 90% CI
never changes a verdict; if the registered N differs from the number of
defined cells the line is "not adjudicated (N ...)".

Outputs: $DATA_DIR/derived/incompleteness/ic_<part>.json (part = units, A, B,
C, D, E, verdict), the long ic_summary.csv (analysis, dataset, judge, key,
value, ci_lo, ci_hi, n) and ic_log.txt (the printed report).

Usage (Colab, repo root):
    python scripts/incompleteness_check.py selftest
    python scripts/incompleteness_check.py paste
    python scripts/incompleteness_check.py run --unseal-dl22      # after commit
CPU only, no API calls.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.qrels import assert_not_sealed, parse_datasets   # noqa: E402
import batch4_correction as b4                             # noqa: E402
import batch5_downstream as b5                             # noqa: E402
import batch6_poolext as b6                                # noqa: E402
import batch7_distill as b7                                # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
# Constants (registered verbatim; not adjustable from the CLI)
# ═══════════════════════════════════════════════════════════════════════════

IC_MARKER = "IC 预注册 — 标注不完整性检查"
OUT_SUB = ("derived", "incompleteness")

DATASETS = ("dl19", "dl20", "dl21", "dl22", "antique")
FAMILY_OF = {"dl19": "msmarco", "dl20": "msmarco", "dl21": "msmarco",
             "dl22": "msmarco", "antique": "antique"}
FAMILIES = ("msmarco", "antique")
LEVELS = (1, 2, 3, 4, 5)
GRADES = (0, 1, 2, 3)

SEED = 42
B_BOOT = 1000
CI_LO, CI_HI = 5.0, 95.0          # 90% percentile interval
REL = 2                           # relevant = RAW_lab >= 2 (E)
A_MIN_N = 10                      # min pairs per grade for A1/A2 to be defined
A_FRAC = 1.0 / 3.0
B_RHO_BAND = 0.30
B_SPLIT_TOL = 0.10
E_LIFT_MAX = 2.0
E_K = 2                           # lift_k used by the E criterion
E_STRATA = (0, 1)                 # h = 0 primary, h = 1 reference
CRITS = ("A1", "A2", "B", "Bq", "E")
CSV_COLS = ("analysis", "dataset", "judge", "key", "value", "ci_lo", "ci_hi",
            "n")
MODELS_YAML = REPO / "config" / "models.yaml"

CRITERIA = {
    "A1": ("off(h) = mean(RAW − h) over the cell's pairs with human grade h "
           "(RAW = L3 record mean, continuous).  PASS iff off(0) ≠ 0, off(1) "
           "and off(2) both have the sign of off(0), and min(|off(1)|, "
           "|off(2)|) ≥ |off(0)|/3.  Defined iff n(h) ≥ 10 for h = 0, 1, 2."),
    "A2": ("floor/ceiling-robust form: Δ = mean(RAW − h) over all the cell's "
           "pairs.  PASS iff Δ ≠ 0, off(1) and off(2) both have the sign of Δ, "
           "and min(|off(1)|, |off(2)|) ≥ |Δ|/3.  Defined iff n(h) ≥ 10 for "
           "h = 1, 2."),
    "B": ("per query: depth = # universe pairs, o_q = mean(RAW − h).  PASS iff "
          "Spearman ρ(depth, o_q) ∈ [−0.30, +0.30] AND |mean o_q(shallow) − "
          "mean o_q(deep)| < 0.10, shallow = depth ≤ median (if that is every "
          "query — more than half at the cap — shallow = depth < median), "
          "deep = the rest; queries unweighted.  Defined iff ≥ 3 queries and "
          "depth not constant."),
    "Bq": ("as B, with depth = # qrels judgments of the query in the full "
           "qrels (uncapped; the universe depth is capped by pair "
           "construction)."),
    "E": ("per dataset: cheap judges, pairs in every cheap judge's universe "
          "with human grade 0; R_j = RAW_lab ≥ 2; c = Σ_j R_j.  lift_2 = "
          "P_obs(c ≥ 2 | c ≥ 1) / P_ind(c ≥ 2 | c ≥ 1), P_ind = "
          "Poisson-binomial with the judges' marginal rates on the same pairs. "
          " PASS iff lift_2 ≤ 2.0.  Defined iff ≥ 2 cheap judges and "
          "P_obs(c ≥ 1) > 0."),
}

NOTES = [
    "纯\"漏标正例\"：偏移集中在 h=0；h≥1 的 off≈0 或为负 → A1/A2 不满足；"
    "浅判 query 偏移更大（ρ<0、浅−深>0）→ B 不满足（方向）；多个 judge "
    "在同一批 h=0 对上同时判相关 → lift ≫ 1 → E 不满足。",
    "纯\"尺度平移\"：h=1、h=2 的 off 与 Δ 同号（h=3 有天花板）→ A 满足；"
    "偏移与判深无关 → B 满足；h=0 上的正判在给定边际率时近似独立 → E 满足。",
    "截断：任何 judge 在 h=0 处 RAW−h ≥ 0、在 h=3 处 ≤ 0（纯噪声也给出 "
    "off(0)>0、off(3)<0）→ 判别力在内部档 1–2；ANTIQUE（Δ<0）上 A1 结构性"
    "不可满足，用 A2。",
    "共享的条目难度会让 judge 之间正相关，即使没有漏标 → lift>1 是预期的；"
    "h=1 层的 lift 作为同一流程的参照一并打印。",
    "universe 判深被构造截断（候选 top-50 ∩ qrels；dl21/dl22/antique 每 query "
    "最多 30 对）→ Bq 用未截断的 qrels 计数；判深本身可能随相关文档数变化"
    "（pool / 判定流程），这是 B 与 Bq 的共同混杂。",
]

ADJUDICATION_RULES = [
    "1. 每条登记行 IC[<判据>|<家族>] 的 token ≥k/N（或 ≤k/N）作用于该家族中"
    "判据有定义的单元（judge×dataset；E 为 dataset）里满足判据的个数。",
    "2. 点估计判定：满足个数 ≥k（≤k）记命中，否则未中；90% CI 单独报告，永不"
    "改变判定。",
    "3. 登记的 N 与实际有定义的单元数不同（缺数据集、单元未定义）→ 不判定，"
    "注明原因。",
    "4. C、D 已发表（non-blind），只报告不计分；E 为探索性（可登记，但标注 "
    "exploratory）。",
]


# ═══════════════════════════════════════════════════════════════════════════
# Small helpers
# ═══════════════════════════════════════════════════════════════════════════

_LOG: list[str] = []


def say(*a) -> None:
    s = " ".join(str(x) for x in a)
    print(s)
    _LOG.append(s)


def out_dir(data_dir: Path) -> Path:
    p = data_dir.joinpath(*OUT_SUB)
    p.mkdir(parents=True, exist_ok=True)
    return p


def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds")


def ci90(x) -> list:
    x = np.asarray(x, dtype=float)
    f = x[np.isfinite(x)]
    if f.size == 0:
        return [None, None]
    return [float(np.percentile(f, CI_LO)), float(np.percentile(f, CI_HI))]


def fnum(x):
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def _f(x, p=3, sign=True) -> str:
    x = fnum(x)
    if x is None:
        return "n/a"
    return f"{x:+.{p}f}" if sign else f"{x:.{p}f}"


def _ci(c, p=3) -> str:
    if not c or c[0] is None:
        return "[n/a]"
    return f"[{c[0]:+.{p}f},{c[1]:+.{p}f}]"


def short(model: str) -> str:
    return model.split("/")[-1]


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(b6._jclean(obj), ensure_ascii=False, indent=1,
                               allow_nan=False), encoding="utf-8")
    say(f"[json] wrote {path}")


def predictions_path(arg: str | None) -> Path:
    return Path(arg) if arg else REPO / "paper" / "PREDICTIONS.md"


def _git_in(path: Path, *a) -> tuple[int, str]:
    try:
        p = subprocess.run(["git", "-C", str(Path(path).resolve().parent), *a],
                           capture_output=True, text=True)
        return p.returncode, (p.stdout.strip() or p.stderr.strip())
    except Exception as e:                                     # noqa: BLE001
        return 1, str(e)


def roster(spec_judges: str | None, spec_cheap: str | None,
           spec_flag: str | None, models_yaml: Path = MODELS_YAML) -> tuple:
    """(judges in roster order, cheap set, flagship set).  Default = the paper
    roster config/models.yaml; CLI lists are TEST hooks / overrides."""
    def lst(s):
        return [x.strip() for x in s.split(",") if x.strip()]
    if spec_judges is None or spec_cheap is None or spec_flag is None:
        import yaml
        spec = yaml.safe_load(models_yaml.read_text(encoding="utf-8"))
        ms = spec["models"]
    judges = lst(spec_judges) if spec_judges is not None else \
        [m["model_id"] for m in ms]
    cheap = frozenset(lst(spec_cheap)) if spec_cheap is not None else \
        frozenset(m["model_id"] for m in ms if m.get("tier") == "cheap")
    flag = frozenset(lst(spec_flag)) if spec_flag is not None else \
        frozenset(m["model_id"] for m in ms if m.get("tier") == "flagship")
    return judges, cheap, flag


def resolve_datasets(spec: str) -> list[str]:
    names = parse_datasets(spec)
    bad = [n for n in names if n not in DATASETS]
    if bad:
        sys.exit(f"ERROR: dataset(s) {bad} not in the IC scope {DATASETS}.")
    return [d for d in DATASETS if d in names]


# ═══════════════════════════════════════════════════════════════════════════
# Registration: locate, parse, gate, sha checks, adjudication
# ═══════════════════════════════════════════════════════════════════════════

_SHA_LINE = re.compile(r"sha256\[([^\]]+)\]\s*[:：]\s*([0-9a-f]{64})")
_PRED_LINE = re.compile(r"^\s*-\s*IC\[([^\]]+)\].*?预测[:：]\s*(.*)$")
_TOK = re.compile(r"^(≥|>=|≤|<=)\s*(\d+)\s*/\s*(\d+)")


def ic_block(text: str) -> str | None:
    """From the '# IC 预注册 — 标注不完整性检查' heading to the next '# '."""
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines)
                  if re.match(r"^#\s", ln) and IC_MARKER in ln), None)
    if start is None:
        return None
    end = next((j for j in range(start + 1, len(lines))
                if re.match(r"^#\s", lines[j])), len(lines))
    return "\n".join(lines[start:end])


def parse_registration(block: str) -> dict:
    shas = {m.group(1).strip(): m.group(2)
            for m in _SHA_LINE.finditer(block)}
    preds, invalid = {}, []
    for line in block.splitlines():
        m = _PRED_LINE.match(line)
        if not m:
            continue
        key = "|".join(p.strip() for p in m.group(1).split("|"))
        mt = _TOK.match(m.group(2).strip())
        if not mt:
            invalid.append((key, m.group(2).strip()))
            continue
        op = "≥" if mt.group(1) in ("≥", ">=") else "≤"
        preds[key] = (op, int(mt.group(2)), int(mt.group(3)))
    return {"shas": shas, "preds": preds, "invalid": invalid}


def ic_gate(dl22: bool, unseal: bool, path: Path,
            clean_fn=b7.file_committed) -> dict:
    """decision_errors.de_gate with the IC heading.  Called BEFORE any data
    is read."""
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    block = ic_block(text)
    if block is None:
        sys.exit(f"ERROR: {path} has no heading '# {IC_MARKER}'.\n  Print the "
                 f"block with `paste`, fill it, COMMIT it — only then `run`.")
    ok, why = clean_fn(path)
    if not ok:
        sys.exit(f"ERROR: {path} {why}. The IC registration counts only once "
                 f"it is committed (commit timestamp = freeze time).")
    if "____" in block:
        sys.exit("ERROR: the IC registration block still contains unfilled "
                 "'____' placeholders.")
    if dl22:
        assert_not_sealed(["dl22"], unseal=unseal)
    reg = parse_registration(block)
    for key, tok in reg["invalid"]:
        print(f"[registration] WARNING: IC[{key}]: unparseable prediction "
              f"{tok!r} — treated as unregistered.", file=sys.stderr)
    rc, out = _git_in(path, "log", "-1", "--format=%H,%cI", "--", path.name)
    reg["commit"] = out if rc == 0 and out else None
    return reg


def check_shas(reg: dict, actual: dict) -> None:
    problems = []
    for name, sha in actual.items():
        want = reg["shas"].get(name)
        if want is None:
            problems.append(f"{name}: not registered (no 'sha256[{name}]: "
                            f"<hex>' line in the IC block)")
        elif want != sha:
            problems.append(f"{name}: sha256 {sha} != registered {want}")
    if problems:
        sys.exit("ERROR: input integrity check failed —\n    "
                 + "\n    ".join(problems)
                 + "\n  The inputs changed after registration (or were never "
                   "registered); refusing to run.")
    for name, sha in actual.items():
        say(f"[sha] {name}: {sha[:16]}… matches the registration")


def adjudicate(pred, n_pass: int, n_def: int) -> str:
    op, k, n_reg = pred
    if n_reg != n_def:
        return (f"not adjudicated (registered N={n_reg} ≠ {n_def} defined "
                f"cells)")
    ok = n_pass >= k if op == "≥" else n_pass <= k
    return "HIT" if ok else "MISS"


# ═══════════════════════════════════════════════════════════════════════════
# Loading: judgments, qrels, cells (batch6 universe)
# ═══════════════════════════════════════════════════════════════════════════

def qrels_sha(qrels: dict, ds: str) -> tuple[str, int]:
    lines = sorted(f"{q}\t{d}\t{int(g)}" for (dd, q, d), g in qrels.items()
                   if dd == ds)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest(), \
        len(lines)


def flagship_file(data_dir: Path, ds: str) -> Path:
    return data_dir / "inputs" / f"pairs_{ds}_flagship40.jsonl"


def input_shas(ctx: dict) -> dict:
    out = {"judgments": ctx["jsha"]}
    for ds in ctx["datasets"]:
        out[f"qrels:{ds}"] = ctx["qshas"][ds][0]
    for ds, s in ctx["fshas"].items():
        out[f"flagship40:{ds}"] = s
    return out


def load_inputs(args, datasets) -> dict:
    data_dir = b5.resolve_data_dir(args.data_dir)
    jpath = Path(args.judgments) if args.judgments else \
        data_dir / "derived" / "judgments.parquet"
    if not jpath.exists():
        sys.exit(f"ERROR: {jpath} not found. Run src/parse.py first.")
    jsha = b6.sha256_file(jpath)
    judges, cheap, flagships = roster(args.judges, args.cheap, args.flagships)
    df = b6.load_judgments(jpath)
    df = df[df.dataset.astype(str).isin(datasets)
            & df.model_id.isin(judges)]
    qrels = b6.load_human(datasets, args.qrels_json)
    qshas = {ds: qrels_sha(qrels, ds) for ds in datasets}
    for ds in datasets:
        if qshas[ds][1] == 0:
            sys.exit(f"ERROR: no qrels loaded for {ds}.")
    present = {ds: set(df.model_id[df.dataset.astype(str) == ds].unique())
               for ds in datasets}
    fshas = {}
    for ds in datasets:
        if present[ds] & set(flagships):
            fp = flagship_file(data_dir, ds)
            if not fp.exists():
                sys.exit(f"ERROR: flagship pairs file {fp} missing.")
            fshas[ds] = b6.sha256_file(fp)
    return {"data_dir": data_dir, "jpath": jpath, "jsha": jsha, "df": df,
            "qrels": qrels, "qshas": qshas, "fshas": fshas,
            "judges": judges, "cheap": cheap, "flagships": flagships,
            "datasets": list(datasets), "present": present}


def build_cells(ctx: dict, seed: int = SEED) -> dict:
    """(ds, model) -> cell (batch6.prepare_cell universe) or {'error': ...}."""
    df, qrels = ctx["df"], ctx["qrels"]
    G = np.arange(4, dtype=float)
    cells: dict = {}
    frozen_pairs: dict = {}
    for ds in ctx["datasets"]:
        dds = df[df.dataset.astype(str) == ds]
        qcount = {}
        for (dd, q, _d) in qrels:
            if dd == ds:
                qcount[q] = qcount.get(q, 0) + 1
        for m in ctx["judges"]:
            if m not in ctx["present"][ds]:
                cells[(ds, m)] = {"model": m, "dataset": ds,
                                  "error": "absent (no rows)"}
                continue
            d, restricted = b6.scope_rows(dds, m, ds, ctx["flagships"],
                                          ctx["data_dir"], frozen_pairs)
            c = b6.prepare_cell(m, ds, d, qrels, seed)
            if "error" in c:
                cells[(ds, m)] = {"model": m, "dataset": ds,
                                  "error": c["error"], "info": c.get("info"),
                                  "flagship": restricted}
                continue
            cnt = c["cnt"]                                # (5, P, 4)
            nrec = cnt.sum(axis=2)
            means = (cnt @ G) / nrec                      # (5, P)
            qnames = sorted(set(c["pair_qid"]))
            cells[(ds, m)] = {
                "model": m, "dataset": ds, "flagship": restricted,
                "cheap": m in ctx["cheap"], "universe": c["universe"],
                "h": c["h"], "pair_q": c["pair_q"], "qnames": qnames,
                "n_queries": c["n_queries"], "means": means,
                "raw": means[2],
                "raw_lab": b6.labels_from_counts(cnt[2], b4.IDENTITY_MAP),
                "nrec": nrec,
                "qdepth": np.array([qcount.get(q, 0) for q in qnames],
                                   dtype=float),
                "info": c["info"]}
    return cells


def check_universe(ctx: dict, cells: dict, keys_only: bool) -> dict:
    """Same universe as batch6/DE where derived/batch6_labels_<ds> exists."""
    status = {}
    for ds in ctx["datasets"]:
        p = None
        base = ctx["data_dir"] / "derived" / f"batch6_labels_{ds}"
        for suf in (".parquet", ".pkl"):
            if base.with_suffix(suf).exists():
                p = base.with_suffix(suf)
                break
        if p is None:
            status[ds] = "not checked (no batch6_labels file)"
            continue
        cols = ["model_id", "qid", "docid"] + ([] if keys_only else
                                               ["human", "RAW"])
        try:
            st = pd.read_parquet(p, columns=cols) if p.suffix == ".parquet" \
                else pd.read_pickle(p)[cols]
        except ImportError:
            status[ds] = "not checked (no parquet engine)"
            continue
        done = []
        for m in sorted(set(st.model_id)):
            c = cells.get((ds, m))
            if c is None or "error" in c:
                continue
            s = st[st.model_id == m]
            keys = list(zip(s.qid.astype(str), s.docid.astype(str)))
            if keys != list(c["universe"]):
                sys.exit(f"ERROR: '{m}' @ {ds}: pair universe differs from "
                         f"{p.name} ({len(keys)} vs {len(c['universe'])}).")
            if not keys_only and not (
                    np.array_equal(s.human.to_numpy(np.int64), c["h"])
                    and np.array_equal(s.RAW.to_numpy(np.int64),
                                       c["raw_lab"])):
                sys.exit(f"ERROR: '{m}' @ {ds}: human / RAW differ from "
                         f"{p.name}.")
            done.append(short(m))
        status[ds] = (f"identical ({'keys' if keys_only else 'keys, human, RAW'}"
                      f") for {len(done)} judge(s): {', '.join(done)}"
                      if done else "not checked (no judge in common)")
    return status


# ═══════════════════════════════════════════════════════════════════════════
# Units (counts only — paste-safe)
# ═══════════════════════════════════════════════════════════════════════════

def _quant(x) -> list:
    x = np.asarray(x, dtype=float)
    return [float(v) for v in np.percentile(x, [0, 25, 50, 75, 100])]


def cell_units(c: dict) -> dict:
    """Counts only.  Never touches c['means'] / c['raw'] / c['raw_lab']."""
    if "error" in c:
        return {"error": c["error"], "info": c.get("info")}
    depth = np.bincount(c["pair_q"], minlength=c["n_queries"])
    return {"n_pairs": int(len(c["h"])), "n_queries": int(c["n_queries"]),
            "flagship40": bool(c["flagship"]), "cheap": bool(c["cheap"]),
            "depth_q": _quant(depth),
            "depth_share_at_max": float((depth == depth.max()).mean()),
            "qrels_depth_q": _quant(c["qdepth"]),
            "n_by_grade": [int((c["h"] == g).sum()) for g in GRADES],
            "info": c["info"]}


def e_units(cells: dict, ds: str, judges, cheap) -> dict:
    cj = [m for m in judges if m in cheap and "error" not in
          cells.get((ds, m), {"error": 1})]
    if len(cj) < 2:
        return {"judges": cj, "n_common": 0, "n_by_grade": [0, 0, 0, 0]}
    common = set(cells[(ds, cj[0])]["universe"])
    for m in cj[1:]:
        common &= set(cells[(ds, m)]["universe"])
    c0 = cells[(ds, cj[0])]
    hmap = dict(zip(c0["universe"], c0["h"].tolist()))
    hs = np.array([hmap[k] for k in common], dtype=np.int64)
    return {"judges": cj, "n_common": len(common),
            "n_by_grade": [int((hs == g).sum()) for g in GRADES]}


def defined_counts(units: dict, eu: dict, datasets) -> dict:
    """Number of cells whose criterion is defined, from counts alone."""
    n = {(k, f): 0 for k in CRITS for f in FAMILIES}
    for (ds, _m), u in units.items():
        if "error" in u:
            continue
        f = FAMILY_OF[ds]
        g = u["n_by_grade"]
        if min(g[0], g[1], g[2]) >= A_MIN_N:
            n[("A1", f)] += 1
        if min(g[1], g[2]) >= A_MIN_N:
            n[("A2", f)] += 1
        ok3 = u["n_queries"] >= 3
        n[("B", f)] += 1 if ok3 and u["depth_q"][0] < u["depth_q"][4] else 0
        n[("Bq", f)] += 1 if ok3 and \
            u["qrels_depth_q"][0] < u["qrels_depth_q"][4] else 0
    for ds in datasets:
        if len(eu[ds]["judges"]) >= 2 and eu[ds]["n_by_grade"][0] > 0:
            n[("E", FAMILY_OF[ds])] += 1
    return n


# ═══════════════════════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════════════════════

def boot_weights(nq: int, B: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """batch6.boot_indices (the single resampling source) -> (idx, weights)."""
    idx = b6.boot_indices(nq, B, seed)
    W = np.zeros((B, nq))
    np.add.at(W, (np.repeat(np.arange(B), nq), idx.ravel()), 1.0)
    return idx, W


def ratio_boot(W: np.ndarray, num: np.ndarray, den: np.ndarray) -> np.ndarray:
    a, b = W @ num, W @ den
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b != 0, a / np.where(b != 0, b, 1.0), np.nan)


def _rank(x: np.ndarray) -> np.ndarray:
    try:
        from scipy.stats import rankdata
        return rankdata(x)
    except ImportError:                                        # pragma: no cover
        return pd.Series(x).rank(method="average").to_numpy()


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def pb_pmf(p: np.ndarray) -> np.ndarray:
    """Poisson-binomial pmf; p (..., m) -> (..., m+1)."""
    p = np.asarray(p, dtype=float)
    shp = p.shape[:-1]
    m = p.shape[-1]
    pmf = np.zeros(shp + (m + 1,))
    pmf[..., 0] = 1.0
    for j in range(m):
        pj = p[..., j:j + 1]
        new = pmf * (1.0 - pj)
        new[..., 1:] += pmf[..., :-1] * pj
        pmf = new
    return pmf


def tail(pmf: np.ndarray) -> np.ndarray:
    """P(c >= k) for k = 0..m along the last axis."""
    return np.flip(np.cumsum(np.flip(pmf, -1), -1), -1)


def sgn(x) -> int:
    return 0 if x is None or x == 0 else (1 if x > 0 else -1)


# ═══════════════════════════════════════════════════════════════════════════
# A — conditional offset by human grade
# ═══════════════════════════════════════════════════════════════════════════

def analysis_A(c: dict, B: int, seed: int) -> dict:
    h, raw, lab, q, nq = c["h"], c["raw"], c["raw_lab"], c["pair_q"], \
        c["n_queries"]
    _idx, W = boot_weights(nq, B, seed)
    diff, dlab = raw - h, lab - h
    out = {"by_grade": {}}
    S_all = np.bincount(q, weights=diff, minlength=nq)
    N_all = np.bincount(q, minlength=nq).astype(float)
    for g in GRADES:
        msk = (h == g).astype(float)
        n = int(msk.sum())
        S = np.bincount(q, weights=diff * msk, minlength=nq)
        Sl = np.bincount(q, weights=dlab * msk, minlength=nq)
        R = np.bincount(q, weights=raw * msk, minlength=nq)
        N = np.bincount(q, weights=msk, minlength=nq)
        if n == 0:
            out["by_grade"][g] = {"n": 0}
            continue
        out["by_grade"][g] = {
            "n": n, "mean_raw": float(R.sum() / n),
            "mean_raw_ci": ci90(ratio_boot(W, R, N)),
            "off": float(S.sum() / n), "off_ci": ci90(ratio_boot(W, S, N)),
            "off_lab": float(Sl.sum() / n),
            "off_lab_ci": ci90(ratio_boot(W, Sl, N))}
    out["n"] = int(len(h))
    out["delta"] = float(diff.mean())
    out["delta_ci"] = ci90(ratio_boot(W, S_all, N_all))
    out["delta_lab"] = float(dlab.mean())
    m0 = (h == 0).astype(float)
    S0 = np.bincount(q, weights=diff * m0, minlength=nq)
    tot = float(S_all.sum())
    out["mass_share_h0"] = float(S0.sum() / tot) if tot != 0 else None
    out["mass_share_h0_ci"] = ci90(ratio_boot(W, S0, S_all))
    out["pair_share_h0"] = float(m0.mean())
    out.update(a_criteria(out))
    return out


def a_criteria(a: dict) -> dict:
    bg = a["by_grade"]
    n = [bg[g]["n"] for g in GRADES]
    off = [bg[g].get("off") for g in GRADES]
    a1 = a2 = None
    if min(n[0], n[1], n[2]) >= A_MIN_N:
        s0 = sgn(off[0])
        a1 = bool(s0 != 0 and sgn(off[1]) == s0 and sgn(off[2]) == s0
                  and min(abs(off[1]), abs(off[2])) >= abs(off[0]) * A_FRAC)
    if min(n[1], n[2]) >= A_MIN_N:
        sd = sgn(a["delta"])
        a2 = bool(sd != 0 and sgn(off[1]) == sd and sgn(off[2]) == sd
                  and min(abs(off[1]), abs(off[2])) >= abs(a["delta"]) * A_FRAC)
    return {"A1_pass": a1, "A2_pass": a2}


# ═══════════════════════════════════════════════════════════════════════════
# B — offset vs judging depth
# ═══════════════════════════════════════════════════════════════════════════

def depth_stats(depth: np.ndarray, oq: np.ndarray, idx: np.ndarray,
                W: np.ndarray) -> dict:
    rho = spearman(depth, oq)
    rb = np.array([spearman(depth[i], oq[i]) for i in idx]) \
        if np.isfinite(rho) else np.array([np.nan])
    med = float(np.median(depth))
    sh = depth <= med
    if sh.all():                  # > half the queries at the max (capped)
        sh = depth < med
    res = {"rho": fnum(rho), "rho_ci": ci90(rb),
           "rho_boot_defined": float(np.isfinite(rb).mean()),
           "median_depth": med, "n_shallow": int(sh.sum()),
           "n_deep": int((~sh).sum())}
    if sh.all() or not sh.any():
        res.update({"off_shallow": None, "off_deep": None, "split_diff": None,
                    "split_diff_ci": [None, None]})
    else:
        s1, s0 = sh.astype(float), (~sh).astype(float)
        ms = ratio_boot(W, oq * s1, s1)
        md = ratio_boot(W, oq * s0, s0)
        res.update({"off_shallow": float(oq[sh].mean()),
                    "off_deep": float(oq[~sh].mean()),
                    "split_diff": float(oq[sh].mean() - oq[~sh].mean()),
                    "split_diff_ci": ci90(ms - md)})
    rho_ok = res["rho"] is not None
    sd = res["split_diff"]
    res["pass"] = (bool(abs(res["rho"]) <= B_RHO_BAND and abs(sd) < B_SPLIT_TOL)
                   if rho_ok and sd is not None else None)
    return res


def analysis_B(c: dict, B: int, seed: int) -> dict:
    h, raw, q, nq = c["h"], c["raw"], c["pair_q"], c["n_queries"]
    idx, W = boot_weights(nq, B, seed)
    depth = np.bincount(q, minlength=nq).astype(float)
    oq = np.bincount(q, weights=raw - h, minlength=nq) / depth
    return {"n_queries": int(nq), "depth_q": _quant(depth),
            "qrels_depth_q": _quant(c["qdepth"]),
            "B": depth_stats(depth, oq, idx, W),
            "Bq": depth_stats(c["qdepth"], oq, idx, W)}


# ═══════════════════════════════════════════════════════════════════════════
# D — tone levels on the identical universe (C = the L3 row of A / D)
# ═══════════════════════════════════════════════════════════════════════════

def analysis_D(c: dict, B: int, seed: int) -> dict:
    h, q, nq = c["h"], c["pair_q"], c["n_queries"]
    _idx, W = boot_weights(nq, B, seed)
    N = np.bincount(q, minlength=nq).astype(float)
    S = {l: np.bincount(q, weights=c["means"][l - 1] - h, minlength=nq)
         for l in LEVELS}
    out = {"n": int(len(h)), "levels": {}}
    for l in LEVELS:
        out["levels"][l] = {
            "off": float(S[l].sum() / N.sum()),
            "off_ci": ci90(ratio_boot(W, S[l], N)),
            "D": float((S[l].sum() - S[3].sum()) / N.sum()),
            "D_ci": ci90(ratio_boot(W, S[l] - S[3], N))}
    offs = [out["levels"][l]["off"] for l in LEVELS]
    out["range"] = float(max(offs) - min(offs))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# E — agreement among cheap judges on h = 0 pairs (EXPLORATORY)
# ═══════════════════════════════════════════════════════════════════════════

def e_matrix(cells_ds: list[dict]) -> tuple:
    common = set(cells_ds[0]["universe"])
    for c in cells_ds[1:]:
        common &= set(c["universe"])
    common = sorted(common)
    R = np.zeros((len(common), len(cells_ds)), dtype=bool)
    h = None
    for j, c in enumerate(cells_ds):
        pos = {k: i for i, k in enumerate(c["universe"])}
        ii = np.array([pos[k] for k in common], dtype=np.int64)
        R[:, j] = c["raw_lab"][ii] >= REL
        hj = c["h"][ii]
        if h is None:
            h = hj
        elif not np.array_equal(h, hj):
            sys.exit("ERROR: E: human grades differ across judges on the "
                     "common pairs — qrels join inconsistent.")
    return common, R, h


def e_stratum(R: np.ndarray, qcode: np.ndarray, nq: int, B: int,
              seed: int) -> dict:
    n, m = R.shape
    if n == 0:
        return {"n": 0, "pass": None}
    c = R.sum(axis=1)
    p = R.mean(axis=0)
    obs = np.array([(c >= k).mean() for k in range(m + 1)])
    ind = tail(pb_pmf(p))
    res = {"n": int(n), "m": int(m), "p": p.tolist(),
           "obs_ge": obs.tolist(), "ind_ge": ind.tolist()}
    with np.errstate(divide="ignore", invalid="ignore"):
        cond_obs = obs / obs[1] if obs[1] > 0 else np.full(m + 1, np.nan)
        cond_ind = ind / ind[1] if ind[1] > 0 else np.full(m + 1, np.nan)
        lift = cond_obs / cond_ind
    res["obs_ge_given_ge1"] = [fnum(x) for x in cond_obs]
    res["ind_ge_given_ge1"] = [fnum(x) for x in cond_ind]
    res["lift_ge"] = [fnum(x) for x in lift]
    pl = []
    for i, j in combinations(range(m), 2):
        if p[i] > 0 and p[j] > 0:
            pl.append(float((R[:, i] & R[:, j]).mean() / (p[i] * p[j])))
    res["pairwise_lift_mean"] = float(np.mean(pl)) if pl else None
    # query bootstrap of lift_E_K
    _idx, W = boot_weights(nq, B, seed)
    Nq = np.bincount(qcode, minlength=nq).astype(float)
    Sj = np.stack([np.bincount(qcode, weights=R[:, j].astype(float),
                               minlength=nq) for j in range(m)], axis=1)
    Ck = np.stack([np.bincount(qcode, weights=(c >= k).astype(float),
                               minlength=nq) for k in range(m + 1)], axis=1)
    den = W @ Nq
    with np.errstate(divide="ignore", invalid="ignore"):
        pB = (W @ Sj) / den[:, None]
        oB = (W @ Ck) / den[:, None]
        iB = tail(pb_pmf(pB))
        lb = (oB[:, E_K] / oB[:, 1]) / (iB[:, E_K] / iB[:, 1]) \
            if m >= E_K else np.full(B, np.nan)
    res["lift_k"] = E_K
    res["lift_k_ci"] = ci90(lb)
    lk = res["lift_ge"][E_K] if m >= E_K else None
    res["pass"] = (bool(lk <= E_LIFT_MAX) if lk is not None else None)
    return res


def analysis_E(cells_ds: list[dict], B: int, seed: int) -> dict:
    if len(cells_ds) < 2:
        return {"judges": [c["model"] for c in cells_ds], "pass": None,
                "note": "fewer than 2 cheap judges"}
    common, R, h = e_matrix(cells_ds)
    qid = [k[0] for k in common]
    out = {"judges": [c["model"] for c in cells_ds], "n_common": len(common),
           "strata": {}}
    for g in E_STRATA:
        sel = h == g
        qs = [q for q, s in zip(qid, sel) if s]
        qn = sorted(set(qs))
        code = {q: i for i, q in enumerate(qn)}
        qc = np.array([code[q] for q in qs], dtype=np.int64)
        out["strata"][g] = e_stratum(R[sel], qc, len(qn), B, seed)
        out["strata"][g]["n_queries"] = len(qn)
    out["pass"] = out["strata"][0]["pass"]
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Summary rows / printing
# ═══════════════════════════════════════════════════════════════════════════

def row(analysis, ds, judge, key, value, ci=None, n=None) -> dict:
    ci = ci or [None, None]
    if isinstance(value, bool):
        value = float(value)
    return {"analysis": analysis, "dataset": ds, "judge": judge, "key": key,
            "value": fnum(value) if value is not None else None,
            "ci_lo": fnum(ci[0]), "ci_hi": fnum(ci[1]), "n": n}


def _fmt_csv(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return "" if not np.isfinite(v) else f"{v:.6f}"
    return str(v)


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLS)
        for r in rows:
            w.writerow([_fmt_csv(r[c]) for c in CSV_COLS])
    say(f"[csv] wrote {path} ({len(rows)} rows)")


def jlabel(c_or_u: dict, model: str) -> str:
    return short(model) + (" [40%]" if c_or_u.get("flagship") or
                           c_or_u.get("flagship40") else "")


def print_A(res: dict, cells: dict, datasets, judges) -> None:
    say("\n" + "=" * 100)
    say("A — conditional offset by human grade  (off(h) = mean(RAW − h), "
        "RAW = L3 record mean; 90% query-bootstrap CI)")
    say("=" * 100)
    for ds in datasets:
        say(f"\n  {ds}  ({FAMILY_OF[ds]})")
        say(f"    {'judge':<26} {'n0/n1/n2/n3':<18} {'off0':>22} "
            f"{'off1':>22} {'off2':>22} {'off3':>7} {'Δ':>7} "
            f"{'mass0':>6} {'pair0':>6}  A1 A2")
        for m in judges:
            a = res.get((ds, m))
            if a is None:
                continue
            bg = a["by_grade"]
            ns = "/".join(str(bg[g]["n"]) for g in GRADES)

            def o(g):
                if bg[g]["n"] == 0:
                    return "n/a"
                return f"{_f(bg[g]['off'])} {_ci(bg[g]['off_ci'])}"

            def yn(x):
                return "-" if x is None else ("Y" if x else "N")
            say(f"    {jlabel(cells[(ds, m)], m):<26} {ns:<18} {o(0):>22} "
                f"{o(1):>22} {o(2):>22} "
                f"{_f(bg[3].get('off')) if bg[3]['n'] else 'n/a':>7} "
                f"{_f(a['delta']):>7} {_f(a['mass_share_h0'], 2, False):>6} "
                f"{_f(a['pair_share_h0'], 2, False):>6}  "
                f"{yn(a['A1_pass']):>2} {yn(a['A2_pass']):>2}")
    say("\n  (RAW_lab = round-half-up RAW offsets are in ic_A.json / "
        "ic_summary.csv as off_lab_h*)")


def print_B(res: dict, cells: dict, datasets, judges) -> None:
    say("\n" + "=" * 100)
    say("B — per-query offset vs judging depth  (Spearman ρ, query bootstrap "
        "90% CI; split at median depth: shallow − deep)")
    say("=" * 100)
    for ds in datasets:
        say(f"\n  {ds}")
        say(f"    {'judge':<26} {'nq':>3} {'depth med':>9} {'ρ(univ)':>22} "
            f"{'split Δ(univ)':>22} {'qrels med':>9} {'ρ(qrels)':>22} "
            f"{'split Δ(qrels)':>22}  B Bq")
        for m in judges:
            b = res.get((ds, m))
            if b is None:
                continue

            def yn(x):
                return "-" if x is None else ("Y" if x else "N")
            u, qq = b["B"], b["Bq"]
            say(f"    {jlabel(cells[(ds, m)], m):<26} {b['n_queries']:>3} "
                f"{b['depth_q'][2]:>9.0f} "
                f"{_f(u['rho']) + ' ' + _ci(u['rho_ci']):>22} "
                f"{_f(u['split_diff']) + ' ' + _ci(u['split_diff_ci']):>22} "
                f"{b['qrels_depth_q'][2]:>9.0f} "
                f"{_f(qq['rho']) + ' ' + _ci(qq['rho_ci']):>22} "
                f"{_f(qq['split_diff']) + ' ' + _ci(qq['split_diff_ci']):>22}"
                f"  {yn(u['pass'])} {yn(qq['pass']):>2}")


def print_C(resA: dict, cells: dict, datasets, judges) -> None:
    say("\n" + "=" * 100)
    say("C — mean offset Δ = mean(RAW − h) per dataset  [NON-BLIND: already "
        "published; reported, not scored]")
    say("=" * 100)
    say("    " + f"{'judge':<26}" + "".join(f"{ds:>10}" for ds in datasets))
    for m in judges:
        vals = [resA.get((ds, m)) for ds in datasets]
        if all(v is None for v in vals):
            continue
        fl = any(cells.get((ds, m), {}).get("flagship") for ds in datasets)
        say("    " + f"{short(m) + (' [40%]' if fl else ''):<26}"
            + "".join(f"{_f(v['delta']) if v else '—':>10}" for v in vals))


def print_D(res: dict, cells: dict, datasets, judges) -> None:
    say("\n" + "=" * 100)
    say("D — offset at each tone level on the identical pair universe  "
        "[NON-BLIND: already published; reported, not scored]")
    say("=" * 100)
    for ds in datasets:
        say(f"\n  {ds}")
        say(f"    {'judge':<26}" + "".join(f"{'L' + str(l):>9}" for l in LEVELS)
            + f"{'range':>9}")
        for m in judges:
            d = res.get((ds, m))
            if d is None:
                continue
            say(f"    {jlabel(cells[(ds, m)], m):<26}"
                + "".join(f"{_f(d['levels'][l]['off']):>9}" for l in LEVELS)
                + f"{_f(d['range'], 3, False):>9}")


def print_E(res: dict, datasets) -> None:
    say("\n" + "=" * 100)
    say("E — EXPLORATORY: cheap-judge agreement on human-grade-0 pairs "
        "(R_j = RAW_lab ≥ 2; lift = observed / independence)")
    say("=" * 100)
    for ds in datasets:
        e = res.get(ds)
        if e is None or "strata" not in e:
            say(f"  {ds}: n/a ({(e or {}).get('note', 'no cells')})")
            continue
        say(f"\n  {ds}: {len(e['judges'])} cheap judges "
            f"({', '.join(short(j) for j in e['judges'])}); "
            f"{e['n_common']} common pairs")
        for g, s in e["strata"].items():
            if s["n"] == 0:
                say(f"    h={g}: no pairs")
                continue
            say(f"    h={g}{' (reference)' if g != 0 else ''}: n={s['n']} "
                f"({s['n_queries']} q)  p_j=" + ",".join(
                    f"{x:.3f}" for x in s["p"]))
            ks = range(1, s["m"] + 1)
            say("      P(c≥k|c≥1) obs: " + "  ".join(
                f"k={k}:{_f(s['obs_ge_given_ge1'][k], 3, False)}" for k in ks))
            say("      P(c≥k|c≥1) ind: " + "  ".join(
                f"k={k}:{_f(s['ind_ge_given_ge1'][k], 3, False)}" for k in ks))
            say("      lift_k        : " + "  ".join(
                f"k={k}:{_f(s['lift_ge'][k], 2, False)}" for k in ks)
                + f"   lift_{s['lift_k']} CI {_ci(s['lift_k_ci'], 2)}  "
                f"pairwise lift {_f(s['pairwise_lift_mean'], 2, False)}"
                + (f"  E {'PASS' if s['pass'] else 'FAIL'}"
                   if g == 0 and s["pass"] is not None else ""))


# ═══════════════════════════════════════════════════════════════════════════
# paste
# ═══════════════════════════════════════════════════════════════════════════

def paste_lines(ctx: dict, units: dict, eu: dict, ustatus: dict) -> list[str]:
    datasets, judges = ctx["datasets"], ctx["judges"]
    head = b4.git_hash()
    dirty = b6.code_dirty()
    L = [f"# {IC_MARKER}（scripts/incompleteness_check.py）", ""]
    L.append(f"登记时间：____。分析代码 git hash：{head}"
             + ("（⚠ scripts/ 或 src/ 有未提交改动 — 先 commit 代码再登记）"
                if dirty else "") + "。")
    L.append("顺序：selftest → paste → 本块填写并 commit → run --unseal-dl22。"
             "只用已有数据，无 API 调用。")
    L.append("问题：MS MARCO / TREC DL 上 judge 偏宽，是 qrels 漏标正例（数据集"
             "伪影）还是 judge 尺度平移（相对这些标注者工作点的错位）？")
    L.append("")
    L.append("## 输入 sha256（run 时逐个核对，不符或缺失即中止）")
    for name, sha in input_shas(ctx).items():
        extra = ""
        if name.startswith("qrels:"):
            dsn = name.split(":", 1)[1]
            extra = (f"（{ctx['qshas'][dsn][1]} 条，规范化 "
                     f"\"qid\\tdocid\\tgrade\" 排序行"
                     + ("，grade 已 −1 映射到 0-3" if dsn == "antique" else "")
                     + "）")
        elif name == "judgments":
            extra = f"（{ctx['jpath'].name}）"
        L.append(f"- sha256[{name}]: {sha}{extra}")
    L.append("")
    L.append("## 定义（逐字；代码同文打印）")
    L.append("- universe = batch6.prepare_cell：人工判过 且 judge 在 L1..L5 每档"
             "都有可用标签的对（B6/DE 同一 universe）；flagship 限其冻结 40% "
             "样本（标 [40%]）。")
    L.append("- RAW = L3 全部可用记录（paraphrase × run）的均值（连续，Part I 的 "
             "Δ 估计量）；RAW_lab = round-half-up(RAW)（E 的相关 = RAW_lab ≥ 2）。"
             f"bootstrap：按 query，B={B_BOOT}，seed {SEED}，90% 百分位 CI。")
    for k in CRITS:
        L.append(f"- 判据 {k}：{CRITERIA[k]}")
    L.append("- 解读：")
    L.extend(f"  - {n}" for n in NOTES)
    L.append("")
    L.append("## 判定纪律（逐字；代码同文打印）")
    L.extend(ADJUDICATION_RULES)
    L.append("")
    L.append("## 单元（只含计数：对数、query 数、判深分位、各人工档对数；"
             "不含任何 judge 分数或一致性数值）")
    for ds in datasets:
        L.append(f"- {ds}（{FAMILY_OF[ds]}；universe 核对：{ustatus.get(ds)}）")
        for m in judges:
            u = units[(ds, m)]
            if "error" in u:
                if u["error"].startswith("absent"):
                    continue
                L.append(f"  - {short(m)}: 无单元（{u['error']}；"
                         f"{(u.get('info') or {}).get('n_usable_by_level')}）")
                continue
            dq = "/".join(f"{x:.0f}" for x in u["depth_q"])
            qq = "/".join(f"{x:.0f}" for x in u["qrels_depth_q"])
            inf = u["info"]
            L.append(f"  - {short(m)}{' [40%]' if u['flagship40'] else ''}"
                     f"{' (cheap)' if u['cheap'] else ''}: "
                     f"{u['n_pairs']} 对 / {u['n_queries']} q；h=0/1/2/3: "
                     + "/".join(str(x) for x in u["n_by_grade"])
                     + f"；universe 判深 min/q25/med/q75/max {dq}（"
                     f"{u['depth_share_at_max']:.0%} query 在最大值）；qrels 判深 "
                     f"{qq}；剔除不可用 {inf['n_dropped_unusable']}，"
                     f"LLM-only {inf['n_llm_only']}")
        e = eu[ds]
        L.append(f"  - E：cheap judge {len(e['judges'])} 个（"
                 + ", ".join(short(j) for j in e["judges"])
                 + f"），共同对 {e['n_common']}，其中 h=0 "
                 f"{e['n_by_grade'][0]}、h=1 {e['n_by_grade'][1]}")
    nd = defined_counts(units, eu, datasets)
    L.append("")
    L.append("## 盲态声明")
    L.append("- A（按人工档条件偏移）：____（建议：blind — 条件偏移从未计算；"
             "总体 Δ 已发表）")
    L.append("- B（偏移 vs 判深）：____（建议：blind）")
    L.append("- E（cheap judge 在 h=0 上的一致率）：____（建议：blind；探索性）")
    L.append("- C、D：non-blind（已发表），只在 run 中打印，不计分。")
    L.append("")
    L.append("## 预测（键 = IC[<判据>|<家族>]；token ≥k/N 或 ≤k/N = 满足判据的"
             "单元数下/上限，N = 判据有定义的单元数（下方已按计数算好）；可增删"
             "行，格式不变）")
    texts = {
        "A1": "off(h=1)、off(h=2) 与 off(h=0) 同号且 min(|off1|,|off2|) ≥ "
              "|off0|/3",
        "A2": "off(h=1)、off(h=2) 与总体 Δ 同号且 min(|off1|,|off2|) ≥ |Δ|/3"
              "（地板/天花板稳健版）",
        "B": "ρ(universe 判深, 每 query 偏移) ∈ [−0.3, +0.3] 且中位数切分差 "
             "< 0.1 grade",
        "Bq": "同 B，判深 = 该 query 的 qrels 条数（未截断）",
        "E": "h=0 对上 cheap judge 一致率 lift_2 = P(c≥2|c≥1)/独立期望 ≤ 2"
             "（每 dataset 一个单元；探索性）",
    }
    for k in CRITS:
        for fam in FAMILIES:
            if not any(FAMILY_OF[d] == fam for d in datasets):
                continue
            note = ""
            if k == "A1" and fam == "antique":
                note = "（ANTIQUE Δ<0：A1 结构性不可满足，建议删去此行或预测 ≤0）"
            L.append(f"- IC[{k}|{fam}] {texts[k]}；N={nd[(k, fam)]}{note}"
                     f"｜预测：____")
    L.append("- 开箱结果（run 后回填）：")
    return L


def cmd_paste(args) -> None:
    datasets = resolve_datasets(args.datasets)
    ctx = load_inputs(args, datasets)
    cells = build_cells(ctx, args.seed)
    units = {k: cell_units(c) for k, c in cells.items()}
    eu = {ds: e_units(cells, ds, ctx["judges"], ctx["cheap"])
          for ds in datasets}
    ustatus = check_universe(ctx, cells, keys_only=True)
    lines = paste_lines(ctx, units, eu, ustatus)
    print("PASTE INTO paper/PREDICTIONS.md  (fill every ____, commit, THEN run;"
          " no judge score entered anything below)")
    print("-" * 96)
    print("\n".join(lines))
    print("-" * 96)


# ═══════════════════════════════════════════════════════════════════════════
# run
# ═══════════════════════════════════════════════════════════════════════════

def run_analyses(ctx: dict, cells: dict, B: int, seed: int) -> dict:
    datasets, judges = ctx["datasets"], ctx["judges"]
    valid = [(ds, m) for ds in datasets for m in judges
             if "error" not in cells[(ds, m)]]
    A = {k: analysis_A(cells[k], B, seed) for k in valid}
    Bres = {k: analysis_B(cells[k], B, seed) for k in valid}
    D = {k: analysis_D(cells[k], B, seed) for k in valid}
    E = {}
    for ds in datasets:
        cj = [cells[(ds, m)] for m in judges
              if m in ctx["cheap"] and (ds, m) in valid]
        E[ds] = analysis_E(cj, B, seed)
    return {"A": A, "B": Bres, "D": D, "E": E, "valid": valid}


def tallies(res: dict, datasets) -> dict:
    """(criterion, family) -> list of (cell id, pass) over DEFINED cells."""
    t = {(k, f): [] for k in CRITS for f in FAMILIES}
    for (ds, m) in res["valid"]:
        f = FAMILY_OF[ds]
        a, b = res["A"][(ds, m)], res["B"][(ds, m)]
        for k, v in (("A1", a["A1_pass"]), ("A2", a["A2_pass"]),
                     ("B", b["B"]["pass"]), ("Bq", b["Bq"]["pass"])):
            if v is not None:
                t[(k, f)].append((f"{ds}|{m}", v))
    for ds in datasets:
        v = res["E"][ds].get("pass")
        if v is not None:
            t[("E", FAMILY_OF[ds])].append((ds, v))
    return t


def verdicts(t: dict, reg: dict | None) -> list[dict]:
    out = []
    preds = (reg or {}).get("preds", {})
    for k in CRITS:
        for f in FAMILIES:
            cl = t[(k, f)]
            if not cl and f"{k}|{f}" not in preds:
                continue
            n_pass = sum(1 for _c, v in cl if v)
            p = preds.get(f"{k}|{f}")
            out.append({"key": f"{k}|{f}", "n_pass": n_pass, "n_def": len(cl),
                        "failing": [c for c, v in cl if not v],
                        "prediction": (f"{p[0]}{p[1]}/{p[2]}" if p else None),
                        "verdict": (adjudicate(p, n_pass, len(cl)) if p
                                    else "unregistered")})
    for key in preds:
        if key.split("|")[0] not in CRITS or \
                (len(key.split("|")) > 1 and key.split("|")[1] not in FAMILIES):
            out.append({"key": key, "n_pass": None, "n_def": None,
                        "failing": [], "prediction": None,
                        "verdict": "unknown criterion/family key"})
    return out


def summary_rows(ctx, cells, res) -> list[dict]:
    rows = []
    for (ds, m), c in cells.items():
        u = cell_units(c)
        if "error" in u:
            continue
        rows += [row("units", ds, m, "n_pairs", u["n_pairs"]),
                 row("units", ds, m, "n_queries", u["n_queries"]),
                 row("units", ds, m, "flagship_40pct", u["flagship40"]),
                 row("units", ds, m, "depth_median", u["depth_q"][2],
                     n=u["n_queries"]),
                 row("units", ds, m, "qrels_depth_median",
                     u["qrels_depth_q"][2], n=u["n_queries"])]
    for (ds, m), a in res["A"].items():
        for g in GRADES:
            bg = a["by_grade"][g]
            n = bg["n"]
            rows.append(row("A", ds, m, f"n_h{g}", n, n=n))
            if n:
                rows += [row("A", ds, m, f"mean_raw_h{g}", bg["mean_raw"],
                             bg["mean_raw_ci"], n),
                         row("A", ds, m, f"off_h{g}", bg["off"], bg["off_ci"],
                             n),
                         row("A", ds, m, f"off_lab_h{g}", bg["off_lab"],
                             bg["off_lab_ci"], n)]
        rows += [row("A", ds, m, "delta", a["delta"], a["delta_ci"], a["n"]),
                 row("A", ds, m, "mass_share_h0", a["mass_share_h0"],
                     a["mass_share_h0_ci"], a["n"]),
                 row("A", ds, m, "pair_share_h0", a["pair_share_h0"],
                     n=a["n"]),
                 row("A", ds, m, "A1_pass", a["A1_pass"], n=a["n"]),
                 row("A", ds, m, "A2_pass", a["A2_pass"], n=a["n"])]
    for (ds, m), b in res["B"].items():
        nq = b["n_queries"]
        for tag, s in (("univ", b["B"]), ("qrels", b["Bq"])):
            rows += [row("B", ds, m, f"rho_depth_{tag}", s["rho"], s["rho_ci"],
                         nq),
                     row("B", ds, m, f"median_depth_{tag}", s["median_depth"],
                         n=nq),
                     row("B", ds, m, f"off_shallow_{tag}", s["off_shallow"],
                         n=s["n_shallow"]),
                     row("B", ds, m, f"off_deep_{tag}", s["off_deep"],
                         n=s["n_deep"]),
                     row("B", ds, m, f"split_diff_{tag}", s["split_diff"],
                         s["split_diff_ci"], nq),
                     row("B", ds, m, f"{'B' if tag == 'univ' else 'Bq'}_pass",
                         s["pass"], n=nq)]
    for (ds, m), a in res["A"].items():
        rows += [row("C", ds, m, "delta", a["delta"], a["delta_ci"], a["n"]),
                 row("C", ds, m, "delta_lab", a["delta_lab"], n=a["n"])]
    for (ds, m), d in res["D"].items():
        for l in LEVELS:
            lv = d["levels"][l]
            rows += [row("D", ds, m, f"off_L{l}", lv["off"], lv["off_ci"],
                         d["n"]),
                     row("D", ds, m, f"D_L{l}", lv["D"], lv["D_ci"], d["n"])]
        rows.append(row("D", ds, m, "range_L1_L5", d["range"], n=d["n"]))
    for ds, e in res["E"].items():
        if "strata" not in e:
            continue
        jn = f"cheap({len(e['judges'])})"
        for g, s in e["strata"].items():
            if s["n"] == 0:
                continue
            pre = f"h{g}:"
            for j, p in zip(e["judges"], s["p"]):
                rows.append(row("E", ds, jn, f"{pre}p:{j}", p, n=s["n"]))
            for k in range(1, s["m"] + 1):
                rows += [row("E", ds, jn, f"{pre}obs_ge{k}_given_ge1",
                             s["obs_ge_given_ge1"][k], n=s["n"]),
                         row("E", ds, jn, f"{pre}ind_ge{k}_given_ge1",
                             s["ind_ge_given_ge1"][k], n=s["n"]),
                         row("E", ds, jn, f"{pre}lift_ge{k}", s["lift_ge"][k],
                             s["lift_k_ci"] if k == s["lift_k"] else None,
                             s["n"])]
            rows.append(row("E", ds, jn, f"{pre}pairwise_lift_mean",
                            s["pairwise_lift_mean"], n=s["n"]))
            if g == 0:
                rows.append(row("E", ds, jn, "E_pass", s["pass"], n=s["n"]))
    return rows


def _jsonable_cells(d: dict) -> dict:
    return {f"{k[0]}|{k[1]}" if isinstance(k, tuple) else str(k): v
            for k, v in d.items()}


def cmd_run(args) -> dict:
    _LOG.clear()
    datasets = resolve_datasets(args.datasets)
    pred_path = predictions_path(args.predictions)
    reg = ic_gate("dl22" in datasets, args.unseal_dl22, pred_path)  # FIRST
    say(f"[gate] IC registration found and committed "
        f"({reg.get('commit')}); {len(reg['preds'])} prediction line(s)")
    ctx = load_inputs(args, datasets)
    shas = input_shas(ctx)
    check_shas(reg, shas)
    cells = build_cells(ctx, args.seed)
    ustatus = check_universe(ctx, cells, keys_only=False)
    for ds, s in ustatus.items():
        say(f"[universe] {ds}: {s}")
    for (ds, m), c in cells.items():
        if "error" in c and not c["error"].startswith("absent"):
            say(f"[cell] {ds} {m}: NO CELL — {c['error']}")
    res = run_analyses(ctx, cells, args.b, args.seed)
    judges = ctx["judges"]
    say(f"\nIC — label-incompleteness check | datasets {datasets} | "
        f"{len(res['valid'])} judge×dataset cells | B={args.b} seed={args.seed}")
    print_A(res["A"], cells, datasets, judges)
    print_B(res["B"], cells, datasets, judges)
    print_C(res["A"], cells, datasets, judges)
    print_D(res["D"], cells, datasets, judges)
    print_E(res["E"], datasets)
    t = tallies(res, datasets)
    vs = verdicts(t, reg)
    say("\n" + "=" * 100)
    say("BACKFILL — registered predictions (point estimates; CI never changes "
        "a verdict)")
    say("=" * 100)
    for r in ADJUDICATION_RULES:
        say("  " + r)
    for v in vs:
        if v["n_def"] is None:
            say(f"- IC[{v['key']}] {v['verdict']}")
            continue
        fails = ", ".join(v["failing"][:12]) + \
            (" …" if len(v["failing"]) > 12 else "")
        say(f"- IC[{v['key']}] 实际 {v['n_pass']}/{v['n_def']} 满足"
            f"｜预测：{v['prediction'] or '—'} → {v['verdict']}"
            + (f"（未满足：{fails}）" if fails else ""))
    od = out_dir(ctx["data_dir"])
    meta = {"script": "scripts/incompleteness_check.py run",
            "git_hash": b4.git_hash(), "code_dirty": b6.code_dirty(),
            "registration_commit": reg.get("commit"),
            "input_sha256": shas, "datasets": datasets, "judges": judges,
            "cheap": sorted(ctx["cheap"]), "flagships_40pct":
                sorted(ctx["flagships"]),
            "B": args.b, "seed": args.seed, "ci": "90% percentile",
            "criteria": CRITERIA, "adjudication_rules": ADJUDICATION_RULES,
            "universe_check": ustatus,
            "blind_note": "C and D are non-blind (already published); "
                          "E is exploratory", "timestamp_utc": now_utc()}
    units = {k: cell_units(c) for k, c in cells.items()}
    write_json(od / "ic_units.json", {"meta": meta,
                                      "cells": _jsonable_cells(units)})
    write_json(od / "ic_A.json", {"meta": meta,
                                  "cells": _jsonable_cells(res["A"])})
    write_json(od / "ic_B.json", {"meta": meta,
                                  "cells": _jsonable_cells(res["B"])})
    write_json(od / "ic_C.json", {"meta": meta, "cells": {
        f"{ds}|{m}": {"delta": a["delta"], "delta_ci": a["delta_ci"],
                      "delta_lab": a["delta_lab"], "n": a["n"],
                      "flagship40": cells[(ds, m)]["flagship"]}
        for (ds, m), a in res["A"].items()}})
    write_json(od / "ic_D.json", {"meta": meta,
                                  "cells": _jsonable_cells(res["D"])})
    write_json(od / "ic_E.json", {"meta": meta, "datasets": res["E"]})
    write_json(od / "ic_verdict.json", {"meta": meta, "verdicts": vs})
    write_csv(od / "ic_summary.csv", summary_rows(ctx, cells, res))
    logp = od / "ic_log.txt"
    logp.write_text("\n".join(_LOG) + "\n", encoding="utf-8")
    print(f"[log] wrote {logp}")
    print("REMINDER: whichever way this falls, the correction stays alignment "
          "to THESE assessors' operating point — never a quality claim.")
    return {"res": res, "verdicts": vs, "out_dir": od, "cells": cells}


# ═══════════════════════════════════════════════════════════════════════════
# selftest — synthetic, offline
# ═══════════════════════════════════════════════════════════════════════════

ST_DATASETS = ("dl21", "dl22", "antique")
ST_SHIFT = {"dl21": 0.6, "dl22": 0.6, "antique": -0.6}
ST_TONE = {1: -0.3, 2: -0.1, 3: 0.0, 4: 0.1, 5: 0.25}
ST_JUDGES = ["stub/shiftA", "stub/shiftB", "stub/missA", "stub/missB",
             "stub/flag"]
ST_CHEAP = ["stub/shiftA", "stub/shiftB", "stub/missA", "stub/missB"]


def _st_world(seed: int = 11, nq: int = 24):
    rng = np.random.default_rng(seed)
    qrels, missed, depth_of = {}, set(), {}
    for ds in ST_DATASETS:
        for qi in range(nq):
            q = f"{ds[:2]}{qi:03d}"
            dep = int(rng.integers(8, 31))
            depth_of[(ds, q)] = dep
            shallow = dep <= 19
            for di in range(dep):
                d = f"d{qi:03d}_{di:02d}"
                g = int(rng.choice(4, p=[0.45, 0.25, 0.18, 0.12]))
                qrels[(ds, q, d)] = g
                # missed positives: h=0 docs, much likelier in shallow queries
                if g == 0 and rng.random() < (0.6 if shallow else 0.05):
                    missed.add((ds, q, d))
    return qrels, missed, depth_of


def _st_records(qrels: dict, missed: set, flag_pairs: dict, seed: int = 5,
                score_jitter: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    # per-(judge, pair) item effect, independent across judges
    item = {(m, ds, q, d): float(rng.normal(0, 0.8)) for m in ST_JUDGES
            for (ds, q, d) in qrels}
    for m in ST_JUDGES:
        runs = 1 if m == "stub/flag" else 2
        for (ds, q, d), g in qrels.items():
            if m == "stub/flag" and (q, d) not in flag_pairs[ds]:
                continue
            for lvl in LEVELS:
                for par in "abc":
                    for run in range(runs):
                        if m.startswith("stub/miss"):
                            if (ds, q, d) in missed:
                                s = 3.0
                            else:
                                s = float(np.clip(g + rng.choice(
                                    [-1, 0, 1], p=[0.1, 0.8, 0.1]), 0, 3))
                        else:
                            x = g + ST_SHIFT[ds] + ST_TONE[lvl] + \
                                item[(m, ds, q, d)] + rng.normal(0, 0.6)
                            s = float(np.clip(np.floor(x + 0.5), 0, 3))
                        if score_jitter:
                            s = float((s + score_jitter) % 4)
                        rows.append((m, f"L{lvl}_{par}", lvl, ds, q, d, run,
                                     s, True))
    df = pd.DataFrame(rows, columns=["model_id", "prompt_id",
                                     "politeness_level", "dataset", "qid",
                                     "docid", "run", "score", "parse_ok"])
    # parse failures: a few scattered records (kept in universe) and ONE pair
    # with every L5 record failed for stub/shiftA on dl21 (dropped).
    k = next(iter(sorted(k for k in qrels if k[0] == "dl21")))
    bad = (df.model_id == "stub/shiftA") & (df.dataset == "dl21") & \
        (df.qid == k[1]) & (df.docid == k[2]) & (df.politeness_level == 5)
    df.loc[bad, "parse_ok"] = False
    df.loc[bad, "score"] = np.nan
    scat = df.sample(frac=0.002, random_state=3).index
    df.loc[scat, "parse_ok"] = False
    df.loc[scat, "score"] = np.nan
    return df


def _st_setup(root: Path, score_jitter: int = 0) -> dict:
    qrels, missed, depth_of = _st_world()
    rng = np.random.default_rng(9)
    flag_pairs = {}
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    (root / "derived").mkdir(parents=True, exist_ok=True)
    for ds in ST_DATASETS:
        keys = sorted((q, d) for (dd, q, d) in qrels if dd == ds)
        sel = [k for k in keys if rng.random() < 0.4]
        flag_pairs[ds] = set(sel)
        with open(root / "inputs" / f"pairs_{ds}_flagship40.jsonl", "w") as f:
            for q, d in sel:
                f.write(json.dumps({"qid": q, "docid": d}) + "\n")
    df = _st_records(qrels, missed, flag_pairs, score_jitter=score_jitter)
    jp = root / "derived" / "judgments.pkl"
    df.to_pickle(jp)
    qj = root / "qrels_stub.json"
    qj.write_text(json.dumps([[ds, q, d, g] for (ds, q, d), g in
                              sorted(qrels.items())]))
    return {"qrels": qrels, "missed": missed, "depth_of": depth_of,
            "jpath": jp, "qjson": qj, "df": df}


def _git_init(d: Path) -> None:
    for a in (["init", "-q"], ["config", "user.email", "st@example.com"],
              ["config", "user.name", "selftest"]):
        subprocess.run(["git", "-C", str(d), *a], check=True,
                       capture_output=True)


def _git_commit(d: Path, msg: str) -> None:
    subprocess.run(["git", "-C", str(d), "add", "-A"], check=True,
                   capture_output=True)
    subprocess.run(["git", "-C", str(d), "commit", "-q", "-m", msg],
                   check=True, capture_output=True)


def _exits(fn) -> bool:
    try:
        fn()
    except SystemExit:
        return True
    return False


def _ns(**kw) -> argparse.Namespace:
    base = dict(data_dir=None, judgments=None, qrels_json=None,
                predictions=None, judges=",".join(ST_JUDGES),
                cheap=",".join(ST_CHEAP), flagships="stub/flag", b=200,
                seed=SEED, unseal_dl22=True,
                datasets=",".join(ST_DATASETS))
    base.update(kw)
    return argparse.Namespace(**base)


def cmd_selftest(args) -> None:
    import contextlib
    import io
    print("=" * 96)
    print("  IC SELF-TEST (synthetic; no Drive, no network)")
    print("=" * 96)
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f"  ({detail})" if detail else ""))
        if not ok:
            fails.append(name)

    # ── 1. statistics primitives ──────────────────────────────────────────
    rng = np.random.default_rng(0)
    x = rng.integers(0, 5, 40).astype(float)
    y = x + rng.normal(0, 2, 40)
    try:
        from scipy.stats import spearmanr
        ref = float(spearmanr(x, y).correlation)
        check("spearman == scipy (ties)", abs(spearman(x, y) - ref) < 1e-12,
              f"{spearman(x, y):.6f}")
    except ImportError:
        check("spearman perfect monotone = 1", abs(spearman(x, x * 2) - 1) < 1e-12)
    check("spearman constant -> nan", np.isnan(spearman(np.ones(5), x[:5])))
    p = np.array([0.1, 0.35, 0.5, 0.8])
    brute = np.zeros(5)
    for bits in product((0, 1), repeat=4):
        pr = np.prod([pi if b else 1 - pi for pi, b in zip(p, bits)])
        brute[sum(bits)] += pr
    check("Poisson-binomial pmf == brute force",
          np.allclose(pb_pmf(p), brute, atol=1e-14))
    check("Poisson-binomial batched == single",
          np.allclose(pb_pmf(np.stack([p, p[::-1]]))[1], brute))
    check("tail(pmf)[0] == 1 and monotone",
          abs(tail(brute)[0] - 1) < 1e-12 and np.all(np.diff(tail(brute)) <= 0))
    idx, W = boot_weights(7, 50, SEED)
    check("boot weights: rows sum to nq, == batch6.boot_indices",
          np.all(W.sum(1) == 7) and np.array_equal(
              idx, b6.boot_indices(7, 50, SEED)))
    # E on hand-built matrices: independent vs shared positives
    n = 4000
    Ri = rng.random((n, 3)) < np.array([0.1, 0.15, 0.2])
    qc = np.arange(n) % 40
    ei = e_stratum(Ri, qc, 40, 100, SEED)
    shared = rng.random(n) < 0.1
    Rs = (rng.random((n, 3)) < 0.03) | shared[:, None]
    es = e_stratum(Rs, qc, 40, 100, SEED)
    check("E: independent judges -> lift_2 ≈ 1, PASS",
          ei["pass"] and abs(ei["lift_ge"][2] - 1) < 0.25,
          f"lift_2 {ei['lift_ge'][2]:.3f}")
    check("E: shared positives -> lift_2 > 2, FAIL",
          es["pass"] is False and es["lift_ge"][2] > 2,
          f"lift_2 {es['lift_ge'][2]:.3f}")
    dcap = np.array([30, 30, 30, 30, 30, 12, 20, 25], dtype=float)
    ocap = np.array([0.1, 0.2, 0.1, 0.2, 0.1, 0.5, 0.4, 0.3])
    ic_, wc_ = boot_weights(8, 50, SEED)
    dsx = depth_stats(dcap, ocap, ic_, wc_)
    check("B split with > half the queries at the cap: < median vs rest",
          dsx["n_shallow"] == 3 and dsx["n_deep"] == 5
          and abs(dsx["split_diff"] - (0.4 - 0.14)) < 1e-12)
    # A criteria on hand-built summaries
    fake = {"by_grade": {0: {"n": 50, "off": 0.9}, 1: {"n": 40, "off": 0.4},
                         2: {"n": 30, "off": 0.35}, 3: {"n": 20, "off": -0.2}},
            "delta": 0.5}
    check("A1/A2 scale-shift pattern -> PASS", a_criteria(fake) ==
          {"A1_pass": True, "A2_pass": True})
    fake["by_grade"][1]["off"] = 0.05
    check("A1/A2 h=1 offset < 1/3 -> FAIL", a_criteria(fake) ==
          {"A1_pass": False, "A2_pass": False})
    fake["by_grade"][0]["n"] = 5
    check("A1 undefined when n(h=0) < 10", a_criteria(fake)["A1_pass"] is None)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        world = _st_setup(root / "data")
        common = dict(data_dir=str(root / "data"),
                      judgments=str(world["jpath"]),
                      qrels_json=str(world["qjson"]))

        # ── 2. cells / universe ───────────────────────────────────────────
        with contextlib.redirect_stdout(io.StringIO()):
            a0 = _ns(**common)
            ctx = load_inputs(a0, list(ST_DATASETS))
            cells = build_cells(ctx, SEED)
        c21 = cells[("dl21", "stub/shiftA")]
        nq21 = sum(1 for k in world["qrels"] if k[0] == "dl21")
        check("universe: all-L5-failed pair dropped, others kept",
              c21["info"]["n_dropped_unusable"] == 1
              and len(c21["h"]) == nq21 - 1, f"{len(c21['h'])}/{nq21}")
        cf = cells[("dl21", "stub/flag")]
        check("flagship restricted to its 40% file",
              cf["flagship"] and 0.25 < len(cf["h"]) / nq21 < 0.55,
              f"{len(cf['h'])}/{nq21}")
        check("RAW = L3 record mean; RAW_lab = round-half-up",
              np.array_equal(c21["raw_lab"],
                             np.clip(np.floor(c21["raw"] + 0.5), 0, 3)))
        # integrity vs a stored batch6 labels file (pickle stand-in)
        st = pd.DataFrame({"model_id": "stub/shiftB",
                           "qid": [k[0] for k in cells[("dl21", "stub/shiftB")]
                                   ["universe"]],
                           "docid": [k[1] for k in cells[("dl21",
                                     "stub/shiftB")]["universe"]],
                           "human": cells[("dl21", "stub/shiftB")]["h"],
                           "RAW": cells[("dl21", "stub/shiftB")]["raw_lab"]})
        stp = root / "data" / "derived" / "batch6_labels_dl21.pkl"
        st.to_pickle(stp)
        s1 = check_universe(ctx, cells, keys_only=False)
        check("universe check vs stored batch6 labels: identical",
              s1["dl21"].startswith("identical"), s1["dl21"])
        st2 = st.copy()
        st2.loc[0, "RAW"] = (st2.loc[0, "RAW"] + 1) % 4
        st2.to_pickle(stp)
        check("universe check: altered RAW -> refuse",
              _exits(lambda: check_universe(ctx, cells, keys_only=False)))
        check("universe check keys-only (paste) ignores RAW",
              check_universe(ctx, cells, keys_only=True)["dl21"]
              .startswith("identical"))
        stp.unlink()

        # ── 3. analyses on the synthetic world ────────────────────────────
        res = run_analyses(ctx, cells, 200, SEED)
        A, Bres, D, E = res["A"], res["B"], res["D"], res["E"]
        for ds in ST_DATASETS:
            sa, ma = A[(ds, "stub/shiftA")], A[(ds, "stub/missA")]
            check(f"A2 {ds}: scale-shift judge PASS, missing-positive judge "
                  f"FAIL", sa["A2_pass"] is True and ma["A2_pass"] is False,
                  f"shift off1/off2 {sa['by_grade'][1]['off']:+.2f}/"
                  f"{sa['by_grade'][2]['off']:+.2f}; miss "
                  f"{ma['by_grade'][1]['off']:+.2f}/"
                  f"{ma['by_grade'][2]['off']:+.2f}")
        check("A: missing-positive judge carries ~all offset mass at h=0",
              A[("dl21", "stub/missA")]["mass_share_h0"] > 0.9,
              f"{A[('dl21', 'stub/missA')]['mass_share_h0']:.2f}")
        check("A1 dl21 scale-shift judge PASS",
              A[("dl21", "stub/shiftA")]["A1_pass"] is True)
        check("A1 antique scale-shift judge FAIL (floor at h=0; use A2)",
              A[("antique", "stub/shiftA")]["A1_pass"] is False)
        bm = Bres[("dl21", "stub/missA")]["Bq"]
        check("B: missed positives in shallow queries -> ρ < −0.3, split > 0",
              bm["rho"] < -0.3 and bm["split_diff"] > 0.1 and
              bm["pass"] is False, f"ρ {bm['rho']:+.2f} split "
                                   f"{bm['split_diff']:+.2f}")
        check("B (universe depth) == Bq for cheap judges (uncapped stub)",
              abs(Bres[("dl21", "stub/missA")]["B"]["rho"] - bm["rho"]) < 1e-12)
        check("D: tone moves the scale-shift judge's offset (range > 0.3)",
              D[("dl21", "stub/shiftA")]["range"] > 0.3,
              f"{D[('dl21', 'stub/shiftA')]['range']:.3f}")
        check("D: tone-insensitive judge range < 0.06; D(L3) ≡ 0",
              D[("dl21", "stub/missA")]["range"] < 0.06 and
              all(D[k]["levels"][3]["D"] == 0 for k in D))
        check("C == A delta == D(L3) offset",
              all(abs(A[k]["delta"] - D[k]["levels"][3]["off"]) < 1e-12
                  for k in A))
        e21 = E["dl21"]
        check("E dl21: 4 cheap judges (flagship excluded), strata h0/h1",
              e21["judges"] == ST_CHEAP and set(e21["strata"]) == {0, 1})
        with contextlib.redirect_stdout(io.StringIO()):
            ctx_s = dict(ctx, cheap=frozenset(["stub/shiftA", "stub/shiftB"]))
            ctx_m = dict(ctx, cheap=frozenset(["stub/missA", "stub/missB"]))
            es_ = run_analyses(ctx_s, cells, 100, SEED)["E"]["dl21"]
            em_ = run_analyses(ctx_m, cells, 100, SEED)["E"]["dl21"]
        l_s = es_["strata"][0]["lift_ge"][2] or 0.0
        check("E (cells): shared-missed judges FAIL, lift_2 above the "
              "independent-noise judges'", em_["pass"] is False
              and em_["strata"][0]["lift_ge"][2] > max(l_s, E_LIFT_MAX),
              f"lift_2 {_f(es_['strata'][0]['lift_ge'][2], 2, False)} vs "
              f"{_f(em_['strata'][0]['lift_ge'][2], 2, False)}")
        res2 = run_analyses(ctx, cells, 200, SEED)
        check("bootstrap deterministic (seed 42)",
              res2["A"][("dl21", "stub/shiftA")]["delta_ci"] ==
              A[("dl21", "stub/shiftA")]["delta_ci"] and
              res2["B"][("dl22", "stub/missB")]["B"]["rho_ci"] ==
              Bres[("dl22", "stub/missB")]["B"]["rho_ci"])
        t = tallies(res, ST_DATASETS)
        check("tallies: A2|msmarco has 10 defined cells (5 judges × 2 ds)",
              len(t[("A2", "msmarco")]) == 10)

        # ── 4. paste: counts only, invariant to judge scores ──────────────
        units = {k: cell_units(c) for k, c in cells.items()}
        eu = {ds: e_units(cells, ds, ctx["judges"], ctx["cheap"])
              for ds in ST_DATASETS}
        txt = "\n".join(paste_lines(ctx, units, eu,
                                    check_universe(ctx, cells, True)))
        world_j = _st_setup(root / "data_j", score_jitter=1)
        with contextlib.redirect_stdout(io.StringIO()):
            ctx_j = load_inputs(_ns(data_dir=str(root / "data_j"),
                                    judgments=str(world_j["jpath"]),
                                    qrels_json=str(world_j["qjson"])),
                                list(ST_DATASETS))
            cells_j = build_cells(ctx_j, SEED)
        units_j = {k: cell_units(c) for k, c in cells_j.items()}
        eu_j = {ds: e_units(cells_j, ds, ctx_j["judges"], ctx_j["cheap"])
                for ds in ST_DATASETS}
        txt_j = "\n".join(paste_lines(ctx_j, units_j, eu_j,
                                      check_universe(ctx_j, cells_j, True)))

        def strip_j(s):
            return "\n".join(ln for ln in s.splitlines()
                             if not ln.startswith("- sha256[judgments]"))
        check("paste: identical when every judge score changes (no score "
              "enters the block)", strip_j(txt) == strip_j(txt_j)
              and txt != txt_j)
        nsha = txt.count("sha256[")
        check("paste: heading, sha lines (judgments + 3 qrels + 3 flagship),"
              " placeholders", txt.startswith(f"# {IC_MARKER}") and nsha == 7
              and "预测：____" in txt and "IC[A2|antique]" in txt, f"{nsha}")

        # ── 5. gate + run end-to-end in a temp git repo ───────────────────
        repo = root / "fake_repo"
        (repo / "paper").mkdir(parents=True)
        _git_init(repo)
        pred = repo / "paper" / "PREDICTIONS.md"
        pred.write_text("# Semi-Blind Prediction Registration\n\nnothing\n")
        _git_commit(repo, "init")
        a_run = _ns(**common, predictions=str(pred), b=100)
        with contextlib.redirect_stdout(io.StringIO()):
            check("gate: no IC heading -> refuse",
                  _exits(lambda: cmd_run(a_run)))
        block = txt.splitlines()
        pred.write_text(pred.read_text() + "\n" + "\n".join(block) + "\n")
        _git_commit(repo, "unfilled")
        check("gate: committed block with '____' -> refuse",
              _exits(lambda: ic_gate(False, False, pred)))
        nd = defined_counts(units, eu, ST_DATASETS)
        filled = []
        for ln in block:
            m_ = re.match(r"^- IC\[(\w+)\|(\w+)\]", ln)
            if m_ and "预测：____" in ln:
                N = nd[(m_.group(1), m_.group(2))]
                tok = f"≥{N}/{N}" if m_.group(1) != "E" else f"≤0/{N}"
                ln = ln.replace("预测：____", f"预测：{tok}")
            filled.append(ln.replace("____", "2026-10-25"))
        pred.write_text("# Semi-Blind Prediction Registration\n\nnothing\n\n"
                        + "\n".join(filled) + "\n# Next section\n")
        check("gate: filled but uncommitted -> refuse",
              _exits(lambda: ic_gate(False, False, pred)))
        _git_commit(repo, "IC registration")
        check("gate: dl22 without --unseal-dl22 -> refuse",
              _exits(lambda: ic_gate(True, False, pred)))
        regp = ic_gate(True, True, pred)
        check("gate: committed + filled + unseal -> passes, commit recorded, "
              "predictions parsed", regp["commit"] is not None
              and len(regp["preds"]) >= 9, f"{len(regp['preds'])} lines")
        bad = {"shas": dict(regp["shas"]), "preds": {}}
        bad["shas"]["qrels:dl21"] = "0" * 64
        with contextlib.redirect_stdout(io.StringIO()):
            check("sha mismatch -> refuse", _exits(lambda: check_shas(
                bad, input_shas(ctx))))
            del bad["shas"]["qrels:dl21"]
            check("sha unregistered -> refuse", _exits(lambda: check_shas(
                bad, input_shas(ctx))))
        a_seal = _ns(**common, predictions=str(pred), unseal_dl22=False)
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            check("run: dl22 without --unseal-dl22 -> refuse",
                  _exits(lambda: cmd_run(a_seal)))
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            out = cmd_run(a_run)
        od = out["out_dir"]
        parts = ["units", "A", "B", "C", "D", "E", "verdict"]
        check("run: every ic_<part>.json + ic_summary.csv + ic_log.txt",
              all((od / f"ic_{p}.json").exists() for p in parts)
              and (od / "ic_summary.csv").exists()
              and (od / "ic_log.txt").exists())
        with open(od / "ic_summary.csv", newline="", encoding="utf-8") as f:
            rd = list(csv.DictReader(f))
        check("csv: long format columns + all analyses present",
              tuple(rd[0].keys()) == CSV_COLS and
              {r["analysis"] for r in rd} >=
              {"units", "A", "B", "C", "D", "E"}, f"{len(rd)} rows")
        vd = {v["key"]: v for v in out["verdicts"]}
        check("verdicts: A2|msmarco ≥N/N MISS (missing-positive judges fail)",
              vd["A2|msmarco"]["verdict"] == "MISS"
              and vd["A2|msmarco"]["n_def"] == 10)
        check("verdicts: adjudicate ≥k / ≤k / N mismatch",
              adjudicate(("≥", 3, 5), 3, 5) == "HIT"
              and adjudicate(("≤", 1, 5), 2, 5) == "MISS"
              and adjudicate(("≥", 3, 6), 4, 5).startswith("not adjudicated"))
        logtxt = (od / "ic_log.txt").read_text(encoding="utf-8")
        check("log: every section + BACKFILL",
              all(s in logtxt for s in ("A — ", "B — ", "C — ", "D — ",
                                        "E — ", "BACKFILL")))
        check("[40%] mark printed for the flagship", "flag [40%]" in logtxt)

    print("-" * 96)
    if fails:
        print(f"SELFTEST FAILED: {len(fails)} check(s): {fails}")
        sys.exit(1)
    print("ALL SELFTESTS PASSED")


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--data-dir", default=None, help="DATA_DIR (default "
                       "$DATA_DIR)")
        p.add_argument("--judgments", default=None, help="judgments file "
                       "(default $DATA_DIR/derived/judgments.parquet)")
        p.add_argument("--qrels-json", default=None, help="TEST hook: stub "
                       "qrels JSON instead of ir_datasets")
        p.add_argument("--datasets", default=",".join(DATASETS))
        p.add_argument("--judges", default=None, help="override the "
                       "config/models.yaml roster (comma list)")
        p.add_argument("--cheap", default=None, help="override the cheap set")
        p.add_argument("--flagships", default=None, help="override the "
                       "flagship (40%%) set")
        p.add_argument("--seed", type=int, default=SEED)

    pp = sub.add_parser("paste", help="print the registration block (counts "
                        "and sha256 only)")
    common(pp)
    pr = sub.add_parser("run", help="gated analysis A-E + adjudication")
    common(pr)
    pr.add_argument("--predictions", default=None,
                    help="PREDICTIONS.md (default paper/PREDICTIONS.md)")
    pr.add_argument("--unseal-dl22", action="store_true")
    pr.add_argument("--b", type=int, default=B_BOOT)
    sub.add_parser("selftest", help="synthetic offline checks")
    args = ap.parse_args()
    if args.cmd == "paste":
        cmd_paste(args)
    elif args.cmd == "run":
        cmd_run(args)
    else:
        cmd_selftest(args)


if __name__ == "__main__":
    main()
