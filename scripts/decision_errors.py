#!/usr/bin/env python3
"""DE — pre-registered DECISION-ERROR study: how often does using an LLM
judge's labels give a different answer than human labels, and how much of
that does operating-point correction remove?

Framing (unchanged from batch 4-7): the correction removes a MEASUREMENT
ARTIFACT (the judge's operating point vs THESE reference annotators); a lower
decision-error rate is artifact removal, never "a better judge".

Two kinds of decisions
----------------------
* WITHIN-SCALE (compare two quantities scored with the SAME labels: A beats
  B, significance).  Raw labels already preserve these (B5 tau 0.69-0.79), so
  correction is predicted not to help.
* CROSS-SCALE / ABSOLUTE (a judge-labelled number against a human-scale
  reference: a published leaderboard number, a K-relevant rule, a spec
  checked on judge-labelled test data).  This is where the offset turns into
  wrong answers and where correction should help.
A fixed uniform tau-grid error rate restates the mean offset, so the bar-
clearing grid (D1b) is SECONDARY / descriptive.

Studies (subcommands)
---------------------
  fit-lodo-rules  msmarco rule refitted on DL19+DL20 ONLY (dial restricted to
                  {L1, L3}: the 325 B5 top-up pairs exist at L1/L3 only), with
                  batch4's pack / delta_D / best_level-rule / fit_monotone_map.
                  Writes derived/decision_errors/rules_lodo.json (never
                  touches correction_rules.json; batch4.run_rules is NOT
                  called).  Refuses to overwrite without --force.
  paste           prints the "# DE 预注册 — 决策错误率" registration block:
                  units / queries / grid sizes, input sha256s, blind-status
                  and prediction placeholders.  Loads NO human labels (batch6
                  label parquets are read for qid/model columns only; eval
                  JSONs only for file names and universe_sha256).
  d2-pool         D2 per-query pool decisions (Application A) on the stored
                  batch6_labels_<ds>.parquet, recomputed + hard-checked with
                  batch6.prepare_cell / crossfit / check_stored, plus the new
                  SHIFT_FROZEN arm.  "relevant" = grade >= 2.
                    D2a  query usable: >= 1 relevant doc (K=1)
                    D2b  >= K relevant docs, K in {3,5,10} (MS MARCO) /
                         {2,3} (ANTIQUE)  [K=1 is D2a]
                    D2c  per-query budget estimate within 20%:
                         |n_L - n_H| / max(n_H, 1) <= 0.2 (error = outside)
                  error = fraction of queries whose verdict differs from the
                  HUMAN verdict; paired query bootstrap via
                  batch6.boot_indices (B=1000, seed 42, 90% percentile CI).
  d1-b5           D1 system decisions on dl21 (B5 pool, 13 systems, 4 cheap
                  judges), NDCG@10 recomputed with batch5's load_runs /
                  guard_dataset / judge_labels / human_labels / ndcg_matrix
                  (condensed-list convention, intersection universe) and
                  integrity-checked against b5_per_system_ndcg.csv (1e-4).
                  Label sets HUMAN / RAW / CORR / SHIFT_FROZEN (+ CORR_LODO /
                  SHIFT_LODO when rules_lodo.json is present).
                    D1a        leaderboard placement, ordered pairs (s, r):
                               predict NDCG_L(s) > NDCG_H(r); truth
                               NDCG_H(s) > NDCG_H(r)  [cross-scale, primary]
                    D1a_place  mean |rank_L(s) - rank_H(s)| when s is placed
                               into the human leaderboard of the others
                               (descriptive; ranks, not an error rate)
                    D1b        bar clearing NDCG(s) >= tau, tau 0.30..0.80
                               step 0.01, error over (system, tau) [descr.]
                    D1b_best   "best system >= tau", error over tau [descr.]
                    D1c        within-scale "X beats Y by >= m", m in
                               {0, 0.01, 0.02, 0.05}, unordered pairs,
                               3-way verdict (X / Y / neither)
                    D1c_sig    paired t-test verdict agreement, alpha 0.05
                  Uncertainty: bootstrap_decisions — shared resampled
                  queries (batch5.paired_bootstrap pattern, B=2000, seed
                  42), truth recomputed inside every replicate, 90% CI.
  d3-validate     D3 on dl22: a practitioner validates a distilled student on
                  a judge-labelled test set.  Student passes iff
                  |mean clip(pred) - mean test label| <= 0.25 AND
                  ratio@2 = #(round_half_up(clip(pred)) >= 2) / #(label >= 2)
                  in [0.8, 1.25] (batch7 per_query_stats / aggregate /
                  boot_weights).  Error = fraction of students whose verdict
                  under the judge test labels (RAW / FROZEN / SHIFT_FROZEN of
                  the EVALUATOR judge) differs from the verdict under human
                  test labels.  Units: registered + ablation students
                  (batch7_summary.SETS); '@tag' exploratory retrains and any
                  other file are excluded and listed.  Students taught by the
                  evaluator are excluded.  Universe = B7 dl22 universe ∩ the
                  evaluator's B6 universe.  Query bootstrap B=1000 seed 42;
                  seed variance is NOT in the CI (as in B7).
  d1-students     D1 on dl22 with the students as "systems": NDCG@10 per query
                  from batch7.query_metrics with the evaluator's labels
                  substituted for the human grades (batch7 convention: re-rank
                  the judged pool, queries with IDCG = 0 excluded per label
                  set).  D1a / D1a_place / D1c / D1c_sig (+ D1b descriptive).
                  CAVEAT: students sit in a narrow NDCG band, which inflates
                  cross-scale error.
  local-calib     OPTIONAL (Priority 5).  R=200 random query-disjoint splits:
                  ~200 calibration pairs (whole queries) estimate
                    SHIFT_n         label shift Δ̂_n = mean(L3 record mean)
                                    - mean(human) over calibration pairs
                    METRIC_SHIFT_n  (D1 only) subtract the mean NDCG(RAW -
                                    HUMAN) of the calibration queries
                  and EVERY arm (incl. RAW / CORR / HUMAN) is recomputed on
                  the held-out queries only, so the comparison stays paired.
                  Reported: mean over splits and 5/95% split percentiles.
  selftest        synthetic, offline: every decision function, the bootstrap,
                  the no-op assertions, the registration gate (temp git repo,
                  fake PREDICTIONS.md), sha checks, the paste format and the
                  long CSV.

Registration gate (every computing subcommand, BEFORE any data is read)
-----------------------------------------------------------------------
The heading "# DE 预注册 — 决策错误率" must exist in a clean, committed
paper/PREDICTIONS.md with no unfilled "____"; dl22 additionally needs
--unseal-dl22 (src.qrels.assert_not_sealed).  Every input sha256 the
subcommand uses must be registered ("sha256[<name>]: <hex>") and match.
Adjudication: point estimate, tie band |Δerr| <= 0.02; a 90% CI never changes
a verdict.  Registered prediction lines:
    - DE[<decision>|<dataset>|<judge>|<arm>|<grid>] ...｜预测：<tok>
    tok = 降 (arm lowers error vs RAW) | 升 | 平 (|Δerr| <= 0.02)
        | ≥x / ≤x (level of that arm's own error; no tie band)
D2 rows of the in-sample datasets (dl21, antique) for FROZEN / SHIFT_FROZEN
are reported but never adjudicated.

Outputs: $DATA_DIR/derived/decision_errors/de_<sub>.json and the long
de_summary.csv (decision, dataset, judge, arm, grid, err, ci_lo, ci_hi,
d_vs_RAW, d_ci, verdict), rebuilt from the per-subcommand
de_summary_<sub>.csv files.  `verdict` = lower / higher / tie (Δerr vs RAW,
tie band 0.02) | ref (RAW) | descr (D1a_place); with a registered prediction
it is suffixed "|HIT", "|MISS", "|TIE" or "|not adjudicated (...)".
For local-calib rows (decision "LC:<d>") err / ci are the mean and 5/95%
percentiles over splits, not a bootstrap.

Usage (Colab, repo root, after the registration commit):
    python scripts/decision_errors.py fit-lodo-rules
    python scripts/decision_errors.py paste
    python scripts/decision_errors.py d2-pool --datasets dl22,dl21,antique --unseal-dl22
    python scripts/decision_errors.py d1-b5
    python scripts/decision_errors.py d3-validate --unseal-dl22
    python scripts/decision_errors.py d1-students --unseal-dl22
    python scripts/decision_errors.py local-calib --unseal-dl22      # optional
    python scripts/decision_errors.py selftest
CPU only, no API calls.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
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
from batch7_summary import SETS as B7_SETS                 # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
# Constants (registered verbatim; not adjustable from the CLI)
# ═══════════════════════════════════════════════════════════════════════════

DE_MARKER = "DE 预注册 — 决策错误率"
OUT_SUB = ("derived", "decision_errors")

SEED = 42
B_D2 = 1000                 # batch6 convention
B_D1 = 2000                 # batch5 convention
B_D3 = 1000                 # batch7 convention
CI_LO, CI_HI = 5.0, 95.0    # 90% percentile interval
TIE = 0.02                  # |Δerr| tie band
EPS = 1e-12                 # float slack on the tie band / exact fractions
REL = 2                     # relevant = grade >= 2

D2_DATASETS = ("dl22", "dl21", "antique")
K_GRID = {"dl22": (1, 3, 5, 10), "dl21": (1, 3, 5, 10), "antique": (1, 2, 3)}
BUDGET_TOL = 0.2
D2_ARMS = ("RAW", "FROZEN", "SHIFT_FROZEN", "REFIT")
D2_INSAMPLE_ARMS = ("FROZEN", "SHIFT_FROZEN")      # never adjudicated off dl22

TAU_GRID = np.round(np.arange(0.30, 0.80 + 1e-9, 0.01), 2)
MARGINS = (0.0, 0.01, 0.02, 0.05)
ALPHA = 0.05
B5_DATASET = "dl21"
B5_JUDGES = ("deepseek/deepseek-v4-flash", "deepseek/deepseek-v4.1-flash",
             "google/gemini-3.5-flash", "google/gemini-3.8-flash")
B5_TOL = 1e-4

LODO_DATASETS = ("dl19", "dl20")
LODO_DIALS = (1, 3)

D3_OFF_TOL = 0.25
D3_RATIO = (0.8, 1.25)
EVALUATORS = ("deepseek/deepseek-v4.1-flash", "deepseek/deepseek-v4-flash",
              "google/gemini-3.5-flash", "google/gemini-3.8-flash")
D3_TEST_SETS = ("HUMAN", "RAW", "FROZEN", "SHIFT_FROZEN")
STUDENT_ARMS = ("minilm-raw", "msmarco-ce")

LC_N = 200
LC_R = 200

SUBS = ("d2-pool", "d1-b5", "d3-validate", "d1-students", "local-calib")
CSV_COLS = ("decision", "dataset", "judge", "arm", "grid", "err", "ci_lo",
            "ci_hi", "d_vs_RAW", "d_ci", "verdict")
DESCRIPTIVE = ("D1a_place",)

ADJUDICATION_RULES = [
    "1. 方向一律按点估计判定：Δerr = err(arm) − err(RAW)；|Δerr| ≤ 0.02 为 tie"
    "（代码硬编码 TIE = 0.02）。",
    "2. 预测\"降\"：Δerr < −0.02 记命中，Δerr > +0.02 记未中，否则 tie；\"升\"方向"
    "相反；\"平\"：|Δerr| ≤ 0.02 记命中，否则未中。tie 既不计命中也不计未中。",
    "3. 水平预测（≥x / ≤x）作用于该 arm 自身 err 的点估计，无 tie 带。",
    "4. 90% CI（query bootstrap 百分位）单独报告为\"显著\"与否，永不改变判定。",
    "5. no-op 单元（冻结规则 identity@L3）：FROZEN/CORR ≡ RAW 逐位相同、Δerr ≡ 0"
    "（含 bootstrap），由代码 hard-fail 断言，不计分。",
    "6. D2 的 dl21（部分 in-sample）与 antique（完全 in-sample）上 FROZEN / "
    "SHIFT_FROZEN 只报告不判定；REFIT（query-disjoint 交叉拟合）处处可判。",
    "7. D1b / D1b_best / D1a_place 为描述性，不登记。",
]


# ═══════════════════════════════════════════════════════════════════════════
# Small helpers
# ═══════════════════════════════════════════════════════════════════════════

def out_dir(data_dir: Path) -> Path:
    p = data_dir.joinpath(*OUT_SUB)
    p.mkdir(parents=True, exist_ok=True)
    return p


def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds")


def rhu_vec(x) -> np.ndarray:
    """Vectorised batch5.round_half_up (floor(x + 0.5), clipped to 0..3)."""
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("rhu_vec got a non-finite value")
    return np.clip(np.floor(x + 0.5), 0, 3).astype(np.int64)


def ci90(x) -> list:
    x = np.asarray(x, dtype=float)
    f = x[np.isfinite(x)]
    if f.size == 0:
        return [None, None]
    return [float(np.percentile(f, CI_LO)), float(np.percentile(f, CI_HI))]


def dir_verdict(d) -> str:
    if d is None or not np.isfinite(d):
        return "n/a"
    if abs(d) <= TIE + EPS:
        return "tie"
    return "lower" if d < 0 else "higher"


def short_judge(model: str) -> str:
    """'deepseek/deepseek-v4-flash' -> 'v4-flash' (b5 CSV naming)."""
    s = model.split("/")[-1]
    return s[len("deepseek-"):] if s.startswith("deepseek-") else s


def _f(x, p=4, sign=True) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "   n/a"
    return f"{x:+.{p}f}" if sign else f"{x:.{p}f}"


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(b6._jclean(obj), ensure_ascii=False, indent=1,
                               allow_nan=False), encoding="utf-8")
    print(f"[json] wrote {path}")


def _git_in(path: Path, *a) -> tuple[int, str]:
    try:
        p = subprocess.run(["git", "-C", str(Path(path).resolve().parent), *a],
                           capture_output=True, text=True)
        return p.returncode, (p.stdout.strip() or p.stderr.strip())
    except Exception as e:                                     # noqa: BLE001
        return 1, str(e)


def predictions_path(arg: str | None) -> Path:
    return Path(arg) if arg else REPO / "paper" / "PREDICTIONS.md"


# ═══════════════════════════════════════════════════════════════════════════
# Registration: locate, parse, gate, sha checks, adjudication
# ═══════════════════════════════════════════════════════════════════════════

_SHA_LINE = re.compile(r"sha256\[([^\]]+)\]\s*[:：]\s*([0-9a-f]{64})")
_PRED_LINE = re.compile(r"^\s*-\s*DE\[([^\]]+)\].*?预测[:：]\s*(.*)$")
_TOK = re.compile(r"^(降|升|平|(?:≥|>=|≤|<=)\s*[0-9]*\.?[0-9]+)")


def de_block(text: str) -> str | None:
    """From the '# DE 预注册 — 决策错误率' heading to the next top-level '# '."""
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines)
                  if re.match(r"^#\s", ln) and DE_MARKER in ln), None)
    if start is None:
        return None
    end = next((j for j in range(start + 1, len(lines))
                if re.match(r"^#\s", lines[j])), len(lines))
    return "\n".join(lines[start:end])


def parse_registration(block: str) -> dict:
    shas: dict = {}
    for m in _SHA_LINE.finditer(block):
        shas[m.group(1).strip()] = m.group(2)
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
        tok = mt.group(1).replace(">=", "≥").replace("<=", "≤").replace(" ", "")
        preds[key] = tok
    return {"shas": shas, "preds": preds, "invalid": invalid}


def de_gate(dl22: bool, unseal: bool, path: Path,
            clean_fn=b7.file_committed) -> dict:
    """Mirror of batch6.gate (strict branch): heading + clean commit + filled;
    dl22 additionally needs --unseal-dl22.  Called BEFORE any data read."""
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    block = de_block(text)
    if block is None:
        sys.exit(f"ERROR: {path} has no heading '# {DE_MARKER}'.\n  Print the "
                 f"block with `paste`, fill it, COMMIT it — only then run the "
                 f"decision-error subcommands.")
    ok, why = clean_fn(path)
    if not ok:
        sys.exit(f"ERROR: {path} {why}. The DE registration counts only once "
                 f"it is committed (commit timestamp = freeze time).")
    if "____" in block:
        sys.exit("ERROR: the DE registration block still contains unfilled "
                 "'____' placeholders.")
    if dl22:
        assert_not_sealed(["dl22"], unseal=unseal)
    reg = parse_registration(block)
    for key, tok in reg["invalid"]:
        print(f"[registration] WARNING: DE[{key}]: unparseable prediction "
              f"{tok!r} — treated as unregistered.", file=sys.stderr)
    rc, out = _git_in(path, "log", "-1", "--format=%H,%cI", "--", path.name)
    reg["commit"] = out if rc == 0 and out else None
    return reg


def check_shas(reg: dict, actual: dict) -> None:
    """actual: name -> sha256 of an input this subcommand uses."""
    problems = []
    for name, sha in actual.items():
        want = reg["shas"].get(name)
        if want is None:
            problems.append(f"{name}: not registered (no 'sha256[{name}]: <hex>' "
                            f"line in the DE block)")
        elif want != sha:
            problems.append(f"{name}: sha256 {sha} != registered {want}")
    if problems:
        sys.exit("ERROR: input integrity check failed —\n    "
                 + "\n    ".join(problems)
                 + "\n  The inputs changed after registration (or were never "
                   "registered); refusing to run.")
    for name, sha in actual.items():
        print(f"[sha] {name}: {sha[:16]}… matches the registration")


def adjudicate(tok: str, err, d) -> str:
    if tok in ("降", "升", "平"):
        if d is None or not np.isfinite(d):
            return "not adjudicated (no Δerr)"
        if tok == "平":
            return "HIT" if abs(d) <= TIE + EPS else "MISS"
        if abs(d) <= TIE + EPS:
            return "TIE"
        return "HIT" if (d < 0) == (tok == "降") else "MISS"
    x = float(tok[1:])
    if err is None or not np.isfinite(err):
        return "not adjudicated (no err)"
    ok = err >= x - EPS if tok[0] == "≥" else err <= x + EPS
    return "HIT" if ok else "MISS"


def row_key(r: dict) -> str:
    return "|".join(str(r[c]) for c in ("decision", "dataset", "judge", "arm",
                                        "grid"))


def apply_registration(rows: list[dict], reg: dict | None) -> list[tuple]:
    """Suffix verdicts of registered rows; returns backfill tuples."""
    if not reg:
        return []
    preds = reg["preds"]
    out = []
    for r in rows:
        k = row_key(r)
        if k not in preds:
            continue
        tok = preds[k]
        if (r["decision"].startswith("D2") and r["dataset"] != "dl22"
                and r["arm"] in D2_INSAMPLE_ARMS):
            v = "not adjudicated (in-sample)"
        else:
            v = adjudicate(tok, r["err"], r["d_vs_RAW"])
        r["verdict"] = f"{r['verdict']}|{v}"
        out.append((k, tok, r["err"], r["d_vs_RAW"], r["d_ci"], v))
    return out


def print_backfill(bf: list[tuple], reg: dict | None, produced_keys: set,
                   owns) -> None:
    print()
    print("  BACKFILL — registered predictions (point estimate decides; "
          f"tie band {TIE}; CI never changes a verdict)")
    if not reg:
        print("    (no registration parsed)")
        return
    for k, tok, err, d, dci, v in bf:
        print(f"    DE[{k}]  预测 {tok}  err {_f(err, sign=False)}  Δerr "
              f"{_f(d)}  90% CI [{dci}]  -> {v}")
    missing = [k for k in reg["preds"] if owns(k) and k not in produced_keys]
    for k in missing:
        print(f"    DE[{k}]  registered but NOT produced by this run -> "
              f"not adjudicated (check the key spelling)")
    tally = defaultdict(int)
    for *_x, v in bf:
        tally[v.split(" ")[0]] += 1
    print(f"    tally: {dict(tally)}")


# ═══════════════════════════════════════════════════════════════════════════
# Long summary rows
# ═══════════════════════════════════════════════════════════════════════════

def make_rows(decision: str, ds: str, judge: str, grid: str, pts: dict,
              reps: dict, arms) -> list[dict]:
    """pts: arm -> point err; reps: arm -> replicate (or split) errs."""
    rows = []
    for arm in arms:
        if arm not in pts:
            continue
        e = pts[arm]
        lo, hi = ci90(reps[arm])
        if arm == "RAW":
            d, dci, v = None, "", "ref"
        else:
            d = float(e - pts["RAW"]) if "RAW" in pts else None
            dl, dh = ci90(np.asarray(reps[arm]) - np.asarray(reps["RAW"])) \
                if "RAW" in reps else (None, None)
            dci = "" if dl is None else f"{dl:+.6f},{dh:+.6f}"
            v = "descr" if decision.split(":")[-1] in DESCRIPTIVE \
                else dir_verdict(d)
        rows.append({"decision": decision, "dataset": ds, "judge": judge,
                     "arm": arm, "grid": grid, "err": float(e),
                     "ci_lo": lo, "ci_hi": hi, "d_vs_RAW": d, "d_ci": dci,
                     "verdict": v})
    return rows


def _fmt_csv(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return "" if not np.isfinite(v) else f"{v:.6f}"
    return str(v)


def write_summary(data_dir: Path, sub: str, rows: list[dict],
                  replace_datasets=None) -> Path:
    od = out_dir(data_dir)
    p = od / f"de_summary_{sub}.csv"
    old = []
    if replace_datasets is not None and p.exists():
        with open(p, newline="", encoding="utf-8") as f:
            old = [r for r in csv.DictReader(f)
                   if r["dataset"] not in set(replace_datasets)]
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLS)
        for r in old:
            w.writerow([r[c] for c in CSV_COLS])
        for r in rows:
            w.writerow([_fmt_csv(r[c]) for c in CSV_COLS])
    # rebuild the long de_summary.csv from every per-subcommand file
    allp = od / "de_summary.csv"
    with open(allp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLS)
        for s in SUBS:
            sp = od / f"de_summary_{s}.csv"
            if not sp.exists():
                continue
            with open(sp, newline="", encoding="utf-8") as g:
                for r in csv.DictReader(g):
                    w.writerow([r[c] for c in CSV_COLS])
    print(f"[csv] wrote {p} and rebuilt {allp}")
    return allp


def print_rows(rows: list[dict], title: str) -> None:
    print()
    print(f"  {title}")
    print(f"    {'decision':<11} {'judge':<30} {'grid':<12} {'arm':<14} "
          f"{'err':>7} {'90% CI':>17} {'Δ vs RAW':>9} {'Δ 90% CI':>19} verdict")
    for r in rows:
        ci = ("" if r["ci_lo"] is None else
              f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]")
        dci = "" if not r["d_ci"] else "[" + ",".join(
            f"{float(x):+.3f}" for x in r["d_ci"].split(",")) + "]"
        print(f"    {r['decision']:<11} {r['judge'][:30]:<30} "
              f"{r['grid'][:12]:<12} {r['arm']:<14} {r['err']:>7.4f} "
              f"{ci:>17} {_f(r['d_vs_RAW']):>9} {dci:>19} {r['verdict']}")


def base_meta(sub: str, reg: dict | None, shas: dict, extra: dict) -> dict:
    return {"script": f"scripts/decision_errors.py {sub}",
            "git_hash": b4.git_hash(), "code_dirty": b6.code_dirty(),
            "registration_commit": (reg or {}).get("commit"),
            "blind_status": "registered" if reg else "selftest/unregistered",
            "input_sha256": shas, "ci": "90% percentile", "tie": TIE,
            "adjudication_rules": ADJUDICATION_RULES,
            "timestamp_utc": now_utc(), **extra}


# ═══════════════════════════════════════════════════════════════════════════
# Label construction shared by D2 / D3 / local-calib (batch6 universe)
# ═══════════════════════════════════════════════════════════════════════════

def b6_labels_path(data_dir: Path, ds: str) -> Path:
    """Path of the stored batch6 labels (same lookup order as batch6's
    load_stored_labels) WITHOUT reading it — paste must not load labels."""
    base = data_dir / "derived" / f"batch6_labels_{ds}"
    for suf in (".parquet", ".pkl"):
        if base.with_suffix(suf).exists():
            return base.with_suffix(suf)
    return base.with_suffix(".parquet")


def cell_labels(model: str, ds: str, df: pd.DataFrame, qrels: dict,
                rules_all: dict, flagships, data_dir: Path, stored,
                frozen_pairs: dict, seed: int = SEED) -> dict:
    """Recompute batch6's universe + RAW/FROZEN/REFIT for one (judge, ds),
    hard-check them against the stored batch6_labels file (batch6
    check_stored, as `followup` does) and add SHIFT_FROZEN."""
    d, restricted = b6.scope_rows(df, model, ds, flagships, data_dir,
                                  frozen_pairs)
    if d.empty:
        return {"model": model, "dataset": ds, "error": f"no {ds} rows"}
    rule = b6.frozen_rule(rules_all, model, ds)
    if rule is None:
        return {"model": model, "dataset": ds,
                "error": f"no frozen {b6.FAMILY_OF[ds]} rule"}
    if rule.get("delta_hat") is None:
        sys.exit(f"ERROR: frozen rule for '{model}' @ {ds} has no delta_hat — "
                 f"SHIFT_FROZEN is undefined.")
    cell = b6.prepare_cell(model, ds, d, qrels, seed)
    if "error" in cell:
        return {"model": model, "dataset": ds, "error": cell["error"]}
    cnt = cell["cnt"]
    dial = int(rule["dial_level"])
    mono = tuple(int(x) for x in rule["monotone_map"])
    labels = {"RAW": b6.labels_from_counts(cnt[2], b4.IDENTITY_MAP),
              "FROZEN": b6.labels_from_counts(cnt[dial - 1], mono)}
    lab, _rules, _pf, why = b6.crossfit(cell, seed)
    if lab is None:
        return {"model": model, "dataset": ds, "error": f"crossfit: {why}"}
    labels["REFIT"] = lab
    status = b6.check_stored(stored, model, cell, labels)
    if status.startswith("not checked"):
        sys.exit(f"ERROR: '{model}' @ {ds}: {status} — the stored "
                 f"batch6_labels file is this study's registered input.")
    l3_mean = (cnt[2] @ np.arange(4, dtype=float)) / cnt[2].sum(axis=1)
    labels["SHIFT_FROZEN"] = rhu_vec(l3_mean - float(rule["delta_hat"]))
    seeds = {}
    s = stored[stored.model_id == model]
    for c in s.columns:
        if c.startswith("REFIT_seed"):
            seeds[c] = s[c].to_numpy(dtype=np.int64)
    return {"model": model, "dataset": ds, "rule": rule,
            "noop": b6.is_noop_rule(rule),
            "in_sample": b6.in_sample_note(rule, ds),
            "flagship": restricted, "universe": cell["universe"],
            "qid": [k[0] for k in cell["universe"]], "human": cell["h"],
            "labels": labels, "l3_mean": l3_mean, "refit_seeds": seeds,
            "check": status}


def load_cells(args, datasets, judges, need_labels=True) -> tuple:
    """Shared loader: judgments + qrels + rules + stored labels per dataset."""
    data_dir = b5.resolve_data_dir(args.data_dir)
    rules_all, rules_path, rsha = b6.load_rules(data_dir, args.rules)
    jpath = Path(args.judgments) if args.judgments else \
        data_dir / "derived" / "judgments.parquet"
    df = b6.load_judgments(jpath)
    df = df[df.dataset.astype(str).isin(datasets)]
    qrels = b6.load_human(datasets, args.qrels_json)
    flagships = b6._flagships(getattr(args, "flagships", None))
    out = {}
    frozen_pairs: dict = {}
    for ds in datasets:
        stored, sp = b6.load_stored_labels(data_dir, ds)
        if stored is None:
            sys.exit(f"ERROR: {sp} not found — run batch6 `run` first.")
        out[ds] = [cell_labels(m, ds, df, qrels, rules_all, flagships,
                               data_dir, stored, frozen_pairs)
                   for m in judges]
    return out, rules_all, rules_path, rsha, data_dir


# ═══════════════════════════════════════════════════════════════════════════
# D2 — per-query pool decisions
# ═══════════════════════════════════════════════════════════════════════════

def d2_indicators(qcode: np.ndarray, nq: int, h: np.ndarray, labels: dict,
                  grid) -> dict:
    """(decision, grid) -> {arm: per-query 0/1 error indicator}."""
    nH = np.bincount(qcode, weights=(h >= REL).astype(float), minlength=nq)
    nL = {a: np.bincount(qcode, weights=(lab >= REL).astype(float),
                         minlength=nq) for a, lab in labels.items()}
    out = {}
    for K in grid:
        key = ("D2a", "K=1") if K == 1 else ("D2b", f"K={K}")
        vh = nH >= K
        out[key] = {a: ((n >= K) != vh).astype(float) for a, n in nL.items()}
    out[("D2c", f"tol={BUDGET_TOL}")] = {
        a: (np.abs(n - nH) / np.maximum(nH, 1.0) > BUDGET_TOL + EPS)
        .astype(float) for a, n in nL.items()}
    return out


def boot_weights_from_idx(idx: np.ndarray, nq: int) -> np.ndarray:
    return np.stack([np.bincount(r, minlength=nq) for r in idx]).astype(float)


def d2_cell(cl: dict, B: int, seed: int) -> dict:
    ds = cl["dataset"]
    qn = sorted(set(cl["qid"]))
    qmap = {q: i for i, q in enumerate(qn)}
    qcode = np.array([qmap[q] for q in cl["qid"]], dtype=np.int64)
    nq = len(qn)
    h = np.asarray(cl["human"], dtype=np.int64)
    labels = {a: cl["labels"][a] for a in D2_ARMS}
    ind = d2_indicators(qcode, nq, h, labels, K_GRID[ds])
    W = boot_weights_from_idx(b6.boot_indices(nq, B, seed), nq)
    pts, reps = {}, {}
    for key, e in ind.items():
        pts[key] = {a: float(v.mean()) for a, v in e.items()}
        reps[key] = {a: W @ v / nq for a, v in e.items()}
    seed_pts = {}
    for c, lab in cl["refit_seeds"].items():
        ie = d2_indicators(qcode, nq, h, {"R": lab}, K_GRID[ds])
        for key, e in ie.items():
            seed_pts.setdefault(f"{key[0]}|{key[1]}", []).append(
                float(e["R"].mean()))
    nH = np.bincount(qcode, weights=(h >= REL).astype(float), minlength=nq)
    corpus = {a: float((labels[a] >= REL).sum() / max((h >= REL).sum(), 1))
              for a in labels}
    return {"model": cl["model"], "dataset": ds, "n_pairs": len(h),
            "n_queries": nq, "noop": cl["noop"], "in_sample": cl["in_sample"],
            "flagship": cl["flagship"], "check": cl["check"],
            "rule": {k: cl["rule"].get(k) for k in
                     ("dial_level", "monotone_map", "delta_hat", "datasets")},
            "labels_identical_FROZEN_RAW": bool(np.array_equal(
                labels["FROZEN"], labels["RAW"])),
            "pts": pts, "reps": reps,
            "refit_seed_err": {k: {"mean": float(np.mean(v)),
                                   "min": float(np.min(v)),
                                   "max": float(np.max(v)), "n": len(v)}
                               for k, v in seed_pts.items()},
            "descriptive": {
                "mean_n_human_rel_per_query": float(nH.mean()),
                "corpus_rel_ratio_vs_human": corpus,
                "queries_with_0_human_rel": int((nH == 0).sum())}}


def noop_violations_d2(res: dict) -> list[str]:
    if not res.get("noop"):
        return []
    bad = []
    if not res["labels_identical_FROZEN_RAW"]:
        bad.append("labels")
    for key, p in res["pts"].items():
        if p["FROZEN"] != p["RAW"]:
            bad.append(f"point:{key}")
        if not np.array_equal(res["reps"][key]["FROZEN"],
                              res["reps"][key]["RAW"]):
            bad.append(f"reps:{key}")
        dl, dh = ci90(res["reps"][key]["FROZEN"] - res["reps"][key]["RAW"])
        if dl != 0.0 or dh != 0.0:
            bad.append(f"dci:{key}")
    return bad


def enforce_noop(named: list[tuple[str, list[str]]], what: str) -> None:
    viol = [(n, v) for n, v in named if v]
    print(f"\n  no-op control ({what}): {len(named)} identity@L3 unit(s) — "
          f"corrected ≡ RAW bit-exactly: {'OK' if not viol else 'VIOLATED'}")
    if viol:
        sys.exit("\nERROR: no-op invariant VIOLATED —\n"
                 + "".join(f"    {n}: {v[:8]}{' ...' if len(v) > 8 else ''}\n"
                           for n, v in viol)
                 + "  identity@L3 must leave every decision error unchanged. "
                   "This is a BUG, not a result. Nothing was written.")


def d2_rows(res: dict) -> list[dict]:
    rows = []
    for (dec, grid), p in res["pts"].items():
        rows += make_rows(dec, res["dataset"], res["model"], grid, p,
                          res["reps"][(dec, grid)], D2_ARMS)
    return rows


def cmd_d2(args) -> None:
    datasets = parse_datasets(args.datasets)
    bad = [d for d in datasets if d not in D2_DATASETS]
    if bad:
        sys.exit(f"ERROR: d2-pool is defined on {list(D2_DATASETS)}; got {bad}.")
    reg = de_gate("dl22" in datasets, args.unseal_dl22,
                  predictions_path(args.predictions))           # BEFORE data
    data_dir = b5.resolve_data_dir(args.data_dir)
    _rules, rpath, rsha = b6.load_rules(data_dir, args.rules)
    shas = {"correction_rules.json": rsha}
    for ds in datasets:
        p = b6_labels_path(data_dir, ds)
        if not p.exists():
            sys.exit(f"ERROR: {p} not found.")
        shas[p.name] = b6.sha256_file(p)
    check_shas(reg, shas)
    run_d2(args, datasets, reg, shas)


def run_d2(args, datasets, reg, shas) -> list[dict]:
    judges = b6._judges(args.judges)
    cells, *_r, data_dir = load_cells(args, datasets, judges)
    print("=" * 100)
    print(f"  DE / D2 — per-query pool decisions  datasets={datasets}  "
          f"B={args.b} seed={args.seed}  relevant = grade >= {REL}")
    print("=" * 100)
    results, rows = [], []
    for ds in datasets:
        print(f"\n  [{ds}]  K grid {K_GRID[ds]}  budget tol {BUDGET_TOL}  "
              + {"dl22": "held-out (frozen msmarco rule never saw dl22) — "
                         "PRIMARY, adjudicated",
                 "dl21": "FROZEN/SHIFT_FROZEN PARTIALLY in-sample — report "
                         "only",
                 "antique": "FROZEN/SHIFT_FROZEN FULLY in-sample — report "
                            "only"}[ds])
        for cl in cells[ds]:
            if "error" in cl:
                print(f"    {cl['model']}: SKIPPED ({cl['error']})")
                results.append({"model": cl["model"], "dataset": ds,
                                "error": cl["error"]})
                continue
            r = d2_cell(cl, args.b, args.seed)
            results.append(r)
            print(f"    {r['model']:<32} pairs {r['n_pairs']:>5} queries "
                  f"{r['n_queries']:>3}  rule L{r['rule']['dial_level']} "
                  f"{r['rule']['monotone_map']} Δ̂={r['rule']['delta_hat']:+.4f}"
                  f"{'  [no-op]' if r['noop'] else ''}"
                  f"{'  [flagship 40%]' if r['flagship'] else ''}  "
                  f"stored check: {r['check']}")
            rows += d2_rows(r)
    ok = [r for r in results if "error" not in r]
    if not ok:
        sys.exit("ERROR: no (judge, dataset) cell produced a result.")
    enforce_noop([(f"{r['model']}@{r['dataset']}", noop_violations_d2(r))
                  for r in ok if r["noop"]], "D2")
    bf = apply_registration(rows, reg)
    print_rows(rows, "D2 error rates (fraction of queries whose verdict "
                     "differs from HUMAN)")
    print_backfill(bf, reg, {row_key(r) for r in rows},
                   lambda k: k.split("|")[0] in ("D2a", "D2b", "D2c")
                   and k.split("|")[1] in datasets)
    od = out_dir(data_dir)
    jp = od / "de_d2-pool.json"
    old = json.loads(jp.read_text()) if jp.exists() else {}
    per_ds = old.get("datasets", {})
    for ds in datasets:
        per_ds[ds] = [_d2_json(r) for r in results if r["dataset"] == ds]
    write_json(jp, {"meta": base_meta("d2-pool", reg, shas,
                                      {"B": args.b, "seed": args.seed,
                                       "k_grid": K_GRID,
                                       "budget_tol": BUDGET_TOL}),
                    "datasets": per_ds})
    write_summary(data_dir, "d2-pool", rows, replace_datasets=datasets)
    return rows


def _d2_json(r: dict) -> dict:
    if "error" in r:
        return r
    o = {k: v for k, v in r.items() if k not in ("pts", "reps")}
    o["point"] = {f"{k[0]}|{k[1]}": v for k, v in r["pts"].items()}
    o["ci90"] = {f"{k[0]}|{k[1]}": {a: ci90(x) for a, x in v.items()}
                 for k, v in r["reps"].items()}
    return o


# ═══════════════════════════════════════════════════════════════════════════
# D1 — system decisions (shared engine for dl21 B5 pool and dl22 students)
# ═══════════════════════════════════════════════════════════════════════════

def verdict3(d: np.ndarray, m: float) -> np.ndarray:
    """+1 'first beats second by >= m', -1 the reverse, 0 neither."""
    return np.where((d >= m - EPS) & (d > 0), 1,
                    np.where((d <= -m + EPS) & (d < 0), -1, 0))


class D1Engine:
    """Decision errors from a value tensor V[set, system, query].

    Set 0 is HUMAN (the truth).  M[set, query] (0/1) marks the queries a
    label set scores (batch7 drops IDCG = 0 queries; batch5 keeps them as 0,
    M = 1).  ``errors(w)`` takes per-query weights (bootstrap counts, or a
    held-out indicator) and recomputes the truth inside, so every replicate is
    self-contained.
    """

    def __init__(self, V: np.ndarray, M: np.ndarray | None = None,
                 tau_grid=TAU_GRID, margins=MARGINS, alpha=ALPHA,
                 sig: bool = True):
        self.V = np.asarray(V, dtype=float)
        S, N, Q = self.V.shape
        self.M = np.ones((S, Q)) if M is None else np.asarray(M, dtype=float)
        self.tau = np.asarray(tau_grid, dtype=float)
        self.margins = tuple(margins)
        self.alpha = alpha
        self.sig = sig
        self.off = ~np.eye(N, dtype=bool)
        self.I, self.J = np.triu_indices(N, 1)
        if sig:
            self.D = self.V[:, self.I, :] - self.V[:, self.J, :]

    def means(self, w: np.ndarray) -> np.ndarray:
        wm = self.M * np.asarray(w, dtype=float)[None, :]
        den = wm.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (self.V * wm[:, None, :]).sum(axis=2) / den[:, None]

    def sig_verdicts(self, w: np.ndarray) -> np.ndarray:
        """Paired t-test (two-sided) verdict per unordered pair, per set."""
        from scipy.stats import t as tdist
        wm = self.M * np.asarray(w, dtype=float)[None, :]
        n = wm.sum(axis=1)                                   # (S,)
        s1 = (self.D * wm[:, None, :]).sum(axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = s1 / n[:, None]
            dev = self.D - mean[:, :, None]
            var = (wm[:, None, :] * dev * dev).sum(axis=2) / (n[:, None] - 1)
            t = mean / np.sqrt(var / n[:, None])
            p = 2.0 * tdist.sf(np.abs(t), (n - 1)[:, None])
        sig = np.isfinite(p) & (p < self.alpha)
        return np.where(sig & (mean > 0), 1, np.where(sig & (mean < 0), -1, 0))

    def errors(self, w: np.ndarray, detail: bool = False) -> dict:
        m = self.means(w)                                      # (S, N)
        H = m[0]
        out = {}
        truth = H[:, None] > H[None, :]
        pred = m[:, :, None] > H[None, None, :]
        out[("D1a", "-")] = ((pred != truth[None]) & self.off[None]).sum(
            axis=(1, 2)) / self.off.sum()
        rank = 1 + ((H[None, None, :] > m[:, :, None]) & self.off[None]).sum(2)
        out[("D1a_place", "-")] = np.abs(rank - rank[0:1]).mean(axis=1)
        clr = m[:, :, None] >= self.tau[None, None, :] - EPS
        per_tau = (clr != clr[0:1]).mean(axis=1)               # (S, T)
        out[("D1b", f"tau={self.tau[0]:.2f}..{self.tau[-1]:.2f}")] = \
            per_tau.mean(axis=1)
        best = m.max(axis=1)
        bclr = best[:, None] >= self.tau[None, :] - EPS
        out[("D1b_best", f"tau={self.tau[0]:.2f}..{self.tau[-1]:.2f}")] = \
            (bclr != bclr[0:1]).mean(axis=1)
        dd = m[:, self.I] - m[:, self.J]
        for mg in self.margins:
            v = verdict3(dd, mg)
            out[("D1c", f"m={mg:g}")] = (v != v[0:1]).mean(axis=1)
        if self.sig:
            v = self.sig_verdicts(w)
            out[("D1c_sig", f"alpha={self.alpha:g}")] = (v != v[0:1]).mean(1)
        if detail:
            out["_per_tau"] = per_tau
            out["_means"] = m
        return out


def bootstrap_decisions(nd: np.ndarray, fn, B: int = B_D1,
                        seed: int = SEED) -> dict:
    """batch5.paired_bootstrap pattern: queries are the resampling unit, ONE
    resampled query set per replicate shared by every label set, truth
    recomputed inside ``fn``.  The replicate draw is batch6.boot_indices
    (same RNG stream as batch5's loop); the resample is passed to ``fn`` as
    per-query counts (identical to indexing nd[:, :, idx])."""
    nq = nd.shape[-1]
    idx = b6.boot_indices(nq, B, seed)
    acc = defaultdict(list)
    for r in idx:
        for k, v in fn(np.bincount(r, minlength=nq).astype(float)).items():
            if not (isinstance(k, str) and k.startswith("_")):
                acc[k].append(v)
    return {k: np.stack(v) for k, v in acc.items()}


def d1_rows(ds: str, judge: str, names: list[str], pts: dict, reps: dict,
            prefix: str = "") -> list[dict]:
    rows = []
    arms = [a for a in names if a != "HUMAN"]
    for key, p in pts.items():
        if isinstance(key, str):
            continue
        dec, grid = key
        rows += make_rows(prefix + dec, ds, judge, grid,
                          {a: float(p[i]) for i, a in enumerate(names)
                           if a != "HUMAN"},
                          {a: reps[key][:, i] for i, a in enumerate(names)
                           if a != "HUMAN"}, arms)
    return rows


# ── B5 pool on dl21 ─────────────────────────────────────────────────────────

def l3_means(d: pd.DataFrame) -> dict:
    """(qid, docid) -> mean over ALL usable L3 records (judge_labels' rows)."""
    ok = d[(d.politeness_level == 3) & d.parse_ok & d.score.notna()]
    g = ok.assign(qid=ok.qid.astype(str), docid=ok.docid.astype(str)) \
        .groupby(["qid", "docid"]).score.agg(["sum", "count"])
    return {k: float(s) / float(c) for k, s, c in
            zip(g.index, g["sum"].to_numpy(float), g["count"].to_numpy(float))}


def shift_labels(means: dict, keys, delta: float) -> dict:
    keys = list(keys)
    x = np.array([means[k] for k in keys]) - float(delta)
    return dict(zip(keys, rhu_vec(x).tolist()))


def b5_setup(model: str, d: pd.DataFrame, rule: dict, human: dict,
             runs: dict, lodo_rule: dict | None) -> dict:
    """The B5 universe (human ∩ RAW ∩ CORR) and every label set on it."""
    dial = int(rule["dial_level"])
    mono = tuple(int(x) for x in rule["monotone_map"])
    raw = b5.judge_labels(d, 3, mono=None)
    corr = b5.judge_labels(d, dial, mono=mono)
    keys = sorted(set(human) & set(raw) & set(corr))
    means3 = l3_means(d)
    sets = {"HUMAN": {k: human[k] for k in keys},
            "RAW": {k: raw[k] for k in keys},
            "CORR": {k: corr[k] for k in keys},
            "SHIFT_FROZEN": shift_labels(means3, keys, rule["delta_hat"])}
    note = None
    if lodo_rule is not None:
        ld = int(lodo_rule["dial_level"])
        lm = tuple(int(x) for x in lodo_rule["monotone_map"])
        cl = b5.judge_labels(d, ld, mono=lm)
        miss = [k for k in keys if k not in cl]
        if miss:
            note = (f"CORR_LODO not evaluable: {len(miss)} universe pair(s) "
                    f"lack an L{ld} label (B5 top-up at L1/L3 only for 3 "
                    f"judges, L3 only for gemini-3.8)")
        else:
            sets["CORR_LODO"] = {k: cl[k] for k in keys}
        sets["SHIFT_LODO"] = shift_labels(means3, keys, lodo_rule["delta_hat"])
    queries = sorted({q for q, _ in keys})
    nd = b5.ndcg_matrix(runs, sets, queries, k=b5.TOP_K)
    return {"keys": keys, "queries": queries, "names": list(sets),
            "sets": sets, "nd": nd, "means3": means3, "lodo_note": note,
            "systems": list(runs), "noop": b6.is_noop_rule(rule),
            "lodo_noop": (lodo_rule is not None
                          and b6.is_noop_rule(lodo_rule))}


def read_b5_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(ln for ln in f if not ln.startswith("#")))


def b5_check(rows: list[dict], model: str, systems: list[str],
             means: np.ndarray, names: list[str]) -> float:
    """Recomputed HUMAN/RAW/CORR system means vs the transcribed B5 table."""
    j = short_judge(model)
    sub = {r["system"]: r for r in rows if r["judge"] == j}
    if not sub:
        sys.exit(f"ERROR: b5 check: judge '{j}' absent from the B5 table.")
    if set(sub) != set(systems):
        sys.exit(f"ERROR: b5 check: system set differs for '{j}': table "
                 f"{sorted(set(sub) ^ set(systems))[:6]} ...")
    worst = 0.0
    for si, s in enumerate(systems):
        for col in ("HUMAN", "RAW", "CORR"):
            diff = abs(float(sub[s][col]) - means[names.index(col), si])
            worst = max(worst, diff)
            if diff > B5_TOL:
                sys.exit(f"ERROR: b5 check FAILED: {j} / {s} / {col}: "
                         f"recomputed {means[names.index(col), si]:.6f} vs "
                         f"table {sub[s][col]} (|diff| {diff:.2e} > {B5_TOL})."
                         f"\n  The recomputation does not reproduce B5; "
                         f"refusing to compute decisions on it.")
    return worst


def noop_violations_d1(st: dict, pts: dict, reps: dict, a: str,
                       b: str = "RAW") -> list[str]:
    names = st["names"]
    if a not in names:
        return []
    ia, ib = names.index(a), names.index(b)
    bad = []
    if st["sets"][a] != st["sets"][b]:
        bad.append("labels")
    if not np.array_equal(st["nd"][ia], st["nd"][ib]):
        bad.append("ndcg")
    for key, p in pts.items():
        if isinstance(key, str):
            continue
        if p[ia] != p[ib]:
            bad.append(f"point:{key}")
        if not np.array_equal(reps[key][:, ia], reps[key][:, ib]):
            bad.append(f"reps:{key}")
    return bad


def load_lodo(path: Path | None) -> tuple[dict | None, str | None]:
    if path is None or not path.exists():
        return None, None
    return json.loads(path.read_text(encoding="utf-8")), b6.sha256_file(path)


def cmd_d1_b5(args) -> None:
    reg = de_gate(False, False, predictions_path(args.predictions))
    data_dir = b5.resolve_data_dir(args.data_dir)
    _r, _p, rsha = b6.load_rules(data_dir, args.rules)
    shas = {"correction_rules.json": rsha}
    lodo_path = Path(args.lodo) if args.lodo else \
        out_dir(data_dir) / "rules_lodo.json"
    if lodo_path.exists():
        shas["rules_lodo.json"] = b6.sha256_file(lodo_path)
    elif "rules_lodo.json" in reg["shas"]:
        sys.exit(f"ERROR: rules_lodo.json is registered but {lodo_path} is "
                 f"missing.")
    b5p = None
    if not args.no_b5_check:
        b5p = Path(args.b5_check) if args.b5_check else \
            out_dir(data_dir) / "b5_per_system_ndcg.csv"
        if not b5p.exists():
            sys.exit(f"ERROR: B5 integrity table {b5p} not found. Copy "
                     f"politeness-llm-internal/paper/sigir27/tables/"
                     f"b5_per_system_ndcg.csv there (or pass --b5-check).")
        shas["b5_per_system_ndcg.csv"] = b6.sha256_file(b5p)
    check_shas(reg, shas)
    run_d1_b5(args, reg, shas, lodo_path, b5p)


def run_d1_b5(args, reg, shas, lodo_path: Path, b5p: Path | None) -> dict:
    data_dir = b5.resolve_data_dir(args.data_dir)
    rules_all, _rp, _rs = b6.load_rules(data_dir, args.rules)
    lodo_all, _ls = load_lodo(lodo_path)
    runs = b5.load_runs(data_dir / "runs" / B5_DATASET)
    jpath = Path(args.judgments) if args.judgments else \
        data_dir / "derived" / "judgments.parquet"
    df = b5.guard_dataset(b6.load_judgments(jpath))
    human = b5.human_labels(b6.load_human([B5_DATASET], args.qrels_json))
    b5rows = read_b5_csv(b5p) if b5p is not None else None
    judges = [s.strip() for s in args.judges.split(",")] if args.judges \
        else list(B5_JUDGES)
    print("=" * 100)
    print(f"  DE / D1 — system decisions on {B5_DATASET} (B5 pool, "
          f"{len(runs)} systems)  B={args.b} seed={args.seed}")
    print("=" * 100)
    print("  CORR on dl21 is PARTIALLY in-sample (msmarco rule fitted on "
          "dl19+dl20+dl21); CORR_LODO / SHIFT_LODO use the dl19+dl20 rule.")
    print("  BLINDNESS: per-system point estimates were already visible (B5 "
          "table / Fig. 5) -> non-blind; bootstrap and LODO arms are new.")
    results, rows, noop_named = [], [], []
    for model in judges:
        d = df[df.model_id == model]
        rule = (rules_all.get("rules", {}).get(model, {}) or {}).get("msmarco")
        if d.empty or rule is None:
            print(f"\n  {model}: SKIPPED ({'no rows' if d.empty else 'no rule'})")
            continue
        lrule = ((lodo_all or {}).get("rules", {}).get(model, {}) or {}) \
            .get("msmarco")
        st = b5_setup(model, d, rule, human, runs, lrule)
        eng = D1Engine(st["nd"])
        full = eng.errors(np.ones(len(st["queries"])), detail=True)
        means = full.pop("_means")
        per_tau = full.pop("_per_tau")
        worst = None
        if b5rows is not None:
            worst = b5_check(b5rows, model, st["systems"], means, st["names"])
        reps = bootstrap_decisions(st["nd"], eng.errors, args.b, args.seed)
        names = st["names"]
        if st["noop"]:
            noop_named.append((f"{model} CORR", noop_violations_d1(
                st, full, reps, "CORR")))
        if st["lodo_noop"]:
            noop_named.append((f"{model} CORR_LODO", noop_violations_d1(
                st, full, reps, "CORR_LODO")))
        offs = {a: float((means[i] - means[0]).mean())
                for i, a in enumerate(names) if a != "HUMAN"}
        print(f"\n  {model}  rule L{rule['dial_level']} {rule['monotone_map']} "
              f"Δ̂={rule['delta_hat']:+.4f}{'  [no-op]' if st['noop'] else ''}"
              + (f"  LODO L{lrule['dial_level']} {lrule['monotone_map']} "
                 f"Δ̂={lrule['delta_hat']:+.4f}" if lrule else "  (no LODO)"))
        print(f"    universe {len(st['keys'])} pairs / {len(st['queries'])} "
              f"queries; b5 check: "
              + ("SKIPPED (--no-b5-check; UNVERIFIED)" if worst is None
                 else f"OK (max |diff| {worst:.1e} <= {B5_TOL})"))
        if st["lodo_note"]:
            print(f"    NOTE: {st['lodo_note']}")
        print("    mean NDCG offset vs HUMAN over systems (mechanistic rule): "
              + "  ".join(f"{a} {o:+.4f}" for a, o in offs.items()))
        r = d1_rows(B5_DATASET, model, names, full, reps)
        rows += r
        results.append({"model": model, "rule": rule, "lodo_rule": lrule,
                        "lodo_note": st["lodo_note"], "noop": st["noop"],
                        "n_pairs": len(st["keys"]),
                        "n_queries": len(st["queries"]),
                        "systems": st["systems"], "label_sets": names,
                        "b5_check_max_absdiff": worst,
                        "system_means": {a: means[i] for i, a in
                                         enumerate(names)},
                        "mean_offset_vs_human": offs,
                        "abs_offset_vs_human": {a: abs(o)
                                                for a, o in offs.items()},
                        "per_tau_point": {a: per_tau[i] for i, a in
                                          enumerate(names)},
                        "tau_grid": TAU_GRID,
                        "point": {f"{k[0]}|{k[1]}": dict(zip(names, v))
                                  for k, v in full.items()},
                        "ci90": {f"{k[0]}|{k[1]}": {a: ci90(v[:, i]) for i, a
                                                    in enumerate(names)}
                                 for k, v in reps.items()}})
    if not results:
        sys.exit("ERROR: no judge produced a result.")
    enforce_noop(noop_named, "D1-b5")
    bf = apply_registration(rows, reg)
    print_rows(rows, "D1 (dl21) decision error rates; D1a_place in ranks")
    print_backfill(bf, reg, {row_key(r) for r in rows},
                   lambda k: k.split("|")[0].startswith("D1")
                   and k.split("|")[1] == B5_DATASET)
    write_json(out_dir(data_dir) / "de_d1-b5.json",
               {"meta": base_meta("d1-b5", reg, shas,
                                  {"B": args.b, "seed": args.seed,
                                   "margins": MARGINS, "alpha": ALPHA,
                                   "blindness": "non-blind (B5 per-system "
                                   "point estimates already visible)"}),
                "judges": results})
    write_summary(data_dir, "d1-b5", rows)
    return {"rows": rows, "results": results}


# ═══════════════════════════════════════════════════════════════════════════
# Students (D3 and D1 on dl22)
# ═══════════════════════════════════════════════════════════════════════════

def scan_students(eval_root: Path, load: bool = True) -> tuple:
    """-> (units, excluded, universe_sha).  Units = registered + ablation
    sets (batch7_summary.SETS); '@tag' retrains and other files excluded."""
    units, excluded, shas = [], [], set()
    for p in sorted(eval_root.glob("*/*/*_dl22.json")):
        arm, slug = p.parent.parent.name, p.parent.name
        stem = p.name[: -len("_dl22.json")]
        set_name, _, seed = stem.rpartition("_s")
        if "@" in set_name:
            excluded.append((str(p.relative_to(eval_root)),
                             "exploratory '@tag' retrain"))
            continue
        if set_name not in B7_SETS or not seed.isdigit():
            excluded.append((str(p.relative_to(eval_root)),
                             "not a registered/ablation B7 set"))
            continue
        u = {"path": p, "arm": arm, "slug": slug, "set": set_name,
             "seed": int(seed)}
        if load:
            r = json.loads(p.read_text(encoding="utf-8"))
            if r.get("universe_sha256") != b7.universe_sha(r["pairs"]):
                sys.exit(f"ERROR: {p}: universe_sha256 does not match its own "
                         f"pairs.")
            u["result"] = r
            shas.add(r["universe_sha256"])
        else:
            m = re.search(r'"universe_sha256":\s*"([0-9a-f]{64})"',
                          p.read_text(encoding="utf-8"))
            if m:
                shas.add(m.group(1))
        units.append(u)
    if not units:
        sys.exit(f"ERROR: no B7 dl22 eval files under {eval_root}.")
    if len(shas) != 1:
        sys.exit(f"ERROR: B7 eval files disagree on the dl22 universe: "
                 f"{sorted(s[:12] for s in shas)}.")
    return units, excluded, shas.pop()


def _same_float(a, b, tol=1e-9) -> bool:
    a, b = float(a), float(b)
    return (np.isnan(a) and np.isnan(b)) or abs(a - b) <= tol


def verify_student(u: dict) -> None:
    """Stored point endpoints recomputed from the file's own pairs."""
    r = u["result"]
    qids, S = b7.per_query_stats(r["pairs"])
    pt = b7.aggregate(np.ones((1, len(qids))), S)
    for k in ("offset", "rel_rate_ratio", "ndcg10"):
        if not _same_float(pt[k][0], r["point"][k]):
            sys.exit(f"ERROR: {u['path']}: recomputed {k} {pt[k][0]!r} != "
                     f"stored {r['point'][k]!r}.")


def student_study(units: list[dict], cl: dict, evaluator: str, B: int,
                  seed: int) -> dict:
    """Per-(student, test label set) batch7 stats on B7 ∩ evaluator universe."""
    ev_keys = {(q, d): i for i, (q, d) in enumerate(cl["universe"])}
    pairs0 = units[0]["result"]["pairs"]
    b7_h = {(str(q), str(d)): int(h) for q, d, _p, h in pairs0}
    keys = sorted(k for k in b7_h if k in ev_keys)
    if not keys:
        sys.exit(f"ERROR: B7 universe ∩ {evaluator} B6 universe is empty.")
    bad = [k for k in keys if b7_h[k] != int(cl["human"][ev_keys[k]])]
    if bad:
        sys.exit(f"ERROR: human grades differ between the B7 eval files and "
                 f"the batch6 labels on {len(bad)} pair(s), e.g. {bad[:3]}.")
    idx = np.array([ev_keys[k] for k in keys])
    test = {"HUMAN": np.array([b7_h[k] for k in keys], dtype=np.int64)}
    for T in D3_TEST_SETS[1:]:
        test[T] = np.asarray(cl["labels"][T])[idx]
    slug_e = b7.judge_slug(evaluator)
    use = [u for u in units if u["slug"] != slug_e]
    dropped = len(units) - len(use)
    S, qids0 = {T: [] for T in D3_TEST_SETS}, None
    for u in use:
        pred = {(str(q), str(d)): float(p) for q, d, p, _h in
                u["result"]["pairs"]}
        if set(pred) != set(b7_h):
            sys.exit(f"ERROR: {u['path']}: pair set differs from the other "
                     f"students.")
        for T in D3_TEST_SETS:
            qids, St = b7.per_query_stats(
                [(k[0], k[1], pred[k], int(g)) for k, g in zip(keys, test[T])])
            if qids0 is None:
                qids0 = qids
            elif qids != qids0:
                sys.exit("ERROR: query order differs across students.")
            S[T].append(St)
    nq = len(qids0)
    W = b7.boot_weights(nq, B, seed)
    one = np.ones((1, nq))
    pas_pt, pas_bt, comp_pt, comp_bt = {}, {}, {}, {}
    for T in D3_TEST_SETS:
        pp, pb, cp, cb = [], [], {"off": [], "ratio": []}, \
            {"off": [], "ratio": []}
        for St in S[T]:
            for w, store, cstore in ((one, pp, cp), (W, pb, cb)):
                a = b7.aggregate(w, St)
                off_ok = np.abs(a["offset"]) <= D3_OFF_TOL + EPS
                rr = a["rel_rate_ratio"]
                with np.errstate(invalid="ignore"):
                    rat_ok = np.isfinite(rr) & (rr >= D3_RATIO[0] - EPS) & \
                        (rr <= D3_RATIO[1] + EPS)
                store.append(off_ok & rat_ok)
                cstore["off"].append(off_ok)
                cstore["ratio"].append(rat_ok)
        pas_pt[T] = np.stack(pp, axis=-1)[0]                 # (n_st,)
        pas_bt[T] = np.stack(pb, axis=-1)                    # (B, n_st)
        comp_pt[T] = {c: np.stack(v, axis=-1)[0] for c, v in cp.items()}
        comp_bt[T] = {c: np.stack(v, axis=-1) for c, v in cb.items()}
    STAT_IDX = {c: i for i, c in enumerate(b7.STAT_COLS)}
    V = np.stack([np.stack([St[:, STAT_IDX["ndcg"]] for St in S[T]])
                  for T in D3_TEST_SETS])                    # (T, n_st, Q)
    Mk = np.stack([S[T][0][:, STAT_IDX["ndcg_ok"]] for T in D3_TEST_SETS])
    for T in D3_TEST_SETS:
        for St in S[T]:
            if not np.array_equal(St[:, STAT_IDX["ndcg_ok"]],
                                  Mk[D3_TEST_SETS.index(T)]):
                sys.exit("ERROR: NDCG query validity differs across students "
                         "under one label set.")
    return {"evaluator": evaluator, "units": use, "n_dropped_teacher": dropped,
            "n_pairs": len(keys), "n_queries": nq, "pass_pt": pas_pt,
            "pass_bt": pas_bt, "comp_pt": comp_pt, "comp_bt": comp_bt,
            "V": V, "M": Mk, "noop": cl["noop"], "rule": cl["rule"]}


def d3_errors(ss: dict) -> tuple[dict, dict, dict]:
    """(grid -> arm -> point), (grid -> arm -> reps), descriptive pass rates."""
    arms_idx = {"all": np.ones(len(ss["units"]), dtype=bool)}
    for a in STUDENT_ARMS:
        m = np.array([u["arm"] == a for u in ss["units"]])
        if m.any():
            arms_idx[a] = m
    pts, reps = {}, {}
    for dec, src_pt, src_bt in (
            ("D3", ss["pass_pt"], ss["pass_bt"]),
            ("D3_off", {T: v["off"] for T, v in ss["comp_pt"].items()},
             {T: v["off"] for T, v in ss["comp_bt"].items()}),
            ("D3_ratio", {T: v["ratio"] for T, v in ss["comp_pt"].items()},
             {T: v["ratio"] for T, v in ss["comp_bt"].items()})):
        for g, m in arms_idx.items():
            pts[(dec, g)] = {T: float((src_pt[T][m] != src_pt["HUMAN"][m])
                                      .mean()) for T in D3_TEST_SETS[1:]}
            reps[(dec, g)] = {T: (src_bt[T][:, m] != src_bt["HUMAN"][:, m])
                              .mean(axis=1) for T in D3_TEST_SETS[1:]}
    desc = defaultdict(lambda: defaultdict(list))
    for i, u in enumerate(ss["units"]):
        for T in D3_TEST_SETS:
            desc[f"{u['arm']}|{u['set']}"][T].append(bool(ss["pass_pt"][T][i]))
    pass_rate = {k: {T: float(np.mean(v)) for T, v in d.items()}
                 for k, d in desc.items()}
    return pts, reps, pass_rate


def students_setup(args) -> tuple:
    data_dir = b5.resolve_data_dir(args.data_dir)
    eval_root = Path(args.eval_root) if args.eval_root else \
        data_dir / "derived" / "batch7" / "eval"
    return data_dir, eval_root


def gate_students(args) -> tuple:
    reg = de_gate(True, args.unseal_dl22, predictions_path(args.predictions))
    data_dir, eval_root = students_setup(args)
    _r, _p, rsha = b6.load_rules(data_dir, args.rules)
    p = b6_labels_path(data_dir, "dl22")
    if not p.exists():
        sys.exit(f"ERROR: {p} not found.")
    _u, _x, usha = scan_students(eval_root, load=False)
    shas = {"correction_rules.json": rsha, p.name: b6.sha256_file(p),
            "b7_universe_dl22": usha}
    check_shas(reg, shas)
    return reg, shas


def load_student_studies(args) -> tuple:
    data_dir, eval_root = students_setup(args)
    units, excluded, usha = scan_students(eval_root, load=True)
    for u in units:
        verify_student(u)
    evaluators = [s.strip() for s in args.evaluators.split(",")] \
        if args.evaluators else list(EVALUATORS)
    cells, *_x = load_cells(args, ["dl22"], evaluators)
    studies = []
    for cl in cells["dl22"]:
        if "error" in cl:
            print(f"  evaluator {cl['model']}: SKIPPED ({cl['error']})")
            continue
        studies.append(student_study(units, cl, cl["model"], args.b,
                                     args.seed))
    print(f"  students: {len(units)} units (registered + ablation sets), "
          f"verified against their stored endpoints; excluded "
          f"{len(excluded)} file(s):")
    for f, why in excluded:
        print(f"    - {f}  ({why})")
    return data_dir, units, excluded, usha, studies


def cmd_d3(args) -> None:
    reg, shas = gate_students(args)
    run_d3(args, reg, shas)


def run_d3(args, reg, shas) -> dict:
    print("=" * 100)
    print(f"  DE / D3 — validating students on judge-labelled dl22 test data"
          f"  B={args.b} seed={args.seed}")
    print(f"  pass rule: |mean clip(pred) - mean label| <= {D3_OFF_TOL} AND "
          f"ratio@2 in [{D3_RATIO[0]}, {D3_RATIO[1]}]")
    print("=" * 100)
    data_dir, units, excluded, usha, studies = load_student_studies(args)
    rows, out, noop_named = [], [], []
    for ss in studies:
        pts, reps, pass_rate = d3_errors(ss)
        ev = ss["evaluator"]
        print(f"\n  evaluator {ev}{'  [PRIMARY]' if ev == EVALUATORS[0] else ''}"
              f"{'  [no-op rule]' if ss['noop'] else ''}: "
              f"{len(ss['units'])} students ({ss['n_dropped_teacher']} taught "
              f"by this judge excluded), {ss['n_pairs']} pairs / "
              f"{ss['n_queries']} queries")
        for (dec, g), p in pts.items():
            rows += make_rows(dec, "dl22", ev, g, p, reps[(dec, g)],
                              ("RAW", "FROZEN", "SHIFT_FROZEN"))
        if ss["noop"]:
            bad = []
            for key in pts:
                if pts[key]["FROZEN"] != pts[key]["RAW"] or not \
                        np.array_equal(reps[key]["FROZEN"], reps[key]["RAW"]):
                    bad.append(str(key))
            noop_named.append((f"{ev} FROZEN", bad))
        print("    pass rate under HUMAN test labels by (arm, training set):")
        for k in sorted(pass_rate):
            pr = pass_rate[k]
            print(f"      {k:<34} " + "  ".join(
                f"{T} {pr[T]:.2f}" for T in D3_TEST_SETS))
        out.append({"evaluator": ev, "n_students": len(ss["units"]),
                    "n_dropped_teacher": ss["n_dropped_teacher"],
                    "n_pairs": ss["n_pairs"], "n_queries": ss["n_queries"],
                    "noop": ss["noop"],
                    "point": {f"{k[0]}|{k[1]}": v for k, v in pts.items()},
                    "ci90": {f"{k[0]}|{k[1]}": {a: ci90(x) for a, x in
                                                v.items()}
                             for k, v in reps.items()},
                    "pass_rate_by_arm_set": pass_rate,
                    "students": [f"{u['arm']}/{u['slug']}/{u['set']}_s"
                                 f"{u['seed']}" for u in ss["units"]]})
    enforce_noop(noop_named, "D3")
    bf = apply_registration(rows, reg)
    print_rows(rows, "D3 error rates (fraction of students whose pass/fail "
                     "differs from the HUMAN-label verdict)")
    print_backfill(bf, reg, {row_key(r) for r in rows},
                   lambda k: k.split("|")[0].startswith("D3"))
    write_json(out_dir(data_dir) / "de_d3-validate.json",
               {"meta": base_meta("d3-validate", reg, shas,
                                  {"B": args.b, "seed": args.seed,
                                   "off_tol": D3_OFF_TOL,
                                   "ratio_band": D3_RATIO,
                                   "excluded_files": excluded,
                                   "b7_universe_sha256": usha,
                                   "seed_variance": "not in the CI (as B7)"}),
                "evaluators": out})
    write_summary(data_dir, "d3-validate", rows)
    return {"rows": rows, "results": out}


def cmd_d1_students(args) -> None:
    reg, shas = gate_students(args)
    run_d1_students(args, reg, shas)


def run_d1_students(args, reg, shas) -> dict:
    print("=" * 100)
    print(f"  DE / D1 on dl22 — students as systems  B={args.b} "
          f"seed={args.seed}")
    print("  CAVEAT (stated before running): students sit in a narrow NDCG "
          "band (~0.43-0.47),")
    print("  which inflates cross-scale (D1a) error for every label set.")
    print("=" * 100)
    data_dir, units, excluded, usha, studies = load_student_studies(args)
    rows, out, noop_named = [], [], []
    names = list(D3_TEST_SETS)
    for ss in studies:
        ev = ss["evaluator"]
        eng = D1Engine(ss["V"], ss["M"])
        full = eng.errors(np.ones(ss["n_queries"]), detail=True)
        means = full.pop("_means")
        full.pop("_per_tau")
        reps = bootstrap_decisions(ss["V"], eng.errors, args.b, args.seed)
        if ss["noop"]:
            ia, ib = names.index("FROZEN"), names.index("RAW")
            bad = [] if np.array_equal(ss["V"][ia], ss["V"][ib]) else ["ndcg"]
            for key, p in full.items():
                if p[ia] != p[ib] or not np.array_equal(reps[key][:, ia],
                                                        reps[key][:, ib]):
                    bad.append(str(key))
            noop_named.append((f"{ev} FROZEN", bad))
        offs = {a: float(np.nanmean(means[i] - means[0]))
                for i, a in enumerate(names) if a != "HUMAN"}
        spread = (float(np.nanmin(means[0])), float(np.nanmax(means[0])))
        print(f"\n  evaluator {ev}: {len(ss['units'])} student-systems, "
              f"{ss['n_queries']} queries; HUMAN NDCG band "
              f"[{spread[0]:.4f}, {spread[1]:.4f}]; mean offset vs HUMAN: "
              + "  ".join(f"{a} {o:+.4f}" for a, o in offs.items()))
        rows += d1_rows("dl22", ev, names, full, reps)
        out.append({"evaluator": ev, "n_systems": len(ss["units"]),
                    "n_queries": ss["n_queries"], "noop": ss["noop"],
                    "human_ndcg_band": spread, "mean_offset_vs_human": offs,
                    "point": {f"{k[0]}|{k[1]}": dict(zip(names, v))
                              for k, v in full.items()},
                    "ci90": {f"{k[0]}|{k[1]}": {a: ci90(v[:, i]) for i, a
                                                in enumerate(names)}
                             for k, v in reps.items()}})
    enforce_noop(noop_named, "D1-students")
    bf = apply_registration(rows, reg)
    print_rows(rows, "D1 (dl22, students as systems) decision error rates")
    print_backfill(bf, reg, {row_key(r) for r in rows},
                   lambda k: k.split("|")[0].startswith("D1")
                   and k.split("|")[1] == "dl22")
    write_json(out_dir(data_dir) / "de_d1-students.json",
               {"meta": base_meta("d1-students", reg, shas,
                                  {"B": args.b, "seed": args.seed,
                                   "excluded_files": excluded,
                                   "b7_universe_sha256": usha,
                                   "convention": "batch7 query_metrics: "
                                   "re-rank the judged pool, test labels as "
                                   "gains, IDCG=0 queries dropped per set"}),
                "evaluators": out})
    write_summary(data_dir, "d1-students", rows)
    return {"rows": rows, "results": out}


# ═══════════════════════════════════════════════════════════════════════════
# local-calib (Priority 5, optional)
# ═══════════════════════════════════════════════════════════════════════════

def calib_split(rng, q_pairs: np.ndarray, n: int) -> np.ndarray | None:
    """Whole random queries until >= n pairs; returns the calibration mask."""
    nq = len(q_pairs)
    cal = np.zeros(nq, dtype=bool)
    tot = 0
    for qi in rng.permutation(nq):
        cal[qi] = True
        tot += int(q_pairs[qi])
        if tot >= n:
            break
    if tot < n or (~cal).sum() < 2:
        return None
    return cal


def _split_summary(vals: dict) -> tuple[dict, dict]:
    return ({a: float(np.mean(v)) for a, v in vals.items()},
            {a: np.asarray(v) for a, v in vals.items()})


def lc_d2(cl: dict, n: int, R: int, seed: int) -> dict:
    ds = cl["dataset"]
    qn = sorted(set(cl["qid"]))
    qmap = {q: i for i, q in enumerate(qn)}
    qcode = np.array([qmap[q] for q in cl["qid"]], dtype=np.int64)
    nq = len(qn)
    h = np.asarray(cl["human"], dtype=np.int64)
    q_pairs = np.bincount(qcode, minlength=nq)
    rng = np.random.default_rng(seed)
    arms = ("RAW", "FROZEN", "SHIFT_FROZEN", "REFIT", f"SHIFT_n{n}")
    acc = defaultdict(lambda: defaultdict(list))
    n_ok = 0
    for _ in range(R):
        cal = calib_split(rng, q_pairs, n)
        if cal is None:
            continue
        cp = cal[qcode]
        dhat = float(cl["l3_mean"][cp].mean() - h[cp].mean())
        labs = {a: cl["labels"][a] for a in arms[:-1]}
        labs[arms[-1]] = rhu_vec(cl["l3_mean"] - dhat)
        ind = d2_indicators(qcode, nq, h, labs, K_GRID[ds])
        held = ~cal
        for key, e in ind.items():
            for a, v in e.items():
                acc[key][a].append(float(v[held].mean()))
        n_ok += 1
    return {"acc": acc, "n_splits": n_ok, "arms": arms}


def lc_d1(st: dict, n: int, R: int, seed: int, runs: dict) -> dict:
    q = st["queries"]
    qi = {x: i for i, x in enumerate(q)}
    pair_q = np.array([qi[k[0]] for k in st["keys"]])
    q_pairs = np.bincount(pair_q, minlength=len(q))
    h = np.array([st["sets"]["HUMAN"][k] for k in st["keys"]], dtype=float)
    m3 = np.array([st["means3"][k] for k in st["keys"]])
    base = ["HUMAN", "RAW", "CORR", "SHIFT_FROZEN"]
    ib = [st["names"].index(a) for a in base]
    rng = np.random.default_rng(seed)
    acc = defaultdict(lambda: defaultdict(list))
    cache: dict = {}
    n_ok = 0
    sname, mname = f"SHIFT_n{n}", f"METRIC_SHIFT_n{n}"
    names = base + [sname, mname]
    for _ in range(R):
        cal = calib_split(rng, q_pairs, n)
        if cal is None:
            continue
        cp = cal[pair_q]
        dhat = float(m3[cp].mean() - h[cp].mean())
        lab = rhu_vec(m3 - dhat)
        ck = lab.tobytes()
        if ck not in cache:
            cache[ck] = b5.ndcg_matrix(
                runs, {"S": dict(zip(st["keys"], lab.tolist()))}, q)[0]
        nd_raw, nd_h = st["nd"][st["names"].index("RAW")], st["nd"][0]
        c = float((nd_raw - nd_h)[:, cal].mean())
        V = np.concatenate([st["nd"][ib], cache[ck][None],
                            (nd_raw - c)[None]], axis=0)
        e = D1Engine(V).errors((~cal).astype(float))
        for key, v in e.items():
            for i, a in enumerate(names):
                if a != "HUMAN":
                    acc[key][a].append(float(v[i]))
        n_ok += 1
    return {"acc": acc, "n_splits": n_ok, "arms": names[1:]}


def lc_rows(ds: str, judge: str, res: dict) -> list[dict]:
    rows = []
    for key, vals in res["acc"].items():
        pts, reps = _split_summary(vals)
        rows += make_rows(f"LC:{key[0]}", ds, judge, key[1], pts, reps,
                          res["arms"])
    return rows


def cmd_local_calib(args) -> None:
    studies = [s.strip() for s in args.studies.split(",") if s.strip()]
    datasets = parse_datasets(args.datasets) if "d2" in studies else []
    reg = de_gate("dl22" in datasets, args.unseal_dl22,
                  predictions_path(args.predictions))
    data_dir = b5.resolve_data_dir(args.data_dir)
    _r, _p, rsha = b6.load_rules(data_dir, args.rules)
    shas = {"correction_rules.json": rsha}
    for ds in datasets:
        p = b6_labels_path(data_dir, ds)
        shas[p.name] = b6.sha256_file(p)
    check_shas(reg, shas)
    run_local_calib(args, reg, shas, studies, datasets)


def run_local_calib(args, reg, shas, studies, datasets) -> list[dict]:
    data_dir = b5.resolve_data_dir(args.data_dir)
    print("=" * 100)
    print(f"  DE / local-calib (OPTIONAL, Priority 5): n={args.n} calibration "
          f"pairs (whole queries), R={args.r} query-disjoint splits, seed "
          f"{args.seed}; every arm on held-out queries only")
    print("=" * 100)
    rows, out = [], {}
    if "d2" in studies:
        judges = b6._judges(args.judges)
        cells, *_x = load_cells(args, datasets, judges)
        for ds in datasets:
            for cl in cells[ds]:
                if "error" in cl:
                    continue
                res = lc_d2(cl, args.n, args.r, args.seed)
                print(f"  [d2 {ds}] {cl['model']}: {res['n_splits']} splits")
                rows += lc_rows(ds, cl["model"], res)
                out[f"d2|{ds}|{cl['model']}"] = res["n_splits"]
    if "d1-b5" in studies:
        rules_all, _p, _s = b6.load_rules(data_dir, args.rules)
        runs = b5.load_runs(data_dir / "runs" / B5_DATASET)
        jpath = Path(args.judgments) if args.judgments else \
            data_dir / "derived" / "judgments.parquet"
        df = b5.guard_dataset(b6.load_judgments(jpath))
        human = b5.human_labels(b6.load_human([B5_DATASET], args.qrels_json))
        judges = [s.strip() for s in args.b5_judges.split(",")] \
            if args.b5_judges else list(B5_JUDGES)
        for model in judges:
            d = df[df.model_id == model]
            rule = (rules_all.get("rules", {}).get(model, {}) or {}) \
                .get("msmarco")
            if d.empty or rule is None:
                continue
            st = b5_setup(model, d, rule, human, runs, None)
            res = lc_d1(st, args.n, args.r, args.seed, runs)
            print(f"  [d1-b5] {model}: {res['n_splits']} splits")
            rows += lc_rows(B5_DATASET, model, res)
            out[f"d1-b5|{model}"] = res["n_splits"]
    if not rows:
        sys.exit("ERROR: local-calib produced no rows.")
    bf = apply_registration(rows, reg)
    print_rows(rows, "local-calib: mean error over splits, [5%, 95%] split "
                     "percentiles (NOT a bootstrap)")
    print_backfill(bf, reg, {row_key(r) for r in rows},
                   lambda k: k.startswith("LC:"))
    write_json(out_dir(data_dir) / "de_local-calib.json",
               {"meta": base_meta("local-calib", reg, shas,
                                  {"n": args.n, "R": args.r,
                                   "seed": args.seed,
                                   "shift_n": "Δ̂_n = mean over calibration "
                                   "pairs of the L3 record mean - mean human "
                                   "(pair-weighted)",
                                   "metric_shift_n": "mean NDCG(RAW-HUMAN) "
                                   "over systems x calibration queries"}),
                "n_splits": out,
                "rows": rows})
    write_summary(data_dir, "local-calib", rows)
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# fit-lodo-rules
# ═══════════════════════════════════════════════════════════════════════════

def lodo_dial(delta: float, D: dict) -> int:
    """batch4.best_level restricted to LODO_DIALS (same tie-break)."""
    return min(LODO_DIALS, key=lambda l: (abs(delta + D[l]), l))


def fit_lodo(df: pd.DataFrame, qrels: dict, judges, flagships: set,
             data_dir: Path, seed: int = SEED) -> dict:
    rules, errors = {}, {}
    use = list(LODO_DATASETS)
    for model in judges:
        frozen, _paths = b4.flagship_pairs_for(data_dir, use)
        d = b4.scope_frame(df, model, use, frozen, model in flagships,
                           f"msmarco-lodo:{','.join(use)}")
        if d.empty:
            errors[model] = f"no {'/'.join(use)} rows"
            continue
        P = b4.pack(d, qrels, model, "msmarco-lodo", seed)
        allrows = np.arange(len(P.h))
        est = b4.delta_D(P, allrows, strict=False)
        if est is None:
            errors[model] = "a politeness level is missing on dl19+dl20"
            continue
        delta, D, _m, _q = est
        dial = lodo_dial(delta, D)
        smap, kfit = b4.fit_monotone_map(P, allrows, dial)
        rules[model] = {"msmarco": {
            "datasets": use, "n_pairs": int(P.n_pairs),
            "delta_hat": float(delta),
            "D_hat": {f"L{l}": float(D[l]) for l in b4.LEVELS},
            "dial_level": int(dial), "dial_candidates": list(LODO_DIALS),
            "dial_unconstrained": int(b4.best_level(delta, D)),
            "monotone_map": [int(x) for x in smap],
            "monotone_map_kappa_insample": (float(kfit) if np.isfinite(kfit)
                                            else None)}}
    return {"_framing": ("LEAVE-DL21-OUT msmarco rules for the DE study: "
                         "fitted on dl19+dl20 only, dial restricted to "
                         "{L1, L3}. Operating-point artifact removal, never "
                         "a quality claim."),
            "script": "scripts/decision_errors.py fit-lodo-rules",
            "git_hash": b4.git_hash(), "timestamp_utc": now_utc(),
            "seed": seed, "datasets_used": use,
            "dial_constraint": list(LODO_DIALS),
            "estimators": "batch4 pack / delta_D / argmin|Δ+D(l)| over "
                          "{L1,L3} / fit_monotone_map",
            "rules": rules, "not_fitted": errors}


def cmd_fit_lodo(args) -> None:
    data_dir = b5.resolve_data_dir(args.data_dir)
    out = Path(args.out) if args.out else out_dir(data_dir) / "rules_lodo.json"
    if out.exists() and not args.force:
        sys.exit(f"ERROR: {out} exists (its sha256 may already be registered)."
                 f" Pass --force to refit deliberately.")
    jpath = Path(args.judgments) if args.judgments else \
        data_dir / "derived" / "judgments.parquet"
    df = b6.load_judgments(jpath)
    df = df[df.dataset.astype(str).isin(LODO_DATASETS)]
    qrels = b6.load_human(list(LODO_DATASETS), args.qrels_json)
    flagships = (b4.flagship_model_ids(REPO / "config" / "models.yaml")
                 if args.flagships is None else
                 {s.strip() for s in args.flagships.split(",") if s.strip()})
    judges = [s.strip() for s in args.judges.split(",")] if args.judges \
        else list(B5_JUDGES)
    res = fit_lodo(df, qrels, judges, flagships, data_dir, args.seed)
    print("=" * 90)
    print("  DE fit-lodo-rules — msmarco rule on dl19+dl20 only, dial ∈ "
          f"{{L{', L'.join(map(str, LODO_DIALS))}}}")
    print("=" * 90)
    for m, fam in res["rules"].items():
        r = fam["msmarco"]
        print(f"  {m:<32} Δ̂={r['delta_hat']:+.4f}  dial=L{r['dial_level']} "
              f"(unconstrained L{r['dial_unconstrained']})  map="
              f"{r['monotone_map']}  n_pairs={r['n_pairs']}")
    for m, why in res["not_fitted"].items():
        print(f"  {m:<32} NOT FITTED: {why}")
    out.write_text(json.dumps(b6._jclean(res), indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"  wrote {out}\n  sha256[rules_lodo.json]: {b6.sha256_file(out)}")


# ═══════════════════════════════════════════════════════════════════════════
# paste — the registration block (blind: no human labels)
# ═══════════════════════════════════════════════════════════════════════════

def paste_lines(data_dir: Path, rules_all: dict, rsha: str, lodo: dict | None,
                lodo_sha: str | None, b5p: Path | None, eval_root: Path,
                judges_d2, flagships, b5_judges, evaluators) -> list[str]:
    head = b4.git_hash()
    dirty = b6.code_dirty()
    L = [f"# {DE_MARKER}（scripts/decision_errors.py）", ""]
    L.append(f"登记时间：____。分析代码 git hash：{head}"
             + ("（⚠ scripts/ 或 src/ 有未提交改动 — 先 commit 代码再登记）"
                if dirty else "") + "。")
    L.append("顺序：fit-lodo-rules → paste → 本块 commit → run（d2-pool / "
             "d1-b5 / d3-validate / d1-students [/ local-calib]）。")
    L.append("")
    L.append("## 输入 sha256（run 时逐个核对，不符或缺失即中止）")
    L.append(f"- sha256[correction_rules.json]: {rsha}")
    if lodo_sha:
        L.append(f"- sha256[rules_lodo.json]: {lodo_sha}")
    else:
        L.append("- sha256[rules_lodo.json]: ____（缺失：先跑 fit-lodo-rules）")
    for ds in D2_DATASETS:
        p = b6_labels_path(data_dir, ds)
        L.append(f"- sha256[{p.name}]: "
                 + (b6.sha256_file(p) if p.exists() else "____（文件缺失）"))
    try:
        units, excluded, usha = scan_students(eval_root, load=False)
    except SystemExit as e:
        units, excluded, usha = [], [], None
        L.append(f"- （B7 eval 扫描失败：{e}）")
    L.append(f"- sha256[b7_universe_dl22]: {usha or '____'}"
             "（B7 dl22 universe_sha256，全部 eval 文件一致）")
    if b5p is not None and b5p.exists():
        L.append(f"- sha256[b5_per_system_ndcg.csv]: {b6.sha256_file(b5p)}")
    else:
        L.append("- sha256[b5_per_system_ndcg.csv]: ____（d1-b5 完整性表缺失；"
                 "或 d1-b5 用 --no-b5-check 并在此声明）")
    L.append("")
    L.append("## LODO 规则（dl19+dl20 拟合，dial 限定 {L1, L3}：B5 的 325 对 "
             "top-up 只有 L1/L3）")
    if lodo:
        for m, fam in lodo.get("rules", {}).items():
            r = fam["msmarco"]
            L.append(f"- {m}: dial=L{r['dial_level']}（无约束 "
                     f"L{r['dial_unconstrained']}）map={r['monotone_map']} "
                     f"Δ̂={r['delta_hat']:+.4f}")
        for m, why in lodo.get("not_fitted", {}).items():
            L.append(f"- {m}: 未拟合（{why}）→ 无 LODO arm")
    else:
        L.append("- ____（未拟合）")
    L.append("")
    L.append("## 判定纪律（逐字；代码同文打印）")
    L.extend(ADJUDICATION_RULES)
    L.append("")
    L.append("## 单元与网格（计数来自文件名/列，不含任何人类标签或一致性数值）")
    L.append(f"- 相关 = grade ≥ {REL}；tie 带 {TIE}；90% 百分位 CI；seed {SEED}。")
    for ds in D2_DATASETS:
        p = b6_labels_path(data_dir, ds)
        if not p.exists():
            L.append(f"- D2 {ds}: batch6 标签文件缺失")
            continue
        q = pd.read_parquet(p, columns=["model_id", "qid"])
        cells = []
        for m in judges_d2:
            rule = b6.frozen_rule(rules_all, m, ds)
            nqm = q[q.model_id == m].qid.nunique()
            tag = ("no-op" if rule is not None and b6.is_noop_rule(rule)
                   else (f"L{rule['dial_level']}" if rule else "无规则"))
            cells.append(f"{short_judge(m)}{'(40%)' if m in flagships else ''}"
                         f"={nqm}q/{tag}")
        L.append(f"- D2 {ds}（{'held-out，主终点' if ds == 'dl22' else 'in-sample，只报告'}"
                 f"）：K ∈ {list(K_GRID[ds])}（K=1 记 D2a），D2c 容差 "
                 f"{BUDGET_TOL}；B={B_D2}；单元 " + "，".join(cells))
    try:
        nsys = len(b5.load_runs(data_dir / "runs" / B5_DATASET))
    except SystemExit:
        nsys = "____"
    L.append(f"- D1-b5 dl21：{nsys} 系统 × {len(b5_judges)} judge（"
             + "，".join(short_judge(m) for m in b5_judges)
             + f"），B5 universe（{b5.N_QUERIES_EXPECTED} judged queries 上限）；"
             f"τ 网格 {TAU_GRID[0]:.2f}–{TAU_GRID[-1]:.2f} 步长 0.01（"
             f"{len(TAU_GRID)} 点）；margin m ∈ {list(MARGINS)}；α={ALPHA}；"
             f"B={B_D1}；arm RAW/CORR/SHIFT_FROZEN/CORR_LODO/SHIFT_LODO。")
    by = defaultdict(int)
    for u in units:
        by[(u["arm"], u["slug"])] += 1
    L.append(f"- D3 / D1-dl22 学生单元：{len(units)} 个 eval 文件（"
             + "，".join(f"{a}/{s}={n}" for (a, s), n in sorted(by.items()))
             + f"）；排除 {len(excluded)} 个（'@tag' 探索性重训等）。")
    for ev in evaluators:
        n_use = sum(1 for u in units if u["slug"] != b7.judge_slug(ev))
        L.append(f"  - evaluator {ev}{'（主）' if ev == evaluators[0] else ''}："
                 f"{n_use} 学生（排除同 teacher {len(units) - n_use}）")
    L.append(f"  通过规则：|mean clip(pred) − mean 测试标签| ≤ {D3_OFF_TOL} 且 "
             f"ratio@2 ∈ [{D3_RATIO[0]}, {D3_RATIO[1]}]；B={B_D3}；seed 方差"
             "不进 CI。D1-dl22 caveat：学生 NDCG 带窄（~0.43–0.47），跨尺度误差"
             "被放大。")
    L.append(f"- local-calib（可选）：n={LC_N} 校准对（整 query），R={LC_R} 次 "
             "query-disjoint 切分；SHIFT_n、METRIC_SHIFT_n；全部 arm 只在留出 "
             "query 上重算。")
    L.append("")
    L.append("## 盲态声明")
    L.append("- D2（dl22 主终点）：____（建议：blind — per-query 决策错误从未"
             "计算；B6 聚合终点已开箱）")
    L.append("- D1-b5（dl21）：____（建议：non-blind — B5 逐系统点估计已见；"
             "bootstrap 与 LODO arm 为新）")
    L.append("- D3（dl22）：____（建议：blind — 学生在 judge 测试标签下的通过/"
             "失败从未计算）")
    L.append("- D1-dl22（学生当系统）：____")
    L.append("")
    L.append("## 预测（键 = decision|dataset|judge|arm|grid；token：降 / 升 / 平 "
             "或 ≥x / ≤x；可增删行，格式不变）")
    L.append("主终点：")
    for m in judges_d2:
        rule = b6.frozen_rule(rules_all, m, "dl22")
        if rule is None or b6.is_noop_rule(rule):
            continue
        for dec, g in (("D2a", "K=1"), ("D2b", "K=3")):
            L.append(f"- DE[{dec}|dl22|{m}|FROZEN|{g}] D2 {g} FROZEN−RAW"
                     f"｜预测：____")
    L.append(f"- DE[D1a|dl21|{b5_judges[0]}|CORR|-] D1a 排行榜放置 CORR−RAW"
             "｜预测：____")
    ev0 = evaluators[0]
    L.append(f"- DE[D3|dl22|{ev0}|FROZEN|all] D3 FROZEN−RAW｜预测：____")
    L.append(f"- DE[D3|dl22|{ev0}|RAW|all] D3 RAW 误判率水平｜预测：____")
    L.append(f"- DE[D1a|dl22|{ev0}|FROZEN|-] D1a 学生当系统 FROZEN−RAW"
             "｜预测：____")
    L.append("次要（可删）：")
    for m in b5_judges:
        rule = (rules_all.get("rules", {}).get(m, {}) or {}).get("msmarco")
        if rule is not None and b6.is_noop_rule(rule):
            continue                        # structural, listed below
        L.append(f"- DE[D1c|dl21|{m}|CORR|m=0] 同尺度 A 胜 B，CORR−RAW"
                 "｜预测：____")
    L.append("结构性（不计分，代码 hard-fail）：")
    for m in sorted(set(judges_d2) | set(b5_judges)):
        rule = (rules_all.get("rules", {}).get(m, {}) or {}).get("msmarco")
        if rule is not None and b6.is_noop_rule(rule):
            L.append(f"- {m}：identity@L3 → FROZEN/CORR ≡ RAW，全部决策 Δerr ≡ 0")
    L.append("- 开箱结果（run 后回填）：")
    return L


def cmd_paste(args) -> None:
    data_dir = b5.resolve_data_dir(args.data_dir)
    rules_all, _p, rsha = b6.load_rules(data_dir, args.rules)
    lodo_path = Path(args.lodo) if args.lodo else \
        out_dir(data_dir) / "rules_lodo.json"
    lodo, lsha = load_lodo(lodo_path)
    b5p = Path(args.b5_check) if args.b5_check else \
        out_dir(data_dir) / "b5_per_system_ndcg.csv"
    eval_root = Path(args.eval_root) if args.eval_root else \
        data_dir / "derived" / "batch7" / "eval"
    evaluators = [s.strip() for s in args.evaluators.split(",")] \
        if args.evaluators else list(EVALUATORS)
    lines = paste_lines(data_dir, rules_all, rsha, lodo, lsha, b5p, eval_root,
                        b6._judges(args.judges), b6._flagships(args.flagships),
                        list(B5_JUDGES), evaluators)
    print("PASTE INTO paper/PREDICTIONS.md  (fill every ____, commit, THEN run;"
          " nothing below was computed from human labels)")
    print("-" * 96)
    print("\n".join(lines))
    print("-" * 96)


# ═══════════════════════════════════════════════════════════════════════════
# selftest — synthetic, offline
# ═══════════════════════════════════════════════════════════════════════════

ST_JUDGES = {
    # lenient at L3 (+1), well-calibrated at L1; frozen msmarco rule L1/identity
    "stub/lenient": {1: 0, 2: 1, 3: 1, 4: 1, 5: 1},
    # calibrated everywhere; identity@L3 frozen rule -> structural no-op
    "stub/noop": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
    # dl22 evaluator: lenient like stub/lenient
    "stub/eval": {1: 0, 2: 1, 3: 1, 4: 1, 5: 1},
}
ST_SHAPE = {"dl21": (24, 14, [0.40, 0.30, 0.18, 0.12]),
            "dl22": (22, 14, [0.45, 0.30, 0.15, 0.10]),
            "antique": (16, 10, [0.15, 0.30, 0.30, 0.25]),
            "dl19": (12, 10, [0.40, 0.30, 0.18, 0.12]),
            "dl20": (12, 10, [0.40, 0.30, 0.18, 0.12])}
ST_SYSTEMS = 6


def _st_build(seed: int = 7):
    rng = np.random.default_rng(seed)
    qrels, rows = {}, []
    for ds, (nq, nd, p) in ST_SHAPE.items():
        for qi in range(nq):
            for di in range(nd):
                qrels[(ds, f"{200 + qi}", f"{ds}-d{qi}-{di}")] = \
                    int(rng.choice(4, p=p))
    for model, offs in ST_JUDGES.items():
        for (ds, q, doc), g in qrels.items():
            if model == "stub/eval" and ds not in ("dl22",):
                continue
            for l in b4.LEVELS:
                for para in "abc":
                    s = min(3, max(0, g + offs[l]))
                    u = rng.random()
                    if u < 0.08:
                        s = min(3, max(0, s + int(rng.choice([-1, 1]))))
                    fail = u > 0.997
                    rows.append({"model_id": model, "prompt_id": f"L{l}_{para}",
                                 "politeness_level": l, "dataset": ds,
                                 "qid": q, "docid": doc, "run": 1,
                                 "score": np.nan if fail else float(s),
                                 "parse_ok": not fail})
    df = pd.DataFrame(rows)
    rules = {"git_hash": "stub", "timestamp_utc": "stub", "rules": {}}
    for m in ST_JUDGES:
        dial, mp = (3, [0, 1, 2, 3]) if m == "stub/noop" else (1, [0, 1, 2, 3])
        rules["rules"].setdefault(m, {})["msmarco"] = {
            "dial_level": dial, "monotone_map": mp,
            "datasets": ["dl19", "dl20", "dl21"], "delta_hat":
            0.02 if m == "stub/noop" else 0.75}
        if m != "stub/eval":
            d = df[(df.model_id == m) & (df.dataset == "antique")]
            P = b4.pack(d, qrels, m, "antique", SEED)
            dh = float(b4.delta_D(P, np.arange(len(P.h)))[0])
            rules["rules"][m]["antique"] = {
                "dial_level": dial, "monotone_map": mp,
                "datasets": ["antique"], "delta_hat": dh}
    return df, qrels, rules


def _st_runs(qrels: dict, root: Path, seed: int = 3) -> None:
    rng = np.random.default_rng(seed)
    rd = root / "runs" / B5_DATASET
    rd.mkdir(parents=True, exist_ok=True)
    byq = defaultdict(list)
    for (ds, q, doc), g in qrels.items():
        if ds == B5_DATASET:
            byq[q].append((doc, g))
    for s in range(ST_SYSTEMS):
        noise = 0.4 + 0.5 * s                     # systems differ in quality
        with open(rd / f"sys{s}.txt", "w") as f:
            for q, docs in sorted(byq.items()):
                sc = [(g + rng.normal(0, noise), doc) for doc, g in docs]
                sc.append((rng.normal(-1, 1), f"unjudged-{q}"))
                for rank, (v, doc) in enumerate(sorted(sc, reverse=True), 1):
                    f.write(f"{q} Q0 {doc} {rank} {v:.6f} sys{s}\n")


def _st_students(root: Path, qrels: dict, seed: int = 5) -> list:
    rng = np.random.default_rng(seed)
    pairs_u = sorted((q, d, g) for (ds, q, d), g in qrels.items()
                     if ds == "dl22")
    bias = {"HUMAN": 0.0, "RAW": 0.75, "CORR_FROZEN": 0.05, "MIX_200": 0.4}
    made = []
    for arm in STUDENT_ARMS:
        for slug in ("stub__teach", "stub__eval"):
            for set_name, bb in bias.items():
                for sd in (1, 2):
                    pairs = [[q, d, float(g + bb + rng.normal(0, 0.3)), int(g)]
                             for q, d, g in pairs_u]
                    res = b7.evaluate_pairs(pairs, b=50, seed=42)
                    out = {"universe_sha256": b7.universe_sha(pairs), **res,
                           "pairs": pairs, "label_set": set_name}
                    p = root / "derived" / "batch7" / "eval" / arm / slug / \
                        f"{set_name}_s{sd}_dl22.json"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(json.dumps(out))
                    made.append(p)
    # an exploratory '@tag' retrain: must be excluded
    p = root / "derived" / "batch7" / "eval" / "minilm-raw" / "stub__teach" / \
        "RAW@hb_s1_dl22.json"
    p.write_text(made[0].read_text())
    return made


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


def _ns(**kw) -> argparse.Namespace:
    base = dict(data_dir=None, rules=None, judgments=None, qrels_json=None,
                predictions=None, judges=None, flagships="", b=200, seed=SEED,
                unseal_dl22=True, eval_root=None, evaluators=None, lodo=None,
                b5_check=None, no_b5_check=False, n=LC_N, r=LC_R,
                studies="d2,d1-b5", datasets="dl22,dl21,antique",
                b5_judges=None, force=False, out=None)
    base.update(kw)
    return argparse.Namespace(**base)


def _exits(fn) -> bool:
    try:
        fn()
    except SystemExit:
        return True
    return False


def cmd_selftest(args) -> None:
    print("=" * 96)
    print("  DE SELF-TEST (synthetic; no Drive, no network)")
    print("=" * 96)
    failures = []

    def check(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  — {detail}" if detail else ""))
        if not cond:
            failures.append(name)

    # ── 1. rounding / shift labels ──────────────────────────────────────────
    xs = np.array([0.5, 1.5, 2.5, 2.4999, -0.7, 3.9, 1.0])
    check("rhu_vec == batch5.round_half_up elementwise",
          rhu_vec(xs).tolist() == [b5.round_half_up(x) for x in xs],
          str(rhu_vec(xs).tolist()))
    sl = shift_labels({("q", "a"): 2.0, ("q", "b"): 1.25, ("q", "c"): 0.5},
                      [("q", "a"), ("q", "b"), ("q", "c")], 0.75)
    check("shift_labels: mean - Δ̂ then round-half-up",
          sl == {("q", "a"): 1, ("q", "b"): 1, ("q", "c"): 0}, str(sl))

    # ── 2. D2 indicators, hand example ──────────────────────────────────────
    qcode = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    h = np.array([2, 2, 0, 0, 1, 3, 2, 2, 0])          # nH = [2, 0, 3]
    labs = {"RAW": np.array([2, 3, 2, 2, 2, 3, 3, 3, 3]),   # nL = [3, 2, 4]
            "FROZEN": np.array([2, 2, 0, 0, 1, 3, 2, 2, 0])}
    ind = d2_indicators(qcode, 3, h, labs, (1, 3))
    check("D2a K=1 errors (RAW wrong on q1 only)",
          ind[("D2a", "K=1")]["RAW"].tolist() == [0, 1, 0]
          and ind[("D2a", "K=1")]["FROZEN"].sum() == 0)
    check("D2b K=3 errors (RAW flips q0: 3>=3 vs 2<3)",
          ind[("D2b", "K=3")]["RAW"].tolist() == [1, 0, 0])
    check("D2c budget: |nL-nH|/max(nH,1) > 0.2",
          ind[("D2c", "tol=0.2")]["RAW"].tolist() == [1, 1, 1]
          and ind[("D2c", "tol=0.2")]["FROZEN"].sum() == 0)

    # ── 3. D1 engine vs brute force ─────────────────────────────────────────
    rng = np.random.default_rng(1)
    S, N, Q = 3, 5, 9
    V = rng.random((S, N, Q))
    V[2] = V[0] + 0.05                                 # constant offset arm
    eng = D1Engine(V)
    e = eng.errors(np.ones(Q), detail=True)
    m = V.mean(axis=2)
    bf_a = []
    for a in range(S):
        wrong = sum((m[a, s] > m[0, r]) != (m[0, s] > m[0, r])
                    for s in range(N) for r in range(N) if s != r)
        bf_a.append(wrong / (N * (N - 1)))
    check("D1a == brute-force ordered-pair loop",
          np.allclose(e[("D1a", "-")], bf_a), f"{e[('D1a', '-')]} vs {bf_a}")
    check("HUMAN arm has zero error on every decision",
          all(v[0] == 0 for k, v in e.items() if not isinstance(k, str)))
    check("constant offset: within-scale D1c(m=0) error == 0 for the shifted "
          "arm", e[("D1c", "m=0")][2] == 0.0)
    rk = [1 + sum(m[0, r] > m[1, s] for r in range(N) if r != s)
          for s in range(N)]
    rh = [1 + sum(m[0, r] > m[0, s] for r in range(N) if r != s)
          for s in range(N)]
    check("D1a_place == brute force",
          np.isclose(e[("D1a_place", "-")][1],
                     np.mean(np.abs(np.array(rk) - np.array(rh)))))
    tau = TAU_GRID
    bb = np.mean([(m[1, s] >= t) != (m[0, s] >= t) for s in range(N)
                  for t in tau])
    check("D1b == brute force over (system, tau)",
          np.isclose(e[(f"D1b", f"tau=0.30..0.80")][1], bb))
    from scipy.stats import ttest_rel
    w = rng.integers(0, 3, size=Q).astype(float)
    w[0] = 2.0
    ex = np.repeat(np.arange(Q), w.astype(int))
    sv = eng.sig_verdicts(w)
    ok = True
    for pi, (i, j) in enumerate(zip(eng.I, eng.J)):
        for a in range(S):
            t, p = ttest_rel(V[a, i, ex], V[a, j, ex])
            want = 0 if not (np.isfinite(p) and p < ALPHA) else \
                (1 if t > 0 else -1)
            ok &= sv[a, pi] == want
    check("weighted paired t-test == scipy ttest_rel on the expanded "
          "resample", bool(ok))
    check("verdict3 margins: d=0.01,m=0.01 -> +1; d=0.005 -> 0; m=0 sign",
          verdict3(np.array([0.01, 0.005, -0.02, 0.0]), 0.01).tolist()
          == [1, 0, -1, 0]
          and verdict3(np.array([1e-9, -1e-9, 0.0]), 0.0).tolist()
          == [1, -1, 0])
    # bootstrap: weights == indexing; same RNG stream as batch5's loop
    idx = b6.boot_indices(Q, 5, SEED)
    r5 = np.random.default_rng(SEED)
    same = all(np.array_equal(idx[i], r5.integers(0, Q, size=Q))
               for i in range(5))
    check("boot_indices == batch5.paired_bootstrap RNG stream", same)
    wv = np.bincount(idx[0], minlength=Q).astype(float)
    check("per-query counts == nd[:, :, idx] means",
          np.allclose(eng.means(wv), V[:, :, idx[0]].mean(axis=2)))
    reps = bootstrap_decisions(V, eng.errors, 300, SEED)
    lo, hi = ci90(reps[("D1a", "-")][:, 1])
    check("bootstrap CI brackets the point estimate (D1a)",
          lo <= e[("D1a", "-")][1] <= hi, f"[{lo:.3f},{hi:.3f}]")
    reps2 = bootstrap_decisions(V, eng.errors, 300, SEED)
    check("bootstrap deterministic under seed 42",
          np.array_equal(reps[("D1a", "-")], reps2[("D1a", "-")]))
    check("truth recomputed inside replicates: HUMAN error 0 in every rep",
          all((v[:, 0] == 0).all() for v in reps.values()))
    # masked mean (batch7: IDCG=0 queries dropped per set)
    Mm = np.ones((S, Q))
    Mm[1, :3] = 0
    em = D1Engine(V, Mm).means(np.ones(Q))
    check("masked means drop invalid queries per label set",
          np.allclose(em[1], V[1, :, 3:].mean(axis=1))
          and np.allclose(em[0], V[0].mean(axis=1)))

    # ── 4. registration parsing / adjudication ──────────────────────────────
    blk = (f"# {DE_MARKER}\n- sha256[x.json]: {'a' * 64}\n"
           "- DE[D2b|dl22|j|FROZEN|K=3] foo｜预测：降\n"
           "- DE[D3|dl22|j|RAW|all] lvl｜预测：≥0.30\n"
           "- DE[D1c|dl21|j|CORR|m=0] x｜预测：平 (note)\n"
           "- DE[D1a|dl21|j|CORR|-] x｜预测：maybe\n# next\n")
    reg = parse_registration(de_block(blk))
    check("registration: sha lines, tokens, invalid token flagged",
          reg["shas"] == {"x.json": "a" * 64}
          and reg["preds"] == {"D2b|dl22|j|FROZEN|K=3": "降",
                               "D3|dl22|j|RAW|all": "≥0.30",
                               "D1c|dl21|j|CORR|m=0": "平"}
          and len(reg["invalid"]) == 1, str(reg))
    check("adjudicate: 降 / tie band / 平 / level",
          [adjudicate("降", 0.1, -0.05), adjudicate("降", 0.1, -0.02),
           adjudicate("降", 0.1, 0.03), adjudicate("平", 0.1, 0.019),
           adjudicate("升", 0.1, 0.021), adjudicate("≥0.30", 0.3, None),
           adjudicate("≤0.10", 0.2, None)]
          == ["HIT", "TIE", "MISS", "HIT", "HIT", "HIT", "MISS"])
    check("de_block stops at the next top-level heading",
          "# next" not in de_block(blk))

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(args.keep_dir) if args.keep_dir else Path(tmp)
        root.mkdir(parents=True, exist_ok=True)
        df, qrels, rules = _st_build()
        (root / "derived").mkdir(parents=True, exist_ok=True)
        df.to_pickle(root / "derived" / "judgments.stub.pkl")
        (root / "derived" / "correction_rules.json").write_text(
            json.dumps(rules, indent=2))
        (root / "qrels_stub.json").write_text(
            json.dumps([[a, b_, c, g] for (a, b_, c), g in qrels.items()]))
        judges = ["stub/lenient", "stub/noop"]
        # stored batch6 labels with batch6's OWN producer (run -> write_labels)
        res6 = b6.analyze_datasets(df, qrels, rules, ["dl21", "antique"],
                                   judges, frozenset(), root, 50, SEED,
                                   None, 10, (42, 43))
        res6dl22 = b6.analyze_datasets(df, qrels, rules, ["dl22"],
                                       judges + ["stub/eval"], frozenset(),
                                       root, 50, SEED, None, 10, (42, 43))
        res6.update(res6dl22)
        for ds in ("dl22", "dl21", "antique"):
            b6.write_labels(res6[ds], ds, root)
        _st_runs(qrels, root)
        _st_students(root, qrels)
        common = dict(data_dir=str(root),
                      judgments=str(root / "derived" / "judgments.stub.pkl"),
                      qrels_json=str(root / "qrels_stub.json"))

        # ── 5. fit-lodo-rules ──────────────────────────────────────────────
        check("lodo_dial restricted to {L1,L3}; batch4 tie-break",
              lodo_dial(0.5, {1: -0.6, 2: -0.5, 3: 0.0, 4: 0.1, 5: 0.2}) == 1
              and b4.best_level(0.5, {1: -0.6, 2: -0.5, 3: 0.0, 4: 0.1,
                                      5: 0.2}) == 2
              and lodo_dial(0.1, {1: -0.3, 2: -0.1, 3: 0.0, 4: 0, 5: 0}) == 3)
        a_l = _ns(**common, judges="stub/lenient,stub/noop,stub/eval")
        cmd_fit_lodo(a_l)
        lp = root / "derived" / "decision_errors" / "rules_lodo.json"
        lodo = json.loads(lp.read_text())
        rl = lodo["rules"]["stub/lenient"]["msmarco"]
        check("fit-lodo: lenient judge -> dial L1 (constrained), Δ̂ > 0",
              rl["dial_level"] == 1 and rl["delta_hat"] > 0.3
              and rl["datasets"] == ["dl19", "dl20"], str(rl))
        check("fit-lodo: every dial in {1,3}; judge without dl19/20 not fitted",
              all(f["msmarco"]["dial_level"] in LODO_DIALS
                  for f in lodo["rules"].values())
              and "stub/eval" in lodo["not_fitted"])
        check("fit-lodo refuses to overwrite without --force",
              _exits(lambda: cmd_fit_lodo(a_l)))

        # ── 6. paste + registration gate in a temp git repo ────────────────
        repo = root / "fake_repo"
        (repo / "paper").mkdir(parents=True, exist_ok=True)
        _git_init(repo)
        pred = repo / "paper" / "PREDICTIONS.md"
        pred.write_text("# Semi-Blind Prediction Registration\n\nnothing\n")
        _git_commit(repo, "init")
        # B5 integrity table from batch5's own evaluate path (independent)
        runs = b5.load_runs(root / "runs" / B5_DATASET)
        dfl = df[df.dataset == B5_DATASET]
        human21 = b5.human_labels({k: g for k, g in qrels.items()
                                   if k[0] == B5_DATASET})
        b5csv = root / "derived" / "decision_errors" / "b5_per_system_ndcg.csv"
        with open(b5csv, "w") as f:
            f.write("# stub B5 table\njudge,system,HUMAN,RAW,CORR\n")
            for mj in judges:
                ev = b5.evaluate_one_judge(mj, dfl[dfl.model_id == mj],
                                           rules["rules"][mj]["msmarco"],
                                           human21, runs, b=10)
                for si, s in enumerate(ev["systems"]):
                    mm = ev["means"]
                    f.write(f"{short_judge(mj)},{s},{mm[0, si]:.4f},"
                            f"{mm[1, si]:.4f},{mm[2, si]:.4f}\n")
        buf = paste_lines(root, rules, b6.sha256_file(
            root / "derived" / "correction_rules.json"), lodo,
            b6.sha256_file(lp), b5csv, root / "derived" / "batch7" / "eval",
            judges + ["stub/eval"], frozenset(), judges,
            ["stub/eval", "stub/lenient"])
        txt = "\n".join(buf)
        check("paste: heading, sha lines for every input, placeholders",
              txt.startswith(f"# {DE_MARKER}")
              and txt.count("sha256[") == 7 and "预测：____" in txt
              and "____" in txt.split("\n")[2], f"{txt.count('sha256[')} sha")
        check("paste: no-op judge gets no prediction line, listed structural",
              "DE[D2a|dl22|stub/noop" not in txt
              and "stub/noop：identity@L3" in txt)
        check("paste: excluded '@tag' file counted",
              "排除 1 个" in txt)
        a_gate = _ns(**common, predictions=str(pred), judges=",".join(judges),
                     b5_check=str(b5csv), lodo=str(lp), datasets="dl22")
        check("gate: no DE heading -> refuse", _exits(lambda: cmd_d2(a_gate)))
        filled = []
        for ln in buf:
            if "预测：____" in ln:
                tok = ("≥0.30" if "|RAW|" in ln else
                       "平" if "D1c" in ln else "降")
                ln = ln.replace("预测：____", f"预测：{tok}")
            filled.append(ln.replace("____", "2026-10-20"))
        pred.write_text(pred.read_text() + "\n" + "\n".join(buf) + "\n")
        _git_commit(repo, "unfilled")
        check("gate: committed block with '____' -> refuse",
              _exits(lambda: de_gate(False, False, pred)))
        pred.write_text("# Semi-Blind Prediction Registration\n\nnothing\n\n"
                        + "\n".join(filled) + "\n")
        check("gate: filled but uncommitted -> refuse",
              _exits(lambda: de_gate(False, False, pred)))
        _git_commit(repo, "DE registration")
        check("gate: dl22 without --unseal-dl22 -> refuse",
              _exits(lambda: de_gate(True, False, pred)))
        regp = de_gate(True, True, pred)
        check("gate: committed + filled + unseal -> passes, commit recorded",
              regp["commit"] is not None and len(regp["preds"]) >= 5,
              f"{len(regp['preds'])} predictions")
        bad_reg = {"shas": dict(regp["shas"]), "preds": {}}
        bad_reg["shas"]["correction_rules.json"] = "0" * 64
        check("sha mismatch -> refuse",
              _exits(lambda: check_shas(bad_reg, {
                  "correction_rules.json":
                      regp["shas"]["correction_rules.json"]})))
        check("unregistered input -> refuse",
              _exits(lambda: check_shas({"shas": {}}, {"x": "a" * 64})))

        # ── 7. D2 end-to-end through the CLI path ───────────────────────────
        a_d2 = _ns(**common, predictions=str(pred), judges=",".join(judges
                   + ["stub/eval"]), datasets="dl22,dl21,antique", b=300)
        cmd_d2(a_d2)
        od = root / "derived" / "decision_errors"
        sm = pd.read_csv(od / "de_summary.csv", keep_default_na=False)
        check("de_summary.csv columns exactly as registered",
              tuple(sm.columns) == CSV_COLS, str(tuple(sm.columns)))

        def g(dec, ds, j, arm, grid, col="err"):
            r = sm[(sm.decision == dec) & (sm.dataset == ds) & (sm.judge == j)
                   & (sm.arm == arm) & (sm.grid == grid)]
            return r.iloc[0][col] if len(r) else None
        e_raw = float(g("D2b", "dl22", "stub/lenient", "RAW", "K=3"))
        e_fz = float(g("D2b", "dl22", "stub/lenient", "FROZEN", "K=3"))
        check("D2 lenient dl22: FROZEN lowers the K=3 error",
              e_fz < e_raw, f"RAW {e_raw:.3f} FROZEN {e_fz:.3f}")
        check("D2 no-op cell: FROZEN Δerr ≡ 0 with CI [0,0]",
              float(g("D2b", "dl22", "stub/noop", "FROZEN", "K=3",
                      "d_vs_RAW")) == 0.0
              and g("D2b", "dl22", "stub/noop", "FROZEN", "K=3", "d_ci")
              == "+0.000000,+0.000000")
        v = g("D2b", "dl22", "stub/lenient", "FROZEN", "K=3", "verdict")
        check("registered prediction adjudicated in the verdict column",
              "|" in str(v) and str(v).split("|")[1] in ("HIT", "MISS", "TIE"),
              str(v))
        v21 = sm[(sm.dataset == "dl21") & (sm.arm == "FROZEN")]
        check("dl21/antique rows present (reported)", len(v21) > 0)
        js = json.loads((od / "de_d2-pool.json").read_text())
        check("de_d2-pool.json: stored check identical for every cell",
              all(c.get("check", "").startswith("identical")
                  for ds in js["datasets"].values() for c in ds
                  if "error" not in c))
        # integrity: a tampered stored label must hard-fail
        p22 = root / "derived" / "batch6_labels_dl22.parquet"
        orig = pd.read_parquet(p22)
        tam = orig.copy()
        tam.loc[tam.index[0], "RAW"] = (int(tam.loc[tam.index[0], "RAW"])
                                        + 1) % 4
        tam.to_parquet(p22, index=False)
        check("tampered batch6_labels -> sha mismatch refuses",
              _exits(lambda: cmd_d2(a_d2)))
        check("tampered batch6_labels -> check_stored hard-fails (sha "
              "bypassed)",
              _exits(lambda: run_d2(_ns(**common, judges="stub/lenient",
                                        datasets="dl22"), ["dl22"], None,
                                    {})))
        orig.to_parquet(p22, index=False)
        # no-op violation must hard-fail
        fake = {"noop": True, "labels_identical_FROZEN_RAW": True,
                "pts": {("D2a", "K=1"): {"RAW": 0.1, "FROZEN": 0.1}},
                "reps": {("D2a", "K=1"): {"RAW": np.array([0.1, 0.2]),
                                          "FROZEN": np.array([0.1, 0.25])}}}
        check("no-op violation (one replicate differs) -> hard-fail",
              _exits(lambda: enforce_noop([("fake", noop_violations_d2(
                  fake))], "D2")))

        # ── 8. D1-b5 end-to-end (b5 check from batch5's own path) ───────────
        a_b5 = _ns(**common, predictions=str(pred), b5_check=str(b5csv),
                   lodo=str(lp), judges=",".join(judges), b=300)
        cmd_d1_b5(a_b5)
        sm = pd.read_csv(od / "de_summary.csv", keep_default_na=False)
        e_raw = float(g("D1a", "dl21", "stub/lenient", "RAW", "-"))
        e_c = float(g("D1a", "dl21", "stub/lenient", "CORR", "-"))
        check("D1-b5 lenient: CORR lowers the cross-scale D1a error",
              e_c < e_raw, f"RAW {e_raw:.3f} CORR {e_c:.3f}")
        check("D1-b5 no-op judge: CORR Δerr ≡ 0 on every decision",
              all(float(x) == 0.0 for x in sm[(sm.judge == "stub/noop") & (
                  sm.dataset == "dl21") & (sm.arm == "CORR")].d_vs_RAW))
        check("D1-b5: LODO arms present", len(sm[(sm.dataset == "dl21") & (
            sm.arm == "CORR_LODO")]) > 0)
        rows_b5 = read_b5_csv(b5csv)
        rows_b5[0]["RAW"] = f"{float(rows_b5[0]['RAW']) + 0.001:.4f}"
        with open(b5csv, "w") as f:
            f.write("judge,system,HUMAN,RAW,CORR\n")
            for r in rows_b5:
                f.write(f"{r['judge']},{r['system']},{r['HUMAN']},{r['RAW']},"
                        f"{r['CORR']}\n")
        check("b5 check: a 1e-3 perturbation of the B5 table hard-fails",
              _exits(lambda: run_d1_b5(a_b5, None, {}, lp, b5csv)))

        # ── 9. D3 + D1-students ─────────────────────────────────────────────
        a_st = _ns(**common, predictions=str(pred), evaluators="stub/eval",
                   b=200)
        cmd_d3(a_st)
        sm = pd.read_csv(od / "de_summary.csv", keep_default_na=False)
        e_raw = float(g("D3", "dl22", "stub/eval", "RAW", "all"))
        e_fz = float(g("D3", "dl22", "stub/eval", "FROZEN", "all"))
        check("D3: lenient evaluator's RAW test labels misclassify students; "
              "FROZEN fewer", e_raw >= 0.3 and e_fz < e_raw,
              f"RAW {e_raw:.3f} FROZEN {e_fz:.3f}")
        jd3 = json.loads((od / "de_d3-validate.json").read_text())
        ev = jd3["evaluators"][0]
        check("D3: students taught by the evaluator excluded; '@tag' excluded",
              ev["n_dropped_teacher"] == 16 and ev["n_students"] == 16
              and len(jd3["meta"]["excluded_files"]) == 1,
              f"{ev['n_students']} used, {ev['n_dropped_teacher']} dropped")
        pr = ev["pass_rate_by_arm_set"]["minilm-raw|HUMAN"]["HUMAN"]
        check("D3: HUMAN-trained students pass under HUMAN test labels",
              pr == 1.0, f"{pr}")
        cmd_d1_students(a_st)
        sm = pd.read_csv(od / "de_summary.csv", keep_default_na=False)
        check("D1-students rows written",
              len(sm[(sm.dataset == "dl22") & (sm.decision == "D1a")]) == 3)
        a_noseal = _ns(**common, predictions=str(pred), evaluators="stub/eval",
                       unseal_dl22=False)
        check("D3 without --unseal-dl22 -> refuse",
              _exits(lambda: cmd_d3(a_noseal)))
        evp = next((root / "derived" / "batch7" / "eval").glob(
            "msmarco-ce/stub__teach/HUMAN_s1_dl22.json"))
        jj = json.loads(evp.read_text())
        jj["pairs"] = jj["pairs"][:-1]
        evp.write_text(json.dumps(jj))
        check("B7 universe drift (sha mismatch in one file) -> refuse",
              _exits(lambda: scan_students(root / "derived" / "batch7" /
                                           "eval")))

        # ── 10. local-calib ────────────────────────────────────────────────
        a_lc = _ns(**common, predictions=str(pred), judges="stub/lenient",
                   b5_judges="stub/lenient", datasets="dl22", n=60, r=25)
        cmd_local_calib(a_lc)
        sm = pd.read_csv(od / "de_summary.csv", keep_default_na=False)
        lc = sm[sm.decision.str.startswith("LC:")]
        check("local-calib: SHIFT_n and METRIC_SHIFT_n rows, both studies",
              {"SHIFT_n60", "METRIC_SHIFT_n60"} <= set(lc.arm)
              and {"dl22", "dl21"} <= set(lc.dataset))
        rngc = np.random.default_rng(0)
        cm = calib_split(rngc, np.array([10, 10, 10, 10, 10]), 25)
        check("calib split: whole queries until >= n pairs, held-out "
              "disjoint", cm.sum() == 3)
        check("de_summary.csv concatenates every subcommand",
              {"D2b", "D1a", "D3", "LC:D2b"} <= set(sm.decision))

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

    def common(p, b=None):
        p.add_argument("--data-dir", default=None,
                       help="DATA_DIR (default: $DATA_DIR)")
        p.add_argument("--rules", default=None,
                       help="correction_rules.json (default: $DATA_DIR/"
                            "derived/correction_rules.json)")
        p.add_argument("--judgments", default=None,
                       help="judgments parquet (.pkl = TEST stub)")
        p.add_argument("--qrels-json", default=None,
                       help="TEST hook: stub qrels instead of ir_datasets")
        p.add_argument("--predictions", default=None,
                       help="PREDICTIONS.md (default paper/PREDICTIONS.md)")
        p.add_argument("--seed", type=int, default=SEED)
        if b is not None:
            p.add_argument("--b", type=int, default=b,
                           help=f"bootstrap replicates (default {b})")

    p = sub.add_parser("fit-lodo-rules", help="msmarco rule on dl19+dl20, "
                       "dial in {L1,L3}")
    common(p)
    p.add_argument("--judges", default=None, help="default: the 4 B5 judges")
    p.add_argument("--flagships", default=None,
                   help="override flagship ids (default config/models.yaml)")
    p.add_argument("--out", default=None)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_fit_lodo)

    p = sub.add_parser("paste", help="print the DE registration block (blind)")
    common(p)
    p.add_argument("--judges", default=None, help="D2 judges (default: B6's)")
    p.add_argument("--flagships", default=None)
    p.add_argument("--lodo", default=None)
    p.add_argument("--b5-check", default=None)
    p.add_argument("--eval-root", default=None)
    p.add_argument("--evaluators", default=None)
    p.set_defaults(func=cmd_paste)

    p = sub.add_parser("d2-pool", help="D2 per-query pool decisions")
    common(p, B_D2)
    p.add_argument("--datasets", default="dl22,dl21,antique")
    p.add_argument("--unseal-dl22", action="store_true")
    p.add_argument("--judges", default=None)
    p.add_argument("--flagships", default=None)
    p.set_defaults(func=cmd_d2)

    p = sub.add_parser("d1-b5", help="D1 system decisions on dl21 (B5 pool)")
    common(p, B_D1)
    p.add_argument("--judges", default=None)
    p.add_argument("--lodo", default=None,
                   help="rules_lodo.json (default derived/decision_errors/)")
    p.add_argument("--b5-check", default=None,
                   help="b5_per_system_ndcg.csv (default derived/"
                        "decision_errors/b5_per_system_ndcg.csv)")
    p.add_argument("--no-b5-check", action="store_true",
                   help="skip the 1e-4 integrity check (stamped UNVERIFIED)")
    p.set_defaults(func=cmd_d1_b5)

    for name, fn, hp in (("d3-validate", cmd_d3, "D3 student validation"),
                         ("d1-students", cmd_d1_students,
                          "D1 on dl22, students as systems")):
        p = sub.add_parser(name, help=hp)
        common(p, B_D3)
        p.add_argument("--unseal-dl22", action="store_true")
        p.add_argument("--eval-root", default=None)
        p.add_argument("--evaluators", default=None,
                       help=f"default {','.join(EVALUATORS)} (first = primary)")
        p.add_argument("--flagships", default=None)
        p.set_defaults(func=fn)

    p = sub.add_parser("local-calib", help="OPTIONAL Priority-5 arms")
    common(p)
    p.add_argument("--studies", default="d2,d1-b5")
    p.add_argument("--datasets", default="dl22,dl21")
    p.add_argument("--unseal-dl22", action="store_true")
    p.add_argument("--judges", default=None, help="D2 judges")
    p.add_argument("--b5-judges", default=None)
    p.add_argument("--flagships", default=None)
    p.add_argument("--n", type=int, default=LC_N)
    p.add_argument("--r", type=int, default=LC_R)
    p.set_defaults(func=cmd_local_calib)

    p = sub.add_parser("selftest", help="synthetic offline self-test")
    p.add_argument("--keep-dir", default=None)
    p.set_defaults(func=cmd_selftest)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
