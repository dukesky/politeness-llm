"""Batch 1R statistical hardening — ROUND-1 (dl19 + dl20) DATA ONLY.

Runs the four P0/P1 analyses on the frozen round-1 batch:

  0. AUTHORITY ASSERTION  — recompute per-(model, level) kappa from the
     parquet on the *authoritative* path (per-paraphrase linear-weighted
     Cohen's kappa, then mean across the 3 paraphrases of a level) and assert
     it matches the unboxed values recorded in paper/PREDICTIONS.md to
     within 5e-4.  Any mismatch prints a diff table and exits 1.
  1. mixed  — MixedLM: dkappa ~ A(l) with a random intercept per model, on
     the 28 (model, level) cells of the 7 non-reasoning models (P0.1).
  2. boot   — query-clustered bootstrap 95% CIs for every
     (non-reasoning model, level in {1,2,4,5}) dkappa vs L3 (P1.3).
  3. tost   — smallest TOST equivalence margin m (0.001 grid) such that every
     level's 90% bootstrap CI lies inside (-m, +m), per non-DeepSeek judge,
     plus the max across the 6 judges (P1.3).
  4. ushape — permutation test for U-shape: statistic = between-level
     variance of the 15 per-variant kappas, level labels exchangeable,
     10,000 permutations (P1.4).

HARD SCOPE: every read of the parquet is filtered to dataset in
("dl19", "dl20").  Round-2 rows (dl21 / dl22 / antique) are never touched, so
this script cannot leak information about the sealed dl22 holdout.

Usage (Colab, repo root):
    python scripts/batch1r_stats.py --data-dir $DATA_DIR
    python scripts/batch1r_stats.py --data-dir $DATA_DIR --skip boot --skip tost
    python scripts/batch1r_stats.py --data-dir $DATA_DIR --b 200      # smoke test

LaTeX fragments (tabular body rows only, no \\begin{tabular}) are written to
    $DATA_DIR/derived/tex/*.tex
They live under DATA_DIR, not the repo: they are derived from data, and this
repo intentionally carries no data-derived artefacts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from src.qrels import load_qrels  # noqa: E402
from validate_model import level_summary_from_table, per_variant_kappa  # noqa: E402

ROUND1_DATASETS = ("dl19", "dl20")

# Cohort definition — mirrors the 口径声明 in paper/PREDICTIONS.md.
NON_REASONING = [
    "deepseek/deepseek-v4-flash",
    "openai/gpt-5.4-mini",
    "anthropic/claude-haiku-4.5",
    "qwen/qwen3.7-plus",
    "google/gemini-3.5-flash",
    "openai/gpt-5.5",
    "anthropic/claude-opus-4.8",
]
REASONING = ["google/gemini-3.1-pro-preview"]
NON_BLIND_SOURCE = "deepseek/deepseek-v4-flash"   # excluded from the TOST claim

LEVELS = [1, 2, 3, 4, 5]
OFF_L3 = [1, 2, 4, 5]
KAPPA_TOL = 5e-4
SEED = 42


# ═══════════════════════════════════════════════════════════════════════════
# 0. Authoritative values from paper/PREDICTIONS.md
# ═══════════════════════════════════════════════════════════════════════════

_MODEL_HEAD = re.compile(r"^###\s+(\S+)\s+—")
_KAPPA_KV = re.compile(r"κ\(L([1-5])\)\s*=\s*(-?\d+\.\d+)")
_KAPPA_RUN = re.compile(r"κ（L1→L5）[：:]\s*(.+)$")
_FLOAT = re.compile(r"-?\d+\.\d+")


def parse_predictions(path: Path) -> dict:
    """Extract the unboxed (开箱结果) per-level kappa per model.

    Handles both recorded shapes:
      * ``κ(L1)=0.3335, κ(L3)=0.3270, ...``            (standard Step C block)
      * ``Step C 验证 κ（L1→L5）：0.4836 / 0.4573 / ...`` (DeepSeek, pre-protocol)
    Lines inside the ``<!-- 登记格式模板 -->`` comment carry ``{}`` placeholders
    and match no float, so they are skipped naturally.
    """
    if not path.exists():
        sys.exit(f"ERROR: {path} not found — cannot run the authority assertion.")

    out: dict[str, dict[int, float]] = {}
    model = None
    for line in path.read_text(encoding="utf-8").splitlines():
        head = _MODEL_HEAD.match(line.strip())
        if head:
            model = head.group(1)
            continue
        if model is None:
            continue
        run = _KAPPA_RUN.search(line)
        if run:
            vals = [float(v) for v in _FLOAT.findall(run.group(1))]
            if len(vals) == 5:
                out.setdefault(model, {}).update(dict(zip(LEVELS, vals)))
            continue
        kvs = _KAPPA_KV.findall(line)
        if kvs:
            got = {int(l): float(v) for l, v in kvs}
            # only the 开箱结果 line carries the full 5-level set
            if len(got) >= 3:
                out.setdefault(model, {}).update(got)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Fast linear-weighted kappa (bootstrap inner loop)
# ═══════════════════════════════════════════════════════════════════════════

_W4 = np.abs(np.subtract.outer(np.arange(4), np.arange(4))).astype(float)


def kappa_from_cm(cm: np.ndarray) -> float:
    """Linear-weighted Cohen's kappa from a 4x4 confusion matrix.

    Identical to ``sklearn.metrics.cohen_kappa_score(..., weights='linear')``
    whenever both label vectors live in {0,1,2,3}: sklearn builds the matrix
    over the sorted union of observed labels, and padding that matrix with
    all-zero rows/columns changes neither the weighted observed sum nor the
    weighted expected sum (the absent label has zero mass in both marginals).
    """
    total = cm.sum()
    if total <= 0:
        return np.nan
    o = cm / total
    e = np.outer(o.sum(axis=1), o.sum(axis=0))
    po_w = float((_W4 * o).sum())
    pe_w = float((_W4 * e).sum())
    if pe_w == 0:
        return np.nan
    return 1.0 - po_w / pe_w


def cms_from_keys(keys: np.ndarray, n_variants: int) -> np.ndarray:
    """Confusion matrices for all variants at once.

    ``keys = variant * 16 + human * 4 + score``.
    """
    counts = np.bincount(keys, minlength=n_variants * 16)
    return counts[: n_variants * 16].reshape(n_variants, 4, 4).astype(float)


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_round1(data_dir: Path) -> pd.DataFrame:
    parquet = data_dir / "derived" / "judgments.parquet"
    if not parquet.exists():
        sys.exit(f"ERROR: {parquet} not found. Run src/parse.py first.")
    df = pd.read_parquet(parquet)
    before = len(df)
    df = df[df.dataset.isin(ROUND1_DATASETS)].copy()
    print(f"[scope] round-1 hard filter dataset in {ROUND1_DATASETS}: "
          f"{before} → {len(df)} rows")
    if df.empty:
        sys.exit("ERROR: no round-1 (dl19/dl20) rows in the parquet.")
    leaked = sorted(set(df.dataset.unique()) - set(ROUND1_DATASETS))
    if leaked:
        sys.exit(f"ERROR: round-1 filter failed, saw {leaked}")
    return df


def flagship_models(models_yaml: Path) -> set[str]:
    """Round-1 flagship model ids (tier: flagship and no `round:` key)."""
    import yaml
    spec = yaml.safe_load(models_yaml.read_text(encoding="utf-8"))
    return {
        m["model_id"] for m in spec["models"]
        if m.get("tier") == "flagship" and int(m.get("round", 1)) == 1
    }


def load_frozen_pairs(paths: list[Path]) -> set[tuple[str, str]]:
    frozen = set()
    for fp in paths:
        with open(fp) as f:
            for line in f:
                p = json.loads(line)
                frozen.add((str(p["qid"]), str(p["docid"])))
    return frozen


def apply_flagship_filter(df: pd.DataFrame, frozen: set, model: str,
                          paths: list) -> pd.DataFrame:
    keys = list(zip(df.qid.astype(str), df.docid.astype(str)))
    mask = np.fromiter((k in frozen for k in keys), dtype=bool, count=len(keys))
    out = df[mask]
    if out.empty:
        sys.exit(
            f"ERROR: flagship pairs filter left ZERO rows for '{model}'.\n"
            f"  pairs files : {[str(p) for p in paths]} "
            f"({len(frozen)} frozen pairs)\n"
            f"  model rows before filter : {len(df)}\n"
            f"  The frozen pairs and this model's collected pairs do not "
            f"intersect — wrong pairs\n"
            f"  files for this round, or the model was collected on a "
            f"different sample. Pass the\n"
            f"  correct --flagship-pairs, or rerun without them if this model "
            f"is not flagship."
        )
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Per-model core quantities
# ═══════════════════════════════════════════════════════════════════════════

def model_core(df_m: pd.DataFrame, qrels: dict) -> dict:
    """Authoritative kappas + Delta/D/A for one model (already scoped)."""
    name = str(df_m.model_id.iloc[0]) if len(df_m) else "(empty)"
    if df_m.empty:
        sys.exit("ERROR: model_core got zero rows — check the round-1 dataset "
                 "filter and the flagship pairs intersection.")
    table = per_variant_kappa(df_m, qrels)
    if table.empty:
        sys.exit(f"ERROR: '{name}' has no scorable (parse_ok, qrels-matched) "
                 f"rows in dl19/dl20; cannot compute kappa.")
    summary = level_summary_from_table(table)
    kappa_level = {int(r.politeness_level): float(r.kappa_mean)
                   for _, r in summary.iterrows()}
    variant_kappa = {
        (str(r.prompt_id), int(r.politeness_level)): float(r.kappa)
        for _, r in table.iterrows()
    }

    # ── Delta / D(l) — reproduced from the Step A (preflight) path ─────────
    human = np.array([qrels.get((str(ds), str(q), str(d)))
                      for ds, q, d in zip(df_m.dataset, df_m.qid, df_m.docid)],
                     dtype=object)
    matched = np.array([h is not None for h in human])
    qrels_mean = float(np.mean([h for h in human[matched]])) if matched.any() else np.nan

    ok = df_m[df_m.parse_ok & df_m.score.notna()]
    empty_levels = [lvl for lvl in LEVELS
                    if not (ok.politeness_level == lvl).any()]
    if empty_levels:
        sys.exit(f"ERROR: '{name}' has no scorable rows at politeness "
                 f"level(s) {empty_levels}; Delta/D(l) and the L3 baseline "
                 f"are undefined.")
    mean_score = {lvl: float(ok[ok.politeness_level == lvl].score.mean())
                  for lvl in LEVELS}
    l3 = mean_score[3]
    delta = l3 - qrels_mean
    D = {lvl: mean_score[lvl] - l3 for lvl in OFF_L3}
    A = {lvl: abs(delta + D[lvl]) - abs(delta) for lvl in OFF_L3}

    return {
        "table": table,
        "kappa_level": kappa_level,
        "variant_kappa": variant_kappa,
        "delta": delta,
        "D": D,
        "A": A,
        "qrels_mean": qrels_mean,
        "l3_mean": l3,
    }


def bootstrap_arrays(df_m: pd.DataFrame, qrels: dict):
    """Pack one model's scorable rows into numpy arrays for the bootstrap."""
    ok = df_m[df_m.parse_ok & df_m.score.notna()].copy()
    ok["human"] = [qrels.get((str(ds), str(q), str(d)))
                   for ds, q, d in zip(ok.dataset, ok.qid, ok.docid)]
    ok = ok.dropna(subset=["human"])

    variants = sorted(ok.prompt_id.unique())
    vidx = {p: i for i, p in enumerate(variants)}
    vlevel = (ok.drop_duplicates("prompt_id")
                .set_index("prompt_id").politeness_level.astype(int).to_dict())

    v = ok.prompt_id.map(vidx).to_numpy(dtype=np.int64)
    h = ok.human.astype(int).to_numpy()
    s = ok.score.astype(int).to_numpy()
    if h.min() < 0 or h.max() > 3 or s.min() < 0 or s.max() > 3:
        sys.exit("ERROR: human/score outside 0-3 — check src/qrels.py mapping.")
    keys = v * 16 + h * 4 + s

    # Cluster row positions by query, kept inside their dataset stratum.
    strata: dict[str, list[np.ndarray]] = {}
    pos = pd.DataFrame({
        "dataset": ok.dataset.astype(str).to_numpy(),
        "qid": ok.qid.astype(str).to_numpy(),
        "pos": np.arange(len(ok)),
    })
    for (ds, _qid), g in pos.groupby(["dataset", "qid"], sort=True):
        strata.setdefault(ds, []).append(g["pos"].to_numpy())

    return {
        "keys": keys,
        "n_variants": len(variants),
        "variants": variants,
        "vlevel": {vidx[p]: int(vlevel[p]) for p in variants},
        "strata": strata,
        "n_rows": len(ok),
    }


def level_kappas_from_keys(keys: np.ndarray, packed: dict) -> dict:
    cms = cms_from_keys(keys, packed["n_variants"])
    per_level: dict[int, list[float]] = {lvl: [] for lvl in LEVELS}
    for vi in range(packed["n_variants"]):
        per_level[packed["vlevel"][vi]].append(kappa_from_cm(cms[vi]))
    return {lvl: float(np.mean(vals)) if vals else np.nan
            for lvl, vals in per_level.items()}


# ═══════════════════════════════════════════════════════════════════════════
# LaTeX output
# ═══════════════════════════════════════════════════════════════════════════

def write_tex(tex_dir: Path, name: str, rows: list[str], caption: str) -> None:
    tex_dir.mkdir(parents=True, exist_ok=True)
    out = tex_dir / name
    body = "% " + caption + "\n" + "\n".join(rows) + "\n"
    out.write_text(body, encoding="utf-8")
    print(f"[tex] wrote {out}")


def esc(s: str) -> str:
    return s.replace("_", r"\_")


def tex_p(p: float) -> str:
    """LaTeX-math p-value: plain decimal, or a x 10^b for tiny values."""
    if not np.isfinite(p):
        return "n/a"
    if p >= 1e-4:
        return f"{p:.4f}"
    mant, exp = f"{p:.2e}".split("e")
    return f"{mant}\\times 10^{{{int(exp)}}}"


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None,
                    help="DATA_DIR (default: $DATA_DIR)")
    ap.add_argument("--skip", action="append", default=[],
                    choices=["mixed", "boot", "tost", "ushape"],
                    help="skip a stage (repeatable); the authority assertion "
                         "always runs")
    ap.add_argument("--b", type=int, default=2000,
                    help="bootstrap replicates (default 2000)")
    ap.add_argument("--n-perm", type=int, default=10000,
                    help="permutations for the U-shape test (default 10000)")
    ap.add_argument("--flagship-pairs", nargs="+", default=None, metavar="JSONL",
                    help="frozen flagship pairs files (default: auto-detect "
                         "$DATA_DIR/inputs/pairs_{dl19,dl20}_flagship40.jsonl)")
    ap.add_argument("--tol", type=float, default=KAPPA_TOL,
                    help=f"authority assertion tolerance (default {KAPPA_TOL})")
    args = ap.parse_args()

    data_dir = Path(args.data_dir or os.environ.get("DATA_DIR", "./data"))
    tex_dir = data_dir / "derived" / "tex"
    skip = set(args.skip)

    df = load_round1(data_dir)

    # qrels: round 1 only — dl22 is never even loaded here.
    qrels = load_qrels(list(ROUND1_DATASETS))

    # flagship pair restriction (matches how Step C reported flagship kappas)
    flag_models = flagship_models(REPO_ROOT / "config" / "models.yaml")
    if args.flagship_pairs:
        paths = [Path(p) for p in args.flagship_pairs]
    else:
        paths = [data_dir / "inputs" / f"pairs_{ds}_flagship40.jsonl"
                 for ds in ROUND1_DATASETS]
    paths = [p for p in paths if p.exists()]
    frozen = load_frozen_pairs(paths) if paths else set()
    if frozen:
        print(f"[flagship] {len(frozen)} frozen pairs from "
              f"{[p.name for p in paths]} → applied to {sorted(flag_models)}")
    else:
        print("[flagship] WARNING: no flagship pairs file found; flagship "
              "models will be recomputed on ALL their parquet rows, which may "
              "break the authority assertion.")

    models = [m for m in NON_REASONING + REASONING
              if m in set(df.model_id.unique())]
    missing = [m for m in NON_REASONING + REASONING if m not in models]
    if missing:
        print(f"[scope] not in parquet, skipped: {missing}")

    scoped: dict[str, pd.DataFrame] = {}
    for m in models:
        d = df[df.model_id == m]
        if m in flag_models and frozen:
            n0 = len(d)
            d = apply_flagship_filter(d, frozen, m, paths)
            print(f"[flagship filter] {m}: {n0} → {len(d)} rows")
        if d.empty:
            sys.exit(f"ERROR: no round-1 rows left for '{m}' after scoping.")
        scoped[m] = d

    core = {m: model_core(scoped[m], qrels) for m in models}

    # ── 0. AUTHORITY ASSERTION ────────────────────────────────────────────
    print()
    print("=" * 78)
    print("0. AUTHORITY ASSERTION vs paper/PREDICTIONS.md")
    print("=" * 78)
    authoritative = parse_predictions(REPO_ROOT / "paper" / "PREDICTIONS.md")
    diffs, checked = [], 0
    for m in models:
        auth = authoritative.get(m)
        if not auth:
            print(f"  [warn] no unboxed kappa recorded for {m} — skipped")
            continue
        for lvl, kauth in sorted(auth.items()):
            kcalc = core[m]["kappa_level"].get(lvl, np.nan)
            checked += 1
            if not np.isfinite(kcalc) or abs(kcalc - kauth) > args.tol:
                diffs.append((m, lvl, kauth, kcalc, kcalc - kauth))
    if diffs:
        print()
        print(f"  !! {len(diffs)} MISMATCH(ES) — recompute does not reproduce "
              f"the authoritative values (tol={args.tol}):")
        print(f"  {'model':<34} {'lvl':>3} {'PREDICTIONS':>12} "
              f"{'recomputed':>12} {'diff':>10}")
        for m, lvl, ka, kc, d in diffs:
            print(f"  {m:<34} {lvl:>3} {ka:>12.4f} {kc:>12.4f} {d:>+10.4f}")
        print()
        print("  Nothing downstream is trustworthy until this is resolved.")
        print("  Check: round-1 dataset filter, flagship pairs restriction, "
              "and that the")
        print("  paraphrase-MEAN path is used (a pooled recompute is NOT "
              "authoritative).")
        sys.exit(1)
    print(f"  OK — {checked} (model, level) kappas reproduce within "
          f"{args.tol} across {len(authoritative)} recorded models.")

    # cross-check the fast bootstrap kappa against the authoritative path.
    # NaN-aware: max(x, nan) silently returns x, so a NaN on exactly one side
    # would otherwise pass unnoticed. One-sided NaN = divergence (fail);
    # two-sided NaN = an undefined cell, reported and skipped.
    packed = {m: bootstrap_arrays(scoped[m], qrels) for m in models}
    worst = 0.0
    nan_both, nan_one = [], []
    for m in models:
        fast = level_kappas_from_keys(packed[m]["keys"], packed[m])
        for lvl in LEVELS:
            a, b = fast[lvl], core[m]["kappa_level"][lvl]
            fa, fb = np.isfinite(a), np.isfinite(b)
            if not fa and not fb:
                nan_both.append((m, lvl))
                continue
            if fa != fb:
                nan_one.append((m, lvl, a, b))
                continue
            worst = max(worst, abs(a - b))
    for m, lvl in nan_both:
        print(f"  [note] {m} L{lvl}: kappa undefined on BOTH paths — cell "
              f"skipped in the cross-check")
    if nan_one:
        for m, lvl, a, b in nan_one:
            print(f"  !! {m} L{lvl}: fast={a}  authoritative={b} "
                  f"(NaN on exactly one path)")
        sys.exit("ERROR: fast kappa and the sklearn path disagree on whether "
                 "a cell is defined; bootstrap CIs would not be comparable to "
                 "the point estimates.")
    print(f"  fast (bincount) kappa vs sklearn path: max |diff| = {worst:.2e}")
    if worst > 1e-9:
        sys.exit("ERROR: fast kappa disagrees with the sklearn path; "
                 "bootstrap CIs would not be comparable to the point estimates.")

    # point-estimate dkappa table
    print()
    print(f"  {'model':<34} " + " ".join(f"{'L'+str(l):>9}" for l in LEVELS))
    for m in models:
        k = core[m]["kappa_level"]
        print(f"  {m:<34} " + " ".join(f"{k[l]:>9.4f}" for l in LEVELS))

    nonreason = [m for m in models if m in NON_REASONING]

    # ── 1. MIXED-EFFECTS (P0.1) ───────────────────────────────────────────
    if "mixed" not in skip:
        run_mixed(nonreason, core, tex_dir)

    # ── 2/3. BOOTSTRAP + TOST (P1.3) ──────────────────────────────────────
    boot = None
    if "boot" not in skip:
        boot = run_bootstrap(nonreason, packed, core, args.b, tex_dir)
    if "tost" not in skip:
        if boot is None:
            print("\n[tost] skipped: needs the bootstrap draws "
                  "(do not pass --skip boot together with tost).")
        else:
            run_tost(nonreason, boot, tex_dir)

    # ── 4. U-SHAPE PERMUTATION (P1.4) ─────────────────────────────────────
    if "ushape" not in skip:
        run_ushape(models, core, args.n_perm, tex_dir)

    print()
    print("done.")


# ───────────────────────────────────────────────────────────────────────────
# 1. Mixed effects
# ───────────────────────────────────────────────────────────────────────────

def run_mixed(models: list[str], core: dict, tex_dir: Path) -> None:
    print()
    print("=" * 78)
    print("1. MIXED-EFFECTS (P0.1)  dkappa ~ A(l) + (1 | model)")
    print("=" * 78)
    rows = []
    for m in models:
        k = core[m]["kappa_level"]
        for lvl in OFF_L3:
            rows.append({"model_id": m, "level": lvl,
                         "A": core[m]["A"][lvl],
                         "dkappa": k[lvl] - k[3]})
    d = pd.DataFrame(rows)
    print(f"  cells: {len(d)} = {d.model_id.nunique()} non-reasoning models "
          f"x {d.level.nunique()} levels")
    if len(d) < 8:
        print("  too few cells for a mixed model; skipped.")
        return

    try:
        import statsmodels.formula.api as smf
    except Exception as e:                                   # noqa: BLE001
        print(f"  statsmodels unavailable ({type(e).__name__}: {e}); "
              f"stage skipped. On Colab: pip install -U statsmodels")
        return

    # Both sides are scaled by 100 for numerical conditioning. The SLOPE is
    # invariant under a common rescaling of x and y (and so is its SE and p);
    # only the intercept changes units (0.01 kappa).
    d = d.assign(y=d.dkappa * 100.0, x=d.A * 100.0)

    import warnings
    with warnings.catch_warnings(record=True) as w_ri:
        warnings.simplefilter("always")
        mdf = smf.mixedlm("y ~ x", d, groups=d["model_id"]).fit(reml=True)
    coef, se, p = mdf.params["x"], mdf.bse["x"], mdf.pvalues["x"]
    gvar = float(mdf.cov_re.iloc[0, 0])
    print()
    print("  ── random intercept per model ──")
    print(f"    fixed effect A : coef = {coef:+.4f}   SE = {se:.4f}   "
          f"p = {p:.4g}")
    print(f"    intercept      : {mdf.params['Intercept']:+.4f} "
          f"(units: 0.01 kappa)")
    print(f"    group var      : {gvar:.4f}   resid var: {float(mdf.scale):.4f}")
    print(f"    converged      : {bool(getattr(mdf, 'converged', True))}")
    for msg in sorted({str(x.message)[:70] for x in w_ri}):
        print(f"    warning: {msg}")
    if gvar < 1e-6:
        print("    NOTE: group variance is at the boundary (~0) — the random "
              "intercept adds")
        print("    nothing over pooled OLS here; a 'converged=False' flag on "
              "a boundary fit is")
        print("    expected and does not invalidate the fixed effect.")
    print("    note: coef is dimensionless (dkappa per unit A) and unchanged "
          "by the x100 scaling.")

    # Robustness: pooled OLS with model-clustered SEs (no random effect).
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ols = smf.ols("y ~ x", d).fit(
                cov_type="cluster", cov_kwds={"groups": d["model_id"]})
        print(f"    [robustness] OLS + model-clustered SE: "
              f"coef = {ols.params['x']:+.4f}  SE = {ols.bse['x']:.4f}  "
              f"p = {ols.pvalues['x']:.4g}")
    except Exception as e:                                   # noqa: BLE001
        print(f"    [robustness] clustered OLS failed: {type(e).__name__}: {e}")

    # random slope attempt
    slope_line = "random slope: not attempted"
    print()
    print("  ── random slope attempt (A | model) ──")
    try:
        with warnings.catch_warnings(record=True) as w_rs:
            warnings.simplefilter("always")
            mdf2 = smf.mixedlm("y ~ x", d, groups=d["model_id"],
                               re_formula="~x").fit(reml=True)
        conv = bool(getattr(mdf2, "converged", False))
        msgs = {str(x.message)[:70] for x in w_rs}
        print(f"    converged = {conv}")
        if msgs:
            for msg in sorted(msgs):
                print(f"    warning: {msg}")
        print(f"    fixed effect A : coef = {mdf2.params['x']:+.4f}   "
              f"SE = {mdf2.bse['x']:.4f}   p = {mdf2.pvalues['x']:.4g}")
        print("    interpretation: with 7 groups x 4 cells the slope variance "
              "is weakly identified;")
        print("    the random-intercept model is the reportable one.")
        slope_line = (f"random slope: converged={conv}, "
                      f"coef={mdf2.params['x']:+.4f}, p={mdf2.pvalues['x']:.4g}")
    except Exception as e:                                   # noqa: BLE001
        print(f"    FAILED to fit: {type(e).__name__}: {e}")
        slope_line = f"random slope: failed to fit ({type(e).__name__})"

    write_tex(
        tex_dir, "batch1r_mixedlm.tex",
        [f"$A(\\ell)$ & ${coef:+.3f}$ & ${se:.3f}$ & ${tex_p(p)}$ \\\\",
         f"intercept & ${mdf.params['Intercept']:+.3f}$ & "
         f"${mdf.bse['Intercept']:.3f}$ & ${tex_p(mdf.pvalues['Intercept'])}$ \\\\",
         f"% n cells = {len(d)}; groups = {d.model_id.nunique()}; {slope_line}"],
        "MixedLM dkappa ~ A(l) + (1|model), non-reasoning models, "
        "both sides x100 (slope scale-invariant)")


# ───────────────────────────────────────────────────────────────────────────
# 2. Query-clustered bootstrap
# ───────────────────────────────────────────────────────────────────────────

def run_bootstrap(models: list[str], packed: dict, core: dict,
                  B: int, tex_dir: Path) -> dict:
    print()
    print("=" * 78)
    print(f"2. QUERY-CLUSTERED BOOTSTRAP (P1.3)  B={B}, seed={SEED}")
    print("=" * 78)
    print("  resampling unit: whole QUERY (all its pairs move together), "
          "with replacement,")
    print("  stratified within dataset (dl19 / dl20 query counts held fixed).")

    draws: dict[str, dict[int, np.ndarray]] = {}
    tex_rows = []
    for m in models:
        pk = packed[m]
        # NOTE: each model re-seeds the SAME seed, so the per-model bootstrap
        # streams are identical-by-construction rather than jointly
        # independent; CIs are per-model and never pooled, so this only makes
        # models share resampling noise (use np.random.SeedSequence(SEED)
        # .spawn(len(models)) if a joint/pooled statistic is ever added).
        rng = np.random.default_rng(SEED)
        strata = {ds: list(gs) for ds, gs in pk["strata"].items()}
        n_q = {ds: len(gs) for ds, gs in strata.items()}
        reps = {lvl: np.empty(B) for lvl in OFF_L3}
        for b in range(B):
            picks = []
            for ds in sorted(strata):
                gs = strata[ds]
                sel = rng.integers(0, len(gs), size=n_q[ds])
                picks.extend(gs[i] for i in sel)
            idx = np.concatenate(picks)
            kl = level_kappas_from_keys(pk["keys"][idx], pk)
            for lvl in OFF_L3:
                reps[lvl][b] = kl[lvl] - kl[3]
        draws[m] = reps

        print()
        print(f"  {m}   (n_queries: "
              f"{', '.join(f'{ds}={n}' for ds, n in sorted(n_q.items()))}; "
              f"rows={pk['n_rows']})")
        print(f"    {'level':<6} {'dkappa':>9} {'95% CI':>22} {'excl. 0':>9}")
        for lvl in OFF_L3:
            point = core[m]["kappa_level"][lvl] - core[m]["kappa_level"][3]
            lo, hi = np.percentile(reps[lvl], [2.5, 97.5])
            excl = "yes" if (lo > 0 or hi < 0) else "no"
            print(f"    L{lvl:<5} {point:>+9.4f} "
                  f"[{lo:>+8.4f}, {hi:>+8.4f}] {excl:>9}")
            tex_rows.append(
                f"{esc(m)} & L{lvl} & ${point:+.4f}$ & "
                f"$[{lo:+.4f},\\,{hi:+.4f}]$ & {excl} \\\\")

    write_tex(tex_dir, "batch1r_bootstrap_ci.tex", tex_rows,
              f"Query-clustered bootstrap 95% CIs for dkappa vs L3 "
              f"(B={B}, seed={SEED}, stratified by dataset)")
    return draws


# ───────────────────────────────────────────────────────────────────────────
# 3. TOST margins
# ───────────────────────────────────────────────────────────────────────────

def run_tost(models: list[str], draws: dict, tex_dir: Path) -> None:
    print()
    print("=" * 78)
    print("3. TOST EQUIVALENCE MARGINS (P1.3)  alpha=0.05 via 90% CI")
    print("=" * 78)
    print("  m = smallest 0.001-grid margin with EVERY level's 90% bootstrap "
          "CI inside (-m, +m).")
    print(f"  Judges: non-reasoning minus {NON_BLIND_SOURCE} (non-blind source).")

    judges = [m for m in models if m != NON_BLIND_SOURCE]
    tex_rows, margins = [], {}
    for m in judges:
        worst, worst_lvl = 0.0, None
        detail = []
        for lvl in OFF_L3:
            lo, hi = np.percentile(draws[m][lvl], [5, 95])
            a = max(abs(lo), abs(hi))
            detail.append((lvl, lo, hi, a))
            if a > worst:
                worst, worst_lvl = a, lvl
        m_grid = (np.floor(worst / 0.001) + 1) * 0.001   # strict containment
        margins[m] = m_grid
        print()
        print(f"  {m}")
        for lvl, lo, hi, a in detail:
            mark = "  <- binding" if lvl == worst_lvl else ""
            print(f"    L{lvl}  90% CI [{lo:+.4f}, {hi:+.4f}]  "
                  f"max|bound|={a:.4f}{mark}")
        print(f"    equivalence margin m = {m_grid:.3f}")
        tex_rows.append(f"{esc(m)} & L{worst_lvl} & ${worst:.4f}$ & "
                        f"${m_grid:.3f}$ \\\\")

    if margins:
        mx = max(margins.values())
        arg = max(margins, key=margins.get)
        print()
        print(f"  MAX across {len(margins)} judges: m = {mx:.3f}  (driven by {arg})")
        print(f"  Claim: tone effects on kappa are statistically equivalent to "
              f"zero within +/-{mx:.3f}.")
        tex_rows.append(r"\midrule")
        tex_rows.append(f"\\textbf{{max}} & -- & -- & $\\mathbf{{{mx:.3f}}}$ \\\\")
    write_tex(tex_dir, "batch1r_tost.tex", tex_rows,
              "TOST equivalence margins (90% bootstrap CI, alpha=0.05), "
              "per judge and max")


# ───────────────────────────────────────────────────────────────────────────
# 4. U-shape permutation
# ───────────────────────────────────────────────────────────────────────────

def run_ushape(models: list[str], core: dict, n_perm: int, tex_dir: Path) -> None:
    print()
    print("=" * 78)
    print(f"4. U-SHAPE PERMUTATION (P1.4)  n_perm={n_perm}, seed={SEED}")
    print("=" * 78)
    print("  statistic: between-level variance of the 5 level means "
          "(level means over 3 paraphrases).")
    print("  H0: the 15 per-variant kappas are exchangeable across level "
          "labels.")
    print("  p = (1 + #{perm >= observed}) / (1 + n_perm)")

    tex_rows = []
    for m in models:
        vk = core[m]["variant_kappa"]
        vals = np.array([v for v in vk.values()], dtype=float)
        levels = np.array([lvl for (_, lvl) in vk.keys()])
        if len(vals) != 15 or np.isnan(vals).any():
            print(f"\n  {m}: expected 15 finite variant kappas, got "
                  f"{len(vals)} ({int(np.isnan(vals).sum())} NaN) — skipped")
            continue

        def stat(v, lv=levels):
            means = np.array([v[lv == l].mean() for l in LEVELS])
            return float(means.var())

        obs = stat(vals)
        rng = np.random.default_rng(SEED)
        perm = np.empty(n_perm)
        for i in range(n_perm):
            perm[i] = stat(rng.permutation(vals))
        p = (1 + int((perm >= obs).sum())) / (1 + n_perm)
        k = core[m]["kappa_level"]
        shape = " ".join(f"L{l}={k[l]:.4f}" for l in LEVELS)
        sig = "***" if p < 0.01 else ("*" if p < 0.05 else "")
        print()
        print(f"  {m}")
        print(f"    {shape}")
        print(f"    between-level var = {obs:.6f}   p = {p:.4f} {sig}")
        tex_rows.append(f"{esc(m)} & ${obs:.5f}$ & ${p:.4f}$ \\\\")

    write_tex(tex_dir, "batch1r_ushape.tex", tex_rows,
              f"U-shape permutation test: between-level variance of the 15 "
              f"per-variant kappas ({n_perm} permutations, seed={SEED})")


if __name__ == "__main__":
    main()
