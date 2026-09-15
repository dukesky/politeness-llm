"""Batch 4 — operating-point calibration ("tone dial" + output remap).

WHAT THIS ANALYSIS CLAIMS (read before quoting any number from it)
------------------------------------------------------------------
The correction studied here removes a **MEASUREMENT ARTIFACT**, not a defect
of the judge's reasoning.  Round 1 and round 2 established that a tone wrapper
moves the judge's *operating point* (its severity/leniency threshold) rather
than its ranking ability, and that kappa against a fixed human reference moves
up or down purely as a function of whether that shift lands closer to or
further from the **reference annotators' own operating point** (sign of
Delta = judge L3 mean - human mean, times the direction of D(l)).

Consequently:

  * Choosing a tone level (C2) or remapping the judge's output labels (C3)
    **re-aligns the judge's operating point with the reference annotators'**.
    Any kappa gain is the artifact being removed, nothing more.
  * It does **NOT** make the judge a better judge: the underlying ordering of
    documents is essentially untouched by both corrections (a monotone label
    map cannot reorder anything at all, and the dial only shifts a threshold).
  * A gain therefore must never be reported as "politeness improves LLM
    judgment", "prompt tone boosts accuracy", or any quality claim.  The
    honest statement is: *the measured agreement of an LLM judge against a
    human reference is confounded by operating-point misalignment; that
    confound is estimable from a small labeled calibration set and can be
    removed, after which the residual tone effect is what deserves to be
    called a real effect.*
  * The correction is also *reference-relative*: it calibrates to THESE
    annotators.  It transfers to another pool only insofar as that pool shares
    the same grade distribution.

Sections (each skippable with --skip):

  0. auth — AUTHORITY ASSERTION.  Recompute per-(model, dataset-scope, level)
     kappa on the authoritative paraphrase-mean path and assert it reproduces
     the round-2 开箱结果 values recorded in paper/PREDICTIONS.md (tol 5e-4).
     Nothing downstream is trustworthy if this fails, so it fails loudly.
  1. c1   — C1 ESTIMATION & SAMPLE EFFICIENCY.  How many labeled pairs does
     the dial need?  RMSE of Delta-hat, P(correct level picked) and expected
     regret, over query-clustered calibration subsamples of size n.
  2. c2   — C2 TONE-DIAL EVALUATION, query-disjoint cross-fit: pick the level
     on fold A, pay for it on fold B.  Reported against the oracle bound and
     against a sign-only heuristic baseline.
  3. c3   — C3 OUTPUT-SIDE REMAP: best monotone map {0..3}->{0..3} fitted on
     fold A at L3, evaluated on fold B; plus C2+C3 combined.  Also reports
     |Delta| held-out before vs after, which is the artifact-removal metric
     that the framing above actually licenses.
  4. (always) SUMMARY + LaTeX + frozen correction rules JSON for C4.

DATASETS: dl19, dl20, dl21, antique.  **dl22 is SEALED** and hard-excluded:
the dataset list goes through src.qrels.assert_not_sealed before a single
kappa is computed, so asking for dl22 aborts with the protocol message rather
than silently leaking the holdout.

Usage (Colab, repo root):
    python scripts/batch4_correction.py --data-dir $DATA_DIR
    python scripts/batch4_correction.py --data-dir $DATA_DIR --b 200      # smoke
    python scripts/batch4_correction.py --data-dir $DATA_DIR --skip c1

Outputs (under DATA_DIR, never in this repo — it carries no data artefacts):
    $DATA_DIR/derived/tex/batch4_*.tex
    $DATA_DIR/derived/correction_rules.json
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from src.qrels import assert_not_sealed, load_qrels, parse_datasets  # noqa: E402
from validate_model import level_summary_from_table, per_variant_kappa  # noqa: E402
from batch1r_stats import (  # noqa: E402
    cms_from_keys,
    esc,
    kappa_from_cm,
    write_tex,
)

# ── scope constants ────────────────────────────────────────────────────────
ALLOWED_DATASETS = ("dl19", "dl20", "dl21", "antique")   # dl22 NEVER here
LEVELS = [1, 2, 3, 4, 5]
OFF_L3 = [1, 2, 4, 5]
SEED = 42
KAPPA_TOL = 5e-4
MIN_N_KAPPA = 10          # matches src/metrics.agreement_table: kappa needs n>10
CAL_SIZES = (25, 50, 100, 200, 400)

# The six round-2 judges (authority assertion scopes: each x dl21, x antique).
ROUND2_MODELS = [
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v4.1-flash",
    "google/gemini-3.5-flash",
    "google/gemini-3.8-flash",
    "anthropic/claude-opus-5",
    "openai/gpt-5.6-sol",
]
# Round-2 flagships additionally re-ran round-1 data as a generational pair.
ROUND2_V1_MODELS = ["anthropic/claude-opus-5", "openai/gpt-5.6-sol"]

# Analysis scopes for C1/C2/C3: dl19+dl20 are pooled into one msmarco-v1 scope.
SCOPES: dict[str, tuple[str, ...]] = {
    "dlv1": ("dl19", "dl20"),
    "dl21": ("dl21",),
    "antique": ("antique",),
}
# Frozen correction rules are fitted per dataset FAMILY, not per scope: the
# msmarco family pools dl19+dl20+dl21 (this is the rule C4 pre-registers for
# the sealed dl22, which is msmarco-v2); antique is its own family.
FAMILIES: dict[str, tuple[str, ...]] = {
    "msmarco": ("dl19", "dl20", "dl21"),
    "antique": ("antique",),
}
FAMILY_OF_SCOPE = {"dlv1": "msmarco", "dl21": "msmarco", "antique": "antique"}


# ═══════════════════════════════════════════════════════════════════════════
# Monotone label maps {0,1,2,3} -> {0,1,2,3}
# ═══════════════════════════════════════════════════════════════════════════

def monotone_maps() -> list[tuple[int, int, int, int]]:
    """All non-decreasing maps of a 4-level ordered scale onto itself.

    There are exactly C(4+4-1, 4) = C(7,3) = 35 of them (multisets of size 4
    drawn from 4 ordered values), constant maps included.  They are the only
    output-side corrections that cannot reorder documents, which is what makes
    C3 an operating-point correction rather than a quality intervention.
    """
    return [m for m in itertools.product(range(4), repeat=4)
            if all(m[i] <= m[i + 1] for i in range(3))]


MONOTONE_MAPS = monotone_maps()
IDENTITY_MAP = (0, 1, 2, 3)


# ═══════════════════════════════════════════════════════════════════════════
# Small helpers
# ═══════════════════════════════════════════════════════════════════════════

def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True, cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:                                        # noqa: BLE001
        return "unknown"


def fold_of(dataset: str, qid: str, seed: int) -> int:
    """Deterministic query-disjoint fold assignment (0 / 1).

    Hash of "seed|dataset|qid" — stable across platforms, python versions and
    row order (unlike hash()), and keyed by dataset because query ids are only
    unique within a collection.
    """
    h = hashlib.md5(f"{seed}|{dataset}|{qid}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 2


def resolve_datasets(spec: str) -> list[str]:
    """Parse --datasets, refuse the sealed holdout, refuse anything else."""
    names = parse_datasets(spec)
    # SEAL CHECK — before any kappa is computed anywhere in this script.
    assert_not_sealed(names)
    bad = [n for n in names if n not in ALLOWED_DATASETS]
    if bad:
        sys.exit(
            f"ERROR: batch 4 correction is defined on {list(ALLOWED_DATASETS)} "
            f"only; got {bad}.\n"
            f"  dl22 is the sealed holdout and is hard-excluded here even with "
            f"--unseal-dl22:\n"
            f"  its correction rule is PRE-REGISTERED from the msmarco family "
            f"rules this script writes."
        )
    return names


def flagship_model_ids(models_yaml: Path) -> set:
    """Flagship model ids from ANY round (round-2 flagships included)."""
    import yaml
    spec = yaml.safe_load(models_yaml.read_text(encoding="utf-8"))
    return {m["model_id"] for m in spec["models"] if m.get("tier") == "flagship"}


_PAIRS_FILE = re.compile(r"^pairs_([A-Za-z0-9.]+)_flagship\d+\.jsonl$")


def dataset_of_pairs_file(path: Path) -> str:
    """Infer the dataset of a frozen pairs file from its filename.

    The pairs files carry no ``dataset`` field, but msmarco-v1, msmarco-v2 and
    ANTIQUE all use bare integer query ids — so a 2-tuple (qid, docid) frozen
    set can silently admit a dl21 pair because dl19 froze the same numbers.
    Every key in this script is therefore dataset-qualified, and the dataset
    comes from the filename the freeze wrote.
    """
    m = _PAIRS_FILE.match(path.name)
    if not m or m.group(1) not in ALLOWED_DATASETS:
        sys.exit(
            f"ERROR: cannot infer the dataset of frozen pairs file "
            f"'{path.name}'.\n"
            f"  Expected inputs/pairs_<dataset>_flagship<NN>.jsonl with "
            f"dataset in {list(ALLOWED_DATASETS)}.\n"
            f"  Pairs files carry no dataset column, and an unqualified "
            f"(qid, docid) key would cross-\n"
            f"  contaminate collections that share bare integer query ids."
        )
    return m.group(1)


def load_frozen_pairs_qualified(paths) -> set:
    """Frozen pairs as dataset-qualified ``(dataset, qid, docid)`` keys."""
    frozen = set()
    for fp in paths:
        ds = dataset_of_pairs_file(Path(fp))
        with open(fp) as f:
            for line in f:
                p = json.loads(line)
                frozen.add((ds, str(p["qid"]), str(p["docid"])))
    return frozen


def flagship_pairs_for(data_dir: Path, datasets) -> tuple:
    """Union of the frozen 40% pairs files for ``datasets`` (missing = empty).

    Returns dataset-qualified keys; see ``dataset_of_pairs_file``.
    """
    paths = [data_dir / "inputs" / f"pairs_{ds}_flagship40.jsonl"
             for ds in datasets]
    paths = [p for p in paths if p.exists()]
    return (load_frozen_pairs_qualified(paths) if paths else set()), paths


def scope_frame(df: pd.DataFrame, model: str, datasets, frozen: set,
                is_flagship: bool, label: str) -> pd.DataFrame:
    """Rows for one (model, dataset-scope), with the flagship restriction."""
    d = df[(df.model_id == model) & (df.dataset.isin(datasets))]
    if d.empty:
        return d
    if is_flagship and frozen:
        keys = list(zip(d.dataset.astype(str), d.qid.astype(str),
                        d.docid.astype(str)))
        mask = np.fromiter((k in frozen for k in keys), dtype=bool,
                           count=len(keys))
        d = d[mask]
        if d.empty:
            sys.exit(
                f"ERROR: flagship pairs filter left ZERO rows for "
                f"'{model}' on {label}.\n"
                f"  {len(frozen)} frozen pairs did not intersect this model's "
                f"collected pairs — wrong\n"
                f"  pairs files for this scope, or the model was collected on "
                f"a different sample."
            )
    return d


# ═══════════════════════════════════════════════════════════════════════════
# 0. Authoritative round-2 values from paper/PREDICTIONS.md
# ═══════════════════════════════════════════════════════════════════════════

_ROUND2_MARK = re.compile(r"^#\s*Round 2\s")
_MODEL_HEAD = re.compile(r"^###\s+(\S+)\s+—")
_DATASET_LINE = re.compile(r"^-\s*数据集[:：]\s*(.+?)\s*$")
_KAPPA_KV = re.compile(r"κ\(L([1-5])\)\s*=\s*(-?\d+\.\d+)")


def parse_round2_predictions(path: Path) -> dict:
    """Extract round-2 开箱结果 kappas keyed by (model_id, dataset-scope).

    The round-2 section registers the SAME model several times, once per
    dataset, so the model heading alone is not a key — the following
    ``- 数据集: X`` line completes it.  Only lines carrying at least three
    ``κ(Lk)=<float>`` values are taken (that is the 开箱结果 line; the
    registration template's ``{}`` placeholders match no float).
    """
    if not path.exists():
        sys.exit(f"ERROR: {path} not found — cannot run the authority assertion.")

    out: dict = {}
    in_round2 = False
    model = None
    ds_key = None
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if _ROUND2_MARK.match(s):
            in_round2 = True
            continue
        if not in_round2:
            continue
        head = _MODEL_HEAD.match(s)
        if head:
            model, ds_key = head.group(1), None
            continue
        if s.startswith("#"):
            # Any other heading (e.g. the trailing 开箱总评 / 盲态备注 prose
            # sections) closes the current block: kappa-like strings there must
            # never be attributed to the last model seen.
            model, ds_key = None, None
            continue
        dsl = _DATASET_LINE.match(s)
        if dsl and model:
            ds_key = dsl.group(1).replace(" ", "")
            continue
        if model is None or ds_key is None:
            continue
        kvs = _KAPPA_KV.findall(s)
        if len(kvs) >= 3:
            out.setdefault((model, ds_key), {}).update(
                {int(l): float(v) for l, v in kvs})
    return out


def authority_scopes() -> list:
    """(model, dataset-key) scopes the assertion must cover."""
    scopes = [(m, ds) for m in ROUND2_MODELS for ds in ("dl21", "antique")]
    scopes += [(m, "dl19,dl20") for m in ROUND2_V1_MODELS]
    return scopes


def run_authority(df: pd.DataFrame, data_dir: Path, qrels_all: dict,
                  flagships: set, tol: float) -> None:
    print()
    print("=" * 78)
    print("0. AUTHORITY ASSERTION vs paper/PREDICTIONS.md (round-2 section)")
    print("=" * 78)
    recorded = parse_round2_predictions(REPO_ROOT / "paper" / "PREDICTIONS.md")
    if not recorded:
        sys.exit("ERROR: parsed ZERO round-2 开箱结果 blocks from "
                 "paper/PREDICTIONS.md — the section layout changed; fix the "
                 "parser before trusting anything downstream.")
    print(f"  parsed {len(recorded)} recorded (model, dataset) kappa blocks")

    diffs, checked, skipped = [], 0, []
    fast_worst, fast_nan_both, fast_nan_one, fast_cells = 0.0, [], [], 0
    for model, ds_key in authority_scopes():
        auth = recorded.get((model, ds_key))
        if not auth:
            skipped.append(f"{model} @ {ds_key} (not recorded)")
            continue
        datasets = ds_key.split(",")
        frozen, _paths = flagship_pairs_for(data_dir, datasets)
        d = scope_frame(df, model, datasets, frozen,
                        model in flagships, ds_key)
        if d.empty:
            skipped.append(f"{model} @ {ds_key} (no rows in parquet)")
            continue
        table = per_variant_kappa(d, qrels_all)
        if table.empty:
            sys.exit(f"ERROR: '{model}' @ {ds_key} has no scorable "
                     f"(parse_ok, qrels-matched) rows; cannot assert.")
        summary = level_summary_from_table(table)
        got = {int(r.politeness_level): float(r.kappa_mean)
               for _, r in summary.iterrows()}
        for lvl, kauth in sorted(auth.items()):
            kcalc = got.get(lvl, np.nan)
            checked += 1
            if not np.isfinite(kcalc) or abs(kcalc - kauth) > tol:
                diffs.append((model, ds_key, lvl, kauth, kcalc))

        # ── fast-path cross-check ─────────────────────────────────────────
        # C1/C2/C3 never call the sklearn path: they run on the Packed /
        # cms_from_keys / kappa_from_cm machinery. The authority assertion
        # must therefore also certify THAT path, or it certifies code nobody
        # downstream uses. NaN-aware: max(x, nan) silently returns x, so a NaN
        # on exactly one side would otherwise pass unnoticed.
        P = pack(d, qrels_all, model, ds_key, SEED)
        fast = level_kappas(P, np.arange(len(P.h)))
        for lvl in LEVELS:
            a, b = fast.get(lvl, np.nan), got.get(lvl, np.nan)
            fa, fb = np.isfinite(a), np.isfinite(b)
            if not fa and not fb:
                fast_nan_both.append((model, ds_key, lvl))
                continue
            if fa != fb:
                fast_nan_one.append((model, ds_key, lvl, a, b))
                continue
            fast_cells += 1
            fast_worst = max(fast_worst, abs(a - b))

    for s in skipped:
        print(f"  [warn] skipped {s}")
    if diffs:
        print()
        print(f"  !! {len(diffs)} MISMATCH(ES) at tol={tol}:")
        print(f"  {'model':<30} {'scope':<10} {'lvl':>3} {'RECORDED':>10} "
              f"{'recomputed':>11} {'diff':>10}")
        for m, ds, lvl, ka, kc in diffs:
            print(f"  {m:<30} {ds:<10} {lvl:>3} {ka:>10.4f} {kc:>11.4f} "
                  f"{kc - ka:>+10.4f}")
        print()
        print("  Nothing downstream is trustworthy until this is resolved.")
        print("  Check: dataset scope, the 40% flagship pairs restriction for "
              "opus-5 / gpt-5.6-sol,")
        print("  and that the paraphrase-MEAN path is used (a pooled "
              "recompute is NOT authoritative).")
        sys.exit(1)
    if checked == 0:
        sys.exit("ERROR: the authority assertion checked ZERO cells — the "
                 "parquet does not contain the round-2 scopes. Refusing to "
                 "continue on unverified data (use --skip auth only for a "
                 "synthetic stub drive).")
    print(f"  OK — {checked} (model, scope, level) kappas reproduce within {tol}.")

    for m, ds, lvl in fast_nan_both:
        print(f"  [note] {m} @ {ds} L{lvl}: kappa undefined on BOTH paths — "
              f"cell skipped in the cross-check")
    if fast_nan_one:
        for m, ds, lvl, a, b in fast_nan_one:
            print(f"  !! {m} @ {ds} L{lvl}: fast={a}  authoritative={b} "
                  f"(NaN on exactly one path)")
        sys.exit("ERROR: the fast kappa path and the sklearn path disagree on "
                 "whether a cell is defined; every C1/C2/C3 number would be "
                 "incomparable with the point estimates.")
    print(f"  fast (bincount) kappa vs sklearn path: {fast_cells} cells, "
          f"max |diff| = {fast_worst:.2e}")
    if fast_worst > 1e-9:
        sys.exit("ERROR: the fast kappa path used by C1/C2/C3 disagrees with "
                 "the authoritative sklearn path; fix it before trusting any "
                 "correction number.")


# ═══════════════════════════════════════════════════════════════════════════
# Packing: one (model, scope) into numpy arrays
# ═══════════════════════════════════════════════════════════════════════════

class Packed:
    """Row arrays for one (model, scope), restricted to qrels-matched rows.

    Rows where the judge failed to parse are KEPT (with score NaN) because the
    human mean that defines Delta is taken over all matched rows, exactly as
    scripts/preflight_distributions.py and scripts/batch1r_stats.py do; the
    ``ok`` mask selects the scorable subset for score means and for kappa.
    """

    __slots__ = ("model", "scope", "h", "s", "lvl", "v", "pair", "q", "fold",
                 "ok", "pos_in_ok", "h_ok", "s_ok", "v_ok", "n_variants",
                 "variants", "vlevel", "pair_rows", "query_pairs", "n_pairs",
                 "n_queries")


def pack(df_m: pd.DataFrame, qrels: dict, model: str, scope: str,
         seed: int) -> Packed:
    d = df_m.copy()
    d["human"] = [qrels.get((str(ds), str(q), str(dd)))
                  for ds, q, dd in zip(d.dataset, d.qid, d.docid)]
    d = d.dropna(subset=["human"])
    if d.empty:
        sys.exit(f"ERROR: '{model}' @ {scope}: no rows survive the qrels join. "
                 f"Check the dataset column and src/qrels.py (ANTIQUE 1-4 -> "
                 f"0-3) mapping.")

    P = Packed()
    P.model, P.scope = model, scope
    P.h = d.human.astype(int).to_numpy()
    P.s = d.score.astype(float).to_numpy()
    P.lvl = d.politeness_level.astype(int).to_numpy()

    P.variants = sorted(d.prompt_id.unique())
    vidx = {p: i for i, p in enumerate(P.variants)}
    P.n_variants = len(P.variants)
    P.v = d.prompt_id.map(vidx).to_numpy(dtype=np.int64)
    vl = (d.drop_duplicates("prompt_id").set_index("prompt_id")
           .politeness_level.astype(int).to_dict())
    P.vlevel = {vidx[p]: int(vl[p]) for p in P.variants}

    ds_s = d.dataset.astype(str).to_numpy()
    qid_s = d.qid.astype(str).to_numpy()
    doc_s = d.docid.astype(str).to_numpy()

    pair_key = pd.Series([f"{a}|{b}|{c}" for a, b, c in zip(ds_s, qid_s, doc_s)])
    q_key = pd.Series([f"{a}|{b}" for a, b in zip(ds_s, qid_s)])
    P.pair = pair_key.astype("category").cat.codes.to_numpy()
    P.q = q_key.astype("category").cat.codes.to_numpy()
    P.n_pairs = int(P.pair.max()) + 1
    P.n_queries = int(P.q.max()) + 1

    fold_cache = {}
    folds = np.empty(len(d), dtype=np.int64)
    for i, (ds, q) in enumerate(zip(ds_s, qid_s)):
        key = (ds, q)
        if key not in fold_cache:
            fold_cache[key] = fold_of(ds, q, seed)
        folds[i] = fold_cache[key]
    P.fold = folds

    P.ok = (d.parse_ok.to_numpy().astype(bool)) & np.isfinite(P.s)
    P.pos_in_ok = np.full(len(d), -1, dtype=np.int64)
    P.pos_in_ok[P.ok] = np.arange(int(P.ok.sum()))
    P.h_ok = P.h[P.ok]
    P.s_ok = P.s[P.ok].astype(np.int64)
    P.v_ok = P.v[P.ok]
    if P.h_ok.size and (P.h_ok.min() < 0 or P.h_ok.max() > 3
                        or P.s_ok.min() < 0 or P.s_ok.max() > 3):
        sys.exit(f"ERROR: '{model}' @ {scope}: human/score outside 0-3 — check "
                 f"src/qrels.py grade mapping and src/parse.py.")

    # pair -> row positions, query -> pair ids (both sorted => deterministic)
    order = np.argsort(P.pair, kind="stable")
    bounds = np.searchsorted(P.pair[order], np.arange(P.n_pairs + 1))
    P.pair_rows = [order[bounds[i]:bounds[i + 1]] for i in range(P.n_pairs)]
    qp: dict = {}
    seen = set()
    for i in range(len(d)):
        p = int(P.pair[i])
        if p in seen:
            continue
        seen.add(p)
        qp.setdefault(int(P.q[i]), []).append(p)
    P.query_pairs = [np.array(sorted(qp.get(qi, [])), dtype=np.int64)
                     for qi in range(P.n_queries)]
    return P


# ═══════════════════════════════════════════════════════════════════════════
# Core estimands on an arbitrary row-index array (duplicates allowed)
# ═══════════════════════════════════════════════════════════════════════════

def delta_D(P: Packed, idx: np.ndarray, strict: bool = True):
    """(Delta, D(l), per-level mean score, human mean) on rows ``idx``.

    Delta = judge L3 mean - human mean; D(l) = level-l mean - L3 mean.  The
    human mean is taken over ALL matched rows in ``idx`` (parse failures
    included) and the score means over the scorable ones — the Step-A
    convention, so these numbers are comparable with PREDICTIONS.md.
    """
    if idx.size == 0:
        if strict:
            sys.exit(f"ERROR: '{P.model}' @ {P.scope}: empty row selection for "
                     f"Delta/D — the scope or fold has no matched pairs.")
        return None
    h = P.h[idx]
    qmean = float(h.mean())
    ok = P.ok[idx]
    lvl = P.lvl[idx]
    s = P.s[idx]
    means = {}
    for l in LEVELS:
        m = ok & (lvl == l)
        if not m.any():
            if strict:
                sys.exit(f"ERROR: '{P.model}' @ {P.scope}: no scorable rows at "
                         f"politeness level L{l}; Delta/D(l) and the L3 "
                         f"baseline are undefined.")
            return None
        means[l] = float(s[m].mean())
    delta = means[3] - qmean
    D = {l: means[l] - means[3] for l in LEVELS}      # D[3] == 0 by construction
    return delta, D, means, qmean


def level_kappas(P: Packed, idx: np.ndarray, smap=None,
                 levels=LEVELS) -> dict:
    """Paraphrase-mean linear-weighted kappa per level on rows ``idx``.

    ``smap`` optionally applies a monotone label map to the judge score first.
    Variants with n <= MIN_N_KAPPA are dropped, mirroring
    src/metrics.agreement_table, so this path stays comparable with the
    authoritative one.
    """
    sel = P.pos_in_ok[idx]
    sel = sel[sel >= 0]
    if sel.size == 0:
        return {l: np.nan for l in levels}
    v = P.v_ok[sel]
    h = P.h_ok[sel]
    s = P.s_ok[sel]
    if smap is not None and tuple(smap) != IDENTITY_MAP:
        s = np.asarray(smap, dtype=np.int64)[s]
    cms = cms_from_keys(v * 16 + h * 4 + s, P.n_variants)
    per_level: dict = {l: [] for l in levels}
    for vi in range(P.n_variants):
        lv = P.vlevel[vi]
        if lv not in per_level:
            continue
        if cms[vi].sum() <= MIN_N_KAPPA:
            continue
        per_level[lv].append(kappa_from_cm(cms[vi]))
    out = {}
    for l in levels:
        vals = [x for x in per_level[l] if np.isfinite(x)]
        out[l] = float(np.mean(vals)) if vals else np.nan
    return out


def mean_score_mapped(P: Packed, idx: np.ndarray, level: int, smap) -> float:
    sel_rows = idx[P.ok[idx] & (P.lvl[idx] == level)]
    if sel_rows.size == 0:
        return np.nan
    s = P.s[sel_rows].astype(np.int64)
    if smap is not None and tuple(smap) != IDENTITY_MAP:
        s = np.asarray(smap, dtype=np.int64)[s]
    return float(s.mean())


def best_level(delta: float, D: dict) -> int:
    """Dial rule: the level whose shift best cancels the misalignment."""
    return min(LEVELS, key=lambda l: (abs(delta + D[l]), l))


def sign_only_level(delta: float, D: dict) -> int:
    """Simpler baseline: ignore magnitudes, only push against the sign of Delta.

    Delta > 0 (judge too lenient) -> the most severity-shifting level
    (most negative D); Delta < 0 -> the most leniency-shifting one.
    """
    if delta > 0:
        return min(LEVELS, key=lambda l: (D[l], l))
    return max(LEVELS, key=lambda l: (D[l], -l))


def fit_monotone_map(P: Packed, idx: np.ndarray, level: int):
    """Monotone map maximising paraphrase-mean kappa at ``level`` on ``idx``."""
    best, best_k = IDENTITY_MAP, -np.inf
    for m in MONOTONE_MAPS:
        k = level_kappas(P, idx, smap=m, levels=[level])[level]
        if np.isfinite(k) and k > best_k:
            best, best_k = m, k
    return tuple(best), (best_k if np.isfinite(best_k) else np.nan)


# ═══════════════════════════════════════════════════════════════════════════
# 1. C1 — estimation & sample efficiency
# ═══════════════════════════════════════════════════════════════════════════

def sample_calibration_idx(P: Packed, n_pairs: int, rng) -> np.ndarray:
    """Query-clustered calibration subsample of ``n_pairs`` DISTINCT pairs.

    Queries are drawn at random (a calibration set is collected query by
    query, and pairs inside a query are not independent), but a pair already
    in the set is never counted twice: ``n`` means n distinct labeled pairs —
    n items a human actually had to grade — so the x-axis of C1 is the real
    annotation budget. Extra queries are drawn until n distinct pairs are
    reached, and the last query is truncated so every size is exact.
    """
    if n_pairs > P.n_pairs:
        sys.exit(f"ERROR: '{P.model}' @ {P.scope}: asked for {n_pairs} distinct "
                 f"calibration pairs but the scope only has {P.n_pairs}.")
    chosen: list = []
    seen: set = set()
    attempts = 0
    max_attempts = 200 * P.n_queries + 1000
    while len(chosen) < n_pairs:
        attempts += 1
        if attempts > max_attempts:
            sys.exit(f"ERROR: '{P.model}' @ {P.scope}: could not assemble "
                     f"{n_pairs} distinct calibration pairs in {attempts} "
                     f"query draws ({P.n_pairs} pairs over {P.n_queries} "
                     f"queries) — the pair/query structure is degenerate.")
        for p in P.query_pairs[int(rng.integers(0, P.n_queries))]:
            p = int(p)
            if p in seen:
                continue
            seen.add(p)
            chosen.append(p)
            if len(chosen) >= n_pairs:
                break
    return np.concatenate([P.pair_rows[p] for p in chosen])


def run_c1(packs: dict, B: int, sizes, seed: int, tex_dir: Path) -> dict:
    print()
    print("=" * 78)
    print(f"1. C1 — CALIBRATION-SET SIZE vs DIAL QUALITY   B={B}, seed={seed}")
    print("=" * 78)
    print("  Delta-hat / D-hat estimated from n query-clustered labeled pairs;")
    print("  dial = argmin_l |Delta-hat + D-hat(l)|.  Regret is measured with "
          "the FULL-data")
    print("  Delta and D, i.e. how much misalignment the small-sample choice "
          "leaves on the table.")

    results: dict = {}
    tex_rows = []
    for (model, scope), P in sorted(packs.items()):
        gt = delta_D(P, np.arange(len(P.h)))
        delta, D, _means, _qm = gt
        lstar = best_level(delta, D)
        results[(model, scope)] = {
            "delta": delta, "D": D, "lstar": lstar, "sizes": {},
            "n_pairs": P.n_pairs, "n_queries": P.n_queries,
        }
        print()
        print(f"  {model}  @ {scope}   "
              f"(pairs={P.n_pairs}, queries={P.n_queries})")
        print(f"    full data: Delta={delta:+.4f}  "
              f"D={{{', '.join(f'L{l}:{D[l]:+.4f}' for l in LEVELS)}}}  "
              f"-> l* = L{lstar}  |Delta+D(l*)|={abs(delta + D[lstar]):.4f}")
        print(f"    {'n':>6} {'RMSE(Delta)':>12} {'P(l*)':>8} "
              f"{'E[regret]':>10} {'n_ok':>6}")
        for n in sizes:
            if n > P.n_pairs:
                print(f"    {n:>6}   WARNING: skipped — this scope has only "
                      f"{P.n_pairs} distinct labeled pairs, fewer than n")
                continue
            rng = np.random.default_rng(seed + 1000 * n)
            errs, hits, regrets, bad = [], 0, [], 0
            for _b in range(B):
                idx = sample_calibration_idx(P, n, rng)
                est = delta_D(P, idx, strict=False)
                if est is None:
                    bad += 1
                    continue
                dh, Dh, _m, _q = est
                lh = best_level(dh, Dh)
                errs.append(dh - delta)
                hits += int(lh == lstar)
                regrets.append(abs(delta + D[lh]) - abs(delta + D[lstar]))
            n_ok = len(errs)
            if n_ok == 0:
                print(f"    {n:>6}   skipped — every replicate missed a "
                      f"politeness level")
                continue
            rmse = float(np.sqrt(np.mean(np.square(errs))))
            phit = hits / n_ok
            reg = float(np.mean(regrets))
            results[(model, scope)]["sizes"][n] = {
                "rmse_delta": rmse, "p_hit": phit, "regret": reg,
                "n_replicates": n_ok, "n_degenerate": bad,
            }
            print(f"    {n:>6} {rmse:>12.4f} {phit:>8.3f} {reg:>10.4f} "
                  f"{n_ok:>6}" + (f"   ({bad} degenerate)" if bad else ""))
            tex_rows.append(
                f"{esc(model)} & {esc(scope)} & {n} & ${rmse:.4f}$ & "
                f"${phit:.3f}$ & ${reg:.4f}$ \\\\")

    write_tex(tex_dir, "batch4_c1_sample_efficiency.tex", tex_rows,
              f"C1: calibration-set size vs dial quality "
              f"(B={B}, query-clustered, seed={seed}). "
              f"RMSE of Delta-hat, P(dial == full-data l*), expected regret "
              f"|Delta+D(l_hat)| - |Delta+D(l*)|")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 2/3. C2 (tone dial) and C3 (output remap), query-disjoint cross-fit
# ═══════════════════════════════════════════════════════════════════════════

def fold_indices(P: Packed):
    a = np.where(P.fold == 0)[0]
    b = np.where(P.fold == 1)[0]
    return a, b


def crossfit(P: Packed, do_c2: bool, do_c3: bool) -> dict:
    """One (model, scope) cross-fit: fit on A, pay on B, then swap."""
    fa, fb = fold_indices(P)
    if fa.size == 0 or fb.size == 0:
        sys.exit(f"ERROR: '{P.model}' @ {P.scope}: the deterministic query "
                 f"split left one fold empty (A={fa.size}, B={fb.size} rows). "
                 f"This scope has too few distinct queries to cross-fit.")

    halves = []
    for fit_idx, ev_idx, name in ((fa, fb, "A->B"), (fb, fa, "B->A")):
        est = delta_D(P, fit_idx, strict=False)
        if est is None:
            sys.exit(f"ERROR: '{P.model}' @ {P.scope}: fold {name[0]} lacks a "
                     f"politeness level; cannot estimate the dial on it.")
        d_fit, D_fit, _m, _q = est
        ev = delta_D(P, ev_idx, strict=False)
        if ev is None:
            sys.exit(f"ERROR: '{P.model}' @ {P.scope}: evaluation fold lacks a "
                     f"politeness level; held-out kappa is undefined.")
        d_ev, _D_ev, _m2, qm_ev = ev

        k_ev = level_kappas(P, ev_idx)
        k3 = k_ev[3]
        half = {"fold": name, "kappa_ev": k_ev, "kappa_L3": k3,
                "delta_fit": d_fit, "D_fit": D_fit, "delta_ev_L3": d_ev}

        if do_c2:
            l_dial = best_level(d_fit, D_fit)
            l_sign = sign_only_level(d_fit, D_fit)
            finite = {l: v for l, v in k_ev.items() if np.isfinite(v)}
            l_oracle = (max(finite, key=lambda l: finite[l]) if finite else 3)
            half.update({
                "dial_level": l_dial,
                "dkappa_dial": k_ev[l_dial] - k3,
                "sign_level": l_sign,
                "dkappa_sign": k_ev[l_sign] - k3,
                "oracle_level": l_oracle,
                "dkappa_oracle": k_ev[l_oracle] - k3,
                "delta_ev_dial": mean_score_mapped(P, ev_idx, l_dial,
                                                   IDENTITY_MAP) - qm_ev,
            })

        if do_c3:
            m3, k_fit3 = fit_monotone_map(P, fit_idx, 3)
            k_ev_m3 = level_kappas(P, ev_idx, smap=m3, levels=[3])[3]
            half.update({
                "map_L3": m3, "map_L3_kappa_fit": k_fit3,
                "dkappa_remap": k_ev_m3 - k3,
                "delta_ev_remap": mean_score_mapped(P, ev_idx, 3, m3) - qm_ev,
            })
            if do_c2:
                l_dial = half["dial_level"]
                mc, k_fitc = fit_monotone_map(P, fit_idx, l_dial)
                k_ev_c = level_kappas(P, ev_idx, smap=mc,
                                      levels=[l_dial])[l_dial]
                half.update({
                    "map_combined": mc, "map_combined_kappa_fit": k_fitc,
                    "dkappa_combined": k_ev_c - k3,
                    "delta_ev_combined": mean_score_mapped(
                        P, ev_idx, l_dial, mc) - qm_ev,
                })
        halves.append(half)

    def avg(key):
        vals = [h[key] for h in halves if key in h and np.isfinite(h[key])]
        return float(np.mean(vals)) if vals else np.nan

    out = {"halves": halves, "kappa_L3": avg("kappa_L3"),
           "abs_delta_L3": float(np.mean([abs(h["delta_ev_L3"])
                                          for h in halves]))}
    for k in ("dkappa_dial", "dkappa_sign", "dkappa_oracle",
              "dkappa_remap", "dkappa_combined"):
        if any(k in h for h in halves):
            out[k] = avg(k)
    for k in ("delta_ev_dial", "delta_ev_remap", "delta_ev_combined"):
        vals = [abs(h[k]) for h in halves if k in h and np.isfinite(h[k])]
        if vals:
            out["abs_" + k] = float(np.mean(vals))
    return out


def run_c2_c3(packs: dict, do_c2: bool, do_c3: bool, tex_dir: Path) -> dict:
    print()
    print("=" * 78)
    title = " + ".join([s for s, on in (("2. C2 TONE DIAL", do_c2),
                                        ("3. C3 OUTPUT REMAP", do_c3)) if on])
    print(f"{title}  — query-disjoint cross-fit (seed-hashed folds)")
    print("=" * 78)
    print("  Fold A fits (dial level and/or monotone map); fold B pays. Then "
          "swapped and averaged.")
    print("  Every reported Dkappa is HELD OUT.  A positive number is "
          "artifact removed, NOT")
    print("  judgment improved: the map cannot reorder documents and the dial "
          "only moves a threshold.")
    print(f"  monotone maps enumerated: {len(MONOTONE_MAPS)} "
          f"(non-decreasing {{0..3}}->{{0..3}}, constants included)")

    out: dict = {}
    rows_c2, rows_c3 = [], []
    for (model, scope), P in sorted(packs.items()):
        r = crossfit(P, do_c2, do_c3)
        out[(model, scope)] = r
        print()
        print(f"  {model}  @ {scope}    held-out kappa(L3) = "
              f"{r['kappa_L3']:.4f}   |Delta| = {r['abs_delta_L3']:.4f}")
        for h in r["halves"]:
            bits = [f"fit {h['fold']}: Delta_fit={h['delta_fit']:+.4f}"]
            if do_c2:
                bits.append(f"dial=L{h['dial_level']} "
                            f"(Dk={h['dkappa_dial']:+.4f})")
                bits.append(f"sign-only=L{h['sign_level']} "
                            f"(Dk={h['dkappa_sign']:+.4f})")
                bits.append(f"oracle=L{h['oracle_level']} "
                            f"(Dk={h['dkappa_oracle']:+.4f})")
            if do_c3:
                bits.append(f"map@L3={h['map_L3']} "
                            f"(Dk={h['dkappa_remap']:+.4f})")
                if do_c2:
                    bits.append(f"combined map={h['map_combined']} "
                                f"(Dk={h['dkappa_combined']:+.4f})")
            print("      " + "\n      ".join(bits))
        if do_c2:
            print(f"    mean held-out: dial {r['dkappa_dial']:+.4f} | "
                  f"sign-only {r['dkappa_sign']:+.4f} | "
                  f"oracle bound {r['dkappa_oracle']:+.4f}")
            rows_c2.append(
                f"{esc(model)} & {esc(scope)} & "
                f"{'/'.join('L%d' % h['dial_level'] for h in r['halves'])} & "
                f"${r['dkappa_dial']:+.4f}$ & ${r['dkappa_sign']:+.4f}$ & "
                f"${r['dkappa_oracle']:+.4f}$ \\\\")
        if do_c3:
            line = f"    mean held-out: remap {r['dkappa_remap']:+.4f}"
            if do_c2:
                line += f" | dial+remap {r['dkappa_combined']:+.4f}"
            print(line)
            print(f"    |Delta| held out: L3 {r['abs_delta_L3']:.4f}"
                  + (f" -> dial {r['abs_delta_ev_dial']:.4f}"
                     if "abs_delta_ev_dial" in r else "")
                  + f" -> remap {r['abs_delta_ev_remap']:.4f}"
                  + (f" -> combined {r['abs_delta_ev_combined']:.4f}"
                     if "abs_delta_ev_combined" in r else ""))
            rows_c3.append(
                f"{esc(model)} & {esc(scope)} & "
                f"{'/'.join(str(h['map_L3']) for h in r['halves'])} & "
                f"${r['dkappa_remap']:+.4f}$ & "
                f"${r.get('dkappa_combined', float('nan')):+.4f}$ & "
                f"${r['abs_delta_L3']:.4f}$ & "
                f"${r.get('abs_delta_ev_combined', r['abs_delta_ev_remap']):.4f}$ "
                f"\\\\")

    if rows_c2:
        write_tex(tex_dir, "batch4_c2_dial.tex", rows_c2,
                  "C2: held-out tone-dial effect (query-disjoint cross-fit). "
                  "Columns: chosen levels (A->B / B->A), Dkappa dial, "
                  "Dkappa sign-only baseline, oracle bound")
    if rows_c3:
        write_tex(tex_dir, "batch4_c3_remap.tex", rows_c3,
                  "C3: held-out monotone output remap and C2+C3 combined, "
                  "with |Delta| before/after (artifact-removal metric)")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 4. Summary + frozen correction rules
# ═══════════════════════════════════════════════════════════════════════════

def run_summary(packs: dict, cf: dict, tex_dir: Path) -> None:
    print()
    print("=" * 78)
    print("4. SUMMARY — held-out kappa after each correction")
    print("=" * 78)
    if not cf:
        print("  (c2 and c3 both skipped — nothing to summarise)")
        return
    hdr = (f"  {'model':<30} {'scope':<8} {'k(L3)':>8} {'+C2':>8} {'+C3':>8} "
           f"{'+C2C3':>8} {'|D|pre':>8} {'|D|post':>8}")
    print(hdr)
    rows = []
    for (model, scope), r in sorted(cf.items()):
        k3 = r["kappa_L3"]

        def k(key):
            return k3 + r[key] if key in r and np.isfinite(r[key]) else np.nan

        k2, k3r, k23 = k("dkappa_dial"), k("dkappa_remap"), k("dkappa_combined")
        dpre = r["abs_delta_L3"]
        dpost = r.get("abs_delta_ev_combined",
                      r.get("abs_delta_ev_remap",
                            r.get("abs_delta_ev_dial", np.nan)))
        print(f"  {model:<30} {scope:<8} {k3:>8.4f} {k2:>8.4f} {k3r:>8.4f} "
              f"{k23:>8.4f} {dpre:>8.4f} {dpost:>8.4f}")
        rows.append(f"{esc(model)} & {esc(scope)} & ${k3:.4f}$ & ${k2:.4f}$ & "
                    f"${k3r:.4f}$ & ${k23:.4f}$ & ${dpre:.4f}$ & "
                    f"${dpost:.4f}$ \\\\")
    write_tex(tex_dir, "batch4_summary.tex", rows,
              "Batch 4 summary: held-out kappa at L3 and after C2 (dial), "
              "C3 (monotone remap) and C2+C3, with |Delta| before/after. "
              "All corrections are query-disjoint cross-fits; gains are "
              "measurement-artifact removal, not judgment quality")


def run_rules(df: pd.DataFrame, qrels: dict, datasets, flagships: set,
              data_dir: Path, cf: dict, seed: int) -> Path:
    """Freeze the full-data, per-(model, family) correction rules for C4."""
    print()
    print("=" * 78)
    print("   FROZEN CORRECTION RULES (per model x dataset family)")
    print("=" * 78)
    print("  msmarco rules are fitted on dl19+dl20+dl21 pooled — these are the "
          "rules C4 will")
    print("  PRE-REGISTER for the sealed dl22 (msmarco-v2). antique rules are "
          "fitted on antique.")

    rules: dict = {}
    for model in sorted(df.model_id.unique()):
        for fam, fam_ds in FAMILIES.items():
            use = [d for d in fam_ds if d in datasets]
            if not use:
                continue
            frozen, _p = flagship_pairs_for(data_dir, use)
            d = scope_frame(df, model, use, frozen, model in flagships,
                            f"{fam}:{','.join(use)}")
            if d.empty:
                continue
            P = pack(d, qrels, model, f"{fam}-all", seed)
            allrows = np.arange(len(P.h))
            est = delta_D(P, allrows, strict=False)
            if est is None:
                print(f"  [warn] {model} @ {fam}: a politeness level is "
                      f"missing on the pooled family data — no rule frozen")
                continue
            delta, D, _m, _q = est
            dial = best_level(delta, D)
            smap, kfit = fit_monotone_map(P, allrows, dial)
            # expected held-out gain from the cross-fit scopes of this family
            exp = [cf[(m, s)]["dkappa_combined"]
                   for (m, s) in cf
                   if m == model and FAMILY_OF_SCOPE.get(s) == fam
                   and np.isfinite(cf[(m, s)].get("dkappa_combined", np.nan))]
            rules.setdefault(model, {})[fam] = {
                "datasets": use,
                "n_pairs": int(P.n_pairs),
                "delta_hat": delta,
                "D_hat": {f"L{l}": D[l] for l in LEVELS},
                "dial_level": int(dial),
                "monotone_map": list(smap),
                "monotone_map_kappa_insample": (float(kfit)
                                                if np.isfinite(kfit) else None),
                "expected_dkappa_from_crossfit": (float(np.mean(exp))
                                                  if exp else None),
            }
            print(f"  {model:<30} {fam:<8} Delta={delta:+.4f}  dial=L{dial}  "
                  f"map={smap}  "
                  f"E[Dk]={'n/a' if not exp else f'{np.mean(exp):+.4f}'}")

    out = {
        "_framing": (
            "Operating-point calibration rules. Applying them removes a "
            "MEASUREMENT ARTIFACT (judge-vs-reference operating-point "
            "misalignment); it does not and cannot improve the judge's "
            "ranking ability. Gains must not be reported as quality gains."
        ),
        "git_hash": git_hash(),
        "timestamp_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "seed": seed,
        "datasets_used": list(datasets),
        "sealed_excluded": ["dl22"],
        "families": {k: list(v) for k, v in FAMILIES.items()},
        "dl22_rule": ("msmarco family rules apply to dl22 (msmarco-v2); "
                      "pre-registered, not fitted on dl22."),
        "rules": rules,
    }
    path = data_dir / "derived" / "correction_rules.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    print(f"  [rules] wrote {path}  ({len(rules)} models)")
    return path


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def build_packs(df: pd.DataFrame, qrels: dict, datasets, flagships: set,
                data_dir: Path, seed: int) -> dict:
    packs: dict = {}
    for model in sorted(df.model_id.unique()):
        for scope, scope_ds in SCOPES.items():
            use = [d for d in scope_ds if d in datasets]
            if not use:
                continue
            frozen, _p = flagship_pairs_for(data_dir, use)
            d = scope_frame(df, model, use, frozen, model in flagships, scope)
            if d.empty:
                continue
            packs[(model, scope)] = pack(d, qrels, model, scope, seed)
    if not packs:
        sys.exit("ERROR: no (model, scope) cells to analyse — the parquet has "
                 "no rows on dl19/dl20/dl21/antique after scoping.")
    return packs


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None, help="DATA_DIR (default: $DATA_DIR)")
    ap.add_argument("--skip", action="append", default=[],
                    choices=["auth", "c1", "c2", "c3"],
                    help="skip a section (repeatable)")
    ap.add_argument("--b", type=int, default=1000,
                    help="C1 resamples per calibration size (default 1000)")
    ap.add_argument("--seed", type=int, default=SEED,
                    help=f"master seed, used for folds and resampling "
                         f"(default {SEED})")
    ap.add_argument("--datasets", default=",".join(ALLOWED_DATASETS),
                    help="datasets to analyse; dl22 is refused (sealed)")
    ap.add_argument("--sizes", default=",".join(str(n) for n in CAL_SIZES),
                    help="C1 calibration-set sizes in labeled pairs")
    ap.add_argument("--tol", type=float, default=KAPPA_TOL,
                    help=f"authority assertion tolerance (default {KAPPA_TOL})")
    args = ap.parse_args(argv)

    skip = set(args.skip)
    datasets = resolve_datasets(args.datasets)
    sizes = [int(x) for x in str(args.sizes).split(",") if x.strip()]
    data_dir = Path(args.data_dir or os.environ.get("DATA_DIR", "./data"))
    tex_dir = data_dir / "derived" / "tex"

    parquet = data_dir / "derived" / "judgments.parquet"
    if not parquet.exists():
        sys.exit(f"ERROR: {parquet} not found. Run src/parse.py first.")
    df = pd.read_parquet(parquet)
    before = len(df)
    df = df[df.dataset.isin(datasets)].copy()
    leaked = sorted(set(df.dataset.astype(str).unique()) - set(datasets))
    if leaked:
        sys.exit(f"ERROR: dataset filter failed, saw {leaked}")
    print(f"[scope] datasets {datasets}: {before} -> {len(df)} rows "
          f"(dl22 hard-excluded: sealed)")
    if df.empty:
        sys.exit("ERROR: no rows on the allowed datasets.")

    qrels = load_qrels(list(datasets))
    flagships = flagship_model_ids(REPO_ROOT / "config" / "models.yaml")

    if "auth" not in skip:
        run_authority(df, data_dir, qrels, flagships, args.tol)
    else:
        print("\n[auth] SKIPPED — downstream numbers are NOT verified against "
              "paper/PREDICTIONS.md.")

    packs = build_packs(df, qrels, datasets, flagships, data_dir, args.seed)
    print(f"\n[cells] {len(packs)} (model, scope) cells: "
          f"{sorted({m for m, _ in packs})}")

    if "c1" not in skip:
        run_c1(packs, args.b, sizes, args.seed, tex_dir)

    cf: dict = {}
    if "c2" not in skip or "c3" not in skip:
        cf = run_c2_c3(packs, "c2" not in skip, "c3" not in skip, tex_dir)
        run_summary(packs, cf, tex_dir)

    run_rules(df, qrels, datasets, flagships, data_dir, cf, args.seed)

    print()
    print("REMINDER: every gain above is operating-point artifact removal "
          "measured against")
    print("THESE reference annotators — never report it as improved judgment "
          "quality.")
    print("done.")


if __name__ == "__main__":
    main()
