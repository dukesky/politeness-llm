#!/usr/bin/env python3
"""Batch 7 (distillation, Application B) — EXPLORATORY post-hoc analyses.

[EXPLORATORY — not pre-registered]  Everything this script prints was asked
for AFTER the B7 unboxing (evaluator review 2026-10-12, sections C/D/F).  None
of it enters the registered B7-P1..P8 adjudication (scripts/batch7_summary.py)
and none of it may be written up as confirmatory.

Inputs are the files batch7_distill.py already wrote — no GPU, no model re-run:
  <eval-root>/<arm>/<judge_slug>/<SET>_s<seed>_dl22.json   (evaluate output;
        "pairs" = [[qid, docid, pred (unclipped), human], ...] for the full
        judged dl22 universe)
  <labels-root>/<judge_slug>/manifest.json                  (build-labels)
  <labels-root>/<judge_slug>/labels_<SET>.jsonl             (optional, item 5)
  --b6-labels batch6_labels_dl22.parquet                    (optional, item 4)

Conventions are batch7_distill.py's (its functions are imported, not copied):
binary relevance = round_half_up(clip(pred, 0, 3)) >= 2 vs human >= 2; NDCG@10
over the judged pool, gain 2^rel - 1, ideal from human grades, ties by docid;
MAP at human >= 2; query-clustered bootstrap B=1000 seed 42, 90% percentile
CI; paired compares on the same resampled queries; a side = mean over its
seeds.  Before anything else every eval file's stored point endpoints (and
CIs, when its bootstrap B/seed equal this run's) are recomputed from its
"pairs" and must match to 1e-9 — a cheap integrity check.

Analyses
--------
1. POST-HOC SHIFT-CALIBRATED RAW STUDENT.  pred' = pred_RAW - shift, with
     SHIFT_n<n>   = delta_hat of the registered CORR_REFIT_n<n> rule (the judge
                    offset estimated on the SAME n human-labelled training
                    pairs that MIX_<n> / CORR_REFIT_n<n> used; record-level
                    L3 score mean - human mean, batch-4 delta_D convention)
     SHIFT_nL<n>  = RAW label mean - HUMAN label mean on those same MIX_<n>
                    human pairs (label-level variant; needs labels_*.jsonl)
     SHIFT_FULL   = RAW label mean - HUMAN label mean over the full training
                    universe (manifest summary)
     ORACLE_SHIFT = the shift that sets the student's OWN dl22 offset (the
                    registered, clipped one) to 0 (IN-SAMPLE oracle,
                    reference only — uses dl22 human labels)
   Shifts use the TEACHER's label offset, not the student's own (except the
   oracle).  NDCG/MAP are invariant to a constant shift (asserted to 1e-9).
2. THRESHOLD-FREE / MATCHED-PREVALENCE endpoints: pooled AUC and PR-AUC
   (average precision) of pred vs human >= 2, per-query-mean AUC (queries with
   both classes), F1@matched-ratio (top-P by score, P = # human positives;
   IN-SAMPLE by construction; ties at the cut taken fractionally = expected
   value under random tie-breaking), prediction sd (unclipped), fraction of
   predictions with rounded grade >= 2 (pred >= 1.5) and >= 1 (pred >= 0.5);
   universe-level: human >= 2 base rate b, trivial all-relevant F1 = 2b/(1+b),
   random-ranking NDCG@10 floor (200 permutations / query, seed 42).
3. KENDALL tau-b between students' scores (same arm, judge, seed), per query
   (mean over queries) and pooled; seed-to-seed tau within RAW and HUMAN as a
   noise reference.  Per-query NDCG@10 paired deltas (FROZEN-RAW, HUMAN-RAW).
4. TEACHER REFERENCE (--b6-labels): the judge's own RAW / FROZEN / REFIT dl22
   labels scored with the same endpoint code on the intersected universe.
5. LABEL-SET DIAGNOSTICS (labels_<SET>.jsonl): grade histogram, Pearson /
   Spearman vs HUMAN on the training universe, grade-1 share, RAW grade-1
   pairs that the set labels 0.

usage:
  python scripts/batch7_posthoc.py run --eval-root $DATA_DIR/derived/batch7/eval \
      --labels-root $DATA_DIR/derived/batch7 \
      [--b6-labels $DATA_DIR/derived/batch6_labels_dl22.parquet] \
      --out $DATA_DIR/derived/batch7/posthoc
  python scripts/batch7_posthoc.py selftest [--keep-dir DIR]

Outputs (--out): posthoc_report.txt (= stdout), posthoc.json,
posthoc_sets.csv (long: arm, judge, set, endpoint, 3-seed mean, seed sd,
90% CI of the seed mean), posthoc_compares.csv (paired A - B with 90% CI).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch7_distill as b7                                       # noqa: E402
from batch7_distill import (BOOT_B, BOOT_SEED, ENDPOINTS, NDCG_K,  # noqa: E402
                            REL_THRESHOLD, STAT_COLS, aggregate, boot_weights,
                            ci90, evaluate_pairs, judge_slug, per_query_stats,
                            universe_sha)

TAG = "[EXPLORATORY — not pre-registered]"
SETS = ("HUMAN", "RAW", "CORR_FROZEN", "CORR_REFIT", "MIX_200", "MIX_1000",
        "CORR_REFIT_n200", "CORR_REFIT_n1000")
MIX_NS = (200, 1000)
SHIFT_SETS = ("SHIFT_n200", "SHIFT_n1000", "SHIFT_nL200", "SHIFT_nL1000",
              "SHIFT_FULL", "ORACLE_SHIFT")
TF_EPS = ("auc", "auc_q", "pr_auc", "f1_matched", "pred_sd", "frac_ge15",
          "frac_ge05")
UNIV_EPS = ("base_rate", "f1_all_relevant")
ALL_EPS = tuple(ENDPOINTS) + TF_EPS
CMP_EPS = ("ndcg10", "map", "offset", "abs_offset", "mae", "f1", "precision",
           "recall", "rel_rate_ratio", "abs_ratio_dev", "ece4", "auc",
           "pr_auc", "f1_matched", "pred_sd", "frac_ge15")
TAU_PAIRS = (("RAW", "HUMAN"), ("RAW", "CORR_FROZEN"), ("RAW", "CORR_REFIT"),
             ("HUMAN", "CORR_FROZEN"), ("MIX_1000", "RAW"),
             ("CORR_REFIT_n1000", "MIX_1000"))
DELTA_PAIRS = (("CORR_FROZEN", "RAW"), ("HUMAN", "RAW"))
TEACHER_SETS = (("RAW", "RAW"), ("FROZEN", "CORR_FROZEN"),
                ("REFIT", "CORR_REFIT"))
TOL = 1e-9
N_RANDOM_PERM = 200
RANDOM_SEED = 42
JUDGE_SHORT = {"deepseek__deepseek-v4-flash": "v4-flash",
               "google__gemini-3.5-flash": "3.5-flash",
               "google__gemini-3.8-flash": "3.8-flash"}


def fail(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


def _div(a, b):
    return b7._div(a, b)


def _f(x, w: int = 8, p: int = 4, sign: bool = True) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return f"{'n/a':>{w}}"
    return f"{x:>+{w}.{p}f}" if sign else f"{x:>{w}.{p}f}"


def _ci(c) -> str:
    if c is None or not all(math.isfinite(v) for v in c):
        return "[   n/a ]"
    return f"[{c[0]:+.4f}, {c[1]:+.4f}]"


def _clean(o):
    """JSON-safe: NaN/inf -> None, numpy -> python."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def build_W(n_q: int, b: int, seed: int) -> np.ndarray:
    """Row 0 = point estimate; rows 1.. = batch7's boot_weights (identical to
    evaluate / compare)."""
    return np.vstack([np.ones((1, n_q)), boot_weights(n_q, b, seed)])


# ═══════════════════════════════════════════════════════════════════════════
# Universe + vectorised pair-level endpoints
# ═══════════════════════════════════════════════════════════════════════════

class Universe:
    """Judged pair universe in (qid, docid) order; qids sorted like
    batch7.per_query_stats so W columns line up with its S rows."""

    def __init__(self, pairs):
        rows = sorted((str(q), str(d), int(h)) for q, d, _p, h in pairs)
        self.keys = [(q, d) for q, d, _h in rows]
        if len(set(self.keys)) != len(self.keys):
            fail("duplicate (qid, docid) in a pair universe.")
        self.pos = {k: i for i, k in enumerate(self.keys)}
        self.h = np.array([h for _q, _d, h in rows], dtype=np.int64)
        self.rel = self.h >= REL_THRESHOLD
        self.qids = sorted({q for q, _d in self.keys})
        qi = {q: i for i, q in enumerate(self.qids)}
        self.qidx = np.array([qi[q] for q, _d in self.keys], dtype=np.int64)
        self.n, self.n_q = len(self.keys), len(self.qids)
        self.q_members = [np.flatnonzero(self.qidx == i)
                          for i in range(self.n_q)]
        self._iu = None

    def align(self, pairs, what: str) -> np.ndarray:
        pred = np.full(self.n, np.nan)
        if len(pairs) != self.n:
            fail(f"{what}: {len(pairs)} pairs vs universe {self.n}.")
        for q, d, p, h in pairs:
            i = self.pos.get((str(q), str(d)))
            if i is None:
                fail(f"{what}: pair ({q}, {d}) not in the universe.")
            if int(h) != int(self.h[i]):
                fail(f"{what}: human grade of ({q}, {d}) differs across files.")
            pred[i] = float(p)
        if not np.all(np.isfinite(pred)):
            fail(f"{what}: missing / non-finite predictions.")
        return pred

    def pairs(self, pred) -> list:
        return [(q, d, float(p), int(h))
                for (q, d), p, h in zip(self.keys, pred, self.h)]

    def triu(self):
        if self._iu is None:
            self._iu = np.triu_indices(self.n, 1)
        return self._iu


def _groups(sorted_scores: np.ndarray) -> np.ndarray:
    return np.r_[0, np.flatnonzero(np.diff(sorted_scores) != 0) + 1]


def w_auc(score, pos, Wp) -> np.ndarray:
    """Weighted ROC-AUC per row of Wp (ties count 1/2)."""
    order = np.argsort(score, kind="mergesort")
    st = _groups(score[order])
    Wo, po = Wp[:, order], pos[order].astype(float)
    pw = np.add.reduceat(Wo * po, st, axis=1)
    nw = np.add.reduceat(Wo * (1.0 - po), st, axis=1)
    below = np.cumsum(nw, axis=1) - nw
    num = (pw * (below + 0.5 * nw)).sum(1)
    return _div(num, pw.sum(1) * nw.sum(1))


def _desc_groups(score, pos, Wp):
    order = np.argsort(-score, kind="mergesort")
    st = _groups(score[order])
    Wo, po = Wp[:, order], pos[order].astype(float)
    pw = np.add.reduceat(Wo * po, st, axis=1)
    nw = np.add.reduceat(Wo * (1.0 - po), st, axis=1)
    return pw, nw


def w_ap(score, pos, Wp) -> np.ndarray:
    """Weighted average precision (sklearn definition: sum dRecall * P)."""
    pw, nw = _desc_groups(score, pos, Wp)
    tp, fp = np.cumsum(pw, 1), np.cumsum(nw, 1)
    prec = np.nan_to_num(_div(tp, tp + fp), nan=0.0)
    return _div((pw * prec).sum(1), pw.sum(1))


def w_f1_matched(score, pos, Wp) -> np.ndarray:
    """F1 when the top-P (weighted) pairs are called relevant, P = weighted #
    human positives; ties at the cut taken fractionally (expected value under
    random tie-breaking).  Then precision = recall = F1 = TP / P."""
    pw, nw = _desc_groups(score, pos, Wp)
    gw = pw + nw
    K = pw.sum(1)
    take = np.clip(K[:, None] - (np.cumsum(gw, 1) - gw), 0.0, gw)
    frac = np.nan_to_num(_div(take, gw), nan=0.0)
    return _div((frac * pw).sum(1), K)


def per_query_auc(U: Universe, score) -> tuple[np.ndarray, np.ndarray]:
    auc = np.zeros(U.n_q)
    ok = np.zeros(U.n_q)
    for i, idx in enumerate(U.q_members):
        r = U.rel[idx]
        if r.all() or (~r).all():
            continue
        sp, sn = score[idx][r], score[idx][~r]
        auc[i] = ((sp[:, None] > sn[None, :]).mean()
                  + 0.5 * (sp[:, None] == sn[None, :]).mean())
        ok[i] = 1.0
    return auc, ok


def pair_level(U: Universe, pred, W) -> dict:
    Wp = W[:, U.qidx]
    tot = Wp.sum(1)
    out = {"auc": w_auc(pred, U.rel, Wp), "pr_auc": w_ap(pred, U.rel, Wp),
           "f1_matched": w_f1_matched(pred, U.rel, Wp)}
    aq, okq = per_query_auc(U, pred)
    out["auc_q"] = _div(W @ (aq * okq), W @ okq)
    pc = pred - pred.mean()
    m1 = (Wp @ pc) / tot
    out["pred_sd"] = np.sqrt(np.maximum((Wp @ (pc * pc)) / tot - m1 * m1, 0))
    rb = np.floor(np.clip(pred, 0.0, 3.0) + 0.5)      # = round_half_up grade
    out["frac_ge15"] = (Wp @ (rb >= 2).astype(float)) / tot
    out["frac_ge05"] = (Wp @ (rb >= 1).astype(float)) / tot
    return out


def compute_unit(U: Universe, pred, W, what: str) -> dict:
    qids, S = per_query_stats(U.pairs(pred))
    if qids != U.qids:
        fail(f"{what}: query order mismatch with per_query_stats.")
    agg = aggregate(W, S)
    agg.update(pair_level(U, pred, W))
    # consistency: rounded-positive share must equal (tp + fp) / n
    c = {n: i for i, n in enumerate(STAT_COLS)}
    T = S.sum(0)
    if abs(agg["frac_ge15"][0] - (T[c["tp"]] + T[c["fp"]]) / T[c["n"]]) > TOL:
        fail(f"{what}: frac_ge15 disagrees with batch7's rounded positives.")
    return {"S": S, "pred": pred, "agg": agg}


def universe_level(U: Universe, W) -> dict:
    Wp = W[:, U.qidx]
    b = (Wp @ U.rel.astype(float)) / Wp.sum(1)
    out = {"base_rate": b, "f1_all_relevant": _div(2 * b, 1 + b)}
    rng = np.random.default_rng(RANDOM_SEED)
    disc = 1.0 / np.log2(np.arange(2, NDCG_K + 2))
    rnd = np.zeros(U.n_q)
    ana = np.zeros(U.n_q)
    ok = np.zeros(U.n_q)
    for i, idx in enumerate(U.q_members):
        h = U.h[idx]
        idcg = b7._dcg(sorted(h.tolist(), reverse=True)[:NDCG_K])
        if idcg <= 0:
            continue
        perms = np.argsort(rng.random((N_RANDOM_PERM, h.size)), axis=1)
        g = (2.0 ** h[perms][:, :NDCG_K]) - 1.0
        rnd[i] = (g * disc[:g.shape[1]]).sum(1).mean() / idcg
        k = min(NDCG_K, h.size)
        ana[i] = ((2.0 ** h - 1.0).mean() * disc[:k].sum()) / idcg
        ok[i] = 1.0
    out["ndcg10_random"] = _div(W @ (rnd * ok), W @ ok)
    out["ndcg10_random_analytic"] = _div(W @ (ana * ok), W @ ok)
    return out


def oracle_shift(pred, h) -> float:
    """c with mean(clip(pred - c, 0, 3)) == mean(h) (the registered, clipped
    offset = 0); bisection — the clipped mean is monotone in c."""
    target = float(np.mean(h))
    lo, hi = float(pred.min()) - 3.0, float(pred.max())
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if np.clip(pred - mid, 0.0, 3.0).mean() > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def tau_b(x, y, iu=None) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    i, j = iu if iu is not None else np.triu_indices(n, 1)
    sx = np.sign(x[i] - x[j])
    sy = np.sign(y[i] - y[j])
    den = math.sqrt(float(np.count_nonzero(sx)) * float(np.count_nonzero(sy)))
    return float((sx * sy).sum() / den) if den > 0 else float("nan")


def tau_stats(U: Universe, x, y) -> dict:
    per_q = [tau_b(x[idx], y[idx]) for idx in U.q_members]
    per_q = np.array(per_q, dtype=float)
    return {"tau_query_mean": float(np.nanmean(per_q)) if np.isfinite(per_q).any()
            else float("nan"),
            "n_queries": int(np.isfinite(per_q).sum()),
            "tau_pooled": tau_b(x, y, U.triu())}


# ═══════════════════════════════════════════════════════════════════════════
# Loading + integrity
# ═══════════════════════════════════════════════════════════════════════════

def load_evals(root: Path) -> dict:
    """{arm: {slug: {SET: {seed: result}}}}"""
    out: dict = {}
    for p in sorted(Path(root).glob("*/*/*_dl22.json")):
        arm, slug = p.parent.parent.name, p.parent.name
        stem = p.name[: -len("_dl22.json")]
        set_name, _, seed = stem.rpartition("_s")
        if not set_name or not seed.isdigit():
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        r["_path"] = str(p)
        out.setdefault(arm, {}).setdefault(slug, {}).setdefault(
            set_name, {})[int(seed)] = r
    if not out:
        fail(f"no <arm>/<judge_slug>/<SET>_s<seed>_dl22.json under {root}.")
    return out


def _close(a, b) -> bool:
    a, b = float(a), float(b)
    if not math.isfinite(a) or not math.isfinite(b):
        return (math.isnan(a) and math.isnan(b)) or a == b
    return abs(a - b) <= TOL


def integrity_check(r: dict, unit: dict, U: Universe, arm: str, slug: str,
                    set_name: str, seed: int, b: int, bseed: int) -> dict:
    what = Path(r["_path"]).name
    bad = []
    for fld, want in (("arm", arm), ("label_set", set_name), ("seed", seed)):
        if r.get(fld) is not None and str(r.get(fld)) != str(want):
            bad.append(f"{fld}={r.get(fld)!r} but path says {want!r}")
    if r.get("judge") and judge_slug(str(r["judge"])) != slug:
        bad.append(f"judge={r['judge']!r} but path slug {slug!r}")
    if r.get("universe_sha256") and \
            r["universe_sha256"] != universe_sha(r["pairs"]):
        bad.append("universe_sha256 does not match its own pairs")
    for k, v in r.get("point", {}).items():
        if k in unit["agg"] and not _close(unit["agg"][k][0], v):
            bad.append(f"point {k}: stored {v} vs recomputed "
                       f"{unit['agg'][k][0]}")
    for k, want in (("n_pairs", U.n), ("n_queries", U.n_q)):
        if r.get(k) is not None and int(r[k]) != want:
            bad.append(f"{k} {r[k]} vs {want}")
    S = unit["S"]
    for k, col in (("n_queries_ndcg", "ndcg_ok"), ("n_queries_map", "ap_ok")):
        if r.get(k) is not None and \
                int(r[k]) != int(S[:, STAT_COLS.index(col)].sum()):
            bad.append(f"{k} mismatch")
    bs = r.get("bootstrap") or {}
    ci_checked = (int(bs.get("B", -1)) == b and int(bs.get("seed", -1)) == bseed)
    if ci_checked:
        for k in ENDPOINTS:
            if k in r.get("ci90", {}):
                mine = ci90(unit["agg"][k][1:])
                if not all(_close(x, y) for x, y in zip(mine, r["ci90"][k])):
                    bad.append(f"ci90 {k}: stored {r['ci90'][k]} vs {mine}")
    if bad:
        fail(f"integrity check failed for {what}:\n  " + "\n  ".join(bad))
    return {"ci_checked": ci_checked}


def check_shift_invariance(S_raw: np.ndarray, S_shift: np.ndarray,
                           what: str) -> None:
    for col in ("ndcg", "ndcg_ok", "ap", "ap_ok"):
        i = STAT_COLS.index(col)
        d = np.max(np.abs(S_raw[:, i] - S_shift[:, i])) if S_raw.size else 0
        if not d <= TOL:
            fail(f"shift invariance VIOLATED for {what}: per-query {col} "
                 f"differs from RAW by {d:.3g} (> {TOL}). A constant shift "
                 f"must not change the ranking.")


# ═══════════════════════════════════════════════════════════════════════════
# Manifest / label files
# ═══════════════════════════════════════════════════════════════════════════

def read_label_file(p: Path) -> dict:
    out = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[(str(r["dataset"]), str(r["qid"]), str(r["docid"]))] = \
                    float(r["label"])
    return out


def load_label_files(labels_root: Path, slug: str) -> dict:
    d = Path(labels_root) / slug
    out = {}
    for s in SETS:
        p = d / f"labels_{s}.jsonl"
        if p.exists():
            out[s] = read_label_file(p)
    return out


def shifts_for(manifest: dict | None, labels: dict) -> tuple[dict, list]:
    """{name: {"value", "source"}} for the manifest/label-based shifts."""
    out, notes = {}, []
    if manifest is None:
        notes.append("manifest.json absent: SHIFT_n / SHIFT_FULL skipped")
        return out, notes
    summ = manifest.get("summary", {})
    if "RAW" in summ and "HUMAN" in summ:
        out["SHIFT_FULL"] = {
            "value": float(summ["RAW"]["mean"]) - float(summ["HUMAN"]["mean"]),
            "source": "manifest summary: RAW label mean - HUMAN label mean "
                      "(full training universe)"}
    else:
        notes.append("manifest summary lacks RAW/HUMAN: SHIFT_FULL skipped")
    eq = manifest.get("equal_budget_rules", {})
    mix = manifest.get("mix", {})
    for n in MIX_NS:
        r = eq.get(f"CORR_REFIT_n{n}")
        if r is None or r.get("delta_hat") is None:
            notes.append(f"CORR_REFIT_n{n} rule absent: SHIFT_n{n} skipped")
        else:
            out[f"SHIFT_n{n}"] = {
                "value": float(r["delta_hat"]),
                "source": f"CORR_REFIT_n{n} rule delta_hat (n_fit_rows "
                          f"{r.get('n_fit_rows')}, {len(r.get('fit_queries', []))}"
                          f" queries)"}
        mq = (mix.get(f"MIX_{n}") or {}).get("human_queries")
        if mq and "RAW" in labels and "HUMAN" in labels:
            qs = set(mq)
            ks = [k for k in labels["HUMAN"] if f"{k[0]}|{k[1]}" in qs]
            if ks and all(k in labels["RAW"] for k in ks):
                raw = np.array([labels["RAW"][k] for k in ks])
                hum = np.array([labels["HUMAN"][k] for k in ks])
                out[f"SHIFT_nL{n}"] = {
                    "value": float(raw.mean() - hum.mean()),
                    "source": f"RAW - HUMAN label mean on the {len(ks)} MIX_{n}"
                              f" human pairs ({len(qs)} queries)"}
        elif mq:
            notes.append(f"labels_RAW/HUMAN.jsonl absent: SHIFT_nL{n} skipped")
    return out, notes


def label_diagnostics(labels: dict, manifest: dict | None) -> dict:
    out = {"sets": {}}
    if manifest:
        out["frozen_rule"] = manifest.get("frozen_rule")
    if "HUMAN" not in labels:
        out["note"] = ("labels_*.jsonl absent — per-pair diagnostics skipped; "
                       "manifest summary only")
        for s, v in ((manifest or {}).get("summary") or {}).items():
            out["sets"][s] = {"mean": v.get("mean"),
                              "offset_vs_human": v.get("offset_vs_human"),
                              "hist": v.get("hist"), "from": "manifest"}
        return out
    keys = sorted(labels["HUMAN"])
    h = np.array([labels["HUMAN"][k] for k in keys])
    raw = (np.array([labels["RAW"][k] for k in keys]) if "RAW" in labels
           else None)
    for s, lab in labels.items():
        if any(k not in lab for k in keys):
            out["sets"][s] = {"note": "key set differs from HUMAN — skipped"}
            continue
        y = np.array([lab[k] for k in keys])
        g = np.floor(y + 0.5)
        d = {"n": int(y.size), "mean": float(y.mean()),
             "offset_vs_human": float(y.mean() - h.mean()),
             "hist": {str(i): int((g == i).sum()) for i in range(4)},
             "share_grade1": float((g == 1).mean()),
             "pearson_vs_human": (float(np.corrcoef(y, h)[0, 1])
                                  if y.std() > 0 and h.std() > 0 else None),
             "spearman_vs_human": (float(pd.Series(y).corr(pd.Series(h),
                                                           method="spearman"))
                                   if y.std() > 0 and h.std() > 0 else None),
             "from": "labels file"}
        if raw is not None:
            gr = np.floor(raw + 0.5)
            m1 = gr == 1
            d["raw1_to_0_share"] = (float((g[m1] == 0).mean()) if m1.any()
                                    else None)
            d["crosstab_raw_rows_set_cols"] = [
                [int(((gr == a) & (g == c)).sum()) for c in range(4)]
                for a in range(4)]
        out["sets"][s] = d
    return out


def load_b6(path: Path | None) -> tuple[pd.DataFrame | None, str]:
    if path is None:
        return None, "--b6-labels not given"
    p = Path(path)
    cands = [p] + ([p.with_suffix(".pkl")] if p.suffix == ".parquet" else [])
    for c in cands:
        if c.exists():
            try:
                df = (pd.read_pickle(c) if c.suffix == ".pkl"
                      else pd.read_parquet(c))
            except Exception as e:                           # noqa: BLE001
                return None, f"could not read {c}: {e}"
            need = {"model_id", "qid", "docid", "human", "RAW", "FROZEN",
                    "REFIT"}
            miss = need - set(df.columns)
            if miss:
                return None, (f"{c} lacks columns {sorted(miss)}; has "
                              f"{list(df.columns)} — teacher reference skipped")
            if "dataset" in df.columns:
                df = df[df.dataset.astype(str) == "dl22"]
            return df, f"read {c} ({len(df)} dl22 rows)"
    return None, f"{p} not found — teacher reference skipped"


# ═══════════════════════════════════════════════════════════════════════════
# Aggregation helpers
# ═══════════════════════════════════════════════════════════════════════════

def side_mean(units: list, k: str) -> np.ndarray:
    return np.mean([u["agg"][k] for u in units], axis=0)


def summarise_set(units: list, eps=ALL_EPS) -> dict:
    out = {}
    for k in eps:
        vals = [float(u["agg"][k][0]) for u in units]
        m = side_mean(units, k)
        out[k] = {"mean": float(m[0]), "ci90": ci90(m[1:]),
                  "sd": (float(np.std(vals, ddof=1)) if len(vals) > 1
                         else float("nan")),
                  "per_seed": vals}
    return out


def paired(units_a: list, units_b: list, eps=CMP_EPS) -> dict:
    """Same arithmetic as batch7.compare_results (side = mean over its seeds
    on the same W rows; diff CI = 5/95 percentiles of the bootstrap rows)."""
    out = {}
    for k in eps:
        a, b = side_mean(units_a, k), side_mean(units_b, k)
        d = a - b
        c = ci90(d[1:])
        out[k] = {"a": float(a[0]), "b": float(b[0]), "diff": float(d[0]),
                  "ci90": c,
                  "excludes_0": bool(all(math.isfinite(x) for x in c)
                                     and (c[0] > 0 or c[1] < 0))}
    return out


def compare_specs() -> list:
    s = []
    for n in MIX_NS:
        s += [(f"SHIFT_n{n}", f"CORR_REFIT_n{n}", "equal-budget shift vs "
               "equal-budget correction"),
              (f"SHIFT_n{n}", f"MIX_{n}", "equal-budget shift vs MIX"),
              (f"SHIFT_nL{n}", f"CORR_REFIT_n{n}", "label-level equal-budget "
               "shift vs correction"),
              (f"SHIFT_nL{n}", f"MIX_{n}", "label-level equal-budget shift "
               "vs MIX")]
    s += [("SHIFT_FULL", "CORR_FROZEN", "full-budget shift vs frozen rule"),
          ("SHIFT_FULL", "CORR_REFIT", "full-budget shift vs cross-fit rule"),
          ("SHIFT_FULL", "HUMAN", "full-budget shift vs human-label student"),
          ("SHIFT_FULL", "RAW", "shift effect itself (NDCG/MAP = 0 exactly)"),
          ("ORACLE_SHIFT", "HUMAN", "ORACLE (in-sample dl22 shift) vs HUMAN"),
          ("ORACLE_SHIFT", "CORR_FROZEN", "ORACLE (in-sample) vs frozen"),
          ("CORR_FROZEN", "RAW", "threshold-free view of P3/P4"),
          ("CORR_REFIT", "RAW", "threshold-free view"),
          ("HUMAN", "RAW", "threshold-free view of P7"),
          ("CORR_FROZEN", "HUMAN", "threshold-free view")]
    return s


# ═══════════════════════════════════════════════════════════════════════════
# run
# ═══════════════════════════════════════════════════════════════════════════

def analyse(eval_root: Path, labels_root: Path, b6_path: Path | None,
            b: int = BOOT_B, bseed: int = BOOT_SEED) -> dict:
    t0 = time.time()
    ev = load_evals(eval_root)
    first = next(iter(next(iter(next(iter(ev.values())).values())).values()))
    r0 = next(iter(first.values()))
    U = Universe(r0["pairs"])
    W = build_W(U.n_q, b, bseed)

    units: dict = {}               # (arm, slug, set, seed) -> unit
    n_files, n_ci = 0, 0
    for arm, judges in ev.items():
        for slug, sets in judges.items():
            for s, seeds in sets.items():
                for seed, r in seeds.items():
                    pred = U.align(r["pairs"], r["_path"])
                    u = compute_unit(U, pred, W, r["_path"])
                    chk = integrity_check(r, u, U, arm, slug, s, seed, b,
                                          bseed)
                    n_files += 1
                    n_ci += int(chk["ci_checked"])
                    u["train_label_mean"] = r.get("train_label_mean")
                    units[(arm, slug, s, seed)] = u
    integ = {"files": n_files, "ci_checked": n_ci,
             "n_pairs": U.n, "n_queries": U.n_q,
             "n_queries_ndcg": int(next(iter(units.values()))["S"][
                 :, STAT_COLS.index("ndcg_ok")].sum()),
             "n_queries_map": int(next(iter(units.values()))["S"][
                 :, STAT_COLS.index("ap_ok")].sum()),
             "universe_sha256": universe_sha(r0["pairs"])}

    univ = universe_level(U, W)
    univ_summary = {k: {"point": float(v[0]), "ci90": ci90(v[1:])}
                    for k, v in univ.items()}

    # ---- per judge: manifest, labels, shifts ------------------------------
    slugs = sorted({k[1] for k in units})
    manifests, labels, shifts, notes, ldiag = {}, {}, {}, defaultdict(list), {}
    for slug in slugs:
        mp = Path(labels_root) / slug / "manifest.json"
        manifests[slug] = (json.loads(mp.read_text(encoding="utf-8"))
                           if mp.exists() else None)
        labels[slug] = load_label_files(labels_root, slug)
        shifts[slug], nn = shifts_for(manifests[slug], labels[slug])
        notes[slug] += nn
        ldiag[slug] = label_diagnostics(labels[slug], manifests[slug])
        m = manifests[slug]
        if m:
            for (arm, sl, s, seed), u in units.items():
                if sl != slug or u.get("train_label_mean") is None:
                    continue
                lm = (m.get("summary", {}).get(s) or {}).get("mean")
                if lm is not None and abs(float(lm) -
                                          float(u["train_label_mean"])) > 1e-6:
                    notes[slug].append(
                        f"{arm}/{s}_s{seed}: train_label_mean "
                        f"{u['train_label_mean']} != manifest mean {lm}")

    # ---- item 1: shifted RAW students -------------------------------------
    shift_vals: dict = {}
    for (arm, slug, s, seed), u in list(units.items()):
        if s != "RAW":
            continue
        cand = dict(shifts[slug])
        cand["ORACLE_SHIFT"] = {
            "value": oracle_shift(u["pred"], U.h),
            "source": "ORACLE / IN-SAMPLE: shift that sets the student's own "
                      "dl22 (clipped) offset to 0 (uses dl22 human labels)"}
        for name, sv in cand.items():
            pred = u["pred"] - sv["value"]
            what = f"{arm}/{slug}/{name}_s{seed}"
            su = compute_unit(U, pred, W, what)
            check_shift_invariance(u["S"], su["S"], what)
            for k in ("ndcg10", "map"):
                if np.nanmax(np.abs(su["agg"][k] - u["agg"][k])) > TOL:
                    fail(f"shift invariance VIOLATED for {what}: {k}.")
            dd = (su["agg"]["offset_unclipped"][0]
                  - (u["agg"]["offset_unclipped"][0] - sv["value"]))
            if abs(dd) > TOL:
                fail(f"{what}: unclipped offset did not move by the shift.")
            units[(arm, slug, name, seed)] = su
            shift_vals.setdefault(arm, {}).setdefault(slug, {}).setdefault(
                name, {"source": sv["source"], "per_seed": {}})[
                "per_seed"][seed] = sv["value"]

    # ---- set summaries -----------------------------------------------------
    groups = defaultdict(list)
    for (arm, slug, s, seed) in sorted(units):
        groups[(arm, slug, s)].append(units[(arm, slug, s, seed)])
    seeds_of = defaultdict(list)
    for (arm, slug, s, seed) in sorted(units):
        seeds_of[(arm, slug, s)].append(seed)
    set_res: dict = {}
    for (arm, slug, s), us in groups.items():
        set_res.setdefault(arm, {}).setdefault(slug, {})[s] = {
            "n_seeds": len(us), "seeds": seeds_of[(arm, slug, s)],
            "kind": ("oracle (in-sample)" if s == "ORACLE_SHIFT" else
                     "post-hoc shift" if s.startswith("SHIFT") else
                     "registered label set"),
            "endpoints": summarise_set(us)}

    # ---- paired compares ---------------------------------------------------
    compares = []
    for arm in sorted(ev):
        for slug in sorted(ev[arm]):
            for a, bb, why in compare_specs():
                ua, ub = groups.get((arm, slug, a)), groups.get((arm, slug, bb))
                if not ua or not ub:
                    continue
                compares.append({"arm": arm, "judge": slug, "a": a, "b": bb,
                                 "why": why,
                                 "seeds_a": seeds_of[(arm, slug, a)],
                                 "seeds_b": seeds_of[(arm, slug, bb)],
                                 "endpoints": paired(ua, ub)})

    # ---- item 3: Kendall tau-b + per-query deltas ---------------------------
    taus, deltas = [], []
    ndi, oki = STAT_COLS.index("ndcg"), STAT_COLS.index("ndcg_ok")
    for arm in sorted(ev):
        for slug in sorted(ev[arm]):
            for a, bb in TAU_PAIRS:
                per = []
                for seed in sorted(set(seeds_of[(arm, slug, a)])
                                   & set(seeds_of[(arm, slug, bb)])):
                    t = tau_stats(U, units[(arm, slug, a, seed)]["pred"],
                                  units[(arm, slug, bb, seed)]["pred"])
                    t["seed"] = seed
                    per.append(t)
                if per:
                    taus.append({"arm": arm, "judge": slug, "a": a, "b": bb,
                                 "kind": "between sets, same seed",
                                 "tau_query_mean": float(np.mean(
                                     [p["tau_query_mean"] for p in per])),
                                 "tau_pooled": float(np.mean(
                                     [p["tau_pooled"] for p in per])),
                                 "per_seed": per})
            for s in ("RAW", "HUMAN"):
                sd = sorted(seeds_of.get((arm, slug, s), []))
                per = []
                for i in range(len(sd)):
                    for j in range(i + 1, len(sd)):
                        t = tau_stats(U, units[(arm, slug, s, sd[i])]["pred"],
                                      units[(arm, slug, s, sd[j])]["pred"])
                        t["seeds"] = [sd[i], sd[j]]
                        per.append(t)
                if per:
                    taus.append({"arm": arm, "judge": slug, "a": s, "b": s,
                                 "kind": "seed-to-seed (noise reference)",
                                 "tau_query_mean": float(np.mean(
                                     [p["tau_query_mean"] for p in per])),
                                 "tau_pooled": float(np.mean(
                                     [p["tau_pooled"] for p in per])),
                                 "per_seed": per})
            for a, bb in DELTA_PAIRS:
                ua, ub = groups.get((arm, slug, a)), groups.get((arm, slug, bb))
                if not ua or not ub:
                    continue
                ok = ua[0]["S"][:, oki] > 0
                na = np.mean([u["S"][:, ndi] for u in ua], axis=0)[ok]
                nb = np.mean([u["S"][:, ndi] for u in ub], axis=0)[ok]
                d = na - nb
                deltas.append({
                    "arm": arm, "judge": slug, "a": a, "b": bb,
                    "n_queries": int(ok.sum()),
                    "share_neg": float((d < -1e-12).mean()),
                    "share_zero": float((np.abs(d) <= 1e-12).mean()),
                    "share_pos": float((d > 1e-12).mean()),
                    "mean_delta": float(d.mean()),
                    "mean_abs_delta": float(np.abs(d).mean()),
                    "basis": "per-query NDCG@10 averaged over seeds per side"})

    # ---- item 4: teacher reference -----------------------------------------
    teacher = {"status": None, "judges": {}}
    b6, msg = load_b6(b6_path)
    teacher["status"] = msg
    if b6 is not None:
        for slug in slugs:
            sub = b6[[judge_slug(str(m)) == slug for m in b6.model_id]]
            if sub.empty:
                teacher["judges"][slug] = {"note": "judge absent from B6 file"}
                continue
            tk = {(str(q), str(d)): i for i, (q, d) in
                  enumerate(zip(sub.qid, sub.docid))}
            inter = [k for k in U.keys if k in tk]
            if not inter:
                teacher["judges"][slug] = {"note": "no overlap with dl22 "
                                           "student universe"}
                continue
            hb6 = sub.human.to_numpy()
            idx_u = np.array([U.pos[k] for k in inter])
            idx_t = np.array([tk[k] for k in inter])
            mism = int((hb6[idx_t].astype(int) != U.h[idx_u]).sum())
            U2 = Universe([(q, d, 0.0, int(U.h[U.pos[(q, d)]]))
                           for q, d in inter])
            W2 = W if U2.n == U.n else build_W(U2.n_q, b, bseed)
            o2 = np.array([U2.pos[k] for k in inter])
            tj = {"coverage_of_student_universe": len(inter) / U.n,
                  "coverage_of_b6_rows": len(inter) / len(sub),
                  "n_pairs": U2.n, "n_queries": U2.n_q,
                  "human_grade_mismatches": mism, "labels": {},
                  "student_on_intersection": None}
            for col, _sset in TEACHER_SETS:
                lab = np.empty(U2.n)
                lab[o2] = sub[col].to_numpy(dtype=float)[idx_t]
                tu = compute_unit(U2, lab, W2, f"teacher {slug} {col}")
                tj["labels"][col] = {
                    k: {"point": float(tu["agg"][k][0]),
                        "ci90": ci90(tu["agg"][k][1:])}
                    for k in ("offset", "rel_rate_ratio", "precision",
                              "recall", "f1", "mae", "ndcg10", "map", "auc",
                              "f1_matched", "mean_pred")}
            if U2.n != U.n:
                st = {}
                for arm in sorted(ev):
                    for _c, sset in TEACHER_SETS + (("HUMAN", "HUMAN"),):
                        us = groups.get((arm, slug, sset))
                        if not us:
                            continue
                        vals = defaultdict(list)
                        for u in us:
                            p2 = np.empty(U2.n)
                            p2[o2] = u["pred"][idx_u]
                            a2 = compute_unit(U2, p2, W2[:1], "restricted")
                            for k in ("offset", "rel_rate_ratio", "f1",
                                      "ndcg10", "auc"):
                                vals[k].append(float(a2["agg"][k][0]))
                        st.setdefault(arm, {})[sset] = {
                            k: float(np.mean(v)) for k, v in vals.items()}
                tj["student_on_intersection"] = st
            teacher["judges"][slug] = tj

    return {"tag": TAG, "code_git": b7.git_hash(),
            "bootstrap": {"B": b, "seed": bseed, "unit": "query",
                          "ci": "90% percentile"},
            "integrity": integ, "universe": univ_summary,
            "shifts": shift_vals, "notes": dict(notes),
            "sets": set_res, "compares": compares, "kendall_tau_b": taus,
            "per_query_ndcg_deltas": deltas, "teacher": teacher,
            "label_diagnostics": ldiag,
            "runtime_s": round(time.time() - t0, 1)}


# ═══════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════

def _pt(R, arm, slug, s, k):
    try:
        return R["sets"][arm][slug][s]["endpoints"][k]["mean"]
    except KeyError:
        return None


def _cmp(R, arm, slug, a, b):
    for c in R["compares"]:
        if (c["arm"], c["judge"], c["a"], c["b"]) == (arm, slug, a, b):
            return c
    return None


def print_report(R: dict) -> None:
    line = "=" * 78
    print(line)
    print(f"  BATCH 7 post-hoc analyses  {TAG}")
    print("  None of this is part of the registered B7-P1..P8 adjudication.")
    print(line)
    it = R["integrity"]
    print(f"  integrity: {it['files']} eval files; stored point endpoints "
          f"recomputed from pairs match to {TOL:g}; CIs re-checked for "
          f"{it['ci_checked']} file(s) (bootstrap B/seed equal to this run).")
    print(f"  universe: {it['n_pairs']} pairs, {it['n_queries']} queries "
          f"(NDCG over {it['n_queries_ndcg']} with IDCG>0, MAP over "
          f"{it['n_queries_map']}); sha {it['universe_sha256'][:12]}")
    u = R["universe"]
    print(f"  human>=2 base rate b = {u['base_rate']['point']:.4f} "
          f"{_ci(u['base_rate']['ci90'])};  trivial all-relevant F1 = 2b/(1+b)"
          f" = {u['f1_all_relevant']['point']:.4f} "
          f"{_ci(u['f1_all_relevant']['ci90'])}")
    print(f"  random-ranking NDCG@10 floor ({N_RANDOM_PERM} perms/query, seed "
          f"{RANDOM_SEED}) = {u['ndcg10_random']['point']:.4f} "
          f"{_ci(u['ndcg10_random']['ci90'])}  (analytic expectation "
          f"{u['ndcg10_random_analytic']['point']:.4f})")
    for slug, nn in R["notes"].items():
        for n in nn:
            print(f"  NOTE [{slug}]: {n}")

    # item 1
    print(f"\n{line}\n  1. POST-HOC SHIFT-CALIBRATED RAW STUDENT  {TAG}\n{line}")
    print("  pred' = pred_RAW - shift.  Shifts use the TEACHER's label offset "
          "(same human budget as\n  the correction), not the student's own; "
          "ORACLE_SHIFT is in-sample on dl22 (reference only).")
    for arm, js in R["shifts"].items():
        for slug, sh in js.items():
            print(f"  [{arm}] {slug}:")
            for name, v in sh.items():
                vals = sorted(set(round(x, 6) for x in v["per_seed"].values()))
                print(f"    {name:<13} shift {', '.join(f'{x:+.4f}' for x in vals)}"
                      f"   ({v['source']})")
    print("  NDCG@10 / MAP of every shifted student equal RAW's to 1e-9 "
          "(asserted).")
    for arm in sorted(R["sets"]):
        for slug in sorted(R["sets"][arm]):
            print(f"\n  [{arm}] {slug} — paired A - B (3-seed sides, 90% CI; "
                  f"* = CI excludes 0)")
            print(f"    {'A vs B':<32} {'ndcg10':>16} {'offset':>16} "
                  f"{'|offset|':>16} {'mae':>16} {'f1':>16} {'ratio':>16} "
                  f"{'ece4':>16}")
            for c in R["compares"]:
                if (c["arm"], c["judge"]) != (arm, slug) or not (
                        c["a"].startswith("SHIFT") or c["a"].startswith("ORACLE")):
                    continue
                cells = []
                for k in ("ndcg10", "offset", "abs_offset", "mae", "f1",
                          "rel_rate_ratio", "ece4"):
                    e = c["endpoints"][k]
                    cells.append(f"{e['diff']:+.4f}{'*' if e['excludes_0'] else ' '}"
                                 f"[{e['ci90'][0]:+.3f},{e['ci90'][1]:+.3f}]"
                                 if math.isfinite(e["diff"]) else "n/a")
                print(f"    {c['a'] + ' vs ' + c['b']:<32} "
                      + " ".join(f"{x:>16}" for x in cells))

    # item 2
    print(f"\n{line}\n  2. THRESHOLD-FREE / MATCHED-PREVALENCE ENDPOINTS  {TAG}"
          f"\n{line}")
    print("  3-seed mean (90% CI of the seed mean).  F1@matched: top-P pairs "
          "called relevant, P = # human\n  positives on dl22 — IN-SAMPLE by "
          "construction.  frac>=1.5 = share with rounded grade >= 2.")
    hdr = ("set", "auc", "auc_q", "pr_auc", "f1_matched", "f1", "pred_sd",
           "frac_ge15", "frac_ge05", "ndcg10")
    for arm in sorted(R["sets"]):
        for slug in sorted(R["sets"][arm]):
            print(f"\n  [{arm}] {slug}")
            print("    " + f"{hdr[0]:<17}" + "".join(f"{h:>11}" for h in hdr[1:]))
            for s in SETS + SHIFT_SETS:
                if s not in R["sets"][arm][slug]:
                    continue
                e = R["sets"][arm][slug][s]["endpoints"]
                print("    " + f"{s:<17}" + "".join(
                    _f(e[k]["mean"], 11, 4, sign=False) for k in hdr[1:]))
            for s in ("RAW", "CORR_FROZEN", "HUMAN"):
                if s in R["sets"][arm][slug]:
                    e = R["sets"][arm][slug][s]["endpoints"]
                    print(f"    CI {s:<14} auc {_ci(e['auc']['ci90'])}  "
                          f"pr_auc {_ci(e['pr_auc']['ci90'])}  f1_matched "
                          f"{_ci(e['f1_matched']['ci90'])}")
            for a, bb in (("CORR_FROZEN", "RAW"), ("HUMAN", "RAW"),
                          ("CORR_FROZEN", "HUMAN")):
                c = _cmp(R, arm, slug, a, bb)
                if c:
                    print(f"    paired {a}-{bb}: " + "  ".join(
                        f"{k} {c['endpoints'][k]['diff']:+.4f}"
                        f"{'*' if c['endpoints'][k]['excludes_0'] else ''}"
                        f"{_ci(c['endpoints'][k]['ci90'])}"
                        for k in ("auc", "pr_auc", "f1_matched", "pred_sd")))

    # item 3
    print(f"\n{line}\n  3. KENDALL tau-b BETWEEN STUDENTS / PER-QUERY NDCG "
          f"DELTAS  {TAG}\n{line}")
    print("  tau-b of pair scores, same arm/judge/seed; per-query mean over "
          "queries (and pooled), 3-seed mean.")
    for t in R["kendall_tau_b"]:
        lab = (f"{t['a']}-{t['b']}" if t["a"] != t["b"]
               else f"{t['a']} seed-to-seed")
        print(f"    [{t['arm']}] {JUDGE_SHORT.get(t['judge'], t['judge']):<10} "
              f"{lab:<28} per-query {t['tau_query_mean']:+.4f}   pooled "
              f"{t['tau_pooled']:+.4f}")
    print("  per-query NDCG@10 deltas (seed-mean per side), share of queries "
          "with delta <0 / =0 / >0:")
    for d in R["per_query_ndcg_deltas"]:
        print(f"    [{d['arm']}] {JUDGE_SHORT.get(d['judge'], d['judge']):<10} "
              f"{d['a'] + '-' + d['b']:<18} {d['share_neg']:.3f} / "
              f"{d['share_zero']:.3f} / {d['share_pos']:.3f}   mean "
              f"{d['mean_delta']:+.4f}  mean|d| {d['mean_abs_delta']:.4f}  "
              f"(n={d['n_queries']})")

    # item 4
    print(f"\n{line}\n  4. TEACHER REFERENCE (judge's own dl22 labels)  {TAG}"
          f"\n{line}")
    print(f"  {R['teacher']['status']}")
    for slug, tj in R["teacher"]["judges"].items():
        if "labels" not in tj:
            print(f"  {slug}: {tj.get('note')}")
            continue
        print(f"  {slug}: coverage {tj['coverage_of_student_universe']:.3f} of "
              f"the student universe ({tj['n_pairs']} pairs, "
              f"{tj['n_queries']} queries); {tj['coverage_of_b6_rows']:.3f} of "
              f"B6 rows; human-grade mismatches {tj['human_grade_mismatches']}")
        print(f"    {'':<20}{'offset':>9}{'ratio':>9}{'P':>9}{'R':>9}{'F1':>9}"
              f"{'ndcg10':>9}{'auc':>9}")
        for col, sset in TEACHER_SETS:
            t = tj["labels"][col]
            print(f"    teacher {col:<12}" + "".join(
                _f(t[k]["point"], 9) for k in ("offset", "rel_rate_ratio",
                                               "precision", "recall", "f1",
                                               "ndcg10", "auc")))
            for arm in sorted(R["sets"]):
                if sset in R["sets"][arm].get(slug, {}):
                    e = R["sets"][arm][slug][sset]["endpoints"]
                    print(f"    student {arm[:12]:<12}" + "".join(
                        _f(e[k]["mean"], 9) for k in (
                            "offset", "rel_rate_ratio", "precision", "recall",
                            "f1", "ndcg10", "auc")) + f"   ({sset})")
        print("    (teacher NDCG ranks integer labels, ties by docid — heavy "
              "ties, read loosely)")
        if tj.get("student_on_intersection"):
            print("    coverage < 1: students restricted to the intersection "
                  "(3-seed mean):")
            for arm, ss in tj["student_on_intersection"].items():
                for s, v in ss.items():
                    print(f"      [{arm}] {s:<12} " + "  ".join(
                        f"{k} {x:+.4f}" for k, x in v.items()))

    # item 5
    print(f"\n{line}\n  5. LABEL-SET DIAGNOSTICS (training universe)  {TAG}"
          f"\n{line}")
    for slug, ld in R["label_diagnostics"].items():
        print(f"  {slug}" + (f"  NOTE: {ld['note']}" if ld.get("note") else ""))
        fr = ld.get("frozen_rule") or {}
        for fam, r in fr.items():
            print(f"    frozen {fam} rule: dial L{r.get('dial_level')} map "
                  f"{r.get('monotone_map')}")
        print(f"    {'set':<17}{'mean':>8}{'offset':>9}  hist 0/1/2/3"
              f"{'':<10}{'grade1':>7}{'RAW1->0':>8}{'pearson':>9}"
              f"{'spearman':>9}")
        for s, d in ld["sets"].items():
            if "mean" not in d:
                print(f"    {s:<17} {d.get('note')}")
                continue
            hist = "/".join(str((d.get("hist") or {}).get(str(g), "?"))
                            for g in range(4))
            print(f"    {s:<17}{_f(d['mean'], 8, 3, False)}"
                  f"{_f(d.get('offset_vs_human'), 9, 3)}  {hist:<22}"
                  f"{_f(d.get('share_grade1'), 7, 3, False)}"
                  f"{_f(d.get('raw1_to_0_share'), 8, 3, False)}"
                  f"{_f(d.get('pearson_vs_human'), 9, 3)}"
                  f"{_f(d.get('spearman_vs_human'), 9, 3)}")

    # compact summary
    print(f"\n{line}\n  COMPACT SUMMARY  {TAG}\n{line}")
    for arm in sorted(R["sets"]):
        for slug in sorted(R["sets"][arm]):
            p = lambda s, k: _f(_pt(R, arm, slug, s, k), 7)   # noqa: E731
            print(f"  [{arm}] {JUDGE_SHORT.get(slug, slug)}")
            print(f"    NDCG   RAW {p('RAW','ndcg10')} HUMAN {p('HUMAN','ndcg10')}"
                  f" FROZEN {p('CORR_FROZEN','ndcg10')} SHIFT_FULL "
                  f"{p('SHIFT_FULL','ndcg10')}")
            print(f"    offset RAW {p('RAW','offset')} FROZEN "
                  f"{p('CORR_FROZEN','offset')} SHIFT_FULL {p('SHIFT_FULL','offset')}"
                  f" SHIFT_n1000 {p('SHIFT_n1000','offset')} HUMAN "
                  f"{p('HUMAN','offset')}")
            print(f"    F1     RAW {p('RAW','f1')} FROZEN {p('CORR_FROZEN','f1')}"
                  f" SHIFT_FULL {p('SHIFT_FULL','f1')} HUMAN {p('HUMAN','f1')}"
                  f" | F1@matched RAW {p('RAW','f1_matched')} FROZEN "
                  f"{p('CORR_FROZEN','f1_matched')} HUMAN "
                  f"{p('HUMAN','f1_matched')}")
            print(f"    AUC    RAW {p('RAW','auc')} FROZEN {p('CORR_FROZEN','auc')}"
                  f" HUMAN {p('HUMAN','auc')}")
            for n in MIX_NS:
                c = _cmp(R, arm, slug, f"SHIFT_n{n}", f"CORR_REFIT_n{n}")
                if not c:
                    continue
                print(f"    SHIFT_n{n}-CORR_REFIT_n{n}: " + "  ".join(
                    f"{k} {c['endpoints'][k]['diff']:+.4f}"
                    f"{'*' if c['endpoints'][k]['excludes_0'] else ''}"
                    f"{_ci(c['endpoints'][k]['ci90'])}"
                    for k in ("offset", "abs_offset", "mae", "f1")))
    print(f"\n  runtime {R['runtime_s']}s; code {R['code_git']}")


def write_outputs(R: dict, out: Path, report: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "posthoc_report.txt").write_text(report, encoding="utf-8")
    (out / "posthoc.json").write_text(
        json.dumps(_clean(R), indent=1, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")
    with open(out / "posthoc_sets.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tag", "arm", "judge", "set", "kind", "n_seeds",
                    "endpoint", "mean", "seed_sd", "ci_lo", "ci_hi",
                    "per_seed"])
        for k, v in R["universe"].items():
            w.writerow([TAG, "*", "*", "UNIVERSE", "universe-level", "",
                        k, v["point"], "", v["ci90"][0], v["ci90"][1], ""])
        for arm, js in R["sets"].items():
            for slug, ss in js.items():
                for s, sr in ss.items():
                    for k, e in sr["endpoints"].items():
                        w.writerow([TAG, arm, slug, s, sr["kind"],
                                    sr["n_seeds"], k, e["mean"], e["sd"],
                                    e["ci90"][0], e["ci90"][1],
                                    ";".join(f"{x:.6g}" for x in e["per_seed"])])
    with open(out / "posthoc_compares.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tag", "arm", "judge", "a", "b", "endpoint", "a_value",
                    "b_value", "diff", "ci_lo", "ci_hi", "ci_excludes_0",
                    "note"])
        for c in R["compares"]:
            for k, e in c["endpoints"].items():
                w.writerow([TAG, c["arm"], c["judge"], c["a"], c["b"], k,
                            e["a"], e["b"], e["diff"], e["ci90"][0],
                            e["ci90"][1], e["excludes_0"], c["why"]])


class _Tee(io.StringIO):
    def __init__(self, stream):
        super().__init__()
        self._s = stream

    def write(self, x):
        self._s.write(x)
        return super().write(x)


def cmd_run(args) -> dict:
    R = analyse(Path(args.eval_root), Path(args.labels_root),
                Path(args.b6_labels) if args.b6_labels else None,
                args.b, args.seed)
    buf = _Tee(sys.stdout if not getattr(args, "quiet", False)
               else io.StringIO())
    with redirect_stdout(buf):
        print_report(R)
    write_outputs(R, Path(args.out), buf.getvalue())
    if not getattr(args, "quiet", False):
        print(f"  -> {args.out}/posthoc.json, posthoc_sets.csv, "
              f"posthoc_compares.csv, posthoc_report.txt")
    return R


# ═══════════════════════════════════════════════════════════════════════════
# selftest
# ═══════════════════════════════════════════════════════════════════════════

ST_SLUGS = ("deepseek__deepseek-v4-flash", "google__gemini-3.8-flash")
ST_ARMS = ("minilm-raw", "msmarco-ce")
ST_SET_OFF = {"HUMAN": 0.0, "RAW": 0.6, "CORR_FROZEN": -0.1,
              "CORR_REFIT": -0.05, "MIX_200": 0.55, "MIX_1000": 0.4,
              "CORR_REFIT_n200": 0.2, "CORR_REFIT_n1000": -0.2}


def _st_universe(rng):
    keys, hum = [], []
    for qi in range(14):
        q = f"q{qi:02d}"
        nd = int(rng.integers(8, 20))
        for di in range(nd):
            if qi == 0:
                g = 1 if di % 2 else 0              # no human-relevant pair
            elif qi == 1:
                g = 3 if di % 2 else 2              # all human-relevant
            else:
                g = int(rng.choice(4, p=[0.45, 0.25, 0.2, 0.1]))
            keys.append((q, f"d{qi:02d}_{di:02d}"))
            hum.append(g)
    return keys, np.array(hum)


def _st_make(root: Path, b: int, rng) -> dict:
    keys, hum = _st_universe(rng)
    ev_root = root / "eval"
    lab_root = root / "labels"
    for slug in ST_SLUGS:
        model = slug.replace("__", "/")
        sets = [s for s in SETS
                if slug == ST_SLUGS[0] or s not in ("MIX_1000",
                                                    "CORR_REFIT_n1000")]
        for arm in ST_ARMS:
            for s in sets:
                for seed in (1, 2, 3):
                    noise = rng.normal(0, 0.7, size=len(keys))
                    pred = (hum * 0.5 + 0.4 + ST_SET_OFF[s] + noise
                            + (0.0 if arm == "minilm-raw" else 0.1))
                    if s == "CORR_FROZEN" and seed == 1:
                        pred[:4] = 1.5                       # exact boundary
                    pairs = [(q, d, float(p), int(h))
                             for (q, d), p, h in zip(keys, pred, hum)]
                    res = evaluate_pairs(pairs, b, BOOT_SEED)
                    r = {"dataset": "dl22", "arm": arm, "label_set": s,
                         "judge": model, "seed": seed,
                         "model_dir": f"/x/{arm}/{slug}/{s}_s{seed}",
                         "base_model": arm, "objective": "regression",
                         "bootstrap": {"B": b, "seed": BOOT_SEED},
                         "universe_sha256": universe_sha(pairs), **res,
                         "pairs": [list(x) for x in pairs]}
                    p = ev_root / arm / slug / f"{s}_s{seed}_dl22.json"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(json.dumps(r), encoding="utf-8")
        # training universe labels (dl19) + manifest
        tr = [("dl19", f"t{qi:02d}", f"e{qi:02d}_{di}")
              for qi in range(30) for di in range(10)]
        th = rng.choice(4, size=len(tr), p=[0.4, 0.3, 0.2, 0.1])
        traw = np.clip(th + rng.choice([0, 1, 1, 2], size=len(tr)), 0, 3)
        tfro = np.array([0, 0, 2, 3])[traw]
        lab = {"HUMAN": th, "RAW": traw, "CORR_FROZEN": tfro,
               "CORR_REFIT": np.array([0, 1, 1, 3])[traw]}
        mixq = [f"dl19|t{qi:02d}" for qi in (3, 7, 11)]
        for n in MIX_NS:
            if n == 1000 and slug != ST_SLUGS[0]:
                continue
            inq = np.array([f"{a}|{b_}" in mixq for a, b_, _c in tr])
            lab[f"MIX_{n}"] = np.where(inq, th, traw)
            lab[f"CORR_REFIT_n{n}"] = np.clip(traw - 1, 0, 3)
        summary = {s: {"mean": float(v.mean()),
                       "offset_vs_human": float(v.mean() - th.mean()),
                       "hist": {str(g): int((v == g).sum()) for g in range(4)}}
                   for s, v in lab.items()}
        m = {"model": model, "summary": summary,
             "frozen_rule": {"msmarco": {"dial_level": 1,
                                         "monotone_map": [0, 0, 2, 3]}},
             "equal_budget_rules": {
                 f"CORR_REFIT_n{n}": {"delta_hat": 0.31 + n / 1e4,
                                      "dial_level": 1,
                                      "monotone_map": [0, 0, 2, 3],
                                      "fit_queries": mixq, "n_fit_rows": 30}
                 for n in MIX_NS if f"MIX_{n}" in lab},
             "mix": {f"MIX_{n}": {"human_queries": mixq}
                     for n in MIX_NS if f"MIX_{n}" in lab}}
        d = lab_root / slug
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
        if slug == ST_SLUGS[0]:                    # label files for one judge
            train0 = (tr, lab, mixq)
            for s, v in lab.items():
                with open(d / f"labels_{s}.jsonl", "w", encoding="utf-8") as f:
                    for (ds, q, doc), y in zip(tr, v):
                        f.write(json.dumps({"dataset": ds, "qid": q,
                                            "docid": doc, "label": float(y),
                                            "set": s}) + "\n")
    # B6 teacher labels: judge 0 only, drop 5 pairs (coverage < 1)
    rows = []
    for (q, d), h in list(zip(keys, hum))[5:]:
        rows.append({"dataset": "dl22", "model_id": ST_SLUGS[0].replace(
            "__", "/"), "qid": q, "docid": d, "human": int(h),
            "RAW": int(min(3, h + rng.integers(0, 2))),
            "FROZEN": int(h), "REFIT": int(max(0, h - rng.integers(0, 2))),
            "fold": 0})
    b6 = pd.DataFrame(rows)
    b6p = root / "batch6_labels_dl22.parquet"
    try:
        b6.to_parquet(b6p, index=False)
    except Exception:                                        # noqa: BLE001
        b6p = b6p.with_suffix(".pkl")
        b6.to_pickle(b6p)
    bad = root / "bad_b6.pkl"
    b6.rename(columns={"FROZEN": "CORR"}).to_pickle(bad)
    return {"eval": ev_root, "labels": lab_root, "b6": b6p, "b6_bad": bad,
            "keys": keys, "hum": hum, "b6_df": b6, "train": train0}


def cmd_selftest(args) -> None:
    passed, failed = [], []

    def check(name, cond, detail=""):
        (passed if cond else failed).append(name)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  ({detail})" if detail and not cond else ""))

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except Exception:                                        # noqa: BLE001
        roc_auc_score = average_precision_score = None
    try:
        from scipy.stats import kendalltau
    except Exception:                                        # noqa: BLE001
        kendalltau = None

    rng = np.random.default_rng(7)
    B = 200
    tmp = Path(args.keep_dir) if args.keep_dir else Path(tempfile.mkdtemp(
        prefix="b7posthoc_"))
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        print("  -- unit checks")
        # AUC / AP / F1@matched vs references, with ties and bootstrap weights
        n = 300
        pos = rng.random(n) < 0.3
        sc = np.round(rng.normal(pos * 0.8, 1.0), 1)            # many ties
        qidx = rng.integers(0, 20, size=n)
        Wq = build_W(20, 5, 3)
        Wp = Wq[:, qidx]
        a = w_auc(sc, pos, Wp)
        ap = w_ap(sc, pos, Wp)
        for r in range(Wp.shape[0]):
            w = Wp[r]
            if roc_auc_score is not None:
                ra = roc_auc_score(pos, sc, sample_weight=w)
                rp = average_precision_score(pos, sc, sample_weight=w)
            else:
                rep = np.repeat(np.arange(n), w.astype(int))
                x, y = sc[rep], pos[rep]
                sp, sn = x[y], x[~y]
                ra = ((sp[:, None] > sn).mean() + 0.5 * (sp[:, None] == sn)
                      .mean())
                o = np.argsort(-x, kind="mergesort")
                th = np.unique(x)[::-1]
                rp, prev = 0.0, 0.0
                for t in th:
                    m = x >= t
                    tp = (m & y).sum()
                    rec = tp / y.sum()
                    rp += (rec - prev) * tp / m.sum()
                    prev = rec
                del o
            check(f"weighted AUC row {r} vs "
                  f"{'sklearn' if roc_auc_score else 'brute force'}",
                  abs(a[r] - ra) < 1e-10, f"{a[r]} vs {ra}")
            check(f"weighted PR-AUC row {r} vs "
                  f"{'sklearn' if average_precision_score else 'brute force'}",
                  abs(ap[r] - rp) < 1e-10, f"{ap[r]} vs {rp}")
        # F1@matched: continuous scores -> exact top-K by replication
        sc2 = rng.normal(pos * 0.8, 1.0)
        f1m = w_f1_matched(sc2, pos, Wp)
        for r in range(Wp.shape[0]):
            rep = np.repeat(np.arange(n), Wp[r].astype(int))
            x, y = sc2[rep], pos[rep]
            K = int(y.sum())
            top = np.argsort(-x, kind="mergesort")[:K]
            check(f"F1@matched row {r} vs replicated top-K",
                  abs(f1m[r] - y[top].sum() / K) < 1e-12)
        f1t = w_f1_matched(np.array([3.0, 2, 2, 1]), np.array([1, 1, 0, 0],
                                                               bool),
                           np.ones((1, 4)))
        check("F1@matched fractional tie at the cut = 0.75",
              abs(f1t[0] - 0.75) < 1e-12, str(f1t))
        # tau-b
        x = np.round(rng.normal(size=60), 1)
        y = np.round(x + rng.normal(size=60), 1)
        if kendalltau is not None:
            ref = float(kendalltau(x, y).correlation)
        else:
            c = d = tx = ty = 0
            for i in range(60):
                for j in range(i + 1, 60):
                    s = np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
                    c += s > 0
                    d += s < 0
                    tx += x[i] == x[j]
                    ty += y[i] == y[j]
            n0 = 60 * 59 / 2
            ref = (c - d) / math.sqrt((n0 - tx) * (n0 - ty))
        check(f"tau-b vs {'scipy' if kendalltau else 'brute force'}",
              abs(tau_b(x, y) - ref) < 1e-12, f"{tau_b(x, y)} vs {ref}")
        # shift-invariance guard fires on a broken S
        S = np.zeros((3, len(STAT_COLS)))
        S2 = S.copy()
        S2[1, STAT_COLS.index("ndcg")] = 1e-6
        try:
            check_shift_invariance(S, S2, "synthetic")
            check("shift-invariance guard fires", False)
        except SystemExit:
            check("shift-invariance guard fires", True)

        print("  -- end-to-end on synthetic eval files")
        D = _st_make(tmp, B, rng)
        out = tmp / "out"
        ns = argparse.Namespace(eval_root=str(D["eval"]),
                                labels_root=str(D["labels"]),
                                b6_labels=str(D["b6"]), out=str(out), b=B,
                                seed=BOOT_SEED, quiet=True)
        R = cmd_run(ns)
        check("integrity: all files recomputed, CIs checked",
              R["integrity"]["files"] == R["integrity"]["ci_checked"] == 2 * (
                  8 + 6) * 3, str(R["integrity"]))
        for f in ("posthoc.json", "posthoc_sets.csv", "posthoc_compares.csv",
                  "posthoc_report.txt"):
            check(f"output {f} written", (out / f).exists())
        J = json.loads((out / "posthoc.json").read_text())
        check("posthoc.json strict JSON (no NaN)", isinstance(J, dict))
        sets_csv = pd.read_csv(out / "posthoc_sets.csv")
        cmp_csv = pd.read_csv(out / "posthoc_compares.csv")
        check("sets csv has rows for all arms and shift sets",
              set(sets_csv.arm) >= set(ST_ARMS)
              and {"SHIFT_FULL", "SHIFT_n200", "SHIFT_nL200",
                   "ORACLE_SHIFT"} <= set(sets_csv.set))
        check("compares csv tagged exploratory",
              (cmp_csv.tag == TAG).all() and len(cmp_csv) > 0)
        # shift invariance + shift magnitude
        man = json.loads((D["labels"] / ST_SLUGS[0] / "manifest.json")
                         .read_text())
        sf = man["summary"]["RAW"]["mean"] - man["summary"]["HUMAN"]["mean"]
        for arm in ST_ARMS:
            ss = R["sets"][arm][ST_SLUGS[0]]
            for s in SHIFT_SETS:
                for k in ("ndcg10", "map"):
                    a_, b_ = (ss[s]["endpoints"][k]["per_seed"],
                              ss["RAW"]["endpoints"][k]["per_seed"])
                    check(f"[{arm}] {s} {k} == RAW (shift invariance)",
                          np.allclose(a_, b_, atol=1e-12, rtol=0))
            ou_raw = np.array(ss["RAW"]["endpoints"]["offset_unclipped"][
                "per_seed"])
            ou_sf = np.array(ss["SHIFT_FULL"]["endpoints"]["offset_unclipped"][
                "per_seed"])
            check(f"[{arm}] SHIFT_FULL moves unclipped offset by RAW-HUMAN "
                  f"label mean", np.allclose(ou_raw - ou_sf, sf, atol=1e-12))
            ou_n = np.array(ss["SHIFT_n200"]["endpoints"]["offset_unclipped"][
                "per_seed"])
            check(f"[{arm}] SHIFT_n200 uses the rule's delta_hat",
                  np.allclose(ou_raw - ou_n, 0.31 + 0.02, atol=1e-12))
            check(f"[{arm}] ORACLE_SHIFT (clipped) offset == 0",
                  np.allclose(ss["ORACLE_SHIFT"]["endpoints"][
                      "offset"]["per_seed"], 0, atol=1e-9))
            # matched F1 invariant to shift; frac_ge15 changes
            check(f"[{arm}] F1@matched / AUC invariant to shift",
                  np.allclose(ss["SHIFT_FULL"]["endpoints"]["f1_matched"][
                      "per_seed"], ss["RAW"]["endpoints"]["f1_matched"][
                      "per_seed"]) and np.allclose(
                      ss["SHIFT_FULL"]["endpoints"]["auc"]["per_seed"],
                      ss["RAW"]["endpoints"]["auc"]["per_seed"]))
        tr, lab, mixq = D["train"]
        inq = np.array([f"{a_}|{b_}" in mixq for a_, b_, _c in tr])
        snl = float(lab["RAW"][inq].mean() - lab["HUMAN"][inq].mean())
        got = R["shifts"]["minilm-raw"][ST_SLUGS[0]]["SHIFT_nL200"][
            "per_seed"][1]
        check("SHIFT_nL200 = label-level offset on MIX human pairs",
              abs(got - snl) < 1e-12, f"{got} vs {snl}")
        check("SHIFT_nL skipped without label files (judge 2)",
              "SHIFT_nL200" not in R["shifts"]["minilm-raw"][ST_SLUGS[1]]
              and any("SHIFT_nL200 skipped" in n
                      for n in R["notes"].get(ST_SLUGS[1], [])))
        check("missing sets skipped (judge 2 has no MIX_1000)",
              "MIX_1000" not in R["sets"]["minilm-raw"][ST_SLUGS[1]]
              and "SHIFT_n1000" not in R["sets"]["minilm-raw"][ST_SLUGS[1]])
        # paired() == batch7.compare_results on the same files
        ev = load_evals(D["eval"])
        sa = [ev["minilm-raw"][ST_SLUGS[0]]["CORR_FROZEN"][s] for s in (1, 2, 3)]
        sb = [ev["minilm-raw"][ST_SLUGS[0]]["RAW"][s] for s in (1, 2, 3)]
        ref = b7.compare_results(sa, sb, b=B, seed=BOOT_SEED)
        mine = _cmp(R, "minilm-raw", ST_SLUGS[0], "CORR_FROZEN", "RAW")
        ok = all(_close(mine["endpoints"][k]["diff"], ref["endpoints"][k]["diff"])
                 and all(_close(x, y) for x, y in zip(
                     mine["endpoints"][k]["ci90"], ref["endpoints"][k]["ci90"]))
                 for k in CMP_EPS if k in ref["endpoints"])
        check("paired() reproduces batch7.compare_results (diff + CI)", ok)
        # pair-level endpoints vs direct numpy on one file
        U = Universe(sb[0]["pairs"])
        pred = U.align(sb[0]["pairs"], "x")
        e = R["sets"]["minilm-raw"][ST_SLUGS[0]]["RAW"]["endpoints"]
        rb = np.floor(np.clip(pred, 0, 3) + 0.5)
        check("frac_ge15 / frac_ge05 / pred_sd vs numpy",
              abs(e["frac_ge15"]["per_seed"][0] - (rb >= 2).mean()) < 1e-12
              and abs(e["frac_ge05"]["per_seed"][0] - (rb >= 1).mean()) < 1e-12
              and abs(e["pred_sd"]["per_seed"][0] - pred.std()) < 1e-12)
        if roc_auc_score is not None:
            check("pooled AUC point vs sklearn",
                  abs(e["auc"]["per_seed"][0] - roc_auc_score(U.rel, pred))
                  < 1e-12)
        aq = [roc_auc_score(U.rel[i], pred[i]) if roc_auc_score else None
              for i in U.q_members if 0 < U.rel[i].sum() < i.size]
        if roc_auc_score is not None:
            check("per-query-mean AUC vs sklearn",
                  abs(e["auc_q"]["per_seed"][0] - np.mean(aq)) < 1e-12)
        b_rate = U.rel.mean()
        check("base rate / trivial F1",
              abs(R["universe"]["base_rate"]["point"] - b_rate) < 1e-12
              and abs(R["universe"]["f1_all_relevant"]["point"]
                      - 2 * b_rate / (1 + b_rate)) < 1e-12)
        un = R["universe"]
        check("random NDCG floor ~ analytic expectation",
              abs(un["ndcg10_random"]["point"]
                  - un["ndcg10_random_analytic"]["point"]) < 0.03,
              f"{un['ndcg10_random']['point']} vs "
              f"{un['ndcg10_random_analytic']['point']}")
        check("integrity: NDCG over 14 queries, MAP over 13 (q00 has no "
              "human>=2 pair)", R["integrity"]["n_queries_ndcg"] == 14
              and R["integrity"]["n_queries_map"] == 13)
        # tau: RAW-RAW same seed would be 1; check range + seed-to-seed rows
        tt = [t for t in R["kendall_tau_b"] if t["kind"].startswith("seed")]
        check("tau-b rows incl. seed-to-seed references",
              len(tt) == 2 * 2 * 2 and all(-1 <= t["tau_pooled"] <= 1
                                           for t in R["kendall_tau_b"]))
        t0 = [t for t in R["kendall_tau_b"]
              if (t["arm"], t["judge"], t["a"], t["b"]) ==
              ("minilm-raw", ST_SLUGS[0], "RAW", "HUMAN")][0]
        ua = U.align(ev["minilm-raw"][ST_SLUGS[0]]["RAW"][2]["pairs"], "x")
        ub = U.align(ev["minilm-raw"][ST_SLUGS[0]]["HUMAN"][2]["pairs"], "x")
        if kendalltau is not None:
            check("pooled tau-b (seed 2) vs scipy",
                  abs(t0["per_seed"][1]["tau_pooled"]
                      - kendalltau(ua, ub).correlation) < 1e-12)
        dd = R["per_query_ndcg_deltas"][0]
        check("per-query delta shares sum to 1",
              abs(dd["share_neg"] + dd["share_zero"] + dd["share_pos"] - 1)
              < 1e-12 and dd["n_queries"] == 14)
        # teacher
        tj = R["teacher"]["judges"][ST_SLUGS[0]]
        b6 = D["b6_df"]
        hb, rw = b6.human.to_numpy(), b6.RAW.to_numpy()
        tp = ((rw >= 2) & (hb >= 2)).sum()
        f1 = 2 * tp / ((rw >= 2).sum() + (hb >= 2).sum())
        check("teacher coverage < 1 reported",
              abs(tj["coverage_of_student_universe"]
                  - (len(D["keys"]) - 5) / len(D["keys"])) < 1e-12)
        check("teacher RAW offset / ratio / F1 vs numpy",
              abs(tj["labels"]["RAW"]["offset"]["point"]
                  - (rw.mean() - hb.mean())) < 1e-12
              and abs(tj["labels"]["RAW"]["rel_rate_ratio"]["point"]
                      - (rw >= 2).sum() / (hb >= 2).sum()) < 1e-12
              and abs(tj["labels"]["RAW"]["f1"]["point"] - f1) < 1e-12)
        check("teacher FROZEN(=human) offset 0, F1 1",
              abs(tj["labels"]["FROZEN"]["offset"]["point"]) < 1e-12
              and abs(tj["labels"]["FROZEN"]["f1"]["point"] - 1) < 1e-12)
        check("students restricted to the intersection",
              bool(tj["student_on_intersection"]))
        check("teacher: judge without B6 rows noted",
              "note" in R["teacher"]["judges"][ST_SLUGS[1]])
        # label diagnostics
        ld = R["label_diagnostics"][ST_SLUGS[0]]["sets"]
        cf = ld["CORR_FROZEN"]
        check("label hist sums to n", sum(cf["hist"].values()) == cf["n"])
        check("label pearson vs numpy",
              abs(cf["pearson_vs_human"]
                  - np.corrcoef(lab["CORR_FROZEN"], lab["HUMAN"])[0, 1]) < 1e-9)
        check("frozen map [0,0,2,3]: RAW grade-1 -> 0 share = 1, grade-1 "
              "share = 0",
              cf["raw1_to_0_share"] == 1.0 and cf["share_grade1"] == 0.0)
        check("label diagnostics fall back to manifest when files absent",
              "note" in R["label_diagnostics"][ST_SLUGS[1]])
        # graceful skips
        ns2 = argparse.Namespace(**{**vars(ns), "b6_labels": str(D["b6_bad"]),
                                    "out": str(tmp / "out2")})
        R2 = cmd_run(ns2)
        check("B6 file with wrong columns -> teacher skipped gracefully",
              not R2["teacher"]["judges"]
              and "lacks columns" in R2["teacher"]["status"])
        ns3 = argparse.Namespace(**{**vars(ns), "b6_labels": None,
                                    "out": str(tmp / "out3")})
        R3 = cmd_run(ns3)
        check("no --b6-labels -> teacher skipped",
              R3["teacher"]["status"] == "--b6-labels not given")
        # integrity guard fires on a tampered file
        bad_root = tmp / "eval_bad"
        shutil.copytree(D["eval"], bad_root)
        p = bad_root / "minilm-raw" / ST_SLUGS[0] / "RAW_s2_dl22.json"
        r = json.loads(p.read_text())
        r["point"]["ndcg10"] += 1e-6
        p.write_text(json.dumps(r))
        try:
            analyse(bad_root, D["labels"], None, B, BOOT_SEED)
            check("integrity guard fires on a tampered point value", False)
        except SystemExit as ex:
            check("integrity guard fires on a tampered point value",
                  "integrity check failed" in str(ex))
        # report smoke
        rep = (out / "posthoc_report.txt").read_text()
        check("report carries the exploratory tag in every section",
              rep.count(TAG) >= 7)
    finally:
        if not args.keep_dir:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n  selftest: {len(passed)} passed, {len(failed)} failed")
    if failed:
        for f in failed:
            print(f"    FAILED: {f}")
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="post-hoc analyses on the B7 eval JSONs")
    r.add_argument("--eval-root", required=True)
    r.add_argument("--labels-root", required=True)
    r.add_argument("--b6-labels", default=None)
    r.add_argument("--out", required=True)
    r.add_argument("--b", type=int, default=BOOT_B)
    r.add_argument("--seed", type=int, default=BOOT_SEED)
    s = sub.add_parser("selftest", help="synthetic end-to-end test")
    s.add_argument("--keep-dir", default=None)
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a)
    else:
        cmd_selftest(a)


if __name__ == "__main__":
    main()
