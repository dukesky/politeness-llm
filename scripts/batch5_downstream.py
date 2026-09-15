"""Batch 5 / gap-B — DOWNSTREAM UTILITY of judge operating-point calibration.

What this asks (and what it does NOT ask)
-----------------------------------------
Batch 4 showed that an operating-point correction (the tone "dial" plus a
monotone output remap) raises judge-vs-human kappa, and C4 showed that on the
sealed dl22 the *frozen* map's kappa gains did not transfer.  Both of those are
statements about **agreement on individual labels**.

This script asks a different, downstream question:

    If you replace NIST's human qrels with an LLM judge's labels and rank a
    pool of retrieval SYSTEMS, do the operating-point-CORRECTED labels
    reproduce the human ranking of systems better than the RAW labels do?

That is a question about **evaluation outcomes**, not about judgment quality.
A judge can be badly calibrated on individual labels and still rank systems
perfectly (NDCG only needs the grade ordering within a query), and it can be
well calibrated and still rank systems badly.  Nothing here should be reported
as evidence that the correction makes the judge "better"; the batch-4 framing
stands — the correction removes a MEASUREMENT ARTIFACT (operating-point
misalignment with the reference annotators).  The question here is only
whether removing that artifact buys anything for the thing practitioners
actually do with an LLM judge: compare systems.

Expected direction, stated up front so the result is interpretable
------------------------------------------------------------------
A monotone label map CANNOT reorder documents within a query, and the dial
only shifts the operating point.  NDCG@10 is invariant to any *strictly*
increasing relabelling of the grade scale, so a strictly-increasing frozen map
must leave every system score untouched and give delta_tau EXACTLY 0.  The
correction can only move system scores when the frozen map is **non-injective**
(it collapses two grades, e.g. (0,0,1,2)) or when the dial level is not L3 (a
different level is a different set of labels, not a relabelling).  So:

  * delta_tau == 0 with an identity/strictly-increasing map at L3 is a
    CORRECTNESS CHECK, not a null result — the script reports it as such.
  * a non-zero delta_tau comes from grade COLLAPSE (which changes the gain
    vector and the ideal DCG) or from switching dial level.

Target collection
-----------------
TREC DL 2021 passage (``dl21`` = msmarco-passage-v2/trec-dl-2021/judged), 53
judged queries.  dl21 is NOT sealed.  **dl22 is hard-excluded** here
(``ALLOWED``): the sealed holdout has its own pre-registered protocol
(scripts/c4_unseal_dl22.py) and must not leak into an unregistered analysis.

Subcommands
-----------
    fetch-runs   acquire / register the system pool  (TREC runs or pyserini)
    coverage     how much of the pool's top-10 is judged, and by whom
    evaluate     the analysis: NDCG@10 -> system ranking -> Kendall tau

Typical Colab sequence
----------------------
    python scripts/batch5_downstream.py fetch-runs --pyserini-pool
    # (generate the runs in the notebook, or drop TREC runs in $DATA_DIR/runs/dl21/)
    python scripts/batch5_downstream.py coverage --models <judge>,<judge>
    python scripts/batch5_downstream.py evaluate --models <judge>,<judge>

Requires: numpy, pandas, scipy (already in requirements.txt); ir_datasets only
for the ``coverage`` top-up writer and for qrels.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from batch1r_stats import write_tex, esc                      # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

# Hard allow-list. dl22 is the sealed holdout; this analysis is not registered
# for it and must never touch it, with or without --unseal-dl22.
ALLOWED: frozenset[str] = frozenset({"dl21"})
DATASET = "dl21"
IR_DATASET_ID = "msmarco-passage-v2/trec-dl-2021/judged"

TOP_K = 10                 # depth of the ranking we evaluate and cover
N_QUERIES_EXPECTED = 53    # dl21 passage judged queries (sanity check only)
IDENTITY_MAP = (0, 1, 2, 3)
GRADE_MIN, GRADE_MAX = 0, 3

LABEL_SETS = ("HUMAN", "RAW", "CORR")

# OpenRouter price table, USD per 1M tokens (in / out). Coordinator updates
# this when the price list moves; it only drives the printed cost estimate.
PRICES: dict[str, tuple[float, float]] = {
    "v4-flash": (0.066, 0.131),
    "v4.1": (0.30, 1.20),
    "gemini-3.5": (1.50, 9.00),
}
TOK_IN, TOK_OUT = 300, 12   # per judge call, order-of-magnitude


# ═══════════════════════════════════════════════════════════════════════════
# The pyserini fallback pool (documented here, GENERATED in the notebook)
# ═══════════════════════════════════════════════════════════════════════════
#
# The official TREC 2021 DL passage runs live under https://trec.nist.gov/
# results/trec30/, which answers every request with HTTP 401 and
# `www-authenticate: Basic realm="results"` — they are released to TREC
# participants only.  So unless the coordinator has participant credentials,
# the system pool is ours to build.
#
# DIVERSITY WARNING (this is a real threat to the analysis, not boilerplate):
# a pool of BM25 parameter variants alone produces highly correlated rankings.
# Kendall tau between near-identical systems is dominated by noise, and a
# low-variance pool makes tau_raw and tau_corr both unstable.  Keep at least
# the d2q-T5 and a learned-sparse family in the pool so the systems actually
# differ.  Flagged as a coordinator decision below.
#
PYSERINI_POOL: list[dict] = [
    # --- BM25 parameter grid on the plain v2 passage index -----------------
    {"runtag": "bm25-k0.9-b0.4",  "index": "msmarco-v2-passage",
     "retriever": "bm25", "params": {"k1": 0.9, "b": 0.4},  "family": "bm25"},
    {"runtag": "bm25-k0.9-b0.68", "index": "msmarco-v2-passage",
     "retriever": "bm25", "params": {"k1": 0.9, "b": 0.68}, "family": "bm25"},
    {"runtag": "bm25-k0.6-b0.4",  "index": "msmarco-v2-passage",
     "retriever": "bm25", "params": {"k1": 0.6, "b": 0.4},  "family": "bm25"},
    {"runtag": "bm25-k1.2-b0.4",  "index": "msmarco-v2-passage",
     "retriever": "bm25", "params": {"k1": 1.2, "b": 0.4},  "family": "bm25"},
    {"runtag": "bm25-k1.2-b0.75", "index": "msmarco-v2-passage",
     "retriever": "bm25", "params": {"k1": 1.2, "b": 0.75}, "family": "bm25"},
    {"runtag": "bm25-k1.5-b0.4",  "index": "msmarco-v2-passage",
     "retriever": "bm25", "params": {"k1": 1.5, "b": 0.4},  "family": "bm25"},
    {"runtag": "bm25-k0.9-b0.2",  "index": "msmarco-v2-passage",
     "retriever": "bm25", "params": {"k1": 0.9, "b": 0.2},  "family": "bm25"},
    {"runtag": "bm25-k0.9-b0.8",  "index": "msmarco-v2-passage",
     "retriever": "bm25", "params": {"k1": 0.9, "b": 0.8},  "family": "bm25"},
    # --- pseudo-relevance feedback ----------------------------------------
    {"runtag": "bm25-rm3", "index": "msmarco-v2-passage", "retriever": "bm25+rm3",
     "params": {"k1": 0.9, "b": 0.4, "fb_terms": 10, "fb_docs": 10,
                "original_query_weight": 0.5}, "family": "prf"},
    {"runtag": "bm25-rm3-wide", "index": "msmarco-v2-passage",
     "retriever": "bm25+rm3",
     "params": {"k1": 0.9, "b": 0.4, "fb_terms": 20, "fb_docs": 20,
                "original_query_weight": 0.3}, "family": "prf"},
    {"runtag": "bm25-rocchio", "index": "msmarco-v2-passage",
     "retriever": "bm25+rocchio",
     "params": {"k1": 0.9, "b": 0.4, "top_fb_terms": 10, "top_fb_docs": 10},
     "family": "prf"},
    # --- doc2query-T5 expanded index --------------------------------------
    {"runtag": "d2q-bm25-k0.9-b0.4", "index": "msmarco-v2-passage-d2q-t5",
     "retriever": "bm25", "params": {"k1": 0.9, "b": 0.4},  "family": "d2q"},
    {"runtag": "d2q-bm25-k1.2-b0.75", "index": "msmarco-v2-passage-d2q-t5",
     "retriever": "bm25", "params": {"k1": 1.2, "b": 0.75}, "family": "d2q"},
    {"runtag": "d2q-bm25-k0.6-b0.4", "index": "msmarco-v2-passage-d2q-t5",
     "retriever": "bm25", "params": {"k1": 0.6, "b": 0.4},  "family": "d2q"},
    {"runtag": "d2q-bm25-rm3", "index": "msmarco-v2-passage-d2q-t5",
     "retriever": "bm25+rm3",
     "params": {"k1": 0.9, "b": 0.4, "fb_terms": 10, "fb_docs": 10,
                "original_query_weight": 0.5}, "family": "d2q"},
    # --- learned sparse / dense (optional: needs the prebuilt index to be
    #     downloadable on Colab; drop them if the download is too big) ------
    {"runtag": "unicoil-0shot", "index": "msmarco-v2-passage-unicoil-0shot",
     "retriever": "impact", "params": {"encoder": "castorini/unicoil-msmarco-passage"},
     "family": "learned-sparse", "optional": True},
    {"runtag": "splade-pp-ed", "index": "msmarco-v2-passage-splade-pp-ed",
     "retriever": "impact",
     "params": {"encoder": "naver/splade-cocondenser-ensembledistil"},
     "family": "learned-sparse", "optional": True},
    {"runtag": "slimr-pp", "index": "msmarco-v2-passage-slimr-pp",
     "retriever": "impact", "params": {}, "family": "learned-sparse",
     "optional": True},
]

PYSERINI_NOTEBOOK_SNIPPET = r'''
# ---- Colab cell: generate the fallback system pool with pyserini ----------
# !pip install pyserini==0.22.1 faiss-cpu
import json, os
from pathlib import Path
from pyserini.search.lucene import LuceneSearcher, LuceneImpactSearcher
import ir_datasets

DATA_DIR = Path(os.environ["DATA_DIR"])
manifest = json.loads((DATA_DIR / "runs" / "dl21" / "pool_manifest.json").read_text())
out_dir  = DATA_DIR / "runs" / "dl21"

ds = ir_datasets.load("msmarco-passage-v2/trec-dl-2021/judged")
queries = {q.query_id: q.text for q in ds.queries_iter()}

for cfg in manifest["configs"]:
    dest = out_dir / f"{cfg['runtag']}.txt"
    if dest.exists():
        print("skip", cfg["runtag"]); continue
    p = cfg["params"]
    try:
        if cfg["retriever"] == "impact":
            s = LuceneImpactSearcher.from_prebuilt_index(
                cfg["index"], p.get("encoder"))
        else:
            s = LuceneSearcher.from_prebuilt_index(cfg["index"])
            s.set_bm25(p.get("k1", 0.9), p.get("b", 0.4))
            if cfg["retriever"] == "bm25+rm3":
                s.set_rm3(p.get("fb_terms", 10), p.get("fb_docs", 10),
                          p.get("original_query_weight", 0.5))
            elif cfg["retriever"] == "bm25+rocchio":
                s.set_rocchio(top_fb_terms=p.get("top_fb_terms", 10),
                              top_fb_docs=p.get("top_fb_docs", 10))
    except Exception as e:                       # optional configs may not exist
        print("SKIP (index unavailable)", cfg["runtag"], type(e).__name__, e)
        continue
    with open(dest, "w") as f:
        for qid, text in sorted(queries.items()):
            for rank, hit in enumerate(s.search(text, k=100), start=1):
                f.write(f"{qid} Q0 {hit.docid} {rank} {hit.score:.6f} "
                        f"{cfg['runtag']}\n")
    print("wrote", dest)
# --------------------------------------------------------------------------
'''


# ═══════════════════════════════════════════════════════════════════════════
# Small helpers
# ═══════════════════════════════════════════════════════════════════════════

def resolve_data_dir(arg: str | None) -> Path:
    d = arg or os.environ.get("DATA_DIR")
    if not d:
        sys.exit("ERROR: pass --data-dir or set $DATA_DIR.")
    return Path(d)


def round_half_up(x: float) -> int:
    """Round-half-up to an int grade in 0..3.

    Python's ``round`` is banker's rounding (round-half-to-EVEN): round(0.5)==0
    and round(1.5)==2, which would make the mean of {0,1} and the mean of {1,2}
    round in opposite directions.  Aggregating paraphrase scores that way would
    put a systematic, grade-dependent bias into every label set, so we round
    half up explicitly.
    """
    if not np.isfinite(x):
        raise ValueError("round_half_up got a non-finite value")
    v = int(math.floor(float(x) + 0.5))
    return max(GRADE_MIN, min(GRADE_MAX, v))


def parse_models(spec: str | None) -> list[str]:
    if not spec:
        return []
    return [s.strip() for s in spec.split(",") if s.strip()]


def guard_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict the frame to dl21 and report what was excluded.

    Batch 5 is defined on ``ALLOWED`` only.  dl22 is the sealed holdout with
    its own pre-registered protocol (scripts/c4_unseal_dl22.py); it is dropped
    here unconditionally, and there is deliberately no --unseal flag on this
    script to turn that off.  The counts are printed rather than silently
    swallowed so an operator can see that the exclusion actually happened.
    """
    if DATASET not in ALLOWED:
        sys.exit(f"ERROR: {DATASET} is not in ALLOWED={sorted(ALLOWED)}.")
    if "dataset" not in df.columns:
        sys.exit("ERROR: parquet has no 'dataset' column.")
    present = df.dataset.astype(str)
    dropped = {ds: int((present == ds).sum())
               for ds in sorted(set(present.unique()) - {DATASET})}
    out = df[present == DATASET]
    if dropped:
        sealed = {k: v for k, v in dropped.items() if k in ("dl22",)}
        print(f"[scope] kept {len(out)} {DATASET} rows; excluded {dropped}"
              + (f"  <- includes the SEALED holdout {sealed}" if sealed else ""))
    if out.empty:
        sys.exit(f"ERROR: no {DATASET} rows in the parquet.")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# TREC run files
# ═══════════════════════════════════════════════════════════════════════════

def load_run_file(path: Path) -> dict[str, list[str]]:
    """Parse a TREC run file -> {qid: [docid, ...]} ordered by rank.

    Format: ``qid Q0 docid rank score tag``.  We order by the *rank* column and
    break ties on (-score, docid) so the ordering is deterministic even when a
    submission emits ranks out of order or repeats a rank.  A (qid, docid)
    repeated within a query keeps its best (smallest) rank.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    rows: dict[str, dict[str, tuple[int, float]]] = defaultdict(dict)
    n_bad = 0
    with opener(path, "rt") as f:                              # type: ignore[operator]
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                if line.strip():
                    n_bad += 1
                continue
            qid, _q0, docid, rank, score = parts[0], parts[1], parts[2], parts[3], parts[4]
            try:
                r, sc = int(rank), float(score)
            except ValueError:
                n_bad += 1
                continue
            prev = rows[qid].get(docid)
            if prev is None or r < prev[0]:
                rows[qid][docid] = (r, sc)
    if n_bad:
        print(f"    [warn] {path.name}: skipped {n_bad} unparseable line(s)")
    return {
        qid: [d for d, _ in sorted(docs.items(), key=lambda t: (t[1][0], -t[1][1], t[0]))]
        for qid, docs in rows.items()
    }


def load_runs(runs_dir: Path) -> dict[str, dict[str, list[str]]]:
    """Load every run file in ``runs_dir`` -> {runtag: run}."""
    if not runs_dir.is_dir():
        sys.exit(f"ERROR: runs dir {runs_dir} does not exist. "
                 f"Run the fetch-runs subcommand first.")
    files = sorted(p for p in runs_dir.iterdir()
                   if p.suffix in (".txt", ".run", ".gz") and p.is_file())
    if not files:
        sys.exit(f"ERROR: no run files (*.txt, *.run, *.gz) in {runs_dir}.")
    runs = {}
    for p in files:
        tag = p.name
        for suf in (".gz", ".txt", ".run"):
            if tag.endswith(suf):
                tag = tag[: -len(suf)]
        run = load_run_file(p)
        if not run:
            print(f"    [warn] {p.name}: no usable lines — skipped")
            continue
        runs[tag] = run
    if not runs:
        sys.exit(f"ERROR: every run file in {runs_dir} was empty/unparseable.")
    return runs


# ═══════════════════════════════════════════════════════════════════════════
# Label construction
# ═══════════════════════════════════════════════════════════════════════════

def judge_labels(d: pd.DataFrame, level: int, mono=None) -> dict:
    """Aggregate one judge's scores at ONE politeness level into pair labels.

    Aggregation = mean over the level's 3 paraphrases AND over runs, then
    round-half-up to an int in 0..3.  ``mono`` (a monotone map from the frozen
    correction rule) is applied PER RECORD, before the mean — that is the same
    order scripts/c4_unseal_dl22.py uses, and it matters: mapping after the
    mean would apply the map to a non-integer.
    """
    ok = d[(d.politeness_level == level) & d.parse_ok & d.score.notna()]
    if ok.empty:
        return {}
    s = ok.score.to_numpy(dtype=float)
    if not np.all(np.isin(s, [0.0, 1.0, 2.0, 3.0])):
        bad = sorted(set(s.tolist()) - {0.0, 1.0, 2.0, 3.0})[:5]
        sys.exit(f"ERROR: parse_ok rows carry scores outside 0..3: {bad}. "
                 f"Scores are never clipped here — fix the parquet.")
    if mono is not None and tuple(mono) != IDENTITY_MAP:
        s = np.asarray(mono, dtype=float)[s.astype(int)]
    acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    for q, doc, v in zip(ok.qid.astype(str), ok.docid.astype(str), s):
        acc[(q, doc)].append(float(v))
    return {k: round_half_up(float(np.mean(v))) for k, v in acc.items()}


def human_labels(qrels: dict) -> dict:
    """dl21 rows of a src.qrels-style dict -> {(qid, docid): grade}."""
    out = {}
    for k, g in qrels.items():
        if len(k) == 3:
            ds, q, doc = k
            if ds != DATASET:
                continue
        else:                                   # legacy 2-tuple, single dataset
            q, doc = k
        out[(str(q), str(doc))] = int(g)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# NDCG@10 on condensed lists
# ═══════════════════════════════════════════════════════════════════════════

def dcg(grades) -> float:
    return float(sum((2.0 ** g - 1.0) / math.log2(i + 2.0)
                     for i, g in enumerate(grades)))


def ndcg_at_k(ranked_docids, labels_q: dict, k: int = TOP_K) -> float:
    """NDCG@k of ONE query under ONE label set, on a CONDENSED list.

    ``ranked_docids`` is the run's ranking for the query; documents the label
    set does not judge are removed first (condensed-list evaluation, Sakai's
    ``condensed lists``), then the top-k of what remains is scored.

    The ideal DCG is taken over ALL documents the label set judges for this
    query (its full judged pool), not only over the retrieved ones — that is
    trec_eval's ``ndcg_cut`` convention and it is what keeps NDCG comparable
    across systems that retrieve different subsets.  A query whose judged pool
    is all-zero has IDCG == 0 and scores 0.0 for every system, so it
    contributes nothing to the between-system comparison.
    """
    condensed = [d for d in ranked_docids if d in labels_q][:k]
    ideal = sorted(labels_q.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    if idcg <= 0.0:
        return 0.0
    return dcg([labels_q[d] for d in condensed]) / idcg


def ndcg_matrix(runs: dict, labelsets: dict, queries: list[str],
                k: int = TOP_K) -> np.ndarray:
    """(n_labelsets, n_systems, n_queries) array of NDCG@k.

    ``labelsets`` maps name -> {(qid, docid): grade}; ``runs`` maps runtag ->
    {qid: [docid...]}.  Each label set is condensed against ITS OWN judged set,
    which is the apples-to-apples rule when the sets differ.  Under the default
    ``--universe intersection`` all three sets judge exactly the same pairs, so
    the condensed lists coincide by construction and any difference in the
    system scores is attributable to the GRADES alone — that is the whole point
    of the restriction.
    """
    names = list(labelsets)
    by_q = {n: defaultdict(dict) for n in names}
    for n in names:
        for (q, doc), g in labelsets[n].items():
            by_q[n][q][doc] = g
    tags = list(runs)
    out = np.zeros((len(names), len(tags), len(queries)), dtype=float)
    for li, n in enumerate(names):
        for si, tag in enumerate(tags):
            run = runs[tag]
            for qi, qid in enumerate(queries):
                lq = by_q[n].get(qid, {})
                out[li, si, qi] = ndcg_at_k(run.get(qid, []), lq, k) if lq else 0.0
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Correlations
# ═══════════════════════════════════════════════════════════════════════════

def kendall_tau_b(a, b) -> float:
    """Kendall's tau-b (ties-corrected). scipy if available, else exact O(n^2).

    System scores can tie (two BM25 variants can score identically on a
    condensed pool), so tau-b, not tau-a.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2:
        return float("nan")
    try:
        from scipy.stats import kendalltau
        t = kendalltau(a, b).correlation
        return float(t)
    except Exception:                                          # noqa: BLE001
        n = a.size
        conc = disc = ta = tb = 0
        for i in range(n):
            for j in range(i + 1, n):
                da, db = a[i] - a[j], b[i] - b[j]
                if da == 0 and db == 0:
                    ta += 1
                    tb += 1
                elif da == 0:
                    ta += 1
                elif db == 0:
                    tb += 1
                elif (da > 0) == (db > 0):
                    conc += 1
                else:
                    disc += 1
        n0 = n * (n - 1) / 2
        den = math.sqrt((n0 - ta) * (n0 - tb))
        return float((conc - disc) / den) if den > 0 else float("nan")


def pearson_r(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def paired_bootstrap(nd: np.ndarray, b: int = 2000, seed: int = 42) -> dict:
    """Paired query bootstrap over delta_tau = tau(CORR,HUMAN) - tau(RAW,HUMAN).

    Queries are the resampling unit (the topics are the sample; the systems are
    fixed).  Every replicate uses the SAME resampled query set for both taus,
    so the two are paired and their difference is not inflated by independent
    query noise.  ``nd`` is (3, n_systems, n_queries) with label-set order
    (HUMAN, RAW, CORR).
    """
    n_q = nd.shape[2]
    rng = np.random.default_rng(seed)
    d_taus, raws, corrs = [], [], []
    for _ in range(b):
        idx = rng.integers(0, n_q, size=n_q)
        means = nd[:, :, idx].mean(axis=2)                     # (3, n_systems)
        tr = kendall_tau_b(means[1], means[0])
        tc = kendall_tau_b(means[2], means[0])
        if not (np.isfinite(tr) and np.isfinite(tc)):
            continue
        raws.append(tr)
        corrs.append(tc)
        d_taus.append(tc - tr)
    if not d_taus:
        return {"n_ok": 0, "ci": (float("nan"), float("nan")),
                "p_gt0": float("nan"), "mean": float("nan")}
    d = np.asarray(d_taus)
    return {
        "n_ok": int(d.size),
        "mean": float(d.mean()),
        "ci": (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))),
        "p_gt0": float((d > 0).mean()),
        "tau_raw_ci": (float(np.percentile(raws, 2.5)),
                       float(np.percentile(raws, 97.5))),
        "tau_corr_ci": (float(np.percentile(corrs, 2.5)),
                        float(np.percentile(corrs, 97.5))),
    }


# ═══════════════════════════════════════════════════════════════════════════
# evaluate — core, injectable (the self-test drives this directly)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_one_judge(model: str, d: pd.DataFrame, rule: dict, human: dict,
                       runs: dict, universe: str = "intersection",
                       b: int = 2000, seed: int = 42, k: int = TOP_K) -> dict:
    """Full per-judge analysis. Returns a result dict (prints nothing)."""
    dial = int(rule["dial_level"])
    mono = tuple(int(x) for x in rule["monotone_map"])

    raw = judge_labels(d, 3, mono=None)
    corr = judge_labels(d, dial, mono=mono)

    if universe == "intersection":
        keys = set(human) & set(raw) & set(corr)
    elif universe == "native":
        keys = None                       # each label set keeps its own pairs
    else:
        raise ValueError(f"unknown universe {universe!r}")

    if keys is not None:
        sets = {"HUMAN": {kk: human[kk] for kk in keys},
                "RAW": {kk: raw[kk] for kk in keys},
                "CORR": {kk: corr[kk] for kk in keys}}
    else:
        sets = {"HUMAN": human, "RAW": raw, "CORR": corr}

    all_keys = set().union(*[set(s) for s in sets.values()])
    queries = sorted({q for q, _ in all_keys})
    if len(queries) < 2:
        return {"model": model, "error": "fewer than 2 queries with labels"}
    if len(runs) < 3:
        return {"model": model,
                "error": f"only {len(runs)} system(s); tau needs >= 3"}

    nd = ndcg_matrix(runs, sets, queries, k=k)
    means = nd.mean(axis=2)                                    # (3, n_systems)

    tau_raw = kendall_tau_b(means[1], means[0])
    tau_corr = kendall_tau_b(means[2], means[0])
    r_raw = pearson_r(means[1], means[0])
    r_corr = pearson_r(means[2], means[0])
    boot = paired_bootstrap(nd, b=b, seed=seed)

    strictly_increasing = all(mono[i] < mono[i + 1] for i in range(3))
    return {
        "model": model,
        "dial_level": dial,
        "monotone_map": list(mono),
        "map_collapses": not strictly_increasing,
        "map_is_identity": mono == IDENTITY_MAP,
        "no_op_expected": strictly_increasing and dial == 3,
        "n_pairs": {n: len(s) for n, s in sets.items()},
        "n_pairs_universe": len(keys) if keys is not None else None,
        "n_dropped": {
            "human_only": len(set(human) - set(raw)) if keys is not None else None,
            "llm_only": len(set(raw) - set(human)) if keys is not None else None,
            "corr_missing": len(set(raw) - set(corr)) if keys is not None else None,
        },
        "n_queries": len(queries),
        "n_systems": len(runs),
        "systems": list(runs),
        "means": means,
        "tau_raw": tau_raw,
        "tau_corr": tau_corr,
        "delta_tau": tau_corr - tau_raw,
        "r_raw": r_raw,
        "r_corr": r_corr,
        "boot": boot,
    }


def print_judge_section(res: dict) -> None:
    print()
    print("-" * 78)
    print(f"  JUDGE: {res['model']}")
    print("-" * 78)
    if "error" in res:
        print(f"  SKIPPED: {res['error']}")
        return
    print(f"  frozen msmarco rule : dial=L{res['dial_level']}  "
          f"map={res['monotone_map']}"
          f"{'  [collapses grades]' if res['map_collapses'] else ''}")
    if res["no_op_expected"]:
        print("  NOTE: the rule is a strictly-increasing map at L3 — NDCG is "
              "invariant to it,")
        print("        so delta_tau MUST be exactly 0. A non-zero value here "
              "is a bug.")
    np_ = res["n_pairs"]
    print(f"  pair universe       : {res['n_pairs_universe']} pairs "
          f"(HUMAN {np_['HUMAN']} / RAW {np_['RAW']} / CORR {np_['CORR']})")
    nd_ = res["n_dropped"]
    if nd_["human_only"] is not None:
        print(f"    dropped: human-only {nd_['human_only']}, "
              f"llm-only {nd_['llm_only']}, "
              f"no CORR label {nd_['corr_missing']}")
    print(f"  queries / systems   : {res['n_queries']} / {res['n_systems']}")
    print()
    print(f"    {'system':<28} {'HUMAN':>8} {'RAW':>8} {'CORR':>8}")
    order = np.argsort(-res["means"][0])
    for si in order:
        m = res["means"]
        print(f"    {res['systems'][si][:28]:<28} "
              f"{m[0, si]:>8.4f} {m[1, si]:>8.4f} {m[2, si]:>8.4f}")
    b = res["boot"]
    print()
    print(f"  tau_b(RAW , HUMAN)  = {res['tau_raw']:+.4f}   "
          f"[boot CI {b.get('tau_raw_ci', (np.nan, np.nan))[0]:+.4f}, "
          f"{b.get('tau_raw_ci', (np.nan, np.nan))[1]:+.4f}]")
    print(f"  tau_b(CORR, HUMAN)  = {res['tau_corr']:+.4f}   "
          f"[boot CI {b.get('tau_corr_ci', (np.nan, np.nan))[0]:+.4f}, "
          f"{b.get('tau_corr_ci', (np.nan, np.nan))[1]:+.4f}]")
    print(f"  delta_tau           = {res['delta_tau']:+.4f}   "
          f"95% CI [{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}]   "
          f"P(delta>0) = {b['p_gt0']:.3f}   (B={b['n_ok']})")
    print(f"  Pearson r (secondary): RAW {res['r_raw']:+.4f}  "
          f"CORR {res['r_corr']:+.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# Subcommand: fetch-runs
# ═══════════════════════════════════════════════════════════════════════════

TREC_INSTRUCTIONS = """
  The official TREC 2021 Deep Learning PASSAGE runs are NOT publicly
  downloadable.  Verified 2026-09-15:

      GET https://trec.nist.gov/results/trec30/deep.passages.input.html
      GET https://trec.nist.gov/results/trec30/
      GET https://trec.nist.gov/results/trec30/deep/input.<runtag>.gz
        -> HTTP 401, www-authenticate: Basic realm="results"

  The whole /results/ tree is HTTP-Basic protected and released to registered
  TREC participants only.  https://trec.nist.gov/data/deep2021.html IS public
  but ships qrels and topics only — no participant runs.

  If the coordinator HAS participant credentials, download the runs by hand
  (this script will never handle credentials) and then:

      python scripts/batch5_downstream.py fetch-runs --from-dir /path/to/runs

  Otherwise use the pyserini fallback pool:

      python scripts/batch5_downstream.py fetch-runs --pyserini-pool
"""


def cmd_fetch_runs(args) -> None:
    data_dir = resolve_data_dir(args.data_dir)
    runs_dir = data_dir / "runs" / DATASET
    runs_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print(f"  BATCH 5 — system pool for {DATASET}  ->  {runs_dir}")
    print("=" * 78)

    if args.pyserini_pool:
        cfgs = [c for c in PYSERINI_POOL
                if args.include_optional or not c.get("optional")]
        manifest = {
            "_note": ("Fallback system pool: the official TREC 2021 DL passage "
                      "runs are login-walled (HTTP 401, Basic realm=results). "
                      "These configs are GENERATED in the notebook; this script "
                      "stays analysis-only."),
            "dataset": DATASET,
            "ir_dataset": IR_DATASET_ID,
            "depth": 100,
            "n_configs": len(cfgs),
            "configs": cfgs,
        }
        path = runs_dir / "pool_manifest.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"  wrote {path}  ({len(cfgs)} configs)")
        fams = defaultdict(int)
        for c in cfgs:
            fams[c["family"]] += 1
        for f, n in sorted(fams.items()):
            print(f"    {f:<16} {n}")
        print("\n  DIVERSITY: a BM25-only pool gives near-identical rankings "
              "and a degenerate tau.")
        print("  Keep the d2q and learned-sparse families in unless the index "
              "download fails.")
        print(PYSERINI_NOTEBOOK_SNIPPET)
        return

    if args.from_dir:
        src = Path(args.from_dir)
        if not src.is_dir():
            sys.exit(f"ERROR: --from-dir {src} is not a directory.")
        n = 0
        for p in sorted(src.iterdir()):
            if p.suffix not in (".txt", ".run", ".gz") or not p.is_file():
                continue
            run = load_run_file(p)
            if not run:
                print(f"  [skip] {p.name}: no usable lines")
                continue
            tag = p.name
            for suf in (".gz", ".txt", ".run"):
                if tag.endswith(suf):
                    tag = tag[: -len(suf)]
            dest = runs_dir / f"{tag}.txt"
            with open(dest, "w") as f:
                for qid in sorted(run):
                    for rank, docid in enumerate(run[qid], start=1):
                        f.write(f"{qid} Q0 {docid} {rank} "
                                f"{1000.0 - rank:.6f} {tag}\n")
            n += 1
            print(f"  normalised {p.name} -> {dest.name} "
                  f"({len(run)} queries)")
        print(f"\n  imported {n} run file(s).")
        if n == 0:
            print(TREC_INSTRUCTIONS)
        return

    # default: report availability
    print(TREC_INSTRUCTIONS)


# ═══════════════════════════════════════════════════════════════════════════
# Subcommand: coverage
# ═══════════════════════════════════════════════════════════════════════════

def cmd_coverage(args) -> None:
    from src.qrels import load_qrels

    data_dir = resolve_data_dir(args.data_dir)
    runs_dir = data_dir / "runs" / DATASET
    runs = load_runs(runs_dir)
    df = guard_dataset(pd.read_parquet(data_dir / "derived" / "judgments.parquet"))
    qrels = load_qrels([DATASET])
    human = human_labels(qrels)
    judged_qids = {q for q, _ in human}

    models = parse_models(args.models)
    known = sorted(df.model_id.unique())
    if models:
        missing = [m for m in models if m not in known]
        if missing:
            sys.exit(f"ERROR: --models not in the parquet: {missing}\n"
                     f"  known: {known}")

    # LLM-judged pairs, overall and per model (any politeness level counts as
    # "collected"; the dial level a rule needs is checked in `evaluate`).
    llm_by_model = {
        m: set(zip(g.qid.astype(str), g.docid.astype(str)))
        for m, g in df.groupby("model_id")
    }
    required = models or list(llm_by_model)
    llm_required = set.intersection(*[llm_by_model[m] for m in required]) \
        if required else set()
    llm_any = set().union(*llm_by_model.values()) if llm_by_model else set()

    print("=" * 78)
    print(f"  BATCH 5 COVERAGE — {DATASET} top-{TOP_K}, {len(runs)} systems")
    print("=" * 78)
    print(f"  judged queries in qrels : {len(judged_qids)} "
          f"(expected {N_QUERIES_EXPECTED})")
    print(f"  'LLM-judged' means judged by ALL of: "
          f"{required if models else '(any model)'}")
    print()
    print(f"  {'run':<30} {'pairs':>7} {'human%':>8} {'llm%':>8} {'both%':>8}")

    union: set[tuple[str, str]] = set()
    for tag in sorted(runs):
        run = runs[tag]
        pairs = set()
        for qid in judged_qids:
            for docid in run.get(qid, [])[:TOP_K]:
                pairs.add((qid, docid))
        union |= pairs
        if not pairs:
            print(f"  {tag[:30]:<30} {0:>7}  (no judged queries in this run)")
            continue
        h = sum(1 for p in pairs if p in human)
        l_ = sum(1 for p in pairs if p in llm_required)
        both = sum(1 for p in pairs if p in human and p in llm_required)
        n = len(pairs)
        print(f"  {tag[:30]:<30} {n:>7} {100*h/n:>7.1f}% "
              f"{100*l_/n:>7.1f}% {100*both/n:>7.1f}%")

    n = len(union)
    if n == 0:
        sys.exit("ERROR: the pool's top-10 contains no pairs on judged queries.")
    h_set = {p for p in union if p in human}
    l_set = {p for p in union if p in llm_required}
    topup = sorted(h_set - l_set)
    neither = union - h_set - l_set
    print()
    print(f"  UNION over all runs      : {n} distinct (qid, docid)")
    print(f"    human-judged           : {len(h_set)} ({100*len(h_set)/n:.1f}%)")
    print(f"    LLM-judged (required)  : {len(l_set)} ({100*len(l_set)/n:.1f}%)")
    print(f"    LLM-judged (any model) : {len(union & llm_any)}")
    print(f"    both                   : {len(h_set & l_set)}")
    print(f"    human only -> TOP-UP   : {len(topup)}  "
          f"(collectable: the human grade exists, so a new LLM label joins "
          f"every label set)")
    print(f"    neither                : {len(neither)}  "
          f"(NOT collectable — no human grade; stays condensed out of ALL "
          f"label sets, for every judge, so it cannot bias the comparison)")

    if args.per_model:
        print()
        print(f"  {'model':<34} {'of union':>10} {'of human':>10}")
        for m in known:
            s = union & llm_by_model[m]
            print(f"  {m[:34]:<34} {100*len(s)/n:>9.1f}% "
                  f"{100*len(s & h_set)/max(1, len(h_set)):>9.1f}%")

    # ---- write the top-up pairs file (same schema as src/build_pairs.py) ---
    out_path = data_dir / "inputs" / f"pairs_{DATASET}_runs_topup.jsonl"
    if not topup:
        print("\n  No top-up needed — nothing written.")
        return
    if args.no_write:
        print(f"\n  --no-write: skipped {out_path}")
    else:
        try:
            import ir_datasets
        except ImportError:
            sys.exit("ERROR: ir_datasets not installed; needed to pull query "
                     "text and passages for the top-up file.")
        ds = ir_datasets.load(IR_DATASET_ID)
        queries = {q.query_id: q.text for q in ds.queries_iter()}
        docstore = ds.docs_store()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with open(out_path, "w") as f:
            for qid, docid in topup:                    # already sorted
                if qid not in queries:
                    sys.exit(f"ERROR: qid {qid!r} missing from {IR_DATASET_ID} "
                             f"queries.")
                try:
                    doc = docstore.get(docid)
                except Exception as e:                  # noqa: BLE001
                    sys.exit(f"ERROR: docid {docid!r} (qid {qid!r}) not in the "
                             f"{IR_DATASET_ID} docstore: {type(e).__name__}: {e}")
                f.write(json.dumps({
                    "dataset": DATASET,
                    "qid": qid,
                    "docid": docid,
                    "query": queries[qid],
                    "passage": doc.text,
                    "rel": human[(qid, docid)],
                }, ensure_ascii=False) + "\n")
                written += 1
        print(f"\n  wrote {written} top-up pairs -> {out_path}")

    # ---- cost estimate ----------------------------------------------------
    n_pairs = len(topup)
    print()
    print(f"  COST ESTIMATE for {n_pairs} top-up pairs "
          f"(~{TOK_IN} in + {TOK_OUT} out tokens per call, {args.runs} run(s))")
    print(f"  {'scenario':<26} {'calls':>9} " +
          " ".join(f"{k:>12}" for k in PRICES))
    for label, n_variants in (("L3 only (3 variants)", 3),
                              ("L3 + dial (6)", 6),
                              ("full grid (15)", 15)):
        calls = n_pairs * n_variants * args.runs
        cells = []
        for _k, (pin, pout) in PRICES.items():
            usd = calls * (TOK_IN * pin + TOK_OUT * pout) / 1e6
            cells.append(f"${usd:>11.2f}")
        print(f"  {label:<26} {calls:>9} " + " ".join(cells))
    print("  (a judge only needs its OWN dial level + L3; 'full grid' is the "
          "ceiling if the\n   coordinator wants the top-up to support every "
          "tone level.)")


# ═══════════════════════════════════════════════════════════════════════════
# Subcommand: evaluate
# ═══════════════════════════════════════════════════════════════════════════

def cmd_evaluate(args) -> None:
    from src.qrels import load_qrels

    data_dir = resolve_data_dir(args.data_dir)
    runs = load_runs(data_dir / "runs" / DATASET)
    df = guard_dataset(pd.read_parquet(data_dir / "derived" / "judgments.parquet"))
    rules_path = Path(args.rules) if args.rules else \
        data_dir / "derived" / "correction_rules.json"
    if not rules_path.exists():
        sys.exit(f"ERROR: correction rules not found at {rules_path}. "
                 f"Run scripts/batch4_correction.py (rules stage) first.")
    rules_all = json.loads(rules_path.read_text())
    qrels = load_qrels([DATASET])
    human = human_labels(qrels)

    models = parse_models(args.models) or sorted(df.model_id.unique())

    print("=" * 78)
    print("  BATCH 5 / gap-B — DOWNSTREAM SYSTEM RANKING "
          f"({DATASET}, NDCG@{TOP_K})")
    print("=" * 78)
    print(f"  rules: {rules_path}  (git {rules_all.get('git_hash')})")
    print(f"  systems: {len(runs)}   universe: {args.universe}   "
          f"bootstrap: B={args.b} seed={args.seed}")
    print("  FRAMING: this measures whether corrected labels reproduce human "
          "evaluation")
    print("  OUTCOMES (system ranking). It is NOT a judgment-quality claim.")

    results = []
    for model in models:
        d = df[df.model_id == model]
        if d.empty:
            print(f"\n  {model}: NO {DATASET} ROWS — skipped")
            continue
        rule = (rules_all.get("rules", {}).get(model, {}) or {}).get("msmarco")
        if rule is None:
            print(f"\n  {model}: no frozen msmarco rule — skipped")
            continue
        res = evaluate_one_judge(model, d, rule, human, runs,
                                 universe=args.universe, b=args.b,
                                 seed=args.seed, k=TOP_K)
        print_judge_section(res)
        results.append(res)

    ok = [r for r in results if "error" not in r]
    if not ok:
        sys.exit("\nERROR: no judge produced a result.")

    # ---- summary ----------------------------------------------------------
    print()
    print("=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print(f"  {'judge':<30} {'dial':>5} {'tau_raw':>9} {'tau_corr':>9} "
          f"{'d_tau':>8} {'P(d>0)':>8} {'r_raw':>8} {'r_corr':>8}")
    for r in ok:
        print(f"  {r['model'][:30]:<30} L{r['dial_level']:<4} "
              f"{r['tau_raw']:>+9.4f} {r['tau_corr']:>+9.4f} "
              f"{r['delta_tau']:>+8.4f} {r['boot']['p_gt0']:>8.3f} "
              f"{r['r_raw']:>+8.4f} {r['r_corr']:>+8.4f}")
    noop = [r for r in ok if r["no_op_expected"]]
    if noop:
        bad = [r["model"] for r in noop if abs(r["delta_tau"]) > 1e-12]
        print(f"\n  no-op control: {len(noop)} judge(s) have a strictly-"
              f"increasing map at L3;")
        print(f"    delta_tau must be exactly 0 for them — "
              f"{'OK' if not bad else 'VIOLATED by ' + ', '.join(bad)}")

    # ---- LaTeX ------------------------------------------------------------
    tex_dir = data_dir / "derived" / "tex"
    rows = []
    for r in ok:
        b = r["boot"]
        rows.append(
            f"{esc(r['model'])} & L{r['dial_level']} & "
            f"{r['n_systems']} & {r['n_queries']} & {r['n_pairs']['HUMAN']} & "
            f"${r['tau_raw']:+.4f}$ & ${r['tau_corr']:+.4f}$ & "
            f"${r['delta_tau']:+.4f}$ & "
            f"$[{b['ci'][0]:+.3f}, {b['ci'][1]:+.3f}]$ & "
            f"{b['p_gt0']:.3f} & ${r['r_raw']:+.3f}$ & ${r['r_corr']:+.3f}$ \\\\"
        )
    write_tex(
        tex_dir, "batch5_system_ranking.tex", rows,
        f"Batch 5 (gap-B): downstream system ranking on {DATASET}. "
        f"NDCG@{TOP_K} on condensed lists over the pair universe judged by "
        f"BOTH the human qrels and the LLM judge; system score = mean over "
        f"queries. tau is Kendall tau-b against the HUMAN system ranking; "
        f"delta_tau CI and P(delta>0) from a paired query bootstrap "
        f"(B={args.b}, seed={args.seed}). Columns: judge, dial level, "
        f"n systems, n queries, n pairs, tau raw, tau corrected, delta tau, "
        f"95\\% CI, P(delta>0), Pearson r raw, Pearson r corrected. "
        f"This is downstream UTILITY (does the correction reproduce human "
        f"evaluation outcomes), not judgment quality.")


# ═══════════════════════════════════════════════════════════════════════════
# Subcommand: selftest  (synthetic stub drive — no network, no ir_datasets)
# ═══════════════════════════════════════════════════════════════════════════

def _stub_frame(model: str, labels_l3: dict, labels_dial: dict,
                dial: int) -> pd.DataFrame:
    """Build a fake parquet frame: 3 paraphrases x 1 run at L3 and at `dial`."""
    rows = []
    levels = [(3, labels_l3)] if dial == 3 else [(3, labels_l3),
                                                 (dial, labels_dial)]
    for level, labs in levels:
        for (qid, docid), score in labs.items():
            for para in ("a", "b", "c"):
                rows.append({
                    "model_id": model, "prompt_id": f"L{level}_{para}",
                    "politeness_level": level, "dataset": DATASET,
                    "qid": qid, "docid": docid, "run": 1,
                    "score": float(score), "parse_ok": True,
                })
    return pd.DataFrame(rows)


def cmd_selftest(args) -> None:
    print("=" * 78)
    print("  BATCH 5 SELF-TEST (synthetic; no Drive, no network)")
    print("=" * 78)
    failures = []

    def check(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" +
              (f"  — {detail}" if detail else ""))
        if not cond:
            failures.append(name)

    # -- 1. rounding -------------------------------------------------------
    check("round_half_up(0.5)==1 (banker's would give 0)",
          round_half_up(0.5) == 1, f"got {round_half_up(0.5)}")
    check("round_half_up(1.5)==2", round_half_up(1.5) == 2)
    check("round_half_up(2.5)==3 (banker's would give 2)",
          round_half_up(2.5) == 3, f"got {round_half_up(2.5)}")
    check("round_half_up clips high", round_half_up(9.9) == 3)
    check("round_half_up clips low", round_half_up(-2.0) == 0)
    check("mean(1,2)=1.5 -> 2", round_half_up(np.mean([1, 2])) == 2)

    # -- 2. condensed-list NDCG -------------------------------------------
    labels_q = {"d1": 3, "d3": 1}          # d2, d4 unjudged -> condensed out
    ranked = ["d2", "d1", "d4", "d3"]
    # condensed = [d1, d3] -> DCG = 7/1 + 1/log2(3)
    want = (7.0 / 1.0 + 1.0 / math.log2(3.0)) / (7.0 / 1.0 + 1.0 / math.log2(3.0))
    got = ndcg_at_k(ranked, labels_q, k=10)
    check("condensation removes unjudged docs", abs(got - want) < 1e-12,
          f"ndcg={got:.6f}")
    check("perfect condensed ranking scores 1.0",
          abs(ndcg_at_k(["d1", "d3"], labels_q, 10) - 1.0) < 1e-12)
    worse = ndcg_at_k(["d3", "d1"], labels_q, 10)
    check("inverted ranking scores lower", worse < 1.0, f"{worse:.4f}")
    check("all-zero label pool -> 0.0",
          ndcg_at_k(["a"], {"a": 0, "b": 0}, 10) == 0.0)
    check("ideal pool includes unretrieved judged docs",
          ndcg_at_k(["d3"], {"d1": 3, "d3": 1}, 10) < 1.0)

    # -- 3. strictly-increasing map at L3 is a NO-OP for NDCG --------------
    # Fixture: human grades in {0,1,2} (see section 4 for why the top grade is
    # left unused), 8 docs x 8 queries, 6 systems with genuinely different
    # orderings so the system ranking has resolution instead of tying out.
    qids = [f"q{i}" for i in range(8)]
    docs = [f"d{j}" for j in range(8)]
    rng = np.random.default_rng(7)
    human = {}
    for q in qids:
        for d in docs:
            human[(q, d)] = int(rng.integers(0, 3))
    perms = {
        "sysA": (0, 1, 2, 3, 4, 5, 6, 7),
        "sysB": (7, 6, 5, 4, 3, 2, 1, 0),
        "sysC": (3, 0, 5, 1, 7, 2, 6, 4),
        "sysD": (1, 2, 0, 4, 3, 6, 5, 7),
        "sysE": (2, 5, 1, 6, 0, 7, 3, 4),
        "sysF": (6, 3, 7, 0, 5, 1, 4, 2),
    }
    runs_stub = {tag: {q: [docs[i] for i in p] for q in qids}
                 for tag, p in perms.items()}
    raw_lab = dict(human)                    # judge == human, for the control
    df_noop = _stub_frame("stub/noop", raw_lab, raw_lab, 3)
    res = evaluate_one_judge("stub/noop", df_noop,
                             {"dial_level": 3, "monotone_map": [0, 1, 2, 3]},
                             human, runs_stub, b=200, seed=42)
    check("identity rule at L3 -> delta_tau exactly 0",
          res["delta_tau"] == 0.0, f"delta_tau={res['delta_tau']!r}")
    check("identity rule flagged as no-op control", res["no_op_expected"])
    check("perfect judge -> tau_raw == 1", abs(res["tau_raw"] - 1.0) < 1e-12,
          f"tau_raw={res['tau_raw']:.4f}")

    # -- 4. corrected labels agree with human MORE than raw ----------------
    # Constructed so the correction genuinely recovers information:
    #
    #   L3 (RAW)  : the judge is LENIENT and CONFLATES — human {0,1,2} is
    #               reported as {1,2,2}, so "not relevant" and "marginal" are
    #               indistinguishable.  That is a non-injective corruption, so
    #               it really does move the NDCG-based system ranking.
    #   dial (L2) : the judge is merely SHIFTED (+1), which is injective — the
    #               grade ordering is intact, only the operating point is off.
    #   frozen map: (0,0,1,2) undoes the shift EXACTLY: 1->0, 2->1, 3->2.
    #
    # Human grades are capped at 2 precisely so the +1 shift cannot saturate at
    # 3; if it did, the dial level would itself lose the top grade and no map
    # could recover it (that was the bug this test was written to catch).
    # Result: CORR reproduces HUMAN exactly (tau_corr == 1) while RAW cannot.
    conflate = {0: 1, 1: 2, 2: 2}
    l3_lab = {k: conflate[v] for k, v in human.items()}
    dial_lab = {k: v + 1 for k, v in human.items()}
    df_corr = _stub_frame("stub/lenient", l3_lab, dial_lab, 2)
    res2 = evaluate_one_judge("stub/lenient", df_corr,
                              {"dial_level": 2, "monotone_map": [0, 0, 1, 2]},
                              human, runs_stub, b=2000, seed=42)
    check("collapsing rule is not treated as a no-op",
          not res2["no_op_expected"] and res2["map_collapses"])
    check("tau_corr > tau_raw", res2["tau_corr"] > res2["tau_raw"],
          f"tau_raw={res2['tau_raw']:+.4f} tau_corr={res2['tau_corr']:+.4f}")
    check("delta_tau > 0", res2["delta_tau"] > 0,
          f"delta_tau={res2['delta_tau']:+.4f}")
    check("corrected recovers human exactly -> tau_corr == 1",
          abs(res2["tau_corr"] - 1.0) < 1e-12, f"{res2['tau_corr']:+.6f}")
    check("raw conflation really does degrade the ranking",
          res2["tau_raw"] < 1.0, f"tau_raw={res2['tau_raw']:+.4f}")
    check("bootstrap CI brackets the point estimate",
          res2["boot"]["ci"][0] <= res2["delta_tau"] + 1e-9,
          f"ci={res2['boot']['ci']}")
    check("bootstrap produced replicates", res2["boot"]["n_ok"] > 0,
          f"n_ok={res2['boot']['n_ok']}")
    check("P(delta>0) in [0,1]", 0.0 <= res2["boot"]["p_gt0"] <= 1.0,
          f"P={res2['boot']['p_gt0']:.3f}")

    # -- 5. universe restriction -------------------------------------------
    partial = {k: v for i, (k, v) in enumerate(l3_lab.items()) if i % 2 == 0}
    df_part = _stub_frame("stub/partial", partial,
                          {k: human[k] + 1 for k in partial}, 2)
    res3 = evaluate_one_judge("stub/partial", df_part,
                              {"dial_level": 2, "monotone_map": [0, 0, 1, 2]},
                              human, runs_stub, b=100, seed=42)
    check("universe = human AND llm intersection",
          res3["n_pairs_universe"] == len(partial) ==
          res3["n_pairs"]["HUMAN"] == res3["n_pairs"]["CORR"],
          f"universe={res3['n_pairs_universe']} partial={len(partial)}")
    check("human-only pairs are dropped and counted",
          res3["n_dropped"]["human_only"] == len(human) - len(partial))

    # -- 6. tau-b fallback matches scipy -----------------------------------
    a = np.array([1.0, 2.0, 2.0, 3.0, 5.0])
    bb = np.array([1.0, 3.0, 2.0, 3.0, 4.0])
    try:
        from scipy.stats import kendalltau
        check("kendall_tau_b matches scipy",
              abs(kendall_tau_b(a, bb) - kendalltau(a, bb).correlation) < 1e-12)
    except ImportError:
        print("  [skip] scipy not installed — tau-b cross-check skipped")

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

    common = dict(help="DATA_DIR (default: $DATA_DIR)")

    p = sub.add_parser("fetch-runs", help="acquire / register the system pool")
    p.add_argument("--data-dir", default=None, **common)
    p.add_argument("--pyserini-pool", action="store_true",
                   help="write the fallback pool manifest + notebook snippet")
    p.add_argument("--include-optional", action="store_true",
                   help="include learned-sparse configs in the manifest")
    p.add_argument("--from-dir", default=None,
                   help="import TREC run files already downloaded by hand")
    p.set_defaults(func=cmd_fetch_runs)

    p = sub.add_parser("coverage", help="judged-fraction of the pool's top-10")
    p.add_argument("--data-dir", default=None, **common)
    p.add_argument("--models", default=None,
                   help="comma-separated model_ids that must ALL have judged a "
                        "pair for it to count as LLM-judged (default: any)")
    p.add_argument("--per-model", action="store_true",
                   help="also print a per-model coverage breakdown")
    p.add_argument("--runs", type=int, default=1,
                   help="runs per (pair, variant) for the cost estimate")
    p.add_argument("--no-write", action="store_true",
                   help="compute coverage but do not write the top-up JSONL")
    p.set_defaults(func=cmd_coverage)

    p = sub.add_parser("evaluate", help="the core system-ranking analysis")
    p.add_argument("--data-dir", default=None, **common)
    p.add_argument("--models", default=None,
                   help="comma-separated judge model_ids (default: all)")
    p.add_argument("--rules", default=None,
                   help="correction_rules.json (default: "
                        "$DATA_DIR/derived/correction_rules.json)")
    p.add_argument("--universe", choices=["intersection", "native"],
                   default="intersection",
                   help="'intersection' (default, the apples-to-apples rule) "
                        "restricts all three label sets to pairs judged by BOTH "
                        "human and LLM; 'native' is a DIAGNOSTIC that lets each "
                        "set condense on its own pool and therefore confounds "
                        "grades with coverage")
    p.add_argument("--b", type=int, default=2000, help="bootstrap replicates")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("selftest", help="synthetic stub drive (offline)")
    p.set_defaults(func=cmd_selftest)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
