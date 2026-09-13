"""Build the pairs file (query-doc inputs) for collection.

Strategy: take judged pairs from qrels intersected with a candidate ranking, so
every LLM judgment can be compared against a human label.

    # v1 behaviour (unchanged): our own BM25 run file
    python -m src.build_pairs --dataset dl19 --run-file inputs/bm25_dl19.txt \
        --top-k 50 --out data/pairs_dl19.jsonl

    # round 2: official scoreddocs ranking shipped with ir_datasets
    python -m src.build_pairs --dataset dl21 --source scoreddocs \
        --top-k 50 --max-per-query 30 --out data/pairs_dl21.jsonl

    # round 2: qrels-only (ANTIQUE has no candidate ranking)
    python -m src.build_pairs --dataset antique --source qrels \
        --max-per-query 30 --out data/pairs_antique.jsonl

Protocol note (round 1 vs round 2)
----------------------------------
* dl19 / dl20 (round 1) were built from **our own BM25 run file** intersected
  with qrels (``--source runfile``, the default). The *selection* of pairs on
  that path is unchanged, but the output is **not** byte-identical to v1:
  every record now carries ``rel``, and rows are emitted in sorted (qid,
  docid) order rather than dict/set iteration order. Re-running this script
  for dl19/dl20 therefore will NOT reproduce the frozen v1 pairs files — the
  **frozen v1 files stay authoritative** for round-1 data; regenerate only if
  you intend to re-freeze (new SHA256).
* dl21 / dl22 (round 2) use the **official candidate ranking (scoreddocs)**
  that ir_datasets ships with TREC DL: top-k by score per query, then
  intersected with qrels. We do not run our own retriever for v2, so the
  candidate pool is the organisers' and is reproducible from ir_datasets
  alone. If the ``/judged`` subset carries no scoreddocs, the parent dataset
  (``/judged`` stripped) supplies them and we restrict to judged qids.
* antique (round 2) has no usable candidate ranking, so it is **qrels-only**:
  every judged (qid, docid) pair, with a stratified per-query cap
  (``--max-per-query``) to keep the grid affordable.

Relevance grades
----------------
Each output record carries ``rel``: the **raw** qrels grade as ir_datasets
reports it. TREC DL is 0-3; **ANTIQUE is 1-4** — grades are stored raw and are
deliberately NOT remapped here. Any collapsing to a binary/4-point scale
belongs in analysis, not in the frozen pairs file.

Determinism
-----------
Queries are sorted by qid and documents by docid before any sampling or
writing (on **all** ``--source`` paths, runfile included), and downsampling
uses ``random.Random(--seed)``. After writing we
print the pair count, the per-grade histogram and the SHA256 of the output
file — that hash is the freeze fingerprint recorded alongside the run.

Requires: pip install ir_datasets
TREC run file format: qid Q0 docid rank score tag
"""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict

import ir_datasets

DATASETS = {
    "dl19": "msmarco-passage/trec-dl-2019/judged",
    "dl20": "msmarco-passage/trec-dl-2020/judged",
    "dl21": "msmarco-passage-v2/trec-dl-2021/judged",
    "dl22": "msmarco-passage-v2/trec-dl-2022/judged",
    "antique": "antique/test",
}


def load_run(path, top_k):
    keep = {}
    with open(path) as f:
        for line in f:
            qid, _, docid, rank, *_ = line.split()
            if int(rank) <= top_k:
                keep.setdefault(qid, set()).add(docid)
    return keep


def load_scoreddocs(ds, ds_name, top_k, judged_qids):
    """Top-k docids per query from the official candidate ranking.

    Ranking is by score descending (docid ascending as a deterministic
    tie-break), NOT by the iteration order of scoreddocs_iter. If the
    ``/judged`` subset has no scoreddocs, fall back to the parent dataset and
    restrict to judged qids.
    """
    source_ds = ds
    if not source_ds.has_scoreddocs():
        parent = ds_name[: -len("/judged")] if ds_name.endswith("/judged") else ds_name
        if parent == ds_name:
            raise SystemExit(f"[ERROR] {ds_name} has no scoreddocs and no parent to fall back to")
        print(f"[info] {ds_name} has no scoreddocs; using parent {parent}")
        source_ds = ir_datasets.load(parent)
        if not source_ds.has_scoreddocs():
            raise SystemExit(f"[ERROR] neither {ds_name} nor {parent} provides scoreddocs")

    # Assumption: the /judged variant shares the parent's doc collection, so a
    # docid from the parent's scoreddocs resolves in the /judged docstore.
    best = defaultdict(dict)  # qid -> docid -> highest score seen
    n_dupe = 0
    for sd in source_ds.scoreddocs_iter():
        if sd.query_id not in judged_qids:
            continue
        prev = best[sd.query_id].get(sd.doc_id)
        if prev is None:
            best[sd.query_id][sd.doc_id] = float(sd.score)
        else:
            n_dupe += 1
            best[sd.query_id][sd.doc_id] = max(prev, float(sd.score))
    if n_dupe:
        print(f"[warn] scoreddocs: {n_dupe} repeated (qid, docid) entries "
              f"deduped before top-k (kept highest score)")

    keep = {}
    for qid, scores in best.items():
        scored = sorted(scores.items(), key=lambda t: (-t[1], t[0]))
        keep[qid] = {docid for docid, _ in scored[:top_k]}
    return keep


def stratified_cap(docids, grades, n, rng):
    """Keep <= n docids, allocating proportionally across qrels grades.

    docids must already be sorted; rng consumption is therefore deterministic.
    """
    if len(docids) <= n:
        return list(docids)

    by_grade = defaultdict(list)
    for d in docids:
        by_grade[grades[d]].append(d)

    total = len(docids)
    alloc, remainders = {}, []
    for g in sorted(by_grade):
        exact = n * len(by_grade[g]) / total
        alloc[g] = int(exact)
        remainders.append((exact - int(exact), g))

    # distribute the leftover slots by largest fractional part (grade as tie-break)
    leftover = n - sum(alloc.values())
    for _, g in sorted(remainders, key=lambda t: (-t[0], t[1]))[:leftover]:
        alloc[g] += 1

    kept = []
    for g in sorted(by_grade):
        k = min(alloc[g], len(by_grade[g]))
        if k:
            kept.extend(rng.sample(by_grade[g], k))
    return sorted(kept)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--source", choices=["runfile", "scoreddocs", "qrels"],
                    default="runfile",
                    help="candidate pool: our run file (v1 default), the official "
                         "scoreddocs ranking, or all judged pairs")
    ap.add_argument("--run-file", help="required for --source runfile")
    ap.add_argument("--top-k", type=int, default=50,
                    help="top-k per query for runfile/scoreddocs (ignored for qrels)")
    ap.add_argument("--max-per-query", type=int, default=None,
                    help="cap judged pairs per query, stratified by qrels grade")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.source == "runfile" and not args.run_file:
        ap.error("--run-file is required with --source runfile")

    ds_name = DATASETS[args.dataset]
    ds = ir_datasets.load(ds_name)
    queries = {q.query_id: q.text for q in ds.queries_iter()}

    # (qid, docid) -> raw relevance grade. Duplicate qrels entries are counted;
    # a duplicate with a *different* grade resolves to max() (deterministic).
    grades = defaultdict(dict)
    n_qrels_dupe = 0
    n_qrels_conflict = 0
    for j in ds.qrels_iter():
        g = int(j.relevance)
        prev = grades[j.query_id].get(j.doc_id)
        if prev is None:
            grades[j.query_id][j.doc_id] = g
            continue
        n_qrels_dupe += 1
        if prev != g:
            n_qrels_conflict += 1
            grades[j.query_id][j.doc_id] = max(prev, g)
    if n_qrels_conflict:
        print(f"[warn] qrels: {n_qrels_conflict} duplicate (qid, docid) entries "
              f"with CONFLICTING grades — kept max grade")

    if args.source == "runfile":
        cand = load_run(args.run_file, args.top_k)
    elif args.source == "scoreddocs":
        cand = load_scoreddocs(ds, ds_name, args.top_k, set(grades))
    else:  # qrels
        cand = {qid: set(docs) for qid, docs in grades.items()}

    docstore = ds.docs_store()
    rng = random.Random(args.seed)

    n = 0
    hist = Counter()
    with open(args.out, "w") as out:
        for qid in sorted(cand):                       # deterministic query order
            if qid not in queries:
                continue
            qgrades = grades.get(qid, {})
            docids = sorted(d for d in cand[qid] if d in qgrades)  # judged only
            if args.max_per_query is not None:
                docids = stratified_cap(docids, qgrades, args.max_per_query, rng)
            for docid in docids:
                try:
                    doc = docstore.get(docid)
                except Exception as e:  # noqa: BLE001
                    raise SystemExit(
                        f"[ERROR] docid {docid!r} (qid {qid!r}) not found in the "
                        f"docstore of {ds_name}: {type(e).__name__}: {e}"
                    )
                out.write(json.dumps({
                    "dataset": args.dataset,
                    "qid": qid,
                    "docid": docid,
                    "query": queries[qid],
                    "passage": doc.text,
                    "rel": qgrades[docid],
                }, ensure_ascii=False) + "\n")
                hist[qgrades[docid]] += 1
                n += 1

    print(f"Wrote {n} judged pairs -> {args.out}")
    print(f"  source={args.source} top_k={args.top_k} "
          f"max_per_query={args.max_per_query} seed={args.seed}")
    print(f"  qrels duplicate (qid, docid) entries: {n_qrels_dupe} "
          f"({n_qrels_conflict} with conflicting grades, resolved to max)")
    print("  grade histogram (raw qrels grades):")
    for g in sorted(hist):
        print(f"    rel={g}: {hist[g]}")
    print(f"  SHA256: {sha256_file(args.out)}")


if __name__ == "__main__":
    main()
