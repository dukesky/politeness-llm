"""Build the pairs file (query-doc inputs) for collection.

Strategy: take judged pairs from qrels intersected with BM25 top-k, so every
LLM judgment can be compared against a human label.

    python -m src.build_pairs --dataset dl19 --run-file inputs/bm25_dl19.txt \
        --top-k 50 --out data/pairs_dl19.jsonl

Requires: pip install ir_datasets
TREC run file format: qid Q0 docid rank score tag
"""

import argparse
import json

import ir_datasets

DATASETS = {
    "dl19": "msmarco-passage/trec-dl-2019/judged",
    "dl20": "msmarco-passage/trec-dl-2020/judged",
    # add BEIR subsets here in W2, e.g. "scifact": "beir/scifact/test"
}


def load_run(path, top_k):
    keep = {}
    with open(path) as f:
        for line in f:
            qid, _, docid, rank, *_ = line.split()
            if int(rank) <= top_k:
                keep.setdefault(qid, set()).add(docid)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--run-file", required=True)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ds = ir_datasets.load(DATASETS[args.dataset])
    queries = {q.query_id: q.text for q in ds.queries_iter()}
    judged = {(j.query_id, j.doc_id) for j in ds.qrels_iter()}
    run = load_run(args.run_file, args.top_k)
    docstore = ds.docs_store()

    n = 0
    with open(args.out, "w") as out:
        for qid, docids in run.items():
            if qid not in queries:
                continue
            for docid in docids:
                if (qid, docid) not in judged:
                    continue  # only pairs with a human label
                doc = docstore.get(docid)
                out.write(json.dumps({
                    "dataset": args.dataset,
                    "qid": qid,
                    "docid": docid,
                    "query": queries[qid],
                    "passage": doc.text,
                }, ensure_ascii=False) + "\n")
                n += 1
    print(f"Wrote {n} judged pairs -> {args.out}")


if __name__ == "__main__":
    main()
