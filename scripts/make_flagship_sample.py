"""Generate deterministic 40% stratified flagship sample files.

Stratification: per-qid, 40% of docids sampled, minimum 1 per qid so no
query is entirely dropped. seed=42 guarantees identical output on every run.

Usage (run once on Colab, outputs go to inputs/):
    python scripts/make_flagship_sample.py --data-dir $DATA_DIR

Outputs:
    {data_dir}/inputs/pairs_dl19_flagship40.jsonl
    {data_dir}/inputs/pairs_dl20_flagship40.jsonl
"""

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

DATASETS = ["dl19", "dl20"]
SAMPLE_RATIO = 0.40
SEED = 42
MIN_PER_QID = 1


def stratified_sample(pairs: list, ratio: float, seed: int, min_per_qid: int) -> list:
    rng = random.Random(seed)

    by_qid = defaultdict(list)
    for p in pairs:
        by_qid[p["qid"]].append(p)

    sampled = []
    for qid in sorted(by_qid):              # sorted → deterministic rng consumption order
        docs = sorted(by_qid[qid], key=lambda p: p["docid"])  # sort within qid too
        n = max(min_per_qid, round(len(docs) * ratio))
        n = min(n, len(docs))
        sampled.extend(rng.sample(docs, n))

    return sampled


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True,
                    help="root data directory (contains inputs/ subfolder)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    inputs_dir = data_dir / "inputs"

    for ds in DATASETS:
        src = inputs_dir / f"pairs_{ds}.jsonl"
        dst = inputs_dir / f"pairs_{ds}_flagship40.jsonl"

        if not src.exists():
            print(f"[SKIP] {src} not found")
            continue

        with open(src) as f:
            pairs = [json.loads(line) for line in f if line.strip()]

        sampled = stratified_sample(pairs, SAMPLE_RATIO, SEED, MIN_PER_QID)

        # write (idempotent — same seed → same output)
        with open(dst, "w") as f:
            for p in sampled:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

        # ── stats ────────────────────────────────────────────────────────
        qids_full    = sorted({p["qid"] for p in pairs})
        qids_sampled = sorted({p["qid"] for p in sampled})
        per_qid      = [sum(1 for p in sampled if p["qid"] == q) for q in qids_sampled]

        print(f"\n{'='*60}")
        print(f"Dataset: {ds}")
        print(f"  全量:  {len(pairs):>6} pairs, {len(qids_full):>3} qids")
        print(f"  抽样:  {len(sampled):>6} pairs, {len(qids_sampled):>3} qids"
              f"  (须 == {len(qids_full)})")
        print(f"  比例:  {len(sampled)/len(pairs):.1%}  (目标 40%)")
        print(f"  每 qid passage 数:  "
              f"min={min(per_qid)}  "
              f"median={statistics.median(per_qid):.0f}  "
              f"max={max(per_qid)}")
        if len(qids_sampled) != len(qids_full):
            missing = set(qids_full) - set(qids_sampled)
            print(f"  ⚠️  缺失 qid: {missing}")
        print(f"  输出:  {dst}")


if __name__ == "__main__":
    main()
