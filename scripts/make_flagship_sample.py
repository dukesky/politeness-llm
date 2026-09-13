"""Generate deterministic stratified flagship sample files (40% by default).

Stratification: per-qid, 40% of docids sampled, minimum 1 per qid so no
query is entirely dropped. seed=42 guarantees identical output on every run.

Usage (run once on Colab, outputs go to inputs/):
    # round 1, unchanged: dl19 + dl20 under {data_dir}/inputs/
    python scripts/make_flagship_sample.py --data-dir $DATA_DIR

    # round 2: any dataset(s) following the same inputs/pairs_{ds}.jsonl layout
    python scripts/make_flagship_sample.py --data-dir $DATA_DIR \
        --datasets dl21,dl22,antique

    # or point at an arbitrary pairs file
    python scripts/make_flagship_sample.py --pairs-file /path/pairs_x.jsonl \
        --out /path/pairs_x_flagship40.jsonl

Outputs (default layout):
    {data_dir}/inputs/pairs_{ds}_flagship40.jsonl
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
    ap.add_argument("--data-dir",
                    help="root data directory (contains inputs/ subfolder)")
    ap.add_argument("--datasets", default=",".join(DATASETS),
                    help=f"comma-separated dataset names under inputs/ "
                         f"(default: {','.join(DATASETS)})")
    ap.add_argument("--pairs-file",
                    help="explicit pairs jsonl; overrides --data-dir/--datasets")
    ap.add_argument("--out",
                    help="output path for --pairs-file (default: <src>_flagship40.jsonl)")
    ap.add_argument("--ratio", type=float, default=SAMPLE_RATIO)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    if args.pairs_file:
        src = Path(args.pairs_file)
        dst = Path(args.out) if args.out else src.with_name(
            f"{src.stem}_flagship{int(round(args.ratio * 100))}{src.suffix}")
        jobs = [(src.stem, src, dst)]
    else:
        if not args.data_dir:
            ap.error("one of --data-dir or --pairs-file is required")
        inputs_dir = Path(args.data_dir) / "inputs"
        names = [s.strip() for s in args.datasets.split(",") if s.strip()]
        suffix = f"_flagship{int(round(args.ratio * 100))}"
        jobs = [(ds,
                 inputs_dir / f"pairs_{ds}.jsonl",
                 inputs_dir / f"pairs_{ds}{suffix}.jsonl")
                for ds in names]

    for ds, src, dst in jobs:
        if not src.exists():
            print(f"[SKIP] {src} not found")
            continue

        with open(src) as f:
            pairs = [json.loads(line) for line in f if line.strip()]

        sampled = stratified_sample(pairs, args.ratio, args.seed, MIN_PER_QID)

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
        print(f"  比例:  {len(sampled)/len(pairs):.1%}  (目标 {args.ratio:.0%})")
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
