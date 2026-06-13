"""Step C validation script: compute kappa and agreement metrics for one model.

!! Run ONLY after Step B (prediction committed to paper/PREDICTIONS.md).

Usage:
    python scripts/validate_model.py --model openai/gpt-5.4-mini --data-dir $DATA_DIR
    python scripts/validate_model.py --model anthropic/claude-haiku-4.5
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

DATASETS = {
    "dl19": "msmarco-passage/trec-dl-2019/judged",
    "dl20": "msmarco-passage/trec-dl-2020/judged",
}


def load_qrels(datasets_present: set) -> dict:
    try:
        import ir_datasets
    except ImportError:
        sys.exit("ERROR: ir_datasets not installed. Run: pip install ir_datasets")
    qrels = {}
    for ds_name in datasets_present:
        if ds_name not in DATASETS:
            continue
        ds = ir_datasets.load(DATASETS[ds_name])
        for j in ds.qrels_iter():
            qrels[(j.query_id, j.doc_id)] = int(j.relevance)
    return qrels


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir or os.environ.get("DATA_DIR", "./data"))
    parquet = data_dir / "derived" / "judgments.parquet"
    if not parquet.exists():
        sys.exit(f"ERROR: {parquet} not found. Run src/parse.py first.")

    df_all = pd.read_parquet(parquet)
    df = df_all[df_all.model_id == args.model].copy()
    if df.empty:
        sys.exit(f"ERROR: no rows for model '{args.model}' in {parquet}.")

    # add repo root to path so src.metrics is importable
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.metrics import attach_qrels, agreement_table, between_vs_within_level

    qrels = load_qrels(set(df.dataset.dropna().unique()))
    df = attach_qrels(df, qrels)

    table = agreement_table(df)

    # ── Per-level κ summary (averaged across paraphrases) ────────────────────
    level_summary = (
        table.groupby("politeness_level")
        .agg(
            kappa_mean=("kappa", "mean"),
            kappa_std=("kappa", "std"),
            fail_rate=("fail_rate", "mean"),
            cost_per_1k=("cost_per_1k", "mean"),
        )
        .reset_index()
    )
    l3_kappa = level_summary.loc[
        level_summary.politeness_level == 3, "kappa_mean"
    ].values[0]

    print()
    print("=" * 72)
    print(f"VALIDATION RESULTS — {args.model}")
    print("=" * 72)
    print()
    print("── Per-level κ (linear weighted, mean ± std across 3 paraphrases) ──")
    print(f"{'Level':<7} {'κ mean':>8} {'κ std':>7} {'Δκ vs L3':>10} "
          f"{'fail%':>7} {'$/1k':>8}")
    print("-" * 55)
    for _, row in level_summary.iterrows():
        lvl = int(row.politeness_level)
        delta_k = row.kappa_mean - l3_kappa if lvl != 3 else 0.0
        print(f"{'L'+str(lvl):<7} {row.kappa_mean:>8.4f} {row.kappa_std:>7.4f} "
              f"{delta_k:>+10.4f} {row.fail_rate:>7.1%} {row.cost_per_1k:>8.4f}")

    # ── Per-prompt detail ─────────────────────────────────────────────────────
    print()
    print("── Per-prompt κ detail ──────────────────────────────────────────────")
    print(f"{'prompt_id':<12} {'level':>6} {'κ':>8} {'exact_acc':>10} {'n':>7}")
    print("-" * 48)
    for _, row in table.sort_values(["politeness_level", "prompt_id"]).iterrows():
        print(f"{row.prompt_id:<12} {int(row.politeness_level):>6} "
              f"{row.kappa:>8.4f} {row.exact_acc:>10.3f} {int(row.n):>7}")

    # ── Between vs within level variance ─────────────────────────────────────
    print()
    print("── Between-level vs within-level κ std ─────────────────────────────")
    bvw = between_vs_within_level(table)
    for _, row in bvw.iterrows():
        print(f"  between_std={row.between_level_std:.4f}  "
              f"within_std={row.within_level_std:.4f}  "
              f"ratio={row.ratio:.2f}x")

    # ── Paste-ready κ values for PREDICTIONS.md ──────────────────────────────
    kmap = dict(zip(level_summary.politeness_level, level_summary.kappa_mean))
    print()
    print("── For PREDICTIONS.md (回填开箱结果) ───────────────────────────────")
    print(f"  κ(L1)={kmap.get(1, float('nan')):.4f}, "
          f"κ(L3)={kmap.get(3, float('nan')):.4f}, "
          f"κ(L5)={kmap.get(5, float('nan')):.4f}，"
          f"κ(L2)={kmap.get(2, float('nan')):.4f}, "
          f"κ(L4)={kmap.get(4, float('nan')):.4f}")
    for lvl, label in [(1, "L1"), (5, "L5")]:
        dk = kmap.get(lvl, float("nan")) - l3_kappa
        tie = abs(dk) < 0.005
        direction = "tie" if tie else ("↑" if dk > 0 else "↓")
        print(f"  {label}: Δκ={dk:+.4f} → {direction}")
    for lvl, label in [(2, "L2"), (4, "L4")]:
        dk = kmap.get(lvl, float("nan")) - l3_kappa
        tie = abs(dk) < 0.005
        direction = "tie" if tie else ("↑" if dk > 0 else "↓")
        print(f"  {label} (次终点): Δκ={dk:+.4f} → {direction}")


if __name__ == "__main__":
    main()
