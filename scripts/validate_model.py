"""Step C validation script: compute kappa and agreement metrics for one model.

!! Run ONLY after Step B (prediction committed to paper/PREDICTIONS.md).

Usage:
    # cheap tier (full pairs, round 1 = default)
    python scripts/validate_model.py --model openai/gpt-5.4-mini --data-dir $DATA_DIR

    # flagship tier (filter to frozen 40% sample)
    python scripts/validate_model.py --model openai/gpt-5.5 --data-dir $DATA_DIR \\
        --flagship-pairs $DATA_DIR/inputs/pairs_dl19_flagship40.jsonl \\
                         $DATA_DIR/inputs/pairs_dl20_flagship40.jsonl

    # round 2 (dl22 is SEALED — it is refused unless --unseal-dl22 is given)
    python scripts/validate_model.py --model deepseek/deepseek-v4.1-flash \\
        --datasets dl21,antique --data-dir $DATA_DIR
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.qrels import assert_not_sealed, load_qrels, parse_datasets  # noqa: E402


# ---------------------------------------------------------------------------
# Authoritative kappa path — imported by scripts/batch1r_stats.py so that the
# statistical hardening reproduces EXACTLY what Step C reported.
# ---------------------------------------------------------------------------

def per_variant_kappa(df: pd.DataFrame, qrels: dict) -> pd.DataFrame:
    """Per (model, prompt_id, level) linear-weighted Cohen's kappa vs human."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.metrics import attach_qrels, agreement_table
    return agreement_table(attach_qrels(df, qrels))


def level_summary_from_table(table: pd.DataFrame) -> pd.DataFrame:
    """Per-level kappa = MEAN over the 3 paraphrases of that level.

    This paraphrase-mean is the AUTHORITATIVE per-level kappa used in
    paper/PREDICTIONS.md.  A pooled recompute (concatenating the 3
    paraphrases' rows and taking one kappa) gives materially different
    numbers — e.g. DeepSeek D = -0.148 pooled vs -0.073 authoritative — and
    must not be used.
    """
    if table.empty:
        sys.exit("ERROR: no scorable (parse_ok, qrels-matched) rows — nothing "
                 "to compute kappa on. Check the dataset filter and, for "
                 "flagship models, the --flagship-pairs files.")
    present = set(int(x) for x in table.politeness_level.unique())
    missing = sorted({1, 2, 3, 4, 5} - present)
    if missing:
        sys.exit(f"ERROR: politeness level(s) {missing} have no rows "
                 f"(present: {sorted(present)}). Every level must be present; "
                 f"a partial grid cannot be compared against L3.")
    return (
        table.groupby("politeness_level")
        .agg(
            kappa_mean=("kappa", "mean"),
            kappa_std=("kappa", "std"),
            fail_rate=("fail_rate", "mean"),
            cost_per_1k=("cost_per_1k", "mean"),
        )
        .reset_index()
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--datasets", default="dl19,dl20",
                    help="comma-separated dataset names (default: dl19,dl20 = round 1)")
    ap.add_argument("--unseal-dl22", action="store_true",
                    help="ONLY after round-2 correction predictions are committed: "
                         "permit kappa computation on the sealed dl22 holdout")
    ap.add_argument("--flagship-pairs", nargs="+", default=None, metavar="JSONL",
                    help="one or more frozen pairs files; restricts analysis to "
                         "those (qid, docid) pairs only (use for flagship models)")
    args = ap.parse_args()

    datasets = parse_datasets(args.datasets)
    # SEAL CHECK — must happen before ANY kappa computation below.
    assert_not_sealed(datasets, args.unseal_dl22)

    data_dir = Path(args.data_dir or os.environ.get("DATA_DIR", "./data"))
    parquet = data_dir / "derived" / "judgments.parquet"
    if not parquet.exists():
        sys.exit(f"ERROR: {parquet} not found. Run src/parse.py first.")

    df_all = pd.read_parquet(parquet)
    df = df_all[df_all.model_id == args.model].copy()
    if df.empty:
        sys.exit(f"ERROR: no rows for model '{args.model}' in {parquet}.")

    before_ds = len(df)
    df = df[df.dataset.isin(datasets)].copy()
    if df.empty:
        sys.exit(f"ERROR: no rows for model '{args.model}' on datasets "
                 f"{datasets} in {parquet}.")
    if len(df) != before_ds:
        print(f"[dataset filter] {before_ds} → {len(df)} rows "
              f"(datasets: {','.join(datasets)})")

    if args.flagship_pairs:
        import json
        frozen = set()
        for fp in args.flagship_pairs:
            with open(fp) as f:
                for line in f:
                    p = json.loads(line)
                    frozen.add((str(p["qid"]), str(p["docid"])))
        before = len(df)
        df = df[df.apply(lambda r: (str(r.qid), str(r.docid)) in frozen, axis=1)]
        print(f"[flagship filter] {before} → {len(df)} rows "
              f"({len(frozen)} frozen pairs × 15 variants × runs)")

    # add repo root to path so src.metrics is importable
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.metrics import between_vs_within_level

    qrels = load_qrels(datasets)

    table = per_variant_kappa(df, qrels)

    # ── Per-level κ summary (averaged across paraphrases) ────────────────────
    level_summary = level_summary_from_table(table)
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
