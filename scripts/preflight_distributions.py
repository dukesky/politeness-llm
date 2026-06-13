"""Preflight distribution check for semi-blind prediction registration.

PURPOSE: Step A of the per-model validation workflow.
Run this script AFTER parse and BEFORE any kappa/consistency metrics.

!! HARD RULE: Do NOT run src/metrics.py or the kappa validation script
!! for this model before running this script and completing Step B
!! (committing the prediction to paper/PREDICTIONS.md).

This script intentionally does NOT compute kappa or any inter-rater
agreement metric. It only reports score distributions and the inputs
needed to apply the prediction rules in paper/PREDICTIONS.md.

Usage (from repo root on Colab):
    python scripts/preflight_distributions.py \\
        --model deepseek/deepseek-v4-flash \\
        --data-dir /content/drive/MyDrive/llm-ranker-tone-data

Output: prints a registration block ready to paste into PREDICTIONS.md.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

DATASETS = {
    "dl19": "msmarco-passage/trec-dl-2019/judged",
    "dl20": "msmarco-passage/trec-dl-2020/judged",
}

LEVELS = [1, 2, 3, 4, 5]
SCORES = [0, 1, 2, 3]
NO_CALL_THRESHOLD = 0.02
DELTA_MIN = 0.01   # |Δ| < this → no-call regardless of D


def load_qrels(datasets_present: set[str]) -> dict:
    """Load qrels from ir_datasets for all datasets in the parquet."""
    try:
        import ir_datasets
    except ImportError:
        print("WARNING: ir_datasets not installed; qrels mean will be 'N/A'.",
              file=sys.stderr)
        return {}

    qrels = {}
    for ds_name in datasets_present:
        if ds_name not in DATASETS:
            continue
        ds = ir_datasets.load(DATASETS[ds_name])
        for j in ds.qrels_iter():
            qrels[(j.query_id, j.doc_id)] = int(j.relevance)
    return qrels


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def predict_direction(delta: float, d: float) -> str:
    """Apply prediction rules from PREDICTIONS.md §2."""
    if abs(d) < NO_CALL_THRESHOLD:
        return "no call  (|D|<0.02)"
    if abs(delta) < DELTA_MIN:
        return "no call  (|Δ|<0.01, model ~aligned with human)"
    if delta > 0 and d < 0:
        return "↑  (model偏宽, 该档漂严 → 更贴近人类工作点)"
    if delta > 0 and d > 0:
        return "↓  (model偏宽, 该档漂宽 → 远离人类工作点)"
    if delta < 0 and d > 0:
        return "↑  (model偏严, 该档漂宽 → 更贴近人类工作点)"
    if delta < 0 and d < 0:
        return "↓  (model偏严, 该档漂严 → 远离人类工作点)"
    return "no call"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="model_id as in models.yaml, e.g. deepseek/deepseek-v4-flash")
    ap.add_argument("--data-dir",
                    default=None,
                    help="path to DATA_DIR (contains derived/judgments.parquet)")
    args = ap.parse_args()

    import os
    data_dir = Path(args.data_dir or os.environ.get("DATA_DIR", "./data"))
    parquet = data_dir / "derived" / "judgments.parquet"

    if not parquet.exists():
        sys.exit(f"ERROR: {parquet} not found. Run src/parse.py first.")

    df_all = pd.read_parquet(parquet)
    df = df_all[df_all.model_id == args.model].copy()
    if df.empty:
        sys.exit(f"ERROR: no rows for model '{args.model}' in {parquet}.")

    # ── Load qrels ────────────────────────────────────────────────────────────
    datasets_present = set(df.dataset.dropna().unique())
    qrels = load_qrels(datasets_present)

    if qrels:
        df["human"] = [qrels.get((str(q), str(d))) for q, d in zip(df.qid, df.docid)]
        df_with_human = df.dropna(subset=["human"])
        qrels_mean = df_with_human["human"].mean()
    else:
        qrels_mean = None

    # ── Per-level stats ───────────────────────────────────────────────────────
    level_stats = {}
    for lvl in LEVELS:
        sub = df[df.politeness_level == lvl]
        ok = sub[sub.parse_ok & sub.score.notna()]
        counts = ok.score.value_counts().reindex(SCORES, fill_value=0)
        total = len(sub)
        level_stats[lvl] = {
            "n_total": total,
            "n_ok": len(ok),
            "parse_ok_rate": len(ok) / total if total else float("nan"),
            "mean_score": ok.score.mean() if len(ok) else float("nan"),
            "dist": {s: counts[s] / len(ok) if len(ok) else float("nan")
                     for s in SCORES},
        }

    l3_mean = level_stats[3]["mean_score"]
    delta = (l3_mean - qrels_mean) if qrels_mean is not None else float("nan")

    d = {}
    for lvl in [1, 2, 4, 5]:
        d[lvl] = level_stats[lvl]["mean_score"] - l3_mean

    # ── Print summary table ───────────────────────────────────────────────────
    print()
    print("=" * 72)
    print(f"PREFLIGHT DISTRIBUTIONS — {args.model}")
    print(f"Parquet: {parquet}")
    print(f"Git hash: {git_hash()}")
    print("=" * 72)
    print()
    print(f"{'Level':<6} {'N':>7} {'parse_ok':>9} {'mean':>7}  "
          f"{'s=0':>6} {'s=1':>6} {'s=2':>6} {'s=3':>6}  {'D(ℓ)':>8}")
    print("-" * 72)
    for lvl in LEVELS:
        s = level_stats[lvl]
        dist = s["dist"]
        d_val = (s["mean_score"] - l3_mean) if lvl != 3 else 0.0
        print(f"{'L'+str(lvl):<6} {s['n_total']:>7} {s['parse_ok_rate']:>9.1%} "
              f"{s['mean_score']:>7.4f}  "
              f"{dist[0]:>6.1%} {dist[1]:>6.1%} {dist[2]:>6.1%} {dist[3]:>6.1%}  "
              f"{d_val:>+8.4f}")
    print()
    if qrels_mean is not None:
        print(f"qrels mean (matched pairs): {qrels_mean:.4f}")
        print(f"Δ = model_L3_mean − qrels_mean = {l3_mean:.4f} − {qrels_mean:.4f} "
              f"= {delta:+.4f}")
    else:
        print("qrels mean: N/A (ir_datasets not available)")
        print(f"Δ: N/A  (model L3 mean = {l3_mean:.4f})")
    print()

    # ── Predictions ──────────────────────────────────────────────────────────
    print("Predictions (apply rules from paper/PREDICTIONS.md §2):")
    for lvl in [1, 5]:
        tag = "(主终点)"
        print(f"  κ(L{lvl}) vs κ(L3): {predict_direction(delta, d[lvl])}  {tag}")
    for lvl in [2, 4]:
        tag = "(次终点)"
        print(f"  κ(L{lvl}) vs κ(L3): {predict_direction(delta, d[lvl])}  {tag}")

    # ── Paste-ready registration block ───────────────────────────────────────
    import datetime
    today = datetime.date.today().isoformat()
    hash_ = git_hash()

    delta_str = f"{delta:+.4f}" if qrels_mean is not None else "N/A"
    l3_str = f"{l3_mean:.4f}"
    qr_str = f"{qrels_mean:.4f}" if qrels_mean is not None else "N/A"

    print()
    print("─" * 72)
    print("PASTE INTO paper/PREDICTIONS.md:")
    print("─" * 72)
    print(f"""
### {args.model} — {today} — git hash {hash_}
- 盲态: blind / non-blind（说明原因）
- Δ = {delta_str}（模型 L3 均分 {l3_str} − qrels 均分 {qr_str}）
- D(L1)={d[1]:+.4f}, D(L2)={d[2]:+.4f}, D(L4)={d[4]:+.4f}, D(L5)={d[5]:+.4f}
- 预测：
  - κ(L1) vs κ(L3): {predict_direction(delta, d[1])}
  - κ(L5) vs κ(L3): {predict_direction(delta, d[5])}  ← 主终点
  - κ(L2) vs κ(L3): {predict_direction(delta, d[2])}  ← 次终点
  - κ(L4) vs κ(L3): {predict_direction(delta, d[4])}  ← 次终点
- 开箱结果（Step C 后回填）：
  - κ(L1)=?, κ(L3)=?, κ(L5)=?，κ(L2)=?, κ(L4)=?
  - 命中：L1 ?，L5 ?
  - 次终点：L2 ?，L4 ?
""")


if __name__ == "__main__":
    main()
