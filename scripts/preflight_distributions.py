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
    # round 1 (default, unchanged v1 behaviour)
    python scripts/preflight_distributions.py \\
        --model deepseek/deepseek-v4-flash \\
        --data-dir /content/drive/MyDrive/llm-ranker-tone-data

    # round 2
    python scripts/preflight_distributions.py \\
        --model deepseek/deepseek-v4.1-flash --datasets dl21,dl22,antique \\
        --data-dir $DATA_DIR

dl22 note: distribution preflight on dl22 is ALLOWED — this stage is blind.
Only kappa / agreement metrics are sealed (see src/qrels.py).

Output: prints a registration block ready to paste into PREDICTIONS.md.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.qrels import SEALED, parse_datasets  # noqa: E402
from src.qrels import load_qrels as _load_qrels  # noqa: E402

LEVELS = [1, 2, 3, 4, 5]
SCORES = [0, 1, 2, 3]
NO_CALL_THRESHOLD = 0.02
DELTA_MIN = 0.01   # |Δ| < this → no-call regardless of D


def load_qrels(datasets: list) -> dict:
    """Dataset-keyed qrels on the common 0-3 scale (src/qrels.py).

    Missing ir_datasets is non-fatal here: this stage still prints the score
    distributions, it just cannot compute Δ.
    """
    try:
        import ir_datasets  # noqa: F401
    except ImportError:
        print("WARNING: ir_datasets not installed; qrels mean will be 'N/A'.",
              file=sys.stderr)
        return {}
    return _load_qrels(list(datasets))


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
    ap.add_argument("--datasets", default="dl19,dl20",
                    help="comma-separated dataset names to analyse "
                         "(default: dl19,dl20 = round 1). Round 2: dl21,dl22,antique")
    ap.add_argument("--flagship-pairs", nargs="+", default=None, metavar="JSONL",
                    help="frozen pairs files; restricts analysis to those (qid,docid) "
                         "pairs only (use for flagship models)")
    args = ap.parse_args()

    import os
    data_dir = Path(args.data_dir or os.environ.get("DATA_DIR", "./data"))
    parquet = data_dir / "derived" / "judgments.parquet"

    if not parquet.exists():
        sys.exit(f"ERROR: {parquet} not found. Run src/parse.py first.")

    datasets = parse_datasets(args.datasets)

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

    if set(datasets) & set(SEALED):
        print()
        print("!! REMINDER: dl22 is a SEALED holdout. This distribution "
              "preflight is BLIND and allowed.")
        print("!! Do NOT run scripts/validate_model.py (kappa) on dl22 until "
              "the round-2 correction")
        print("!! predictions are committed to paper/PREDICTIONS.md.")
        print()

    if args.flagship_pairs:
        frozen = set()
        for fp in args.flagship_pairs:
            with open(fp) as f:
                for line in f:
                    p = json.loads(line)
                    frozen.add((str(p["qid"]), str(p["docid"])))
        before = len(df)
        df = df[df.apply(lambda r: (str(r.qid), str(r.docid)) in frozen, axis=1)]
        print(f"[flagship filter] {before} → {len(df)} rows ({len(frozen)} frozen pairs)")

    # ── Load qrels ────────────────────────────────────────────────────────────
    qrels = load_qrels(datasets)

    if qrels:
        df["human"] = [qrels.get((str(ds), str(q), str(d)))
                       for ds, q, d in zip(df.dataset, df.qid, df.docid)]
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
    print(f"Datasets: {','.join(datasets)}")
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
- 数据集: {','.join(datasets)}
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
