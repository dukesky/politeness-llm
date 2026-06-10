"""Core metrics: agreement with qrels, stability across tone, cost.

Used from notebooks/02_analysis.ipynb. qrels format: dict[(qid, docid)] = grade.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


# ---------------------------------------------------------------------------
# Agreement with human qrels
# ---------------------------------------------------------------------------

def attach_qrels(df: pd.DataFrame, qrels: dict) -> pd.DataFrame:
    df = df.copy()
    df["human"] = [qrels.get((q, d)) for q, d in zip(df.qid, df.docid)]
    return df.dropna(subset=["human"])


def agreement_table(df: pd.DataFrame) -> pd.DataFrame:
    """Cohen's kappa (linear weights) per (model, prompt) against human grades.
    Only rows with parse_ok and a non-null score are scored; failure rate is
    reported separately so it isn't silently absorbed."""
    rows = []
    for (m, p, lvl), g in df.groupby(["model_id", "prompt_id", "politeness_level"]):
        ok = g[g.parse_ok & g.score.notna()]
        rows.append({
            "model_id": m,
            "prompt_id": p,
            "politeness_level": lvl,
            "n": len(g),
            "fail_rate": 1 - len(ok) / len(g) if len(g) else np.nan,
            "kappa": cohen_kappa_score(ok.human.astype(int), ok.score.astype(int),
                                       weights="linear") if len(ok) > 10 else np.nan,
            "exact_acc": (ok.human.astype(int) == ok.score.astype(int)).mean()
                          if len(ok) else np.nan,
            "mean_tokens_in": g.tokens_in.mean(),
            "cost_per_1k": g.cost_usd.mean() * 1000,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stability across tone
# ---------------------------------------------------------------------------

def flip_rate(df: pd.DataFrame, prompt_a: str, prompt_b: str) -> float:
    """Share of (qid, docid) pairs whose label changes between two prompts
    (same model handled by caller via pre-filtering; run 1 only)."""
    a = df[(df.prompt_id == prompt_a) & (df.run == 1)].set_index(["qid", "docid"]).score
    b = df[(df.prompt_id == prompt_b) & (df.run == 1)].set_index(["qid", "docid"]).score
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    return float((j.a != j.b).mean())


def between_vs_within_level(table: pd.DataFrame) -> pd.DataFrame:
    """Key design check: is variance BETWEEN politeness levels larger than
    paraphrase variance WITHIN a level? Reports per-model std of kappa."""
    rows = []
    for m, g in table.groupby("model_id"):
        within = g.groupby("politeness_level").kappa.std().mean()
        between = g.groupby("politeness_level").kappa.mean().std()
        rows.append({"model_id": m,
                     "within_level_std": within,
                     "between_level_std": between,
                     "ratio": between / within if within else np.nan})
    return pd.DataFrame(rows)
