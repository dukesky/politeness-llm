"""Parse raw API logs (append-only JSONL) into tidy analysis tables.

    python -m src.parse --data-dir /content/drive/MyDrive/llm-ranker-tone-data

Never edits raw/. Re-run freely whenever parsing logic changes.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_raw(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for fp in sorted(raw_dir.rglob("*.jsonl")):
        with open(fp) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn final line from a crash
                u = r.get("usage") or {}
                details = u.get("completion_tokens_details") or {}
                rows.append({
                    "model_id": r["model_id"],
                    "provider": r.get("provider"),
                    "prompt_id": r["prompt_id"],
                    "politeness_level": r["politeness_level"],
                    "dataset": r["dataset"],
                    "qid": str(r["qid"]),
                    "docid": str(r["docid"]),
                    "run": r["run"],
                    "score": r["score"],
                    "parse_ok": r["parse_ok"],
                    "tokens_in": u.get("prompt_tokens"),
                    "tokens_out": u.get("completion_tokens"),
                    "tokens_reasoning": details.get("reasoning_tokens"),
                    "cost_usd_reported": u.get("cost"),
                    "latency_s": r.get("latency_s"),
                    "code_version": r.get("code_version"),
                })
    df = pd.DataFrame(rows)
    # Dedupe on the experiment key, keep first occurrence
    df = df.drop_duplicates(
        subset=["model_id", "prompt_id", "qid", "docid", "run"], keep="first"
    )
    return df


def add_cost(df: pd.DataFrame, models_yaml: dict) -> pd.DataFrame:
    price = {
        m["model_id"]: (m.get("price_per_m_input") or 0,
                        m.get("price_per_m_output") or 0)
        for m in models_yaml["models"]
    }
    def cost(row):
        pi, po = price.get(row.model_id, (0, 0))
        ti = row.tokens_in or 0
        to = row.tokens_out or 0
        return ti / 1e6 * pi + to / 1e6 * po
    df["cost_usd"] = df.apply(cost, axis=1)
    return df


def main():
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    df = load_raw(data_dir / "raw")
    models_yaml = yaml.safe_load(open("config/models.yaml"))
    df = add_cost(df, models_yaml)

    out = data_dir / "derived"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "judgments.parquet", index=False)
    print(f"Wrote {len(df)} rows -> {out / 'judgments.parquet'}")
    print(df.groupby(['model_id', 'politeness_level'])['parse_ok'].mean().round(3))


if __name__ == "__main__":
    main()
