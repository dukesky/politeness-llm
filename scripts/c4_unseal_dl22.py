"""C4 — pre-registered DL22 unsealing (the ONLY analysis allowed to open dl22).

Protocol: this script may be run ONLY after the C4 registration block
("# C4 预注册 — DL22 解封") exists in paper/PREDICTIONS.md (committed), which
records (a) the blind tone-direction predictions from the dl22 distribution
preflight and (b) the FROZEN correction rules (msmarco family, fitted on
dl19+dl20+dl21, never on dl22) with their expected corrected-vs-raw kappa
gains.  The script refuses to start if that marker is absent.

Framing: any kappa gain from the correction is MEASUREMENT-ARTIFACT REMOVAL
(operating-point alignment with the reference annotators), not a judgment-
quality improvement.

Outputs per model on dl22 (flagships restricted to the frozen 40% pairs):
  raw per-level kappa L1..L5 (tone unboxing, scored against the registered
  predictions), Delta_raw at L3, then the frozen rule applied:
  kappa(dial level, raw scores), kappa(map applied at dial level), and
  Delta_after for the corrected pipeline.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.qrels import load_qrels, parse_datasets, assert_not_sealed  # noqa: E402
from validate_model import per_variant_kappa  # noqa: E402

MODELS = [
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v4.1-flash",
    "google/gemini-3.5-flash",
    "google/gemini-3.8-flash",
    "anthropic/claude-opus-5",
    "openai/gpt-5.6-sol",
]
FLAGSHIPS = {"anthropic/claude-opus-5", "openai/gpt-5.6-sol"}
C4_MARKER = "C4 预注册 — DL22 解封"


def require_registration() -> None:
    p = REPO / "paper" / "PREDICTIONS.md"
    if C4_MARKER not in p.read_text():
        sys.exit(
            "ERROR: paper/PREDICTIONS.md does not contain the C4 registration "
            f"block ('{C4_MARKER}').\nCommit the registration FIRST — this "
            "script is the unsealing step and must never precede it."
        )


def level_kappa(df_lvl: pd.DataFrame, qrels: dict) -> float:
    """Paraphrase-mean kappa for the rows of ONE politeness level."""
    if df_lvl.empty:
        return float("nan")
    table = per_variant_kappa(df_lvl, qrels)
    return float(table["kappa"].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--unseal-dl22", action="store_true")
    args = ap.parse_args()
    import os
    data_dir = Path(args.data_dir or os.environ["DATA_DIR"])

    require_registration()
    # The one sanctioned unsealing: still demands the explicit flag.
    assert_not_sealed(parse_datasets("dl22"), unseal=args.unseal_dl22)

    df = pd.read_parquet(data_dir / "derived" / "judgments.parquet")
    df = df[df.dataset == "dl22"]
    if df.empty:
        sys.exit("ERROR: no dl22 rows in the parquet.")

    frozen = set()
    fp = data_dir / "inputs" / "pairs_dl22_flagship40.jsonl"
    with open(fp) as f:
        for line in f:
            p = json.loads(line)
            frozen.add((str(p["qid"]), str(p["docid"])))

    rules_all = json.loads(
        (data_dir / "derived" / "correction_rules.json").read_text())
    qrels = load_qrels(["dl22"])

    print("=" * 78)
    print("C4 — DL22 UNSEALING (pre-registered; rules git "
          f"{rules_all.get('git_hash')})")
    print("=" * 78)

    for model in MODELS:
        d = df[df.model_id == model]
        if model in FLAGSHIPS:
            keys = list(zip(d.qid.astype(str), d.docid.astype(str)))
            mask = np.fromiter((k in frozen for k in keys), dtype=bool,
                               count=len(keys))
            d = d[mask]
        if d.empty:
            print(f"\n{model}: NO ROWS — skipped")
            continue

        rule = rules_all["rules"][model]["msmarco"]
        dial = int(rule["dial_level"])
        mono = [int(x) for x in rule["monotone_map"]]

        ks = {lvl: level_kappa(d[d.politeness_level == lvl], qrels)
              for lvl in (1, 2, 3, 4, 5)}

        ok = d[d.parse_ok & d.score.notna()]
        matched = d[[ (r.dataset, str(r.qid), str(r.docid)) in qrels
                      for r in d.itertuples() ]]
        human_mean = float(np.mean([
            qrels[(r.dataset, str(r.qid), str(r.docid))]
            for r in matched.itertuples()]))
        l3_mean = float(ok[ok.politeness_level == 3].score.mean())
        delta_raw = l3_mean - human_mean

        d_dial = d[d.politeness_level == dial]
        k_dial = ks[dial]
        d_corr = d_dial.copy()
        sc = d_corr.score.to_numpy(dtype=float)
        val = np.isfinite(sc)
        mapped = sc.copy()
        mapped[val] = np.array(mono, dtype=float)[sc[val].astype(int)]
        d_corr = d_corr.assign(score=mapped)
        k_corr = level_kappa(d_corr, qrels)
        ok_c = d_corr[d_corr.parse_ok & d_corr.score.notna()]
        delta_after = float(ok_c.score.mean()) - human_mean

        print(f"\n{model}")
        print("  raw kappa: " + "  ".join(
            f"L{l}={ks[l]:.4f}" for l in (1, 2, 3, 4, 5)))
        print(f"  tone unboxing: "
              f"dL1={ks[1]-ks[3]:+.4f} dL2={ks[2]-ks[3]:+.4f} "
              f"dL4={ks[4]-ks[3]:+.4f} dL5={ks[5]-ks[3]:+.4f}")
        print(f"  Delta_raw(L3) = {delta_raw:+.4f}")
        print(f"  rule: dial=L{dial} map={mono}")
        print(f"  kappa(dial raw)      = {k_dial:.4f}  "
              f"(vs L3 {ks[3]:.4f}: {k_dial-ks[3]:+.4f})")
        print(f"  kappa(dial + map)    = {k_corr:.4f}  "
              f"(vs L3 raw: {k_corr-ks[3]:+.4f}; expected "
              f"{rule.get('expected_dkappa_from_crossfit', float('nan')):+.4f})")
        print(f"  |Delta| before/after = {abs(delta_raw):.4f} / "
              f"{abs(delta_after):.4f}")

    print("\nDone. Backfill paper/PREDICTIONS.md C4 block with these numbers.")


if __name__ == "__main__":
    main()
