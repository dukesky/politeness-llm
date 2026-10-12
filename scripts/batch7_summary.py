#!/usr/bin/env python3
"""Batch 7 (distillation) — post-unboxing summary and pre-registration
adjudication.  Reads the evaluate/compare JSONs written by
scripts/batch7_distill.py and prints (1) a per-(arm, judge, label set)
table of seed-averaged endpoints, (2) the paired compares, (3) the
B7-P1..P8 verdicts under the rules registered in paper/PREDICTIONS.md
(direction by 3-seed mean point estimate; tie bands NDCG/MAP/F1 0.005,
offset / relevant-rate ratio 0.02; ties count neither hit nor miss;
90% CI excluding 0 is reported separately as "significant" and never
changes a verdict; main arm minilm-raw is scored, msmarco-ce is the
robustness arm).  Pure reporting: no analysis quantity is recomputed
from raw data, every number is read from the JSON files.

usage: python scripts/batch7_summary.py --eval-root $DATA_DIR/derived/batch7/eval
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ARMS = ("minilm-raw", "msmarco-ce")
MAIN_ARM = "minilm-raw"
SETS = ("HUMAN", "RAW", "CORR_FROZEN", "CORR_REFIT", "MIX_200", "MIX_1000",
        "CORR_REFIT_n200", "CORR_REFIT_n1000")
SHOW = ("ndcg10", "map", "offset", "mae", "f1", "precision", "recall",
        "rel_rate_ratio", "ece4", "mean_pred")
TIE_RANK = 0.005      # NDCG / MAP / F1
TIE_OFF = 0.02        # mean offset / relevant-rate ratio
JUDGE_SHORT = {"deepseek__deepseek-v4-flash": "v4-flash",
               "google__gemini-3.5-flash": "3.5-flash",
               "google__gemini-3.8-flash": "3.8-flash"}


def load_evals(root: Path) -> dict:
    """{arm: {slug: {SET: {seed: result}}}}"""
    out: dict = {}
    for p in sorted(root.glob("*/*/*_dl22.json")):
        arm, slug = p.parent.parent.name, p.parent.name
        stem = p.name[: -len("_dl22.json")]
        set_name, _, seed = stem.rpartition("_s")
        r = json.loads(p.read_text(encoding="utf-8"))
        out.setdefault(arm, {}).setdefault(slug, {}).setdefault(
            set_name, {})[int(seed)] = r
    return out


def seed_mean(rs: dict, key: str) -> tuple[float, float, int]:
    vals = [r["point"][key] for r in rs.values()]
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return sum(vals) / len(vals), sd, len(vals)


def fmt_table(arm: str, slug: str, sets: dict) -> None:
    print(f"\n  [{arm}] {slug}  (3-seed mean; sd in parentheses = seed noise)")
    print("  " + f"{'set':<18}" + "".join(f"{k:>16}" for k in SHOW))
    for s in SETS:
        if s not in sets:
            continue
        cells = []
        for k in SHOW:
            m, sd, _n = seed_mean(sets[s], k)
            cells.append(f"{m:+.4f}({sd:.3f})")
        n = len(sets[s])
        print("  " + f"{s:<18}" + "".join(f"{c:>16}" for c in cells)
              + ("" if n == 3 else f"   <-- only {n} seed(s)"))
    r0 = next(iter(next(iter(sets.values())).values()))
    print(f"  pairs {r0['n_pairs']}  queries {r0['n_queries']}  "
          f"universe sha {r0['universe_sha256'][:12]}  code {r0.get('code_git')}")
    shas = {r["universe_sha256"] for ss in sets.values() for r in ss.values()}
    if len(shas) != 1:
        print(f"  !! universe differs across result files: {shas}")


def pt(sets: dict, s: str, k: str) -> float | None:
    if s not in sets:
        return None
    return seed_mean(sets[s], k)[0]


def verdict_gt(diff: float, tie: float, want_positive: bool = True) -> str:
    """prediction 'diff > 0' (or '< 0' if not want_positive) with a tie band."""
    if abs(diff) <= tie:
        return "tie"
    return "HIT" if (diff > 0) == want_positive else "MISS"


def adjudicate(arm: str, slug: str, sets: dict, calls: dict) -> list[tuple]:
    """Return rows (pred, judge, value_str, verdict)."""
    j = JUDGE_SHORT.get(slug, slug)
    rows = []
    h = lambda k: pt(sets, "HUMAN", k)  # noqa: E731
    g = lambda s, k: pt(sets, s, k)     # noqa: E731
    # P1  offset(RAW) - offset(HUMAN) > +0.02
    if g("RAW", "offset") is not None and h("offset") is not None:
        d = g("RAW", "offset") - h("offset")
        rows.append(("P1", j, f"off(RAW)-off(HUMAN) = {d:+.4f}",
                     "HIT" if d > TIE_OFF else ("tie" if d >= -TIE_OFF else "MISS")))
    # P2  |off(FROZEN)-off(H)| < |off(RAW)-off(H)|   (3.8-flash: invariant)
    if all(x is not None for x in (g("CORR_FROZEN", "offset"), g("RAW", "offset"), h("offset"))):
        a = abs(g("CORR_FROZEN", "offset") - h("offset"))
        b = abs(g("RAW", "offset") - h("offset"))
        if calls.get(j, {}).get("P2") == "invariant":
            dn = abs(g("CORR_FROZEN", "ndcg10") - g("RAW", "ndcg10"))
            do = abs(g("CORR_FROZEN", "offset") - g("RAW", "offset"))
            rows.append(("P2", j, f"no-op invariant: |dNDCG| {dn:.4f} |doff| {do:.4f}",
                         "N/A (invariant " + ("OK" if dn < 1e-6 and do < 1e-6 else
                                               f"NOT bit-identical; within tie bands: {dn <= TIE_RANK and do <= TIE_OFF}") + ")"))
        else:
            rows.append(("P2", j, f"|dev FROZEN| {a:.4f} vs |dev RAW| {b:.4f}  (b-a {b-a:+.4f})",
                         verdict_gt(b - a, TIE_OFF)))
    # P3  NDCG(FROZEN)-NDCG(RAW) within +-0.01
    if g("CORR_FROZEN", "ndcg10") is not None and calls.get(j, {}).get("P3"):
        d = g("CORR_FROZEN", "ndcg10") - g("RAW", "ndcg10")
        rows.append(("P3", j, f"dNDCG@10 FROZEN-RAW = {d:+.4f}",
                     "HIT" if abs(d) <= 0.01 else "MISS"))
    # P4  F1(FROZEN)-F1(RAW) >= 0   (v4-flash call only)
    if g("CORR_FROZEN", "f1") is not None and calls.get(j, {}).get("P4"):
        d = g("CORR_FROZEN", "f1") - g("RAW", "f1")
        rows.append(("P4", j, f"dF1@2 FROZEN-RAW = {d:+.4f}", verdict_gt(d, TIE_RANK)))
    elif g("CORR_FROZEN", "f1") is not None and calls.get(j, {}).get("P4") is None and j == "3.5-flash":
        d = g("CORR_FROZEN", "f1") - g("RAW", "f1")
        rows.append(("P4", j, f"dF1@2 FROZEN-RAW = {d:+.4f}", "no call (registered)"))
    # P5  |ratio(FROZEN)-1| < |ratio(RAW)-1|
    if g("CORR_FROZEN", "rel_rate_ratio") is not None and calls.get(j, {}).get("P5"):
        a = abs(g("CORR_FROZEN", "rel_rate_ratio") - 1)
        b = abs(g("RAW", "rel_rate_ratio") - 1)
        rows.append(("P5", j, f"|ratio-1| FROZEN {a:.4f} vs RAW {b:.4f}  (b-a {b-a:+.4f})",
                     verdict_gt(b - a, TIE_OFF)))
    # P6  equal budget: |off(REFIT_n1000)-off(H)| < |off(MIX_1000)-off(H)| ; n200 likewise
    for n in (1000, 200):
        r, m = f"CORR_REFIT_n{n}", f"MIX_{n}"
        if g(r, "offset") is not None and g(m, "offset") is not None and calls.get(j, {}).get("P6"):
            a = abs(g(r, "offset") - h("offset"))
            b = abs(g(m, "offset") - h("offset"))
            rows.append((f"P6-n{n}", j, f"|dev REFIT_n{n}| {a:.4f} vs |dev MIX_{n}| {b:.4f}  (b-a {b-a:+.4f})",
                         verdict_gt(b - a, TIE_OFF)))
            dfa = g(r, "f1") - g(m, "f1")
            rows.append((f"P6-n{n}-F1", j, f"dF1@2 REFIT_n{n}-MIX_{n} = {dfa:+.4f}", "no call (registered)"))
    # P7  NDCG(HUMAN) >= NDCG(RAW) - 0.005
    if h("ndcg10") is not None and calls.get(j, {}).get("P7"):
        d = h("ndcg10") - g("RAW", "ndcg10")
        rows.append(("P7", j, f"dNDCG@10 HUMAN-RAW = {d:+.4f}",
                     "HIT" if d >= -TIE_RANK else "MISS"))
    return rows


CALLS = {  # registered calls (paper/PREDICTIONS.md, B7 block)
    "v4-flash": {"P1": True, "P2": True, "P3": True, "P4": True, "P5": True, "P6": True, "P7": True},
    "3.5-flash": {"P1": True, "P2": True, "P3": True, "P4": None, "P5": True, "P6": True, "P7": True},
    "3.8-flash": {"P1": True, "P2": "invariant"},
}


def load_compares(root: Path) -> dict:
    out: dict = {}
    for p in sorted(root.glob("*/*/compare_*_vs_*.json")):
        arm, slug = p.parent.parent.name, p.parent.name
        out.setdefault(arm, {}).setdefault(slug, {})[p.stem[len("compare_"):]] = \
            json.loads(p.read_text(encoding="utf-8"))
    return out


def write_csvs(d: Path, ev: dict, cmp_: dict) -> None:
    import csv
    d.mkdir(parents=True, exist_ok=True)
    with (d / "b7_sets.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "judge", "set", "n_seeds", "n_pairs", "n_queries"]
                   + [f"{k}_{x}" for k in SHOW for x in ("mean", "sd")]
                   + [f"{k}_ci_lo_s1" for k in SHOW] + [f"{k}_ci_hi_s1" for k in SHOW])
        for arm, judges in ev.items():
            for slug, sets in judges.items():
                for s_, rs in sets.items():
                    r0 = rs[min(rs)]
                    row = [arm, slug, s_, len(rs), r0["n_pairs"], r0["n_queries"]]
                    for k in SHOW:
                        m, sd, _n = seed_mean(rs, k)
                        row += [f"{m:.6f}", f"{sd:.6f}"]
                    row += [f"{r0['ci90'][k][0]:.6f}" for k in SHOW]
                    row += [f"{r0['ci90'][k][1]:.6f}" for k in SHOW]
                    w.writerow(row)
    with (d / "b7_compares.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "judge", "compare", "endpoint", "a", "b", "diff", "ci_lo", "ci_hi", "tost_ndcg10"])
        for arm, judges in cmp_.items():
            for slug, cs in judges.items():
                for name, r in cs.items():
                    for k, e in r["endpoints"].items():
                        w.writerow([arm, slug, name, k, f"{e['a']:.6f}", f"{e['b']:.6f}", f"{e['diff']:.6f}",
                                    f"{e['ci90'][0]:.6f}", f"{e['ci90'][1]:.6f}",
                                    r["tost_ndcg10"]["statement"].replace("\n", " ")])
    print(f"  csv -> {d / 'b7_sets.csv'}, {d / 'b7_compares.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--csv-dir", default=None,
                    help="also write b7_sets.csv / b7_compares.csv here (for the paper table builder)")
    args = ap.parse_args()
    root = Path(args.eval_root)
    ev = load_evals(root)
    cmp_ = load_compares(root)
    if args.csv_dir:
        write_csvs(Path(args.csv_dir), ev, cmp_)
    n_files = sum(len(ss) for a in ev.values() for sl in a.values() for ss in sl.values())
    print(f"BATCH 7 SUMMARY  eval files {n_files}  arms {sorted(ev)}")
    print("FRAMING: operating-point artifact removal against THESE annotators; "
          "all absolute endpoints relative to the same-arm HUMAN student (recipe baseline).")

    for arm in ARMS:
        for slug, sets in ev.get(arm, {}).items():
            fmt_table(arm, slug, sets)

    print("\n" + "=" * 100 + "\n  PAIRED COMPARES (3-seed side means, paired query bootstrap 90% CI; "
          "* = CI excludes 0)\n" + "=" * 100)
    keys = ("ndcg10", "map", "offset", "f1", "rel_rate_ratio", "mae")
    for arm in ARMS:
        for slug, cs in cmp_.get(arm, {}).items():
            print(f"\n  [{arm}] {slug}")
            print("  " + f"{'A vs B':<36}" + "".join(f"{k:>26}" for k in keys) + "   TOST(NDCG@10)")
            for name, r in cs.items():
                cells = []
                for k in keys:
                    e = r["endpoints"][k]
                    lo, hi = e["ci90"]
                    star = "*" if (lo > 0 or hi < 0) else " "
                    cells.append(f"{e['diff']:+.4f}[{lo:+.3f},{hi:+.3f}]{star}")
                tost = r["tost_ndcg10"]["statement"]
                tost = ("equivalent" if "equivalent" in tost and "not" not in tost.split("equivalent")[0][-5:]
                        else tost)[:40]
                print("  " + f"{name:<36}" + "".join(f"{c:>26}" for c in cells) + f"   {tost}")

    print("\n" + "=" * 100 + "\n  B7 PRE-REGISTRATION ADJUDICATION (registered rules; main arm scored, "
          "msmarco-ce = robustness)\n" + "=" * 100)
    tallies = {}
    main_dir = {}
    for arm in ARMS:
        print(f"\n  arm {arm}{'  [SCORED]' if arm == MAIN_ARM else '  [robustness, not scored]'}")
        for slug, sets in ev.get(arm, {}).items():
            for pred, j, val, v in adjudicate(arm, slug, sets, CALLS):
                print(f"    {pred:<10} {j:<10} {val:<64} -> {v}")
                if arm == MAIN_ARM and v in ("HIT", "MISS", "tie"):
                    tallies[v] = tallies.get(v, 0) + 1
                if pred in ("P1", "P2", "P5") and v in ("HIT", "MISS", "tie"):
                    main_dir.setdefault((pred, j), {})[arm] = v
        if arm == MAIN_ARM:
            print(f"    main-arm tally: HIT {tallies.get('HIT', 0)} / MISS {tallies.get('MISS', 0)} "
                  f"/ tie {tallies.get('tie', 0)}")
    print("\n  P8 (msmarco-ce same direction as minilm-raw on P1/P2/P5; not scored):")
    for (pred, j), d in sorted(main_dir.items()):
        if len(d) == 2:
            same = d[MAIN_ARM] == d["msmarco-ce"]
            print(f"    {pred} {j:<10} minilm-raw {d[MAIN_ARM]:<5} msmarco-ce {d['msmarco-ce']:<5} -> "
                  f"{'same' if same else 'DIFFERENT'}")


if __name__ == "__main__":
    main()
