#!/usr/bin/env python3
"""BA — deterministic sampler + gated analysis for the AUTHOR BLIND AUDIT of
the label-incompleteness (IC) flags (analysis only, no API).

Why
---
The IC relabel bound (scripts/incompleteness_check.py addons, part `bound`)
relabels every human-0 pair that >= 2 of the m cheap judges label >= 2.  It is
circular (built from LLM labels).  On DL19/DL20 (m = 5) it is largest, so an
external read is needed to say how much of it could be genuine missed
positives (evaluator report 2026-11-06 §F 2, §G5).  The author grades a
blinded sample from text with the rubric; the sheet carries query and passage
ONLY.  Exploratory; the annotator is the author, not a NIST assessor.

Data (nothing new is collected)
-------------------------------
* judge records: $DATA_DIR/derived/judgments.parquet; human grades
  src.qrels.load_qrels — both loaded through incompleteness_check.load_inputs
  / build_cells / cheap_common VERBATIM (the IC universe and RAW_lab =
  round-half-up of the L3 record mean), datasets dl19, dl20 only.
* per-pair votes are NOT stored in ic_addons_bound.json (it stores only the
  flagged pair ids per rule k = 1, 2, 3 and per-judge Δ before/after), so the
  votes are recomputed here; the recomputed h = 0 flag sets for >= 2/5 and
  >= 3/5 are asserted identical to ic_addons_bound.json, and the judgments /
  qrels sha256 to its meta.
* pair texts: $DATA_DIR/inputs/pairs_<ds>.jsonl (src/build_pairs.py; fields
  dataset, qid, docid, query, passage[, rel]); pairs_<ds>_flagship40.jsonl is
  read as a fallback (it is a subset).

Sampling procedure (registered verbatim; PROCEDURE below is the same text)
-------------------------------------------------------------------------
1. Universe per dataset d in {dl19, dl20}: the five cheap judges' common
   universe exactly as incompleteness_check.cheap_common; votes
   v = #{cheap judges j : RAW_lab_j >= 2}; h = human grade (0-3).
2. Strata (per dataset; quota dl19 / dl20):
     F2  h = 0 and v = 2          8 / 8
     F3  h = 0 and v >= 3         8 / 8
     C0  h = 0 and v = 0          4 / 4   (same-query preference)
     H1  h = 1 and v >= 2         5 / 5   (same-query preference)
     H2  h = 2 and v >= 2         5 / 5   (same-query preference)
   Total 60.
3. One generator rng = numpy.random.default_rng(seed) is used for every draw,
   in this fixed order: dl19 F2, F3, C0, H1, H2, then dl20 F2, F3, C0, H1, H2,
   then the item shuffle (step 6).
4. Candidates of a stratum are sorted by (qid, docid).  F strata: walk the
   candidates in the order of rng.permutation(n).  Control strata (C0, H1,
   H2): split the candidates into tier A = qid among the qids of the F2 + F3
   pairs already sampled for the same dataset, and tier B = the rest; draw
   pA = rng.permutation(len A) and then pB = rng.permutation(len B) (both are
   always drawn); walk tier A in pA order, then tier B in pB order.
5. Walking: a candidate is skipped if its passage (whitespace-stripped) is
   shorter than 20 characters, or is identical (after stripping) to the
   passage of any pair already sampled (any stratum, any dataset); otherwise
   it is taken, until the quota is met.  If the candidates run out the
   stratum keeps all it took and the shortfall is recorded in the manifest.
6. Items: with N sampled pairs (sampling order 0..N-1), perm =
   rng.permutation(N); item k (1-based) is the sampled pair perm[k-1].  The
   sheet lists items 1..N; it carries no id, dataset, stratum, vote or grade.

Subcommands
-----------
  sample    writes to --out (default $DATA_DIR/derived/incompleteness/
            blind_audit/): audit_sheet.csv (item, query, passage, score, note),
            audit_sheet.md (the same, for a text editor), mapping_sealed.json
            (item -> dataset, qid, docid, stratum, votes, human grade,
            per-judge RAW_lab) and audit_manifest.json (seed, counts, sha256s,
            procedure, timestamp); prints the "# BA 预注册 — 作者盲审" draft
            (aggregate counts, sha256 lines, definitions — no per-item data).
            Refuses to overwrite an existing sheet without --force.
  analyze   GATED: needs --unseal AND a file UNSEAL_OK next to
            mapping_sealed.json (created by hand only after the registration
            commit and after the scores exist).  Reads a CSV with columns
            item, score (0-3 for every item; extra columns ignored), checks
            the mapping sha256 against the manifest, and computes per stratum
            the share graded >= 2 and >= 1 with exact (Clopper-Pearson) 90%
            CIs; m̂ = clip((p_F − p_H1) / (p_H2 − p_H1), 0, 1) on the >= 2
            shares for F2, F3, F pooled (the 32 F items) and F weighted (F2/F3
            shares weighted by their population sizes in the >= 2/5 flag set,
            from ic_addons_bound.json), datasets pooled, with a bootstrap 90%
            CI (resampling items within each stratum x dataset cell, B = 2000,
            seed 42; draws with p_H2 <= p_H1 are undefined and counted); and
            the implied genuine-miss Δ drop = m̂ × the bound's Δ drop (rule
            >= 2/5, cheap judges; flagships on their 40% universe listed
            separately) per dataset.  Prints a BACKFILL block; writes
            ba_analysis.json.
  selftest  synthetic pairs + judgments stub, offline; must print
            "ALL SELFTESTS PASSED".

Usage (Colab, repo root):
    python scripts/blind_audit_sample.py selftest
    python scripts/blind_audit_sample.py sample --seed 20261106
    python scripts/blind_audit_sample.py analyze --unseal \\
        --scores <filled csv> --mapping <out>/mapping_sealed.json
CPU only, no API calls.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import incompleteness_check as ic                          # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
# Constants (registered verbatim)
# ═══════════════════════════════════════════════════════════════════════════

BA_MARKER = "BA 预注册 — 作者盲审"
DATASETS = ("dl19", "dl20")
M_PANEL = 5
REL = 2                                  # judge "relevant" = RAW_lab >= 2
SEED = 20261106
MIN_PASSAGE_CHARS = 20
STRATA = ("F2", "F3", "C0", "H1", "H2")
F_STRATA = ("F2", "F3")
CONTROL = ("C0", "H1", "H2")
QUOTA = {"F2": 8, "F3": 8, "C0": 4, "H1": 5, "H2": 5}     # per dataset
STRATUM_DEF = {
    "F2": "h = 0 且恰好 2/5 个 cheap judge RAW_lab ≥ 2",
    "F3": "h = 0 且 ≥ 3/5 个 cheap judge RAW_lab ≥ 2",
    "C0": "h = 0 且 0/5 个 cheap judge RAW_lab ≥ 2（同 query 优先）",
    "H1": "h = 1 且 ≥ 2/5 个 cheap judge RAW_lab ≥ 2（同 query 优先）",
    "H2": "h = 2 且 ≥ 2/5 个 cheap judge RAW_lab ≥ 2（同 query 优先）",
}
SHEET_COLS = ("item", "query", "passage", "score", "note")
SHEET_CSV, SHEET_MD = "audit_sheet.csv", "audit_sheet.md"
MAPPING, MANIFEST = "mapping_sealed.json", "audit_manifest.json"
UNSEAL_FILE = "UNSEAL_OK"
ANALYSIS_JSON = "ba_analysis.json"
B_BOOT, BOOT_SEED = 2000, 42
CI_LEVEL = 0.90
BOUND_RULE = "2"                         # primary relabel rule >= 2/m

PROCEDURE = __doc__.split("Sampling procedure (registered verbatim; "
                          "PROCEDURE below is the same text)\n")[1] \
    .split("\nSubcommands\n")[0].split("\n", 1)[1].strip("\n")

ANALYSIS_DEF = [
    "每层（数据集合并，并分数据集列出）：作者评分 ≥2 的占比与 ≥1 的占比，"
    "精确二项（Clopper-Pearson）90% CI。",
    "m̂ = clip((p_F − p_H1)/(p_H2 − p_H1), 0, 1)，p = 评分 ≥2 占比、DL19+DL20 "
    "合并；F ∈ {F2, F3, F 合并（32 条等权）, F 加权（F2/F3 按 ≥2/5 flag 集中"
    "的总体规模加权）}；p_H2 ≤ p_H1 时 m̂ 无定义。",
    "m̂ 的 90% CI：在每个 层×数据集 单元内对条目有放回重抽（单元大小不变），"
    "B = 2000，seed 42，百分位区间；无定义的重抽次数单独报告。",
    "隐含真漏标 Δ 降幅（每数据集）= m̂ × relabel bound（≥2/5 规则）的 Δ 降幅"
    "（cheap judge，取自 ic_addons_bound.json；flagship 40% universe 另列）；"
    "主报告用 F 合并，F 加权并列。",
    "描述性、探索性、不计分；标注者为作者本人（非 NIST 评估员），盲于 qid/docid/"
    "数据集/分层/票数/人工等级；m̂ 是“若作者 = 评估员尺度”下的换算，不是漏标率。",
]


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1,
                               allow_nan=False) + "\n", encoding="utf-8")


def in_stratum(s: str, h: int, v: int) -> bool:
    if s == "F2":
        return h == 0 and v == 2
    if s == "F3":
        return h == 0 and v >= 3
    if s == "C0":
        return h == 0 and v == 0
    if s == "H1":
        return h == 1 and v >= 2
    if s == "H2":
        return h == 2 and v >= 2
    raise ValueError(s)


def default_out(data_dir: Path) -> Path:
    return data_dir / "derived" / "incompleteness" / "blind_audit"


# ═══════════════════════════════════════════════════════════════════════════
# Loading: panel (IC verbatim), texts, bound cross-check
# ═══════════════════════════════════════════════════════════════════════════

def load_panel(args) -> tuple[dict, dict]:
    """ds -> {'judges', 'keys', 'lab' (n, m), 'h'} from the IC loaders."""
    ns = argparse.Namespace(data_dir=args.data_dir, judgments=args.judgments,
                            qrels_json=args.qrels_json, judges=args.judges,
                            cheap=args.cheap, flagships=args.flagships)
    ctx = ic.load_inputs(ns, list(DATASETS))
    cells = ic.build_cells(ctx, ic.SEED)
    panel = {}
    for ds in DATASETS:
        cc = ic.cheap_common(cells, ds, ctx["judges"], ctx["cheap"])
        if cc is None or len(cc["judges"]) != M_PANEL:
            sys.exit(f"ERROR: {ds}: expected a cheap panel of m = {M_PANEL}, "
                     f"got {None if cc is None else cc['judges']}.")
        panel[ds] = cc
    return panel, ctx


def load_texts(data_dir: Path, ds: str) -> tuple[dict, dict]:
    """(qid, docid) -> (query, passage); main pairs file, flagship40 fallback."""
    out, shas = {}, {}
    for name in (f"pairs_{ds}.jsonl", f"pairs_{ds}_flagship40.jsonl"):
        p = data_dir / "inputs" / name
        if not p.exists():
            continue
        shas[name] = sha256_file(p)
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                k = (str(r["qid"]), str(r["docid"]))
                if "query" not in r or "passage" not in r:
                    continue
                t = (str(r["query"]), str(r["passage"]))
                if k in out and out[k] != t:
                    sys.exit(f"ERROR: {ds}: pair text differs between pairs "
                             f"files for one (qid, docid).")
                out.setdefault(k, t)
    if not out:
        sys.exit(f"ERROR: no pair texts for {ds} under {data_dir}/inputs.")
    return out, shas


def check_bound(bound: dict, panel: dict, ctx: dict) -> None:
    """Recomputed h=0 flag sets (>=2/5, >=3/5) and input shas == the bound."""
    meta = bound.get("meta", {})
    want = meta.get("input_sha256", {})
    actual = ic.input_shas(ctx)
    for name in ["judgments"] + [f"qrels:{d}" for d in DATASETS]:
        if want.get(name) != actual.get(name):
            sys.exit(f"ERROR: sha256[{name}] {actual.get(name)} != "
                     f"ic_addons_bound.json meta {want.get(name)} — inputs "
                     f"drifted since the IC add-ons; refusing to sample.")
    for ds in DATASETS:
        b = bound["datasets"].get(ds)
        cc = panel[ds]
        if b is None or "rules" not in b:
            sys.exit(f"ERROR: ic_addons_bound.json has no rules for {ds}.")
        if list(b["cheap_judges"]) != list(cc["judges"]):
            sys.exit(f"ERROR: {ds}: cheap panel {cc['judges']} != bound "
                     f"panel {b['cheap_judges']}.")
        v = (cc["lab"] >= REL).sum(axis=1)
        for k in ("2", "3"):
            got = {f"{q}|{d}" for (q, d), hh, vv in zip(cc["keys"], cc["h"], v)
                   if hh == 0 and vv >= int(k)}
            if got != set(b["rules"][k]["flagged_pairs"]):
                sys.exit(f"ERROR: {ds}: recomputed h=0 flag set (≥{k}/5) "
                         f"differs from ic_addons_bound.json.")


def bound_drops(bound: dict, cheap_judges: dict) -> dict:
    """ds -> {'cheap': drop, 'judges': {model: drop}} (drop = −Δ change)."""
    out = {}
    for ds in DATASETS:
        r = bound["datasets"][ds]["rules"][BOUND_RULE]
        js = {m: -float(j["delta_change"]) for m, j in r["judges"].items()}
        cj = [js[m] for m in cheap_judges[ds] if m in js]
        if not cj or max(cj) - min(cj) > 1e-9:
            sys.exit(f"ERROR: {ds}: cheap-judge Δ drops under ≥2/5 are "
                     f"missing or not identical: {cj}.")
        out[ds] = {"cheap": cj[0], "judges": js,
                   "flagship40": {m: js[m] for m, j in r["judges"].items()
                                  if j.get("flagship40")}}
    return out


def bound_pop(bound: dict) -> dict:
    """ds -> population sizes of F2 (exactly 2/5) and F3 (>=3/5)."""
    out = {}
    for ds in DATASETS:
        r = bound["datasets"][ds]["rules"]
        n2, n3 = int(r["2"]["n_flagged"]), int(r["3"]["n_flagged"])
        out[ds] = {"F2": n2 - n3, "F3": n3}
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Sampling
# ═══════════════════════════════════════════════════════════════════════════

def sample_pairs(panel: dict, texts: dict, seed: int) -> tuple[list, dict,
                                                               np.ndarray]:
    """Returns (sampled pairs in sampling order, stats, item permutation)."""
    rng = np.random.default_rng(seed)
    sampled: list[dict] = []
    seen_passages: set = set()
    stats: dict = {}
    for ds in DATASETS:
        cc = panel[ds]
        v = (cc["lab"] >= REL).sum(axis=1)
        idx_of = {k: i for i, k in enumerate(cc["keys"])}
        fq: set = set()
        stats[ds] = {}
        for s in STRATA:
            cand = sorted(k for k, hh, vv in zip(cc["keys"], cc["h"], v)
                          if in_stratum(s, int(hh), int(vv)))
            if s in F_STRATA:
                tiers = [("all", [cand[i] for i in rng.permutation(len(cand))])]
            else:
                A = [k for k in cand if k[0] in fq]
                B = [k for k in cand if k[0] not in fq]
                pA, pB = rng.permutation(len(A)), rng.permutation(len(B))
                tiers = [("A", [A[i] for i in pA]), ("B", [B[i] for i in pB])]
            st = {"quota": QUOTA[s], "candidates": len(cand), "taken": 0,
                  "skipped_short": 0, "skipped_dup": 0}
            for tname, order in tiers:
                st[f"cand_{tname}"] = len(order)
                st[f"taken_{tname}"] = 0
                st[f"exhausted_{tname}"] = False
                for k in order:
                    if st["taken"] >= QUOTA[s]:
                        break
                    if k not in texts[ds]:
                        sys.exit(f"ERROR: {ds}: a universe pair is missing "
                                 f"from the pairs files.")
                    query, passage = texts[ds][k]
                    p = passage.strip()
                    if len(p) < MIN_PASSAGE_CHARS:
                        st["skipped_short"] += 1
                        continue
                    if p in seen_passages:
                        st["skipped_dup"] += 1
                        continue
                    seen_passages.add(p)
                    i = idx_of[k]
                    sampled.append({
                        "dataset": ds, "qid": k[0], "docid": k[1],
                        "stratum": s, "votes": int(v[i]),
                        "human_grade": int(cc["h"][i]),
                        "raw_lab": {m: int(cc["lab"][i, j])
                                    for j, m in enumerate(cc["judges"])},
                        "query": query, "passage": passage, "tier": tname})
                    st["taken"] += 1
                    st[f"taken_{tname}"] += 1
                else:
                    st[f"exhausted_{tname}"] = True
            st["shortfall"] = QUOTA[s] - st["taken"]
            if s in F_STRATA:
                fq |= {r["qid"] for r in sampled
                       if r["dataset"] == ds and r["stratum"] == s}
            stats[ds][s] = st
    perm = rng.permutation(len(sampled))
    return sampled, stats, perm


def md_text(rows: list[dict]) -> str:
    out = ["# Blind audit sheet", "",
           "Grade each passage for the query with the rubric (0-3). "
           "Fill the score in audit_sheet.csv (column `score`).", ""]
    for r in rows:
        out += [f"### Item {r['item']}", "", f"**Query:** {r['query']}", "",
                f"**Passage:** {r['passage']}", "", "Score: ___", "",
                "Note: ___", ""]
    return "\n".join(out)


def registration_lines(man: dict) -> list[str]:
    L = [f"# {BA_MARKER}（探索性；作者本人盲评 IC flag 对）", "",
         f"- 生成时间（UTC）：{man['timestamp_utc']}；脚本 "
         f"scripts/blind_audit_sample.py @ {man['git_hash']}；seed "
         f"{man['seed']}",
         f"- 范围：DL19、DL20；cheap panel m = {M_PANEL}（"
         + ", ".join(man["cheap_panel"]) + "）；票数 v = #{j: RAW_lab_j ≥ 2}"
         "（IC universe 原样，RAW_lab = L3 记录均值 round-half-up）",
         "- 分层定义（h = 人工等级）："]
    for s in STRATA:
        L.append(f"  - {s}：{STRATUM_DEF[s]}；每数据集 {QUOTA[s]} 条")
    L.append("- 抽样数（取得/配额，候选数）：")
    for ds in DATASETS:
        parts = []
        for s in STRATA:
            st = man["counts"][ds][s]
            parts.append(f"{s} {st['taken']}/{st['quota']}"
                         f"（候选 {st['candidates']}）")
        L.append(f"  - {ds}：" + "；".join(parts))
    sf = [f"{ds} {s} 缺 {man['counts'][ds][s]['shortfall']}"
          for ds in DATASETS for s in STRATA
          if man["counts"][ds][s]["shortfall"] > 0]
    L.append(f"- 合计 {man['n_items']} 条" + (
        "；不足配额：" + "，".join(sf) if sf else "；各层均满额"))
    L.append(f"- 排除规则：passage 去空白后 < {MIN_PASSAGE_CHARS} 字符，或与已抽"
             "中条目 passage 完全相同 → 跳过")
    L.append("- 抽样程序：见 audit_manifest.json 的 procedure 字段（脚本 "
             "docstring 原文）；同 query 优先 = 对照层先从本数据集已抽 F2/F3 的 "
             "query 中均匀抽，不足再从其余 query 均匀抽")
    for name in (SHEET_CSV, SHEET_MD, MAPPING):
        L.append(f"- sha256[{name}]: {man['sha256'][name]}")
    for name, sha in man["input_sha256"].items():
        L.append(f"- sha256[{name}]: {sha}")
    L.append("- 盲评流程：作者只打开 audit_sheet（query + passage），按 "
             "config/prompts.yaml 的 rubric 给 0–3 分；评分完成并 commit 评分文件"
             "（或记录其 sha256）之前不打开 mapping_sealed.json；之后才创建 "
             f"{UNSEAL_FILE} 并运行 analyze --unseal")
    L.append("- 分析定义：")
    for a in ANALYSIS_DEF:
        L.append(f"  - {a}")
    L.append("- 预测（作者填写，可选）：____")
    return L


def cmd_sample(args) -> dict:
    data_dir = ic.b5.resolve_data_dir(args.data_dir)
    od = Path(args.out) if args.out else default_out(data_dir)
    if (od / SHEET_CSV).exists() and not args.force:
        sys.exit(f"ERROR: {od / SHEET_CSV} exists — the sample is frozen once "
                 f"drawn; pass --force only if it was never registered.")
    bpath = Path(args.bound) if args.bound else \
        data_dir / "derived" / "incompleteness" / "ic_addons_bound.json"
    if not bpath.exists():
        sys.exit(f"ERROR: {bpath} missing — run incompleteness_check.py "
                 f"addons first.")
    bound = json.loads(bpath.read_text(encoding="utf-8"))
    panel, ctx = load_panel(args)
    check_bound(bound, panel, ctx)
    texts, tshas = {}, {}
    for ds in DATASETS:
        texts[ds], s = load_texts(data_dir, ds)
        tshas.update({f"pairs:{k}": v for k, v in s.items()})
    sampled, stats, perm = sample_pairs(panel, texts, args.seed)
    rows = []
    for item, si in enumerate(perm, start=1):
        r = dict(sampled[int(si)])
        r["item"] = item
        r["sample_order"] = int(si)
        rows.append(r)
    od.mkdir(parents=True, exist_ok=True)
    with open(od / SHEET_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(SHEET_COLS)
        for r in rows:
            w.writerow([r["item"], r["query"], r["passage"], "", ""])
    (od / SHEET_MD).write_text(md_text(rows), encoding="utf-8")
    mapping = {
        "note": "SEALED — do not open before the scores are committed.",
        "seed": args.seed,
        "items": {str(r["item"]): {
            k: r[k] for k in ("dataset", "qid", "docid", "stratum", "votes",
                              "human_grade", "raw_lab", "sample_order",
                              "tier")} for r in rows}}
    write_json(od / MAPPING, mapping)
    ishas = {"judgments": ctx["jsha"],
             **{f"qrels:{d}": ctx["qshas"][d][0] for d in DATASETS},
             "ic_addons_bound.json": sha256_file(bpath), **tshas}
    man = {"script": "scripts/blind_audit_sample.py sample",
           "git_hash": ic.b4.git_hash(), "seed": args.seed,
           "datasets": list(DATASETS),
           "cheap_panel": list(panel[DATASETS[0]]["judges"]),
           "strata": STRATUM_DEF, "quota_per_dataset": QUOTA,
           "min_passage_chars": MIN_PASSAGE_CHARS, "n_items": len(rows),
           "counts": stats,
           "shortfall": {f"{ds}|{s}": stats[ds][s]["shortfall"]
                         for ds in DATASETS for s in STRATA
                         if stats[ds][s]["shortfall"] > 0},
           "sha256": {n: sha256_file(od / n)
                      for n in (SHEET_CSV, SHEET_MD, MAPPING)},
           "input_sha256": ishas, "bound_path": str(bpath),
           "procedure": PROCEDURE, "analysis": ANALYSIS_DEF,
           "timestamp_utc": now_utc()}
    write_json(od / MANIFEST, man)
    print(f"[ba] wrote {od / SHEET_CSV}, {SHEET_MD}, {MAPPING}, {MANIFEST} "
          f"({len(rows)} items)")
    if man["shortfall"]:
        print(f"[ba] WARNING: strata below quota (all candidates taken): "
              f"{man['shortfall']}")
    print("\n" + "=" * 30 + " REGISTRATION BLOCK (paste into "
          "paper/PREDICTIONS.md) " + "=" * 30)
    print("\n".join(registration_lines(man)))
    print("=" * 96)
    return {"out": od, "manifest": man, "rows": rows}


# ═══════════════════════════════════════════════════════════════════════════
# Analysis (gated)
# ═══════════════════════════════════════════════════════════════════════════

def cp_ci(k: int, n: int, level: float = CI_LEVEL) -> list:
    from scipy.stats import beta
    if n == 0:
        return [None, None]
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return [lo, hi]


def mhat(pF, pH1, pH2):
    den = pH2 - pH1
    if not np.isfinite(den) or den <= 0 or not np.isfinite(pF):
        return None
    return float(min(1.0, max(0.0, (pF - pH1) / den)))


def m_estimates(ge2: dict, w: dict) -> dict:
    """ge2: (stratum, ds) -> 0/1 array.  Datasets pooled."""
    def pool(*ss):
        x = np.concatenate([ge2[(s, d)] for s in ss for d in DATASETS])
        return float(x.mean()) if x.size else float("nan")
    pH1, pH2 = pool("H1"), pool("H2")
    p2, p3, pF = pool("F2"), pool("F3"), pool("F2", "F3")
    pw = w["F2"] * p2 + w["F3"] * p3
    return {"F2": mhat(p2, pH1, pH2), "F3": mhat(p3, pH1, pH2),
            "F_pooled": mhat(pF, pH1, pH2), "F_weighted": mhat(pw, pH1, pH2)}


def gate_unseal(unseal: bool, od: Path) -> None:
    if not unseal:
        sys.exit("ERROR: analyze refuses to run without --unseal (the mapping "
                 "is sealed until the scores are in and the BA registration "
                 "is committed).")
    if not (od / UNSEAL_FILE).exists():
        sys.exit(f"ERROR: {od / UNSEAL_FILE} does not exist — create it only "
                 f"after the BA registration commit and after the scores "
                 f"arrive.")


def read_scores(path: Path, items: set) -> dict:
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        if not rd.fieldnames or "item" not in rd.fieldnames or \
                "score" not in rd.fieldnames:
            sys.exit(f"ERROR: {path} needs columns 'item' and 'score'.")
        out = {}
        for r in rd:
            it = str(r["item"]).strip()
            if it in out:
                sys.exit(f"ERROR: item {it} scored twice.")
            s = str(r["score"]).strip()
            if s not in ("0", "1", "2", "3"):
                sys.exit(f"ERROR: item {it}: score {s!r} is not one of 0-3.")
            out[it] = int(s)
    if set(out) != items:
        miss = sorted(items - set(out), key=int)
        extra = sorted(set(out) - items)
        sys.exit(f"ERROR: scores do not cover the items exactly (missing "
                 f"{miss[:10]}, unknown {extra[:10]}).")
    return out


def _p(x, p=3):
    return "n/a" if x is None else f"{x:.{p}f}"


def _c(c, p=3):
    return "[n/a]" if not c or c[0] is None else f"[{c[0]:.{p}f},{c[1]:.{p}f}]"


def cmd_analyze(args) -> dict:
    mpath = Path(args.mapping)
    od = mpath.parent
    gate_unseal(args.unseal, od)                       # FIRST: before any read
    manp = Path(args.manifest) if args.manifest else od / MANIFEST
    man = json.loads(manp.read_text(encoding="utf-8"))
    if sha256_file(mpath) != man["sha256"][MAPPING]:
        sys.exit("ERROR: mapping_sealed.json sha256 differs from the manifest "
                 "— the mapping changed after sampling; refusing.")
    bpath = Path(args.bound) if args.bound else Path(man["bound_path"])
    if not bpath.exists():
        sys.exit(f"ERROR: {bpath} missing (pass --bound).")
    if sha256_file(bpath) != man["input_sha256"]["ic_addons_bound.json"]:
        sys.exit("ERROR: ic_addons_bound.json sha256 differs from the one "
                 "recorded at sampling time.")
    bound = json.loads(bpath.read_text(encoding="utf-8"))
    mapping = json.loads(mpath.read_text(encoding="utf-8"))["items"]
    scores = read_scores(Path(args.scores), set(mapping))
    ssha = sha256_file(Path(args.scores))

    cells: dict = {(s, d): [] for s in STRATA for d in DATASETS}
    for it in sorted(mapping, key=int):
        m = mapping[it]
        cells[(m["stratum"], m["dataset"])].append(scores[it])
    sc = {k: np.array(v, dtype=int) for k, v in cells.items()}
    ge2 = {k: (v >= 2).astype(float) for k, v in sc.items()}

    shares: dict = {}
    for s in STRATA:
        for d in list(DATASETS) + ["pooled"]:
            x = np.concatenate([sc[(s, dd)] for dd in DATASETS]) \
                if d == "pooled" else sc[(s, d)]
            n = int(x.size)
            k2, k1 = int((x >= 2).sum()), int((x >= 1).sum())
            shares[f"{s}|{d}"] = {
                "n": n, "k_ge2": k2, "p_ge2": k2 / n if n else None,
                "ci_ge2": cp_ci(k2, n), "k_ge1": k1,
                "p_ge1": k1 / n if n else None, "ci_ge1": cp_ci(k1, n)}

    pop = bound_pop(bound)
    n2 = sum(pop[d]["F2"] for d in DATASETS)
    n3 = sum(pop[d]["F3"] for d in DATASETS)
    w = {"F2": n2 / (n2 + n3), "F3": n3 / (n2 + n3)}
    est = m_estimates(ge2, w)
    rng = np.random.default_rng(args.seed)
    draws = {k: [] for k in est}
    for _ in range(args.b):
        bs = {}
        for k, v in ge2.items():
            bs[k] = v[rng.integers(0, v.size, v.size)] if v.size else v
        e = m_estimates(bs, w)
        for k in est:
            draws[k].append(np.nan if e[k] is None else e[k])
    boot = {}
    for k, v in draws.items():
        a = np.array(v, dtype=float)
        f = a[np.isfinite(a)]
        boot[k] = {"ci90": [float(np.percentile(f, 5)),
                            float(np.percentile(f, 95))] if f.size else
                   [None, None], "n_undefined": int((~np.isfinite(a)).sum()),
                   "draws": a}

    cheap_judges = {d: bound["datasets"][d]["cheap_judges"] for d in DATASETS}
    drops = bound_drops(bound, cheap_judges)
    implied: dict = {}
    for d in DATASETS:
        implied[d] = {"bound_drop_cheap": drops[d]["cheap"],
                      "bound_drop_flagship40": drops[d]["flagship40"]}
        for k in ("F_pooled", "F_weighted"):
            mv = est[k]
            dr = boot[k]["draws"] * drops[d]["cheap"]
            f = dr[np.isfinite(dr)]
            implied[d][k] = {
                "genuine_drop": None if mv is None else mv * drops[d]["cheap"],
                "ci90": [float(np.percentile(f, 5)),
                         float(np.percentile(f, 95))] if f.size else
                [None, None],
                "genuine_drop_flagship40": {
                    mm: (None if mv is None else mv * x)
                    for mm, x in drops[d]["flagship40"].items()}}

    res = {"meta": {"script": "scripts/blind_audit_sample.py analyze",
                    "git_hash": ic.b4.git_hash(),
                    "scores_sha256": ssha, "mapping_sha256":
                    man["sha256"][MAPPING], "B": args.b, "seed": args.seed,
                    "ci": "exact binomial (Clopper-Pearson) 90% for shares; "
                          "item bootstrap within stratum x dataset, 90% "
                          "percentile for m̂", "weights": w,
                    "population": pop, "timestamp_utc": now_utc()},
           "shares": shares,
           "mhat": {k: {"point": est[k], "ci90": boot[k]["ci90"],
                        "n_undefined": boot[k]["n_undefined"]} for k in est},
           "implied": implied}
    write_json(od / ANALYSIS_JSON, res)

    print("\nBA ANALYSIS (exploratory; author blind audit)")
    for s in STRATA:
        r = shares[f"{s}|pooled"]
        per = "; ".join(f"{d} {shares[f'{s}|{d}']['k_ge2']}/"
                        f"{shares[f'{s}|{d}']['n']}" for d in DATASETS)
        print(f"  {s}: ≥2 {r['k_ge2']}/{r['n']} = {_p(r['p_ge2'])} "
              f"{_c(r['ci_ge2'])} ({per}) | ≥1 {r['k_ge1']}/{r['n']} = "
              f"{_p(r['p_ge1'])} {_c(r['ci_ge1'])}")
    for k in est:
        print(f"  m̂({k}) = {_p(est[k])} {_c(boot[k]['ci90'])} "
              f"(undefined {boot[k]['n_undefined']}/{args.b})")
    print("\n" + "=" * 30 + " BACKFILL (paste under the BA registration) "
          + "=" * 30)
    L = ["## BA 开箱结果（作者盲审，回填）", "",
         f"- 评分文件 sha256[scores]: {ssha}；mapping sha256 与 manifest 一致",
         "- 作者评分 ≥2 占比（DL19+DL20 合并；精确二项 90% CI；分数据集计数）："]
    for s in STRATA:
        r = shares[f"{s}|pooled"]
        per = "，".join(f"{d} {shares[f'{s}|{d}']['k_ge2']}/"
                       f"{shares[f'{s}|{d}']['n']}" for d in DATASETS)
        L.append(f"  - {s}：{r['k_ge2']}/{r['n']} = {_p(r['p_ge2'])} "
                 f"{_c(r['ci_ge2'])}（{per}）；≥1：{r['k_ge1']}/{r['n']} = "
                 f"{_p(r['p_ge1'])} {_c(r['ci_ge1'])}")
    L.append(f"- m̂（bootstrap 90% CI，B = {args.b}，seed {args.seed}）：")
    for k in est:
        L.append(f"  - m̂({k}) = {_p(est[k])} {_c(boot[k]['ci90'])}"
                 f"（无定义 {boot[k]['n_undefined']}/{args.b}）")
    L.append(f"  - F 加权权重：F2 {w['F2']:.3f} / F3 {w['F3']:.3f}"
             f"（≥2/5 flag 集总体：F2 {n2}，F3 {n3}）")
    L.append("- 隐含真漏标 Δ 降幅 = m̂ × relabel bound（≥2/5）Δ 降幅（cheap "
             "judge）：")
    for d in DATASETS:
        ip = implied[d]
        fl = list(ip["bound_drop_flagship40"].values())
        flt = (f"；flagship 40% bound 降幅 {min(fl):.3f}–{max(fl):.3f}"
               if fl else "")
        L.append(f"  - {d}：bound 降幅 {ip['bound_drop_cheap']:.3f} → F 合并 "
                 f"{_p(ip['F_pooled']['genuine_drop'])} "
                 f"{_c(ip['F_pooled']['ci90'])}；F 加权 "
                 f"{_p(ip['F_weighted']['genuine_drop'])} "
                 f"{_c(ip['F_weighted']['ci90'])}{flt}")
    L.append("- 披露：标注者为作者本人（非 NIST 评估员），探索性、不计分；m̂ "
             "假定作者与评估员同尺度，H1/H2 层校准作者对“judge 判相关”对的"
             "基线。")
    print("\n".join(L))
    print("=" * 96)
    print(f"[ba] wrote {od / ANALYSIS_JSON}")
    return res


# ═══════════════════════════════════════════════════════════════════════════
# selftest — synthetic, offline
# ═══════════════════════════════════════════════════════════════════════════

ST_CHEAP = [f"stub/cheap{i}" for i in range(M_PANEL)]
ST_FLAG = "stub/flag"
_WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet "
          "kilo lima mike november oscar papa quebec romeo sierra tango "
          "uniform victor whiskey xray yankee zulu").split()


def _st_build(root: Path, seed: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    qrels, latent, texts = {}, {}, {}
    for ds in DATASETS:
        for qi in range(24):
            q = f"Q{ds[-2:]}x{qi:02d}"
            for di in range(26):
                d = f"D{ds[-2:]}x{qi:02d}y{di:02d}"
                g = int(rng.choice(4, p=[0.45, 0.25, 0.18, 0.12]))
                qrels[(ds, q, d)] = g
                bump = 1.6 if (g == 0 and rng.random() < 0.25) else 0.0
                latent[(ds, q, d)] = g + bump
                words = " ".join(rng.choice(_WORDS, size=12))
                texts[(ds, q, d)] = f"{words} {qi * 26 + di} {ds[-1]}"
            # short and duplicate passages to exercise the exclusions
            ks = sorted(k for k in qrels if k[0] == ds and k[1] == q)
            for k in ks[:3]:
                texts[k] = "n/a"
            texts[ks[5]] = texts[ks[4]]
            texts[ks[7]] = texts[ks[6]]
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    (root / "derived").mkdir(parents=True, exist_ok=True)
    fsel = {}
    for ds in DATASETS:
        keys = sorted(k for k in qrels if k[0] == ds)
        fsel[ds] = {(q, d) for (_, q, d) in keys if rng.random() < 0.4}
        with open(root / "inputs" / f"pairs_{ds}.jsonl", "w") as f, \
                open(root / "inputs" / f"pairs_{ds}_flagship40.jsonl",
                     "w") as g:
            for (_, q, d) in keys:
                line = json.dumps({"dataset": ds, "qid": q, "docid": d,
                                   "query": f"what is {q[-2:]} about",
                                   "passage": texts[(ds, q, d)]}) + "\n"
                f.write(line)
                if (q, d) in fsel[ds]:
                    g.write(line)
    import pandas as pd
    rows = []
    for m in ST_CHEAP + [ST_FLAG]:
        bias = rng.normal(0, 0.15)
        for (ds, q, d), x0 in latent.items():
            if m == ST_FLAG and (q, d) not in fsel[ds]:
                continue
            item = rng.normal(0, 0.75)
            for lvl in ic.LEVELS:
                for par in "ab":
                    x = x0 + bias + item + rng.normal(0, 0.3)
                    s = float(np.clip(np.floor(x + 0.5), 0, 3))
                    rows.append((m, f"L{lvl}_{par}", lvl, ds, q, d, 0, s,
                                 True))
    df = pd.DataFrame(rows, columns=["model_id", "prompt_id",
                                     "politeness_level", "dataset", "qid",
                                     "docid", "run", "score", "parse_ok"])
    jp = root / "derived" / "judgments.pkl"
    df.to_pickle(jp)
    qj = root / "qrels_stub.json"
    qj.write_text(json.dumps([[ds, q, d, g] for (ds, q, d), g in
                              sorted(qrels.items())]))
    ns = argparse.Namespace(data_dir=str(root), judgments=str(jp),
                            qrels_json=str(qj),
                            judges=",".join(ST_CHEAP + [ST_FLAG]),
                            cheap=",".join(ST_CHEAP), flagships=ST_FLAG)
    ctx = ic.load_inputs(ns, list(DATASETS))
    cells = ic.build_cells(ctx, ic.SEED)
    res = ic.run_addons(ctx, cells, ["bound"], 50, ic.SEED)
    bp = root / "derived" / "incompleteness" / "ic_addons_bound.json"
    bp.parent.mkdir(parents=True, exist_ok=True)
    bp.write_text(json.dumps(ic.b6._jclean(
        {"meta": {"input_sha256": ic.input_shas(ctx)},
         "datasets": res["bound"]}), ensure_ascii=False))
    return {"jp": jp, "qj": qj, "bp": bp, "texts": texts}


def _ns(root: Path, w: dict, **kw) -> argparse.Namespace:
    base = dict(data_dir=str(root), judgments=str(w["jp"]),
                qrels_json=str(w["qj"]), judges=",".join(ST_CHEAP + [ST_FLAG]),
                cheap=",".join(ST_CHEAP), flagships=ST_FLAG, seed=SEED,
                out=None, bound=None, force=False)
    base.update(kw)
    return argparse.Namespace(**base)


def _exits(fn) -> bool:
    import contextlib
    import io
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fn()
    except SystemExit:
        return True
    return False


def _quiet(fn):
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn()
    return out, buf.getvalue()


def cmd_selftest(_args) -> None:
    print("=" * 96)
    print("  BA SELF-TEST (synthetic; no Drive, no network)")
    print("=" * 96)
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" +
              (f" — {detail}" if detail and not ok else ""))
        if not ok:
            fails.append(name)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        w, _ = _quiet(lambda: _st_build(root))
        od = root / "derived" / "incompleteness" / "blind_audit"
        out, log = _quiet(lambda: cmd_sample(_ns(root, w)))
        man, rows = out["manifest"], out["rows"]
        mp = json.loads((od / MAPPING).read_text())["items"]

        keys = {(r["dataset"], r["qid"], r["docid"]) for r in rows}
        check("60 items, 60 unique pairs", len(rows) == 60 and len(keys) == 60)
        cnt = {(d, s): 0 for d in DATASETS for s in STRATA}
        for r in mp.values():
            cnt[(r["dataset"], r["stratum"])] += 1
        check("strata counts per dataset = quota",
              all(cnt[(d, s)] == QUOTA[s] for d in DATASETS for s in STRATA),
              str(cnt))
        check("no shortfall recorded", man["shortfall"] == {})
        check("stratum membership (h, votes) correct",
              all(in_stratum(r["stratum"], r["human_grade"], r["votes"])
                  and r["votes"] == sum(v >= 2 for v in r["raw_lab"].values())
                  for r in mp.values()))
        ps = [r["passage"].strip() for r in rows]
        check("no passage < 20 chars, no duplicate passage",
              min(len(p) for p in ps) >= MIN_PASSAGE_CHARS
              and len(set(ps)) == len(ps))
        sk = sum(man["counts"][d][s]["skipped_short"] +
                 man["counts"][d][s]["skipped_dup"]
                 for d in DATASETS for s in STRATA)
        check("exclusion rules exercised (skips > 0)", sk > 0, str(sk))

        with open(od / SHEET_CSV, newline="") as f:
            rd = list(csv.reader(f))
        check("sheet columns exactly item,query,passage,score,note",
              tuple(rd[0]) == SHEET_COLS)
        body = rd[1:]
        check("sheet: items 1..60 in order, score/note empty",
              [int(r[0]) for r in body] == list(range(1, 61))
              and all(r[3] == "" and r[4] == "" for r in body))
        raw_csv = (od / SHEET_CSV).read_text()
        raw_md = (od / SHEET_MD).read_text()
        leak = [t for r in mp.values() for t in (r["qid"], r["docid"])
                if t in raw_csv or t in raw_md]
        leak += [t for t in ("dl19", "dl20", "stratum", "votes", "human",
                             "F2", "F3", "C0", "H1", "H2", "qid", "docid")
                 if t in raw_csv or t in raw_md]
        check("no leakage (ids, dataset, stratum, votes, grade) in sheets",
              not leak, str(leak[:5]))
        check("md: 60 '### Item k' blocks matching the csv",
              all(f"### Item {r['item']}\n\n**Query:** {r['query']}\n\n"
                  f"**Passage:** {r['passage']}\n\nScore: ___\n\nNote: ___"
                  in raw_md for r in rows)
              and raw_md.count("### Item ") == 60)
        check("sheet text = mapping pair text",
              all(w["texts"][(mp[r[0]]["dataset"], mp[r[0]]["qid"],
                              mp[r[0]]["docid"])] == r[2] for r in body))

        order = np.array([mp[str(i)]["sample_order"] for i in range(1, 61)])
        check("item order is a permutation of the sampling order",
              sorted(order.tolist()) == list(range(60)))
        check("item order != sampling order",
              not np.array_equal(order, np.arange(60)))
        seq = [mp[str(i)]["stratum"] + mp[str(i)]["dataset"]
               for i in range(1, 61)]
        runs = 1 + sum(a != b for a, b in zip(seq, seq[1:]))
        rho = ic.spearman(np.arange(60, dtype=float), order.astype(float))
        check("no stratum-correlated ordering (runs ≥ 30, |ρ(item, "
              "sample order)| < 0.4)", runs >= 30 and abs(rho) < 0.4,
              f"runs {runs}, rho {rho:.3f}")

        check("manifest sha256s match the files",
              all(man["sha256"][n] == sha256_file(od / n)
                  for n in (SHEET_CSV, SHEET_MD, MAPPING)))
        reg = log.split("REGISTRATION BLOCK")[1]
        check("registration block: heading, 3 sha lines, no per-item data",
              f"# {BA_MARKER}" in reg
              and all(f"sha256[{n}]: {man['sha256'][n]}" in reg
                      for n in (SHEET_CSV, SHEET_MD, MAPPING))
              and not any(t in reg for r in mp.values()
                          for t in (r["qid"], r["docid"]))
              and not any(r["passage"][:30] in reg for r in rows))
        ctl_ok = True
        for d in DATASETS:
            for s in CONTROL:
                st = man["counts"][d][s]
                if st["taken_B"] > 0 and not st["exhausted_A"]:
                    ctl_ok = False
            fq = {r["qid"] for r in mp.values()
                  if r["dataset"] == d and r["stratum"] in F_STRATA}
            for r in mp.values():
                if r["dataset"] == d and r["stratum"] in CONTROL:
                    ctl_ok &= (r["tier"] == "A") == (r["qid"] in fq)
        check("same-query preference: tier A first, tier B only if A "
              "exhausted", ctl_ok)

        check("refuses to overwrite without --force",
              _exits(lambda: cmd_sample(_ns(root, w))))
        od2 = root / "again"
        _quiet(lambda: cmd_sample(_ns(root, w, out=str(od2))))
        check("deterministic: same seed -> identical sheet and mapping",
              sha256_file(od2 / SHEET_CSV) == man["sha256"][SHEET_CSV]
              and sha256_file(od2 / MAPPING) == man["sha256"][MAPPING])
        od3 = root / "other"
        _quiet(lambda: cmd_sample(_ns(root, w, out=str(od3), seed=7)))
        check("different seed -> different sheet",
              sha256_file(od3 / SHEET_CSV) != man["sha256"][SHEET_CSV])

        QUOTA["F2"] = 10 ** 6
        try:
            o4, log4 = _quiet(lambda: cmd_sample(_ns(root, w,
                                                     out=str(root / "big"))))
        finally:
            QUOTA["F2"] = 8
        m4 = o4["manifest"]
        st4 = [m4["counts"][d]["F2"] for d in DATASETS]
        check("stratum with too few candidates: takes all eligible, shortfall "
              "in manifest + block",
              all(s["exhausted_all"] and s["shortfall"] > 0
                  and s["taken"] + s["skipped_short"] + s["skipped_dup"]
                  == s["candidates"] for s in st4)
              and set(m4["shortfall"]) == {f"{d}|F2" for d in DATASETS}
              and "不足配额" in log4)
        print("  --- registration block as printed (synthetic run) ---")
        print("\n".join("  | " + ln for ln in
                        log.split("REGISTRATION BLOCK")[1].splitlines()[1:]))

        b = json.loads(w["bp"].read_text())
        b["datasets"]["dl20"]["rules"]["3"]["flagged_pairs"].pop()
        tb = root / "tampered_bound.json"
        tb.write_text(json.dumps(b))
        check("refuses when the bound's flag set differs",
              _exits(lambda: cmd_sample(_ns(root, w, out=str(root / "t"),
                                            bound=str(tb)))))
        b = json.loads(w["bp"].read_text())
        b["meta"]["input_sha256"]["judgments"] = "0" * 64
        tb.write_text(json.dumps(b))
        check("refuses when the judgments sha differs from the bound meta",
              _exits(lambda: cmd_sample(_ns(root, w, out=str(root / "t2"),
                                            bound=str(tb)))))

        # ── analyze: known m̂ ──
        k_target = {"F2": 5, "F3": 11, "C0": 1, "H1": 2, "H2": 8}
        by_s: dict = {s: [] for s in STRATA}
        for it in sorted(mp, key=int):
            by_s[mp[it]["stratum"]].append(it)
        score = {}
        for s, its in by_s.items():
            for j, it in enumerate(its):
                score[it] = (3 if j % 2 else 2) if j < k_target[s] else \
                    (1 if j % 2 else 0)
        sp = root / "scores.csv"
        with open(sp, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["item", "score"])
            for it in sorted(score, key=int):
                wr.writerow([it, score[it]])
        an = _ns(root, w, scores=str(sp), mapping=str(od / MAPPING),
                 manifest=None, bound=None, unseal=False, b=B_BOOT,
                 seed=BOOT_SEED)
        check("analyze refuses without --unseal", _exits(
            lambda: cmd_analyze(an)))
        an.unseal = True
        check("analyze refuses without UNSEAL_OK", _exits(
            lambda: cmd_analyze(an)))
        (od / UNSEAL_FILE).write_text("ok\n")
        res, alog = _quiet(lambda: cmd_analyze(an))
        exp = {"F2": (5 / 16 - 0.2) / 0.6, "F3": (11 / 16 - 0.2) / 0.6,
               "F_pooled": (16 / 32 - 0.2) / 0.6}
        check("analyze reproduces known m̂ (F2 0.1875, F3 0.8125, F 0.5)",
              all(abs(res["mhat"][k]["point"] - v) < 1e-12
                  for k, v in exp.items()),
              str({k: res["mhat"][k]["point"] for k in exp}))
        wts = res["meta"]["weights"]
        pw = wts["F2"] * 5 / 16 + wts["F3"] * 11 / 16
        check("F weighted m̂ = population-weighted p_F",
              abs(res["mhat"]["F_weighted"]["point"] - (pw - 0.2) / 0.6)
              < 1e-12)
        check("shares + Clopper-Pearson: H1 2/10, CI brackets p; C0 ≥1 "
              "consistent",
              res["shares"]["H1|pooled"]["k_ge2"] == 2
              and res["shares"]["H1|pooled"]["ci_ge2"][0] < 0.2
              < res["shares"]["H1|pooled"]["ci_ge2"][1]
              and abs(cp_ci(0, 10)[1] - (1 - 0.05 ** 0.1)) < 1e-9)
        bd = bound_drops(json.loads(w["bp"].read_text()),
                         {d: ST_CHEAP for d in DATASETS})
        check("implied drop = m̂(F pooled) × bound Δ drop per dataset",
              all(abs(res["implied"][d]["F_pooled"]["genuine_drop"]
                      - 0.5 * bd[d]["cheap"]) < 1e-12 for d in DATASETS)
              and all(bd[d]["cheap"] > 0 for d in DATASETS))
        check("bootstrap CI brackets the point estimate",
              all(res["mhat"][k]["ci90"][0] - 1e-12 <= res["mhat"][k]["point"]
                  <= res["mhat"][k]["ci90"][1] + 1e-12 for k in exp))
        check("BACKFILL block printed", "## BA 开箱结果" in alog)
        sp2 = root / "scores_bad.csv"
        sp2.write_text("item,score\n" + "\n".join(
            f"{it},{score[it]}" for it in sorted(score, key=int)[:-1]) + "\n")
        an.scores = str(sp2)
        check("analyze refuses incomplete scores",
              _exits(lambda: cmd_analyze(an)))
        an.scores = str(sp)
        mtxt = (od / MAPPING).read_text()
        (od / MAPPING).write_text(mtxt.replace('"F2"', '"F3"', 1))
        check("analyze refuses a mapping whose sha256 changed",
              _exits(lambda: cmd_analyze(an)))
        (od / MAPPING).write_text(mtxt)

    print("-" * 96)
    if fails:
        print(f"SELFTEST FAILED: {len(fails)} check(s): {fails}")
        sys.exit(1)
    print("ALL SELFTESTS PASSED")


# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("sample", help="draw the blinded 60-item sheet")
    ps.add_argument("--data-dir", default=None)
    ps.add_argument("--judgments", default=None)
    ps.add_argument("--qrels-json", default=None, help="TEST hook")
    ps.add_argument("--judges", default=None, help="override roster (TEST)")
    ps.add_argument("--cheap", default=None, help="override cheap set (TEST)")
    ps.add_argument("--flagships", default=None, help="override (TEST)")
    ps.add_argument("--bound", default=None, help="ic_addons_bound.json "
                    "(default $DATA_DIR/derived/incompleteness/)")
    ps.add_argument("--seed", type=int, default=SEED)
    ps.add_argument("--out", default=None, help="default $DATA_DIR/derived/"
                    "incompleteness/blind_audit/")
    ps.add_argument("--force", action="store_true")
    pa = sub.add_parser("analyze", help="GATED: needs --unseal and UNSEAL_OK")
    pa.add_argument("--scores", required=True)
    pa.add_argument("--mapping", required=True)
    pa.add_argument("--manifest", default=None)
    pa.add_argument("--bound", default=None)
    pa.add_argument("--unseal", action="store_true")
    pa.add_argument("--b", type=int, default=B_BOOT)
    pa.add_argument("--seed", type=int, default=BOOT_SEED)
    sub.add_parser("selftest", help="synthetic offline checks")
    args = ap.parse_args()
    if args.cmd == "sample":
        cmd_sample(args)
    elif args.cmd == "analyze":
        cmd_analyze(args)
    else:
        cmd_selftest(args)


if __name__ == "__main__":
    main()
