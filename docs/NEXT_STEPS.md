# Roadmap: Full Paper → SIGIR 2027

Target: SIGIR 2027 full paper (deadline ~mid/late Jan 2027).
Parallel track: RecSys 2026 R&P Notes (2-page, finalized) — submit by Jul 15,
2026 AoE; notification Aug 10; camera-ready Aug 17.

## Status snapshot (Jul 2026)

- Data collection complete: 8 judges × 15 prompt variants × TREC DL19/DL20
  (3,498 pairs; flagship tier on frozen 40% subsample).
- Headline results locked (hand-verified, see PREDICTIONS.md): Spearman
  ρ = −0.62 (p = 0.0005) over 28 (model, level) cells; ρ = −0.68 excl.
  DeepSeek; all ΔNDCG@10 ≤ 0.033; Kendall τ ∈ [0.74, 0.94].
- Notes-version improvements folded into full .tex (hedged framing,
  pre-specified wording, κ-independent corroboration). Cleaned
  references.bib delivered (jedidi2025overthink key rename; zheng2023judging
  added; all note fields stripped).

## IMPORTANT: data-source discipline

All new analyses MUST reproduce the authoritative numbers from the
hand-verified analysis path (PREDICTIONS.md) before extending them. A fresh
recompute from derived/judgments.parquet yields a slightly different DeepSeek
D (−0.148 vs script −0.073); rank-based conclusions are unaffected, but any
new script must first assert agreement with the authoritative κ / Δ / D
tables and fail loudly if they diverge.

## Batch 1 — Statistical hardening (pure analysis, $0) — Jul–Aug

1. **P0.1 Independence fixes** (addresses known reviewer objection: 28 cells
   share per-model Δ, so pooled Spearman p is optimistic)
   - Model-clustered permutation test: permute level labels within each
     model, recompute ρ, build null distribution (≥10,000 permutations).
   - Leave-one-model-out table: ρ and p for each of the 7 held-out configs.
   - Mixed-effects model: Δκ ~ A(ℓ) with random intercept (and slope if it
     converges) per judge; statsmodels MixedLM or R lme4.
   - Outcome: replace Limitations item 2 with a Methods/Results subsection.
2. **P1.3 Equivalence testing** (upgrade null results to bounded claims)
   - Query-level bootstrap (resample queries with replacement, keep all
     pairs of a query together) → 95% CI for every (model, level) Δκ.
   - TOST-style statement: smallest margin m such that |Δκ| < m is supported
     at 95% for the six non-DeepSeek judges (target claim: m ≈ 0.02).
3. **P1.4 U-shape significance** (replace btw/in heuristic with a test)
   - Paraphrase-level permutation: under H0 level labels are exchangeable
     across the 15 variants; test statistic = between-level κ variance.
   - Report p per model; expected: significant for DeepSeek only. Resolves
     the Opus edge case (btw/in = 1.37, no U-shape) within one framework.

Deliverables: one analysis script (or notebook) + LaTeX-ready tables +
updated figures if needed. Keep raw outputs append-only per repo convention.

## Batch 2 — Dataset generalization (P0.2, ~$100–200) — Aug–Sep

- Extend to TREC DL21/DL22 (MS MARCO v2 passage corpus, NIST graded qrels).
- Prep step: obtain DL21/22 qrels, build BM25 top-50 ∩ qrels intersection,
  freeze pair set before collection (mirror DL19/20 protocol).
- Run cost-efficient tier first (5 models × 15 variants × 2 runs); decide on
  flagship tier after seeing Batch 1 + cost.
- Reuse existing collection pipeline (resume keys: model_id, prompt_id, qid,
  docid, run).

## Batch 3 — Extensions — Oct onward

4. **P1.5 Reasoning judges** (~$50–100): add 2–3 reasoning-capable judges
   (candidates: DeepSeek reasoner, Qwen thinking mode, GPT-5.5 with
   reasoning enabled) on the 40% subsample. Goal: upgrade tone × reasoning
   from n=1 exploratory (Gemini 3.1 Pro) to a systematic finding. Budget
   note: L1 (rude) inflates reasoning tokens ~+120%, so cost is level-skewed.
5. **P2 Pairwise judges** (if time permits): mechanism predicts pairwise
   judging is tone-immune (consumes only relative order). Small-scale run
   (1 dataset, ~3 models) = third independent evidence line.

## Writing & logistics (parallel)

- Related Work: expand the Arabzadeh & Clarke (SIGIR '25) positioning to
  2–3 sentences (what they did; orthogonality: calibrated tone + mechanism).
- Anonymization for submission build: \documentclass[sigconf,anonymous,review]
  and an anonymized supplementary link (anonymous.4open.science).
- Legal (before any signed public release): verify Snap employment contract
  for IP-assignment / publication-approval clauses; resolve between RecSys
  notification (Aug 10) and arXiv/camera-ready.

## Timeline

| Window        | Work                                              |
|---------------|---------------------------------------------------|
| Jul 2026      | Submit RecSys Notes (Jul 15); finish .tex/.bib    |
| Jul–Aug 2026  | Batch 1 (P0.1 + P1.3 + P1.4)                      |
| Aug 2026      | RecSys notification (Aug 10); Snap IP review      |
| Aug–Sep 2026  | Batch 2 (DL21/22 collection + analysis)           |
| Oct 2026      | Batch 3a (reasoning judges)                       |
| Nov–Dec 2026  | Writing integration; P2 pairwise if time permits  |
| Jan 2027      | SIGIR 2027 submission                             |
