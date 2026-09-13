"""Multi-dataset qrels loading, grade normalisation, and the dl22 seal.

This module is the single source of truth for turning human relevance
judgments into the 4-point scale the LLM judge rubric uses.

ir_datasets ids
---------------
    dl19     msmarco-passage/trec-dl-2019/judged      grades 0-3   (round 1)
    dl20     msmarco-passage/trec-dl-2020/judged      grades 0-3   (round 1)
    dl21     msmarco-passage-v2/trec-dl-2021/judged   grades 0-3   (round 2)
    dl22     msmarco-passage-v2/trec-dl-2022/judged   grades 0-3   (round 2)
    antique  antique/test                             grades 1-4   (round 2)

ANTIQUE GRADE MAPPING  (read this before using ANTIQUE numbers anywhere)
------------------------------------------------------------------------
ANTIQUE ships **raw 1-4** relevance grades, while TREC DL ships 0-3 and our
judge rubric (config/prompts.yaml) emits 0-3.  ``load_qrels`` maps ANTIQUE to
the common scale by **subtracting 1**:

    0-3 grade = raw ANTIQUE grade - 1        (1->0, 2->1, 3->2, 4->3)

Why this mapping and why it is safe:

* Linear-weighted Cohen's kappa needs judge and human on **one common
  4-point scale**; comparing a 0-3 judge against a 1-4 human would inflate
  every disagreement by a constant offset and make kappa meaningless.
* The map ``g -> g - 1`` is **affine and strictly order-preserving**, so it
  changes neither the ordering of documents nor the *shape* of the grade
  distribution.
* Linear-weighted kappa depends on the grades only through the pairwise
  distances ``|i - j|`` and the marginal distributions, both of which are
  **invariant** under a common shift of the two rating scales.  Applying the
  same shift to the human side that the rubric already implies for the judge
  side therefore leaves the weighted kappa unchanged relative to any other
  order-preserving affine alignment of the two scales.
* This is a *presentation* alignment, not a semantic claim: ANTIQUE's grade 1
  ("not relevant / off-topic") is treated as the analogue of TREC DL's 0.
  Anything downstream that needs the raw ANTIQUE grade must read the ``rel``
  field on the round-2 pairs files, which stores grades **unmapped** (see
  src/build_pairs.py).

The mapping is applied **only** to ANTIQUE; every other dataset is passed
through unchanged.

Sealed holdout
--------------
``dl22`` is a SEALED holdout.  No kappa or any other judge-vs-human agreement
metric may be computed on dl22 until the round-2 correction predictions are
committed.  Distribution preflight (score histograms, Delta, D(l)) is BLIND
and therefore allowed on dl22.  Call :func:`assert_not_sealed` before any
agreement computation; the only override is an explicit ``--unseal-dl22``
flag threaded through from the CLI.
"""

from __future__ import annotations

import sys

# ir_datasets identifiers, keyed by our short dataset name (as stored in the
# ``dataset`` column of derived/judgments.parquet).
DATASETS: dict[str, str] = {
    "dl19": "msmarco-passage/trec-dl-2019/judged",
    "dl20": "msmarco-passage/trec-dl-2020/judged",
    "dl21": "msmarco-passage-v2/trec-dl-2021/judged",
    "dl22": "msmarco-passage-v2/trec-dl-2022/judged",
    "antique": "antique/test",
}

ROUND1 = ("dl19", "dl20")
ROUND2 = ("dl21", "dl22", "antique")

# Datasets whose *raw* grades are 1-4 and are shifted down by 1 (see module
# docstring).  Everything not listed here is assumed to already be 0-3.
GRADE_OFFSET: dict[str, int] = {
    "antique": -1,
}

# Sealed holdout: no agreement metric until correction predictions are frozen.
SEALED: frozenset[str] = frozenset({"dl22"})

SEAL_MESSAGE = """
================================================================================
  PROTOCOL VIOLATION BLOCKED - dl22 IS A SEALED HOLDOUT
================================================================================
  You asked for a judge-vs-human agreement metric (kappa or equivalent) on a
  dataset list that includes dl22:

      requested datasets : {requested}
      sealed             : {sealed}

  dl22 stays sealed until the round-2 correction predictions are COMMITTED to
  paper/PREDICTIONS.md.  Looking at dl22 kappa before that destroys the blind
  status of every round-2 prediction and cannot be undone.

  Allowed on dl22 right now:
      - scripts/preflight_distributions.py  (blind: distributions, Delta, D)

  Not allowed on dl22 right now:
      - src/metrics.py agreement_table / scripts/validate_model.py
      - anything reporting kappa, accuracy, or correlation against qrels

  If (and only if) the correction predictions are already committed, re-run
  with the explicit override flag:  --unseal-dl22
================================================================================
"""


def assert_not_sealed(datasets, unseal: bool = False) -> None:
    """Abort loudly if a sealed dataset is about to be used for agreement.

    Call this immediately before ANY kappa / agreement computation, never
    before a purely distributional (blind) computation.

    Args:
        datasets: iterable of short dataset names.
        unseal:   True only when the operator passed ``--unseal-dl22``.
    """
    requested = [str(d) for d in datasets]
    hit = sorted(set(requested) & set(SEALED))
    if not hit:
        return
    if unseal:
        print(
            "[UNSEALED] dl22 agreement metrics enabled via --unseal-dl22. "
            "This is only legitimate if round-2 correction predictions are "
            "already committed to paper/PREDICTIONS.md.",
            file=sys.stderr,
        )
        return
    sys.exit(
        SEAL_MESSAGE.format(
            requested=",".join(requested) or "(none)",
            sealed=",".join(sorted(SEALED)),
        )
    )


def parse_datasets(spec: str) -> list[str]:
    """Parse a ``--datasets`` comma list, validating every name."""
    names = [s.strip() for s in str(spec).split(",") if s.strip()]
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        sys.exit(
            f"ERROR: unknown dataset(s) {unknown}. "
            f"Known: {sorted(DATASETS)}"
        )
    if not names:
        sys.exit("ERROR: --datasets is empty.")
    # de-duplicate, preserve order
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def load_qrels(datasets: list[str], quiet: bool = False) -> dict:
    """Load human grades for ``datasets``, normalised to a common 0-3 scale.

    Returns:
        dict keyed by ``(dataset, qid, docid)`` -> int grade in 0-3.
        Keys carry the dataset name because query ids are only unique
        *within* a collection (msmarco-passage v1 / v2 / ANTIQUE all use bare
        integers), so a 2-tuple key would silently cross-contaminate.

    ANTIQUE grades are shifted by -1; see the module docstring for the full
    justification.  Grades are NEVER clipped: a raw grade outside the range
    the dataset is documented to use (0-3 for TREC DL, 1-4 for ANTIQUE) means
    an ir_datasets version or dataset-id surprise, and silently mutating it
    would corrupt every kappa downstream — so it aborts instead.
    """
    try:
        import ir_datasets
    except ImportError:
        sys.exit("ERROR: ir_datasets not installed. Run: pip install ir_datasets")

    qrels: dict[tuple[str, str, str], int] = {}
    for name in datasets:
        if name not in DATASETS:
            sys.exit(f"ERROR: unknown dataset '{name}'. Known: {sorted(DATASETS)}")
        offset = GRADE_OFFSET.get(name, 0)
        raw_lo, raw_hi = -offset, 3 - offset     # documented raw range
        ds = ir_datasets.load(DATASETS[name])
        n = 0
        for j in ds.qrels_iter():
            raw = int(j.relevance)
            if raw < raw_lo or raw > raw_hi:
                sys.exit(
                    f"ERROR: {name} ({DATASETS[name]}) returned relevance "
                    f"grade {raw} for (qid={j.query_id}, docid={j.doc_id}), "
                    f"outside the documented raw range {raw_lo}-{raw_hi}.\n"
                    f"  Grades are never clipped: a shifted or clipped grade "
                    f"would silently corrupt every\n"
                    f"  linear-weighted kappa. Check the ir_datasets version "
                    f"and the dataset id, then update\n"
                    f"  GRADE_OFFSET / this range in src/qrels.py deliberately."
                )
            qrels[(name, str(j.query_id), str(j.doc_id))] = raw + offset
            n += 1
        if not quiet:
            note = ""
            if offset:
                note = (f"  [grades shifted by {offset:+d}: "
                        f"raw {raw_lo}-{raw_hi} -> 0-3]")
            print(f"[qrels] {name:<8} {DATASETS[name]:<42} {n:>8} judgments{note}")
    return qrels


def attach(df, qrels: dict):
    """Attach a dataset-aware ``human`` column and drop unmatched rows.

    ``df`` must carry ``dataset``, ``qid``, ``docid``.  This is the
    multi-dataset replacement for ``src.metrics.attach_qrels``'s 2-tuple
    lookup (that function still accepts both key shapes).
    """
    df = df.copy()
    df["human"] = [
        qrels.get((str(ds), str(q), str(d)))
        for ds, q, d in zip(df.dataset, df.qid, df.docid)
    ]
    return df.dropna(subset=["human"])
