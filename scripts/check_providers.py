"""Live-fire provider audit for config/models.yaml.

Two-layer check per model:
  1. Static:  GET /api/v1/models/{id}/endpoints — verify provider_order names
              are case-exact matches of live provider_name values.
  2. Live:    POST a real relevance-scoring request with the exact payload
              structure used by src/collect.py (pinned provider, reasoning
              params, temperature=0, max_tokens=200).

Usage:
    OPENROUTER_API_KEY=sk-... python scripts/check_providers.py
    python scripts/check_providers.py --skip-live   # static check only

Prints one result row per model, then a PASS/FAIL summary.
Does NOT modify models.yaml — fixes require a separate commit.
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

import yaml

YAML_PATH = Path("config/models.yaml")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# Fixed mini prompt for live-fire test — same rubric structure as real runs
LIVE_PROMPT = (
    "You are evaluating the relevance of a passage to a search query.\n\n"
    "Query: climate change effects\n"
    "Passage: Rising temperatures are causing glaciers to melt worldwide.\n\n"
    "Relevance scale:\n"
    "3 = dedicated and exact answer  2 = partial answer  "
    "1 = related but no answer  0 = unrelated\n\n"
    'Provide your assessment as the JSON object {"score": <0-3>} with no additional text.'
)


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only — no extra deps beyond pyyaml)
# ---------------------------------------------------------------------------

def _get(url: str, timeout: int = 15) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def _post(url: str, payload: dict, api_key: str, timeout: int = 60) -> tuple[int, dict]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


# ---------------------------------------------------------------------------
# Layer 1: static endpoint check
# ---------------------------------------------------------------------------

def static_check(model_id: str, pinned: list[str]) -> tuple[bool, str, list[str]]:
    """Return (ok, detail, available_provider_names)."""
    try:
        data = _get(f"{OPENROUTER_BASE}/models/{model_id}/endpoints")
        endpoints = data.get("data", {}).get("endpoints", [])
    except Exception as e:
        return False, f"endpoint fetch error: {e}", []

    live_names = [ep["provider_name"] for ep in endpoints]
    live_set = set(live_names)

    for name in pinned:
        if name not in live_set:
            suggestion = next((k for k in live_names if k.lower() == name.lower()), None)
            detail = f"'{name}' not in live providers"
            if suggestion:
                detail += f" (did you mean '{suggestion}'?)"
            return False, detail, live_names

    return True, "ok", live_names


# ---------------------------------------------------------------------------
# Layer 2: live-fire POST
# ---------------------------------------------------------------------------

def build_payload(model: dict, defaults: dict) -> dict:
    payload = {
        "model": model["model_id"],
        "temperature": defaults["temperature"],
        "top_p": defaults["top_p"],
        "max_tokens": defaults["max_tokens"],
        "messages": [{"role": "user", "content": LIVE_PROMPT}],
    }
    if model.get("provider_order"):
        payload["provider"] = {
            "order": model["provider_order"],
            "allow_fallbacks": False,
        }
    if model.get("reasoning_mode") == "disabled":
        payload["reasoning"] = {"enabled": False}
    else:
        effort = model.get("reasoning_effort", defaults.get("reasoning_effort"))
        if effort:
            payload["reasoning"] = {"effort": effort}
    return payload


def live_check(model: dict, defaults: dict, api_key: str) -> dict:
    """Return a result dict with all fields needed for the summary row."""
    payload = build_payload(model, defaults)
    t0 = time.time()
    status, body = _post(f"{OPENROUTER_BASE}/chat/completions", payload, api_key)
    latency = round(time.time() - t0, 2)

    result = {
        "status": status,
        "latency_s": latency,
        "actual_provider": None,
        "reasoning_tokens": None,
        "finish_reason": None,
        "content_preview": None,
        "cost": None,
        "ok": False,
        "fail_reason": None,
    }

    if status != 200:
        result["fail_reason"] = f"HTTP {status}: {str(body)[:120]}"
        return result

    choices = body.get("choices")
    if not choices:
        result["fail_reason"] = f"HTTP 200 but no choices: {str(body)[:120]}"
        return result

    choice = choices[0]
    content = choice.get("message", {}).get("content")
    usage = body.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}

    result["actual_provider"] = body.get("provider")
    result["reasoning_tokens"] = details.get("reasoning_tokens")
    result["finish_reason"] = choice.get("finish_reason")
    result["content_preview"] = (content or "")[:40]
    result["cost"] = usage.get("cost")

    # Four-condition PASS gate
    pinned = (model.get("provider_order") or [None])[0]
    fails = []
    if pinned and result["actual_provider"] != pinned:
        fails.append(
            f"provider='{result['actual_provider']}' (want '{pinned}')"
        )
    rsn = result["reasoning_tokens"]
    if rsn is not None and rsn != 0:
        fails.append(f"reasoning_tokens={rsn} (want 0)")
    if result["finish_reason"] != "stop":
        fails.append(f"finish_reason='{result['finish_reason']}' (want 'stop')")

    if fails:
        result["fail_reason"] = "; ".join(fails)
        return result

    result["ok"] = True
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-live", action="store_true",
                    help="run static endpoint check only (no API key required)")
    args = ap.parse_args()

    cfg = yaml.safe_load(YAML_PATH.read_text())
    defaults = cfg["defaults"]
    models = cfg["models"]

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not args.skip_live and not api_key:
        print("ERROR: OPENROUTER_API_KEY not set. Use --skip-live for static check only.",
              file=sys.stderr)
        sys.exit(1)

    # ── column headers ──────────────────────────────────────────────────────
    hdr = "{:<40} {:<8} {:<12} {:<18} {:<10} {:<12} {:<42} {}"
    div = "-" * 160
    print(hdr.format(
        "model_id", "HTTP", "provider", "actual_provider",
        "rsn_tok", "finish", "content[:40]", "cost_usd",
    ))
    print(div)

    summary: list[tuple[str, bool, str]] = []  # (model_id, pass, reason)

    for m in models:
        model_id = m["model_id"]
        pinned = (m.get("provider_order") or [None])[0]

        # — static check —
        s_ok, s_detail, live_names = static_check(model_id, m.get("provider_order", []))
        if not s_ok:
            row = hdr.format(
                model_id[:39], "—", pinned or "—", "—", "—", "—", "—",
                f"STATIC FAIL: {s_detail}",
            )
            print(row)
            if live_names:
                print(f"  available providers: {live_names}")
            summary.append((model_id, False, f"static: {s_detail}"))
            continue

        if args.skip_live:
            print(hdr.format(model_id[:39], "—", pinned or "—",
                             "—", "—", "—", "—", "STATIC OK (live skipped)"))
            summary.append((model_id, True, "static ok"))
            continue

        # — live-fire check —
        r = live_check(m, defaults, api_key)
        cost_str = f"{r['cost']:.6f}" if r["cost"] is not None else "—"
        status_ok = "PASS" if r["ok"] else f"FAIL: {r['fail_reason']}"
        print(hdr.format(
            model_id[:39],
            str(r["status"]),
            pinned or "—",
            (r["actual_provider"] or "—")[:17],
            str(r["reasoning_tokens"] if r["reasoning_tokens"] is not None else "—"),
            (r["finish_reason"] or "—")[:11],
            (r["content_preview"] or "—")[:41],
            f"{cost_str}  {status_ok}",
        ))
        summary.append((model_id, r["ok"], r["fail_reason"] or ""))

    # ── summary ─────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"{'SUMMARY':^60}")
    print("=" * 60)
    passes = sum(1 for _, ok, _ in summary if ok)
    fails = len(summary) - passes
    for mid, ok, reason in summary:
        tag = "PASS" if ok else "FAIL"
        note = f"  ← {reason}" if not ok else ""
        print(f"  [{tag}] {mid}{note}")
    print("-" * 60)
    print(f"  {passes}/{len(summary)} passed" +
          (f"  ← {fails} FAILED" if fails else "  ✓ all clear"))

    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
