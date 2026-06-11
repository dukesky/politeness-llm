"""Audit provider_order entries in config/models.yaml against OpenRouter.

For each model, fetches GET /api/v1/models/{model_id}/endpoints and checks:
  - Every name in provider_order exists (case-exact) in the live endpoint list.
  - Prints a table: model | pinned provider | quantization | status.
  - Exits non-zero if any mismatch is found.

Usage:
    python scripts/check_providers.py            # read-only audit
    python scripts/check_providers.py --fix      # patch models.yaml in place
"""

import argparse
import sys
import urllib.request
import json
import yaml
from pathlib import Path

ENDPOINT_URL = "https://openrouter.ai/api/v1/models/{model_id}/endpoints"
YAML_PATH = Path("config/models.yaml")


def fetch_endpoints(model_id: str) -> list[dict]:
    url = ENDPOINT_URL.format(model_id=model_id)
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.loads(r.read())
        return data.get("data", {}).get("endpoints", [])
    except Exception as e:
        print(f"  [ERROR fetching {model_id}] {e}", file=sys.stderr)
        return []


def audit(models: list[dict], fix: bool) -> bool:
    col = "{:<45} {:<22} {:<20} {}"
    print(col.format("model_id", "pinned provider", "quantization", "status"))
    print("-" * 105)

    any_mismatch = False
    for m in models:
        model_id = m["model_id"]
        pinned = m.get("provider_order", [])
        endpoints = fetch_endpoints(model_id)
        live_by_name = {ep["provider_name"]: ep for ep in endpoints}

        if not pinned:
            print(col.format(model_id, "(none pinned)", "—", "SKIP"))
            continue

        for name in pinned:
            if name in live_by_name:
                quant = live_by_name[name].get("quantization") or "—"
                print(col.format(model_id, name, quant, "OK"))
            else:
                # Try case-insensitive match to suggest correct spelling
                match = next(
                    (k for k in live_by_name if k.lower() == name.lower()), None
                )
                if match:
                    status = f"MISMATCH → correct name: '{match}'"
                    if fix:
                        idx = m["provider_order"].index(name)
                        m["provider_order"][idx] = match
                        quant = live_by_name[match].get("quantization") or "—"
                        status += f"  [FIXED] quant={quant}"
                    else:
                        quant = live_by_name[match].get("quantization") or "—"
                else:
                    status = f"NOT FOUND in live endpoints ({len(live_by_name)} available)"
                    quant = "—"
                print(col.format(model_id, name, quant, status))
                any_mismatch = True

    return any_mismatch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true",
                    help="patch provider names in models.yaml in place")
    args = ap.parse_args()

    cfg = yaml.safe_load(YAML_PATH.read_text())
    models = cfg["models"]

    any_mismatch = audit(models, fix=args.fix)

    if args.fix and any_mismatch:
        YAML_PATH.write_text(yaml.dump(cfg, allow_unicode=True, sort_keys=False,
                                       default_flow_style=False))
        print("\nmodels.yaml patched.")

    sys.exit(1 if any_mismatch and not args.fix else 0)


if __name__ == "__main__":
    main()
