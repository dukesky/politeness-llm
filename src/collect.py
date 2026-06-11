"""Data collection runner: async OpenRouter calls with resume + durable logging.

Usage (from a Colab notebook or shell):

    python -m src.collect --pairs data/pairs_dl19.jsonl --dry-run
    python -m src.collect --pairs data/pairs_dl19.jsonl

Inputs
------
--pairs : JSONL file with one record per query-doc pair:
          {"dataset": "dl19", "qid": "...", "docid": "...",
           "query": "...", "passage": "..."}
Config  : config/prompts.yaml, config/models.yaml
Env     : OPENROUTER_API_KEY  (in Colab, load from Secrets and os.environ)
Output  : {DATA_DIR}/raw/{model_slug}__{prompt_id}__run{n}.jsonl  (append-only)

Resume
------
On startup we scan raw/ and skip every (model, prompt_id, qid, docid, run)
already present, so the script can be killed/restarted at any time.
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import time
from itertools import product
from pathlib import Path

import aiohttp
import yaml

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_CONCURRENCY = 10
MAX_RETRIES = 4


# --------------------------------------------------------------------------
# Config / inputs
# --------------------------------------------------------------------------

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def compose_prompt(variant, rubric, query, passage):
    """wrapper_prefix + shared rubric + wrapper_suffix. Rubric is identical
    across variants by construction; only the tone wrapper changes."""
    body = rubric.format(query=query, passage=passage)
    return f"{variant['wrapper_prefix'].strip()}\n\n{body}\n{variant['wrapper_suffix'].strip()}"


def parse_score(text):
    """Return (score, parse_ok). Strict-ish: look for a JSON object with an
    integer 'score' in 0..3; fall back to a bare digit. Failures are data."""
    try:
        m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if m:
            obj = json.loads(m.group(0))
            s = int(obj.get("score"))
            if s in (0, 1, 2, 3):
                return s, True
    except Exception:
        pass
    m = re.fullmatch(r"\s*([0-3])\s*", text)
    if m:
        return int(m.group(1)), True
    return None, False


# --------------------------------------------------------------------------
# Resume logic
# --------------------------------------------------------------------------

def key(model_id, prompt_id, qid, docid, run):
    return f"{model_id}|{prompt_id}|{qid}|{docid}|{run}"


def scan_done(raw_dir: Path):
    done = set()
    for fp in raw_dir.rglob("*.jsonl"):
        with open(fp) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add(key(r["model_id"], r["prompt_id"],
                                 r["qid"], r["docid"], r["run"]))
                except Exception:
                    continue  # tolerate a torn final line from a crash
    return done


# --------------------------------------------------------------------------
# API call
# --------------------------------------------------------------------------

async def call_one(session, sem, task, cfg, out_files, code_version):
    model, variant, pair, run = task
    async with sem:
        payload = {
            "model": model["model_id"],
            "temperature": cfg["temperature"],
            "top_p": cfg["top_p"],
            "max_tokens": cfg["max_tokens"],
            "messages": [{
                "role": "user",
                "content": compose_prompt(variant, cfg["rubric"],
                                          pair["query"], pair["passage"]),
            }],
        }
        if model.get("provider_order"):
            payload["provider"] = {"order": model["provider_order"],
                                   "allow_fallbacks": False}
        effort = model.get("reasoning_effort", cfg.get("reasoning_effort"))
        if effort:
            payload["reasoning"] = {"effort": effort}

        headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                t0 = time.time()
                async with session.post(OPENROUTER_URL, json=payload,
                                        headers=headers, timeout=120) as resp:
                    body = await resp.json()
                    if resp.status != 200:
                        raise RuntimeError(f"HTTP {resp.status}: {body}")
                latency = time.time() - t0
                text = body["choices"][0]["message"]["content"]
                score, ok = parse_score(text)
                record = {
                    "model_id": model["model_id"],
                    "provider": body.get("provider"),
                    "prompt_id": variant["prompt_id"],
                    "politeness_level": variant["politeness_level"],
                    "dataset": pair["dataset"],
                    "qid": pair["qid"],
                    "docid": pair["docid"],
                    "run": run,
                    "score": score,
                    "parse_ok": ok,
                    "raw_output": text,
                    "usage": body.get("usage"),
                    "latency_s": round(latency, 3),
                    "code_version": code_version,
                    "ts": time.time(),
                }
                fkey = (model["model_id"], variant["prompt_id"], run)
                f = out_files[fkey]
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()  # durability on Drive: flush every record
                return
            except Exception as e:  # noqa: BLE001
                last_err = e
                await asyncio.sleep(2 ** attempt)
        print(f"[FAILED after {MAX_RETRIES} tries] {task_desc(task)}: {last_err}")


def task_desc(task):
    model, variant, pair, run = task
    return f"{model['model_id']} {variant['prompt_id']} {pair['qid']}/{pair['docid']} run{run}"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def slug(s):
    return re.sub(r"[^a-zA-Z0-9.-]+", "_", s)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "./data"))
    ap.add_argument("--dry-run", action="store_true",
                    help="limit to 5 pairs x 2 prompts x 1 model x 1 run")
    ap.add_argument("--models", default=None,
                    help="comma-separated model_id substrings; omit to run all")
    args = ap.parse_args()

    prompts = load_yaml("config/prompts.yaml")
    models_cfg = load_yaml("config/models.yaml")
    cfg = {**models_cfg["defaults"], "rubric": prompts["rubric"]}

    pairs = [json.loads(l) for l in open(args.pairs)]
    variants = prompts["variants"]
    models = models_cfg["models"]

    if args.models:
        filters = [s.strip() for s in args.models.split(",") if s.strip()]
        models = [m for m in models if any(f in m["model_id"] for f in filters)]

    if args.dry_run:
        pairs, variants, models = pairs[:5], variants[:2], models[:1]

    print(f"Selected models ({len(models)}): {[m['model_id'] for m in models]}")

    raw_dir = Path(args.data_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    done = scan_done(raw_dir)
    print(f"Resume: {len(done)} records already collected.")

    tasks = []
    for model, variant, pair in product(models, variants, pairs):
        if model.get("subsample") and hash(pair["qid"]) % 100 >= model["subsample"] * 100:
            continue  # deterministic subsample for flagship tier
        model_runs = 1 if args.dry_run else model.get("n_runs", cfg["n_runs"])
        for run in range(1, model_runs + 1):
            if key(model["model_id"], variant["prompt_id"],
                   pair["qid"], pair["docid"], run) not in done:
                tasks.append((model, variant, pair, run))
    print(f"To collect: {len(tasks)} records.")

    out_files = {}
    for model, variant, pair, run in tasks:
        fkey = (model["model_id"], variant["prompt_id"], run)
        if fkey not in out_files:
            model_dir = raw_dir / slug(model['model_id'])
            model_dir.mkdir(parents=True, exist_ok=True)
            fp = model_dir / f"{variant['prompt_id']}__run{run}.jsonl"
            out_files[fkey] = open(fp, "a")

    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    code_version = git_hash()
    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*[
            call_one(session, sem, t, cfg, out_files, code_version) for t in tasks
        ])
    for f in out_files.values():
        f.close()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
