"""Send every item to every panel model, plus repeat runs of the first model.

Each response is cached as one JSON file under outputs/raw/<task set>/, so an interrupted
run resumes where it stopped. Failed calls are cached with an "error" and retried next run.

Run:
  export MODEL_ACCESS_KEY=...
  python run_calls.py --check-models          # confirm the model slugs exist
  python run_calls.py --dry-run               # count calls and estimate cost
  python run_calls.py --limit 2               # smoke test: 2 items per task set
  python run_calls.py                         # everything
  python run_calls.py --sets research         # one task set
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import (BASE_URL, DATA, JUDGE_MODEL, MAX_TOKENS, MODELS, RAW, TASK_SETS,
                    calls_for, chat, load_prices, make_client, read_jsonl, safe)

# Rough per-call token guesses for --dry-run only. Reasoning models can use far more output.
GUESS = {"classification": (900, 1500), "extraction": (450, 1500), "research": (350, 8000),
         "extraction_hard": (500, 1500)}


def raw_path(task_set, item_id, model, rep):
    return RAW / task_set / f"{safe(item_id)}__{safe(model)}__{rep}.json"


def check_models():
    client = make_client(timeout=60)
    available = {m.id for m in client.models.list().data}
    missing = [m for m in MODELS + [JUDGE_MODEL] if m not in available]
    print(f"{len(available)} models available at {BASE_URL}")
    for m in MODELS + [JUDGE_MODEL]:
        print(f"  {'ok     ' if m in available else 'MISSING'} {m}")
    if missing:
        print("\nSet MODELS / JUDGE_MODEL to slugs from this list:")
        for m in sorted(available):
            print(f"  {m}")
        sys.exit(1)


def dry_run(sets, limit):
    prices = load_prices()
    total = 0.0
    for s in sets:
        items = read_jsonl(DATA / f"{s}.jsonl")[:limit]
        n_in, n_out = GUESS[s]
        cost = sum((n_in * prices[m]["input"] + n_out * prices[m]["output"]) / 1e6
                   for _ in items for m, _ in calls_for(None))
        total += cost
        print(f"{s:15} {len(items):3} items x {len(calls_for(None))} calls  ~${cost:.2f}")
    print(f"{'panel total':15} ~${total:.2f} (judge calls are extra; see judge_research.py --dry-run)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=TASK_SETS, choices=TASK_SETS)
    ap.add_argument("--limit", type=int, default=None, help="items per task set")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--check-models", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.check_models:
        return check_models()
    if args.dry_run:
        return dry_run(args.sets, args.limit)

    client = make_client(timeout=1200)
    jobs = []
    for s in args.sets:
        for item in read_jsonl(DATA / f"{s}.jsonl")[:args.limit]:
            for model, rep in calls_for(item["id"]):
                path = raw_path(s, item["id"], model, rep)
                if path.exists() and json.loads(path.read_text()).get("error") is None:
                    continue
                jobs.append((s, item, model, rep, path))
    print(f"{len(jobs)} calls to make")

    def work(job):
        s, item, model, rep, path = job
        out = chat(client, model, item["prompt"], MAX_TOKENS[s])
        out.update({"item_id": item["id"], "task_set": s, "repeat": rep})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out))
        return out

    failed = 0
    with ThreadPoolExecutor(args.workers) as pool:
        for i, fut in enumerate(as_completed(pool.submit(work, j) for j in jobs), 1):
            out = fut.result()
            if out["error"]:
                failed += 1
                print(f"  FAILED {out['item_id']} {out['model']}: {out['error'][:120]}")
            elif out["finish_reason"] == "length":
                print(f"  truncated (max_tokens) {out['item_id']} {out['model']}")
            if i % 25 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} done")
    if failed:
        print(f"{failed} call(s) failed; rerun the same command to retry them.")


if __name__ == "__main__":
    main()
