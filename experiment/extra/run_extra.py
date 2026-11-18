"""Model calls for the follow-up checks.

  python extra/run_extra.py list-models    # find small-model slugs; confirm SMALL and STRONG exist
  python extra/run_extra.py run            # math (small panel + repeats + strong) and cascade calls
  python extra/run_extra.py fix-skipped    # rerun main-experiment classification calls that hit
                                           # the 8,000-token limit, with 32,000 tokens

Calls are cached under extra/outputs/raw/; rerun to retry failures.
Set SMALL_MODELS=a,b,c or STRONG_MODEL=x to change models.
"""
import argparse
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

from extra_common import (DATA, FAKE, MAIN_DATA, MAIN_RAW, MAX_TOKENS, RAW, REPEATS, SMALL,
                          STRONG, client, final_answer, prices, raw_path, read_jsonl)

from common import MODELS as MAIN_MODELS, REPEATS as MAIN_REPEATS, chat  # noqa: E402


def jobs(sets):
    out = []
    if "math" in sets:
        for item in read_jsonl(DATA / "math.jsonl"):
            calls = [(m, 0) for m in SMALL] + [(SMALL[0], r) for r in range(1, REPEATS)] + [(STRONG, 0)]
            out += [("math", item, m, r) for m, r in calls]
    for task_set in ("classification", "extraction_hard"):
        if "cascade" in sets:
            for item in read_jsonl(MAIN_DATA / f"{task_set}.jsonl"):
                out += [(task_set, item, m, 0) for m in SMALL[:2]]
    return out


def list_models():
    c = client(timeout=60)
    ids = sorted(m.id for m in c.models.list().data)
    small = [m for m in ids if re.search(r"flash|nano|mini|oss|lite|small|haiku|luna", m, re.I)]
    print("Small/cheap models available:")
    for m in small:
        print(f"  {m}{'' if m in prices() else '   (no price in extra_common.EXTRA_PRICES)'}")
    print("\nConfigured:")
    for m in SMALL + [STRONG]:
        print(f"  {'ok     ' if m in ids else 'MISSING'} {m}")


def run(workers, sets):
    c = client()
    todo = []
    for task_set, item, model, rep in jobs(sets):
        p = raw_path(RAW, task_set, item["id"], model, rep)
        if p.exists() and json.loads(p.read_text()).get("error") is None:
            continue
        todo.append((task_set, item, model, rep, p))
    print(f"{len(todo)} calls to make")

    def work(job):
        task_set, item, model, rep, p = job
        out = chat(c, model, item["prompt"], MAX_TOKENS[task_set])
        out.update({"item_id": item["id"], "task_set": task_set, "repeat": rep})
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out))
        return out

    failed = 0
    with ThreadPoolExecutor(workers) as pool:
        for i, fut in enumerate(as_completed(pool.submit(work, j) for j in todo), 1):
            o = fut.result()
            if o["error"]:
                failed += 1
                print(f"  FAILED {o['item_id']} {o['model']}: {o['error'][:120]}")
            elif o["finish_reason"] == "length":
                print(f"  truncated (max_tokens) {o['item_id']} {o['model']}")
            if i % 50 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} done")
    if failed:
        print(f"{failed} call(s) failed; rerun the same command to retry them.")


def fix_skipped():
    """Rerun main classification calls with no FINAL line at a 32,000-token budget. The original
    file is kept as .bak; the new reply replaces it only if it has a FINAL line."""
    c = client()
    calls = [(m, 0) for m in MAIN_MODELS] + [(MAIN_MODELS[0], r) for r in range(1, MAIN_REPEATS)]
    todo = []
    for item in read_jsonl(MAIN_DATA / "classification.jsonl"):
        for m, r in calls:
            p = raw_path(MAIN_RAW, "classification", item["id"], m, r)
            if p.exists() and final_answer(json.loads(p.read_text())["text"]) is None:
                todo.append((item, m, r, p))
    print(f"{len(todo)} classification calls to rerun with 32,000 tokens")
    for item, m, r, p in todo:
        out = chat(c, m, item["prompt"], 32000)
        out.update({"item_id": item["id"], "task_set": "classification", "repeat": r,
                    "rerun_max_tokens": 32000})
        ok = not out["error"] and final_answer(out["text"]) is not None
        if ok and not FAKE:
            shutil.copy(p, p.with_suffix(".json.bak"))
            p.write_text(json.dumps(out))
        u = out.get("usage") or {}
        print(f"  {item['id']} {m} run {r}: {'fixed' if ok else 'still no FINAL'} "
              f"(finish={out['finish_reason']}, output tokens={u.get('completion_tokens')})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["list-models", "run", "fix-skipped"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--sets", default="math,cascade", help="comma-separated: math, cascade")
    args = ap.parse_args()
    {"list-models": list_models, "run": lambda: run(args.workers, args.sets.split(",")),
     "fix-skipped": fix_skipped}[args.step]()
