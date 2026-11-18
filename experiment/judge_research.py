"""Grade every research (DRACO) answer against its rubric with a judge model outside the panel.

Each answer gets one judge call that returns MET or UNMET for every rubric criterion.
DRACO's own protocol judges one criterion per call; grading all criteria in one call is a
cost-saving deviation, so report it. One answer per task (MODELS[0], first run) is judged a
second time to measure how much the judge disagrees with itself.

Run after run_calls.py:
  python judge_research.py --dry-run
  python judge_research.py
"""
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import (DATA, JUDGE, JUDGE_MAX_TOKENS, JUDGE_MODEL, MODELS, RAW, calls_for, chat,
                    load_prices, make_client, read_jsonl, safe)

JUDGE_PROMPT = """You are grading a response to a research question against a rubric.

For each criterion, decide whether the response satisfies it.
- MET: the response clearly does what the criterion describes.
- UNMET: it does not, or only vaguely gestures at it.
Some criteria describe mistakes (for example, "States X incorrectly"). For those, MET means the
response makes that mistake. Judge only what the response says; do not reward length.

Question:
{question}

Response:
{response}

Rubric (one criterion per line, id in brackets):
{criteria}

Reply with only a JSON object mapping every criterion id to "MET" or "UNMET":
{{"verdicts": {{"criterion-id": "MET", ...}}}}"""


def criteria(item):
    return [c for s in item["rubric"]["sections"] for c in s["criteria"]]


def parse_verdicts(text, ids):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    raw = re.sub(r':\s*(MET|UNMET)\b', r': "\1"', m.group(0))  # judge sometimes omits quotes
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    verdicts = data.get("verdicts", data)
    out = {i: str(verdicts.get(i, "")).upper() for i in ids}
    if any(v not in ("MET", "UNMET") for v in out.values()):
        return None
    return {i: v == "MET" for i, v in out.items()}


def judge_jobs(items):
    for item in items:
        for model, rep in calls_for(item["id"]):
            passes = [0, 1] if (model, rep) == (MODELS[0], 0) else [0]
            for j in passes:
                yield item, model, rep, j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    items = read_jsonl(DATA / "research.jsonl")[:args.limit]
    jobs = list(judge_jobs(items))
    if args.dry_run:
        p = load_prices().get(JUDGE_MODEL)
        if not p:
            raise SystemExit(f"Add {JUDGE_MODEL} to prices.json")
        cost = len(jobs) * (9000 * p["input"] + 3000 * p["output"]) / 1e6
        print(f"{len(jobs)} judge calls with {JUDGE_MODEL}  ~${cost:.2f} (guess: 9K in, 3K out each)")
        return

    client = make_client(timeout=600)
    todo = []
    for item, model, rep, j in jobs:
        out_path = JUDGE / f"{safe(item['id'])}__{safe(model)}__{rep}__j{j}.json"
        if out_path.exists() and json.loads(out_path.read_text()).get("verdicts") is not None:
            continue
        raw = RAW / "research" / f"{safe(item['id'])}__{safe(model)}__{rep}.json"
        if not raw.exists():
            continue
        answer = json.loads(raw.read_text())
        if answer.get("error") or not answer.get("text"):
            continue
        todo.append((item, model, rep, j, answer["text"], out_path))
    print(f"{len(todo)} judge calls to make")

    def work(job):
        item, model, rep, j, text, out_path = job
        crit = criteria(item)
        prompt = JUDGE_PROMPT.format(
            question=item["prompt"], response=text,
            criteria="\n".join(f"- [{c['id']}] {c['requirement']}" for c in crit))
        out = chat(client, JUDGE_MODEL, prompt, JUDGE_MAX_TOKENS, temperature=0)
        out["verdicts"] = parse_verdicts(out["text"], [c["id"] for c in crit]) if out["text"] else None
        out.update({"item_id": item["id"], "answer_model": model, "repeat": rep, "judge_pass": j})
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out))
        return out

    bad = 0
    with ThreadPoolExecutor(args.workers) as pool:
        for i, fut in enumerate(as_completed(pool.submit(work, t) for t in todo), 1):
            out = fut.result()
            if out["verdicts"] is None:
                bad += 1
                print(f"  unusable judge reply for {out['item_id']} {out['answer_model']}"
                      f" ({out['error'] or 'could not parse verdicts'})")
            if i % 10 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} done")
    if bad:
        print(f"{bad} judge call(s) unusable; rerun to retry them.")


if __name__ == "__main__":
    main()
