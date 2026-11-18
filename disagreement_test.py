"""Disagreement test: how often do different models give different answers to your requests?

Input: a JSONL file, one request per line:
  {"prompt": "...", "task_type": "classification", "answer": "billing"}
"task_type" and "answer" are optional. "answer" enables the two ground-truth columns.

Run: API_KEY=... python disagreement_test.py requests.jsonl
Prints one row per task type and writes every model's full response to
results.jsonl, for scoring open-ended answers against a rubric.
"""
import collections
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("BASE_URL", "https://inference.do-ai.run/v1"),
    api_key=os.environ["API_KEY"],
)

MODELS = ["glm-5.2", "kimi-k2.6", "deepseek-v4-pro"]
REPEATS = 3  # runs of MODELS[0] on the same request, for the noise floor

INSTRUCTION = (
    "\n\nEnd your reply with one line in the form FINAL: <answer>, "
    "where <answer> is only the label, option, or value."
)


def ask(model, prompt):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt + INSTRUCTION}],
    )
    return response.choices[0].message.content or ""


def final_answer(text):
    for line in reversed(text.strip().splitlines()):
        if line.strip().upper().startswith("FINAL:"):
            return line.split(":", 1)[1].strip().lower()
    return None


def differs(answers):
    answers = [a for a in answers if a is not None]
    if len(answers) < 2:
        return None
    return len(set(answers)) > 1


def run(row):
    prompt = row["prompt"]
    with ThreadPoolExecutor() as pool:
        texts = list(pool.map(lambda m: ask(m, prompt), MODELS))
        repeats = list(pool.map(lambda _: ask(MODELS[0], prompt), range(REPEATS - 1)))
    return {
        **row,
        "responses": dict(zip(MODELS, texts)),
        "cross": [final_answer(t) for t in texts],
        "self": [final_answer(t) for t in [texts[0]] + repeats],
    }


def rate(results, key):
    flags = [differs(r[key]) for r in results]
    flags = [f for f in flags if f is not None]
    return sum(flags) / len(flags) if flags else float("nan")


def ground_truth(results):
    """Share of requests where every model agreed and was wrong, and where
    at least one model was right but a strict-majority vote would not pick it."""
    graded = [r for r in results
              if "answer" in r and any(a is not None for a in r["cross"])]
    if not graded:
        return None, None
    agreed_wrong = rescuable = 0
    for r in graded:
        gold = str(r["answer"]).strip().lower()
        answers = [a for a in r["cross"] if a is not None]
        top, votes = collections.Counter(answers).most_common(1)[0]
        vote_picks_gold = top == gold and votes > len(answers) / 2
        if len(answers) > 1 and len(set(answers)) == 1 and answers[0] != gold:
            agreed_wrong += 1
        elif gold in answers and not vote_picks_gold:
            rescuable += 1
    return agreed_wrong / len(graded), rescuable / len(graded)


def complete(r):
    return all(a is not None for a in r["cross"] + r["self"])


def report(results):
    done = [r for r in results if complete(r)]
    by_type = collections.defaultdict(list)
    for r in done:
        by_type[r.get("task_type", "all")].append(r)
    print(f"{'task type':<16}{'n':>5}{'models differ':>15}{'same model differs':>20}"
          f"{'excess':>8}{'all agree, wrong':>18}{'one right, vote misses':>27}")
    for task_type, rows in sorted(by_type.items()):
        cross, same = rate(rows, "cross"), rate(rows, "self")
        wrong, rescue = ground_truth(rows)
        gt = (f"{wrong:>18.0%}{rescue:>27.0%}" if wrong is not None
              else f"{'-':>18}{'-':>27}")
        print(f"{task_type:<16}{len(rows):>5}{cross:>15.0%}{same:>20.0%}{cross - same:>8.0%}{gt}")
    skipped = len(results) - len(done)
    if skipped:
        print(f"\n{skipped} request(s) skipped: a reply had no FINAL: line (see results.jsonl).")


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        requests = [json.loads(line) for line in f if line.strip()]
    results = [run(r) for r in requests]
    report(results)
    with open("results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
