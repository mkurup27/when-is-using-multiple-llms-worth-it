"""Build a worksheet for hand-checking 20 of the rubric judge's verdicts on the main DRACO run.

Picks the 4 shortest judged answers (from different tasks) and 5 criteria from each, mixing
MET and UNMET verdicts. Read each answer, mark each criterion yourself, then count how often
you disagree with the judge.

Run: python extra/make_worksheet.py      -> extra/outputs/judge_check.md
"""
import json
import random

from extra_common import MAIN_DATA, OUT, read_jsonl, safe

from common import JUDGE, MODELS, RAW, REPEATS  # noqa: E402

SEED = 4155

if __name__ == "__main__":
    items = {i["id"]: i for i in read_jsonl(MAIN_DATA / "research.jsonl")}
    candidates = []
    for item_id, item in items.items():
        for m, r in [(m, 0) for m in MODELS] + [(MODELS[0], r) for r in range(1, REPEATS)]:
            jp = JUDGE / f"{safe(item_id)}__{safe(m)}__{r}__j0.json"
            ap = RAW / "research" / f"{safe(item_id)}__{safe(m)}__{r}.json"
            if not (jp.exists() and ap.exists()):
                continue
            verdicts = json.loads(jp.read_text()).get("verdicts")
            text = json.loads(ap.read_text()).get("text", "")
            if verdicts and text:
                candidates.append((len(text), item_id, m, r, text, verdicts))
    candidates.sort()
    chosen, seen = [], set()
    for c in candidates:
        if c[1] not in seen:
            chosen.append(c)
            seen.add(c[1])
        if len(chosen) == 4:
            break

    rng = random.Random(SEED)
    lines = ["# Judge hand-check", "",
             "For each answer, read it and decide MET or UNMET for each criterion yourself. "
             "Fill in the last column, then count how many of the 20 differ from the judge.", ""]
    n = 0
    for k, (length, item_id, m, r, text, verdicts) in enumerate(chosen, 1):
        item = items[item_id]
        crit = [c for s in item["rubric"]["sections"] for c in s["criteria"]]
        met = [c for c in crit if verdicts.get(c["id"])]
        unmet = [c for c in crit if not verdicts.get(c["id"])]
        pick = rng.sample(met, min(3, len(met)))
        pick += rng.sample(unmet, min(5 - len(pick), len(unmet)))
        rng.shuffle(pick)
        lines += [f"## Answer {k}: {item['domain']} task, {m} (run {r}), {length:,} characters", "",
                  "### Question", "", item["prompt"], "", "### Answer", "", text, "",
                  "### Criteria to check", "",
                  "| # | criterion | weight | judge said | you say (MET / UNMET) |", "|---|---|---|---|---|"]
        for c in pick:
            n += 1
            req = c["requirement"].replace("|", "/")
            lines.append(f"| {n} | {req} | {c['weight']} | {'MET' if verdicts[c['id']] else 'UNMET'} | |")
        lines += ["", "Negative weights mark mistakes: MET means the answer makes that mistake.", "", "---", ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "judge_check.md").write_text("\n".join(lines))
    print(f"Wrote {n} criteria from {len(chosen)} answers to {OUT / 'judge_check.md'}")
