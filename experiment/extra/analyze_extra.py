"""Tables for the follow-up checks. Reads cached calls only; makes no API calls.

  python extra/analyze_extra.py
Writes extra/outputs/extra_summary.md and prints it.

Sections:
  1. Math positive control: does a vote of three small models beat each of them?
  2. Cascade: two cheap models answer; only disagreements go to the strong model.
  3. Held-out comparison: majority vote vs. a best model picked on the other half of the data,
     plus exact McNemar tests of the vote against each model.
  4. Classification messages where all three main models agreed on a wrong intent.
"""
import random
from collections import Counter
from math import comb

from extra_common import (MAIN_DATA, MAIN_RAW, OUT, RAW, REPEATS, SMALL, STRONG, DATA,
                          call_cost, load_raw, parsed, prices, read_jsonl)

from analyze import answer_rows  # noqa: E402
from common import MODELS as MAIN_MODELS  # noqa: E402


def pct(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def vote_ok(answers, gold):
    top, votes = Counter(answers).most_common(1)[0]
    return top == gold and votes > len(answers) / 2


def mcnemar(a_right, b_right):
    """Exact two-sided McNemar test on paired right/wrong lists."""
    b = sum(x and not y for x, y in zip(a_right, b_right))
    c = sum(y and not x for x, y in zip(a_right, b_right))
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return b, c, min(1.0, p)


def cost_of(call, table):
    return call_cost(call["model"], call.get("usage"), table) if call else None


# ---- 1. math positive control ----

def math_missing():
    """Count unparseable math replies per (model, run), split into truncated and other."""
    out = Counter()
    for item in read_jsonl(DATA / "math.jsonl"):
        calls = [(m, 0) for m in SMALL] + [(SMALL[0], r) for r in range(1, REPEATS)] + [(STRONG, 0)]
        for m, r in calls:
            c = load_raw(RAW, "math", item["id"], m, r)
            if c and parsed(c, "int") is None:
                why = "truncated" if c.get("finish_reason") == "length" else "no FINAL line"
                out[(m, r, why)] += 1
    return out


def math_rows():
    rows, skipped = [], 0
    for item in read_jsonl(DATA / "math.jsonl"):
        cross_calls = [load_raw(RAW, "math", item["id"], m, 0) for m in SMALL]
        self_calls = [load_raw(RAW, "math", item["id"], SMALL[0], r) for r in range(REPEATS)]
        strong_call = load_raw(RAW, "math", item["id"], STRONG, 0)
        if any(c is None for c in cross_calls + self_calls + [strong_call]):
            continue
        cross = [parsed(c, "int") for c in cross_calls]
        self_ = [parsed(c, "int") for c in self_calls]
        strong = parsed(strong_call, "int")
        if any(a is None for a in cross + self_ + [strong]):
            skipped += 1  # kept: a reply with no readable answer scores as wrong
        rows.append({**item, "cross": cross, "self": self_, "strong": strong, "gold": item["answer"],
                     "cross_calls": cross_calls, "strong_call": strong_call})
    return rows, skipped


def math_section(lines, table):
    rows, skipped = math_rows()
    lines += ["## 1. Math positive control", ""]
    if not rows:
        return lines + ["No complete math rows yet.", ""]
    n = len(rows)
    acc = {m: mean(r["cross"][i] == r["gold"] for r in rows) for i, m in enumerate(SMALL)}
    vote = mean(vote_ok(r["cross"], r["gold"]) for r in rows)
    selfc = mean(vote_ok(r["self"], r["gold"]) for r in rows)
    strong = mean(r["strong"] == r["gold"] for r in rows)
    cross_d = mean(len(set(r["cross"])) > 1 for r in rows)
    self_d = mean(len(set(r["self"])) > 1 for r in rows)
    small_cost = mean(sum(cost_of(c, table) or 0 for c in r["cross_calls"]) for r in rows)
    strong_cost = mean(cost_of(r["strong_call"], table) or 0 for r in rows)
    lines += [f"{n} MATH-500 problems (levels 4-5, integer answers). On {skipped}, at least one reply "
              "had no readable answer; those replies count as wrong:", ""]
    lines += [f"- {m} run {r}: {k} {why}" for (m, r, why), k in sorted(math_missing().items())]
    lines += ["", "| measure | value |", "|---|---|"]
    lines += [f"| {m} alone | {pct(a)} |" for m, a in acc.items()]
    lines += [f"| average small model | {pct(mean(acc.values()))} |",
              f"| best small model | {pct(max(acc.values()))} |",
              f"| **majority vote of the three small models** | **{pct(vote)}** |",
              f"| self-consistency: 3 runs of {SMALL[0]}, majority | {pct(selfc)} |",
              f"| {STRONG} alone | {pct(strong)} |",
              f"| small models differ | {pct(cross_d)} |",
              f"| {SMALL[0]} differs from itself | {pct(self_d)} |",
              f"| excess disagreement | {100 * (cross_d - self_d):+.0f} pts |",
              f"| cost per problem: three small models | ${small_cost:.4f} |",
              f"| cost per problem: {STRONG} | ${strong_cost:.4f} |", ""]
    vote_right = [vote_ok(r["cross"], r["gold"]) for r in rows]
    for i, m in enumerate(SMALL):
        b, c, p = mcnemar(vote_right, [r["cross"][i] == r["gold"] for r in rows])
        lines.append(f"- Vote vs {m}: vote alone right on {b}, {m} alone right on {c}, exact McNemar p = {p:.3f}")
    b, c, p = mcnemar(vote_right, [r["strong"] == r["gold"] for r in rows])
    lines += [f"- Vote vs {STRONG}: vote alone right on {b}, {STRONG} alone right on {c}, p = {p:.3f}", ""]
    return lines


# ---- 2. cascade ----

def cascade_rows(task_set):
    rows = []
    if task_set == "math":
        items, base_cheap, base_strong = read_jsonl(DATA / "math.jsonl"), RAW, RAW
    else:
        items, base_cheap, base_strong = read_jsonl(MAIN_DATA / f"{task_set}.jsonl"), RAW, MAIN_RAW
    for item in items:
        cheap = [load_raw(base_cheap, task_set, item["id"], m, 0) for m in SMALL[:2]]
        strong = load_raw(base_strong, task_set, item["id"], STRONG, 0)
        answers = [parsed(c, item["kind"]) for c in cheap] + [parsed(strong, item["kind"])]
        if any(a is None for a in answers):
            continue
        gold = parsed({"text": f"FINAL: {item['answer']}"}, item["kind"])
        rows.append({"a": answers[0], "b": answers[1], "s": answers[2], "gold": gold,
                     "cheap_calls": cheap, "strong_call": strong})
    return rows


def cascade_section(lines, table):
    lines += ["## 2. Cascade: two cheap models, escalate only when they disagree", "",
              f"Cheap pair: {SMALL[0]} and {SMALL[1]}. Strong model: {STRONG}.", "",
              "| task | n | escalated | cascade accuracy | strong model always | cheap model always | "
              "cascade cost per item | strong cost per item | cascade cost vs strong |",
              "|---|---|---|---|---|---|---|---|---|"]
    for task_set in ("classification", "extraction_hard", "math"):
        rows = cascade_rows(task_set)
        if not rows:
            lines.append(f"| {task_set} | 0 | | | | | | | |")
            continue
        esc = [r["a"] != r["b"] for r in rows]
        final = [r["s"] if e else r["a"] for r, e in zip(rows, esc)]
        acc_c = mean(f == r["gold"] for f, r in zip(final, rows))
        acc_s = mean(r["s"] == r["gold"] for r in rows)
        acc_a = mean(r["a"] == r["gold"] for r in rows)
        cheap_cost = [sum(cost_of(c, table) or 0 for c in r["cheap_calls"]) for r in rows]
        strong_cost = [cost_of(r["strong_call"], table) or 0 for r in rows]
        c_cost = mean(cc + (sc if e else 0) for cc, sc, e in zip(cheap_cost, strong_cost, esc))
        s_cost = mean(strong_cost)
        lines.append(f"| {task_set} | {len(rows)} | {pct(mean(esc))} | {pct(acc_c)} | {pct(acc_s)} | "
                     f"{pct(acc_a)} | ${c_cost:.4f} | ${s_cost:.4f} | {c_cost / s_cost:.2f}x |")
    lines += ["", "Strong-model answers for classification and hard extraction are reused from the "
              "main experiment.", ""]
    return lines


# ---- 3. held-out comparison ----

def held_out(rows, n_models, splits=2000, seed=4155):
    rng = random.Random(seed)
    diffs, n = [], len(rows)
    for _ in range(splits):
        idx = list(range(n))
        rng.shuffle(idx)
        a, b = idx[: n // 2], idx[n // 2:]
        scores = [sum(rows[i]["cross"][k] == rows[i]["gold"] for i in a) for k in range(n_models)]
        best = rng.choice([k for k in range(n_models) if scores[k] == max(scores)])
        pick = mean(rows[i]["cross"][best] == rows[i]["gold"] for i in b)
        vote = mean(vote_ok(rows[i]["cross"], rows[i]["gold"]) for i in b)
        diffs.append(vote - pick)
    return mean(diffs), mean(d > 0 for d in diffs), mean(d == 0 for d in diffs)


def held_out_section(lines):
    lines += ["## 3. Vote vs. a best model chosen on held-out data", "",
              "Each of 2,000 random splits picks the most accurate single model on one half and "
              "scores it and the majority vote on the other half.", "",
              "| task | n | vote minus picked model (mean) | vote better | tie | vote worse |",
              "|---|---|---|---|---|---|"]
    sets = [("classification", answer_rows("classification")[0], MAIN_MODELS),
            ("extraction_hard", answer_rows("extraction_hard")[0], MAIN_MODELS),
            ("math (small models)", math_rows()[0], SMALL)]
    tests = []
    for name, rows, models in sets:
        if not rows:
            continue
        d, better, tie = held_out(rows, len(models))
        lines.append(f"| {name} | {len(rows)} | {100 * d:+.1f} pts | {pct(better)} | {pct(tie)} | "
                     f"{pct(1 - better - tie)} |")
        vote_right = [vote_ok(r["cross"], r["gold"]) for r in rows]
        for i, m in enumerate(models):
            b, c, p = mcnemar(vote_right, [r["cross"][i] == r["gold"] for r in rows])
            tests.append(f"- {name}, vote vs {m}: vote alone right on {b}, model alone right on {c}, p = {p:.3f}")
    return lines + ["", "Exact McNemar tests on the full sets:", ""] + tests + [""]


# ---- 4. agreed-wrong classification messages ----

def agreed_wrong_section(lines):
    rows, _ = answer_rows("classification")
    wrong = [r for r in rows if len(set(r["cross"])) == 1 and r["cross"][0] != r["gold"]]
    lines += ["## 4. Classification messages where all three models agreed on a wrong intent", "",
              "Decide for each whether the gold label or the models' answer is better.", "",
              "| id | message | gold label | all three models said |", "|---|---|---|---|"]
    lines += [f"| {r['id']} | {r['message'].replace('|', '/')} | {r['gold']} | {r['cross'][0]} |" for r in wrong]
    return lines + [""]


if __name__ == "__main__":
    table = prices()
    lines = ["# Follow-up checks", ""]
    lines = math_section(lines, table)
    lines = cascade_section(lines, table)
    lines = held_out_section(lines)
    lines = agreed_wrong_section(lines)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "extra_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
