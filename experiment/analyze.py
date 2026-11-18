"""Turn cached responses and judge verdicts into the tables for the article.

Run after run_calls.py (and judge_research.py for the research rows):
  python analyze.py
Writes results/summary.md and results/summary.json, and prints the markdown.
"""
import json
import statistics
from collections import Counter, defaultdict

from common import (DATA, JUDGE, MODELS, RAW, REPEATS, RESULTS, TASK_SETS, call_cost,
                    final_answer, load_prices, normalize, read_jsonl, safe)


def pct(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def load_raw(task_set, item_id, model, rep):
    p = RAW / task_set / f"{safe(item_id)}__{safe(model)}__{rep}.json"
    return json.loads(p.read_text()) if p.exists() else None


# ---- classification and extraction ----

def answer_rows(task_set):
    rows, skipped = [], 0
    for item in read_jsonl(DATA / f"{task_set}.jsonl"):
        cross = [load_raw(task_set, item["id"], m, 0) for m in MODELS]
        self_ = [load_raw(task_set, item["id"], MODELS[0], r) for r in range(REPEATS)]
        if any(c is None for c in cross + self_):
            continue  # not run yet
        parse = lambda c: normalize(final_answer(c["text"]), item["kind"]) if not c["error"] else None
        cross_a, self_a = [parse(c) for c in cross], [parse(c) for c in self_]
        if any(a is None for a in cross_a + self_a):
            skipped += 1
            continue
        rows.append({**item, "cross": cross_a, "self": self_a,
                     "gold": normalize(item["answer"], item["kind"])})
    return rows, skipped


def answer_metrics(rows):
    if not rows:
        return None
    n = len(rows)
    acc = {m: mean(r["cross"][i] == r["gold"] for r in rows) for i, m in enumerate(MODELS)}
    vote_right = agreed_wrong = rescuable = 0
    for r in rows:
        top, votes = Counter(r["cross"]).most_common(1)[0]
        vote_ok = top == r["gold"] and votes > len(r["cross"]) / 2
        vote_right += vote_ok
        if len(set(r["cross"])) == 1 and r["cross"][0] != r["gold"]:
            agreed_wrong += 1
        elif r["gold"] in r["cross"] and not vote_ok:
            rescuable += 1
    cross = mean(len(set(r["cross"])) > 1 for r in rows)
    self_ = mean(len(set(r["self"])) > 1 for r in rows)
    return {
        "n": n, "accuracy": acc, "best_single": max(acc.values()),
        "majority_vote": vote_right / n,
        "any_right": mean(r["gold"] in r["cross"] for r in rows),
        "models_differ": cross, "same_model_differs": self_, "excess": cross - self_,
        "all_agree_wrong": agreed_wrong / n, "one_right_vote_misses": rescuable / n,
    }


# ---- research (DRACO) ----

def load_verdicts(item_id, model, rep, j=0):
    p = JUDGE / f"{safe(item_id)}__{safe(model)}__{rep}__j{j}.json"
    return json.loads(p.read_text()).get("verdicts") if p.exists() else None


def research_metrics():
    per_task, per_model_cov, penalties = [], defaultdict(list), defaultdict(list)
    judge_flips = []
    for item in read_jsonl(DATA / "research.jsonl"):
        crit = [c for s in item["rubric"]["sections"] for c in s["criteria"]]
        pos = [c for c in crit if c["weight"] > 0]
        neg = [c for c in crit if c["weight"] < 0]
        total_w = sum(c["weight"] for c in pos)
        cross = [load_verdicts(item["id"], m, 0) for m in MODELS]
        self_ = [load_verdicts(item["id"], MODELS[0], r) for r in range(REPEATS)]
        if any(v is None for v in cross + self_):
            continue

        cover = lambda met: sum(c["weight"] for c in pos if met(c["id"])) / total_w
        cov = [cover(lambda i, v=v: v[i]) for v in cross]
        self_cov = [cover(lambda i, v=v: v[i]) for v in self_]
        for m, c, v in zip(MODELS, cov, cross):
            per_model_cov[m].append(c)
            penalties[m].append(sum(v[c["id"]] for c in neg))
        split = lambda vs: mean(len({v[c["id"]] for v in vs}) > 1 for c in pos)
        per_task.append({
            "id": item["id"], "domain": item["domain"],
            "criteria_models_differ": split(cross),
            "criteria_same_model_differs": split(self_),
            "union_gain_models": cover(lambda i: any(v[i] for v in cross)) - max(cov),
            "union_gain_same_model": cover(lambda i: any(v[i] for v in self_)) - max(self_cov),
        })
        rejudged = load_verdicts(item["id"], MODELS[0], 0, j=1)
        if rejudged:
            judge_flips.append(mean(cross[0][c["id"]] != rejudged[c["id"]] for c in crit))

    if not per_task:
        return None
    agg = lambda k: mean(t[k] for t in per_task)
    return {
        "n": len(per_task),
        "coverage": {m: mean(v) for m, v in per_model_cov.items()},
        "penalties_triggered": {m: mean(v) for m, v in penalties.items()},
        "criteria_models_differ": agg("criteria_models_differ"),
        "criteria_same_model_differs": agg("criteria_same_model_differs"),
        "criteria_judge_differs": mean(judge_flips),
        "union_gain_models": agg("union_gain_models"),
        "union_gain_same_model": agg("union_gain_same_model"),
        "tasks": per_task,
    }


# ---- cost and latency ----

def cost_latency():
    prices = load_prices()
    by = defaultdict(lambda: {"calls": 0, "errors": 0, "cost": 0.0, "in": 0, "out": 0,
                              "latency": [], "unpriced": 0})
    sources = [(s, RAW / s) for s in TASK_SETS] + [("research judge", JUDGE)]
    for label, folder in sources:
        for p in folder.glob("*.json") if folder.exists() else []:
            d = json.loads(p.read_text())
            b = by[(label, d["model"])]
            b["calls"] += 1
            if d.get("error"):
                b["errors"] += 1
                continue
            u = d.get("usage") or {}
            b["in"] += u.get("prompt_tokens", 0)
            b["out"] += u.get("completion_tokens", 0)
            c = call_cost(d["model"], u, prices)
            if c is None:
                b["unpriced"] += 1
            else:
                b["cost"] += c
            if d.get("latency_s") is not None:
                b["latency"].append(d["latency_s"])
    return by


def per_item_panel_cost(task_set):
    """Cost of one call to MODELS[0] vs. the three-model panel calls alone, per item."""
    prices = load_prices()
    single, panel = [], []
    for item in read_jsonl(DATA / f"{task_set}.jsonl"):
        calls = [load_raw(task_set, item["id"], m, 0) for m in MODELS]
        if any(c is None or c["error"] for c in calls):
            continue
        costs = [call_cost(c["model"], c["usage"], prices) for c in calls]
        if None in costs:
            continue
        single.append(costs[0])
        panel.append(sum(costs))
    return (mean(single), mean(panel)) if single else (None, None)


# ---- report ----

def main():
    out, md = {}, ["# Disagreement experiment results", ""]
    md.append(f"Panel: {', '.join(MODELS)}. Noise floor: {REPEATS} runs of {MODELS[0]}.")
    md.append("")

    md += ["## Classification and extraction", "",
           "| task | n | skipped | models differ | same model differs | excess | "
           "best single | majority vote | any model right | all agree, wrong | "
           "one right, vote misses |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    answer_sets = [s for s in ["classification", "extraction", "extraction_hard"]
                   if (DATA / f"{s}.jsonl").exists()]
    for task_set in answer_sets:
        rows, skipped = answer_rows(task_set)
        groups = [(task_set, rows)]
        if task_set.startswith("extraction"):
            kinds = sorted({r["difficulty"] for r in rows})
            groups += [(f"{task_set} ({d})", [r for r in rows if r["difficulty"] == d])
                       for d in kinds]
        for name, g in groups:
            m = answer_metrics(g)
            out[name] = m
            if m:
                md.append(
                    f"| {name} | {m['n']} | {skipped if name == task_set else ''} | "
                    f"{pct(m['models_differ'])} | {pct(m['same_model_differs'])} | "
                    f"{100 * m['excess']:+.0f} pts | {pct(m['best_single'])} | "
                    f"{pct(m['majority_vote'])} | {pct(m['any_right'])} | "
                    f"{pct(m['all_agree_wrong'])} | {pct(m['one_right_vote_misses'])} |")
    md.append("")
    md.append("Per-model accuracy:")
    md.append("")
    for name in answer_sets:
        if out.get(name):
            md.append(f"- {name}: " + ", ".join(f"{k} {pct(v)}" for k, v in out[name]["accuracy"].items()))
    md.append("")
    md.append("Skipped = at least one reply had no parseable FINAL: line.")
    md.append("")

    r = research_metrics()
    out["research"] = r
    md += ["## Research (DRACO, no web access)", ""]
    if r:
        md += [f"Tasks fully judged: {r['n']}", "",
               "| measure | value |", "|---|---|",
               f"| rubric criteria where the models' verdicts differ | {pct(r['criteria_models_differ'])} |",
               f"| rubric criteria where {MODELS[0]}'s own runs differ | {pct(r['criteria_same_model_differs'])} |",
               f"| rubric criteria where the judge disagrees with itself | {pct(r['criteria_judge_differs'])} |",
               f"| coverage gain, union of 3 models vs. best single answer | {100 * r['union_gain_models']:+.1f} pts |",
               f"| coverage gain, union of 3 {MODELS[0]} runs vs. best single run | {100 * r['union_gain_same_model']:+.1f} pts |",
               "", "Mean weighted coverage of positive rubric criteria, and penalty criteria triggered:", ""]
        for m in MODELS:
            md.append(f"- {m}: {pct(r['coverage'][m])} coverage, "
                      f"{r['penalties_triggered'][m]:.1f} penalties per answer")
        md += ["", "Coverage here is our simplification (share of positive rubric weight met), "
               "not DRACO's official score. The union rows are an upper bound on what a perfect "
               "synthesizer could merge, not what the synthesis tool actually produces.", ""]
    else:
        md += ["No fully judged research tasks yet. Run judge_research.py.", ""]

    md += ["## Cost and latency", "",
           "| task set | model | calls | errors | input tokens | output tokens | cost | median latency |",
           "|---|---|---|---|---|---|---|---|"]
    by = cost_latency()
    grand = 0.0
    for (label, model), b in sorted(by.items()):
        grand += b["cost"]
        lat = f"{statistics.median(b['latency']):.1f} s" if b["latency"] else "n/a"
        note = f" ({b['unpriced']} unpriced)" if b["unpriced"] else ""
        md.append(f"| {label} | {model} | {b['calls']} | {b['errors']} | {b['in']:,} | "
                  f"{b['out']:,} | ${b['cost']:.2f}{note} | {lat} |")
    md += ["", f"Total spend: ${grand:.2f}", ""]
    out["total_cost"] = grand

    md += ["Per-item cost, one model vs. the three panel calls (before any judge or synthesizer):", ""]
    for task_set in TASK_SETS:
        s, p = per_item_panel_cost(task_set)
        if s:
            md.append(f"- {task_set}: {MODELS[0]} ${s:.4f}, three-model panel ${p:.4f} ({p / s:.1f}x)")
    md.append("")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "summary.md").write_text("\n".join(md))
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    main()
