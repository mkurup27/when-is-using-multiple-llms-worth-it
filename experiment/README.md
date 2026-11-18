# Disagreement experiment

Harness used for the article's measured results. For the short afternoon script and a
quick start, see the [repository README](../README.md).

Measures how often three different models disagree on the same request, compared with how
often one model disagrees with itself, on three task types:

- **classification**: 50 Banking77 messages (PolyAI, CC-BY-4.0), 77 allowed intents.
- **extraction**: 50 synthetic invoices and billing emails built from known field values.
  Half are clean; half have distractors (subtotal vs. total, issue vs. due date, a later
  correction, an old PO number).
- **research**: 25 DRACO tasks (perplexity-ai/draco) across 10 domains, scored per rubric
  criterion by a judge model outside the panel. No web access, so expect low coverage.

The panel is called directly (one call per model); the synthesis tool is not used.

## Run

```bash
pip install openai
export MODEL_ACCESS_KEY=...            # DigitalOcean model access key

python run_calls.py --check-models     # confirm slugs; fix MODELS / JUDGE_MODEL if any are missing
python run_calls.py --dry-run          # rough panel cost
python judge_research.py --dry-run     # rough judge cost
python run_calls.py --limit 2          # smoke test, then read a few files in outputs/raw/
python run_calls.py                    # all 625 calls; rerun to retry failures
python judge_research.py               # 150 judge calls; rerun to retry failures
python analyze.py                      # writes results/summary.md
```

Override models without editing code:

```bash
MODELS=glm-5.2,kimi-k2.6,deepseek-v4-pro JUDGE_MODEL=openai-gpt-5.5 python run_calls.py
```

If you change `JUDGE_MODEL` or `MODELS`, add the slug and its prices to `prices.json`.
`build_datasets.py` has already been run; rerunning it reproduces the same files (seed 4155).

## What to disclose in the article

- Extraction documents are synthetic.
- Research answers had no web access; DRACO is built for research agents with search.
- The judge grades all criteria of an answer in one call; DRACO's protocol uses one call
  per criterion. The "judge disagrees with itself" row shows how noisy that judge is.
- Coverage is our simplification (share of positive rubric weight met), not DRACO's score.
- Union coverage is an upper bound for merging answers, not what the synthesis tool produces.
