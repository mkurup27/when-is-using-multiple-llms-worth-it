# Multi-LLM Disagreement: A Measured Test Before You Ensemble

**A reproducible measurement of when different models give different answers on the same request, compared with how often one model disagrees with itself, plus the vote and cascade follow-ups that check what that disagreement predicts.**

- **Main run:** three models on DigitalOcean Serverless Inference (`glm-5.2`, `kimi-k2.6`, `deepseek-v4-pro`), with three GLM-5.2 runs per request for the noise floor
- **Judge:** Claude Sonnet 5.5, not on the panel, scoring DRACO rubric criteria as met or unmet
- **Request sets:** 50 Banking77 classification messages, 50 easy + 50 hard synthetic extraction docs, 25 DRACO research tasks (no web search)
- **Follow-ups:** 100 MATH-500 problems (levels 4–5, integer answers) with two three-model panels; cascade with DeepSeek V4.1 Flash + GLM-5.3 Flash escalating to GLM-5.2 on disagreement
- **Cost:** about **$7.50** for the main panel + judge run; math vote of three comparable small models at **$0.0032** per problem
- **Everything in this repository is what actually ran:** the afternoon disagreement script, the full experiment harness, the request sets, the prices file used for cost estimates, and the analysis that produced the tables

This repository is the evidence base for a companion article, *"When Is Using Multiple LLMs Worth It? Ensembling vs. Routing vs. One Model"* (DigitalOcean Community).

---

## Abstract

Ensembling spends latency and money to lower the chance that one model's mistake reaches the user. Whether that pays off starts with a cheap measurement: how often the models you'd combine give different answers on your requests.

This study sends the same requests to three models in parallel, extracts a comparable `FINAL:` answer from each, and subtracts a noise floor: how often the model you'd otherwise deploy disagrees with *itself* across three samples. The gap, *excess* disagreement, is a rough signal that a panel might do something a single model would not. Where labeled answers exist, it also counts how often every model agrees and is wrong (no vote can fix that) and how often at least one model is right but majority voting misses it (the most a judge or synthesizer could recover).

Disagreement depends on the task, and a majority vote rarely beats the best single model. On easy extraction, three models never disagreed. On classification they disagreed on 26% of messages against a 10% noise floor (+16 pts), and the vote scored 78%, behind the best single model at 80%. On hard extraction, excess disagreement of +14 pts pointed at one weak model's vendor-name errors rather than a gap only a panel fills. On research, cross-model disagreement barely cleared the noise floor (+3 pts), and a perfect union of three models would add almost nothing over three samples of one model.

A follow-up on math with one final number, where voting is known to help, went the other way: a vote of three comparable small models scored **100%** and beat each of them, while a panel that included one weak model did worse than its best member. A disagreement-triggered cascade matched GLM-5.2 at **0.19× to 0.46×** its cost.

Run the disagreement test on your own requests before you pay for a panel. Results from one task type do not transfer to another, and cheaper patterns (one strong model, a cascade, or a vote of comparable cheap models) often capture the gain.

---

## Key findings

### 1. Excess disagreement varies from zero to a clear signal

| Task type | n | Models differ | Same model differs | Excess | Average single | Best single | Majority vote |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Classification | 50 | 26% | 10% | **+16 pts** | 75% | 80% | 78% |
| Extraction, easy | 50 | 0% | 0% | 0 pts | 100% | 100% | 100% |
| Extraction, hard | 50 | 14% | 0% | **+14 pts** | 95% | 100% | 100% |
| Research (rubric criteria) | 25 | 26% | 23% | +3 pts | 43% coverage | 44% coverage | n/a |

On the three sets with a vote, the vote never beat the best single model, but wherever the models disagreed, it beat the average one. Panel calls alone cost 2.0× to 2.9× a single GLM-5.2 call, before any judge or synthesizer.

### 2. High disagreement can point to a weaker model to drop

On hard extraction, GLM-5.2 and Kimi K2.6 got all 50 right. DeepSeek V4 Pro got 86%, and every disagreement came from its misses (often naming a factoring or forwarding company instead of the supplier). The vote scored 100%, and so did either of the other two models alone. The vote's 7-to-0 lead over DeepSeek V4 Pro was the only statistically significant vote-vs-single gap across the three labeled sets (exact McNemar, p = 0.016).

### 3. On research, mixing models barely beats sampling one model three times

Models split on 26% of rubric criteria; GLM-5.2's own runs split on 23%. The union of three models' answers would cover at most **7.5 points** more rubric weight than the best single answer; the union of three GLM-5.2 runs covers **7.4 points**. The judge disagreed with itself on 2% of criteria when grading the same answer twice.

### 4. Voting wins when members are comparable and make different mistakes

| | Panel of comparable models | Panel with one weak model |
| --- | --- | --- |
| Models (accuracy alone) | DeepSeek V4.1 Flash (98%), gpt-oss-120b (96%), GLM-5.3 Flash (95%) | Claude Haiku 4.5 (96%), GPT-5 nano (93%), GPT-4o mini (48%) |
| Average single model | 96% | 79% |
| Majority vote | **100%** | 93% |
| Vote vs. best model chosen on held-out data | 3.9 pts ahead on average; never behind | 2.3 pts behind on average; behind in 77% of splits |
| Cost per problem, all three models | $0.0032 | $0.0050 |

The comparable panel also beat GLM-5.2 alone (98%) at about a fifth of the cost ($0.0032 vs $0.0154). The margin on the full set is 2 problems in 100 and is not statistically significant. Adding a weaker model can make the vote worse than its best member.

### 5. A disagreement-triggered cascade can match the escalator at a fraction of the cost

| Task | Requests | Escalated | Cascade accuracy | GLM-5.2 alone | Cascade cost vs. GLM-5.2 |
| --- | --- | --- | --- | --- | --- |
| Classification | 49 | 12% | 76% | 76% | **0.46×** |
| Extraction, hard | 50 | 0% | 100% | 100% | **0.22×** |
| Math | 96 | 4% | 100% | 100% | **0.19×** |

Two cheap models answered first; only disagreements went to GLM-5.2. The cascade matched the escalator, but it did not beat the cheap model alone when that model already matched GLM-5.2. Measure that first.

### 6. Shared wrong answers and vote misses

On classification, all three models agreed on the same wrong intent on **12%** of messages (6 of 50); no vote can fix that. On **4%** (2 of 50), at least one model was right but the vote missed it, which is the most a perfect judge or synthesizer could have recovered on that set. Held-out splits (2,000 random halves) found the "best" single model and the vote tied on average.

---

## Methodology

### Fixed variables

- DigitalOcean Serverless Inference, OpenAI-compatible endpoint (`https://inference.do-ai.run/v1`)
- Panel: `glm-5.2`, `kimi-k2.6`, `deepseek-v4-pro` (one call each per request)
- Noise floor: three runs of `MODELS[0]` (`glm-5.2`) on the same request
- Answers compared as exact strings after lowercasing, taken from a required `FINAL:` line; extraction values normalized before comparison
- Research judge: Claude Sonnet 5.5, outside the panel; all criteria of one answer graded in a single call
- Dataset seed **4155** for reproducible sampling

### Main run — four request sets

1. **Classification.** 50 Banking77 test messages (PolyAI, CC-BY-4.0), one from each of 50 intents; all 77 intents listed in the prompt.
2. **Extraction, easy.** 50 synthetic invoices and billing emails from known field values; half with simple distractors (e.g. subtotal next to total).
3. **Extraction, hard.** 50 synthetic documents with harder traps (credit notes, payment-term due dates, factoring letterhead, cancelled then replaced POs).
4. **Research.** 25 DRACO tasks across domains; answered with no web access; scored per rubric criterion.

### Follow-ups

- **Math positive control.** 100 MATH-500 problems, levels 4–5, integer answers only. Two panels of three small models; unreadable final answers counted wrong.
- **Cascade.** DeepSeek V4.1 Flash and GLM-5.3 Flash first; escalate to GLM-5.2 only when they disagree. Same classification, hard-extraction, and math requests.
- **Held-out comparison.** Random half-splits pick the best single model on one half and score it against the vote on the other; exact McNemar tests on paired right/wrong.

### Afternoon script

`disagreement_test.py` is the portable version of steps 3–5 from the article: any OpenAI-compatible endpoint, your own `requests.jsonl`, optional `task_type` and `answer` fields for the ground-truth columns.

### Known limitations, stated plainly

- Samples are small (50 per classification/extraction set, 25 research tasks, 100 math problems). Gaps of a few points (e.g. 78% vs 80%) are within noise.
- Extraction documents are synthetic.
- Research answers had no web access; DRACO is built for search agents. Coverage here is share of positive rubric weight met, not DRACO's official score.
- The judge grades all criteria of an answer in one call; DRACO's protocol uses one call per criterion. A hand-check of 20 verdicts on four short answers agreed with 16.
- Five classification replies hit the 8,000-token output budget without a `FINAL:` line and were rerun with 32,000 tokens.
- Cascade rows leave out requests where one of the three models gave no readable answer.
- This repository ships the harness and request sets. Live API calls regenerate raw outputs under `experiment/outputs/` and `experiment/extra/outputs/`.
- Upstream dataset licenses still apply (Banking77 CC-BY-4.0; MATH MIT; check DRACO on Hugging Face before redistributing).

---

## Repository layout

```bash
disagreement_test.py         Afternoon script from the article: your requests → disagreement table.
examples/
  requests.jsonl             Tiny sample input (classification + open-ended).
requirements.txt             openai>=1.0.0
experiment/
  common.py                  Shared settings, chat helper, FINAL: parsing.
  run_calls.py               Panel + noise-floor calls for the four main sets.
  judge_research.py          Rubric judge for research answers.
  analyze.py                 Summary tables → results/summary.md.
  build_datasets.py          Regenerate data/ (seed 4155).
  prices.json                USD per 1M tokens for cost estimates.
  data/
    classification.jsonl
    extraction.jsonl
    extraction_hard.jsonl
    research.jsonl
  extra/
    build_math.py            Download MATH-500 → extra/data/math.jsonl.
    run_extra.py             Math panel + cascade calls.
    analyze_extra.py         Math / cascade / held-out tables.
    make_worksheet.py        Hand-check worksheet for judge verdicts.
    extra_common.py          Shared settings for follow-ups.
```

## Reproducing

Against your own requests (the numbers that matter for *your* workload):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export MODEL_ACCESS_KEY=...          # DigitalOcean model access key, or set API_KEY directly
API_KEY=$MODEL_ACCESS_KEY python disagreement_test.py examples/requests.jsonl
```

Edit `MODELS` at the top of `disagreement_test.py` to match your endpoint. Put the model you'd deploy first: the noise floor is measured on `MODELS[0]`. Optional: `BASE_URL` (default `https://inference.do-ai.run/v1`).

Against the article's request sets (live API; regenerates outputs):

```bash
cd experiment
export MODEL_ACCESS_KEY=...

python run_calls.py --check-models     # confirm slugs; fix MODELS / JUDGE_MODEL if needed
python run_calls.py --dry-run          # rough panel cost
python judge_research.py --dry-run     # rough judge cost
python run_calls.py --limit 2          # smoke test
python run_calls.py                    # full panel run; rerun to retry failures
python judge_research.py               # research judge; rerun to retry failures
python analyze.py                      # writes results/summary.md
```

Override models without editing code:

```bash
MODELS=glm-5.2,kimi-k2.6,deepseek-v4-pro JUDGE_MODEL=openai-gpt-5.5 python run_calls.py
```

If you change `JUDGE_MODEL` or `MODELS`, add the slug and its prices to `prices.json`. `build_datasets.py` has already been run; rerunning it reproduces the same files (seed 4155).

Math and cascade follow-ups:

```bash
cd experiment
python extra/build_math.py             # writes extra/data/math.jsonl
python extra/run_extra.py list-models  # find small-model slugs
python extra/run_extra.py run          # math panel + cascade calls
python extra/analyze_extra.py          # writes extra/outputs/extra_summary.md
```

Model availability and pricing change. If your results differ from anything reported in the article, that difference is the finding. Re-run on your endpoint and your requests.

---

## Citation

If you use this data or method, please cite:

```bash
When Is Using Multiple LLMs Worth It? Ensembling vs. Routing vs. One Model (2026).
https://github.com/mkurup27/when-is-using-multiple-llms-worth-it
```

## License

MIT. See [LICENSE](LICENSE).
