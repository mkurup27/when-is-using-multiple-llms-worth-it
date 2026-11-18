"""Shared settings and helpers for the disagreement experiment."""
import json
import os
import random
import re
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = ROOT / "outputs" / "raw"
JUDGE = ROOT / "outputs" / "judge"
RESULTS = ROOT / "results"

BASE_URL = os.environ.get("BASE_URL", "https://inference.do-ai.run/v1")
MODELS = os.environ.get("MODELS", "glm-5.2,kimi-k2.6,deepseek-v4-pro").split(",")
REPEATS = 3  # runs of MODELS[0] per item, for the noise floor
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "anthropic-claude-sonnet-5.5")
TASK_SETS = ["classification", "extraction", "research", "extraction_hard"]

MAX_TOKENS = {"classification": 8000, "extraction": 8000, "research": 16000,
              "extraction_hard": 8000}
JUDGE_MAX_TOKENS = 8000

FINAL_INSTRUCTION = (
    "\n\nEnd your reply with one line in the form FINAL: <answer>, "
    "where <answer> is only the label, option, or value."
)


def calls_for(item_id):
    """(model, repeat) pairs to run for one item: every model once, plus repeats of MODELS[0]."""
    return [(m, 0) for m in MODELS] + [(MODELS[0], r) for r in range(1, REPEATS)]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def safe(name):
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def load_prices():
    with open(ROOT / "prices.json") as f:
        return json.load(f)


def call_cost(model, usage, prices):
    p = prices.get(model)
    if not p or not usage:
        return None
    return (usage.get("prompt_tokens", 0) * p["input"]
            + usage.get("completion_tokens", 0) * p["output"]) / 1_000_000


def make_client(timeout):
    if os.environ.get("FAKE_LLM"):
        return FakeClient()
    from openai import OpenAI

    key = os.environ.get("MODEL_ACCESS_KEY") or os.environ.get("API_KEY")
    if not key:
        raise SystemExit("Set MODEL_ACCESS_KEY (a DigitalOcean model access key) first.")
    return OpenAI(base_url=BASE_URL, api_key=key, timeout=timeout, max_retries=0)


def chat(client, model, prompt, max_tokens, temperature=None, attempts=4):
    """One chat completion with retries. Returns a dict that is safe to cache as JSON."""
    kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}],
              "max_tokens": max_tokens}
    if temperature is not None:
        kwargs["temperature"] = temperature
    error = None
    for attempt in range(attempts):
        start = time.time()
        try:
            r = client.chat.completions.create(**kwargs)
            usage = r.usage.model_dump() if getattr(r, "usage", None) else {}
            return {
                "model": model,
                "text": r.choices[0].message.content or "",
                "finish_reason": r.choices[0].finish_reason,
                "usage": usage,
                "latency_s": round(time.time() - start, 2),
                "error": None,
            }
        except Exception as e:  # network errors, 429s, 5xx
            error = f"{type(e).__name__}: {e}"
            time.sleep(min(60, 5 * 2 ** attempt))
    return {"model": model, "text": "", "finish_reason": None, "usage": {},
            "latency_s": None, "error": error}


class FakeClient:
    """Stand-in for the API (FAKE_LLM=1), so the pipeline can be tested without a key."""

    class _Obj:
        def __init__(self, **kw):
            self.__dict__.update(kw)

        def model_dump(self):
            return dict(self.__dict__)

    def __init__(self):
        self.chat = self._Obj(completions=self._Obj(create=self._create))
        listed = [self._Obj(id=m) for m in MODELS + [JUDGE_MODEL]]
        self.models = self._Obj(list=lambda: self._Obj(data=listed))

    def _create(self, model, messages, max_tokens, temperature=None):
        prompt = messages[0]["content"]
        rng = random.Random(f"{model}|{prompt}|{time.time_ns() % 3}")
        if "rubric" in prompt and "criterion" in prompt:
            ids = re.findall(r"^- \[([^\]]+)\]", prompt, re.M)
            text = json.dumps({"verdicts": {i: rng.choice(["MET", "UNMET"]) for i in ids}})
        elif "Allowed intents:" in prompt:
            labels = re.findall(r"^- (\S+)$", prompt, re.M)
            text = f"Reasoning.\nFINAL: {rng.choice(labels[:3])}"
        elif "extract" in prompt.lower():
            text = "Reasoning.\nFINAL: " + rng.choice(["2026-03-14", "1,234.50", "INV-1001"])
        else:
            text = "A long research answer. " * 20
        usage = self._Obj(prompt_tokens=len(prompt) // 4, completion_tokens=len(text) // 4)
        msg = self._Obj(content=text)
        return self._Obj(choices=[self._Obj(message=msg, finish_reason="stop")], usage=usage)


# ---- answer normalization (classification and extraction) ----

def final_answer(text):
    for line in reversed(text.strip().splitlines()):
        if line.strip().upper().startswith("FINAL:"):
            return line.split(":", 1)[1].strip()
    return None


def normalize(value, kind):
    """Normalize a FINAL answer so formatting differences don't count as disagreement."""
    if value is None:
        return None
    v = value.strip().strip("`*\"' ").rstrip(".")
    if kind == "amount":
        v = re.sub(r"[^\d.\-]", "", v)
        try:
            return f"{float(v):.2f}"
        except ValueError:
            return v.lower()
    if kind == "date":
        m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", v)
        if m:
            try:
                return date(int(m[1]), int(m[2]), int(m[3])).isoformat()
            except ValueError:
                pass
        return v.lower()
    return re.sub(r"\s+", " ", v).lower()
