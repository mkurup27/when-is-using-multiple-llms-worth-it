"""Shared settings for the follow-up checks: math positive control, cascade test, held-out
comparison, and the judge worksheet."""
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (DATA as MAIN_DATA, RAW as MAIN_RAW, call_cost, final_answer,  # noqa: E402,F401
                    load_prices, make_client, normalize, read_jsonl, safe, write_jsonl)

ROOT = Path(__file__).resolve().parent
FAKE = bool(os.environ.get("FAKE_LLM"))
DATA = ROOT / "data"
OUT = ROOT / ("outputs_fake" if FAKE else "outputs")
RAW = OUT / "raw"

SMALL = os.environ.get("SMALL_MODELS", "deepseek-v4.1-flash,glm-5.3-flash,openai-gpt-oss-120b").split(",")
STRONG = os.environ.get("STRONG_MODEL", "glm-5.2")
REPEATS = 3  # runs of SMALL[0] per math problem, for the noise floor and self-consistency
MAX_TOKENS = {"math": 16000, "classification": 8000, "extraction_hard": 8000}
MATH_INSTRUCTION = ("\n\nEnd your reply with one line in the form FINAL: <answer>, "
                    "where <answer> is only the final integer.")

# Small models from the DigitalOcean pricing page (USD per 1M tokens). Check slugs with
# `python extra/run_extra.py list-models`; add any slug you switch to here.
EXTRA_PRICES = {
    "deepseek-v4-flash": {"input": 0.14, "output": 0.28},
    "deepseek-v4.1-flash": {"input": 0.30, "output": 1.20},
    "glm-5.3-flash": {"input": 0.15, "output": 0.50},
    "openai-gpt-oss-120b": {"input": 0.10, "output": 0.70},
    "gpt-oss-120b": {"input": 0.10, "output": 0.70},
    "minimax-m3.1-flash": {"input": 0.15, "output": 0.60},
    "openai-gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "anthropic-claude-haiku-4.5": {"input": 1.00, "output": 5.00},
    "openai-gpt-5-nano": {"input": 0.05, "output": 0.40},
    "openai-gpt-5.4-nano": {"input": 0.20, "output": 1.25},
}


def prices():
    return {**load_prices(), **EXTRA_PRICES}


def int_answer(value):
    """Normalize a math answer to an integer string, so '1,234', '$1234$' and '1234.0' match."""
    if value is None:
        return None
    v = value.replace(",", "").replace("\\!", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", v)
    if not nums:
        return v.strip().lower()
    x = nums[-1]
    if "." in x:
        f = float(x)
        return str(int(f)) if f.is_integer() else x
    return str(int(x))


def norm(value, kind):
    return int_answer(value) if kind == "int" else normalize(value, kind)


def raw_path(base, task_set, item_id, model, rep):
    return base / task_set / f"{safe(item_id)}__{safe(model)}__{rep}.json"


def load_raw(base, task_set, item_id, model, rep):
    p = raw_path(base, task_set, item_id, model, rep)
    return json.loads(p.read_text()) if p.exists() else None


def math_final(text):
    """FINAL answer for math, tolerating Markdown such as '**FINAL: 12**' or '### FINAL: 12'."""
    for line in reversed(text.strip().splitlines()):
        m = re.match(r"[\s*#>_`]*FINAL\s*(?:ANSWER)?\s*[:=]\s*(.*)", line, re.I)
        if m:
            return m.group(1).strip().strip("*_` ")
    return last_boxed(text)


def last_boxed(text):
    """Contents of the last \\boxed{...}, with nested braces; MATH's own answer convention."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i, depth = start + len("\\boxed{"), 1
    for j in range(i, len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[i:j]
    return None


def parsed(call, kind):
    """Normalized FINAL answer of a cached call, or None if it failed or had no FINAL line."""
    if not call or call.get("error"):
        return None
    if kind == "int":
        return norm(math_final(call["text"]), kind)
    return norm(final_answer(call["text"]), kind)


class FakeExtraClient:
    """FAKE_LLM=1 stand-in: math prompts get a random integer, other prompts use common's fake."""

    def __init__(self):
        from common import FakeClient
        self.base = FakeClient()
        self.models = self.base.models
        self.chat = type("Chat", (), {})()
        self.chat.completions = type("Comp", (), {})()
        self.chat.completions.create = self._create

    def _create(self, model, messages, max_tokens, temperature=None):
        prompt = messages[0]["content"]
        if "only the final integer" not in prompt:
            return self.base._create(model, messages, max_tokens, temperature)
        rng = random.Random(f"{model}|{prompt}|{time.time_ns() % 3}")
        O = self.base._Obj
        text = f"Working.\nFINAL: {rng.choice(['12', '12', '7', '30'])}"
        return O(choices=[O(message=O(content=text), finish_reason="stop")],
                 usage=O(prompt_tokens=len(prompt) // 4, completion_tokens=900))


def client(timeout=1200):
    return FakeExtraClient() if FAKE else make_client(timeout=timeout)
