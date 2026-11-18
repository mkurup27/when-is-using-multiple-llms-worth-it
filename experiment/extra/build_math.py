"""Build the math positive-control set: 100 MATH-500 problems, levels 4-5, integer answers only
(so answers compare exactly). MATH is MIT-licensed (Hendrycks et al., 2021).

Run: python extra/build_math.py
"""
import json
import random
import re
import urllib.request

from extra_common import DATA, MATH_INSTRUCTION, write_jsonl

URL = "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl"
SEED = 4155

if __name__ == "__main__":
    with urllib.request.urlopen(URL, timeout=60) as r:
        rows = [json.loads(line) for line in r.read().decode().splitlines() if line.strip()]
    pool = [p for p in rows if p["level"] >= 4 and re.fullmatch(r"-?\d+", p["answer"].strip())]
    picked = random.Random(SEED).sample(pool, min(100, len(pool)))
    out = [{"id": f"math-{i:03d}", "task_type": "math", "kind": "int", "level": p["level"],
            "subject": p["subject"], "source_id": p["unique_id"],
            "prompt": p["problem"] + MATH_INSTRUCTION, "answer": p["answer"].strip()}
           for i, p in enumerate(picked)]
    write_jsonl(DATA / "math.jsonl", out)
    print(f"{len(rows)} problems, {len(pool)} at level 4-5 with integer answers, "
          f"{len(out)} written to extra/data/math.jsonl")
