"""
prepare_finetune.py
───────────────────
Reads results.csv → filters to CORRECT rows → saves clean JSONL for fine-tuning.

Keeps only:
  - question
  - distilled_reasoning  (compact teaching chain-of-thought)
  - true_answer

Filters:
  - agent2_verdict == CORRECT   (verifier confirmed the answer)
  - distilled_reasoning not empty

Output:
  data/train.jsonl
  data/valid.jsonl
  data/test.jsonl

Usage:
    python prepare_finetune.py
    python prepare_finetune.py --csv results.csv --out data/ --split 0.85 0.10 0.05
"""

import argparse
import csv
import json
import random
from pathlib import Path

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--csv",   default="results2.csv")
parser.add_argument("--out",   default="data")
parser.add_argument("--split", nargs=3, type=float, default=[0.85, 0.10, 0.05],
                    metavar=("TRAIN", "VALID", "TEST"))
parser.add_argument("--seed",  type=int, default=42)
args = parser.parse_args()

assert abs(sum(args.split) - 1.0) < 1e-6, "Split values must sum to 1.0"

SYSTEM_PROMPT = (
    "You are a precise math competition solver. "
    "Think through the problem step by step, then state the answer clearly."
)

# ── Load & filter ──────────────────────────────────────────────────────────────
rows = []
skipped = 0

with open(args.csv, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        verdict   = row.get("agent2_verdict", "").strip().upper()
        question  = row.get("question", "").strip()
        reasoning = row.get("distilled_reasoning", "").strip()
        answer    = row.get("true_answer", "").strip()

        if verdict != "CORRECT":
            skipped += 1
            continue
        if not question or not reasoning or not answer:
            skipped += 1
            continue

        # Ensure reasoning ends with the answer
        answer_line = f"Answer: {answer}"
        if not reasoning.rstrip().endswith(answer_line):
            assistant = f"{reasoning}\n\n{answer_line}"
        else:
            assistant = reasoning

        rows.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": question},
                {"role": "assistant", "content": assistant},
            ]
        })

print(f"Loaded  : {len(rows)} usable rows")
print(f"Skipped : {skipped}  (wrong verdict, empty fields)")

if len(rows) < 5:
    raise ValueError("Not enough data — need at least 5 CORRECT rows in the CSV.")

# ── Split ──────────────────────────────────────────────────────────────────────
random.seed(args.seed)
random.shuffle(rows)

n       = len(rows)
n_train = int(n * args.split[0])
n_valid = int(n * args.split[1])

splits = {
    "train": rows[:n_train],
    "valid": rows[n_train : n_train + n_valid],
    "test":  rows[n_train + n_valid :],
}

# ── Write JSONL ────────────────────────────────────────────────────────────────
out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)

for name, records in splits.items():
    path = out_dir / f"{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  {name:5s} → {path}  ({len(records)} examples)")

# ── Preview first record ───────────────────────────────────────────────────────
print("\n── First training record ──────────────────────────────────")
print(json.dumps(splits["train"][0], indent=2, ensure_ascii=False))
print("\n✅  Done — ready to pass to finetune.py")