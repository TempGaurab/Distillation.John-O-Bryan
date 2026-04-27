#!/usr/bin/env python3
"""
qwen_tuned_math500.py  –  Run the fine-tuned / fused Qwen model on the MATH-500
                           dataset and use DeepSeek to judge whether the answer
                           is correct.

Reads  : results-math500.csv  (columns: unique_id, subject, level, problem,
                                solution, true_answer, agent1_answer,
                                agent1_reasoning, agent2_verdict,
                                agent2_justification, correct_flag)
Writes : qwen_tuned_math500_results.csv

Requires:
    pip install mlx-lm openai python-dotenv

.env (same folder):
    Deepseek_api=your_key_here

Usage:
    python qwen_tuned_math500.py \
        --model   /path/to/fused_model \
        --data    /Users/gauurab/Documents/Projects/Distillation.John-O-Bryan/results-math500.csv \
        --output  qwen_tuned_math500_results.csv \
        --max-tokens 1500 \
        [--resume]
"""

import os
import re
import csv
import time
import argparse
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    default="fused_model/",
    help="Path to fused Qwen model directory",
)
parser.add_argument(
    "--data",
    default="/Users/gauurab/Documents/Projects/Distillation.John-O-Bryan/results-math500.csv",
    help="Input MATH-500 CSV file",
)
parser.add_argument(
    "--output",
    default="qwen_tuned_math500_results.csv",
    help="Output CSV filename",
)
parser.add_argument(
    "--max-tokens",
    type=int,
    default=1500,
    help="Max tokens for Qwen generation",
)
parser.add_argument(
    "--resume",
    action="store_true",
    help="Skip rows whose unique_id is already present in the output CSV",
)
args = parser.parse_args()

# ── Load env & build DeepSeek client ──────────────────────────────────────────
load_dotenv()
API_KEY = os.environ.get("Deepseek_api")
if not API_KEY:
    raise EnvironmentError(
        "Could not find 'Deepseek_api' in environment or .env file.\n"
        "Create a .env file with:  Deepseek_api=your_key_here"
    )

ds_client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
DEEPSEEK_MODEL = "deepseek-chat"

# ── Retry / rate-limit config ──────────────────────────────────────────────────
MAX_RETRIES   = 3
RETRY_DELAY   = 5.0
REQUEST_DELAY = 0.5   # seconds between consecutive DeepSeek calls

# ── Load Qwen model (mlx-lm) ──────────────────────────────────────────────────
print(f"Loading Qwen model from: {args.model}")
from mlx_lm import load, generate

qwen_model, qwen_tokenizer = load(args.model)
print("Qwen model loaded.\n")

# ── Helpers ────────────────────────────────────────────────────────────────────
def call_with_retry(fn, *fn_args, label="API call"):
    """Call fn(*fn_args), retrying up to MAX_RETRIES times on exception."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*fn_args)
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            print(f"  ⚠  {label} failed (attempt {attempt}/{MAX_RETRIES}): {exc}")
            time.sleep(RETRY_DELAY)


def qwen_generate(problem: str) -> str:
    """Run the fused Qwen model on a single MATH-500 problem."""
    messages = [{"role": "user", "content": problem}]
    try:
        prompt = qwen_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = problem   # plain fallback

    response = generate(
        qwen_model,
        qwen_tokenizer,
        prompt=prompt,
        max_tokens=args.max_tokens,
        verbose=False,
    )
    return response.strip()


# ── DeepSeek judge ─────────────────────────────────────────────────────────────
JUDGE_SYSTEM = (
    "You are a strict math answer checker. "
    "You will be given a competition math problem, the correct answer, and a model's response. "
    "Determine whether the model's final answer matches the correct answer "
    "(mathematically equivalent). "
    "Reply with ONLY one of:\n"
    "  CORRECT\n"
    "  INCORRECT\n"
    "Do not add any explanation."
)


def _judge_request(problem: str, true_answer: str, model_response: str) -> str:
    prompt = (
        f"Problem:\n{problem}\n\n"
        f"Correct answer: {true_answer}\n\n"
        f"Model response:\n{model_response}"
    )
    resp = ds_client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        max_tokens=16,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.0,
    )
    verdict = (resp.choices[0].message.content or "").strip().upper()
    if "CORRECT" in verdict and "INCORRECT" not in verdict:
        return "CORRECT"
    elif "INCORRECT" in verdict:
        return "INCORRECT"
    return verdict   # pass through unexpected values for inspection


def judge_answer(problem: str, true_answer: str, model_response: str) -> str:
    return call_with_retry(
        _judge_request, problem, true_answer, model_response,
        label="DeepSeek judge"
    )


# ── Extract final boxed / numeric answer from model output ────────────────────
def extract_boxed(text: str) -> list:
    r"""
    Extract all \boxed{...} contents with correct nested-brace handling.
    e.g. \boxed{\frac{1}{2}} → '\frac{1}{2}'
    """
    results = []
    i = 0
    while i < len(text):
        idx = text.find(r"\boxed{", i)
        if idx == -1:
            break
        start = idx + len(r"\boxed{")
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        results.append(text[start : j - 1])
        i = j
    return results


def extract_final_answer(text: str) -> str:
    """
    Priority:
      1. Last \\boxed{...}  (nested-brace aware)
      2. Last 'Answer: ...' line
      3. Last non-empty line of the response
    """
    boxes = extract_boxed(text)
    if boxes:
        return boxes[-1].strip()
    answer_lines = re.findall(r"(?i)answer\s*[:\-]\s*(.+)", text)
    if answer_lines:
        return answer_lines[-1].strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


# ── Resume support: load already-processed unique_ids ─────────────────────────
done_ids: set = set()
output_path = Path(args.output)
if args.resume and output_path.exists():
    with open(output_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid = row.get("unique_id", "")
            if uid:
                done_ids.add(uid)
    print(f"Resuming – {len(done_ids)} rows already done.\n")

# ── Load source data ───────────────────────────────────────────────────────────
input_path = Path(args.data)
if not input_path.exists():
    raise FileNotFoundError(f"Input file '{input_path}' not found.")

with open(input_path, newline="", encoding="utf-8") as f:
    reader     = csv.DictReader(f)
    src_fields = list(reader.fieldnames or [])
    rows       = list(reader)

print(f"Loaded {len(rows)} rows from '{input_path}'.\n")

# ── Output columns ─────────────────────────────────────────────────────────────
# Preserve all original columns; append our three new ones
NEW_COLS = ["qwen_response", "qwen_extracted_answer", "deepseek_verdict"]
out_fields = src_fields + [c for c in NEW_COLS if c not in src_fields]

# ── Counters ───────────────────────────────────────────────────────────────────
correct = incorrect = skipped = errors = 0
total = len(rows)

# Per-subject accuracy tracking
subject_stats: dict[str, dict] = {}   # { subject: {"correct": n, "total": n} }

# ── Main loop ──────────────────────────────────────────────────────────────────
write_mode = "a" if (args.resume and output_path.exists()) else "w"

with open(output_path, write_mode, newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=out_fields, extrasaction="ignore")
    if write_mode == "w":
        writer.writeheader()

    for i, row in enumerate(rows, 1):
        unique_id   = row.get("unique_id", "").strip()
        subject     = row.get("subject", "").strip()
        level       = row.get("level", "").strip()
        problem     = row.get("problem", "").strip()
        true_answer = row.get("true_answer", "").strip()
        flag        = row.get("correct_flag", "").strip()

        print(f"[{i}/{total}]  id={unique_id}  subject={subject}  level={level}  flag={flag}")

        # ── Resume: skip already-done rows ────────────────────────────────────
        if unique_id in done_ids:
            print("  ⏭  Already done, skipping.")
            skipped += 1
            continue

        # ── Skip placeholder / invalid rows ───────────────────────────────────
        if flag == "-1":
            print("  ⏭  Placeholder row (flag=-1), skipping.")
            out_row = dict(row)
            out_row["qwen_response"]         = "SKIPPED"
            out_row["qwen_extracted_answer"] = "SKIPPED"
            out_row["deepseek_verdict"]      = "SKIPPED"
            writer.writerow(out_row)
            csvfile.flush()
            skipped += 1
            continue

        if not problem:
            print("  ⏭  Empty problem field, skipping.")
            skipped += 1
            continue

        # ── Qwen generation ───────────────────────────────────────────────────
        qwen_response  = ""
        qwen_extracted = ""
        verdict        = "ERROR"

        try:
            print("  🤖 Generating with Qwen…", end=" ", flush=True)
            qwen_response  = qwen_generate(problem)
            qwen_extracted = extract_final_answer(qwen_response)
            print(f"done.  Extracted: '{qwen_extracted}'")
        except Exception as exc:
            print(f"\n  ✗  Qwen generation failed: {exc}")
            qwen_response  = f"GENERATION_ERROR: {exc}"
            qwen_extracted = "ERROR"
            errors += 1
            out_row = dict(row)
            out_row["qwen_response"]         = qwen_response
            out_row["qwen_extracted_answer"] = qwen_extracted
            out_row["deepseek_verdict"]      = verdict
            writer.writerow(out_row)
            csvfile.flush()
            continue

        # ── DeepSeek verdict ──────────────────────────────────────────────────
        try:
            print("  🔍 Asking DeepSeek to judge…", end=" ", flush=True)
            verdict = judge_answer(problem, true_answer, qwen_response)
            print(f"→ {verdict}")

            # Global counters
            if verdict == "CORRECT":
                correct += 1
            else:
                incorrect += 1

            # Per-subject counters
            if subject not in subject_stats:
                subject_stats[subject] = {"correct": 0, "total": 0}
            subject_stats[subject]["total"] += 1
            if verdict == "CORRECT":
                subject_stats[subject]["correct"] += 1

        except Exception as exc:
            print(f"\n  ✗  DeepSeek judge failed: {exc}")
            verdict = f"JUDGE_ERROR: {exc}"
            errors += 1

        time.sleep(REQUEST_DELAY)

        out_row = dict(row)
        out_row["qwen_response"]         = qwen_response
        out_row["qwen_extracted_answer"] = qwen_extracted
        out_row["deepseek_verdict"]      = verdict
        writer.writerow(out_row)
        csvfile.flush()

# ── Summary ────────────────────────────────────────────────────────────────────
judged   = correct + incorrect
accuracy = (correct / judged * 100) if judged else 0.0

print(f"\n{'='*65}")
print(f"✅  Done!  Results saved to '{output_path}'.")
print(f"   Total rows      : {total}")
print(f"   Judged          : {judged}")
print(f"   Correct         : {correct}  ({accuracy:.1f}%)")
print(f"   Incorrect       : {incorrect}")
print(f"   Skipped         : {skipped}")
print(f"   Errors          : {errors}")

if subject_stats:
    print(f"\n{'─'*65}")
    print("   Per-subject breakdown:")
    for subj, stats in sorted(subject_stats.items()):
        s_acc = (stats["correct"] / stats["total"] * 100) if stats["total"] else 0.0
        print(f"     {subj:<35} {stats['correct']:>3}/{stats['total']:<3}  ({s_acc:.1f}%)")

print(f"{'='*65}")
