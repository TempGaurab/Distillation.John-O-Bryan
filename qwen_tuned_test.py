#!/usr/bin/env python3
"""
qwen_tuned_test.py  –  Run the fine-tuned / fused Qwen model on the full dataset
                        and use DeepSeek to judge whether the answer is correct.

Reads  : results.csv   (must contain columns: year, section, question_id,
                         question, true_answer)
Writes : qwen_tuned_results.csv

Requires:
    pip install mlx-lm openai python-dotenv

.env (same folder):
    Deepseek_api=your_key_here

Usage:
    python qwen_tuned_test.py \
        --model   /path/to/fused_model \
        --data    results.csv \
        --output  qwen_tuned_results.csv \
        --max-tokens 600
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
parser.add_argument("--model",      default="fused_model/",
                    help="Path to fused Qwen model directory")
parser.add_argument("--data",       default="results.csv",
                    help="Input CSV with question/answer columns")
parser.add_argument("--output",     default="qwen_tuned_results_200iterations.csv",
                    help="Output CSV filename")
parser.add_argument("--max-tokens", type=int, default=1500,
                    help="Max tokens for Qwen generation")
parser.add_argument("--resume",     action="store_true",
                    help="Skip rows already present in output CSV")
args = parser.parse_args()

# ── Load env & build DeepSeek client ─────────────────────────────────────────
load_dotenv()
API_KEY = os.environ.get("Deepseek_api")
if not API_KEY:
    raise EnvironmentError(
        "Could not find 'Deepseek_api' in environment or .env file.\n"
        "Create a .env file with:  Deepseek_api=your_key_here"
    )

ds_client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
DEEPSEEK_MODEL = "deepseek-chat"

# ── Retry config ───────────────────────────────────────────────────────────────
MAX_RETRIES   = 3
RETRY_DELAY   = 5.0
REQUEST_DELAY = 0.5   # between DeepSeek calls

# ── Load Qwen model (mlx-lm) ──────────────────────────────────────────────────
print(f"Loading Qwen model from: {args.model}")
from mlx_lm import load, generate

qwen_model, qwen_tokenizer = load(args.model)
print("Qwen model loaded.\n")

# ── Helpers ────────────────────────────────────────────────────────────────────
def call_with_retry(fn, *fn_args, label="API call"):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*fn_args)
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            print(f"  ⚠  {label} failed (attempt {attempt}/{MAX_RETRIES}): {exc}")
            time.sleep(RETRY_DELAY)


def qwen_generate(question: str) -> str:
    """Run the fused Qwen model on a single question."""
    # Build a chat-style prompt using the tokenizer's chat template if available
    messages = [{"role": "user", "content": question}]
    try:
        prompt = qwen_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback: plain prompt
        prompt = question

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
    "Determine whether the model's final answer matches the correct answer (mathematically equivalent). "
    "Reply with ONLY one of:\n"
    "  CORRECT\n"
    "  INCORRECT\n"
    "Do not add any explanation."
)


def _judge_request(question: str, true_answer: str, model_response: str) -> str:
    prompt = (
        f"Problem:\n{question}\n\n"
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
    # Normalise – sometimes the model adds punctuation
    if "CORRECT" in verdict and "INCORRECT" not in verdict:
        return "CORRECT"
    elif "INCORRECT" in verdict:
        return "INCORRECT"
    return verdict   # pass through unexpected values for inspection


def judge_answer(question: str, true_answer: str, model_response: str) -> str:
    return call_with_retry(
        _judge_request, question, true_answer, model_response,
        label="DeepSeek judge"
    )


# ── Extract final boxed / numeric answer from model output ────────────────────
def extract_boxed(text: str) -> list:
    """
    Extract all \\boxed{...} contents with correct nested-brace handling.
    e.g. \\boxed{\\frac{1}{2}} correctly returns '\\frac{1}{2}'
    instead of stopping at the first closing brace.
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


# ── Resume support: load already-processed question IDs ───────────────────────
done_ids: set = set()
output_path = Path(args.output)
if args.resume and output_path.exists():
    with open(output_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done_ids.add(
                (row.get("year",""), row.get("section",""), row.get("question_id",""))
            )
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
# Preserve all original columns, add our new ones
NEW_COLS = ["qwen_response", "qwen_extracted_answer", "deepseek_verdict"]
out_fields = src_fields + [c for c in NEW_COLS if c not in src_fields]

# ── Main loop ──────────────────────────────────────────────────────────────────
correct = incorrect = skipped = errors = 0
total = len(rows)

write_mode = "a" if (args.resume and output_path.exists()) else "w"

with open(output_path, write_mode, newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=out_fields, extrasaction="ignore")
    if write_mode == "w":
        writer.writeheader()

    for i, row in enumerate(rows, 1):
        year    = row.get("year", "")
        section = row.get("section", "")
        q_id    = row.get("question_id", "")
        flag    = row.get("correct_flag", "")

        uid = (year, section, q_id)
        print(f"[{i}/{total}] {year} / {section} / {q_id}  (flag={flag})")

        # ── Skip resume ────────────────────────────────────────────────────────
        if uid in done_ids:
            print("  ⏭  Already done, skipping.")
            skipped += 1
            continue

        # ── Skip placeholder rows ──────────────────────────────────────────────
        if flag == "-1":
            print("  ⏭  Placeholder row (flag=-1), skipping.")
            out_row = dict(row)
            out_row["qwen_response"]          = "SKIPPED"
            out_row["qwen_extracted_answer"]  = "SKIPPED"
            out_row["deepseek_verdict"]       = "SKIPPED"
            writer.writerow(out_row)
            csvfile.flush()
            skipped += 1
            continue

        question    = row.get("question", "").strip()
        true_answer = row.get("true_answer", "").strip()

        if not question:
            print("  ⏭  Empty question, skipping.")
            skipped += 1
            continue

        # ── Qwen generation ────────────────────────────────────────────────────
        qwen_response         = ""
        qwen_extracted        = ""
        verdict               = "ERROR"

        try:
            print("  🤖 Generating with Qwen…", end=" ", flush=True)
            qwen_response  = qwen_generate(question)
            qwen_extracted = extract_final_answer(qwen_response)
            print(f"done. Extracted: '{qwen_extracted}'")
        except Exception as exc:
            print(f"\n  ✗  Qwen generation failed: {exc}")
            qwen_response  = f"GENERATION_ERROR: {exc}"
            qwen_extracted = "ERROR"
            errors += 1
            # Still write the row so we don't lose progress
            out_row = dict(row)
            out_row["qwen_response"]         = qwen_response
            out_row["qwen_extracted_answer"] = qwen_extracted
            out_row["deepseek_verdict"]      = verdict
            writer.writerow(out_row)
            csvfile.flush()
            continue

        # ── DeepSeek verdict ───────────────────────────────────────────────────
        try:
            print("  🔍 Asking DeepSeek to judge…", end=" ", flush=True)
            verdict = judge_answer(question, true_answer, qwen_response)
            print(f"→ {verdict}")
            if verdict == "CORRECT":
                correct += 1
            else:
                incorrect += 1
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
judged = correct + incorrect
accuracy = (correct / judged * 100) if judged else 0.0

print(f"\n{'='*65}")
print(f"✅  Done!  Results saved to '{output_path}'.")
print(f"   Total rows      : {total}")
print(f"   Judged          : {judged}")
print(f"   Correct         : {correct}  ({accuracy:.1f}%)")
print(f"   Incorrect       : {incorrect}")
print(f"   Skipped         : {skipped}")
print(f"   Errors          : {errors}")
print(f"{'='*65}")