"""
Qwen Tester: reads results.csv, runs each question through Qwen 2.5 7B
(Hugging Face Inference API), then has DeepSeek evaluate correctness.
Saves output to results_qwen.csv.

All original columns are preserved. New columns added:
  - qwen_answer       : raw answer from Qwen 2.5 7B
  - qwen_correct_flag : 1 if correct, 0 if wrong, -1 if skipped/error
  - qwen_eval_notes   : DeepSeek's brief evaluation note

Dependencies:
    pip install openai huggingface_hub python-dotenv

.env file (same folder as this script):
    Deepseek_api=your_deepseek_key_here
    HF_api=your_huggingface_token_here
"""

import os
import re
import csv
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from huggingface_hub import InferenceClient

# ── Load env & build clients ───────────────────────────────────────────────────
load_dotenv()

DEEPSEEK_KEY = os.environ.get("Deepseek_api")
if not DEEPSEEK_KEY:
    raise EnvironmentError(
        "Could not find 'Deepseek_api' in environment or .env file.\n"
        "Add to your .env:\n    Deepseek_api=your_key_here"
    )

HF_KEY = os.environ.get("HF_api")
if not HF_KEY:
    raise EnvironmentError(
        "Could not find 'HF_api' in environment or .env file.\n"
        "Add to your .env:\n    HF_api=your_huggingface_token_here"
    )

deepseek_client = OpenAI(
    api_key=DEEPSEEK_KEY,
    base_url="https://api.deepseek.com",
)

hf_client = InferenceClient(
    provider="hf-inference",
    api_key=HF_KEY,
)

QWEN_MODEL     = "Qwen/Qwen2.5-7B-Instruct"
DEEPSEEK_MODEL = "deepseek-chat"

# ── Retry config ───────────────────────────────────────────────────────────────
MAX_RETRIES   = 3
RETRY_DELAY   = 5.0
REQUEST_DELAY = 0.5

# ── Prompts ────────────────────────────────────────────────────────────────────
QWEN_SYSTEM = (
    "You are a precise math competition solver. "
    "Work through the problem step by step, then state your final answer clearly. "
    "End your response with: 'Answer: <your answer>' on its own line."
)

EVALUATOR_SYSTEM = (
    "You are a strict math competition answer evaluator. "
    "You will be given a problem, the correct answer, and a model's response. "
    "Determine if the model's final answer matches the correct answer (mathematically equivalent). "
    "Reply in this exact format:\n"
    "CORRECT: <yes or no>\n"
    "NOTE: <one short sentence explaining your judgment>"
)


# ── Retry wrapper ──────────────────────────────────────────────────────────────
def call_with_retry(fn, *args, label="API call"):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args)
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            print(f"  ⚠  {label} failed (attempt {attempt}/{MAX_RETRIES}): {exc}")
            time.sleep(RETRY_DELAY)


# ── Qwen agent ────────────────────────────────────────────────────────────────
def _qwen_request(question: str) -> str:
    messages = [
        {"role": "system", "content": QWEN_SYSTEM},
        {"role": "user",   "content": question},
    ]
    completion = hf_client.chat.completions.create(
        model=QWEN_MODEL,
        messages=messages,
        max_tokens=1024,
    )
    return (completion.choices[0].message.content or "").strip()

def call_qwen(question: str) -> str:
    return call_with_retry(_qwen_request, question, label="Qwen 2.5 7B")


# ── DeepSeek evaluator ─────────────────────────────────────────────────────────
def _evaluator_request(question: str, true_answer: str, model_response: str) -> str:
    prompt = (
        f"Problem:\n{question}\n\n"
        f"Correct answer: {true_answer}\n\n"
        f"Model's response:\n{model_response}"
    )
    response = deepseek_client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        max_tokens=256,
        messages=[
            {"role": "system", "content": EVALUATOR_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
    )
    return (response.choices[0].message.content or "").strip()

def call_evaluator(question: str, true_answer: str, model_response: str) -> tuple[int, str]:
    """Returns (flag, note) where flag is 1=correct, 0=wrong, -1=parse error."""
    raw = call_with_retry(
        _evaluator_request, question, true_answer, model_response,
        label="DeepSeek Evaluator"
    )
    # Parse CORRECT: yes/no
    match = re.search(r"CORRECT:\s*(yes|no)", raw, re.IGNORECASE)
    note_match = re.search(r"NOTE:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)

    flag = -1
    if match:
        flag = 1 if match.group(1).lower() == "yes" else 0

    note = note_match.group(1).strip() if note_match else raw[:200]
    return flag, note


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    input_csv  = Path("results.csv")
    output_csv = Path("results_qwen.csv")

    if not input_csv.exists():
        raise FileNotFoundError(f"'{input_csv}' not found.")

    with open(input_csv, newline="", encoding="utf-8") as f:
        reader     = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows       = list(reader)

    if not rows:
        raise ValueError("results.csv is empty.")

    new_cols = ["qwen_answer", "qwen_correct_flag", "qwen_eval_notes"]
    out_fieldnames = fieldnames + new_cols

    print(f"Loaded {len(rows)} rows from '{input_csv}'.")
    print(f"Model  : {QWEN_MODEL}")
    print(f"Judge  : {DEEPSEEK_MODEL}")
    print(f"Output : '{output_csv}'\n")

    total = correct = wrong = skipped = errors = 0

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=out_fieldnames)
        writer.writeheader()

        for i, row in enumerate(rows, 1):
            year    = row.get("year", "")
            section = row.get("section", "")
            q_id    = row.get("question_id", "")
            flag    = row.get("correct_flag", "")

            print(f"[{i}/{len(rows)}] {year} / {section} / {q_id}  (orig_flag={flag})")

            question    = row.get("question", "").strip()
            true_answer = row.get("true_answer", "").strip()

            qwen_answer      = ""
            qwen_flag        = -1
            qwen_eval_notes  = ""

            # Skip placeholder rows (no question content)
            if not question or flag == "-1":
                print(f"  ⏭  Skipping — placeholder row")
                skipped += 1
            else:
                total += 1
                # Step 1: Get Qwen's answer
                try:
                    qwen_answer = call_qwen(question)
                    # Extract just the answer line for display
                    ans_preview = qwen_answer.splitlines()[-1][:80] if qwen_answer else ""
                    print(f"  🤖 Qwen: {ans_preview}")
                except Exception as exc:
                    print(f"  ✗  Qwen failed after {MAX_RETRIES} retries: {exc}")
                    qwen_answer     = f"ERROR: {exc}"
                    qwen_flag       = -1
                    qwen_eval_notes = "Qwen API error"
                    errors += 1

                    out_row = dict(row)
                    out_row["qwen_answer"]       = qwen_answer
                    out_row["qwen_correct_flag"] = qwen_flag
                    out_row["qwen_eval_notes"]   = qwen_eval_notes
                    writer.writerow(out_row)
                    csvfile.flush()
                    time.sleep(REQUEST_DELAY)
                    continue

                time.sleep(REQUEST_DELAY)

                # Step 2: DeepSeek evaluates
                try:
                    qwen_flag, qwen_eval_notes = call_evaluator(
                        question, true_answer, qwen_answer
                    )
                    verdict = "✓ CORRECT" if qwen_flag == 1 else "✗ WRONG"
                    print(f"  {verdict}  — {qwen_eval_notes[:80]}")
                    if qwen_flag == 1:
                        correct += 1
                    else:
                        wrong += 1
                except Exception as exc:
                    print(f"  ✗  Evaluator failed after {MAX_RETRIES} retries: {exc}")
                    qwen_flag       = -1
                    qwen_eval_notes = f"EVAL_ERROR: {exc}"
                    errors += 1

                time.sleep(REQUEST_DELAY)

            out_row = dict(row)
            out_row["qwen_answer"]       = qwen_answer
            out_row["qwen_correct_flag"] = qwen_flag
            out_row["qwen_eval_notes"]   = qwen_eval_notes
            writer.writerow(out_row)
            csvfile.flush()

    # ── Summary ────────────────────────────────────────────────────────────────
    attempted = correct + wrong
    accuracy  = (correct / attempted * 100) if attempted else 0.0

    print(f"\n{'='*65}")
    print(f"✅  Done!  Results saved to '{output_csv}'.")
    print(f"{'='*65}")
    print(f"   Model tested      : {QWEN_MODEL}")
    print(f"   Rows attempted    : {total}")
    print(f"   Correct           : {correct}")
    print(f"   Wrong             : {wrong}")
    print(f"   Accuracy          : {accuracy:.1f}%")
    print(f"   Skipped           : {skipped}  (placeholder rows)")
    print(f"   Errors            : {errors}")
    print(f"{'='*65}")
    print("   New columns: qwen_answer | qwen_correct_flag | qwen_eval_notes")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()