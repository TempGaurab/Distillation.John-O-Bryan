"""
Re-Distiller: reads results.csv, runs distilled_reasoning through the
Distiller agent a second time, and saves to results2.csv.

All rows are preserved (including skipped ones with flag=-1).
Only rows with non-empty distilled_reasoning are re-distilled.

Dependencies:
    pip install openai python-dotenv

.env file (same folder as this script):
    Deepseek_api=your_key_here
"""

import os
import re
import csv
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# ── Load env & build client ────────────────────────────────────────────────────
load_dotenv()

API_KEY = os.environ.get("Deepseek_api")
if not API_KEY:
    raise EnvironmentError(
        "Could not find 'Deepseek_api' in environment or .env file.\n"
        "Create a .env file in the same folder with:\n"
        "    Deepseek_api=your_key_here"
    )

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com",
)

DISTILLER_MODEL = "deepseek-chat"

# ── Retry config ───────────────────────────────────────────────────────────────
MAX_RETRIES   = 3
RETRY_DELAY   = 5.0
REQUEST_DELAY = 0.5

# ── Distiller prompt (same as original) ───────────────────────────────────────
DISTILLER_SYSTEM = (
    "You are preparing training data for a small math model. "
    "You will receive a competition math problem, its correct answer, and a chain-of-thought "
    "that reached that answer. "
    "Your task: rewrite the reasoning as an even SHORTER, MORE PRECISE chain-of-thought "
    "that a smaller model can learn from. "
    "Rules:\n"
    "  • Keep only the essential steps — cut all dead-ends, restarts, and repetition.\n"
    "  • Each step must be one clear sentence or equation.\n"
    "  • End with: 'Answer: <answer>' on its own line.\n"
    "  • Target length: 3–10 steps. Never exceed 15.\n"
    "  • Do NOT add any preamble, greeting, or meta-commentary — output the steps only."
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


# ── Distiller ──────────────────────────────────────────────────────────────────
def _distiller_request(question: str, answer: str, reasoning: str):
    prompt = (
        f"Problem:\n{question}\n\n"
        f"Correct answer: {answer}\n\n"
        f"Chain-of-thought to compress further:\n{reasoning}"
    )
    response = client.chat.completions.create(
        model=DISTILLER_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": DISTILLER_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
    )
    return (response.choices[0].message.content or "").strip()

def call_distiller(question: str, answer: str, reasoning: str) -> str:
    return call_with_retry(
        _distiller_request, question, answer, reasoning,
        label="Distiller (pass 2)"
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    input_csv  = Path("results.csv")
    output_csv = Path("results2.csv")

    if not input_csv.exists():
        raise FileNotFoundError(f"'{input_csv}' not found. Run the original pipeline first.")

    with open(input_csv, newline="", encoding="utf-8") as f:
        reader     = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows       = list(reader)

    if not rows:
        raise ValueError("results.csv is empty.")

    # Replace distilled_reasoning with distilled_reasoning2 in output columns
    out_fieldnames = []
    for col in fieldnames:
        if col == "distilled_reasoning":
            out_fieldnames.append("distilled_reasoning")   # keep original
            out_fieldnames.append("distilled_reasoning2")  # new column
        else:
            out_fieldnames.append(col)

    print(f"Loaded {len(rows)} rows from '{input_csv}'.")
    print(f"Output will be written to '{output_csv}'.\n")

    total = redone = skipped = errors = 0

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=out_fieldnames)
        writer.writeheader()

        for i, row in enumerate(rows, 1):
            year    = row.get("year", "")
            section = row.get("section", "")
            q_id    = row.get("question_id", "")
            flag    = row.get("correct_flag", "")

            print(f"[{i}/{len(rows)}] {year} / {section} / {q_id}  (flag={flag})")

            question            = row.get("question", "").strip()
            true_answer         = row.get("true_answer", "").strip()
            distilled_reasoning = row.get("distilled_reasoning", "").strip()

            distilled_reasoning2 = ""

            # Only re-distill rows that have content and weren't skipped
            skip_conditions = (
                not distilled_reasoning
                or distilled_reasoning.startswith("DISTILLATION_ERROR")
                or flag == "-1"
                or row.get("agent1_answer", "") in ("SKIPPED", "ERROR")
            )

            if skip_conditions:
                reason = (
                    "flag=-1 (placeholder)"  if flag == "-1"
                    else "no distilled_reasoning to compress"
                )
                print(f"  ⏭  Skipping — {reason}")
                skipped += 1
            else:
                total += 1
                try:
                    distilled_reasoning2 = call_distiller(
                        question, true_answer, distilled_reasoning
                    )
                    before = len(distilled_reasoning)
                    after  = len(distilled_reasoning2)
                    steps  = len([l for l in distilled_reasoning2.splitlines() if l.strip()])
                    print(f"  📝 Re-distilled to {steps} steps "
                          f"({before} chars → {after} chars)")
                    redone += 1
                except Exception as exc:
                    print(f"  ✗  Distiller failed after {MAX_RETRIES} retries: {exc}")
                    distilled_reasoning2 = f"DISTILLATION_ERROR: {exc}"
                    errors += 1

                time.sleep(REQUEST_DELAY)

            out_row = dict(row)
            out_row["distilled_reasoning2"] = distilled_reasoning2
            writer.writerow(out_row)
            csvfile.flush()

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"✅  Done!  Results saved to '{output_csv}'.")
    print(f"   Rows re-distilled : {redone}")
    print(f"   Skipped           : {skipped}  (placeholder / no content)")
    print(f"   Errors            : {errors}")
    print(f"{'='*65}")
    print("   New column: distilled_reasoning2 = second-pass compact CoT")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()