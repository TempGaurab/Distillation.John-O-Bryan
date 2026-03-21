"""
Math Competition Question Solver & Verifier — MATH-500 Edition
Uses DeepSeek API with two agents:
  - Agent 1 (deepseek-reasoner): Solves the question, returns only the answer
  - Agent 2 (deepseek-chat):     Given the true answer AND Agent 1's answer,
                                  decides if they are mathematically equivalent

Flag logic:
   1  → Agent 2 says CORRECT   (answers are equivalent)
   0  → Agent 2 says INCORRECT (Agent 1 is wrong — needs your review)
  -1  → Question was skipped   (placeholder / missing content)

Input:  a CSV file with columns:
          problem, solution, answer, subject, level, unique_id
Output: results.csv with all original columns plus grading columns

Dependencies:
    pip install openai python-dotenv

.env file (place in the same folder as this script):
    Deepseek_api=your_key_here
"""

import os
import re
import csv
import time
import argparse
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

SOLVER_MODEL   = "deepseek-reasoner"
VERIFIER_MODEL = "deepseek-chat"

# ── Retry / rate-limit config ──────────────────────────────────────────────────
MAX_RETRIES   = 3
RETRY_DELAY   = 5.0   # seconds between retries
REQUEST_DELAY = 0.5   # seconds between normal requests

# ── MATH-500 CSV column names ──────────────────────────────────────────────────
COL_PROBLEM    = "problem"
COL_SOLUTION   = "solution"
COL_ANSWER     = "answer"
COL_SUBJECT    = "subject"
COL_LEVEL      = "level"
COL_UNIQUE_ID  = "unique_id"

REQUIRED_COLS  = {COL_PROBLEM, COL_ANSWER}

# ── Placeholder detection ──────────────────────────────────────────────────────
PLACEHOLDER_PATTERNS = [
    re.compile(r"not visible",             re.IGNORECASE),
    re.compile(r"content not",             re.IGNORECASE),
    re.compile(r"see (figure|diagram|image|picture|graph)", re.IGNORECASE),
    re.compile(r"\[image\]",               re.IGNORECASE),
    re.compile(r"\[figure\]",              re.IGNORECASE),
    re.compile(r"^\s*question\s+\d+\s*$",  re.IGNORECASE),
]

def is_placeholder(text: str) -> bool:
    """Return True if the question text is empty or a placeholder."""
    if not text or len(text.strip()) < 10:
        return True
    for pat in PLACEHOLDER_PATTERNS:
        if pat.search(text):
            return True
    return False


# ── System prompts ─────────────────────────────────────────────────────────────
SOLVER_SYSTEM = (
    "You are an expert math competition solver. "
    "Work through the problem carefully using step-by-step reasoning. "
    "Your final response must contain ONLY the answer — no explanation, no LaTeX, "
    "no \\boxed{}, no units, no extra words. "
    "Match exactly the format the problem specifies "
    "(e.g. integer, reduced fraction like 2/3, decimal, expression like 36√2, etc.)."
)

VERIFIER_SYSTEM = (
    "You are a strict math answer checker. "
    "You will be given a problem, the correct answer, and a student's answer. "
    "Decide if the student's answer is mathematically equivalent to the correct answer. "
    "Account for equivalent forms (e.g. 0.5 = 1/2, 36√2 = 36*sqrt(2), -3 = -(3), "
    "\\frac{1}{2} = 0.5, boxed expressions equal their contents). "
    "You MUST respond with EXACTLY one of these two words on the very first line of your "
    "response: CORRECT  or  INCORRECT  (all caps, nothing else on that line). "
    "On the second line write one short sentence explaining why. "
    "Do not add any preamble, greeting, or extra text before the verdict word."
)


# ── API wrappers ───────────────────────────────────────────────────────────────
def call_with_retry(fn, *args, label="API call"):
    """Call fn(*args), retrying up to MAX_RETRIES times on exception."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args)
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            print(f"  ⚠  {label} failed (attempt {attempt}/{MAX_RETRIES}): {exc}")
            time.sleep(RETRY_DELAY)


def _solver_request(question: str):
    response = client.chat.completions.create(
        model=SOLVER_MODEL,
        max_tokens=8000,
        messages=[
            {"role": "system", "content": SOLVER_SYSTEM},
            {"role": "user",   "content": question},
        ],
    )
    msg       = response.choices[0].message
    answer    = (msg.content or "").strip()
    reasoning = (getattr(msg, "reasoning_content", None) or "").strip()
    return answer, reasoning


def call_solver(question: str) -> tuple[str, str]:
    """Agent 1 — solve the question. Returns (answer, reasoning)."""
    return call_with_retry(_solver_request, question, label="Agent 1 (solver)")


def _verifier_request(question: str, true_answer: str, agent1_answer: str):
    prompt = (
        f"Problem:\n{question}\n\n"
        f"Correct answer: {true_answer}\n"
        f"Student's answer: {agent1_answer}"
    )
    response = client.chat.completions.create(
        model=VERIFIER_MODEL,
        max_tokens=512,
        messages=[
            {"role": "system", "content": VERIFIER_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def parse_verdict(raw: str) -> tuple[str, str]:
    """
    Robustly extract CORRECT / INCORRECT from the verifier's raw response.

    Strategy:
      1. Check each line for the verdict word (exact or embedded).
      2. Fall back to scanning the full text.
    """
    if not raw:
        return "UNKNOWN", "(empty response)"

    lines = [l.strip() for l in raw.splitlines() if l.strip()]

    # Pass 1: line that IS the verdict (ignoring punctuation)
    for i, line in enumerate(lines):
        clean = re.sub(r"[^A-Za-z]", "", line).upper()
        if clean == "CORRECT":
            note = lines[i + 1] if i + 1 < len(lines) else ""
            return "CORRECT", note
        if clean == "INCORRECT":
            note = lines[i + 1] if i + 1 < len(lines) else ""
            return "INCORRECT", note

    # Pass 2: verdict word embedded anywhere in a line
    for i, line in enumerate(lines):
        upper = line.upper()
        if "INCORRECT" in upper:
            note = lines[i + 1] if i + 1 < len(lines) else ""
            return "INCORRECT", note
        if "CORRECT" in upper:
            note = lines[i + 1] if i + 1 < len(lines) else ""
            return "CORRECT", note

    preview = raw[:120].replace("\n", " ")
    return "UNKNOWN", f"(unparseable response: {preview})"


def call_verifier(question: str, true_answer: str, agent1_answer: str) -> tuple[str, str]:
    """Agent 2 — check if Agent 1's answer is equivalent to the true answer."""
    raw = call_with_retry(
        _verifier_request, question, true_answer, agent1_answer,
        label="Agent 2 (verifier)"
    )
    return parse_verdict(raw)


# ── CSV loader ─────────────────────────────────────────────────────────────────
def load_math500_csv(filepath: Path) -> list[dict]:
    """
    Parse the MATH-500 CSV file.

    Expected columns (flexible — extras are kept as-is):
        problem, solution, answer, subject, level, unique_id

    Returns a list of record dicts, each with a '_skip' boolean.
    """
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows   = list(reader)

    if not rows:
        raise ValueError("CSV file is empty or has no data rows.")

    headers = set(rows[0].keys())
    missing = REQUIRED_COLS - headers
    if missing:
        raise ValueError(
            f"CSV is missing required column(s): {missing}. "
            f"Found columns: {sorted(headers)}"
        )

    records  = []
    skipped  = []

    for i, row in enumerate(rows, start=2):          # row 1 = header
        problem     = row.get(COL_PROBLEM,   "").strip()
        true_answer = row.get(COL_ANSWER,    "").strip()
        subject     = row.get(COL_SUBJECT,   "").strip()
        level       = row.get(COL_LEVEL,     "").strip()
        unique_id   = row.get(COL_UNIQUE_ID, f"row_{i}").strip()
        solution    = row.get(COL_SOLUTION,  "").strip()

        skip = is_placeholder(problem)

        if skip:
            skipped.append(unique_id or f"row_{i}")

        records.append({
            "unique_id":   unique_id,
            "subject":     subject,
            "level":       level,
            "problem":     problem,
            "solution":    solution,
            "true_answer": true_answer,
            "_skip":       skip,
            "_skip_reason": "placeholder or missing problem text" if skip else "",
        })

    if skipped:
        print(f"  ℹ  {len(skipped)} placeholder row(s) will be skipped: "
              f"{', '.join(skipped[:10])}{'…' if len(skipped) > 10 else ''}")

    return records


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Solve & grade MATH-500 problems using DeepSeek."
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        default= "math500_output.csv",
        help="Path to the MATH-500 CSV file (default: math500.csv)",
    )
    parser.add_argument(
        "--output",
        default="results-math500.csv",
        help="Path for the output CSV (default: results.csv)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N questions (useful for testing)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Input file not found: '{csv_path}'\n"
            "Pass the path as the first argument, e.g.:\n"
            "    python math500_solver.py path/to/math500.csv"
        )

    print(f"Loading {csv_path} …")
    records = load_math500_csv(csv_path)
    print(f"  {len(records)} rows loaded.")

    if args.limit:
        records = records[: args.limit]
        print(f"  ⚠  --limit {args.limit}: only processing first {len(records)} rows.")

    output_csv = Path(args.output)
    fieldnames = [
        "unique_id",
        "subject",
        "level",
        "problem",
        "solution",             # reference solution from MATH-500
        "true_answer",
        "agent1_answer",
        "agent1_reasoning",
        "agent2_verdict",
        "agent2_justification",
        "correct_flag",
        # 1  = correct
        # 0  = incorrect (needs review)
        # -1 = skipped (placeholder / missing content)
    ]

    total_q = correct_q = skipped_q = 0

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for idx, rec in enumerate(records, start=1):
            unique_id   = rec["unique_id"]
            subject     = rec["subject"]
            level       = rec["level"]
            problem     = rec["problem"]
            solution    = rec["solution"]
            true_answer = rec["true_answer"]

            print(f"\n[{idx}/{len(records)}] {unique_id}  ({subject}, Level {level})")

            # ── Skip placeholder problems ───────────────────────────────────
            if rec["_skip"]:
                reason = rec["_skip_reason"]
                print(f"  ⏭  SKIPPED — {reason}")
                writer.writerow({
                    "unique_id":            unique_id,
                    "subject":              subject,
                    "level":                level,
                    "problem":              problem,
                    "solution":             solution,
                    "true_answer":          true_answer,
                    "agent1_answer":        "SKIPPED",
                    "agent1_reasoning":     "",
                    "agent2_verdict":       "SKIPPED",
                    "agent2_justification": reason,
                    "correct_flag":         -1,
                })
                csvfile.flush()
                skipped_q += 1
                continue

            total_q += 1
            q_preview = problem[:90] + ("…" if len(problem) > 90 else "")
            print(f"  Q : {q_preview}")
            print(f"  ✓ : {true_answer}")

            # ── Agent 1: solve ──────────────────────────────────────────────
            try:
                agent1_answer, agent1_reasoning = call_solver(problem)
            except Exception as exc:
                print(f"  ✗  Agent 1 failed after {MAX_RETRIES} retries: {exc}")
                agent1_answer, agent1_reasoning = "ERROR", str(exc)

            print(f"  A1: {agent1_answer}")
            if agent1_reasoning:
                preview = agent1_reasoning[:120].replace("\n", " ")
                print(f"  R : {preview}…")

            # ── Agent 2: verify ─────────────────────────────────────────────
            try:
                verdict, justification = call_verifier(
                    problem, true_answer, agent1_answer
                )
            except Exception as exc:
                print(f"  ✗  Agent 2 failed after {MAX_RETRIES} retries: {exc}")
                verdict, justification = "UNKNOWN", str(exc)

            print(f"  A2: {verdict} — {justification}")

            # ── Flag ────────────────────────────────────────────────────────
            if verdict == "CORRECT":
                correct_flag = 1
                correct_q   += 1
                icon = "✓  correct"
            elif verdict == "INCORRECT":
                correct_flag = 0
                icon = "✗  needs review"
            else:
                correct_flag = 0
                icon = "?  unknown verdict — needs review"

            print(f"  Flag: {correct_flag}  {icon}")

            writer.writerow({
                "unique_id":            unique_id,
                "subject":              subject,
                "level":                level,
                "problem":              problem,
                "solution":             solution,
                "true_answer":          true_answer,
                "agent1_answer":        agent1_answer,
                "agent1_reasoning":     agent1_reasoning,
                "agent2_verdict":       verdict,
                "agent2_justification": justification,
                "correct_flag":         correct_flag,
            })
            csvfile.flush()

            time.sleep(REQUEST_DELAY)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"✅  Done!  Results saved to '{output_csv}'.")
    print(f"   Questions attempted : {total_q}")
    if total_q:
        pct = 100 * correct_q / total_q
        print(f"   Correct (flag=1)    : {correct_q}  ({pct:.1f}%)")
        print(f"   Wrong   (flag=0)    : {total_q - correct_q}  ({100-pct:.1f}%)")
    print(f"   Skipped (flag=-1)   : {skipped_q}  (placeholder/missing content)")
    print(f"{'='*65}")
    print("   correct_flag= 1 → Agent 1 got it right")
    print("   correct_flag= 0 → Agent 1 wrong or verdict unclear, review the row")
    print("   correct_flag=-1 → Problem had no content, skipped")


if __name__ == "__main__":
    main()