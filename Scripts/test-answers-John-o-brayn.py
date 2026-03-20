"""
Math Competition Question Solver & Verifier
Uses DeepSeek Reasoner API with two agents:
  - Agent 1 (deepseek-reasoner): Solves the question, returns only the answer
  - Agent 2 (deepseek-reasoner): Given the true answer AND Agent 1's answer,
                                  decides if they are mathematically equivalent

Flag logic:
   1  → Agent 2 says CORRECT   (answers are equivalent)
   0  → Agent 2 says INCORRECT (Agent 1 is wrong — needs your review)
  -1  → Question was skipped   (placeholder / missing content in JSON)

CSV columns include agent1_reasoning so you can trace Agent 1's thinking.

Dependencies:
    pip install openai python-dotenv

.env file (place in the same folder as this script):
    Deepseek_api=your_key_here
"""

import os
import re
import json
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

SOLVER_MODEL   = "deepseek-reasoner"
VERIFIER_MODEL = "deepseek-chat"

# ── Retry config ───────────────────────────────────────────────────────────────
MAX_RETRIES    = 3
RETRY_DELAY    = 5.0   # seconds between retries
REQUEST_DELAY  = 0.5   # seconds between normal requests

# ── Placeholder patterns — questions with these are skipped ───────────────────
PLACEHOLDER_PATTERNS = [
    re.compile(r"not visible", re.IGNORECASE),
    re.compile(r"content not", re.IGNORECASE),
    re.compile(r"see (figure|diagram|image|picture|graph)", re.IGNORECASE),
    re.compile(r"\[image\]", re.IGNORECASE),
    re.compile(r"\[figure\]", re.IGNORECASE),
    re.compile(r"^\s*question\s+\d+\s*$", re.IGNORECASE),      # bare "Question 11"
    re.compile(r"^\s*question\s+\d+\s*\(", re.IGNORECASE),     # "Question 11 (..."
    re.compile(r"provide the problem", re.IGNORECASE),
]

def is_placeholder(text: str) -> bool:
    """Return True if the question text is a placeholder or missing content."""
    if not text or len(text.strip()) < 10:
        return True
    for pat in PLACEHOLDER_PATTERNS:
        if pat.search(text):
            return True
    return False


# ── Prompts ────────────────────────────────────────────────────────────────────
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
    "Account for equivalent forms (e.g. 0.5 = 1/2, 36√2 = 36*sqrt(2), -3 = -(3)). "
    "You MUST respond with EXACTLY one of these two words on the very first line of your "
    "response: CORRECT  or  INCORRECT  (all caps, nothing else on that line). "
    "On the second line write one short sentence explaining why. "
    "Do not add any preamble, greeting, or extra text before the verdict word."
)


# ── Agent wrappers ─────────────────────────────────────────────────────────────
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
    msg = response.choices[0].message
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
        max_tokens=1024,
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
      1. Check the first non-empty line for the verdict word.
      2. If that fails, scan every line of the response.
      3. If still not found, search for the word anywhere in the full text.
    """
    if not raw:
        return "UNKNOWN", "(empty response)"

    lines = [l.strip() for l in raw.splitlines() if l.strip()]

    # Pass 1: look for a line that IS the verdict (possibly with punctuation)
    for line in lines:
        clean = re.sub(r"[^A-Za-z]", "", line).upper()
        if clean == "CORRECT":
            note = lines[1] if len(lines) > 1 else ""
            return "CORRECT", note
        if clean == "INCORRECT":
            note = lines[1] if len(lines) > 1 else ""
            return "INCORRECT", note

    # Pass 2: look for the verdict word anywhere inside a line
    for i, line in enumerate(lines):
        upper = line.upper()
        if "INCORRECT" in upper:
            note = lines[i + 1] if i + 1 < len(lines) else ""
            return "INCORRECT", note
        if "CORRECT" in upper:
            note = lines[i + 1] if i + 1 < len(lines) else ""
            return "CORRECT", note

    # Give up
    preview = raw[:120].replace("\n", " ")
    return "UNKNOWN", f"(unparseable response: {preview})"


def call_verifier(question: str, true_answer: str, agent1_answer: str) -> tuple[str, str]:
    """Agent 2 — check if Agent 1's answer is equivalent to the true answer."""
    raw = call_with_retry(
        _verifier_request, question, true_answer, agent1_answer,
        label="Agent 2 (verifier)"
    )
    return parse_verdict(raw)


# ── JSON loader ────────────────────────────────────────────────────────────────
def load_year(filepath: Path) -> list[dict]:
    """
    Parse one JSON file.  Expected structure (flexible):

        {
          "section_name": {
            "question_1": { "question": "...", "answer": "..." },
            ...
          },
          ...
        }

    Returns a list of record dicts, one per question.
    Skips entries where the question text is missing or a placeholder.
    """
    with open(filepath, encoding="utf-8") as f:
        raw = f.read()

    if not raw.strip():
        raise ValueError("File is empty.")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON parse error: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object, got {type(data).__name__}.")

    year    = filepath.stem
    records = []
    skipped = []

    for section_name, section in data.items():
        if not isinstance(section, dict):
            print(f"  ⚠  Skipping non-dict section '{section_name}'")
            continue

        for q_key, q_obj in section.items():
            if not isinstance(q_obj, dict):
                print(f"  ⚠  Skipping non-dict entry '{section_name}/{q_key}'")
                continue

            question    = q_obj.get("question", "").strip()
            true_answer = str(q_obj.get("answer", "")).strip()

            if is_placeholder(question):
                skipped.append(f"{section_name}/{q_key}")
                records.append({
                    "year":        year,
                    "section":     section_name,
                    "question_id": q_key,
                    "question":    question,
                    "true_answer": true_answer,
                    "_skip":       True,
                    "_skip_reason": "placeholder or missing question text",
                })
                continue

            records.append({
                "year":        year,
                "section":     section_name,
                "question_id": q_key,
                "question":    question,
                "true_answer": true_answer,
                "_skip":       False,
            })

    if skipped:
        print(f"  ℹ  Skipped {len(skipped)} placeholder question(s): {', '.join(skipped)}")

    return records


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    datafile_dir = Path("datafile")
    if not datafile_dir.exists():
        raise FileNotFoundError(
            f"Directory '{datafile_dir}' not found. "
            "Run this script from the folder that contains 'datafile/'."
        )

    json_files = sorted(datafile_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError("No JSON files found inside 'datafile/'.")

    print(f"Found {len(json_files)} JSON file(s): {[f.name for f in json_files]}")

    output_csv = Path("results.csv")
    fieldnames = [
        "year", "section", "question_id", "question",
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
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for filepath in json_files:
            print(f"\n{'='*65}")
            print(f"  Processing {filepath.name}")
            print(f"{'='*65}")

            try:
                records = load_year(filepath)
            except Exception as exc:
                print(f"  ✗  Could not parse {filepath.name}: {exc}")
                continue

            for rec in records:
                year        = rec["year"]
                section     = rec["section"]
                q_id        = rec["question_id"]
                question    = rec["question"]
                true_answer = rec["true_answer"]

                print(f"\n  [{year}] {section} / {q_id}")

                # ── Skip placeholder questions ──────────────────────────────
                if rec.get("_skip"):
                    reason = rec.get("_skip_reason", "unknown")
                    print(f"  ⏭  SKIPPED — {reason}")
                    writer.writerow({
                        "year":                 year,
                        "section":              section,
                        "question_id":          q_id,
                        "question":             question,
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
                q_preview = question[:90] + ("…" if len(question) > 90 else "")
                print(f"  Q : {q_preview}")

                # ── Agent 1: solve ──────────────────────────────────────────
                try:
                    agent1_answer, agent1_reasoning = call_solver(question)
                except Exception as exc:
                    print(f"  ✗  Agent 1 failed after {MAX_RETRIES} retries: {exc}")
                    agent1_answer, agent1_reasoning = "ERROR", str(exc)

                print(f"  A1: {agent1_answer}")
                print(f"  ✓ : {true_answer}")
                if agent1_reasoning:
                    preview = agent1_reasoning[:120].replace("\n", " ")
                    print(f"  R : {preview}…")

                # ── Agent 2: verify ─────────────────────────────────────────
                try:
                    verdict, justification = call_verifier(
                        question, true_answer, agent1_answer
                    )
                except Exception as exc:
                    print(f"  ✗  Agent 2 failed after {MAX_RETRIES} retries: {exc}")
                    verdict, justification = "UNKNOWN", str(exc)

                print(f"  A2: {verdict} — {justification}")

                # ── Flag ────────────────────────────────────────────────────
                if verdict == "CORRECT":
                    correct_flag = 1
                    correct_q += 1
                    icon = "✓  correct"
                elif verdict == "INCORRECT":
                    correct_flag = 0
                    icon = "✗  needs review"
                else:
                    correct_flag = 0
                    icon = "?  unknown verdict — needs review"

                print(f"  Flag: {correct_flag}  {icon}")

                writer.writerow({
                    "year":                 year,
                    "section":              section,
                    "question_id":          q_id,
                    "question":             question,
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
        print(f"   Wrong   (flag=0)    : {total_q - correct_q}")
    print(f"   Skipped (flag=-1)   : {skipped_q}  (placeholder/missing content)")
    print(f"{'='*65}")
    print("   correct_flag= 1 → Agent 1 got it right")
    print("   correct_flag= 0 → Agent 1 wrong or verdict unclear, review the row")
    print("   correct_flag=-1 → Question had no content in the JSON, skipped")


if __name__ == "__main__":
    main()