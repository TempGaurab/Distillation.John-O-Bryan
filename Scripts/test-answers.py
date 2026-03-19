"""
Math Competition Question Solver & Verifier
Uses DeepSeek Reasoner API with two agents:
  - Agent 1 (deepseek-reasoner): Solves the question, returns only the answer
  - Agent 2 (deepseek-reasoner): Given the true answer AND Agent 1's answer,
                                  decides if they are mathematically equivalent

Flag logic:
   1  → Agent 2 says CORRECT   (answers are equivalent)
   0  → Agent 2 says INCORRECT (Agent 1 is wrong — needs your review)

CSV columns include agent1_reasoning so you can trace Agent 1's thinking.

Dependencies:
    pip install openai python-dotenv

.env file (place in the same folder as this script):
    Deepseek_api=your_key_here
"""

import os
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
VERIFIER_MODEL = "deepseek-reasoner"

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
    "You are a math answer checker. "
    "You will be given a problem, the correct answer, and a student's answer. "
    "Your only job is to decide if the student's answer is mathematically equivalent "
    "to the correct answer. Account for equivalent forms (e.g. 0.5 = 1/2, 36√2 = 36*sqrt(2)). "
    "Respond with EXACTLY one word on the first line: CORRECT or INCORRECT. "
    "On the second line, write one short sentence explaining why."
)


# ── Agent wrappers ─────────────────────────────────────────────────────────────
def call_solver(question: str) -> tuple[str, str]:
    """Agent 1 — solve the question.

    Returns (answer, reasoning) where reasoning is the chain-of-thought.
    """
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


def call_verifier(question: str, true_answer: str, agent1_answer: str) -> tuple[str, str]:
    """Agent 2 — given the true answer, check if Agent 1's answer is equivalent."""
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
    raw   = (response.choices[0].message.content or "").strip()
    lines = raw.split("\n", 1)

    verdict = lines[0].strip().upper()
    note    = lines[1].strip() if len(lines) > 1 else ""

    if verdict.startswith("CORRECT"):
        verdict = "CORRECT"
    elif verdict.startswith("INCORRECT"):
        verdict = "INCORRECT"
    else:
        verdict = "UNKNOWN"

    return verdict, note


# ── JSON loader ────────────────────────────────────────────────────────────────
def load_year(filepath: Path) -> list[dict]:
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    year, records = filepath.stem, []

    for section_name, section in data.items():
        if not isinstance(section, dict):
            continue
        for q_key, q_obj in section.items():
            if not isinstance(q_obj, dict):
                continue
            question    = q_obj.get("question", "").strip()
            true_answer = q_obj.get("answer",   "").strip()
            if not question:
                continue
            records.append({
                "year":        year,
                "section":     section_name,
                "question_id": q_key,
                "question":    question,
                "true_answer": true_answer,
            })

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

    output_csv = Path("results.csv")
    fieldnames = [
        "year", "section", "question_id", "question",
        "true_answer",
        "agent1_answer",
        "agent1_reasoning",        # ← chain-of-thought from deepseek-reasoner
        "agent2_verdict",
        "agent2_justification",
        "correct_flag",
        # flag: 1 = correct, 0 = wrong (needs review)
    ]

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
                print(f"  ⚠  Could not parse {filepath.name}: {exc}")
                continue

            for rec in records:
                year        = rec["year"]
                section     = rec["section"]
                q_id        = rec["question_id"]
                question    = rec["question"]
                true_answer = rec["true_answer"]

                print(f"\n  [{year}] {section} / {q_id}")
                print(f"  Q : {question[:90]}{'…' if len(question) > 90 else ''}")

                # ── Agent 1: solve ──────────────────────────────────────────
                try:
                    agent1_answer, agent1_reasoning = call_solver(question)
                except Exception as exc:
                    print(f"  ✗  Agent 1 error: {exc}")
                    agent1_answer, agent1_reasoning = "ERROR", str(exc)

                print(f"  A1: {agent1_answer}")
                print(f"  ✓ : {true_answer}")
                if agent1_reasoning:
                    preview = agent1_reasoning[:120].replace("\n", " ")
                    print(f"  R : {preview}…")

                # ── Agent 2: check against known answer ─────────────────────
                try:
                    verdict, justification = call_verifier(question, true_answer, agent1_answer)
                except Exception as exc:
                    print(f"  ✗  Agent 2 error: {exc}")
                    verdict, justification = "UNKNOWN", str(exc)

                print(f"  A2: {verdict} — {justification}")

                # ── Flag ────────────────────────────────────────────────────
                correct_flag = 1 if verdict == "CORRECT" else 0
                icon = "✓  correct" if correct_flag else "✗  needs review"
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

                time.sleep(0.3)

    print(f"\n✅  Done! Results saved to '{output_csv}'.")
    print("     correct_flag=1 → Agent 1 got it right")
    print("     correct_flag=0 → Agent 1 wrong, review the row")


if __name__ == "__main__":
    main()