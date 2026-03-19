"""
Math Competition Question Solver & Verifier
Uses DeepSeek API with two agents:
  - Agent 1: Solves questions and returns only the answer
  - Agent 2: Independently verifies Agent 1's answer
Results are saved to results.csv with a correctness flag.
"""

import os
import json
import csv
import re
import time
from pathlib import Path
from openai import OpenAI

# ── DeepSeek client ────────────────────────────────────────────────────────────
api_key = os.environ.get("DA")
if not api_key:
    raise EnvironmentError("Environment variable 'Deepseek_api' is not set.")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)

MODEL = "deepseek-chat"

# ── Prompts ────────────────────────────────────────────────────────────────────
SOLVER_SYSTEM = (
    "You are an expert math competition solver. "
    "When given a problem, reason through it carefully and then output ONLY the final answer — "
    "no explanation, no units, no extra words. "
    "Match the format the problem asks for (fraction, decimal, integer, expression, etc.)."
)

VERIFIER_SYSTEM = (
    "You are an expert math competition judge. "
    "You will be given a problem and a proposed answer. "
    "Solve the problem independently, then decide if the proposed answer is correct. "
    "Reply with EXACTLY one word: CORRECT or INCORRECT, "
    "followed by a newline and a very brief justification (one sentence max)."
)


# ── API helpers ─────────────────────────────────────────────────────────────────
def call_solver(question: str) -> str:
    """Agent 1 — solve and return the bare answer."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SOLVER_SYSTEM},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
        max_tokens=256,
    )
    return response.choices[0].message.content.strip()


def call_verifier(question: str, proposed_answer: str) -> tuple[str, str]:
    """Agent 2 — verify Agent 1's answer.

    Returns (verdict, justification) where verdict is 'CORRECT' or 'INCORRECT'.
    """
    prompt = (
        f"Problem:\n{question}\n\n"
        f"Proposed answer: {proposed_answer}"
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": VERIFIER_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=256,
    )
    raw = response.choices[0].message.content.strip()
    lines = raw.split("\n", 1)
    verdict = lines[0].strip().upper()
    justification = lines[1].strip() if len(lines) > 1 else ""
    # Normalise — sometimes the model adds punctuation
    if verdict.startswith("CORRECT"):
        verdict = "CORRECT"
    elif verdict.startswith("INCORRECT"):
        verdict = "INCORRECT"
    else:
        verdict = "UNKNOWN"
    return verdict, justification


# ── Answer comparison ──────────────────────────────────────────────────────────
def normalise(text: str) -> str:
    """Lowercase, strip whitespace and common punctuation for loose comparison."""
    text = text.lower().strip()
    text = re.sub(r"[^\w./$√π^-]", "", text)   # keep alphanumeric + math chars
    return text


def answers_match(agent_answer: str, true_answer: str) -> bool:
    """Return True if the answers look equivalent after normalisation."""
    return normalise(agent_answer) == normalise(true_answer)


# ── JSON loader ────────────────────────────────────────────────────────────────
def load_year(filepath: Path) -> list[dict]:
    """Parse one year's JSON file into a flat list of question records."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    year = filepath.stem          # e.g. "2013"
    records = []

    for section_name, section in data.items():
        if not isinstance(section, dict):
            continue
        for q_key, q_obj in section.items():
            if not isinstance(q_obj, dict):
                continue
            question = q_obj.get("question", "").strip()
            true_answer = q_obj.get("answer", "").strip()
            if not question:
                continue
            records.append({
                "year": year,
                "section": section_name,
                "question_id": q_key,
                "question": question,
                "true_answer": true_answer,
            })

    return records


# ── Main pipeline ──────────────────────────────────────────────────────────────
def main():
    datafile_dir = Path("datafile")
    if not datafile_dir.exists():
        raise FileNotFoundError(f"Directory '{datafile_dir}' not found. "
                                "Run this script from the folder that contains 'datafile/'.")

    json_files = sorted(datafile_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError("No JSON files found inside 'datafile/'.")

    output_csv = Path("results.csv")
    fieldnames = [
        "year",
        "section",
        "question_id",
        "question",
        "true_answer",
        "agent1_answer",
        "agent2_verdict",
        "agent2_justification",
        "correct_flag",   # 1 = agent1 correct, 0 = agent1 wrong (needs review)
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for filepath in json_files:
            print(f"\n{'='*60}")
            print(f"Processing {filepath.name} …")
            print(f"{'='*60}")

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
                print(f"  Q: {question[:80]}{'…' if len(question)>80 else ''}")

                # ── Agent 1: solve ──────────────────────────────────────────
                try:
                    agent1_answer = call_solver(question)
                except Exception as exc:
                    print(f"  ✗ Agent 1 error: {exc}")
                    agent1_answer = "ERROR"

                print(f"  Agent 1 answer : {agent1_answer}")
                print(f"  True answer    : {true_answer}")

                # ── Agent 2: verify ─────────────────────────────────────────
                try:
                    verdict, justification = call_verifier(question, agent1_answer)
                except Exception as exc:
                    print(f"  ✗ Agent 2 error: {exc}")
                    verdict, justification = "UNKNOWN", str(exc)

                print(f"  Agent 2 verdict: {verdict} — {justification}")

                # ── Correctness flag ────────────────────────────────────────
                # Primary signal: direct string comparison (normalised).
                # Secondary signal: agent 2's verdict (used if primary is close call).
                direct_match = answers_match(agent1_answer, true_answer)
                verifier_says_correct = verdict == "CORRECT"

                # Flag is 1 only if BOTH the direct comparison AND verifier agree.
                # This makes 0-flagged rows maximally useful for manual review.
                if direct_match and verifier_says_correct:
                    correct_flag = 1
                elif not direct_match and not verifier_says_correct:
                    correct_flag = 0
                else:
                    # Disagreement — conservative: mark for review
                    correct_flag = 0

                icon = "✓" if correct_flag else "✗ → needs review"
                print(f"  Flag           : {correct_flag}  {icon}")

                writer.writerow({
                    "year":                 year,
                    "section":              section,
                    "question_id":          q_id,
                    "question":             question,
                    "true_answer":          true_answer,
                    "agent1_answer":        agent1_answer,
                    "agent2_verdict":       verdict,
                    "agent2_justification": justification,
                    "correct_flag":         correct_flag,
                })
                csvfile.flush()   # write incrementally so progress is saved

                # Polite rate-limit buffer
                time.sleep(0.5)

    print(f"\n✅  Done!  Results saved to '{output_csv}'.")
    print("    Rows with correct_flag=0 are flagged for your manual review.")


if __name__ == "__main__":
    main()