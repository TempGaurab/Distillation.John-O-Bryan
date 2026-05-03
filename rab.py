#!/usr/bin/env python3
"""
reasoning_accuracy_benchmark.py

Tests all 5 reasoning-level LoRA adapters against results.csv.
For each adapter:
  - Generates answers using the base model + adapter (mlx_lm)
  - Uses DeepSeek to judge correctness
  - Saves per-row results to runs/reasoning_N/results.csv

After all 5 are done, prints and saves:
  - Overall accuracy per level
  - Accuracy by question type (section) per level
  - Accuracy by year per level

Usage:
    python reasoning_accuracy_benchmark.py [--resume]

Edit CONFIG below to match your paths.
"""

import os
import re
import csv
import time
import argparse
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv
from openai import OpenAI

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
BASE_MODEL   = "mlx-community/Qwen2.5-7B-Instruct-4bit"
RUNS_DIR     = Path("runs")
INPUT_CSV    = "results.csv"
MAX_TOKENS   = 600

REASONING_LEVELS = [1, 2, 3, 4, 5]

MAX_RETRIES   = 3
RETRY_DELAY   = 5.0
REQUEST_DELAY = 0.5
# ══════════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true",
                    help="Skip rows already written in each level's results.csv")
args = parser.parse_args()

# ── Load DeepSeek client ───────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.environ.get("Deepseek_api")
if not API_KEY:
    raise EnvironmentError("'Deepseek_api' not found in .env")

ds_client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

# ── Load source data ───────────────────────────────────────────────────────────
input_path = Path(INPUT_CSV)
if not input_path.exists():
    raise FileNotFoundError(f"'{INPUT_CSV}' not found.")

with open(input_path, newline="", encoding="utf-8") as f:
    reader     = csv.DictReader(f)
    src_fields = list(reader.fieldnames or [])
    all_rows   = list(reader)

print(f"Loaded {len(all_rows)} rows from '{INPUT_CSV}'.")

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


def extract_boxed(text: str) -> list:
    results, i = [], 0
    while i < len(text):
        idx = text.find(r"\boxed{", i)
        if idx == -1:
            break
        start, depth, j = idx + len(r"\boxed{"), 1, idx + len(r"\boxed{")
        while j < len(text) and depth > 0:
            if text[j] == "{":   depth += 1
            elif text[j] == "}": depth -= 1
            j += 1
        results.append(text[start: j - 1])
        i = j
    return results


def extract_final_answer(text: str) -> str:
    boxes = extract_boxed(text)
    if boxes:
        return boxes[-1].strip()
    answer_lines = re.findall(r"(?i)answer\s*[:\-]\s*(.+)", text)
    if answer_lines:
        return answer_lines[-1].strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


JUDGE_SYSTEM = (
    "You are a strict math answer checker. "
    "Given a competition math problem, the correct answer, and a model's response, "
    "determine whether the model's final answer is mathematically equivalent to the correct answer. "
    "Reply with ONLY one of:\n  CORRECT\n  INCORRECT\nNo explanation."
)


def _judge_request(question, true_answer, model_response):
    resp = ds_client.chat.completions.create(
        model="deepseek-chat",
        max_tokens=16,
        temperature=0.0,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": (
                f"Problem:\n{question}\n\n"
                f"Correct answer: {true_answer}\n\n"
                f"Model response:\n{model_response}"
            )},
        ],
    )
    v = (resp.choices[0].message.content or "").strip().upper()
    if "INCORRECT" in v: return "INCORRECT"
    if "CORRECT"   in v: return "CORRECT"
    return v


def judge_answer(question, true_answer, model_response):
    return call_with_retry(
        _judge_request, question, true_answer, model_response,
        label="DeepSeek judge"
    )


# ── Per-level evaluation ───────────────────────────────────────────────────────

NEW_COLS = ["qwen_response", "qwen_extracted_answer", "deepseek_verdict"]

def evaluate_level(level: int) -> list[dict]:
    """
    Load the base model + LoRA adapter for this level,
    run every question in results.csv through it,
    judge with DeepSeek, and save to runs/reasoning_N/results.csv.
    Returns list of result dicts (one per non-skipped row).
    """
    adapter_dir = RUNS_DIR / f"reasoning_{level}" / "adapters"
    out_csv     = RUNS_DIR / f"reasoning_{level}" / "results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Resume: load already-done question IDs
    done_ids: set = set()
    if args.resume and out_csv.exists():
        with open(out_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done_ids.add((row.get("year",""), row.get("section",""), row.get("question_id","")))
        print(f"  Resuming — {len(done_ids)} rows already done.")

    # Load model with adapter
    print(f"\n  Loading model + adapter from {adapter_dir} …")
    from mlx_lm import load, generate

    if adapter_dir.exists():
        model, tokenizer = load(
            BASE_MODEL,
            adapter_path=str(adapter_dir),
            tokenizer_config={"trust_remote_code": True},
        )
    else:
        print(f"  [WARN] Adapter not found at {adapter_dir}, using base model only.")
        model, tokenizer = load(BASE_MODEL, tokenizer_config={"trust_remote_code": True})

    print(f"  Model loaded.\n")

    out_fields = src_fields + [c for c in NEW_COLS if c not in src_fields]
    write_mode = "a" if (args.resume and out_csv.exists()) else "w"

    level_results = []
    total = len(all_rows)

    with open(out_csv, write_mode, newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=out_fields, extrasaction="ignore")
        if write_mode == "w":
            writer.writeheader()

        for i, row in enumerate(all_rows, 1):
            year    = row.get("year", "")
            section = row.get("section", "")
            q_id    = row.get("question_id", "")
            flag    = row.get("correct_flag", "")
            uid     = (year, section, q_id)

            print(f"  [{i}/{total}] {year} / {section} / {q_id}")

            # Skip resume
            if uid in done_ids:
                print("    ⏭  Already done.")
                continue

            # Skip placeholder rows
            if flag == "-1" or not row.get("question", "").strip():
                out_row = dict(row)
                out_row["qwen_response"]         = "SKIPPED"
                out_row["qwen_extracted_answer"] = "SKIPPED"
                out_row["deepseek_verdict"]      = "SKIPPED"
                writer.writerow(out_row)
                csvfile.flush()
                continue

            question    = row["question"].strip()
            true_answer = row.get("true_answer", "").strip()

            # Qwen generation
            qwen_response = ""
            try:
                messages = [{"role": "user", "content": question}]
                try:
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    prompt = question

                qwen_response = generate(
                    model, tokenizer,
                    prompt=prompt,
                    max_tokens=MAX_TOKENS,
                    verbose=False,
                ).strip()
                extracted = extract_final_answer(qwen_response)
                print(f"    🤖 Extracted: '{extracted}'", end="  ")
            except Exception as exc:
                print(f"\n    ✗ Qwen failed: {exc}")
                qwen_response = f"ERROR: {exc}"
                extracted     = "ERROR"
                out_row = dict(row)
                out_row["qwen_response"]         = qwen_response
                out_row["qwen_extracted_answer"] = extracted
                out_row["deepseek_verdict"]      = "ERROR"
                writer.writerow(out_row)
                csvfile.flush()
                continue

            # Judge
            verdict = "ERROR"
            try:
                verdict = judge_answer(question, true_answer, qwen_response)
                print(f"→ {verdict}")
            except Exception as exc:
                print(f"\n    ✗ Judge failed: {exc}")

            time.sleep(REQUEST_DELAY)

            out_row = dict(row)
            out_row["qwen_response"]         = qwen_response
            out_row["qwen_extracted_answer"] = extracted
            out_row["deepseek_verdict"]      = verdict
            writer.writerow(out_row)
            csvfile.flush()

            if verdict in ("CORRECT", "INCORRECT"):
                level_results.append({
                    "year":    year,
                    "section": section,
                    "q_id":    q_id,
                    "correct": verdict == "CORRECT",
                })

    # Also load any already-done rows from the CSV for the breakdown
    with open(out_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = row.get("deepseek_verdict", "")
            if v in ("CORRECT", "INCORRECT"):
                entry = {
                    "year":    row.get("year", ""),
                    "section": row.get("section", ""),
                    "q_id":    row.get("question_id", ""),
                    "correct": v == "CORRECT",
                }
                # Avoid duplicates
                if entry not in level_results:
                    level_results.append(entry)

    # De-duplicate by (year, section, q_id)
    seen, unique = set(), []
    for r in level_results:
        key = (r["year"], r["section"], r["q_id"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique


# ── Breakdown helpers ──────────────────────────────────────────────────────────

def accuracy(items: list[dict]) -> tuple[int, int, float]:
    """Returns (correct, total, pct)."""
    if not items:
        return 0, 0, 0.0
    c = sum(1 for r in items if r["correct"])
    return c, len(items), c / len(items) * 100


def breakdown(results: list[dict], key: str) -> dict[str, tuple]:
    groups = defaultdict(list)
    for r in results:
        groups[r[key]].append(r)
    return {k: accuracy(v) for k, v in sorted(groups.items())}


# ── Print helpers ──────────────────────────────────────────────────────────────

def print_breakdown_table(level_data: dict[int, list[dict]]):
    """Print overall, by-section, by-year tables across all 5 levels."""

    levels = sorted(level_data.keys())

    # ── Overall ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print("  OVERALL ACCURACY")
    print(f"{'═'*80}")
    hdr = f"  {'Level':<14}" + "".join(f"{'reasoning_'+str(l):>16}" for l in levels)
    print(hdr)
    print("─" * 80)

    def row_str(label, vals):
        return f"  {label:<14}" + "".join(f"{v:>16}" for v in vals)

    overall_acc  = [f"{accuracy(level_data[l])[2]:.1f}%" for l in levels]
    overall_c    = [f"{accuracy(level_data[l])[0]}/{accuracy(level_data[l])[1]}" for l in levels]
    print(row_str("Accuracy",  overall_acc))
    print(row_str("Correct",   overall_c))

    # ── By Section ────────────────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print("  ACCURACY BY SECTION (question type)")
    print(f"{'═'*80}")
    print(hdr)
    print("─" * 80)

    all_sections = sorted({r["section"] for l in levels for r in level_data[l]})
    for sec in all_sections:
        vals = []
        for l in levels:
            items = [r for r in level_data[l] if r["section"] == sec]
            c, t, pct = accuracy(items)
            vals.append(f"{pct:.1f}% ({c}/{t})")
        print(row_str(sec[:14], vals))

    # ── By Year ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print("  ACCURACY BY YEAR")
    print(f"{'═'*80}")
    print(hdr)
    print("─" * 80)

    all_years = sorted({r["year"] for l in levels for r in level_data[l]})
    for yr in all_years:
        vals = []
        for l in levels:
            items = [r for r in level_data[l] if r["year"] == yr]
            c, t, pct = accuracy(items)
            vals.append(f"{pct:.1f}% ({c}/{t})")
        print(row_str(str(yr)[:14], vals))

    print(f"{'═'*80}\n")


def save_summary(level_data: dict[int, list[dict]]):
    """Save a CSV with all breakdowns for easy import into spreadsheets."""
    levels = sorted(level_data.keys())
    out_path = RUNS_DIR / "accuracy_summary.csv"

    rows_out = []

    # Overall
    for l in levels:
        c, t, pct = accuracy(level_data[l])
        rows_out.append({
            "level": f"reasoning_{l}",
            "breakdown": "overall",
            "group": "all",
            "correct": c, "total": t, "accuracy_pct": round(pct, 2),
        })

    # By section
    all_sections = sorted({r["section"] for l in levels for r in level_data[l]})
    for sec in all_sections:
        for l in levels:
            items = [r for r in level_data[l] if r["section"] == sec]
            c, t, pct = accuracy(items)
            rows_out.append({
                "level": f"reasoning_{l}",
                "breakdown": "section",
                "group": sec,
                "correct": c, "total": t, "accuracy_pct": round(pct, 2),
            })

    # By year
    all_years = sorted({r["year"] for l in levels for r in level_data[l]})
    for yr in all_years:
        for l in levels:
            items = [r for r in level_data[l] if r["year"] == yr]
            c, t, pct = accuracy(items)
            rows_out.append({
                "level": f"reasoning_{l}",
                "breakdown": "year",
                "group": str(yr),
                "correct": c, "total": t, "accuracy_pct": round(pct, 2),
            })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["level","breakdown","group","correct","total","accuracy_pct"]
        )
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Accuracy summary saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    level_data: dict[int, list[dict]] = {}

    for level in REASONING_LEVELS:
        print(f"\n{'█'*70}")
        print(f"  EVALUATING  reasoning_{level}/5")
        print(f"{'█'*70}")
        level_data[level] = evaluate_level(level)
        c, t, pct = accuracy(level_data[level])
        print(f"\n  → reasoning_{level}: {c}/{t} correct  ({pct:.1f}%)")

    print_breakdown_table(level_data)
    save_summary(level_data)


if __name__ == "__main__":
    main()