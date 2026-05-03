#!/usr/bin/env python3
"""
filtered_accuracy_benchmark.py

Reads the per-level results CSVs already produced by reasoning_accuracy_benchmark.py
(runs/reasoning_N/results.csv) and re-scores them, but ONLY for questions whose
'question' text appears in results2.csv with correct_flag == 1.

Matching is done on the 'question' column (stripped, case-sensitive).

Outputs the same table format as the original benchmark:
  - Overall accuracy per level
  - Accuracy by section per level
  - Accuracy by year per level
  - Saves runs/accuracy_summary_filtered.csv

Usage:
    python filtered_accuracy_benchmark.py

Edit CONFIG below to match your paths.
"""

import csv
from pathlib import Path
from collections import defaultdict

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
RUNS_DIR        = Path("runs")
RESULTS2_CSV    = "results2.csv"
REASONING_LEVELS = [1, 2, 3, 4, 5]
# ══════════════════════════════════════════════════════════════════════════════


# ── Step 1: Build the set of "allowed" questions from results2.csv ─────────────

results2_path = Path(RESULTS2_CSV)
if not results2_path.exists():
    raise FileNotFoundError(f"'{RESULTS2_CSV}' not found.")

allowed_questions: set[str] = set()

with open(results2_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        flag = str(row.get("correct_flag", "")).strip()
        q    = row.get("question", "").strip()
        if flag == "1" and q:
            allowed_questions.add(q)

print(f"Loaded {len(allowed_questions)} questions with correct_flag=1 from '{RESULTS2_CSV}'.\n")


# ── Step 2: Load each level's results.csv and filter ──────────────────────────

def load_level(level: int) -> list[dict]:
    """
    Load runs/reasoning_N/results.csv, keep only rows whose 'question' is in
    allowed_questions and whose deepseek_verdict is CORRECT or INCORRECT.
    Returns list of dicts with keys: year, section, q_id, correct.
    """
    csv_path = RUNS_DIR / f"reasoning_{level}" / "results.csv"
    if not csv_path.exists():
        print(f"  [WARN] {csv_path} not found — skipping level {level}.")
        return []

    results = []
    skipped_flag = 0
    skipped_verdict = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q       = row.get("question", "").strip()
            verdict = row.get("deepseek_verdict", "").strip()

            # Must be in allowed set
            if q not in allowed_questions:
                skipped_flag += 1
                continue

            # Must have a real verdict
            if verdict not in ("CORRECT", "INCORRECT"):
                skipped_verdict += 1
                continue

            results.append({
                "year":    row.get("year", ""),
                "section": row.get("section", ""),
                "q_id":    row.get("question_id", ""),
                "correct": verdict == "CORRECT",
            })

    # De-duplicate by (year, section, q_id) — keep first occurrence
    seen, unique = set(), []
    for r in results:
        key = (r["year"], r["section"], r["q_id"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    print(f"  reasoning_{level}: {len(unique)} matched questions  "
          f"(skipped {skipped_flag} not-in-filter, {skipped_verdict} non-verdict rows)")
    return unique


# ── Step 3: Accuracy helpers ───────────────────────────────────────────────────

def accuracy(items: list[dict]) -> tuple[int, int, float]:
    if not items:
        return 0, 0, 0.0
    c = sum(1 for r in items if r["correct"])
    return c, len(items), c / len(items) * 100


# ── Step 4: Print the same table format ───────────────────────────────────────

def print_breakdown_table(level_data: dict[int, list[dict]]):
    levels = sorted(level_data.keys())
    col_w  = 18  # width per level column

    hdr = f"  {'Level':<16}" + "".join(f"{'reasoning_'+str(l):>{col_w}}" for l in levels)
    sep = "═" * (16 + col_w * len(levels) + 2)
    thin = "─" * (16 + col_w * len(levels) + 2)

    def row_str(label, vals):
        return f"  {label:<16}" + "".join(f"{v:>{col_w}}" for v in vals)

    # ── Overall ───────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  OVERALL ACCURACY  (correct_flag == 1 questions only)")
    print(sep)
    print(hdr)
    print(thin)

    overall_acc = [f"{accuracy(level_data[l])[2]:.1f}%" for l in levels]
    overall_c   = [f"{accuracy(level_data[l])[0]}/{accuracy(level_data[l])[1]}" for l in levels]
    print(row_str("Accuracy", overall_acc))
    print(row_str("Correct",  overall_c))

    # ── By Section ────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  ACCURACY BY SECTION (question type)")
    print(sep)
    print(hdr)
    print(thin)

    all_sections = sorted({r["section"] for l in levels for r in level_data[l]})
    for sec in all_sections:
        vals = []
        for l in levels:
            items = [r for r in level_data[l] if r["section"] == sec]
            c, t, pct = accuracy(items)
            vals.append(f"{pct:.1f}% ({c}/{t})")
        print(row_str(sec[:16], vals))

    # ── By Year ───────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  ACCURACY BY YEAR")
    print(sep)
    print(hdr)
    print(thin)

    all_years = sorted({r["year"] for l in levels for r in level_data[l]})
    for yr in all_years:
        vals = []
        for l in levels:
            items = [r for r in level_data[l] if r["year"] == yr]
            c, t, pct = accuracy(items)
            vals.append(f"{pct:.1f}% ({c}/{t})")
        print(row_str(str(yr)[:16], vals))

    print(f"{sep}\n")


# ── Step 5: Save summary CSV ───────────────────────────────────────────────────

def save_summary(level_data: dict[int, list[dict]]):
    levels    = sorted(level_data.keys())
    out_path  = RUNS_DIR / "accuracy_summary_filtered.csv"

    rows_out = []

    # Overall
    for l in levels:
        c, t, pct = accuracy(level_data[l])
        rows_out.append({
            "level": f"reasoning_{l}", "breakdown": "overall", "group": "all",
            "correct": c, "total": t, "accuracy_pct": round(pct, 2),
        })

    # By section
    all_sections = sorted({r["section"] for l in levels for r in level_data[l]})
    for sec in all_sections:
        for l in levels:
            items = [r for r in level_data[l] if r["section"] == sec]
            c, t, pct = accuracy(items)
            rows_out.append({
                "level": f"reasoning_{l}", "breakdown": "section", "group": sec,
                "correct": c, "total": t, "accuracy_pct": round(pct, 2),
            })

    # By year
    all_years = sorted({r["year"] for l in levels for r in level_data[l]})
    for yr in all_years:
        for l in levels:
            items = [r for r in level_data[l] if r["year"] == yr]
            c, t, pct = accuracy(items)
            rows_out.append({
                "level": f"reasoning_{l}", "breakdown": "year", "group": str(yr),
                "correct": c, "total": t, "accuracy_pct": round(pct, 2),
            })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["level","breakdown","group","correct","total","accuracy_pct"]
        )
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Filtered accuracy summary saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    level_data: dict[int, list[dict]] = {}

    print("Loading per-level results…")
    for level in REASONING_LEVELS:
        level_data[level] = load_level(level)

    # Quick per-level summary line (mirrors original output style)
    print()
    for level in REASONING_LEVELS:
        c, t, pct = accuracy(level_data[level])
        print(f"  → reasoning_{level}: {c}/{t} correct  ({pct:.1f}%)")

    print_breakdown_table(level_data)
    save_summary(level_data)


if __name__ == "__main__":
    main()