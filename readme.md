# John O'Bryan Mathematics Competition — Distillation & Analysis

## ✅ Project Roadmap

- [x] **1. Gather Data** — Collected all questions from the John O'Bryan Mathematics Competition at NKU (2011–2025).
- [x] **2. Descriptive Analysis** — Analyzed question distribution by year and section.
- [x] **3. DeepSeek Reasoning Generation** — Ran a Solver + Verifier dual-agent pipeline using DeepSeek to generate and verify answers. All outputs stored in `results.csv`.
- [x] **4. Reasoning Chain Distillation** — Used DeepSeek to generate 5 progressive levels of reasoning per question (full → minimal), saved to `reasoning_chain.csv`.
- [x] **5. Fine-tune Qwen2.5** — Fine-tuned `Qwen2.5-7B-Instruct-4bit` via `mlx_lm.lora` on each of the 5 reasoning levels.
- [x] **6. Benchmark Evaluation** — Evaluated all 5 LoRA adapters against the full question set using DeepSeek as an automated judge.

---

## 📁 Key Files

| File | Description |
|---|---|
| `results.csv` | DeepSeek dual-agent outputs (solver + verifier) for all 651 competition questions |
| `results2.csv` | Extended results with additional metadata |
| `reasoning_chain.csv` | 5-level progressive reasoning chains for each question |
| `reduce_reasoning.py` | Script to generate the 5-level reasoning chain via DeepSeek |
| `rab.py` | Benchmark script — evaluates all 5 LoRA adapters on `results.csv` |
| `tatata.py` | Filtered benchmark — re-scores using only `correct_flag == 1` questions |
| `average_words.py` | Computes average response word count per reasoning level |
| `fine-tune-math-500.py` | Fine-tuning script (MLX LoRA) |
| `runs/` | Per-level adapter weights and evaluation results |
| `runs/accuracy_summary.csv` | Accuracy breakdown by level, section, and year |

---

## 🧠 Pipeline Overview

### Stage 1 — Dual-Agent DeepSeek Evaluation

Two DeepSeek agents process each question:

- **Agent 1 (Solver):** `deepseek-reasoner` → generates `agent1_answer` + `agent1_reasoning`
- **Agent 2 (Verifier):** `deepseek-reasoner` → compares solver answer to ground truth, outputs `agent2_verdict` + `agent2_justification`

```text
═══════════════════════════════════════════════════
✅  Done! Results saved to 'results.csv'
   Questions attempted : 651
   Correct (flag=1)    : 594  (91.2%)
   Wrong   (flag=0)    : 57
   Skipped (flag=-1)   : 20   (placeholder/missing)
═══════════════════════════════════════════════════
```

---

### Stage 2 — Reasoning Chain Distillation (`reduce_reasoning.py`)

For each question, DeepSeek generates 5 progressively compressed reasoning chains:

| Level | Description | Target Length |
|---|---|---|
| `reasoning_1` | Full step-by-step reasoning | ~800 tokens |
| `reasoning_2` | ~40% reduction | ~480 tokens |
| `reasoning_3` | ~50% reduction | ~400 tokens |
| `reasoning_4` | ~65% reduction | ~280 tokens |
| `reasoning_5` | Minimal — 1–3 sentence insight | ~80 tokens |

Output saved to `reasoning_chain.csv` with columns:
`year, section, question_id, question, true_answer, reasoning_1 … reasoning_5`

---

### Stage 3 — LoRA Fine-tuning (MLX)

Each reasoning level used to fine-tune a separate LoRA adapter on top of `Qwen2.5-7B-Instruct-4bit`.

```text
════════════════════════════════════════════════════════════════════════════════
  FINE-TUNING COMPARISON
════════════════════════════════════════════════════════════════════════════════
Level        Examples    Train    Valid     Test   TestLoss    TestPPL
────────────────────────────────────────────────────────────────────────────────
  reasoning_1    671      536       67       68      0.593      1.809
  reasoning_2    671      536       67       68      1.137      3.117
  reasoning_3    671      536       67       68      1.411      4.100
  reasoning_4    671      536       67       68      1.752      5.766
  reasoning_5    671      536       67       68      2.169      8.750
════════════════════════════════════════════════════════════════════════════════
```

> **Key insight:** Models trained on more detailed reasoning (lower level number) achieve significantly lower test loss and perplexity.

---

### Stage 4 — Benchmark Accuracy (`rab.py`)

Each LoRA adapter was evaluated against the full question set. Qwen generates an answer; DeepSeek judges correctness.

```text
════════════════════════════════════════════════════════════════════════════════
  OVERALL ACCURACY
════════════════════════════════════════════════════════════════════════════════
  Level          reasoning_1  reasoning_2  reasoning_3  reasoning_4  reasoning_5
────────────────────────────────────────────────────────────────────────────────
  Accuracy           57.6%        56.8%        47.6%        40.7%        39.6%
  Correct          373/648      370/651      310/651      265/651      258/651
════════════════════════════════════════════════════════════════════════════════

  ACCURACY BY SECTION
════════════════════════════════════════════════════════════════════════════════
  freshman_soph  56.8%(151/266) 59.6%(159/267) 54.3%(145/267) 41.9%(112/267) 44.2%(118/267)
  junior_senior  59.3%(159/268) 57.2%(154/269) 48.7%(131/269) 43.5%(117/269) 42.8%(115/269)
  two_person_sp  55.3% (63/114) 49.6% (57/115) 29.6% (34/115) 31.3% (36/115) 21.7% (25/115)

  ACCURACY BY YEAR
════════════════════════════════════════════════════════════════════════════════
  2011           51.4%(19/37) 54.1%(20/37) 37.8%(14/37) 32.4%(12/37) 32.4%(12/37)
  2012           59.2%(29/49) 50.0%(25/50) 48.0%(24/50) 34.0%(17/50) 42.0%(21/50)
  2013           47.7%(21/44) 53.3%(24/45) 44.4%(20/45) 42.2%(19/45) 46.7%(21/45)
  2014           57.8%(26/45) 52.2%(24/46) 50.0%(23/46) 41.3%(19/46) 32.6%(15/46)
  2015           62.8%(27/43) 51.2%(22/43) 30.2%(13/43) 39.5%(17/43) 37.2%(16/43)
  2016           53.3%(24/45) 62.2%(28/45) 48.9%(22/45) 28.9%(13/45) 35.6%(16/45)
  2017           74.4%(32/43) 79.1%(34/43) 72.1%(31/43) 58.1%(25/43) 51.2%(22/43)
  2018           51.2%(22/43) 37.2%(16/43) 41.9%(18/43) 25.6%(11/43) 27.9%(12/43)
  2019           61.5%(16/26) 61.5%(16/26) 53.8%(14/26) 50.0%(13/26)  34.6%(9/26)
  2020           45.9%(17/37) 59.5%(22/37) 56.8%(21/37) 40.5%(15/37) 48.6%(18/37)
  2021           50.0%(23/46) 58.7%(27/46) 39.1%(18/46) 34.8%(16/46) 39.1%(18/46)
  2022           54.2%(26/48) 58.3%(28/48) 47.9%(23/48) 58.3%(28/48) 37.5%(18/48)
  2023           55.3%(26/47) 59.6%(28/47) 48.9%(23/47) 48.9%(23/47) 36.2%(17/47)
  2024           66.7%(32/48) 62.5%(30/48) 45.8%(22/48) 31.2%(15/48) 41.7%(20/48)
  2025           70.2%(33/47) 55.3%(26/47) 51.1%(24/47) 46.8%(22/47) 48.9%(23/47)
════════════════════════════════════════════════════════════════════════════════
```

Full breakdown saved to `runs/accuracy_summary.csv`.

---

## 📊 Results Dataset Codebook (`results.csv`)

| Column | Type | Description |
|---|---|---|
| `year` | String | Competition year (e.g. `2024`) |
| `section` | String | Section name (e.g. `freshman_sophomore_test`) |
| `question_id` | String | Unique ID within section (e.g. `question_1`) |
| `question` | String | Full problem text |
| `true_answer` | String | Ground-truth answer |
| `agent1_answer` | String | Solver's final answer |
| `agent1_reasoning` | String | Chain-of-thought from solver |
| `agent2_verdict` | Categorical | `CORRECT` / `INCORRECT` / `UNKNOWN` |
| `agent2_justification` | String | Verifier explanation |
| `correct_flag` | Integer | `1` = correct, `0` = incorrect, `-1` = skipped |

---

## ⚙️ Evaluation Logic

```text
If agent2_verdict == "CORRECT" → correct_flag = 1
Else                           → correct_flag = 0
Placeholder rows               → correct_flag = -1
```