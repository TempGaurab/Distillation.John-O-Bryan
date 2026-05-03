import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

API_KEY = os.getenv("Deepseek_api")
BASE_URL = "https://api.deepseek.com"
INPUT_CSV = "results.csv"
OUTPUT_CSV = "reasoning_chain.csv"

def call_deepseek(client, messages, max_tokens=1000):
    """Call DeepSeek API and return the response text."""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.1,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  API error: {e}")
        time.sleep(2)
        return None

def generate_reasoning_1(client, question, true_answer):
    """Generate initial full reasoning (reasoning_1)."""
    messages = [
        {"role": "system", "content": "You are a math expert. Solve the given problem step by step with detailed reasoning."},
        {"role": "user", "content": f"Solve this math problem with full reasoning:\n\n{question}\n\nThe correct answer is {true_answer}. Show detailed step-by-step reasoning that leads to this answer."}
    ]
    return call_deepseek(client, messages, max_tokens=800)

def reduce_reasoning(client, question, previous_reasoning, step_num, true_answer):
    """Progressively reduce the reasoning at each step."""
    reduction_instructions = {
        2: "Reduce the following reasoning by about 40%. Remove verbose explanations, keep key steps and logic. Stay mathematically accurate.",
        3: "Reduce the following reasoning by about 50% from the original. Keep only the essential steps. Be concise but correct.",
        4: "Reduce the following reasoning by about 65% from the original. Compress to core logical steps only. No fluff.",
        5: "This is the final reduction. Compress to the absolute minimum reasoning — just the critical insight(s) needed to reach the answer. Should be 1-3 sentences maximum."
    }

    instruction = reduction_instructions.get(step_num, "Reduce this reasoning further.")

    messages = [
        {"role": "system", "content": "You are a math expert tasked with condensing mathematical reasoning while preserving correctness."},
        {"role": "user", "content": f"""Problem: {question}
Correct Answer: {true_answer}

Previous reasoning:
{previous_reasoning}

Task: {instruction}
Output only the condensed reasoning, nothing else."""}
    ]
    max_tokens = max(50, 600 - (step_num - 1) * 120)
    return call_deepseek(client, messages, max_tokens=max_tokens)

def process_question(client, row, idx, total):
    """Process a single question through all 5 reasoning stages."""
    question = row['question']
    true_answer = row['true_answer']
    year = row['year']
    section = row['section']
    question_id = row['question_id']

    print(f"\n[{idx+1}/{total}] {year} | {question_id}")
    print(f"  Q: {str(question)[:80]}...")

    result = {
        'year': year,
        'section': section,
        'question_id': question_id,
        'question': question,
        'true_answer': true_answer,
        'reasoning_1': None,
        'reasoning_2': None,
        'reasoning_3': None,
        'reasoning_4': None,
        'reasoning_5': None
    }

    # Step 1: Full reasoning
    print("  Generating reasoning_1 (full)...")
    r1 = generate_reasoning_1(client, question, true_answer)
    if not r1:
        print("  Failed at reasoning_1, skipping.")
        return result
    result['reasoning_1'] = r1
    time.sleep(0.5)

    # Steps 2-5: Progressive reduction
    prev = r1
    for step in range(2, 6):
        print(f"  Generating reasoning_{step} (reduction {step-1})...")
        reduced = reduce_reasoning(client, question, prev, step, true_answer)
        if not reduced:
            print(f"  Failed at reasoning_{step}, using previous.")
            reduced = prev
        result[f'reasoning_{step}'] = reduced
        prev = reduced
        time.sleep(0.5)

    token_counts = [len(str(result.get(f'reasoning_{i}', '')).split()) for i in range(1, 6)]
    print(f"  Token approx: {' -> '.join(map(str, token_counts))} words")
    return result

def main():
    if not API_KEY:
        print("Error: Deepseek_api not found in .env file.")
        return

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # Read CSV
    try:
        df = pd.read_csv(INPUT_CSV)
    except Exception as e:
        print(f"Error reading {INPUT_CSV}: {e}")
        return

    # Keep only required columns
    required_cols = ['year', 'section', 'question_id', 'question', 'true_answer']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Missing columns in CSV: {missing}")
        return

    df = df[required_cols].dropna(subset=['question', 'true_answer']).reset_index(drop=True)
    total = len(df)
    print(f"Loaded {total} questions from {INPUT_CSV}")
    print(f"Output will be saved to {OUTPUT_CSV}\n")

    results = []
    for idx, row in df.iterrows():
        result = process_question(client, row, idx, total)
        results.append(result)

        # Save progress every 5 questions
        if (idx + 1) % 5 == 0 or (idx + 1) == total:
            out_df = pd.DataFrame(results)
            out_df.to_csv(OUTPUT_CSV, index=False)
            print(f"\n  Progress saved: {idx+1}/{total} questions -> {OUTPUT_CSV}")

    # Final save
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n{'='*60}")
    print(f"DONE. {total} questions processed.")
    print(f"Output saved to: {OUTPUT_CSV}")
    print(f"Columns: year, section, question_id, question, true_answer, reasoning_1 ... reasoning_5")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()