import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tabulate import tabulate
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables from .env file
load_dotenv()

# Configuration
API_KEY = os.getenv("Deepseek_api")
BASE_URL = "https://api.deepseek.com"
INPUT_CSV = "results.csv"
OUTPUT_CSV = "yearly_difficulty_average.csv"
MAX_WORKERS = 10 # Number of parallel requests

def get_difficulty_from_deepseek(client, question, index):
    """
    Asks DeepSeek to categorize a single question's difficulty.
    """
    prompt = f"""
    Analyze the following math question and categorize its difficulty on a scale of 1 to 10.
    1 is extremely easy (basic arithmetic), and 10 is extremely hard (Olympiad-level).
    
    Question:
    {question}
    
    Provide your response exactly in this format:
    Difficulty: [Score]
    """
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a math expert who categorizes the difficulty of math problems accurately. You only output the difficulty score."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=20
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse difficulty score
        for line in content.split('\n'):
            if "Difficulty:" in line:
                score_str = line.split("Difficulty:")[1].strip()
                score = "".join(filter(lambda x: x.isdigit() or x == '.', score_str))
                return index, float(score) if score else None
    except Exception:
        return index, None
    return index, None

def main():
    if not API_KEY:
        print("Error: Deepseek_api not found in .env file.")
        return

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # Read the results.csv
    try:
        df = pd.read_csv(INPUT_CSV)
    except Exception as e:
        print(f"Error reading {INPUT_CSV}: {e}")
        return

    df = df[df['question'].notna() & df['year'].notna()].copy()
    total = len(df)
    print(f"Starting parallel analysis of {total} questions (Workers: {MAX_WORKERS})...")
    
    difficulties = [None] * total
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(get_difficulty_from_deepseek, client, row['question'], i): i 
                   for i, (idx, row) in enumerate(df.iterrows())}
        
        completed = 0
        for future in as_completed(futures):
            index, diff = future.result()
            difficulties[index] = diff
            completed += 1
            if completed % 20 == 0 or completed == total:
                print(f"Progress: {completed}/{total} questions analyzed...")

    df['difficulty'] = difficulties
    
    # Calculate final average difficulty per year
    avg_df = df.groupby('year')['difficulty'].mean().reset_index()
    avg_df.columns = ['Year', 'Average Difficulty']
    
    # Calculate Percentage Difficulty (Assuming 10 is 100%)
    avg_df['Difficulty Percentage'] = (avg_df['Average Difficulty'] / 10 * 100).round(2).astype(str) + "%"
    
    # Save to final CSV
    avg_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nFull analysis complete. Results saved to {OUTPUT_CSV}")
    
    # Print the table
    print("\n" + "="*50)
    print("AVERAGE DIFFICULTY BY YEAR (FULL ANALYSIS)")
    print("="*50)
    print(tabulate(avg_df, headers="keys", tablefmt="grid", showindex=False))
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
