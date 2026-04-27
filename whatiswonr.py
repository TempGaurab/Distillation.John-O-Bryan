import pandas as pd
import re

# 1. Load and clean the data
file_path = 'qwen_tuned_results_200iterations.csv'

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: Could not find '{file_path}'. Make sure it's in the same directory as this script.")
    exit()

# Clean column names just in case there are hidden spaces
df.columns = df.columns.str.strip()

# 2. Filter for incorrect answers (case insensitive and stripped of whitespace)
incorrect_df = df[df['deepseek_verdict'].astype(str).str.strip().str.upper() == 'INCORRECT'].copy()

if incorrect_df.empty:
    print("No incorrect answers found.")
else:
    print(f"Found {len(incorrect_df)} incorrect answers. Running Error Analysis...\n")

    # 3. Define the categorization logic
    def categorize_error(row):
        true_ans = str(row['true_answer']).strip()
        qwen_ans = str(row['qwen_extracted_answer']).strip()
        
        # A. False Negative: The text matches exactly, but the evaluator marked it wrong
        if true_ans == qwen_ans or true_ans.lower() == qwen_ans.lower():
            return "False Negative (Evaluator Error)"
        
        # B. Extraction / Formatting Failure: Contains English words, equations (S_8 =), or raw LaTeX commands (\frac)
        # We use a regex looking for words 3+ letters long (excluding 'sqrt'), equals signs, or LaTeX markers
        if re.search(r'[a-zA-Z]{3,}(?!rt)|=|\$|\\text|\\frac', qwen_ans) and qwen_ans != true_ans:
            return "Extraction / Formatting Failure"
        
        # C. Format / Type Mismatch: e.g., % vs decimal, or fraction vs decimal
        if "%" in qwen_ans or ("/" in true_ans and "." in qwen_ans) or ("/" in qwen_ans and "." in true_ans):
            return "Format / Type Mismatch"
        
        # D. Rounding / Precision Error: The numbers are very close (off by < 0.5)
        try:
            # Strip commas just in case (e.g., 4,000 -> 4000)
            t_num = float(true_ans.replace(',', ''))
            q_num = float(qwen_ans.replace(',', ''))
            if abs(t_num - q_num) < 0.5 and t_num != q_num:
                return "Rounding / Precision Error"
        except ValueError:
            pass # Fails if the answers aren't clean numbers (e.g., they contain variables or square roots)
        
        # E. Default fallback
        return "Calculation / Logic Error"

    # 4. Apply the logic to create a new column
    incorrect_df['error_category'] = incorrect_df.apply(categorize_error, axis=1)

    # 5. Output the Analysis Report
    print("========================================")
    print("        ERROR ANALYSIS SUMMARY          ")
    print("========================================")
    category_counts = incorrect_df['error_category'].value_counts()
    
    for cat, count in category_counts.items():
        print(f"{count:3d} | {cat}")

    print("\n========================================")
    print("          DETAILED BREAKDOWN            ")
    print("========================================")
    
    for category in category_counts.index:
        print(f"\n[{category.upper()}]")
        # Filter down to just this category
        subset = incorrect_df[incorrect_df['error_category'] == category]
        
        for _, row in subset.iterrows():
            print(f"  Question ID : {row['question_id']}")
            print(f"  True Answer : {row['true_answer']}")
            print(f"  Qwen Output : {row['qwen_extracted_answer']}")
            print("-" * 40)