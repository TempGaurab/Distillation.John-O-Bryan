from datasets import load_dataset
import pandas as pd

dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
df = pd.DataFrame(dataset)

df.to_csv("math500_output.csv", index=False)