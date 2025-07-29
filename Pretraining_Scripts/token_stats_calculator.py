import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

LOCAL_DATA_DIR = "/Users/niklascanova/Desktop/SwissAI/Scripts/romansh_data"
TOKENIZER_NAME = "alehc/swissai-tokenizer"
OUTPUT_DIR     = "stats_local_final"

DATA_SUBDIRS = [
    'monolingual',
    os.path.join('polylingual', 'Aligned'),
    os.path.join('polylingual', 'Non_Aligned'),
    'synthetic'
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
print("\nInitializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
print("Tokenizer initialized.")

def count_tokens(example):
    """Counts the number of tokens in the 'text' field of a dataset example."""
    
    text = example.get("text", "") or ""
    tokens = tokenizer.tokenize(text)
    return {"token_count": len(tokens)}

print("\n--- Starting Data Loading and Tokenization from Local Files ---")
df_list = []
for sub_dir in tqdm(DATA_SUBDIRS, desc="Processing subdirectories"):
    dir_path = os.path.join(LOCAL_DATA_DIR, sub_dir)
    
    if not os.path.isdir(dir_path):
        print(f"Warning: Directory not found, skipping: {dir_path}")
        continue

    file_list = glob.glob(os.path.join(dir_path, "**", "*.jsonl.gz"), recursive=True)

    if not file_list:
        print(f"Warning: No .jsonl.gz files found in {dir_path}, skipping.")
        continue

    ds = load_dataset('json', data_files=file_list, split='train')
    ds = ds.map(count_tokens, num_proc=os.cpu_count()) 

    df_temp = pd.DataFrame(ds)
    df_temp['source_dir'] = sub_dir
    df_list.append(df_temp)

if not df_list:
    print(" ERROR: No data was loaded. Please check your LOCAL_DATA_DIR and subdirectories.")
    exit()

df = pd.concat(df_list, ignore_index=True)
print("--- Data Loading and Tokenization Complete ---\n")

print("--- Aggregating and Plotting Language Stats ---")
lang_stats = (
    df.groupby("language")
      .agg(rows=("text", "count"), tokens=("token_count", "sum"))
      .reset_index()
      .sort_values(by='tokens', ascending=False)
)
lang_stats.to_csv(os.path.join(OUTPUT_DIR, "tokens_by_language.csv"), index=False)

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(12, 7))
bars = plt.bar(lang_stats["language"], lang_stats["tokens"], color='skyblue')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval/1e6:.1f}M', va='bottom', ha='center', fontsize=11)
plt.ylabel("Total Tokens", fontsize=12)
plt.title("Tokens by Language", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "tokens_by_language.png"), dpi=300)
plt.close()
print("Language stats processing complete.\n")

print("--- Aggregating and Plotting Idiom Stats ---")
roh_df = df[df["language"] == "roh"].copy()
if 'idiom' not in roh_df.columns:
    roh_df['idiom'] = 'N/A'
roh_df['idiom'] = roh_df['idiom'].fillna("Unknown")

idiom_stats = (
    roh_df.groupby("idiom")
           .agg(rows=("text", "count"), tokens=("token_count", "sum"))
           .reset_index()
           .sort_values(by='tokens', ascending=False)
)
idiom_stats.to_csv(os.path.join(OUTPUT_DIR, "tokens_by_idiom_roh.csv"), index=False)

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(12, 7))
bars = plt.bar(idiom_stats["idiom"], idiom_stats["tokens"], color='coral')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval/1e6:.1f}M', va='bottom', ha='center', fontsize=11)
plt.ylabel("Total Tokens", fontsize=12)
plt.title("Tokens by Idiom (for 'roh' language)", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "tokens_by_idiom_roh.png"), dpi=300)
plt.close()
print("Idiom stats processing complete.\n")

print("--- Aggregating and Plotting Source Directory Stats ---")
dir_stats = (
    df.groupby("source_dir")
      .agg(rows=("text", "count"), tokens=("token_count", "sum"))
      .reset_index()
      .sort_values(by='tokens', ascending=False)
)
dir_stats.to_csv(os.path.join(OUTPUT_DIR, "tokens_by_source_dir.csv"), index=False)

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(12, 7))
bars = plt.bar(dir_stats["source_dir"], dir_stats["tokens"], color='mediumseagreen')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval/1e6:.1f}M', va='bottom', ha='center', fontsize=11)
plt.ylabel("Total Tokens", fontsize=12)
plt.title("Tokens by Source Directory", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "tokens_by_source_dir.png"), dpi=300)
plt.close()
print("Source directory stats processing complete.\n")


print(f" All done! Stats and plots are in the '{OUTPUT_DIR}' folder.")
