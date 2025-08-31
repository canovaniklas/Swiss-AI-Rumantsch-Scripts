# Swiss-AI_Rumansh_Scripts

Utilities and scripts for building and analysing Rumantsch datasets for **pretraining** and **SFT** (supervised fine‑tuning).  
The repo includes helpers to convert data to chat formats, create Hugging Face datasets, compute token statistics, and generate/score synthetic SFT pairs.
---

## 📁Repository structure

```
Swiss-AI_Rumansh_Scripts/
├─ Human_SFT_Scripts/
│  ├─ convert_to_chat_format.py
│  ├─ create_hf_dataset.py
├─ Pretraining_Scripts/
│  ├─ jsonl_reader.py
│  ├─ token_stats_calculator.py
│  └─ universal_jsonl_builder.py
├─ Synthetic_SFT_Scripts/
│  ├─ dictionary_sft.py
│  ├─ idiom_identification_sft.py
│  ├─ translation_sft_idiom.py
│  ├─ translation_sft_quality_scoring.py
│  └─ translation_sft_RG.py
├─ requirements.txt
├─ .gitignore
```
---

## Quick start (step‑by‑step)

### 1) Clone or create the repo locally
```bash
# If you already created the GitHub repo:
git clone https://github.com/canovaniklas/Swiss-AI_Rumansh_Scripts.git
cd Swiss-AI_Rumansh_Scripts
```

If you’re starting locally and will push later:
```bash
mkdir Swiss-AI_Rumansh_Scripts && cd Swiss-AI_Rumansh_Scripts
git init -b main
```

### 2) Create and activate a virtual environment
```bash
# macOS/Linux
python -m venv venv
source venv/bin/activate

# Windows PowerShell
# python -m venv venv
# .\venv\Scripts\Activate
```

### 3) Install Python dependencies
You can install them from the provided requirements file **or** directly from the list.

```bash
# Using requirements.txt
pip install -r requirements.txt

# OR install the main libs explicitly
pip install datasets tqdm transformers pandas matplotlib nltk torch sentence-transformers openai
```

> If you’re on Apple Silicon and want GPU acceleration, consult the PyTorch website for the latest install command. CPU-only works out of the box.

### 4) (Optional) Prepare NLTK data
If any scripts use NLTK tokenizers, download common packages once:
```bash
python - << 'PY'
import nltk
for pkg in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.download(pkg)
    except Exception as e:
        print("Could not download", pkg, e)
PY
```

### 5) (Optional) Set API keys
Some synthetic generation or scoring scripts may call external APIs (e.g. OpenAI).  
Export your key as an environment variable:

```bash
# macOS/Linux
export OPENAI_API_KEY="sk-..."
# Windows PowerShell
# setx OPENAI_API_KEY "sk-..."
```

### 6) Verify your environment
```bash
python -c "import torch, datasets, transformers, pandas, nltk; print('OK')"
```

---

##  Typical workflows

### A) Convert raw data to chat format (human SFT prep)
Target: Convert JSON/JSONL or other structure into a chat-style format for SFT.

```bash
# Show options
python Human_SFT_Scripts/convert_to_chat_format.py -h

# Example pattern
python Human_SFT_Scripts/convert_to_chat_format.py \
  --input data/raw.jsonl \
  --output data/chat_train.jsonl \
  --system-prompt "You are a helpful assistant for Rumantsch." \
  --max-turns 6
```

### B) Build a Hugging Face dataset from local files
```bash
python Human_SFT_Scripts/create_hf_dataset.py -h

# Example pattern
python Human_SFT_Scripts/create_hf_dataset.py \
  --input data/chat_train.jsonl \
  --repo-id your-hf-username/rumantsch-sft \
  --push-to-hub
```

### C) Universal JSONL builder (pretraining)
Aggregate multiple sources into a unified JSONL for pretraining.

```bash
python Pretraining_Scripts/universal_jsonl_builder.py -h

# Example pattern
python Pretraining_Scripts/universal_jsonl_builder.py \
  --sources data/corpus1/*.jsonl data/corpus2/*.jsonl \
  --output data/pretrain_corpus.jsonl \
  --shuffle
```

### D) Quick JSONL inspection
```bash
python Pretraining_Scripts/jsonl_reader.py -h

# Example pattern
python Pretraining_Scripts/jsonl_reader.py \
  --input data/pretrain_corpus.jsonl \
  --head 5
```

### E) Token statistics (lengths, distribution, vocab estimates)
```bash
python Pretraining_Scripts/token_stats_calculator.py -h

# Example pattern
python Pretraining_Scripts/token_stats_calculator.py \
  --input data/pretrain_corpus.jsonl \
  --tokenizer "gpt2" \
  --sample 10000 \
  --report out/token_stats.md
```

### F) Synthetic SFT generation & scoring
> The following scripts help create and assess synthetic pairs (prompt/response, translations, idioms, etc.). Check `-h` for exact parameters.

```bash
# Dictionary-style SFT creation
python Synthetic_SFT_Scripts/dictionary_sft.py -h

# Idiom identification data
python Synthetic_SFT_Scripts/idiom_identification_sft.py -h

# Translation with idiom awareness
python Synthetic_SFT_Scripts/translation_sft_idiom.py -h

# Translation quality scoring
python Synthetic_SFT_Scripts/translation_sft_quality_scoring.py -h

# Rule-guided / rubric-based generation or filtering
python Synthetic_SFT_Scripts/translation_sft_RG.py -h
```

---

##  Data layout tips

- Prefer **JSONL** (one JSON object per line) for large corpora.  
- Keep raw sources under `data/raw/` and derived artifacts under `data/processed/`.  
- Use stable field names like `{"text": "...", "lang": "roh", "meta": {...}}` for pretraining, and chat-style fields like:
  ```json
  {"messages":[
    {"role":"system","content":"..."},
    {"role":"user","content":"..."},
    {"role":"assistant","content":"..."}
  ]}
  ```


##  License

Add your preferred license (e.g., MIT).

---
