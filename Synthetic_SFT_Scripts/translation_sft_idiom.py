import gzip
import json
import nltk
import os
import torch
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util


INPUT_FILE_PATH = "/Users/niklascanova/Desktop/SwissAI/romansh_data/polylingual/Aligned/Laws_Sagogn.jsonl.gz"
OUTPUT_DIR = "SFT_Data"
TARGET_IDIOM = "Sursilvan" 

EMBEDDING_MODEL = 'paraphrase-multilingual-mpnet-base-v2'
SIMILARITY_THRESHOLD = 0.65
MAX_LINES_TO_PROCESS = None

def clean_sentence(sentence: str) -> str:
    """
    Applies specific cleaning rules to a sentence.
    """
    cleaned = sentence.replace('\nArt', '\n')
    cleaned = re.sub(r'^\d+\s*\n', '', cleaned)
    cleaned = cleaned.replace('\t', '')
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()

def is_valid_sentence(sentence: str) -> bool:
    """
    Applies a set of rules to filter out noisy or unhelpful sentences.
    """
    if len(sentence.strip()) < 50:
        return False
    if len(set(sentence.strip())) < 5:
        return False
    alnum_chars = sum(1 for char in sentence if char.isalnum())
    if len(sentence) > 0 and (alnum_chars / len(sentence)) < 0.5:
        return False
    if re.fullmatch(r'Art\.\s*[\d\s\*]*', sentence.strip()):
        return False
    return True

def download_nltk_resources():
    """
    Downloads the NLTK resources required for sentence tokenization ('punkt').
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("NLTK 'punkt' tokenizer not found. Downloading...")
        nltk.download('punkt')
        print("Download complete.")

def parse_line(line: str):
    """
    Parses a single JSON object (one line) and extracts metadata and text.
    """
    try:
        data = json.loads(line)
        text = data.get("text", "")
        return data, text
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse JSON from line. Skipping. Error: {e}\nLine: {line[:150]}...")
        return None, None

def create_sft_data_with_alignment(input_path: str, output_dir: str):
    """
    Reads an aligned, gzipped JSONL file, uses sentence embeddings to find the
    best translation pairs, and creates a single consolidated JSONL file containing
    records for both translation directions.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at '{input_path}'")
        return
    os.makedirs(output_dir, exist_ok=True)
    target_idiom_filename = TARGET_IDIOM.replace(' ', '')
    output_path = os.path.join(output_dir, f"sft_{target_idiom_filename}_bidirectional.jsonl")

    print("Starting SFT data generation...")
    print(f"All data will be saved to a single file: '{output_path}'")
    download_nltk_resources()

    print(f"Loading sentence embedding model: '{EMBEDDING_MODEL}'...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Model loaded.")

    prompt_template_de_to_idiom = f"Übersetze den folgenden Satz ins {TARGET_IDIOM}: {{german_sentence}}"
    prompt_template_idiom_to_de = f"Übersetze den folgenden Satz ins Deutsche: {{romansh_sentence}}"

    total_records_generated = 0

    with gzip.open(input_path, 'rt', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        lines = infile.readlines()
        print(f"Read {len(lines)} total lines from the input file.")

        if MAX_LINES_TO_PROCESS is not None:
            lines = lines[:MAX_LINES_TO_PROCESS]
            print(f"Processing only the first {len(lines)} lines as requested.")

        for i in range(0, len(lines), 2):
            if i + 1 >= len(lines):
                print(f"Warning: Odd number of lines. Skipping last line.")
                continue
            
            print(f"\n--- Processing document pair {i//2 + 1} ---")

            meta1, text1 = parse_line(lines[i])
            meta2, text2 = parse_line(lines[i+1])

            if text1 is None or text2 is None:
                continue

            if meta1.get('language') == 'roh' and meta2.get('language') == 'de':
                romansh_text, german_text = text1, text2
            elif meta1.get('language') == 'de' and meta2.get('language') == 'roh':
                german_text, romansh_text = text1, text2
            else:
                print(f"Warning: Could not find a 'roh'/'de' pair at lines {i+1}-{i+2}. Skipping.")
                continue

            romansh_sentences = [clean_sentence(s) for s in nltk.sent_tokenize(romansh_text, language='german')]
            german_sentences = [clean_sentence(s) for s in nltk.sent_tokenize(german_text, language='german')]
            
            if not romansh_sentences or not german_sentences:
                print("Warning: No sentences found after cleaning. Skipping pair.")
                continue

            print(f"Tokenized {len(german_sentences)} German and {len(romansh_sentences)} Romansh sentences.")
            print("Encoding sentences into embeddings...")
            
            german_embeddings = model.encode(german_sentences, convert_to_tensor=True, show_progress_bar=False)
            romansh_embeddings = model.encode(romansh_sentences, convert_to_tensor=True, show_progress_bar=False)

            cosine_scores = util.cos_sim(german_embeddings, romansh_embeddings)

            print("Finding best bidirectional matches...")
            
            g2r_scores, g2r_indices = torch.max(cosine_scores, dim=1)
            r2g_scores, r2g_indices = torch.max(cosine_scores, dim=0)

            for j in tqdm(range(len(german_sentences)), desc="Aligning and writing to single file"):
                best_romansh_idx = g2r_indices[j].item()
                
                if r2g_indices[best_romansh_idx].item() == j:
                    score = g2r_scores[j].item()
                    
                    if score >= SIMILARITY_THRESHOLD:
                        de_sent = german_sentences[j]
                        roh_sent = romansh_sentences[best_romansh_idx]

                        if is_valid_sentence(de_sent) and is_valid_sentence(roh_sent):
                            prompt1 = prompt_template_de_to_idiom.format(german_sentence=de_sent)
                            sft_record1 = {"Prompt": prompt1, "Answer": roh_sent}
                            outfile.write(json.dumps(sft_record1, ensure_ascii=False) + '\n')
                            total_records_generated += 1

                            prompt2 = prompt_template_idiom_to_de.format(romansh_sentence=roh_sent)
                            sft_record2 = {"Prompt": prompt2, "Answer": de_sent}
                            outfile.write(json.dumps(sft_record2, ensure_ascii=False) + '\n')
                            total_records_generated += 1

    print("\n-------------------------------------------------")
    print("SFT data generation complete!")
    print(f"Total records generated (for both directions): {total_records_generated}")
    print(f"Output file saved to: '{output_path}'")
    print("-------------------------------------------------")


if __name__ == '__main__':
    create_sft_data_with_alignment(INPUT_FILE_PATH, OUTPUT_DIR)