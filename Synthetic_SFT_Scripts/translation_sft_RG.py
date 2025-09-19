import gzip
import json
import nltk
import os
import torch
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

INPUT_FILE_PATH = "/Users/niklascanova/Desktop/SwissAI/romansh_data/polylingual/Aligned/swiss_constitutions.jsonl.gz" # Path to your new multi-language file
OUTPUT_FILE_PATH = "SFT_Data/sft_multilang_to_grischun.jsonl"
TARGET_IDIOM = "Rumantsch Grischun"
EMBEDDING_MODEL = 'paraphrase-multilingual-mpnet-base-v2'
SIMILARITY_THRESHOLD = 0.65
MAX_LINES_TO_PROCESS = None 
MAX_WORD_COUNT_RATIO = 1.3

LANG_NAME_MAP = {
    'de': 'Deutschen',
    'en': 'Englischen',
    'fr': 'Französischen',
    'it': 'Italienischen'
}

def clean_sentence(sentence: str) -> str:
    """
    Applies specific cleaning rules to a sentence.
    """
    cleaned = re.sub(r'[\t\n]+', ' ', sentence)
    cleaned = re.sub(r' {2,}', ' ', cleaned)
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
    if re.fullmatch(r'Art\.\s*[\d\s\*]*', sentence.strip(), re.IGNORECASE):
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
    Parses a single JSON object from a line, handling two possible formats.
    """
    try:
        data = json.loads(line)
        text = data.get("text", "")
        return data, text
    except json.JSONDecodeError:
        try:
            meta_part, text_part = line.split('|', 1)
            metadata = json.loads(meta_part.strip())
            return metadata, text_part.strip()
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Could not parse line in any known format. Skipping. Error: {e}\nLine: {line[:150]}...")
            return None, None


def create_sft_data_with_alignment(input_path: str, output_path: str):
    """
    Reads a multi-language, gzipped JSONL file, aligns each source language
    against Romansh, and creates a new JSONL file for SFT.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at '{input_path}'")
        return

    print("Starting SFT data generation with multi-language alignment...")
    download_nltk_resources()

    print(f"Loading sentence embedding model: '{EMBEDDING_MODEL}'...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Model loaded.")

    total_pairs_generated = 0
    
    with gzip.open(input_path, 'rt', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        lines = infile.readlines()
        print(f"Read {len(lines)} total lines from the input file.")

        if MAX_LINES_TO_PROCESS is not None:
            lines = lines[:MAX_LINES_TO_PROCESS]
            print(f"Processing only the first {len(lines)} lines as requested.")

        num_languages = 5 
        for i in range(0, len(lines), num_languages):
            chunk = lines[i:i+num_languages]
            if len(chunk) < num_languages:
                print(f"Warning: Incomplete language chunk at the end of the file. Skipping.")
                continue
            
            print(f"\n--- Processing document chunk starting at line {i+1} ---")

            texts = {}
            for line in chunk:
                meta, text = parse_line(line)
                if meta and text:
                    lang_code = meta.get('language')
                    if lang_code:
                        texts[lang_code] = text
            
            if 'roh' not in texts:
                print("Warning: Romansh ('roh') text not found in this chunk. Skipping.")
                continue
            
            romansh_text = texts.pop('roh')
            romansh_sentences = [clean_sentence(s) for s in nltk.sent_tokenize(romansh_text, language='german')]
            if not romansh_sentences:
                print("Warning: No valid sentences found in Romansh text. Skipping chunk.")
                continue
            
            romansh_embeddings = model.encode(romansh_sentences, convert_to_tensor=True, show_progress_bar=False)

            for lang_code, source_text in texts.items():
                if lang_code not in LANG_NAME_MAP:
                    continue

                print(f"Aligning {LANG_NAME_MAP[lang_code]} against {TARGET_IDIOM}...")
                
                source_sentences = [clean_sentence(s) for s in nltk.sent_tokenize(source_text, language='english')]
                if not source_sentences:
                    continue

                source_embeddings = model.encode(source_sentences, convert_to_tensor=True, show_progress_bar=False)

                cosine_scores = util.cos_sim(source_embeddings, romansh_embeddings)
                
                source_scores, source_indices = torch.max(cosine_scores, dim=1)
                target_scores, target_indices = torch.max(cosine_scores, dim=0)

                for j in tqdm(range(len(source_sentences)), desc=f"Finding pairs for '{lang_code}'", leave=False):
                    best_target_idx = source_indices[j].item()
                    if target_indices[best_target_idx].item() == j:
                        score = source_scores[j].item()
                        if score >= SIMILARITY_THRESHOLD:
                            source_sent = source_sentences[j]
                            target_sent = romansh_sentences[best_target_idx]
                            
                            source_word_count = len(source_sent.split())
                            target_word_count = len(target_sent.split())

                            if source_word_count == 0 or target_word_count == 0:
                                continue

                            ratio = source_word_count / target_word_count
                            if ratio > MAX_WORD_COUNT_RATIO or ratio < (1 / MAX_WORD_COUNT_RATIO):
                                continue
                            if is_valid_sentence(source_sent) and is_valid_sentence(target_sent):
                                lang_name = LANG_NAME_MAP[lang_code]
                                prompt = f"Übersetze diesen Text aus dem {lang_name} ins {TARGET_IDIOM}: {source_sent}"
                                answer = target_sent
                                sft_record = {"Prompt": prompt, "Answer": answer}
                                outfile.write(json.dumps(sft_record, ensure_ascii=False) + '\n')
                                total_pairs_generated += 1

    print("\n-------------------------------------------------")
    print("SFT data generation complete!")
    print(f"Total Question/Answer pairs produced: {total_pairs_generated}")
    print(f"Output file saved to: '{output_path}'")
    print("-------------------------------------------------")


if __name__ == '__main__':
    create_sft_data_with_alignment(INPUT_FILE_PATH, OUTPUT_FILE_PATH)
