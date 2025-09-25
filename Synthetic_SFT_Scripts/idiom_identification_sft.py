import gzip
import json
import nltk
import os
import random
from tqdm import tqdm


INPUT_FILE_PATH = "/Users/niklascanova/Desktop/SwissAI/romansh_data/monolingual/FMR-La_Quotidiana_2008ff.jsonl.gz"
OUTPUT_FILE_PATH = "SFT_Data/sft_idiom_identification.jsonl"
SAMPLES_PER_IDIOM = 3000
MIN_SENTENCE_WORDS = 10
PROMPT_TEMPLATE = "Sag mir in welche Idiom der folgende Satz is: {romansh_sentence}"


def download_nltk_resources():
    """
    Downloads the NLTK resources required for sentence tokenization ('punkt')
    if they are not already present.
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("NLTK 'punkt' tokenizer not found. Downloading...")
        nltk.download('punkt')
        print("Download complete.")

def create_idiom_sft_data(input_path: str, output_path: str):
    """
    Reads a gzipped JSONL file containing Romansh text, samples sentences
    from each idiom, and creates a new JSONL file for an idiom identification task.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at '{input_path}'")
        return

    download_nltk_resources()

    idiom_sentences = {
        "Rumantsch Grischun": [],
        "Sursilvan": [],
        "Surmiran": [],
        "Sutsilvan": [],
        "Puter": [],
        "Vallader": [],
    }

    print(f"Processing input file: {input_path}")
    
    with gzip.open(input_path, 'rt', encoding='utf-8') as infile:
        for line in tqdm(infile, desc="Reading and tokenizing"):
            try:
                data = json.loads(line)
                
                text_content = data.get("text")
                idiom = data.get("idiom")

                if text_content and idiom and idiom in idiom_sentences:
                    sentences = nltk.sent_tokenize(text_content)
                    
                    for sentence in sentences:
                        cleaned_sentence = sentence.strip()
                        if len(cleaned_sentence.split()) >= MIN_SENTENCE_WORDS:
                            idiom_sentences[idiom].append(cleaned_sentence)

            except json.JSONDecodeError as e:
                continue
    
    print("\nFinished processing file. Sentence counts per idiom:")
    for idiom, sentences in idiom_sentences.items():
        print(f"- {idiom}: {len(sentences)} sentences")
    final_sft_data = []
    print(f"\nSampling up to {SAMPLES_PER_IDIOM} examples per idiom...")

    for idiom, sentences in idiom_sentences.items():

        num_to_sample = min(SAMPLES_PER_IDIOM, len(sentences))
        
        if num_to_sample == 0:
            print(f"Warning: No valid sentences found for {idiom}. Skipping.")
            continue

        sampled_sentences = random.sample(sentences, num_to_sample)
        
        for sentence in sampled_sentences:
            prompt = PROMPT_TEMPLATE.format(romansh_sentence=sentence)
            sft_record = {"Prompt": prompt, "Answer": idiom}
            final_sft_data.append(sft_record)

    random.shuffle(final_sft_data)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"\nWriting {len(final_sft_data)} total records to output file...")
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for record in final_sft_data:
            outfile.write(json.dumps(record, ensure_ascii=False) + '\n')

    print("\n-------------------------------------------------")
    print("SFT data generation complete!")
    print(f"Total records produced: {len(final_sft_data)}")
    print(f"Output file saved to: '{output_path}'")
    print("-------------------------------------------------")


if __name__ == '__main__':
    create_idiom_sft_data(INPUT_FILE_PATH, OUTPUT_FILE_PATH)
