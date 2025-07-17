import json
import os
import random
from tqdm import tqdm

INPUT_FILE_PATH = "/Users/niklascanova/Desktop/Data/pledarigrond_export_json_rumantschgrischun.json"
OUTPUT_FILE_PATH = "SFT_Data/sft_dictionary_RG.jsonl"

MIN_WORDS_PER_PROMPT = 30
MAX_WORDS_PER_PROMPT = 40
PROMPT_TEMPLATE_RG_TO_DE = "Übersetze die folgende Liste von Rumantsch Grischun-Begriffen ins Deutsche:\n{romansh_list}"
PROMPT_TEMPLATE_DE_TO_RG = "Übersetze die folgende Liste deutscher Begriffe ins Rumatsch Grischun:\n{german_list}"


def format_list_randomly(items: list) -> str:
    """
    Formats a list of strings into a single string with a random list style.
    """
    styles = ['numeric_dot', 'numeric_paren', 'alpha_paren', 'dash', 'asterisk']
    chosen_style = random.choice(styles)
    
    formatted_lines = []
    for i, item in enumerate(items):
        if chosen_style == 'numeric_dot':
            prefix = f"{i + 1}."
        elif chosen_style == 'numeric_paren':
            prefix = f"{i + 1})"
        elif chosen_style == 'alpha_paren':
            prefix = f"{chr(97 + i)})"
        elif chosen_style == 'dash':
            prefix = "-"
        elif chosen_style == 'asterisk':
            prefix = "*"
        
        formatted_lines.append(f"{prefix} {item}")
        
    return "\n".join(formatted_lines)


def create_grouped_dictionary_sft_data(input_path: str, output_path: str):
    """
    Reads a JSON dictionary, groups word pairs into randomly sized and
    formatted lists, and creates a bidirectional SFT dataset.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at '{input_path}'")
        return

    print(f"Reading and cleaning dictionary file: {input_path}")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            dictionary_data = json.load(infile)
    except json.JSONDecodeError as e:
        print(f"Error: Could not parse the JSON file. It might be corrupted. Error: {e}")
        return

    valid_pairs = []
    for entry in tqdm(dictionary_data, desc="Extracting valid pairs"):
        if isinstance(entry, dict):
            romansh_term = entry.get("RStichwort", "").strip()
            german_term = entry.get("DStichwort", "").strip()
            if romansh_term and german_term:
                valid_pairs.append((romansh_term, german_term))

    if not valid_pairs:
        print("No valid word pairs found. Aborting.")
        return
        
    print(f"\nFound {len(valid_pairs)} valid word pairs.")

    random.shuffle(valid_pairs)
    final_sft_data = []
    
    cursor = 0
    with tqdm(total=len(valid_pairs), desc="Grouping into prompts") as pbar:
        while cursor < len(valid_pairs):
            chunk_size = random.randint(MIN_WORDS_PER_PROMPT, MAX_WORDS_PER_PROMPT)
            chunk = valid_pairs[cursor : cursor + chunk_size]
            if not chunk:
                break

            romansh_terms = [pair[0] for pair in chunk]
            german_terms = [pair[1] for pair in chunk]
            formatted_romansh_list = format_list_randomly(romansh_terms)
            prefixes = [line.split(' ', 1)[0] for line in formatted_romansh_list.split('\n')]
            formatted_german_list = "\n".join([f"{p} {term}" for p, term in zip(prefixes, german_terms)])

            prompt_rg_de = PROMPT_TEMPLATE_RG_TO_DE.format(romansh_list=formatted_romansh_list)
            sft_record_1 = {"Prompt": prompt_rg_de, "Answer": formatted_german_list}
            final_sft_data.append(sft_record_1)

            prompt_de_rg = PROMPT_TEMPLATE_DE_TO_RG.format(german_list=formatted_german_list)
            sft_record_2 = {"Prompt": prompt_de_rg, "Answer": formatted_romansh_list}
            final_sft_data.append(sft_record_2)

            cursor += len(chunk)
            pbar.update(len(chunk))

    random.shuffle(final_sft_data)
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"\nWriting {len(final_sft_data)} total records to output file...")
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for record in final_sft_data:
            outfile.write(json.dumps(record, ensure_ascii=False) + '\n')

    print("\n-------------------------------------------------")
    print("SFT data generation from dictionary complete!")
    print(f"Total word pairs processed: {len(valid_pairs)}")
    print(f"Total grouped prompts produced: {len(final_sft_data)}")
    print(f"Output file saved to: '{output_path}'")
    print("-------------------------------------------------")


if __name__ == '__main__':
    create_grouped_dictionary_sft_data(INPUT_FILE_PATH, OUTPUT_FILE_PATH)