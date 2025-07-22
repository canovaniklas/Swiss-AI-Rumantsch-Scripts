import openai
import json
import os
import re
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

INPUT_FILE_PATH = "SFT_Data/sft_multilang_to_grischun.jsonl"
OUTPUT_FILE_PATH = "SFT_Data/sft_grischun_quality_filtered.jsonl"


API_KEY = "YOUR_API_KEY"  # Replace with your actual API key
BASE_URL = "https://api.swissai.cscs.ch/v1"
MODEL_NAME = "Qwen/Qwen3-32B" 
MAX_WORKERS = 20


QUALITY_SCORE_THRESHOLD = 7
MAX_LINES_TO_PROCESS = None 

SYSTEM_PROMPT = """
You are a meticulous translation quality evaluator. Your task is to provide a single integer score.

First, check for automatic failures. Assign a score of **0** if any of the following are true:
- The Source and Translation are in the same language.
- The Translation is not in the correct target language implied by the prompt.
- The Translation contains significantly more or less information than the Source.

If none of the above apply, rate the translation on a scale from 1 to 10 based on these criteria:
- **Accuracy (1-10):** Does the translation correctly convey the meaning?
- **Fluency (1-10):** Does the translation sound natural and grammatically correct?

Your final response MUST be a single integer and nothing else. Do not provide explanations. Just the number. For example: 8
"""

def get_quality_score(client, source_sentence: str, translated_sentence: str) -> int:
    """
    Sends a translation pair to the AI model and returns a numerical quality score.
    """
    user_prompt = f"Source: \"{source_sentence}\"\nTranslation: \"{translated_sentence}\""
    
    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0, 
            max_tokens=10,   
            stream=False  
        )
        
        content = res.choices[0].message.content
        match = re.search(r'\d+', content)
        
        if match:
            return int(match.group(0))
        else:
            print(f"\nWarning: Could not parse score from model response: '{content}'. Assigning score of 0.")
            return 0
            
    except openai.APIError as e:
        print(f"\nAn API error occurred: {e}. Skipping this entry.")
        return 0
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}. Skipping this entry.")
        return 0

def process_line(line: str, client):
    """
    Processes a single line from the input file. This function is run in a separate thread.
    It parses the JSON, calls the API for a score, and returns the result.
    """
    try:
        data = json.loads(line)
        prompt = data.get("Prompt", "")
        answer = data.get("Answer", "")

        source_match = re.search(r':\s*(.*)', prompt)
        if not source_match:
            return "parse_error", line

        source_sentence = source_match.group(1).strip()
        translated_sentence = answer
        score = get_quality_score(client, source_sentence, translated_sentence)
        return score, line
    except json.JSONDecodeError:
        return "json_error", line
    except Exception as e:
        return f"unknown_error: {e}", line


def filter_translations_by_quality(input_path: str, output_path: str):
    """
    Reads a JSONL file, gets a quality score for each entry from an AI model in parallel,
    and writes only the high-quality entries to a new file.
    """
    if API_KEY == "[YOUR_API_KEY]" or not API_KEY:
        print("Error: API_KEY is not set. Please replace '[YOUR_API_KEY]' with your actual key.")
        return

    if not os.path.exists(input_path):
        print(f"Error: Input file not found at '{input_path}'")
        return

    print("Initializing Swiss AI client...")
    client = openai.Client(api_key=API_KEY, base_url=BASE_URL)
    
    with open(input_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    
    if MAX_LINES_TO_PROCESS is not None:
        lines = lines[:MAX_LINES_TO_PROCESS]
        print(f"Processing a maximum of {len(lines)} lines.")

    kept_count = 0
    discarded_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        future_to_line = {executor.submit(process_line, line, client): line for line in lines}
        
        for future in tqdm(as_completed(future_to_line), total=len(lines), desc="Evaluating in parallel"):
            try:
                score, original_line = future.result()
                
                if isinstance(score, int) and score >= QUALITY_SCORE_THRESHOLD:
                    outfile.write(original_line)
                    kept_count += 1
                else:
                    discarded_count += 1
                    if isinstance(score, str):
                         print(f"\nWarning: Discarding line due to error: {score}. Line: {original_line[:100]}")

            except Exception as exc:
                print(f'\nLine generated an exception: {exc}')
                discarded_count += 1

    print("\n-------------------------------------------------")
    print("Quality filtering complete!")
    print(f"Total entries evaluated: {len(lines)}")
    print(f"Entries kept (score >= {QUALITY_SCORE_THRESHOLD}): {kept_count}")
    print(f"Entries discarded (score < {QUALITY_SCORE_THRESHOLD} or errored): {discarded_count}")
    print(f"High-quality data saved to: '{output_path}'")
    print("-------------------------------------------------")


if __name__ == '__main__':
    filter_translations_by_quality(INPUT_FILE_PATH, OUTPUT_FILE_PATH)
