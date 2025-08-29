from datasets import load_dataset, DatasetDict

input_jsonl_file = '/Users/niklascanova/Desktop/SwissAI/Scripts/SFT_Data/SFT_Romansh.jsonl'
output_hf_dataset_path = 'SFT_Romansh' 

print(f"Loading data from '{input_jsonl_file}'...")
dataset = load_dataset('json', data_files=input_jsonl_file, split='train')

print("Renaming columns: 'Prompt' -> 'input', 'Answer' -> 'output'")
dataset = dataset.rename_column("Prompt", "input")
dataset = dataset.rename_column("Answer", "output")
final_dataset = DatasetDict({
    "train": dataset
})

print(f"Saving dataset to disk at '{output_hf_dataset_path}'...")
final_dataset.save_to_disk(output_hf_dataset_path)

print("\n Dataset is now ready for the standardization script.")
print(f"Run the standardization script on the '{output_hf_dataset_path}' directory.")