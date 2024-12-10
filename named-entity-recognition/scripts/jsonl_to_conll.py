import json
import os
from transformers import AutoTokenizer
import pandas as pd

# Step 1: Load .jsonl Data
def load_jsonl_data(json_path):
    texts = []
    entities = []
    with open(json_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            texts.append(data['text'])
            entities.append([(int(ent['start']), int(ent['end']), ent['label']) for ent in data['spans']])
    return texts, entities


# Step 2: Tokenize and Align Labels
tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')

def tokenize_and_align_labels(texts, entities, vocab, max_len=512):
    tokenized_inputs = tokenizer(texts, padding=False, truncation=True, max_length=max_len, return_offsets_mapping=True)

    labels = []
    for i, (text, entity_list) in enumerate(zip(texts, entities)):
        offsets = tokenized_inputs['offset_mapping'][i]
        label_list = ['O'] * len(offsets)  # Initialize all tokens as 'O' (no entity)

        # Process each entity
        for start, end, label in entity_list:
            entity_started = False  # Flag to indicate if the entity has started
            for j, (start_offset, end_offset) in enumerate(offsets):
                # Check if the token's offset overlaps with the entity span
                if start_offset < end and end_offset > start:
                    if not entity_started:
                        # First token of the entity should be labeled as 'B-PK' (beginning of entity)
                        label_list[j] = 'B-PK'
                        entity_started = True
                    else:
                        # Subsequent tokens of the entity should be labeled as 'I-PK' (inside entity)
                        label_list[j] = 'I-PK'

        labels.append(label_list)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# Step 3: Convert Tokenized Data to .tsv
def create_tsv(tokenized_data, out_path, filename):
    tsv_path = os.path.join(out_path, f"{filename}.tsv")

    rows = []
    for idx, (input_ids, labels) in enumerate(zip(tokenized_data['input_ids'], tokenized_data['labels'])):
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        # Write token-label pairs
        for token, label in zip(tokens, labels):
            rows.append({
                "uid": str(idx),
                "token": token,
                "label": label
            })
    
    # Convert rows to DataFrame and save as .tsv
    df = pd.DataFrame(rows)
    df.to_csv(tsv_path, sep='\t', index=False, header=False)
    print(f"TSV file saved to {tsv_path}")


# Step 4: Convert .tsv to .conll
def tsv_to_conll(tsv_path, conll_path):
    tsv_data = pd.read_csv(tsv_path, sep='\t', names=["uid", "token", "label"])
    
    with open(conll_path, 'w', encoding='utf-8') as conll_file:
        for _, row in tsv_data.iterrows():
            token = row['token']
            label = row['label']
            conll_file.write(f"{token} {label}\n")
        
        conll_file.write("\n")   # Add a blank line between sentences


# Step 5: Overall Data Processing
def process_data(jsonl_path, tsv_out_path, conll_out_path):
    filename = os.path.splitext(os.path.basename(jsonl_path))[0]  # Get the base name without extension

    # Load data from jsonl file
    texts, entities = load_jsonl_data(jsonl_path)

    # Get the tokenizer's vocabulary (this should be passed to handle OOV words)
    vocab = tokenizer.get_vocab()

    # Tokenize and align labels, passing the vocab to handle OOV
    tokenized_data = tokenize_and_align_labels(texts, entities, vocab)

    # Convert to TSV format
    create_tsv(tokenized_data, tsv_out_path, filename)  # Use filename for the tsv file

    # Convert to CoNLL format
    tsv_to_conll(os.path.join(tsv_out_path, f"{filename}.tsv"), os.path.join(conll_out_path, f"{filename}.conll"))


# Step 6: Main Execution to Process All JSONL Files
if __name__ == "__main__":
    jsonl_folder = './pk_dataset'  # Folder with .jsonl files
    tsv_out_path = './lara_preprocessed_tsv'  # Path to save the TSV files
    conll_out_path = './lara_preprocessed_conll'  # Path to save the CoNLL files

    # Iterate over all files in the folder
    for jsonl_file in os.listdir(jsonl_folder):
        if jsonl_file.endswith('.jsonl'):
            jsonl_path = os.path.join(jsonl_folder, jsonl_file)
            process_data(jsonl_path, tsv_out_path, conll_out_path)
