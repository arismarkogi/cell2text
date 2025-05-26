import torch
import pandas as pd
from transformers import AutoTokenizer
from cell2text_model.configuration import Cell2TextConfig
from cell2text_model.model import Cell2TextModel
from datasets import load_from_disk

# Load dataset from disk
dataset = load_from_disk("/home/arismarkog/Desktop/datasets/final_dataset_with_descriptions")
# Convert to pandas DataFrame if needed
df = dataset.to_pandas()

# Use only a small subset of the data
sample_size = 1 # You can adjust this number based on your needs
df_sample = df.head(sample_size)

# Process the sampled data
expression_tokens_list = df_sample["input_ids"].tolist()

# Convert to padded tensor
max_len = max(len(seq) for seq in expression_tokens_list)
expression_tokens = torch.zeros((len(expression_tokens_list), max_len), dtype=torch.long)
for i, seq in enumerate(expression_tokens_list):
    expression_tokens[i, :len(seq)] = torch.tensor(seq)

print("Expression tokens shape:", expression_tokens.shape)

expression_token_lengths = df_sample["length"].tolist()
expression_token_lengths = torch.tensor(expression_token_lengths, dtype=torch.long)
print("Expression token lengths:", expression_token_lengths)

# Initialize tokenizer for Llama
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# Create configuration with Llama parameters
config = Cell2TextConfig(
    geneformer_path="/home/arismarkog/Desktop/cell2text/Geneformer"
)

# Initialize model with our new implementation
model = Cell2TextModel(config)
model.warm_up()
model.eval()

# Test 1: Forward pass with tokenized input
print(f"Running forward pass on {sample_size} samples...")

# Create tokenized input (using parameter names that match the updated forward method)
input_ids = torch.ones((expression_tokens.shape[0], 1), dtype=torch.long) * tokenizer.bos_token_id
attention_mask = torch.ones_like(input_ids)

with torch.no_grad():
    outputs = model(
        expression_tokens=expression_tokens,
        expression_token_lengths=expression_token_lengths,
        input_ids=input_ids,  # Changed from text_input_ids
        attention_mask=attention_mask  # Changed from text_attention_mask
    )

print(f"Forward pass successful. Output logits shape: {outputs.logits.shape if hasattr(outputs, 'logits') else 'No logits attribute'}")
print(f"Processed {sample_size} samples out of {len(df)} total samples")

# Test 2: Generate description without prompt (using BOS token only)
print("\nTest 2: Generating description for first sample (no prompt)...")
description = model.generate_cell_description(
    expression_tokens=expression_tokens[0:1],
    expression_token_lengths=expression_token_lengths[0:1],
    max_new_tokens=50,
    num_beams=2
)
print(f"Generated description (no prompt): {description}")

# Test 3: Generate description with a custom tokenized prompt
print("\nTest 3: Generating description with custom prompt...")
prompt_text = "Describe this cell: "
prompt_tokens = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
prompt_input_ids = prompt_tokens.input_ids
prompt_attention_mask = prompt_tokens.attention_mask

description_with_prompt = model.generate_cell_description(
    expression_tokens=expression_tokens[0:1],
    expression_token_lengths=expression_token_lengths[0:1],
    inputs=prompt_input_ids,  # Pass tokenized prompt
    attention_mask=prompt_attention_mask,
    max_new_tokens=50,
    num_beams=2
)
print(f"Generated description (with prompt): {description_with_prompt}")

