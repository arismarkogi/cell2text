import torch
import pandas as pd
from transformers import AutoTokenizer
from cell2text_model.configuration import Cell2TextConfig
from cell2text_model.model import Cell2TextModel  # Use our new function

# Load .parquet
df = pd.read_parquet("/home/arismarkog/Desktop/datasets/processed.parquet")

# Use only a small subset of the data
sample_size = 3  # You can adjust this number based on your needs
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

# BOS token for decoder input
text_input_ids = torch.ones((expression_tokens.shape[0], 1), dtype=torch.long) * tokenizer.bos_token_id
text_attention_mask = torch.ones_like(text_input_ids)

print(f"Running forward pass on {sample_size} samples...")

# Run forward pass
with torch.no_grad():
    outputs = model(
        expression_tokens=expression_tokens,
        expression_token_lengths=expression_token_lengths,
        text_input_ids=text_input_ids,
        text_attention_mask=text_attention_mask
    )

print(f"Forward pass successful. Output shape: {outputs}")
print(f"Processed {sample_size} samples out of {len(df)} total samples")

# # Generate a description for the first sample
# print("\nGenerating description for first sample...")
# description = model.generate_cell_description(
#     expression_tokens=expression_tokens[0:1],
#     expression_token_lengths=expression_token_lengths[0:1],
#     max_new_tokens=50,  # Adjust as needed
#     num_beams=2     # Use a smaller beam size for testing
# )

# print(f"Generated description: {description}")