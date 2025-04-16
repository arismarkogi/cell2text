import torch
from transformers import GPT2Config
from cell2text_model.model import Cell2TextModel  # Adjust the path if needed

# Dummy config for testing
from cell2text_model.configuration import Cell2TextConfig
from cell2text_model.model import Cell2TextModel

# create the config
config = Cell2TextConfig(gpt_config={"add_cross_attention": True})


# now this works
model = Cell2TextModel(config)


model = Cell2TextModel(config)

# Set model to eval mode
model.eval()

# Dummy input: batch size 2
expression_tokens = torch.randint(0, 100, (2, 20))  # e.g., 20 tokens per sample
expression_token_lengths = torch.tensor([20, 18])   # sample lengths

input_tokens = torch.randint(0, 100, (2, 30))        # PubMedBERT dummy input
input_token_lengths = torch.tensor([30, 28])         # lengths

decoder_input_ids = torch.ones((2, 1), dtype=torch.long) * 0  # BOS token

# Run forward pass
with torch.no_grad():
    outputs = model(
        expression_tokens=expression_tokens,
        expression_token_lengths=expression_token_lengths,
        input_tokens=input_tokens,
        input_token_lengths=input_token_lengths,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=torch.ones_like(decoder_input_ids)
    )

print("Forward pass successful. Output shape:", outputs.logits.shape)
