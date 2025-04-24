import torch
from transformers import GPT2Config

# Dummy config for testing
from cell2text_model.configuration import Cell2TextConfig
from cell2text_model.model import Cell2TextModel

# create the config
config = Cell2TextConfig(gpt_config={"add_cross_attention": True})


# now this works
model = Cell2TextModel(config)

# Set model to eval mode
model.eval()

# Dummy input: batch size 2
expression_tokens = torch.randint(0, 100, (2, 20))  # e.g., 20 tokens per sample
expression_token_lengths = torch.tensor([20, 18])   # sample lengths


decoder_input_ids = torch.ones((2, 1), dtype=torch.long) * 0  # BOS token

# Run forward pass
with torch.no_grad():
    outputs = model(
        expression_tokens=expression_tokens,
        expression_token_lengths=expression_token_lengths,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=torch.ones_like(decoder_input_ids)
    )

print("Forward pass successful. Output shape:", outputs.logits.shape)


# from transformers import GPT2Tokenizer

# # Load tokenizer for GPT2 (or your corresponding decoder model)
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# tokenizer.pad_token = tokenizer.eos_token  # needed for batch inputs
# tokenizer.bos_token = tokenizer.bos_token  # just to be explicit

# # Dummy input: batch size 1 for simplicity
# expression_tokens = torch.randint(0, 100, (1, 20))  # e.g., 20 tokens per sample
# expression_token_lengths = torch.tensor([20])

# # Run generation
# with torch.no_grad():
#     generated_text = model.generate_cell_description(
#         expression_tokens=expression_tokens,
#         expression_token_lengths=expression_token_lengths,
#         tokenizer=tokenizer,
#         device='cpu'
#     )

# print("Generated description:", generated_text)
