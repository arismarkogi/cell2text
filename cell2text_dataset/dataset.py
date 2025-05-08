import torch
import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class Cell2TextDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        """
        Dataset for cell expression data
        Args:
            data_path: Path to the parquet file containing cell data
            tokenizer:  tokenizer for text descriptions
        """
        self.data = pd.read_parquet(data_path)
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        
        # Get input_ids from the dataset (tokenized gene expression)
        expression_tokens = torch.tensor(sample["input_ids"])
        expression_token_length = torch.tensor(sample["length"])
        
        # Process text description if tokenizer is provided
        if self.tokenizer and "text_desc" in sample:
            label_inputs = self.tokenizer(
                sample["text_desc"],
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            print(label_inputs)
            labels = label_inputs.input_ids.squeeze(0)
            
            # Set padding tokens to -100 so they're ignored in loss calculation
            labels[labels == self.tokenizer.pad_token_id] = -100
        else:
            # If no tokenizer, just return empty tensors

            labels = torch.tensor([], dtype=torch.long)
        print(f"labels: {labels}")
        return {
            "expression_tokens": expression_tokens,
            "expression_token_length": expression_token_length,
            "labels": labels
        }

def collate_fn(batch):
    expression_tokens = [item["expression_tokens"] for item in batch]
    expression_token_lengths = torch.tensor([item["expression_token_length"] for item in batch], dtype=torch.long)

    labels_list = [item["labels"] for item in batch]
    has_labels = all(label.numel() > 0 for label in labels_list)

    # Pad expression tokens
    max_expr_len = max(len(tokens) for tokens in expression_tokens)
    padded_expression_tokens = torch.zeros((len(batch), max_expr_len), dtype=torch.long)
    for i, tokens in enumerate(expression_tokens):
        padded_expression_tokens[i, :len(tokens)] = tokens

    # Pad labels (if present)
    if has_labels:
        max_label_len = max(label.size(0) for label in labels_list)
        padded_labels = torch.full((len(batch), max_label_len), fill_value=-100, dtype=torch.long)
        for i, label in enumerate(labels_list):
            padded_labels[i, :label.size(0)] = label
    else:
        padded_labels = None

    return {
        "expression_tokens": padded_expression_tokens,
        "expression_token_lengths": expression_token_lengths,
        "labels": padded_labels
    }
