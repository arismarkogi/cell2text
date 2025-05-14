import torch
import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
import sys
import pickle

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

        try:
            expression_ids = sample["input_ids"]

            expression_tokens = torch.tensor(expression_ids, dtype=torch.long)
            expression_token_length = torch.tensor(len(expression_ids), dtype=torch.long)
                
        except KeyError as e:
            raise KeyError(f"Missing gene expression field: {e}")

        # Use a structured format like: "Prompt: <X>. Description: <Y>"
        if "text_desc" not in sample or not sample["text_desc"]:
            raise ValueError(f"No 'text_desc' for index {idx}")
        
        target_text = sample["text_desc"]
        

        # Tokenize prompt and target separately
        target_enc = self.tokenizer(target_text, add_special_tokens=False)

        # Combine them into a single input
        input_ids = target_enc["input_ids"]
        attention_mask = [1] * len(input_ids)

        # Labels: ignore prompt, supervise target
        labels =  target_enc["input_ids"]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return {
            "expression_tokens": expression_tokens,
            "expression_token_length": expression_token_length,
            "text_input_ids": input_ids,
            "text_attention_mask": attention_mask,
            "labels": labels
        }
    
    def collate_fn(self, geneformer_pad_token_id):
        def collate(batch):
            expression_tokens = [item["expression_tokens"] for item in batch]
            expression_token_lengths = torch.tensor([item["expression_token_length"] for item in batch], dtype=torch.long)

            text_input_ids_list = [item["text_input_ids"] for item in batch]
            attention_masks_list = [item["text_attention_mask"] for item in batch]
            labels_list = [item["labels"] for item in batch]

            # Pad expression tokens with geneformer_pad_token_id
            max_expr_len = max(len(t) for t in expression_tokens)
            padded_expr = torch.full((len(batch), max_expr_len), fill_value=geneformer_pad_token_id, dtype=torch.long)
            for i, t in enumerate(expression_tokens):
                padded_expr[i, :len(t)] = t

            # Pad text inputs
            max_text_len = max(len(t) for t in text_input_ids_list)
            padded_input_ids = torch.full((len(batch), max_text_len), fill_value=self.tokenizer.pad_token_id, dtype=torch.long)
            padded_attention_mask = torch.zeros((len(batch), max_text_len), dtype=torch.long)
            padded_labels = torch.full((len(batch), max_text_len), fill_value=-100, dtype=torch.long)

            for i, (inp, attn, lbl) in enumerate(zip(text_input_ids_list, attention_masks_list, labels_list)):
                padded_input_ids[i, :len(inp)] = inp
                padded_attention_mask[i, :len(attn)] = attn
                padded_labels[i, :len(lbl)] = lbl

            return {
                "expression_tokens": padded_expr,
                "expression_token_lengths": expression_token_lengths,
                "text_input_ids": padded_input_ids,
                "text_attention_mask": padded_attention_mask,
                "labels": padded_labels
            }
    
        return collate

