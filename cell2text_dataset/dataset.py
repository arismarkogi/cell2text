#https://github.com/ColinFX/Prot2Text-V2 This code was really helpful

import torch
import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
import sys
import pickle
from datasets import load_from_disk
import random


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class Cell2TextDataset(Dataset):
    def __init__(self, data_path, tokenizer, geneformer_tokenizer=None, 
                 system_message="You are a scientific assistant specialized in analyzing single-cell gene expression data. Given the gene expression profile, describe the cell type and its characteristics clearly and concisely in professional language.",
                 placeholder_token='<|reserved_special_token_1|>', top_k=None, projector="qformer", num_latents=None,
                 sort_by_depth=False):
        """
        Dataset for cell expression data
        Args:
            data_path: Path to the parquet file containing cell data
            tokenizer: tokenizer for text descriptions (e.g., Llama tokenizer)
            geneformer_tokenizer: tokenizer for gene expression data
            system_message: System message for chat template
            placeholder_token: Token to use as placeholder for expression embeddings
            top_k: If specified, only use top k gene expression tokens
            sort_by_depth: If True, sort data by cl_depth in ascending order
        """
        self.data = load_from_disk(data_path)
        self.tokenizer = tokenizer
        self.geneformer_tokenizer = geneformer_tokenizer
        self.system_message = system_message
        self.placeholder_token = placeholder_token
        self.projector = projector
        self.top_k = top_k
        self.num_latents = num_latents
        
        # def assign_depth_bin(example):
        #     depth = example['cl_depth']
        #     if depth <= 2:
        #         return {'cl_depth_new': 0}
        #     elif depth <= 4:
        #         return {'cl_depth_new': 1}
        #     elif depth <= 6:
        #         return {'cl_depth_new': 2}
        #     else:
        #         return {'cl_depth_new': 3}

        # # Sort directly by the new int column
        # if sort_by_depth:
        #     if 'cl_depth' in self.data.column_names:
        #         self.data = self.data.map(assign_depth_bin, desc="Assigning cl_depth bins")
        #         self.data = self.data.sort('cl_depth_new')
        #         print("Dataset sorted by cl_depth_new (0=shallow, ..., 3=deep)")
        #     else:
        #         print("Warning: cl_depth_new not found, skipping sort")



        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]

        # Get expression data
        try:
            expression_ids = sample["input_ids"]
            expression_tokens = torch.tensor(expression_ids, dtype=torch.long)
            expression_token_length = sample["length"]

        except KeyError as e:
            raise KeyError(f"Missing gene expression field: {e}")

        # Get target description
        if "natural_desc" not in sample or not sample["natural_desc"]:
            raise ValueError(f"No 'natural_desc' for index {idx}")
        
        description = sample["natural_desc"]
        
        # Create chat template with placeholder tokens
        if (self.projector == "mlp" or self.projector == "qformer") and self.top_k is not None:
                placeholder_length = min(len(expression_ids), self.top_k)
        
        elif self.projector == "perceiver" and self.num_latents is not None:
            placeholder_length = self.num_latents
        
        placeholder_string = self.placeholder_token * placeholder_length
        
        user_message = f"Gene expression embeddings: {placeholder_string}"
        
        prompt_conversation = [
            {"role": "system", "content": self.system_message}, 
            {"role": "user", "content": user_message}
        ]
        
        # Apply chat template to create prompt
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt_conversation,
            add_generation_prompt=True, 
            tokenize=True, 
            padding=False, 
            return_tensors="pt"
        )
        
        # Tokenize description with EOS token
        description_ids = self.tokenizer(
            [description + self.tokenizer.eos_token],
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors="pt"
        )["input_ids"]

        # Return data in format similar to protein dataset
        # Keep tensors as (1, seq_length) for batch formation
        return {
            "expression_tokens": expression_tokens.unsqueeze(0),  # (1, expr_len)
            "expression_token_length": expression_token_length,
            "prompt_input_ids": prompt_ids,  # (1, prompt_len)
            "description_input_ids": description_ids,  # (1, desc_len)
        }
    
    def collate_fn(self, geneformer_pad_token_id=0, mode="train"):
        """
        Create collate function with simplified right padding for all sequences
        Args:
            geneformer_pad_token_id: Pad token ID for gene expression data
            mode: "train" or "inference"
        """
        def collate(batch):
            # Extract components from batch
            expression_tokens = [item["expression_tokens"][0] for item in batch]  # Remove extra dimension
            expression_lengths = torch.tensor([item["expression_token_length"] for item in batch], dtype=torch.long)
            prompt_input_ids = [item["prompt_input_ids"][0] for item in batch]
            description_input_ids = [item["description_input_ids"][0] for item in batch]

            # Pad expression tokens (right padding)
            max_expr_len = max(len(t) for t in expression_tokens)
            padded_expr = torch.full((len(batch), max_expr_len), fill_value=geneformer_pad_token_id, dtype=torch.long)
            padded_expr_mask = torch.zeros((len(batch), max_expr_len), dtype=torch.long)
            
            for i, t in enumerate(expression_tokens):
                padded_expr[i, :len(t)] = t
                padded_expr_mask[i, :len(t)] = 1

            # Pad prompts (right padding)
            max_prompt_len = max(len(p) for p in prompt_input_ids)
            padded_prompt_ids = torch.full((len(batch), max_prompt_len), fill_value=self.tokenizer.pad_token_id, dtype=torch.long)
            padded_prompt_mask = torch.zeros((len(batch), max_prompt_len), dtype=torch.long)
            
            for i, p in enumerate(prompt_input_ids):
                padded_prompt_ids[i, :len(p)] = p
                padded_prompt_mask[i, :len(p)] = 1

            # Pad descriptions (right padding)
            max_desc_len = max(len(d) for d in description_input_ids)
            padded_desc_ids = torch.full((len(batch), max_desc_len), fill_value=self.tokenizer.pad_token_id, dtype=torch.long)
            padded_desc_mask = torch.zeros((len(batch), max_desc_len), dtype=torch.long)
            padded_labels = torch.full((len(batch), max_desc_len), fill_value=-100, dtype=torch.long)
            
            for i, d in enumerate(description_input_ids):
                padded_desc_ids[i, :len(d)] = d
                padded_desc_mask[i, :len(d)] = 1
                padded_labels[i, :len(d)] = d  # Labels same as description for training

            # Combine based on mode
            if mode == "train":
                # Concatenate prompt and description for training
                combined_input_ids = torch.cat([padded_prompt_ids, padded_desc_ids], dim=1)
                combined_attention_mask = torch.cat([padded_prompt_mask, padded_desc_mask], dim=1)
                combined_labels = torch.cat([
                    torch.full_like(padded_prompt_ids, fill_value=-100),  # Ignore prompt in loss
                    padded_labels
                ], dim=1)
                
                return {
                    "expression_tokens": padded_expr,
                    "expression_attention_mask": padded_expr_mask,
                    "expression_token_lengths": expression_lengths,
                    "input_ids": combined_input_ids,
                    "attention_mask": combined_attention_mask,
                    "labels": combined_labels
                }
            
            elif mode == "inference":
                # Only return prompt for generation
                return {
                    "expression_tokens": padded_expr,
                    "expression_attention_mask": padded_expr_mask,
                    "expression_token_lengths": expression_lengths,
                    "input_ids": padded_prompt_ids,
                    "attention_mask": padded_prompt_mask,
                    "description_input_ids": padded_desc_ids,  # For evaluation
                }
            
            else:
                raise ValueError(f"Invalid mode: {mode}")

        return collate

class SanityDataset(Dataset):
    """Wrapper to create a small subset of data for sanity check"""
    
    def __init__(self, full_dataset, num_samples=8, seed=42):
        self.full_dataset = full_dataset
        self.num_samples = min(num_samples, len(full_dataset))
        
        # Set seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        # Select random indices
        self.indices = random.sample(range(len(full_dataset)), self.num_samples)
        print(f"Selected {self.num_samples} samples for sanity check: {self.indices}")
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.full_dataset[self.indices[idx]]