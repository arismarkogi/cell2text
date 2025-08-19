import torch
import torch.nn as nn
from torch.utils.data import Dataset
import random
import numpy as np
from datasets import load_from_disk


class CellPuzzlesDataset(Dataset):
    def __init__(self, data_path, tokenizer, geneformer_tokenizer=None,
                 placeholder_token='<|reserved_special_token_1|>', projector="qformer", 
                 num_query_tokens=32,  pad_cells=True):
        """
        Dataset for CellPuzzles multi-cell annotation task
        
        Args:
            data_path: Path to converted CellPuzzles dataset 
            tokenizer: LLaMA tokenizer object (not string!)
            geneformer_tokenizer: Not used, kept for compatibility
            placeholder_token: Token to use as placeholder for cell embeddings
            projector: Type of projector ("qformer", "mlp", "perceiver")
            num_query_tokens: Number of output tokens from projector (32 for qformer)
            pad_cells: Whether to pad sequences to max_cells
        """
        self.data = load_from_disk(data_path)
        self.tokenizer = tokenizer
        self.placeholder_token = placeholder_token
        self.projector = projector
        self.num_query_tokens = num_query_tokens
        self.pad_cells = pad_cells
        
        # Validate tokenizer
        if not hasattr(self.tokenizer, 'apply_chat_template'):
            raise ValueError("tokenizer must be a tokenizer object with apply_chat_template method, not a string")
        if not hasattr(self.tokenizer, 'pad_token_id'):
            raise ValueError("tokenizer must have pad_token_id attribute")
        if not hasattr(self.tokenizer, 'eos_token'):
            raise ValueError("tokenizer must have eos_token attribute")
        
        print(f"Loaded {len(self.data)} examples")
        
    def _replace_gene_lists_with_placeholders(self, user_msg, num_cells):
        """
        Replace gene expression lists in user message with placeholder tokens.
        
        Args:
            user_msg: Original user message with gene lists
            num_cells: Number of cells to replace
            
        Returns:
            Modified user message with gene lists replaced by placeholders
        """
        import re
        
        # Split the message into parts: context + cells + matching instruction
        parts = user_msg.split('\n\n')
        context_part = parts[0]  # "Context: The cell is from..."
        
        # Find the matching instruction part
        match_part = None
        for part in parts:
            if part.startswith("Match the cells"):
                match_part = part
                break
        
        if match_part is None:
            # Fallback: look for the match instruction in the last part
            match_part = parts[-1] if "Match the cells" in parts[-1] else ""
        
        # Create placeholder cells
        placeholder_cells = []
        for i in range(num_cells):
            cell_placeholders = self.placeholder_token * self.num_query_tokens
            placeholder_cells.append(f"Cell {i+1}: {cell_placeholders}")
        
        # Reconstruct the message
        modified_msg = context_part + "\n\n" + "\n".join(placeholder_cells)
        if match_part:
            modified_msg += "\n\n" + match_part
        
        return modified_msg
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Get multiple cell expression data
        input_ids_batch = sample["input_ids_batch"]  # List of tokenized cell expressions
        system_msg = sample["system_msg"]
        user_msg = sample["user_msg"] 
        assistant_msg = sample["assistant_msg"]
        
        # Limit number of cells and pad if needed
        num_cells = len(input_ids_batch)
        cell_expressions = input_ids_batch[:num_cells]
        
        # Convert to tensors and get lengths
        cell_tokens_list = []
        cell_lengths = []
        
        for cell_expr in cell_expressions:
            if len(cell_expr) > 0:  # Skip empty cells
                tokens = torch.tensor(cell_expr, dtype=torch.long)
                cell_tokens_list.append(tokens)
                cell_lengths.append(len(cell_expr))
        
        
        # Replace gene lists with placeholders in the original user message
        user_message = self._replace_gene_lists_with_placeholders(user_msg, len(cell_tokens_list))
        
        # Create conversation
        conversation = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_message}
        ]

        print(conversation)
        
        # Tokenize prompt
        prompt_ids = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            padding=False,
            return_tensors="pt"
        )
        
        # Tokenize target response
        target_ids = self.tokenizer(
            [assistant_msg + self.tokenizer.eos_token],
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors="pt"
        )["input_ids"]
        
        return {
            "cell_expressions": cell_tokens_list,  # List of tensors
            "cell_lengths": torch.tensor(cell_lengths, dtype=torch.long),
            "num_cells": num_cells,
            "prompt_input_ids": prompt_ids,
            "target_input_ids": target_ids,
            "raw_user_msg": user_msg,  # Original for reference
            "raw_assistant_msg": assistant_msg
        }
    
    def collate_fn(self, geneformer_pad_token_id=0, mode="train"):
        """
        Collate function for batching multi-cell data
        """
        def collate(batch):
            batch_size = len(batch)
            
            # Find max cells and max expression length across batch
            max_cells_in_batch = max(len(item["cell_expressions"]) for item in batch)
            max_expr_len = 0
            for item in batch:
                for cell_expr in item["cell_expressions"]:
                    max_expr_len = max(max_expr_len, len(cell_expr))
            
            # Prepare cell expression tensors: (batch_size, max_cells, max_expr_len)
            batch_cell_expressions = torch.full(
                (batch_size, max_cells_in_batch, max_expr_len), 
                fill_value=geneformer_pad_token_id, 
                dtype=torch.long
            )
            batch_cell_masks = torch.zeros(
                (batch_size, max_cells_in_batch, max_expr_len), 
                dtype=torch.long
            )
            batch_cell_lengths = torch.zeros(
                (batch_size, max_cells_in_batch), 
                dtype=torch.long
            )
            
            # Fill in the data
            for batch_idx, item in enumerate(batch):
                for cell_idx, cell_expr in enumerate(item["cell_expressions"]):
                    expr_len = len(cell_expr)
                    batch_cell_expressions[batch_idx, cell_idx, :expr_len] = cell_expr
                    batch_cell_masks[batch_idx, cell_idx, :expr_len] = 1
                    batch_cell_lengths[batch_idx, cell_idx] = item["cell_lengths"][cell_idx]
            
            # Handle text sequences (similar to your original code)
            prompt_input_ids = [item["prompt_input_ids"][0] for item in batch]
            target_input_ids = [item["target_input_ids"][0] for item in batch]
            
            # Pad prompts
            max_prompt_len = max(len(p) for p in prompt_input_ids)
            padded_prompt_ids = torch.full(
                (batch_size, max_prompt_len), 
                fill_value=self.tokenizer.pad_token_id, 
                dtype=torch.long
            )
            padded_prompt_mask = torch.zeros((batch_size, max_prompt_len), dtype=torch.long)
            
            for i, p in enumerate(prompt_input_ids):
                padded_prompt_ids[i, :len(p)] = p
                padded_prompt_mask[i, :len(p)] = 1
            
            # Pad targets
            max_target_len = max(len(t) for t in target_input_ids)
            padded_target_ids = torch.full(
                (batch_size, max_target_len), 
                fill_value=self.tokenizer.pad_token_id, 
                dtype=torch.long
            )
            padded_target_mask = torch.zeros((batch_size, max_target_len), dtype=torch.long)
            padded_labels = torch.full((batch_size, max_target_len), fill_value=-100, dtype=torch.long)
            
            for i, t in enumerate(target_input_ids):
                padded_target_ids[i, :len(t)] = t
                padded_target_mask[i, :len(t)] = 1
                padded_labels[i, :len(t)] = t
            
            if mode == "train":
                # Combine prompt and target for training
                combined_input_ids = torch.cat([padded_prompt_ids, padded_target_ids], dim=1)
                combined_attention_mask = torch.cat([padded_prompt_mask, padded_target_mask], dim=1)
                combined_labels = torch.cat([
                    torch.full_like(padded_prompt_ids, fill_value=-100),
                    padded_labels
                ], dim=1)
                
                return {
                    "expression_tokens": batch_cell_expressions,  # (B, max_cells, max_expr_len)
                    "expression_attention_mask": batch_cell_masks,
                    "expression_token_lengths": batch_cell_lengths,  # (B, max_cells)
                    "num_cells": torch.tensor([item["num_cells"] for item in batch]),
                    "input_ids": combined_input_ids,
                    "attention_mask": combined_attention_mask,
                    "labels": combined_labels
                }
            
            elif mode == "inference":
                return {
                    "expression_tokens": batch_cell_expressions,
                    "expression_attention_mask": batch_cell_masks,
                    "expression_token_lengths": batch_cell_lengths,
                    "num_cells": torch.tensor([item["num_cells"] for item in batch]),
                    "input_ids": padded_prompt_ids,
                    "attention_mask": padded_prompt_mask,
                    "target_input_ids": padded_target_ids,  # For evaluation
                    "raw_user_msgs": [item["raw_user_msg"] for item in batch],
                    "raw_assistant_msgs": [item["raw_assistant_msg"] for item in batch]
                }
            else:
                raise ValueError(f"Invalid mode: {mode}")
        
        return collate

