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
        self.data = load_from_disk(data_path).select(range(20))
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
        input_ids_batch = sample["input_ids_batch"]  # list of lists
        num_cells = len(input_ids_batch)

        # find max length for this sample
        max_len = max(len(c) for c in input_ids_batch if len(c) > 0)

        # create tensor (num_cells, max_len)
        cell_expressions = torch.full(
            (num_cells, max_len), 
            fill_value=self.tokenizer.pad_token_id, 
            dtype=torch.long
        )
        cell_masks = torch.zeros((num_cells, max_len), dtype=torch.long)
        cell_lengths = []

        for i, cell_expr in enumerate(input_ids_batch):
            expr_len = len(cell_expr)
            if expr_len > 0:
                cell_expressions[i, :expr_len] = torch.tensor(cell_expr, dtype=torch.long)
                cell_masks[i, :expr_len] = 1
                cell_lengths.append(expr_len)
            else:
                cell_lengths.append(0)

        # Prepare conversation tokens
        user_message = self._replace_gene_lists_with_placeholders(sample["user_msg"], num_cells)
        conversation = [
            {"role": "system", "content": sample["system_msg"]},
            {"role": "user", "content": user_message}
        ]
        prompt_ids = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            padding=False,
            return_tensors="pt"
        )[0]  # (seq_len,)

        target_ids = self.tokenizer(
            [sample["assistant_msg"] + self.tokenizer.eos_token],
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors="pt"
        )["input_ids"][0]  # (target_len,)

        return {
            "cell_expressions": cell_expressions,  # (num_cells, max_len)
            "cell_masks": cell_masks,              # (num_cells, max_len)
            "cell_lengths": torch.tensor(cell_lengths, dtype=torch.long),
            "num_cells": num_cells,
            "prompt_input_ids": prompt_ids,
            "target_input_ids": target_ids,
            "raw_user_msg": sample["user_msg"],
            "raw_assistant_msg": sample["assistant_msg"]
        }

    
    def collate_fn(self, geneformer_pad_token_id=0, mode="train"):
        def collate(batch):
            batch_size = len(batch)

            # Pad cell_expressions (B, max_cells, max_len)
            max_cells = max(item["num_cells"] for item in batch)
            max_len = max(item["cell_expressions"].shape[1] for item in batch)

            cell_expressions = torch.full(
                (batch_size, max_cells, max_len),
                fill_value=geneformer_pad_token_id,
                dtype=torch.long
            )
            cell_masks = torch.zeros((batch_size, max_cells, max_len), dtype=torch.long)
            cell_lengths = torch.zeros((batch_size, max_cells), dtype=torch.long)

            for i, item in enumerate(batch):
                ncells, clen = item["cell_expressions"].shape
                cell_expressions[i, :ncells, :clen] = item["cell_expressions"]
                cell_masks[i, :ncells, :clen] = item["cell_masks"]
                cell_lengths[i, :ncells] = item["cell_lengths"]

            # Pad prompts
            prompt_lens = [len(item["prompt_input_ids"]) for item in batch]
            max_prompt_len = max(prompt_lens)
            prompt_ids = torch.full(
                (batch_size, max_prompt_len),
                fill_value=self.tokenizer.pad_token_id,
                dtype=torch.long
            )
            prompt_mask = torch.zeros((batch_size, max_prompt_len), dtype=torch.long)
            for i, item in enumerate(batch):
                L = len(item["prompt_input_ids"])
                prompt_ids[i, :L] = item["prompt_input_ids"]
                prompt_mask[i, :L] = 1

            # Pad targets
            target_lens = [len(item["target_input_ids"]) for item in batch]
            max_target_len = max(target_lens)
            target_ids = torch.full(
                (batch_size, max_target_len),
                fill_value=self.tokenizer.pad_token_id,
                dtype=torch.long
            )
            target_mask = torch.zeros((batch_size, max_target_len), dtype=torch.long)
            labels = torch.full((batch_size, max_target_len), -100, dtype=torch.long)
            for i, item in enumerate(batch):
                L = len(item["target_input_ids"])
                target_ids[i, :L] = item["target_input_ids"]
                target_mask[i, :L] = 1
                labels[i, :L] = item["target_input_ids"]

            if mode == "train":
                input_ids = torch.cat([prompt_ids, target_ids], dim=1)
                attention_mask = torch.cat([prompt_mask, target_mask], dim=1)
                combined_labels = torch.cat([
                    torch.full_like(prompt_ids, -100), labels
                ], dim=1)

                return {
                    "expression_tokens": cell_expressions,
                    "expression_attention_mask": cell_masks,
                    "expression_token_lengths": cell_lengths,
                    "num_cells": torch.tensor([item["num_cells"] for item in batch]),
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": combined_labels
                }

            elif mode == "inference":
                return {
                    "expression_tokens": cell_expressions,
                    "expression_attention_mask": cell_masks,
                    "expression_token_lengths": cell_lengths,
                    "num_cells": torch.tensor([item["num_cells"] for item in batch]),
                    "input_ids": prompt_ids,
                    "attention_mask": prompt_mask,
                    "target_input_ids": target_ids,
                    "raw_user_msgs": [item["raw_user_msg"] for item in batch],
                    "raw_assistant_msgs": [item["raw_assistant_msg"] for item in batch]
                }
        return collate


