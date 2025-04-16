#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import random
from tensorboardX import SummaryWriter
from datasets import load_from_disk  
from datasets import load_dataset

from cell2text_model.cell2text_encoder import Cell2TextEncoder, Cell2TextEncoderConfig


# Import from custom modules
from cell2text_model.model import Cell2TextModel
from train.utils import (
    set_seed, 
    get_linear_schedule_with_warmup,
    ContrastiveLoss,
    save_checkpoint,
    load_checkpoint
)

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

@dataclass
class EncoderTrainingArguments:
    """
    Arguments for training the Cell2Text encoder model.
    """
    # Input/output paths
    data_dir: str = field(
        default="data",
        metadata={"help": "Path to the directory containing cell and text data"}
    )
    output_dir: str = field(
        default="output/encoder",
        metadata={"help": "Output directory for model checkpoints and logs"}
    )
    cell_data_path: str = field(
        default="cell_data.pkl",
        metadata={"help": "Filename for cell expression data"}
    )
    text_data_path: str = field(
        default="text_data.pkl",
        metadata={"help": "Filename for text data"}
    )
    checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a specific checkpoint to load"}
    )
    
    # Model parameters
    model_type: str = field(
        default="cell2text",
        metadata={"help": "Type of model to train"}
    )
    pretrained_cell_encoder: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained Geneformer model"}
    )
    pretrained_text_encoder: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained PubMedBERT model"}
    )
    embedding_fusion_method: str = field(
        default="attention",
        metadata={"help": "Method to fuse embeddings: 'concatenate', 'attention', 'sum', 'linear'"}
    )
    
    # Training parameters
    num_train_epochs: int = field(
        default=5,
        metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per device for training"}
    )
    per_device_eval_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per device for evaluation"}
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "Initial learning rate"}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay rate"}
    )
    warmup_steps: int = field(
        default=500,
        metadata={"help": "Number of warmup steps for learning rate scheduler"}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of steps to accumulate gradients before updating weights"}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Maximum gradient norm for gradient clipping"}
    )
    
    # Loss parameters
    cell_contrastive_loss_weight: float = field(
        default=0.5,
        metadata={"help": "Weight for cell-cell contrastive loss"}
    )
    text_contrastive_loss_weight: float = field(
        default=0.5,
        metadata={"help": "Weight for text-text contrastive loss"}
    )
    multimodal_contrastive_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for cell-text contrastive loss"}
    )
   
    temperature: float = field(
        default=0.07,
        metadata={"help": "Temperature parameter for contrastive loss"}
    )
    
    # Data augmentation parameters
    cell_dropout_rate: float = field(
        default=0.2,
        metadata={"help": "Dropout rate for cell data augmentation"}
    )
    text_dropout_rate: float = field(
        default=0.1,
        metadata={"help": "Dropout rate for text data augmentation"}
    )
    
    
    # Logging and saving parameters
    logging_steps: int = field(
        default=100,
        metadata={"help": "Number of steps between logging updates"}
    )
    save_steps: int = field(
        default=1000,
        metadata={"help": "Number of steps between saving checkpoints"}
    )
    eval_steps: int = field(
        default=1000,
        metadata={"help": "Number of steps between evaluations"}
    )
    
    # Other parameters
    seed: int = field(
        default=42,
        metadata={"help": "Random seed"}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use mixed precision training"}
    )
    local_rank: int = field(
        default=-1,
        metadata={"help": "Local rank for distributed training"}
    )
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        

class CellTextDataset(Dataset):
    """
    Dataset for paired cell-text data loaded from .arrow or .parquet files
    with contrastive learning support.
    """
    def __init__(
        self,
        data_path: str,  # Path to the directory containing the .arrow/.parquet files
        text_tokenizer,
        max_cell_length: int = 2048,
        max_text_length: int = 512,
        augment_cell: bool = False,  # Adjust based on what 'input_ids' represents
        augment_text: bool = True,
        cell_dropout_rate: float = 0.2,  # May not be applicable to token IDs
        text_dropout_rate: float = 0.1
    ):
        self.dataset = load_dataset("parquet", data_files=os.path.join(data_path, "processed.parquet"))["train"]
        logger.info(f"Loaded dataset with {len(self.dataset)} samples from {data_path}")

        self.text_tokenizer = text_tokenizer
        self.max_cell_length = max_cell_length
        self.max_text_length = max_text_length
        self.augment_cell = augment_cell
        self.augment_text = augment_text
        self.cell_dropout_rate = cell_dropout_rate
        self.text_dropout_rate = text_dropout_rate

    def __len__(self):
        return len(self.dataset)

    def _apply_cell_augmentation(self, cell_tokens: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to cell tokens (if meaningful).
        Consider masking random tokens or other sequence-based augmentations.
        Current implementation is a placeholder and might not be suitable.
        """
        if not self.augment_cell:
            return cell_tokens

        # Example: Randomly mask some tokens (replace with 0, assuming 0 is padding)
        mask_prob = self.cell_dropout_rate
        mask = torch.rand(cell_tokens.shape) > mask_prob
        augmented_cell_tokens = cell_tokens * mask.long()
        return augmented_cell_tokens

    def _apply_text_augmentation(self, text: str) -> str:
        """Apply word dropout to text data"""
        if not self.augment_text:
            return text

        words = text.split()
        mask = np.random.binomial(1, 1 - self.text_dropout_rate, len(words))
        augmented_words = [word for i, word in enumerate(words) if mask[i]]
        return " ".join(augmented_words) if augmented_words else text

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        cell_tokens = sample['input_ids']
        text = sample['text_desc']

        # Convert to PyTorch tensors
        cell_tokens = torch.tensor(cell_tokens, dtype=torch.long)

        # Apply cell augmentation (if enabled and a meaningful operation is defined)
        augmented_cell_tokens = self._apply_cell_augmentation(cell_tokens.clone())

        # Pad or truncate cell tokens
        if len(cell_tokens) > self.max_cell_length:
            cell_tokens = cell_tokens[:self.max_cell_length]
        augmented_cell_tokens = augmented_cell_tokens[:self.max_cell_length]
        cell_length = len(cell_tokens)
        cell_length_augmented = len(augmented_cell_tokens)

        # Apply text augmentation
        augmented_text = self._apply_text_augmentation(text)

        # Tokenize text data
        text_tokens = self.text_tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=self.max_text_length,
            truncation=True,
            padding='max_length',  # Pad here for simplicity in __getitem__
            return_tensors='pt'
        ).squeeze(0)  # Remove batch dimension

        text_tokens_augmented = self.text_tokenizer.encode(
            augmented_text,
            add_special_tokens=True,
            max_length=self.max_text_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        ).squeeze(0)

        text_length = (text_tokens != self.text_tokenizer.pad_token_id).sum()
        text_length_augmented = (text_tokens_augmented != self.text_tokenizer.pad_token_id).sum()

        return {
            'cell_tokens': cell_tokens,
            'cell_tokens_augmented': augmented_cell_tokens,
            'cell_length': torch.tensor(cell_length, dtype=torch.long),
            'cell_length_augmented': torch.tensor(cell_length_augmented, dtype=torch.long),
            'text_tokens': text_tokens,
            'text_tokens_augmented': text_tokens_augmented,
            'text_length': text_length,
            'text_length_augmented': text_length_augmented
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable length cell sequences.
    Text sequences are already padded in __getitem__.
    """
    # Extract items from batch
    cell_tokens = [item['cell_tokens'] for item in batch]
    cell_tokens_augmented = [item['cell_tokens_augmented'] for item in batch]
    cell_lengths = torch.stack([item['cell_length'] for item in batch])
    cell_lengths_augmented = torch.stack([item['cell_length_augmented'] for item in batch])

    text_tokens = torch.stack([item['text_tokens'] for item in batch])
    text_lengths = torch.stack([item['text_length'] for item in batch])
    text_lengths_augmented = torch.stack([item['text_length_augmented'] for item in batch])

    # Pad cell sequences
    cell_tokens_padded = pad_sequence(cell_tokens, batch_first=True, padding_value=0)
    cell_tokens_augmented_padded = pad_sequence(cell_tokens_augmented, batch_first=True, padding_value=0)

    return {
        'cell_tokens': cell_tokens_padded,
        'cell_tokens_augmented': cell_tokens_augmented_padded,
        'cell_lengths': cell_lengths,
        'cell_lengths_augmented': cell_lengths_augmented,
        'text_tokens': text_tokens,
        'text_lengths': text_lengths,
        'text_lengths_augmented': text_lengths_augmented
    }


class EncoderTrainer:
    """
    Trainer class for the Cell2Text encoder components.
    Focuses on contrastive learning between cell and text modalities.
    """
    def __init__(
        self,
        model: Cell2TextModel,
        args: EncoderTrainingArguments,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        cell_tokenizer=None,
        text_tokenizer=None
    ):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.cell_tokenizer = cell_tokenizer
        self.text_tokenizer = text_tokenizer
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Initialize loss functions
        self.contrastive_loss = ContrastiveLoss(temperature=args.temperature)
        
        # Set up optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        
        # Set up data loaders
        if train_dataset is not None:
            self.train_dataloader = DataLoader(
                train_dataset,
                batch_size=args.per_device_train_batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=4
            )
        else:
            self.train_dataloader = None
            
        if eval_dataset is not None:
            self.eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=args.per_device_eval_batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=4
            )
        else:
            self.eval_dataloader = None
            
        # Set up learning rate scheduler
        if self.train_dataloader is not None:
            num_update_steps_per_epoch = len(self.train_dataloader) // args.gradient_accumulation_steps
            num_training_steps = args.num_train_epochs * num_update_steps_per_epoch
            
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=num_training_steps
            )
        else:
            self.lr_scheduler = None
            
        # Set up mixed precision training if available
        self.scaler = torch.cuda.amp.GradScaler() if args.fp16 and torch.cuda.is_available() else None
        
        # Set up tensorboard
        self.tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))
        
        # Track best model
        self.best_eval_loss = float("inf")
    
    def train(self):
        """
        Run the training loop focusing on contrastive losses.
        """
        if self.train_dataloader is None:
            logger.warning("No training data provided, skipping training.")
            return
        
        # Initialize counters
        start_epoch = 0
        global_step = 0
        
        # Load checkpoint if provided
        if self.args.checkpoint_path:
            start_epoch, global_step = load_checkpoint(
                self.model, 
                self.optimizer, 
                self.lr_scheduler, 
                self.args.checkpoint_path
            )
            
        logger.info(f"Starting training from epoch {start_epoch}, global step {global_step}")
        
        total_loss = 0
        logging_loss = 0.0

        for epoch in range(start_epoch, self.args.num_train_epochs):
            self.model.train()
            epoch_loss = 0.0
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}")
            
            for step, batch in enumerate(progress_bar):
                inputs = {k: v.to(self.device) for k, v in batch.items()}

                with torch.cuda.amp.autocast(enabled=bool(self.scaler)):
                    outputs = self.model(**inputs)

                    # Retrieve embeddings
                    cell_embeds = outputs["cell_embeds"]
                    text_embeds = outputs["text_embeds"]
                    cell_embeds_aug = outputs["cell_embeds_aug"]
                    text_embeds_aug = outputs["text_embeds_aug"]

                    # Compute losses
                    loss_cell = self.contrastive_loss(cell_embeds, cell_embeds_aug)
                    loss_text = self.contrastive_loss(text_embeds, text_embeds_aug)
                    loss_multimodal = self.contrastive_loss(cell_embeds, text_embeds)

                    total_batch_loss = (
                        self.args.cell_contrastive_loss_weight * loss_cell +
                        self.args.text_contrastive_loss_weight * loss_text +
                        self.args.multimodal_contrastive_loss_weight * loss_multimodal
                    )
                
                if self.scaler:
                    self.scaler.scale(total_batch_loss).backward()
                else:
                    total_batch_loss.backward()

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    if global_step % self.args.logging_steps == 0:
                        avg_loss = (total_loss - logging_loss) / self.args.logging_steps
                        logger.info(f"Step {global_step}: avg loss = {avg_loss:.4f}")
                        self.tb_writer.add_scalar("loss/train", avg_loss, global_step)
                        logging_loss = total_loss

                    if global_step % self.args.eval_steps == 0 and self.eval_dataloader:
                        eval_loss = self.evaluate()
                        self.tb_writer.add_scalar("loss/eval", eval_loss, global_step)
                        logger.info(f"Evaluation loss: {eval_loss:.4f}")
                        if eval_loss < self.best_eval_loss:
                            self.best_eval_loss = eval_loss
                            save_checkpoint(self.model, self.optimizer, self.lr_scheduler, self.args.output_dir, global_step)

                epoch_loss += total_batch_loss.item()
                total_loss += total_batch_loss.item()
                progress_bar.set_postfix(loss=total_batch_loss.item())

        logger.info("Training complete!")

    def evaluate(self):
        """
        Run evaluation on the eval dataset and return average loss.
        """
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**inputs)

                cell_embeds = outputs["cell_embeds"]
                text_embeds = outputs["text_embeds"]
                cell_embeds_aug = outputs["cell_embeds_aug"]
                text_embeds_aug = outputs["text_embeds_aug"]

                loss_cell = self.contrastive_loss(cell_embeds, cell_embeds_aug)
                loss_text = self.contrastive_loss(text_embeds, text_embeds_aug)
                loss_multimodal = self.contrastive_loss(cell_embeds, text_embeds)

                batch_loss = (
                    self.args.cell_contrastive_loss_weight * loss_cell +
                    self.args.text_contrastive_loss_weight * loss_text +
                    self.args.multimodal_contrastive_loss_weight * loss_multimodal
                )
                total_loss += batch_loss.item()
        
        avg_loss = total_loss / len(self.eval_dataloader)
        return avg_loss
