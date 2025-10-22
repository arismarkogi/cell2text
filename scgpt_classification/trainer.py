"""
Training and evaluation utilities for scGPT classification fine-tuning
"""
import time
import copy
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from typing import Dict, Tuple, Optional, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
from tqdm import tqdm
import torch.distributed as dist
from transformers import get_linear_schedule_with_warmup


class ClassificationTrainer:
    def __init__(self, model, config, vocab, device="cuda", logger=None):
        self.model = model
        self.config = config
        self.vocab = vocab
        self.device = device
        self.logger = logger
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.best_model = None
        self.training_history = {"train_loss": [], "val_loss": [], "val_acc": []}
        
        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        
        # Setup
        self.pad_token = "<pad>"
    
    def setup_training(self):
        """Setup optimizer, scheduler, and other training components"""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            eps=1e-4 if self.config.amp else 1e-8
        )
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=1,
            gamma=self.config.schedule_ratio
        )
        
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.amp)
    
    def train_epoch(self, train_loader: DataLoader, set_sampler_epoch: bool = True) -> Dict[str, float]:
        """Train for one epoch (DDP-safe)
        
        Args:
            train_loader: DataLoader for training
            set_sampler_epoch: Whether to call set_epoch on the sampler (default True).
                              Set to False when training on chunks within an epoch.
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        start_time = time.time()

        # Set epoch for DistributedSampler (important for shuffling)
        # Only do this if set_sampler_epoch is True
        if set_sampler_epoch and hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(self.current_epoch)

        # tqdm only on rank 0
        if hasattr(self, "rank") and self.rank == 0:
            pbar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {self.current_epoch} [Train]",
                leave=False,
                ncols=100
            )
        else:
            pbar = enumerate(train_loader)

        for batch_idx, batch_data in pbar:
            # Move data to device
            gene_ids = batch_data["gene_ids"].to(self.device)
            values = batch_data["values"].to(self.device)
            labels = batch_data["labels"].to(self.device)
            batch_labels = batch_data.get("batch_labels", torch.zeros(len(labels))).to(self.device)

            # Create padding mask
            src_key_padding_mask = gene_ids.eq(self.vocab[self.pad_token])

            with torch.cuda.amp.autocast(enabled=self.config.amp):
                output_dict = self.model(
                    gene_ids,
                    values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if self.config.DSBN else None,
                    CLS=True,
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=False,
                )
                cls_output = output_dict["cls_output"]
                loss = self.criterion(cls_output, labels)

            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Local stats
            total_loss += loss.item() * labels.size(0)
            predictions = cls_output.argmax(1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            # Update tqdm only on rank 0
            if hasattr(self, "rank") and self.rank == 0:
                current_loss = total_loss / total_samples if total_samples > 0 else 0
                current_acc = total_correct / total_samples if total_samples > 0 else 0
                pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.4f}',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })

        if hasattr(self, "rank") and self.rank == 0:
            pbar.close()

        # DDP Reduce across processes
        loss_tensor = torch.tensor(total_loss, device=self.device)
        correct_tensor = torch.tensor(total_correct, device=self.device)
        samples_tensor = torch.tensor(total_samples, device=self.device)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(correct_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(samples_tensor, op=torch.distributed.ReduceOp.SUM)

        avg_loss = loss_tensor.item() / samples_tensor.item() if samples_tensor.item() > 0 else 0
        avg_acc = correct_tensor.item() / samples_tensor.item() if samples_tensor.item() > 0 else 0

        return {
            "train_loss_avg": avg_loss,
            "train_loss_total": loss_tensor.item(),
            "train_acc": avg_acc,
            "train_samples": samples_tensor.item(),
            "train_correct": correct_tensor.item()
        }

    
    def evaluate(self, eval_loader: DataLoader, return_predictions: bool = False) -> Dict[str, float]:
        """Evaluate model on validation/test set (DDP-safe)"""
        self.model.eval()
        total_loss = 0.0
        all_predictions, all_labels = [], []

        # tqdm only on rank 0
        if hasattr(self, "rank") and self.rank == 0:
            pbar = tqdm(
                enumerate(eval_loader),
                total=len(eval_loader),
                desc="Evaluating",
                leave=False,
                ncols=100
            )
        else:
            pbar = enumerate(eval_loader)

        with torch.no_grad():
            for batch_idx, batch_data in pbar:
                gene_ids = batch_data["gene_ids"].to(self.device)
                values = batch_data["values"].to(self.device)
                labels = batch_data["labels"].to(self.device)
                batch_labels = batch_data.get("batch_labels", torch.zeros(len(labels))).to(self.device)
                src_key_padding_mask = gene_ids.eq(self.vocab[self.pad_token])

                with torch.cuda.amp.autocast(enabled=self.config.amp):
                    output_dict = self.model(
                        gene_ids,
                        values,
                        src_key_padding_mask=src_key_padding_mask,
                        batch_labels=batch_labels if self.config.DSBN else None,
                        CLS=True,
                        CCE=False,
                        MVC=False,
                        ECS=False,
                        do_sample=False,
                    )
                    cls_output = output_dict["cls_output"]
                    loss = self.criterion(cls_output, labels)
                
                total_loss += loss.item() * labels.size(0)
                preds = cls_output.argmax(1).cpu().numpy()
                labels_np = labels.cpu().numpy()
                all_predictions.extend(preds)
                all_labels.extend(labels_np)

                if hasattr(self, "rank") and self.rank == 0:
                    pbar.set_postfix({'Loss': f'{total_loss / (batch_idx + 1):.4f}'})

        if hasattr(self, "rank") and self.rank == 0:
            pbar.close()

        # DDP Gather predictions/labels
        def gather_numpy_array(np_array):
            tensor = torch.tensor(np_array, device=self.device, dtype=torch.long)
            length = torch.tensor([tensor.numel()], device=self.device)
            lengths = [torch.zeros(1, device=self.device, dtype=torch.long) for _ in range(self.world_size)]
            torch.distributed.all_gather(lengths, length)
            max_len = max([l.item() for l in lengths])

            if tensor.numel() < max_len:
                pad = torch.zeros(max_len - tensor.numel(), device=self.device, dtype=tensor.dtype)
                tensor = torch.cat([tensor, pad], dim=0)

            gathered = [torch.zeros(max_len, device=self.device, dtype=tensor.dtype) for _ in range(self.world_size)]
            torch.distributed.all_gather(gathered, tensor)

            result = []
            for i, g in enumerate(gathered):
                n = lengths[i].item()
                result.extend(g[:n].cpu().numpy().tolist())
            return np.array(result)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            loss_tensor = torch.tensor(total_loss, device=self.device)
            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
            
            total_loss_global = loss_tensor.item()

            preds_all = gather_numpy_array(all_predictions)
            labels_all = gather_numpy_array(all_labels)
        else:
            total_loss_global = total_loss
            preds_all = np.array(all_predictions)
            labels_all = np.array(all_labels)

        # Metrics only on rank 0
        if not torch.distributed.is_available() or not torch.distributed.is_initialized() or self.rank == 0:
            avg_loss = total_loss_global / len(labels_all) if len(labels_all) > 0 else 0
            
            accuracy = accuracy_score(labels_all, preds_all)
            precision = precision_score(labels_all, preds_all, average="macro", zero_division=0)
            recall = recall_score(labels_all, preds_all, average="macro", zero_division=0)
            f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
            weighted_f1 = f1_score(labels_all, preds_all, average="weighted", zero_division=0)

            metrics = {
                "loss": avg_loss,
                "acc": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "weighted_f1": weighted_f1
            }
            if return_predictions:
                metrics["predictions"] = preds_all
                metrics["labels"] = labels_all
        else:
            metrics = None

        return metrics


    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, save_dir: str = None):
        """Complete training loop - skips validation and saves model after each epoch"""
        self.setup_training()
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Training only - skip validation
            train_metrics = self.train_epoch(train_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            if save_dir and self.rank == 0:
                epoch_model_path = os.path.join(save_dir, f"model_epoch_{epoch}.pt")
                # if DDP, save module state_dict
                state = self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()
                torch.save(state, epoch_model_path)
                if self.logger:
                    self.logger.info(f"Model saved for epoch {epoch} at {epoch_model_path}")

            
            # Update history with training metrics only
            self.training_history["train_loss"].append(train_metrics["train_loss_avg"])
            # Keep validation history empty or with default values
            self.training_history["val_loss"].append(float("inf"))
            self.training_history["val_acc"].append(0.0)
            
            # Log epoch results
            elapsed = time.time() - epoch_start
            if self.logger:
                self.logger.info(
                    f"Epoch {epoch}/{epochs} | Time: {elapsed:.2f}s | "
                    f"Train Loss: {train_metrics['train_loss_avg']:.4f} | "
                    f"Train Acc: {train_metrics['train_acc']:.4f}"
                )
        
        # Set best model to the final model since we're not doing validation
        self.best_model = copy.deepcopy(self.model.state_dict())
        
        return self.training_history
    
    def get_best_model(self):
        """Get the best model state dict"""
        return self.best_model
    
    def test(self, test_loader: DataLoader, id_to_label: Dict[int, str] = None):
        """Test the best model"""
        if self.best_model is None:
            raise ValueError("No best model found. Train first.")
        
        # Load best model
        self.model.load_state_dict(self.best_model)
        
        # Evaluate
        test_metrics = self.evaluate(test_loader, return_predictions=True)
        
        # Add this check to handle None metrics in non-rank-0 processes
        if test_metrics is None:
            return None  # Return None for non-rank-0 processes
        
        if self.logger:
            self.logger.info(
                f"Test Results - Loss: {test_metrics.get('loss', float('inf')):.4f} | "
                f"Acc: {test_metrics.get('acc', 0.0):.4f} | "
                f"F1: {test_metrics.get('f1', 0.0):.4f} | "
                f"Weighted F1: {test_metrics.get('weighted_f1', 0.0):.4f}"
            )
        return test_metrics
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_model': self.best_model,
            'training_history': self.training_history,
            'config': self.config,
            'vocab': self.vocab
        }
        
        torch.save(checkpoint, path)
        if self.logger:
            self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_model = checkpoint['best_model']
        self.training_history = checkpoint['training_history']
        
        if self.logger:
            self.logger.info(f"Checkpoint loaded from {path}")
    
    def plot_confusion_matrix(self, labels, predictions, class_names=None, save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(labels, predictions)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return cm, cm_norm
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.training_history["train_loss"], label="Train Loss")
        ax1.plot(self.training_history["val_loss"], label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot - only training accuracy since we skip validation
        ax2.plot(self.training_history["val_acc"], label="Val Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Validation Accuracy")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

    def train_on_chunks(self, data_loader, task, label_to_id, epochs: int, save_dir: str = None):
        """Train on data chunks sequentially (DDP-safe) - FIXED VERSION
        
        Key fix: We create a unique sampler for each chunk with a chunk-specific seed
        that combines the epoch number and chunk index. This ensures proper shuffling
        across epochs while maintaining deterministic behavior.
        """
        self.setup_training()
        
        train_batches = data_loader.train_batches
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train on each chunk
            epoch_total_loss = 0.0
            epoch_total_correct = 0
            epoch_total_samples = 0
            
            for chunk_idx, train_batch in enumerate(train_batches):
                if self.rank == 0:
                    print(f"\n[Epoch {epoch}] Processing training chunk {chunk_idx + 1}/{len(train_batches)}")
                
                # ALL RANKS: Process the data to get the dataset
                file_path, cell_indices = train_batch
                temp_loader = data_loader.process_and_create_loader(
                    file_path,
                    cell_indices,
                    task,
                    label_to_id,
                    self.config.batch_size,
                    shuffle=False,
                    rank=self.rank,
                    world_size=self.world_size,
                )
                chunk_dataset = temp_loader.dataset
                del temp_loader
                
                # ALL RANKS: Create DDP sampler with epoch-aware seeding
                sampler = None
                shuffle = True
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    # FIXED: Create sampler with chunk-specific seed based on epoch
                    # This ensures different shuffling for each epoch but same shuffling
                    # across ranks within the same epoch
                    sampler = DistributedSampler(
                        chunk_dataset, 
                        num_replicas=self.world_size, 
                        rank=self.rank, 
                        shuffle=True,
                        seed=42 + epoch  # Seed changes with epoch, not chunk
                    )
                    # FIXED: Manually set the epoch on the sampler for this chunk
                    # This combines the epoch with the chunk_idx for unique shuffling
                    sampler.set_epoch(epoch * 1000 + chunk_idx)
                    shuffle = False
                
                chunk_loader = DataLoader(
                    chunk_dataset, 
                    batch_size=self.config.batch_size,
                    shuffle=shuffle,
                    sampler=sampler,
                    num_workers=4, 
                    pin_memory=True
                )
                
                # ALL RANKS: Train on this chunk
                # FIXED: Pass set_sampler_epoch=False to prevent train_epoch from 
                # resetting the sampler epoch we just set
                chunk_metrics = self.train_epoch(chunk_loader, set_sampler_epoch=False)
                
                # Accumulate metrics (all ranks have the same DDP-synced values)
                epoch_total_loss += chunk_metrics["train_loss_total"]
                epoch_total_correct += chunk_metrics["train_correct"]
                epoch_total_samples += chunk_metrics["train_samples"]
                
                # Clean up (all ranks)
                del chunk_loader, chunk_dataset, sampler
                gc.collect()
                torch.cuda.empty_cache()
                
                # Barrier so all ranks wait for each other before loading next chunk
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.barrier()
            
            # Calculate epoch metrics (all ranks compute the same value)
            epoch_loss = epoch_total_loss / epoch_total_samples if epoch_total_samples > 0 else 0
            epoch_acc = epoch_total_correct / epoch_total_samples if epoch_total_samples > 0 else 0
            
            # Update learning rate
            self.scheduler.step()
            
            # Save model (only rank 0)
            if save_dir and self.rank == 0:
                epoch_model_path = os.path.join(save_dir, f"model_epoch_{epoch}.pt")
                state = self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()
                torch.save(state, epoch_model_path)
                if self.logger:
                    self.logger.info(f"Model saved for epoch {epoch} at {epoch_model_path}")
            
            # Update history
            self.training_history["train_loss"].append(epoch_loss)
            self.training_history["val_loss"].append(float("inf"))
            self.training_history["val_acc"].append(0.0)
            
            # Log epoch results (only rank 0)
            elapsed = time.time() - epoch_start
            if self.logger and self.rank == 0:
                self.logger.info(
                    f"Epoch {epoch}/{epochs} | Time: {elapsed:.2f}s | "
                    f"Train Loss: {epoch_loss:.4f} | "
                    f"Train Acc: {epoch_acc:.4f}"
                )
        
        # Set best model to final model
        self.best_model = copy.deepcopy(self.model.state_dict())
        
        return self.training_history