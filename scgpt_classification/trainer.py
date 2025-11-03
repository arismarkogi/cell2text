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
        # Filter parameters that require gradients (for "freeze" option)
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())

        self.optimizer = torch.optim.Adam(
            trainable_params, # Pass only trainable params
            lr=self.config.lr,
            eps=1e-4 if self.config.amp else 1e-8
        )
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=1,
            gamma=self.config.schedule_ratio
        )
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.amp)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch (DDP-safe)"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        start_time = time.time()

        # Set epoch for DistributedSampler (important for shuffling)
        if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(self.current_epoch)

        # tqdm only on rank 0
        pbar_desc = f"Epoch {self.current_epoch} [Train]"
        if hasattr(self, "rank") and self.rank == 0:
            pbar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=pbar_desc,
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

    
    def evaluate(self, eval_loader: DataLoader, desc: str = "Evaluate") -> Dict[str, float]:
        """Evaluate model on validation/test set (DDP-safe)"""
        self.model.eval()
        total_loss = 0.0
        all_predictions, all_labels = [], []

        # tqdm only on rank 0
        pbar_desc = f"Epoch {self.current_epoch} [{desc}]"
        if hasattr(self, "rank") and self.rank == 0:
            pbar = tqdm(
                enumerate(eval_loader),
                total=len(eval_loader),
                desc=pbar_desc,
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
                    avg_loss_so_far = total_loss / (len(all_labels)) if len(all_labels) > 0 else 0
                    pbar.set_postfix({'Loss': f'{avg_loss_so_far:.4f}'})

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
            metrics["predictions"] = preds_all
            metrics["labels"] = labels_all
            
        else:
            metrics = None # Return None for non-rank-0 processes

        return metrics

    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, save_dir: str = None):
        """Complete training loop with validation"""
        self.setup_training()
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # --- Training ---
            train_metrics = self.train_epoch(train_loader) 
            
            # --- Validation ---
            val_metrics = self.evaluate(val_loader, desc="Validate")

            # Update learning rate
            self.scheduler.step()
            
            # Only rank 0 handles logging, history, and saving
            if self.rank == 0:
                val_loss = val_metrics.get('loss', float("inf"))
                val_acc = val_metrics.get('acc', 0.0)
                
                # Update history
                self.training_history["train_loss"].append(train_metrics["train_loss_avg"])
                self.training_history["val_loss"].append(val_loss)
                self.training_history["val_acc"].append(val_acc)
                
                # Log epoch results
                elapsed = time.time() - epoch_start
                if self.logger:
                    self.logger.info(
                        f"Epoch {epoch}/{epochs} | Time: {elapsed:.2f}s | "
                        f"Train Loss: {train_metrics['train_loss_avg']:.4f} | "
                        f"Train Acc: {train_metrics['train_acc']:.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Val Acc: {val_acc:.4f}"
                    )
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model = copy.deepcopy(self.model.state_dict())
                    if self.logger:
                        self.logger.info(f"New best model found! Val Loss: {val_loss:.4f}")
                    if save_dir:
                        best_model_path = os.path.join(save_dir, "best_model.pt")
                        state = self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()
                        torch.save(state, best_model_path)
                        if self.logger:
                            self.logger.info(f"Best model saved to {best_model_path}")
        
        # Ensure all processes have the final model
        if self.world_size > 1:
            dist.barrier()
            
        # If no validation was done or all val_loss were inf, use final model
        if self.best_model is None:
            self.best_model = self.model.state_dict()
        
        return self.training_history
    
    def get_best_model(self):
        """Get the best model state dict"""
        return self.best_model
    
    def test(self, test_loader: DataLoader, id_to_label: Dict[int, str] = None):
        """Test the best model"""
        if self.best_model is None:
            if self.rank == 0:
                self.logger.warning("No best model found. Using final model for testing.")
            self.best_model = self.model.state_dict()
        
        # Load best model
        try:
            self.model.load_state_dict(self.best_model)
        except RuntimeError: # Handle DDP/non-DDP state_dict mismatch
             # create new OrderedDict that does not contain `module.`
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in self.best_model.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)

        
        # Evaluate
        test_metrics = self.evaluate(test_loader, desc="Test")
        
        # Add this check to handle None metrics in non-rank-0 processes
        if test_metrics is None:
            return None
        
        if self.logger and self.rank == 0:
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
        
        # Accuracy plot
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

    # -----------------------------------------------------------------
    # --- OBSOLETE: This function is the cause of the BatchNorm bug ---
    # --- and is no longer needed. It has been removed. ---
    # -----------------------------------------------------------------
    # def train_on_chunks(...):
    #     ...
