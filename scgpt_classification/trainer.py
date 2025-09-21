"""
Training and evaluation utilities for scGPT classification fine-tuning
"""
import time
import copy
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm


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
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        start_time = time.time()
        
        # Create progress bar
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {self.current_epoch} [Train]",
            leave=False,
            ncols=100
        )
        
        for batch_idx, batch_data in pbar:
            # Move data to device
            gene_ids = batch_data["gene_ids"].to(self.device)
            values = batch_data["values"].to(self.device)
            labels = batch_data["labels"].to(self.device)
            batch_labels = batch_data.get("batch_labels", torch.zeros(len(labels))).to(self.device)
            
            # Create padding mask
            src_key_padding_mask = gene_ids.eq(self.vocab[self.pad_token])

            # Check for NaN/Inf values
            for k, v in batch_data.items():
                if isinstance(v, torch.Tensor):
                    if torch.isnan(v).any() or torch.isinf(v).any():
                        print(f"⚠️ NaN or Inf detected in {k}")
                        print("Tensor stats:", v.shape, v.dtype)
                        print("NaN count:", torch.isnan(v).sum().item())
                        print("Inf count:", torch.isinf(v).sum().item())
                        raise ValueError(f"Invalid values in {k}")
            
            # Forward pass with mixed precision
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
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    1.0,
                    error_if_nonfinite=False if self.scaler.is_enabled() else True,
                )
                if len(w) > 0 and self.logger:
                    self.logger.warning(f"Found infinite gradient at batch {batch_idx}")
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Statistics
            total_loss += loss.item()
            predictions = cls_output.argmax(1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            # Update progress bar
            current_loss = total_loss / (batch_idx + 1)
            current_acc = total_correct / total_samples
            
            # Update progress bar description with current metrics
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Log progress less frequently since we have tqdm
            if batch_idx % 500 == 0 and batch_idx > 0 and self.logger:
                elapsed = time.time() - start_time
                self.logger.info(
                    f"Epoch {self.current_epoch} | Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {current_loss:.4f} | Acc: {current_acc:.4f} | "
                    f"Time: {elapsed:.2f}s"
                )
        
        # Close progress bar
        pbar.close()
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = total_correct / total_samples
        
        return {
            "train_loss": avg_loss,
            "train_acc": avg_acc,
            "train_samples": total_samples
        }
    
    def evaluate(self, eval_loader: DataLoader, return_predictions: bool = False) -> Dict[str, float]:
        """Evaluate model on validation/test set"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        # Create progress bar for evaluation
        pbar = tqdm(
            enumerate(eval_loader),
            total=len(eval_loader),
            desc="Evaluating",
            leave=False,
            ncols=100
        )
        
        with torch.no_grad():
            for batch_idx, batch_data in pbar:
                # Move data to device
                gene_ids = batch_data["gene_ids"].to(self.device)
                values = batch_data["values"].to(self.device)
                labels = batch_data["labels"].to(self.device)
                batch_labels = batch_data.get("batch_labels", torch.zeros(len(labels))).to(self.device)
                
                # Create padding mask
                src_key_padding_mask = gene_ids.eq(self.vocab[self.pad_token])
                
                # Forward pass
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
                
                # Collect results
                total_loss += loss.item()
                predictions = cls_output.argmax(1).cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                all_predictions.extend(predictions)
                all_labels.extend(labels_np)
                
                # Update progress bar with current metrics
                current_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Batches': f'{batch_idx + 1}/{len(eval_loader)}'
                })
        
        # Close progress bar
        pbar.close()
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        avg_loss = total_loss / len(eval_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average="macro", zero_division=0)
        recall = recall_score(all_labels, all_predictions, average="macro", zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0)
        weighted_f1 = f1_score(all_labels, all_predictions, average="weighted", zero_division=0)
        
        metrics = {
            "val_loss": avg_loss,
            "val_acc": accuracy,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1,
            "val_weighted_f1": weighted_f1
        }
        
        if return_predictions:
            metrics["predictions"] = all_predictions
            metrics["labels"] = all_labels
        
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
            
            # Save model after each epoch if save_dir is provided
            if save_dir:
                epoch_model_path = os.path.join(save_dir, f"model_epoch_{epoch}.pt")
                torch.save(self.model.state_dict(), epoch_model_path)
                if self.logger:
                    self.logger.info(f"Model saved for epoch {epoch} at {epoch_model_path}")
            
            # Update history with training metrics only
            self.training_history["train_loss"].append(train_metrics["train_loss"])
            # Keep validation history empty or with default values
            self.training_history["val_loss"].append(float("inf"))
            self.training_history["val_acc"].append(0.0)
            
            # Log epoch results
            elapsed = time.time() - epoch_start
            if self.logger:
                self.logger.info(
                    f"Epoch {epoch}/{epochs} | Time: {elapsed:.2f}s | "
                    f"Train Loss: {train_metrics['train_loss']:.4f} | "
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
        
        if self.logger:
            self.logger.info(
                f"Test Results - Loss: {test_metrics.get('val_loss', float('inf')):.4f} | "
                f"Acc: {test_metrics.get('val_acc', 0.0):.4f} | "
                f"F1: {test_metrics.get('val_f1', 0.0):.4f} | "
                f"Weighted F1: {test_metrics.get('val_weighted_f1', 0.0):.4f}"
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