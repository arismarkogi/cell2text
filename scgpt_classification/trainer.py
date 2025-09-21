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
from typing import Dict, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class ClassificationTrainer:
    def __init__(self, model, config, vocab, device="cuda", logger=None):
        self.model = model
        self.config = config
        self.vocab = vocab
        self.device = device
        self.logger = logger
        
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
        
        for batch_idx, batch_data in enumerate(train_loader):
            # Move data to device
            gene_ids = batch_data["gene_ids"].to(self.device)
            values = batch_data["values"].to(self.device)
            labels = batch_data["labels"].to(self.device)
            batch_labels = batch_data.get("batch_labels", torch.zeros(len(labels))).to(self.device)
            
            # Create padding mask
            src_key_padding_mask = gene_ids.eq(self.vocab[self.pad_token])
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.config.amp):
                output_dict = self.model(
                    gene_ids,
                    values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if self.config.DSBN else None,
                    CLS=True,  # Enable classification
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=False,
                )
                
                # Classification loss
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
            
            # Log progress
            if batch_idx % 100 == 0 and batch_idx > 0:
                current_loss = total_loss / (batch_idx + 1)
                current_acc = total_correct / total_samples
                elapsed = time.time() - start_time
                
                if self.logger:
                    self.logger.info(
                        f"Epoch {self.current_epoch} | Batch {batch_idx}/{len(train_loader)} | "
                        f"Loss: {current_loss:.4f} | Acc: {current_acc:.4f} | "
                        f"Time: {elapsed:.2f}s"
                    )
        
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
        
        with torch.no_grad():
            for batch_data in eval_loader:
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
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        avg_loss = total_loss / len(eval_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average="macro", zero_division=0)
        recall = recall_score(all_labels, all_predictions, average="macro", zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0)
        
        metrics = {
            "val_loss": avg_loss,
            "val_acc": accuracy,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1
        }
        
        if return_predictions:
            metrics["predictions"] = all_predictions
            metrics["labels"] = all_labels
        
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        """Complete training loop"""
        self.setup_training()
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.evaluate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Track best model
            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.best_model = copy.deepcopy(self.model.state_dict())
                
                if self.logger:
                    self.logger.info(f"New best model at epoch {epoch} with val_loss: {self.best_val_loss:.4f}")
            
            # Update history
            self.training_history["train_loss"].append(train_metrics["train_loss"])
            self.training_history["val_loss"].append(val_metrics["val_loss"])
            self.training_history["val_acc"].append(val_metrics["val_acc"])
            
            # Log epoch results
            elapsed = time.time() - epoch_start
            if self.logger:
                self.logger.info(
                    f"Epoch {epoch}/{epochs} | Time: {elapsed:.2f}s | "
                    f"Train Loss: {train_metrics['train_loss']:.4f} | "
                    f"Val Loss: {val_metrics['val_loss']:.4f} | "
                    f"Val Acc: {val_metrics['val_acc']:.4f}"
                )
        
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
                f"Test Results - Loss: {test_metrics['val_loss']:.4f} | "
                f"Acc: {test_metrics['val_acc']:.4f} | "
                f"F1: {test_metrics['val_f1']:.4f}"
            )
        
        return test_metrics
    
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