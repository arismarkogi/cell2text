import time
import copy
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict
from sklearn.metrics import f1_score, jaccard_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
from tqdm import tqdm
import torch.distributed as dist

class FocalLoss(nn.Module):
    """Focal Loss for multi-label classification"""
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class PathwayTrainer:
    def __init__(self, model, config, vocab, device="cuda", logger=None, k_pathways=2):
        self.model = model
        self.config = config
        self.vocab = vocab
        self.device = device
        self.logger = logger
        self.k_pathways = k_pathways  # Top-k predictions
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.criterion = FocalLoss(alpha=1.0, gamma=2.0)
        
        # Training state
        self.current_epoch = 0
        self.best_val_f1 = 0.0
        self.best_model = None
        self.training_history = {"train_loss": [], "val_loss": [], "val_f1": []}
        
        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        
        self.pad_token = "<pad>"
    
    def setup_training(self):
        """Setup optimizer, scheduler, and other training components"""
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.lr,
            weight_decay=0.01
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.amp)
    
    def get_topk_predictions(self, logits, k=None):
        """Get top-k predictions (multi-hot encoding)"""
        if k is None:
            k = self.k_pathways
        batch_size, num_pathways = logits.shape
        predictions = torch.zeros_like(logits)
        _, top_indices = torch.topk(logits, k, dim=1)
        predictions.scatter_(1, top_indices, 1)
        return predictions.long()
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(self.current_epoch)

        pbar_desc = f"Epoch {self.current_epoch} [Train]"
        if self.rank == 0:
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=pbar_desc, leave=False, ncols=100)
        else:
            pbar = enumerate(train_loader)

        for batch_idx, batch_data in pbar:
            gene_ids = batch_data["gene_ids"].to(self.device)
            values = batch_data["values"].to(self.device)
            labels = batch_data["labels"].to(self.device).float()  # Multi-hot labels

            src_key_padding_mask = gene_ids.eq(self.vocab[self.pad_token])

            with torch.cuda.amp.autocast(enabled=self.config.amp):
                logits = self.model(gene_ids, values, src_key_padding_mask)
                loss = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            if self.rank == 0:
                current_loss = total_loss / total_samples if total_samples > 0 else 0
                pbar.set_postfix({'Loss': f'{current_loss:.4f}'})

        if self.rank == 0:
            pbar.close()

        # DDP Reduce
        loss_tensor = torch.tensor(total_loss, device=self.device)
        samples_tensor = torch.tensor(total_samples, device=self.device)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(samples_tensor, op=torch.distributed.ReduceOp.SUM)

        avg_loss = loss_tensor.item() / samples_tensor.item() if samples_tensor.item() > 0 else 0

        return {"train_loss": avg_loss}
    
    def evaluate(self, eval_loader: DataLoader, desc: str = "Evaluate") -> Dict[str, float]:
        """Evaluate model on validation/test set"""
        self.model.eval()
        total_loss = 0.0
        all_predictions, all_labels = [], []

        pbar_desc = f"Epoch {self.current_epoch} [{desc}]"
        if self.rank == 0:
            pbar = tqdm(enumerate(eval_loader), total=len(eval_loader), desc=pbar_desc, leave=False, ncols=100)
        else:
            pbar = enumerate(eval_loader)

        with torch.no_grad():
            for batch_idx, batch_data in pbar:
                gene_ids = batch_data["gene_ids"].to(self.device)
                values = batch_data["values"].to(self.device)
                labels = batch_data["labels"].to(self.device).float()
                
                src_key_padding_mask = gene_ids.eq(self.vocab[self.pad_token])

                with torch.cuda.amp.autocast(enabled=self.config.amp):
                    logits = self.model(gene_ids, values, src_key_padding_mask)
                    loss = self.criterion(logits, labels)
                
                total_loss += loss.item() * labels.size(0)
                
                # Get top-k predictions
                preds = self.get_topk_predictions(logits, self.k_pathways)
                
                all_predictions.append(preds.cpu())
                all_labels.append(labels.cpu())

                if self.rank == 0:
                    avg_loss = total_loss / (len(all_labels) * labels.size(0)) if len(all_labels) > 0 else 0
                    pbar.set_postfix({'Loss': f'{avg_loss:.4f}'})

        if self.rank == 0:
            pbar.close()

        # DDP Gather
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

        # Concatenate predictions and labels
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            loss_tensor = torch.tensor(total_loss, device=self.device)
            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
            total_loss_global = loss_tensor.item()

            # Flatten for gathering
            preds_flat = all_predictions.view(-1).numpy()
            labels_flat = all_labels.view(-1).numpy()
            
            preds_all = gather_numpy_array(preds_flat)
            labels_all = gather_numpy_array(labels_flat)
            
            # Reshape back
            num_pathways = all_predictions.shape[1]
            preds_all = preds_all.reshape(-1, num_pathways)
            labels_all = labels_all.reshape(-1, num_pathways)
        else:
            total_loss_global = total_loss
            preds_all = all_predictions.numpy()
            labels_all = all_labels.numpy()

        # Calculate metrics (only on rank 0)
        if not torch.distributed.is_available() or not torch.distributed.is_initialized() or self.rank == 0:
            avg_loss = total_loss_global / len(labels_all) if len(labels_all) > 0 else 0
            
            # Multi-label metrics
            subset_acc = accuracy_score(labels_all, preds_all)
            weighted_f1 = f1_score(labels_all, preds_all, average="weighted", zero_division=0)
            macro_f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
            jaccard = jaccard_score(labels_all, preds_all, average="samples", zero_division=0)

            metrics = {
                "loss": avg_loss,
                "subset_accuracy": subset_acc,
                "weighted_f1": weighted_f1,
                "macro_f1": macro_f1,
                "jaccard_similarity": jaccard
            }
            metrics["predictions"] = preds_all
            metrics["labels"] = labels_all
        else:
            metrics = None

        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, save_dir: str = None):
        """Complete training loop with validation"""
        self.setup_training()
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader, desc="Validate")

            if self.rank == 0:
                val_loss = val_metrics.get('loss', float("inf"))
                val_f1 = val_metrics.get('weighted_f1', 0.0)
                
                self.training_history["train_loss"].append(train_metrics["train_loss"])
                self.training_history["val_loss"].append(val_loss)
                self.training_history["val_f1"].append(val_f1)
                
                elapsed = time.time() - epoch_start
                if self.logger:
                    self.logger.info(
                        f"Epoch {epoch}/{epochs} | Time: {elapsed:.2f}s | "
                        f"Train Loss: {train_metrics['train_loss']:.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Val Weighted F1: {val_f1:.4f} | "
                        f"Val Subset Acc: {val_metrics.get('subset_accuracy', 0.0):.4f}"
                    )
                
                # Save best model based on weighted F1
                if val_f1 > self.best_val_f1:
                    self.best_val_f1 = val_f1
                    self.best_model = copy.deepcopy(self.model.state_dict())
                    if self.logger:
                        self.logger.info(f"New best model! Weighted F1: {val_f1:.4f}")
                    if save_dir:
                        best_model_path = os.path.join(save_dir, "best_model.pt")
                        state = self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()
                        torch.save(state, best_model_path)
                        if self.logger:
                            self.logger.info(f"Best model saved to {best_model_path}")
                
                # Update scheduler
                self.scheduler.step(val_f1)
        
        if self.world_size > 1:
            dist.barrier()
        
        if self.best_model is None:
            self.best_model = self.model.state_dict()
        
        return self.training_history
    
    def test(self, test_loader: DataLoader):
        """Test the best model"""
        if self.best_model is None:
            if self.rank == 0:
                self.logger.warning("No best model found. Using final model.")
            self.best_model = self.model.state_dict()
        
        try:
            self.model.load_state_dict(self.best_model)
        except RuntimeError:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in self.best_model.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)
        
        test_metrics = self.evaluate(test_loader, desc="Test")
        
        if test_metrics is None:
            return None
        
        if self.logger and self.rank == 0:
            self.logger.info(
                f"Test Results - "
                f"Weighted F1: {test_metrics.get('weighted_f1', 0.0):.4f} | "
                f"Jaccard: {test_metrics.get('jaccard_similarity', 0.0):.4f} | "
                f"Subset Acc: {test_metrics.get('subset_accuracy', 0.0):.4f}"
            )
        return test_metrics
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.training_history["train_loss"], label="Train Loss")
        ax1.plot(self.training_history["val_loss"], label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(self.training_history["val_f1"], label="Val Weighted F1")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("F1 Score")
        ax2.set_title("Validation F1 Score")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()