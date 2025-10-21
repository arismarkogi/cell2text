"""
Training and evaluation utilities for scGPT classification fine-tuning (DDP Version)
"""
import time
import copy
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict
from sklearn.metrics import accuracy_score, f1_score
import os
import gc
from tqdm import tqdm
import torch.distributed as dist


class ClassificationTrainer:
    def __init__(self, model, config, vocab, device="cuda", logger=None):
        self.model = model
        self.config = config
        self.vocab = vocab
        self.device = device
        self.logger = logger
        
        self.model = self.model.to(self.device)
        
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.criterion = nn.CrossEntropyLoss()
        
        self.current_epoch = 0
        self.best_model = None
        self.training_history = {"train_loss": [], "val_loss": [], "val_acc": []}
        
        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        
        self.pad_token = "<pad>"
    
    def setup_training(self):
        """Setup optimizer, scheduler, and other training components"""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            eps=1e-4 if self.config.amp else 1e-8
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=self.config.schedule_ratio
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.amp)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch (DDP-safe)"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        if self.world_size > 1 and hasattr(train_loader, 'sampler'):
            train_loader.sampler.set_epoch(self.current_epoch)

        pbar = None
        if self.rank == 0:
            pbar = tqdm(
                enumerate(train_loader), total=len(train_loader),
                desc=f"Epoch {self.current_epoch} [Train]", leave=False, ncols=100
            )
        else:
            pbar = enumerate(train_loader)

        for batch_idx, batch_data in pbar:
            gene_ids = batch_data["gene_ids"].to(self.device)
            values = batch_data["values"].to(self.device)
            labels = batch_data["labels"].to(self.device)
            src_key_padding_mask = gene_ids.eq(self.vocab[self.pad_token])

            with torch.cuda.amp.autocast(enabled=self.config.amp):
                output_dict = self.model(
                    gene_ids, values, src_key_padding_mask=src_key_padding_mask,
                    CLS=True, CCE=False, MVC=False, ECS=False, do_sample=False,
                )
                cls_output = output_dict["cls_output"]
                loss = self.criterion(cls_output, labels)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            total_correct += (cls_output.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)

            if self.rank == 0:
                pbar.set_postfix({
                    'Loss': f'{total_loss / (batch_idx + 1):.4f}',
                    'Acc': f'{total_correct / total_samples:.4f}'
                })
        
        if self.rank == 0 and pbar is not None:
            pbar.close()

        if self.world_size > 1:
            metrics = torch.tensor([total_loss, total_correct, total_samples], device=self.device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            total_loss, total_correct, total_samples = metrics.tolist()

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_acc = total_correct / total_samples if total_samples > 0 else 0
        return {"train_loss": avg_loss, "train_acc": avg_acc}
    
    def evaluate(self, eval_loader: DataLoader, return_predictions: bool = False) -> Dict[str, float]:
        """Evaluate model on a validation/test set (DDP-safe)"""
        self.model.eval()
        all_predictions, all_labels = [], []

        if self.world_size > 1 and hasattr(eval_loader, 'sampler'):
            eval_loader.sampler.set_epoch(self.current_epoch)

        with torch.no_grad():
            for batch_data in eval_loader:
                gene_ids = batch_data["gene_ids"].to(self.device)
                values = batch_data["values"].to(self.device)
                labels = batch_data["labels"].to(self.device)
                src_key_padding_mask = gene_ids.eq(self.vocab[self.pad_token])

                with torch.cuda.amp.autocast(enabled=self.config.amp):
                    output_dict = self.model(
                        gene_ids, values, src_key_padding_mask=src_key_padding_mask,
                        CLS=True, CCE=False, MVC=False, ECS=False, do_sample=False,
                    )
                all_predictions.extend(output_dict["cls_output"].argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        if self.world_size > 1:
            gathered_preds = [None] * self.world_size
            gathered_labels = [None] * self.world_size
            dist.gather_object(all_predictions, gathered_preds if self.rank == 0 else None, dst=0)
            dist.gather_object(all_labels, gathered_labels if self.rank == 0 else None, dst=0)
            
            if self.rank == 0:
                all_predictions = [item for sublist in gathered_preds for item in sublist]
                all_labels = [item for sublist in gathered_labels for item in sublist]

        metrics = {}
        if self.rank == 0:
            accuracy = accuracy_score(all_labels, all_predictions)
            f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0)
            weighted_f1 = f1_score(all_labels, all_predictions, average="weighted", zero_division=0)
            metrics = {"acc": accuracy, "f1": f1, "weighted_f1": weighted_f1}
            if return_predictions:
                metrics["predictions"] = all_predictions
                metrics["labels"] = all_labels
        
        return metrics

    def train_on_chunks(self, data_loader, task, label_to_id, epochs: int, save_dir: str = None):
        """Train on data chunks sequentially (DDP-safe)"""
        self.setup_training()
        train_files = data_loader.train_files
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()
            epoch_chunk_losses, epoch_chunk_accs = [], []
            
            for file_idx, train_file_path in enumerate(train_files):
                if self.rank == 0:
                    self.logger.info(f"\n[Epoch {epoch}] Processing file {file_idx + 1}/{len(train_files)}: {train_file_path.name}")

                n_cells = data_loader.get_n_obs(train_file_path)
                if n_cells == 0:
                    if self.rank == 0:
                        self.logger.warning(f"Skipping empty file: {train_file_path.name}")
                    continue

                for i in range(0, n_cells, self.config.cell_chunk_size):
                    cell_indices = range(i, min(i + self.config.cell_chunk_size, n_cells))
                    
                    if self.rank == 0:
                        self.logger.info(f"  Processing cell chunk {i} to {cell_indices.stop-1}")
                    
                    chunk_loader = data_loader.process_and_create_loader(
                        train_file_path, cell_indices, task, label_to_id, self.config.batch_size, 
                        shuffle=True, rank=self.rank, world_size=self.world_size
                    )
                    
                    chunk_metrics = self.train_epoch(chunk_loader)
                    epoch_chunk_losses.append(chunk_metrics["train_loss"])
                    epoch_chunk_accs.append(chunk_metrics["train_acc"])
                    
                    del chunk_loader
                    gc.collect()
                    torch.cuda.empty_cache()
            
            self.scheduler.step()
            
            if self.rank == 0:
                epoch_loss = np.mean(epoch_chunk_losses)
                epoch_acc = np.mean(epoch_chunk_accs)
                
                if save_dir:
                    torch.save(self.model.module.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch}.pt"))
                    self.logger.info(f"Model saved for epoch {epoch}")
                
                self.training_history["train_loss"].append(epoch_loss)
                self.training_history["val_acc"].append(epoch_acc)
                
                elapsed = time.time() - epoch_start
                self.logger.info(
                    f"Epoch {epoch}/{epochs} Summary | Time: {elapsed:.2f}s | "
                    f"Avg Chunk Train Loss: {epoch_loss:.4f} | "
                    f"Avg Chunk Train Acc: {epoch_acc:.4f}"
                )
        
        self.best_model = copy.deepcopy(self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict())
        return self.training_history

    def get_best_model(self):
        """Get the best model state dict"""
        if self.best_model is None:
            if self.rank == 0:
                self.logger.info("No best model found, returning last trained model.")
            return self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict()
        return self.best_model

