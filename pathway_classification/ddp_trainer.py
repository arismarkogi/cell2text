import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from tqdm import tqdm
import os
import json
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, roc_auc_score
from datetime import datetime
from dataset import MultiDatasetPathwayDataset  # Import the dataset class
from classifier import GeneformerPathwayClassifier  # Import the model class
from torch.nn.parallel import DistributedDataParallel as DDP  # Import DDP

from trainer import PathwayTrainer, FocalLoss  # Import the base trainer and loss


class DDPPathwayTrainer(PathwayTrainer):
    """DDP-enhanced trainer for multi-label pathway classification"""

    def __init__(self, model, train_dataset, val_dataset, train_sampler, val_sampler, 
                 rank, world_size, config, learning_rate=1e-4, batch_size=128, 
                 num_epochs=2, gradient_accumulation_steps=1, k_pathways=2, 
                 device=None, use_ddp=None, ddp_port=None):

        # Store DDP-specific attributes
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = device or f'cuda:{rank}'

        # Extract training parameters from config
        training_config = self.config.get('training', {})
        learning_rate = float(training_config.get('learning_rate', learning_rate))
        batch_size = training_config.get('batch_size', batch_size)
        self.num_epochs = training_config.get('num_epochs', num_epochs)
        self.gradient_accumulation_steps = training_config.get('gradient_accumulation_steps', gradient_accumulation_steps)
        self.k_pathways = training_config.get('k_pathways', k_pathways)

        # Store datasets and samplers
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler

        # Move model to device (should already be wrapped with DDP)
        self.model = model.to(self.device)

        # Create DataLoaders with DDP samplers
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            sampler=self.train_sampler,
            collate_fn=self._collate_fn, 
            num_workers=4, 
            pin_memory=True, 
            persistent_workers=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=batch_size, 
            sampler=self.val_sampler,
            collate_fn=self._collate_fn, 
            num_workers=2
        )
        
        # Initialize metrics calculator only on rank 0
        if self.rank == 0:
            from metrics import ComprehensiveMetrics
            self.metrics_calc = ComprehensiveMetrics(
                pathway_names=self.train_dataset.target_pathways
            )
        
        # Loss function
        self.criterion = FocalLoss(alpha=1.0, gamma=2.0)
        
        # Optimizer - handle DDP model parameters
        if self.model.module.freeze_geneformer:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
        else:
            # Get classifier and geneformer parameters
            try:
                classifier_params = list(self.model.module.classifier.parameters())
                geneformer_params = list(self.model.module.geneformer.parameters())
            except AttributeError:
                # Fallback: separate params manually
                geneformer_param_ids = {id(p) for p in self.model.module.geneformer.parameters()}
                classifier_params = [p for p in self.model.parameters() if id(p) not in geneformer_param_ids]
                geneformer_params = list(self.model.module.geneformer.parameters())

            self.optimizer = torch.optim.AdamW([
                {'params': classifier_params, 'lr': learning_rate},
                {'params': geneformer_params, 'lr': learning_rate * 0.1}
            ], weight_decay=0.01)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
        self.scaler = torch.cuda.amp.GradScaler()

        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}

        


    def train_epoch(self):
        """DDP-enabled training epoch"""
        self.model.train()
        
        # Set epoch for distributed sampler
        self.train_sampler.set_epoch(self.current_epoch)
        
        total_loss = 0.0
        self.optimizer.zero_grad()
        
        # Only show progress bar on rank 0
        if self.rank == 0:
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} Training")
        else:
            progress_bar = self.train_loader

        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                logits = self.model(input_ids, attention_mask)
                labels = labels.float()
                loss = self.criterion(logits, labels)
            
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.gradient_accumulation_steps

            if self.rank == 0 and hasattr(progress_bar, 'set_postfix'):
                progress_bar.set_postfix({'loss': loss.item()})
        
        # Average loss across all processes
        avg_loss = total_loss / len(self.train_loader)
        
        # Synchronize loss across processes
        loss_tensor = torch.tensor(avg_loss, device=self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / self.world_size
        
        return avg_loss

    def validate(self):
        """DDP-enabled validation"""
        self.model.eval()
        total_loss = 0.0
        all_preds_threshold = []
        all_preds_topk = []
        all_probs = []
        all_labels = []
        all_dataset_ids = []
        all_cell_ids = []

        # Only show progress bar on rank 0
        if self.rank == 0:
            progress_bar = tqdm(self.val_loader, desc="Validation")
        else:
            progress_bar = self.val_loader

        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device).float()

                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                probs = torch.sigmoid(logits)
                preds_threshold = (probs >= 0.5).long()
                preds_topk = self._get_topk_predictions(logits, self.k_pathways)

                all_probs.append(probs.cpu().numpy())
                all_preds_threshold.append(preds_threshold.cpu().numpy())
                all_preds_topk.append(preds_topk.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_dataset_ids.extend(batch['dataset_ids'])
                all_cell_ids.extend(batch['cell_ids'])

                if batch_idx == 0 and self.rank == 0:
                    self._debug_print_sample_batch(
                        labels=labels, logits=logits, probs=probs,
                        preds_threshold=preds_threshold, preds_topk=preds_topk,
                        dataset_ids=batch['dataset_ids'], cell_ids=batch['cell_ids'],
                        num_samples=3
                    )

                if self.rank == 0 and hasattr(progress_bar, 'set_postfix'):
                    progress_bar.set_postfix({'loss': loss.item()})

        # Gather results from all processes
        all_probs = self._gather_from_all_processes(np.vstack(all_probs))
        all_preds_threshold = self._gather_from_all_processes(np.vstack(all_preds_threshold))
        all_preds_topk = self._gather_from_all_processes(np.vstack(all_preds_topk))
        all_labels = self._gather_from_all_processes(np.vstack(all_labels))

        # Average loss across processes
        avg_loss = total_loss / len(self.val_loader)
        loss_tensor = torch.tensor(avg_loss, device=self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / self.world_size

        # Calculate metrics only on rank 0 to avoid redundant computation
        if self.rank == 0:
            # Threshold-based metrics
            subset_acc_threshold = accuracy_score(all_labels, all_preds_threshold)
            hamming_threshold = hamming_loss(all_labels, all_preds_threshold)
            micro_f1_threshold = f1_score(all_labels, all_preds_threshold, average="micro", zero_division=0)
            macro_f1_threshold = f1_score(all_labels, all_preds_threshold, average="macro", zero_division=0)
            
            # Top-k metrics
            subset_acc_topk = accuracy_score(all_labels, all_preds_topk)
            hamming_topk = hamming_loss(all_labels, all_preds_topk)
            micro_f1_topk = f1_score(all_labels, all_preds_topk, average="micro", zero_division=0)
            macro_f1_topk = f1_score(all_labels, all_preds_topk, average="macro", zero_division=0)

            try:
                auc = roc_auc_score(all_labels, all_probs, average="macro")
            except ValueError:
                auc = np.nan

            metrics = {
                "threshold_predictions": {
                    "subset_accuracy": subset_acc_threshold,
                    "hamming_loss": hamming_threshold,
                    "micro_f1": micro_f1_threshold,
                    "macro_f1": macro_f1_threshold,
                },
                f"top{self.k_pathways}_predictions": {
                    "subset_accuracy": subset_acc_topk,
                    "hamming_loss": hamming_topk,
                    "micro_f1": micro_f1_topk,
                    "macro_f1": macro_f1_topk,
                },
                "roc_auc_macro": auc,
            }
        else:
            metrics = None

        return avg_loss, metrics, all_preds_topk, all_probs, all_labels, all_dataset_ids

    def _gather_from_all_processes(self, data):
        """Gather data from all processes"""
        if not dist.is_initialized():
            return data
            
        # Convert to tensor
        data_tensor = torch.from_numpy(data).to(self.device)
        
        # Gather tensor sizes first
        size_tensor = torch.tensor(data_tensor.shape, device=self.device)
        size_list = [torch.zeros_like(size_tensor) for _ in range(self.world_size)]
        dist.all_gather(size_list, size_tensor)
        
        # Flatten data for gathering
        flat_data = data_tensor.flatten()
        
        # Gather all flattened data
        gathered_flat = [torch.zeros(torch.prod(size).item(), dtype=flat_data.dtype, device=self.device) 
                        for size in size_list]
        dist.all_gather(gathered_flat, flat_data)
        
        # Reshape and concatenate
        gathered_data = []
        for i, flat in enumerate(gathered_flat):
            shape = size_list[i].cpu().numpy()
            gathered_data.append(flat.reshape(shape))
        
        # Concatenate along batch dimension
        result = torch.cat(gathered_data, dim=0).cpu().numpy()
        return result

    def train(self, save_path=None):
        """DDP-enabled training loop"""
        if self.rank == 0:
            print("Starting DDP pathway classification training...")
            print(f"World size: {self.world_size}")
            print(f"Training samples: {len(self.train_dataset)}")
            print(f"Validation samples: {len(self.val_dataset)}")
            print(f"Geneformer frozen: {self.model.module.freeze_geneformer}")
            print(f"Number of pathways: {self.model.module.num_pathways}")
            print(f"Using top-{self.k_pathways} predictions")
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_metrics, _, _, _, _ = self.validate()

            # Only process results on rank 0
            if self.rank == 0:
                monitor_metric = val_metrics[f"top{self.k_pathways}_predictions"]["macro_f1"]
                self.scheduler.step(monitor_metric)

                # Save history
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['val_metrics'].append(val_metrics)

                # Print results
                print(f"\nEpoch {epoch+1}/{self.num_epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                
                print("  Threshold (≥0.5) Metrics:")
                thresh_metrics = val_metrics["threshold_predictions"]
                for k, v in thresh_metrics.items():
                    print(f"    {k}: {v:.4f}")
                
                print(f"  Top-{self.k_pathways} Metrics:")
                topk_metrics = val_metrics[f"top{self.k_pathways}_predictions"]
                for k, v in topk_metrics.items():
                    print(f"    {k}: {v:.4f}")
                
                if not np.isnan(val_metrics["roc_auc_macro"]):
                    print(f"  ROC AUC (macro): {val_metrics['roc_auc_macro']:.4f}")

                # Save best model
                current_best = val_metrics[f"top{self.k_pathways}_predictions"]["macro_f1"]
                if current_best > self.best_val_acc:
                    self.best_val_acc = current_best
                    if save_path:
                        self.save_model(save_path, epoch, val_metrics)
                    print(f"  New best validation Top-{self.k_pathways} Macro F1: {current_best:.4f}")
        
        if self.rank == 0:
            print(f"Training completed. Best validation Top-{self.k_pathways} Macro F1: {self.best_val_acc:.4f}")
        
        return self.history

    def save_model(self, save_path, epoch, metrics):
        """Save model checkpoint (only on rank 0)"""
        if self.rank != 0:
            return
            
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict(),  # Save underlying model, not DDP wrapper
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'metrics': metrics,
            'k_pathways': self.k_pathways,
        }
        
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")