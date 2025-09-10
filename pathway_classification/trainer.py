import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, roc_auc_score
import os
import json
from tqdm import tqdm


import warnings
warnings.filterwarnings('ignore')



class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # BCE with logits
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)
        # Compute focal weight
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class PathwayTrainer:
    """Enhanced trainer for multi-class pathway classification"""
    
    def __init__(self, model, train_dataset, val_dataset, 
                 learning_rate=1e-4, batch_size=16, num_epochs=50,
                 gradient_accumulation_steps=4, device='cuda'):
        
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        
        # Use CrossEntropyLoss for multi-class classification
        self.criterion = FocalLoss(alpha=1.0, gamma=2.0)
        learning_rate = float(learning_rate)
        # Optimizer
        if model.freeze_geneformer:
            trainable_params = []
            trainable_params.extend([p for p in model.parameters() if p.requires_grad])
            self.optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
        else:
            classifier_params = list(model.parameters()) 
            geneformer_params = list(model.geneformer.parameters())
            self.optimizer = torch.optim.AdamW([
                {'params': classifier_params, 'lr': learning_rate},
                {'params': geneformer_params, 'lr': learning_rate * 0.1}
            ], weight_decay=0.01)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
        

        self.gradient_accumulation_steps = gradient_accumulation_steps
        # Enable automatic mixed precision
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Adjust effective batch size
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=self._collate_fn, 
            num_workers=16,  # Increase workers
            pin_memory=True,  # Speed up GPU transfer
            persistent_workers=True  # Keep workers alive
        )


        
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, 
            collate_fn=self._collate_fn, num_workers=2
        )
        
        # Training state
        self.current_epoch = 0
        self.num_epochs = num_epochs
        self.best_val_acc = 0.0
        self.history = {
            'train_loss': [], 'val_loss': [], 
            'val_pathway1_acc': [], 'val_pathway2_acc': [],
            'val_both_acc': [], 'val_metrics': []
        }

        
        # # Compile model for PyTorch 2.0+ (if available)
        # try:
        #     self.model = torch.compile(self.model, mode='max-autotune')
        #     print("Model compiled for faster execution")
        # except:
        #     print("Model compilation not available")
    
        
    def _collate_fn(self, batch):
        # Pre-compute max length
        max_len = max(len(item['input_ids']) for item in batch)
        batch_size = len(batch)

        input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        attention_masks = torch.zeros(batch_size, max_len, dtype=torch.long)

        # Detect number of pathways dynamically
        num_pathways = len(batch[0]['labels'])
        labels = torch.zeros(batch_size, num_pathways, dtype=torch.float)

        dataset_ids = []
        cell_ids = []

        for i, item in enumerate(batch):
            seq_len = len(item['input_ids'])
            input_ids[i, :seq_len] = item['input_ids']
            attention_masks[i, :seq_len] = item['attention_mask']
            labels[i, :num_pathways] = torch.tensor(item['labels'], dtype=torch.float)
            dataset_ids.append(item['dataset_id'])
            cell_ids.append(item['cell_id'])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'labels': labels,
            'dataset_ids': dataset_ids,
            'cell_ids': cell_ids
        }   

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        
        # Zero gradients at the start
        self.optimizer.zero_grad()
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} Training")

        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            # Use automatic mixed precision
            with torch.cuda.amp.autocast():
                logits = self.model(input_ids, attention_mask)  # Single output now
                loss = self.criterion(logits, labels)
            
            # Scaled backward pass
            self.scaler.scale(loss).backward()
            
            # Update weights every N steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping with scaler
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.gradient_accumulation_steps

            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate the model with richer metrics"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []
        all_dataset_ids = []
        all_cell_ids = []  # <-- Add this to collect cell_ids

        progress_bar = tqdm(self.val_loader, desc="Validation")

        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device).float()

                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long()

                all_probs.append(probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_dataset_ids.extend(batch['dataset_ids'])
                all_cell_ids.extend(batch['cell_ids'])  # <-- Collect cell_ids

                # >>> DEBUG: Print first batch only <<<
                if batch_idx == 0:
                    self._debug_print_sample_batch(
                        labels=labels,
                        logits=logits,
                        probs=probs,
                        preds=preds,
                        dataset_ids=batch['dataset_ids'],
                        cell_ids=batch['cell_ids'],
                        num_samples=3
                    )

                progress_bar.set_postfix({'loss': loss.item()})

        # Concatenate
        all_probs = np.vstack(all_probs)
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        avg_loss = total_loss / len(self.val_loader)

        # --- Metrics ---
        num_pathways = all_labels.shape[1]

        # Subset accuracy (all labels must match per sample)
        subset_acc = accuracy_score(all_labels, all_preds)

        # Hamming loss
        hamming = hamming_loss(all_labels, all_preds)

        # F1 scores
        micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        # AUROC (macro) – only if both classes present
        try:
            auc = roc_auc_score(all_labels, all_probs, average="macro")
        except ValueError:
            auc = np.nan

        metrics = {
            "subset_accuracy": subset_acc,
            "hamming_loss": hamming,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "roc_auc_macro": auc,
        }

        # --- Special breakdown for 2-pathway case ---
        if num_pathways == 2:
            both_correct = np.sum(
                np.all(all_labels == all_preds, axis=1)
            )
            one_correct = np.sum(
                np.sum(all_labels == all_preds, axis=1) == 1
            )
            none_correct = np.sum(
                np.sum(all_labels == all_preds, axis=1) == 0
            )
            total = len(all_labels)

            breakdown = {
                "both_correct": both_correct / total,
                "one_correct": one_correct / total,
                "none_correct": none_correct / total,
            }
            metrics.update(breakdown)

        return avg_loss, metrics, all_preds, all_probs, all_labels, all_dataset_ids

    
    def train(self, save_path=None):
        """Full training loop"""
        print("Starting pathway classification training...")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        print(f"Geneformer frozen: {self.model.freeze_geneformer}")
        print(f"Number of pathways: {self.model.num_pathways}")
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_metrics, _, _, _, _ = self.validate()

            # Use macro F1 for scheduler
            monitor_metric = val_metrics.get("macro_f1", 0.0)
            self.scheduler.step(monitor_metric)

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)

            # Print results
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            for k, v in val_metrics.items():
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

            # Save best model (based on macro F1)
            if val_metrics["macro_f1"] > self.best_val_acc:
                self.best_val_acc = val_metrics["macro_f1"]
                if save_path:
                    self.save_model(save_path, epoch, val_metrics)
                print(f"  New best validation Macro F1: {val_metrics['macro_f1']:.4f}")
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)
            
            # Print results
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            
        
        print(f"Training completed. Best validation Both Accuracy: {self.best_val_acc:.4f}")
        return self.history
    
    def save_model(self, save_path, epoch, metrics):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'metrics': metrics,
        }
        
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")

    def _debug_print_sample_batch(self, labels, logits, probs, preds, dataset_ids, cell_ids, num_samples=3):
        """
        Print sample predictions vs targets for manual inspection.
        Useful for sanity-checking model behavior.
        """
        print("\n" + "="*60)
        print("DEBUG: Sample Predictions vs Targets")
        print("="*60)
        
        # Convert to numpy if tensors
        if torch.is_tensor(labels): labels = labels.cpu().numpy()
        if torch.is_tensor(logits): logits = logits.cpu().numpy()
        if torch.is_tensor(probs):  probs = probs.cpu().numpy()
        if torch.is_tensor(preds):  preds = preds.cpu().numpy()

        # Print first `num_samples` samples
        for i in range(min(num_samples, len(labels))):
            print(f"\nSample {i+1}:")
            print(f"  Dataset ID: {dataset_ids[i] if i < len(dataset_ids) else 'N/A'}")
            print(f"  Cell ID: {cell_ids[i] if i < len(cell_ids) else 'N/A'}")
            print(f"  Labels (true):    {labels[i]}")
            print(f"  Logits:           {logits[i]}")
            print(f"  Probs (sigmoid):  {probs[i]}")
            print(f"  Predictions (≥0.5): {preds[i]}")
            
            # Check if prediction matches
            match = np.array_equal(labels[i], preds[i])
            print(f"  ✅ Match: {match}")
        print("="*60 + "\n")
    
    def evaluate_test_set(self, test_dataset, batch_size=32, save_results=True, save_path=None):
        """Comprehensive test set evaluation"""
        print("Evaluating on test set...")
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=self._collate_fn, num_workers=2
        )
        
        self.model.eval()
        all_logits = []
        all_labels = []
        all_dataset_ids = []
        all_cell_ids = []

        progress_bar = tqdm(test_loader, desc="Test Evaluation")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_dataset_ids.extend(batch['dataset_ids'])
                all_cell_ids.extend(batch['cell_ids'])
                
                if batch_idx % 10 == 0:
                    print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")
        
        # Process results
        all_logits = np.vstack(all_logits)
        all_labels = np.vstack(all_labels)
        all_probs = 1 / (1 + np.exp(-all_logits))  # Sigmoid
        
        # Calculate comprehensive metrics
        test_metrics = self.metrics_calc.calculate_all_metrics(all_labels, all_probs)
        
        # Print detailed results
        self._print_test_results(test_metrics, len(all_labels))
        
        # Prepare results dictionary
        test_results = {
            'metrics': test_metrics,
            'predictions_proba': all_probs,
            'predictions_binary': (all_probs >= 0.5).astype(int),
            'labels': all_labels,
            'logits': all_logits,
            'dataset_ids': all_dataset_ids,
            'cell_ids': all_cell_ids
        }
        
        # Save results if requested
        if save_results and save_path:
            self._save_test_results(test_results, save_path)
        
        return test_results
    
    def _print_test_results(self, metrics, n_samples):
        """Print comprehensive test results"""
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST SET RESULTS")
        print("="*80)
        print(f"Total samples: {n_samples}")
        
        # Overall metrics
        overall = metrics['overall']
        print(f"\nOVERALL METRICS:")
        print(f"  Subset Accuracy (exact match): {overall['subset_accuracy']:.4f}")
        print(f"  Hamming Loss: {overall['hamming_loss']:.4f}")
        print(f"  Micro F1: {overall['micro_f1']:.4f}")
        print(f"  Macro F1: {overall['macro_f1']:.4f}")
        print(f"  Weighted F1: {overall['weighted_f1']:.4f}")
        print(f"  Total Positives: {overall['total_positives']}")
        print(f"  Total Predictions: {overall['total_predictions']}")
        print(f"  Correct Predictions: {overall['correct_predictions']}")
        print(f"  Micro Precision: {overall['micro_precision']:.4f}")
        print(f"  Macro Precision: {overall['macro_precision']:.4f}")
        print(f"  Micro Recall: {overall['micro_recall']:.4f}")
        print(f"  Macro Recall: {overall['macro_recall']:.4f}")
        
        # Pathway-specific metrics
        for pathway in ['pathway1', 'pathway2']:
            if pathway in metrics:
                p_metrics = metrics[pathway]
                print(f"\n{pathway.upper()} METRICS:")
                print(f"  Accuracy: {p_metrics['accuracy']:.4f}")
                print(f"  Precision: {p_metrics['precision']:.4f}")
                print(f"  Recall: {p_metrics['recall']:.4f}")
                print(f"  F1 Score: {p_metrics['f1_score']:.4f}")
                print(f"  Specificity: {p_metrics['specificity']:.4f}")
                print(f"  Balanced Accuracy: {p_metrics['balanced_accuracy']:.4f}")
                
                if not np.isnan(p_metrics.get('roc_auc', np.nan)):
                    print(f"  ROC AUC: {p_metrics['roc_auc']:.4f}")
                    print(f"  PR AUC: {p_metrics['pr_auc']:.4f}")
                    print(f"  Log Loss: {p_metrics['log_loss']:.4f}")
                    print(f"  Brier Score: {p_metrics['brier_score']:.4f}")
                
                print(f"  Matthews Correlation: {p_metrics['matthews_corrcoef']:.4f}")
                print(f"  Cohen's Kappa: {p_metrics['cohen_kappa']:.4f}")
                
                # Confusion matrix stats
                if 'true_positives' in p_metrics:
                    print(f"  True Positives: {p_metrics['true_positives']}")
                    print(f"  False Positives: {p_metrics['false_positives']}")
                    print(f"  True Negatives: {p_metrics['true_negatives']}")
                    print(f"  False Negatives: {p_metrics['false_negatives']}")
                
                # Optimal thresholds
                if 'optimal_threshold_youden' in p_metrics:
                    print(f"  Optimal Threshold (Youden): {p_metrics['optimal_threshold_youden']:.4f}")
                    print(f"  Optimal Threshold (F1): {p_metrics['optimal_threshold_f1']:.4f}")
        
        print("="*80)
    
    def _save_test_results(self, results, save_path):
        """Save comprehensive test results"""
        import pickle
        
        # Create save directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save the full results
        with open(f"{save_path}_full_results.pkl", 'wb') as f:
            pickle.dump(results, f)
        
        # Save metrics as JSON for easy reading
        metrics_for_json = {}
        for key, value in results['metrics'].items():
            if isinstance(value, dict):
                metrics_for_json[key] = {}
                for k, v in value.items():
                    if isinstance(v, (np.integer, int)):
                        metrics_for_json[key][k] = int(v)
                    elif isinstance(v, (np.floating, float)):
                        if np.isnan(v):
                            metrics_for_json[key][k] = None
                        else:
                            metrics_for_json[key][k] = float(v)
                    else:
                        metrics_for_json[key][k] = v
        
        with open(f"{save_path}_metrics.json", 'w') as f:
            json.dump(metrics_for_json, f, indent=2)
        
        # Save predictions as CSV
        predictions_df = pd.DataFrame({
            'cell_id': results['cell_ids'],
            'dataset_id': results['dataset_ids'],
            'pathway1_true': results['labels'][:, 0],
            'pathway2_true': results['labels'][:, 1],
            'pathway1_prob': results['predictions_proba'][:, 0],
            'pathway2_prob': results['predictions_proba'][:, 1],
            'pathway1_pred': results['predictions_binary'][:, 0],
            'pathway2_pred': results['predictions_binary'][:, 1],
        })
        predictions_df.to_csv(f"{save_path}_predictions.csv", index=False)
        
        print(f"Results saved to:")
        print(f"  {save_path}_full_results.pkl")
        print(f"  {save_path}_metrics.json")
        print(f"  {save_path}_predictions.csv")
    
    
    
    def load_model(self, load_path):
        """Load model checkpoint"""
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_f1 = checkpoint['best_val_f1']
        self.history = checkpoint['history']
        self.current_epoch = checkpoint['epoch']
        
        print(f"Model loaded from {load_path}")
        print(f"Best validation F1: {self.best_val_f1:.4f}")
        return checkpoint['metrics']
