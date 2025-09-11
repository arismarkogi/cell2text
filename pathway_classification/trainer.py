import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, roc_auc_score
import os
import json
from tqdm import tqdm
from metrics import ComprehensiveMetrics  # Import the metrics class

import warnings
warnings.filterwarnings('ignore')


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification - Fixed version
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        # Compute binary cross entropy manually to maintain gradient flow
        # BCE = -[y*log(p) + (1-y)*log(1-p)]
        eps = 1e-7  # Small epsilon to prevent log(0)
        bce_loss = -(targets * torch.log(probs + eps) + (1 - targets) * torch.log(1 - probs + eps))
        
        # Compute focal weight: (1-pt)^gamma where pt is the probability of the correct class
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        # Apply focal weight to BCE loss
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class PathwayTrainer:
    """Enhanced trainer for multi-label pathway classification with top-k predictions"""
    
    def __init__(self, model, train_dataset, val_dataset, 
                 learning_rate=1e-4, batch_size=16, num_epochs=50,
                 gradient_accumulation_steps=4, device='cuda', k_pathways=2):
        
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.k_pathways = k_pathways  # Number of top pathways to predict
        
        # Initialize metrics calculator
        self.metrics_calc = ComprehensiveMetrics(
            pathway_names=train_dataset.target_pathways
        )
        
        # Use Focal Loss for multi-label classification
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
            'train_loss': [], 'val_loss': [], 'val_metrics': []
        }
        
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

    def _get_topk_predictions(self, logits, k=None):
        """
        Get top-k predictions based on logits
        """
        if k is None:
            k = self.k_pathways
            
        batch_size, num_pathways = logits.shape
        predictions = torch.zeros_like(logits)
        
        # Get top-k indices for each sample
        _, top_indices = torch.topk(logits, k, dim=1)
        
        # Set top-k positions to 1
        predictions.scatter_(1, top_indices, 1)
        
        return predictions.long()

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
                # Ensure labels are float32 for focal loss
                labels = labels.float()
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
        """Validate the model with top-k predictions and richer metrics"""
        self.model.eval()
        total_loss = 0.0
        all_preds_threshold = []  # 0.5 threshold predictions
        all_preds_topk = []       # top-k predictions
        all_probs = []
        all_labels = []
        all_dataset_ids = []
        all_cell_ids = []

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
                preds_threshold = (probs >= 0.5).long()
                preds_topk = self._get_topk_predictions(logits, self.k_pathways)

                all_probs.append(probs.cpu().numpy())
                all_preds_threshold.append(preds_threshold.cpu().numpy())
                all_preds_topk.append(preds_topk.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_dataset_ids.extend(batch['dataset_ids'])
                all_cell_ids.extend(batch['cell_ids'])

                # DEBUG: Print first batch only
                if batch_idx == 0:
                    self._debug_print_sample_batch(
                        labels=labels,
                        logits=logits,
                        probs=probs,
                        preds_threshold=preds_threshold,
                        preds_topk=preds_topk,
                        dataset_ids=batch['dataset_ids'],
                        cell_ids=batch['cell_ids'],
                        num_samples=3
                    )

                progress_bar.set_postfix({'loss': loss.item()})

        # Concatenate
        all_probs = np.vstack(all_probs)
        all_preds_threshold = np.vstack(all_preds_threshold)
        all_preds_topk = np.vstack(all_preds_topk)
        all_labels = np.vstack(all_labels)

        avg_loss = total_loss / len(self.val_loader)

        # --- Metrics for both prediction methods ---
        
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

        # AUROC (macro) – only if both classes present
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

        return avg_loss, metrics, all_preds_topk, all_probs, all_labels, all_dataset_ids
    
    def train(self, save_path=None):
        """Full training loop"""
        print("Starting pathway classification training...")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        print(f"Geneformer frozen: {self.model.freeze_geneformer}")
        print(f"Number of pathways: {self.model.num_pathways}")
        print(f"Using top-{self.k_pathways} predictions")
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_metrics, _, _, _, _ = self.validate()

            # Use top-k macro F1 for scheduler (more appropriate for your use case)
            monitor_metric = val_metrics[f"top{self.k_pathways}_predictions"]["macro_f1"]
            self.scheduler.step(monitor_metric)

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)

            # Print results
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            
            # Print both threshold and top-k results
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

            # Save best model (based on top-k macro F1)
            current_best = val_metrics[f"top{self.k_pathways}_predictions"]["macro_f1"]
            if current_best > self.best_val_acc:
                self.best_val_acc = current_best
                if save_path:
                    self.save_model(save_path, epoch, val_metrics)
                print(f"  New best validation Top-{self.k_pathways} Macro F1: {current_best:.4f}")
        
        print(f"Training completed. Best validation Top-{self.k_pathways} Macro F1: {self.best_val_acc:.4f}")
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
            'k_pathways': self.k_pathways,
        }
        
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")

    def _debug_print_sample_batch(self, labels, logits, probs, preds_threshold, preds_topk, dataset_ids, cell_ids, num_samples=3):
        """
        Print sample predictions vs targets for manual inspection.
        Now shows both threshold and top-k predictions.
        """
        print("\n" + "="*70)
        print("DEBUG: Sample Predictions vs Targets")
        print("="*70)
        
        # Convert to numpy if tensors
        if torch.is_tensor(labels): labels = labels.cpu().numpy()
        if torch.is_tensor(logits): logits = logits.cpu().numpy()
        if torch.is_tensor(probs):  probs = probs.cpu().numpy()
        if torch.is_tensor(preds_threshold):  preds_threshold = preds_threshold.cpu().numpy()
        if torch.is_tensor(preds_topk):  preds_topk = preds_topk.cpu().numpy()

        # Print first `num_samples` samples
        for i in range(min(num_samples, len(labels))):
            print(f"\nSample {i+1}:")
            print(f"  Dataset ID: {dataset_ids[i] if i < len(dataset_ids) else 'N/A'}")
            print(f"  Cell ID: {cell_ids[i] if i < len(cell_ids) else 'N/A'}")
            print(f"  Labels (true):           {labels[i]}")
            print(f"  Logits:                  {logits[i]}")
            print(f"  Probs (sigmoid):         {probs[i]}")
            print(f"  Predictions (≥0.5):      {preds_threshold[i]}")
            print(f"  Predictions (top-{self.k_pathways}):      {preds_topk[i]}")
            
            # Check matches
            match_threshold = np.array_equal(labels[i], preds_threshold[i])
            match_topk = np.array_equal(labels[i], preds_topk[i])
            print(f"  ✅ Threshold Match: {match_threshold}")
            print(f"  ✅ Top-{self.k_pathways} Match: {match_topk}")
            
            # Show which pathways are predicted vs true
            true_pathways = np.where(labels[i] == 1)[0]
            pred_topk_pathways = np.where(preds_topk[i] == 1)[0]
            print(f"  True pathway indices: {true_pathways}")
            print(f"  Top-{self.k_pathways} predicted indices: {pred_topk_pathways}")
        print("="*70 + "\n")
    
    def evaluate_test_set(self, test_dataset, batch_size=32, save_results=True, save_path=None):
        """Comprehensive test set evaluation with both threshold and top-k predictions"""
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
        
        # Get both types of predictions
        all_preds_threshold = (all_probs >= 0.5).astype(int)
        
        # Top-k predictions
        all_preds_topk = np.zeros_like(all_logits, dtype=int)
        for i in range(len(all_logits)):
            top_indices = np.argsort(all_logits[i])[-self.k_pathways:]
            all_preds_topk[i, top_indices] = 1
        
        # Calculate comprehensive metrics using the metrics calculator for both methods
        test_metrics_threshold = self.metrics_calc.calculate_all_metrics(all_labels, all_probs, predictions=all_preds_threshold)
        test_metrics_topk = self.metrics_calc.calculate_all_metrics(all_labels, all_probs, predictions=all_preds_topk)
        
        # Combine metrics
        combined_metrics = {
            'threshold_predictions': test_metrics_threshold,
            f'top{self.k_pathways}_predictions': test_metrics_topk
        }
        
        # Print detailed results
        self._print_test_results(combined_metrics, len(all_labels))
        
        # Prepare results dictionary
        test_results = {
            'metrics': combined_metrics,
            'predictions_proba': all_probs,
            'predictions_threshold': all_preds_threshold,
            'predictions_topk': all_preds_topk,
            'labels': all_labels,
            'logits': all_logits,
            'dataset_ids': all_dataset_ids,
            'cell_ids': all_cell_ids,
            'k_pathways': self.k_pathways
        }
        
        # Save results if requested
        if save_results and save_path:
            self._save_test_results(test_results, save_path)
        
        return test_results
    
    def _print_test_results(self, metrics, n_samples):
        """Print comprehensive test results for both prediction methods"""
        print("\n" + "="*90)
        print("COMPREHENSIVE TEST SET RESULTS")
        print("="*90)
        print(f"Total samples: {n_samples}")
        
        for pred_method, method_metrics in metrics.items():
            print(f"\n{pred_method.upper().replace('_', ' ')} RESULTS:")
            print("-" * 50)
            
            # Overall metrics
            overall = method_metrics['overall']
            print(f"  OVERALL METRICS:")
            print(f"    Subset Accuracy (exact match): {overall['subset_accuracy']:.4f}")
            print(f"    Hamming Loss: {overall['hamming_loss']:.4f}")
            print(f"    Micro F1: {overall['micro_f1']:.4f}")
            print(f"    Macro F1: {overall['macro_f1']:.4f}")
            print(f"    Weighted F1: {overall['weighted_f1']:.4f}")
            print(f"    Total Positives: {overall['total_positives']}")
            print(f"    Total Predictions: {overall['total_predictions']}")
            print(f"    Correct Predictions: {overall['correct_predictions']}")
            print(f"    Micro Precision: {overall['micro_precision']:.4f}")
            print(f"    Macro Precision: {overall['macro_precision']:.4f}")
            print(f"    Micro Recall: {overall['micro_recall']:.4f}")
            print(f"    Macro Recall: {overall['macro_recall']:.4f}")
            
            # Show just a few pathway examples to keep output manageable
            pathway_keys = [k for k in method_metrics.keys() if k.startswith('pathway_') and k != 'pathway_names']
            sample_pathways = pathway_keys[:3]  # Show first 3 pathways as examples
            
            if sample_pathways:
                print(f"  SAMPLE PATHWAY METRICS (showing first 3):")
                for pathway in sample_pathways:
                    if pathway in method_metrics:
                        p_metrics = method_metrics[pathway]
                        print(f"    {pathway.upper()}:")
                        print(f"      F1 Score: {p_metrics['f1_score']:.4f}")
                        print(f"      Precision: {p_metrics['precision']:.4f}")
                        print(f"      Recall: {p_metrics['recall']:.4f}")
                        if not np.isnan(p_metrics.get('roc_auc', np.nan)):
                            print(f"      ROC AUC: {p_metrics['roc_auc']:.4f}")
        
        print("="*90)
    
    def _save_test_results(self, results, save_path):
        """Save comprehensive test results with both prediction methods"""
        import pickle
        
        # Create save directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save the full results
        with open(f"{save_path}_full_results.pkl", 'wb') as f:
            pickle.dump(results, f)
        
        # Save metrics as JSON for easy reading
        metrics_for_json = {}
        for pred_method, method_metrics in results['metrics'].items():
            metrics_for_json[pred_method] = {}
            for key, value in method_metrics.items():
                if isinstance(value, dict):
                    metrics_for_json[pred_method][key] = {}
                    for k, v in value.items():
                        if isinstance(v, (np.integer, int)):
                            metrics_for_json[pred_method][key][k] = int(v)
                        elif isinstance(v, (np.floating, float)):
                            if np.isnan(v):
                                metrics_for_json[pred_method][key][k] = None
                            else:
                                metrics_for_json[pred_method][key][k] = float(v)
                        else:
                            metrics_for_json[pred_method][key][k] = v
        
        with open(f"{save_path}_metrics.json", 'w') as f:
            json.dump(metrics_for_json, f, indent=2)
        
        # Save predictions as CSV
        predictions_df = pd.DataFrame({
            'cell_id': results['cell_ids'],
            'dataset_id': results['dataset_ids'],
        })
        
        # Add columns for each pathway
        num_pathways = results['labels'].shape[1]
        for i in range(num_pathways):
            predictions_df[f'pathway_{i}_true'] = results['labels'][:, i]
            predictions_df[f'pathway_{i}_prob'] = results['predictions_proba'][:, i]
            predictions_df[f'pathway_{i}_pred_threshold'] = results['predictions_threshold'][:, i]
            predictions_df[f'pathway_{i}_pred_top{results["k_pathways"]}'] = results['predictions_topk'][:, i]
        
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
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
        self.current_epoch = checkpoint['epoch']
        
        # Load k_pathways if available
        if 'k_pathways' in checkpoint:
            self.k_pathways = checkpoint['k_pathways']
        
        print(f"Model loaded from {load_path}")
        print(f"Best validation Top-{self.k_pathways} Macro F1: {self.best_val_acc:.4f}")
        return checkpoint['metrics']