import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, balanced_accuracy_score,
    top_k_accuracy_score
)
import os
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class CellTypeTrainer:
    """Trainer for single-label cell type classification"""
    
    def __init__(self, model, train_dataset, val_dataset, 
                 learning_rate=2e-5, batch_size=128, num_epochs=5,
                 gradient_accumulation_steps=4, device='cuda', 
                 use_class_weights=True, label_smoothing=0.1):
        
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.num_cell_types = train_dataset.num_cell_types
        self.idx_to_cell_type = train_dataset.idx_to_cell_type
        
        # Loss function with optional class weighting and label smoothing
        if use_class_weights:
            class_weights = train_dataset.get_class_weights()
            if class_weights is not None:
                class_weights = class_weights.to(device)
                print(f"Using class weights. Min weight: {class_weights.min():.4f}, Max weight: {class_weights.max():.4f}")
        else:
            class_weights = None
        
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        )
        
        learning_rate = float(learning_rate)
        
        # Optimizer with different learning rates for different parts
        if model.freeze_geneformer:
            # Only train the classifier head
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
            print(f"Training only classifier head with {sum(p.numel() for p in trainable_params)} parameters")
        else:
            # Train everything with different learning rates
            classifier_params = list(model.model.classifier.parameters())
            bert_params = list(model.model.bert.parameters())
            
            self.optimizer = torch.optim.AdamW([
                {'params': classifier_params, 'lr': learning_rate},
                {'params': bert_params, 'lr': learning_rate * 0.1}  # Lower LR for pre-trained weights
            ], weight_decay=0.01)
            print("Training full model with different learning rates")
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
        
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=self._collate_fn, 
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )

        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=self._collate_fn, 
            num_workers=4
        )
        
        # Training state
        self.current_epoch = 0
        self.num_epochs = num_epochs
        self.best_val_f1 = 0.0
        self.history = {
            'train_loss': [], 
            'train_accuracy': [],
            'val_loss': [], 
            'val_accuracy': [],
            'val_f1_macro': [],
            'val_f1_weighted': []
        }
        
    def _collate_fn(self, batch):
        """Collate function for single-label classification"""
        max_len = max(len(item['input_ids']) for item in batch)
        batch_size = len(batch)

        input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        attention_masks = torch.zeros(batch_size, max_len, dtype=torch.long)
        labels = torch.zeros(batch_size, dtype=torch.long)  # Single label per sample

        dataset_ids = []
        cell_ids = []

        for i, item in enumerate(batch):
            seq_len = len(item['input_ids'])
            input_ids[i, :seq_len] = item['input_ids']
            attention_masks[i, :seq_len] = item['attention_mask']
            labels[i] = item['labels']  # Single integer label
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
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        self.optimizer.zero_grad()
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
            
            # Scale loss for gradient accumulation
            scaled_loss = loss / self.gradient_accumulation_steps
            self.scaler.scale(scaled_loss).backward()
            
            # Update weights every N steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            # Calculate accuracy
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
            
            total_loss += loss.item()
            
            # Update progress bar
            current_accuracy = correct_predictions / total_predictions
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_accuracy:.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_accuracy = correct_predictions / total_predictions
        
        return avg_loss, avg_accuracy
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_dataset_ids = []
        
        progress_bar = tqdm(self.val_loader, desc="Validation")
        
        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                # Get predictions and probabilities
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_dataset_ids.extend(batch['dataset_ids'])
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        balanced_acc = balanced_accuracy_score(all_labels, all_predictions)
        f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        precision_macro = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
        recall_macro = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
        
        # Top-k accuracy (useful for cell type classification)
        all_probabilities = np.array(all_probabilities)
        top3_accuracy = top_k_accuracy_score(all_labels, all_probabilities, k=3)
        top5_accuracy = top_k_accuracy_score(all_labels, all_probabilities, k=min(5, self.num_cell_types))
        
        metrics = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'top3_accuracy': top3_accuracy,
            'top5_accuracy': top5_accuracy,
            'loss': avg_loss
        }
        
        return metrics, all_predictions, all_labels, all_probabilities, all_dataset_ids
    
    def train(self, save_path=None):
        """Full training loop"""
        print("Starting cell type classification training...")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        print(f"Number of cell types: {self.num_cell_types}")
        print(f"Geneformer frozen: {self.model.freeze_geneformer}")
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_metrics, _, _, _, _ = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_metrics['f1_macro'])
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_f1_macro'].append(val_metrics['f1_macro'])
            self.history['val_f1_weighted'].append(val_metrics['f1_weighted'])
            
            # Print results
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Val Balanced Accuracy: {val_metrics['balanced_accuracy']:.4f}")
            print(f"  Val F1 (Macro): {val_metrics['f1_macro']:.4f}")
            print(f"  Val F1 (Weighted): {val_metrics['f1_weighted']:.4f}")
            print(f"  Val Top-3 Accuracy: {val_metrics['top3_accuracy']:.4f}")
            print(f"  Val Top-5 Accuracy: {val_metrics['top5_accuracy']:.4f}")
            
            # Save best model
            if val_metrics['f1_macro'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1_macro']
                if save_path:
                    self.save_model(save_path, epoch, val_metrics)
                print(f"  New best validation F1 (Macro): {self.best_val_f1:.4f}")
        
        print(f"\nTraining completed. Best validation F1 (Macro): {self.best_val_f1:.4f}")
        return self.history
    
    def save_model(self, save_path, epoch, metrics):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_f1': self.best_val_f1,
            'history': self.history,
            'metrics': metrics,
            'num_cell_types': self.num_cell_types,
            'idx_to_cell_type': self.idx_to_cell_type,
        }
        
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, load_path, weights_only=False):
        """Load model checkpoint"""
        checkpoint = torch.load(load_path, map_location=self.device, weights_only=weights_only)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_f1 = checkpoint['best_val_f1']
        self.history = checkpoint['history']
        self.current_epoch = checkpoint['epoch']
        
        print(f"Model loaded from {load_path}")
        print(f"Best validation F1 (Macro): {self.best_val_f1:.4f}")
        return checkpoint['metrics']
    
    def evaluate_test_set(self, test_dataset, save_results=True, save_path=None):
        """Comprehensive test set evaluation"""
        print("Evaluating on test set...")
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=256, 
            shuffle=False,
            collate_fn=self._collate_fn, 
            num_workers=4
        )
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_dataset_ids = []
        all_cell_ids = []

        progress_bar = tqdm(test_loader, desc="Test Evaluation")
        
        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']
                
                logits = self.model(input_ids, attention_mask)
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_dataset_ids.extend(batch['dataset_ids'])
                all_cell_ids.extend(batch['cell_ids'])
        
        # Calculate comprehensive metrics
        test_metrics = self._calculate_comprehensive_metrics(
            all_labels, all_predictions, all_probabilities
        )
        
        # Print results
        self._print_test_results(test_metrics, len(all_labels))
        
        # Prepare results
        test_results = {
            'metrics': test_metrics,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities,
            'dataset_ids': all_dataset_ids,
            'cell_ids': all_cell_ids,
            'idx_to_cell_type': self.idx_to_cell_type
        }
        
        if save_results and save_path:
            self._save_test_results(test_results, save_path)
        
        return test_results
    
    def _calculate_comprehensive_metrics(self, labels, predictions, probabilities):
        """Calculate comprehensive metrics"""
        probabilities = np.array(probabilities)
        
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'balanced_accuracy': balanced_accuracy_score(labels, predictions),
            'f1_macro': f1_score(labels, predictions, average='macro', zero_division=0),
            'f1_weighted': f1_score(labels, predictions, average='weighted', zero_division=0),
            'f1_micro': f1_score(labels, predictions, average='micro', zero_division=0),
            'precision_macro': precision_score(labels, predictions, average='macro', zero_division=0),
            'precision_weighted': precision_score(labels, predictions, average='weighted', zero_division=0),
            'recall_macro': recall_score(labels, predictions, average='macro', zero_division=0),
            'recall_weighted': recall_score(labels, predictions, average='weighted', zero_division=0),
            'top3_accuracy': top_k_accuracy_score(labels, probabilities, k=3),
            'top5_accuracy': top_k_accuracy_score(labels, probabilities, k=min(5, self.num_cell_types)),
        }
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Per-class metrics
        class_report = classification_report(labels, predictions, output_dict=True, zero_division=0)
        metrics['per_class_metrics'] = class_report
        
        return metrics
    
    def _print_test_results(self, metrics, n_samples):
        """Print test results"""
        print("\n" + "="*60)
        print("TEST SET RESULTS")
        print("="*60)
        print(f"Total samples: {n_samples}")
        print(f"Number of cell types: {self.num_cell_types}")
        print()
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
        print(f"F1 Score (Weighted): {metrics['f1_weighted']:.4f}")
        print(f"F1 Score (Micro): {metrics['f1_micro']:.4f}")
        print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
        print(f"Precision (Weighted): {metrics['precision_weighted']:.4f}")
        print(f"Recall (Macro): {metrics['recall_macro']:.4f}")
        print(f"Recall (Weighted): {metrics['recall_weighted']:.4f}")
        print(f"Top-3 Accuracy: {metrics['top3_accuracy']:.4f}")
        print(f"Top-5 Accuracy: {metrics['top5_accuracy']:.4f   }")
        print("="*60 + "\n")

    def _save_test_results(self, results, save_path):
        """Save test results to disk"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save metrics as JSON
        metrics_path = save_path.replace('.pt', '_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(results['metrics'], f, indent=4)
        print(f"Test metrics saved to {metrics_path}")
        
        # Save detailed results as a DataFrame
        df = pd.DataFrame({
            'dataset_id': results['dataset_ids'],
            'cell_id': results['cell_ids'],
            'true_label': [results['idx_to_cell_type'][lbl] for lbl in results['labels']],
            'predicted_label': [results['idx_to_cell_type'][pred] for pred in results['predictions']],
            'predicted_label_idx': results['predictions'],
            'true_label_idx': results['labels'],
            'probabilities': results['probabilities']
        })
        
        results_path = save_path.replace('.pt', '_detailed_results.csv')
        df.to_csv(results_path, index=False)
        print(f"Detailed test results saved to {results_path}")
        
        # Save the entire results object as a PyTorch file
        torch.save(results, save_path)
        print(f"Full test results saved to {save_path}")
        print(f"Confusion matrix saved to {results_path.replace('.csv', '_confusion_matrix.csv')}")
        cm_df = pd.DataFrame(
            results['metrics']['confusion_matrix'], 
            index=[self.idx_to_cell_type[i] for i in range(self.num_cell_types)],
            columns=[self.idx_to_cell_type[i] for i in range(self.num_cell_types)]
        )
        cm_df.to_csv(results_path.replace('.csv', '_confusion_matrix.csv'))
        print("="*60)
        print("Test results saved successfully.")
        print("="*60)

    