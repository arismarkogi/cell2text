# test_evaluation.py
import torch
from dataset import MultiDatasetPathwayDataset
from classifier import GeneformerPathwayClassifier
from trainer import PathwayTrainer
import yaml
import os
import numpy as np

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Load configuration
    config = load_config('configs/default.yaml')
    
    # Create test dataset
    print("Loading test dataset...")
    test_dataset = MultiDatasetPathwayDataset(
        config['data']['base_data_path'], 
        split='test'
    )
    
    # Create model with same architecture as training
    model = GeneformerPathwayClassifier(
        config['model']['geneformer_model_path'],
        num_pathways=test_dataset.num_pathways,
        freeze_geneformer=config['model']['freeze_geneformer']
    )
    
    # Create trainer (we'll use it just for evaluation)
    trainer = PathwayTrainer(
        model=model,
        train_dataset=test_dataset,  # Dummy, won't be used
        val_dataset=test_dataset,    # Dummy, won't be used
        batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        num_epochs=1,  # Not used for evaluation
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps']
    )
    
    # Load the trained model checkpoint with weights_only=False
    checkpoint_path = "results/best_model.pt"
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # Use custom loading function to handle the weights_only issue
    def load_model_custom(load_path):
        checkpoint = torch.load(load_path, map_location=trainer.device, weights_only=False)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.best_val_acc = checkpoint['best_val_acc']
        trainer.history = checkpoint['history']
        trainer.current_epoch = checkpoint['epoch']
        
        # Load k_pathways if available
        if 'k_pathways' in checkpoint:
            trainer.k_pathways = checkpoint['k_pathways']
        
        print(f"Model loaded from {load_path}")
        print(f"Best validation Top-{trainer.k_pathways} Macro F1: {trainer.best_val_acc:.4f}")
        return checkpoint['metrics']
    
    # Load using custom function
    metrics = load_model_custom(checkpoint_path)
    
    # Run evaluation on test set
    print("Running evaluation on test set...")
    
    # Use a simplified evaluation approach to avoid the method signature issue
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False,
        collate_fn=trainer._collate_fn,
        num_workers=2
    )
    
    trainer.model.eval()
    all_logits = []
    all_labels = []
    all_dataset_ids = []
    all_cell_ids = []

    print("Processing test batches...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(trainer.device)
            attention_mask = batch['attention_mask'].to(trainer.device)
            labels = batch['labels'].to(trainer.device)
            
            logits = trainer.model(input_ids, attention_mask)
            
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
        top_indices = np.argsort(all_logits[i])[-trainer.k_pathways:]
        all_preds_topk[i, top_indices] = 1
    
    # Calculate metrics manually to avoid the method signature issue
    from sklearn.metrics import accuracy_score, f1_score, hamming_loss
    
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
    
    # Print results
    print("\n" + "="*60)
    print("TEST SET EVALUATION RESULTS")
    print("="*60)
    print(f"Threshold (≥0.5) Metrics:")
    print(f"  Subset Accuracy: {subset_acc_threshold:.4f}")
    print(f"  Hamming Loss: {hamming_threshold:.4f}")
    print(f"  Micro F1: {micro_f1_threshold:.4f}")
    print(f"  Macro F1: {macro_f1_threshold:.4f}")
    
    print(f"\nTop-{trainer.k_pathways} Metrics:")
    print(f"  Subset Accuracy: {subset_acc_topk:.4f}")
    print(f"  Hamming Loss: {hamming_topk:.4f}")
    print(f"  Micro F1: {micro_f1_topk:.4f}")
    print(f"  Macro F1: {macro_f1_topk:.4f}")
    
    # Save predictions to CSV
    print("\nSaving predictions to CSV...")
    os.makedirs("test_results", exist_ok=True)
    
    import pandas as pd
    
    # Create DataFrame with predictions
    predictions_df = pd.DataFrame({
        'cell_id': all_cell_ids,
        'dataset_id': all_dataset_ids,
    })
    
    # Add columns for each pathway
    num_pathways = all_labels.shape[1]
    for i in range(num_pathways):
        predictions_df[f'pathway_{i}_true'] = all_labels[:, i]
        predictions_df[f'pathway_{i}_prob'] = all_probs[:, i]
        predictions_df[f'pathway_{i}_pred_threshold'] = all_preds_threshold[:, i]
        predictions_df[f'pathway_{i}_pred_top{trainer.k_pathways}'] = all_preds_topk[:, i]
    
    # Save to CSV
    csv_path = "test_results/predictions.csv"
    predictions_df.to_csv(csv_path, index=False)
    print(f"Predictions saved to {csv_path}")
    
    # Save metrics to text file
    with open("test_results/metrics.txt", "w") as f:
        f.write("TEST SET EVALUATION METRICS\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Threshold (≥0.5) Metrics:\n")
        f.write(f"  Subset Accuracy: {subset_acc_threshold:.4f}\n")
        f.write(f"  Hamming Loss: {hamming_threshold:.4f}\n")
        f.write(f"  Micro F1: {micro_f1_threshold:.4f}\n")
        f.write(f"  Macro F1: {macro_f1_threshold:.4f}\n\n")
        
        f.write(f"Top-{trainer.k_pathways} Metrics:\n")
        f.write(f"  Subset Accuracy: {subset_acc_topk:.4f}\n")
        f.write(f"  Hamming Loss: {hamming_topk:.4f}\n")
        f.write(f"  Micro F1: {micro_f1_topk:.4f}\n")
        f.write(f"  Macro F1: {macro_f1_topk:.4f}\n")
    
    print(f"Metrics saved to test_results/metrics.txt")
    print("\nEvaluation Complete!")

if __name__ == "__main__":
    main()