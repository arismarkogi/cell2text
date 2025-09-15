import os
import torch
import argparse
from datasets import load_from_disk
from torch.utils.data import random_split

# Import your classes from the previous code
from classifier import GeneformerCellTypeClassifier
from dataset import CellTypeDataset
from trainer import CellTypeTrainer


def main():
    parser = argparse.ArgumentParser(description='Train Geneformer for cell type classification')
    parser.add_argument('--base_path', type=str, required=True,
                        help='Base path containing train/validation/test folders')
    parser.add_argument('--geneformer_path', type=str, default='ctheodoris/Geneformer',
                        help='Path to Geneformer model (local or Hugging Face ID)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save model checkpoints and results')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--min_samples_per_type', type=int, default=10,
                        help='Minimum samples per cell type')
    parser.add_argument('--freeze_geneformer', action='store_true',
                        help='Freeze Geneformer parameters during training')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = CellTypeDataset(
        base_path=args.base_path,
        split='train',
        min_samples_per_type=args.min_samples_per_type
    )
    
    val_dataset = CellTypeDataset(
        base_path=args.base_path,
        split='validation',
        min_samples_per_type=args.min_samples_per_type
    )
    
    test_dataset = CellTypeDataset(
        base_path=args.base_path,
        split='test',
        min_samples_per_type=args.min_samples_per_type
    )
    
    # Verify dataset consistency
    # assert train_dataset.num_cell_types == val_dataset.num_cell_types == test_dataset.num_cell_types, \
    #     "Inconsistent cell types across datasets!"
    
    print(f"\nDataset Statistics:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Unique cell types: {train_dataset.num_cell_types}")
    
    # Initialize model
    print("\nInitializing model...")
    model = GeneformerCellTypeClassifier(
        geneformer_model_path=args.geneformer_path,
        num_cell_types=train_dataset.num_cell_types,
        freeze_geneformer=args.freeze_geneformer,
        dropout_rate=0.1
    )
    
    # Initialize trainer
    trainer = CellTypeTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        device=args.device,
        use_class_weights=True,
        label_smoothing=0.1
    )
    
    # Train model
    print("\nStarting training...")
    trainer.train(save_path=os.path.join(args.output_dir, "best_model.pt"))
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate_test_set(
        test_dataset=test_dataset,
        save_results=True,
        save_path=os.path.join(args.output_dir, "test_results.pt")
    )
    
    print("\nTraining completed successfully!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Best validation F1 (Macro): {trainer.best_val_f1:.4f}")

    print("\nTest Set Results:")
    for metric, value in test_results.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()