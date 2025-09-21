"""
Main script for scGPT fine-tuning on classification tasks
"""
import os
import sys
import logging
import pickle
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

# Add paths (adjust according to your directory structure)
sys.path.insert(0, "../")

# Import scGPT components (adjust imports based on your scGPT installation)
#from scgpt.model import TransformerModel
#from scgpt.tokenizer.gene_tokenizer import GeneVocab

# Import our custom modules
from config import  get_task_specific_config
from model_setup import ModelManager
from data_loader import ScGPTDataLoader, create_dataloader
from trainer import ClassificationTrainer

# Suppress warnings
warnings.filterwarnings("ignore")

def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_logger(save_dir: Path) -> logging.Logger:
    """Setup logger"""
    logger = logging.getLogger("scGPT_finetune")
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    fh = logging.FileHandler(save_dir / "training.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def main():
    """Main training function"""
    
    # Configuration
    config = get_task_specific_config("celltype")  # Change task as needed
    config.validate()
    
    # Set device
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(f"Using device: {device}")
    
    # Set seed
    set_seed(config.seed)
    
    # Create save directory
    save_dir = Path(f"./save/scGPT_finetune_{config.classification_task}_{config.dataset_name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(save_dir)
    logger.info(f"Starting fine-tuning for task: {config.classification_task}")
    logger.info(f"Save directory: {save_dir}")
    
    # Save configuration
    config.to_json(save_dir / "config.json")
    
    # Data paths (UPDATE THESE PATHS TO YOUR DATA)
    data_paths = {
        "train": "train_data.h5ad",
        "val": "val_data.h5ad", 
        "test": "test_data.h5ad"
    }
    
    # Check if data files exist
    for split, path in data_paths.items():
        if not Path(path).exists():
            logger.error(f"Data file not found: {path}")
            raise FileNotFoundError(f"Please provide {split} data at {path}")
    
    # Initialize data loader
    data_loader = ScGPTDataLoader(config)
    logger.info("Loading and preprocessing data...")
    
    # Load data
    adata_all, adata_train, adata_val, adata_test = data_loader.load_data(
        data_paths["train"], data_paths["val"], data_paths["test"]
    )
    logger.info(f"Loaded data: Train={len(adata_train)}, Val={len(adata_val)}, Test={len(adata_test)}")
    
    # Setup vocabulary
    vocab_path = None
    if Path(config.load_model).exists():
        potential_vocab_path = Path(config.load_model) / "vocab.json"
        if potential_vocab_path.exists():
            vocab_path = str(potential_vocab_path)
    
    vocab = data_loader.setup_vocabulary(adata_all, vocab_path)
    logger.info(f"Vocabulary size: {len(vocab)}")
    
    # Filter genes by vocabulary
    adata_all = data_loader.filter_genes_by_vocab(adata_all)
    
    # Preprocess data
    adata_all = data_loader.preprocess_data(adata_all, is_raw_data=True)
    
    # Prepare dataset splits
    adata_train, adata_val, adata_test, num_classes, id_to_label = data_loader.prepare_dataset_splits(
        adata_all, config.classification_task
    )
    
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Classes: {list(id_to_label.values())}")
    
    # Save label mappings
    with open(save_dir / "label_mappings.pkl", "wb") as f:
        pickle.dump({"id_to_label": id_to_label}, f)
    
    # Tokenize data
    train_tokenized, _, _ = data_loader.tokenize_data(adata_train, "train")
    val_tokenized, _, _ = data_loader.tokenize_data(adata_val, "validation") 
    test_tokenized, _, _ = data_loader.tokenize_data(adata_test, "test")
    
    # Create data dictionaries
    train_data_dict = data_loader.create_data_dict(
        train_tokenized, adata_train, config.classification_task, mask_ratio=config.mask_ratio
    )
    val_data_dict = data_loader.create_data_dict(
        val_tokenized, adata_val, config.classification_task, mask_ratio=0.0
    )
    test_data_dict = data_loader.create_data_dict(
        test_tokenized, adata_test, config.classification_task, mask_ratio=0.0
    )
    
    # Create data loaders
    train_loader = create_dataloader(
        train_data_dict, config.batch_size, shuffle=True
    )
    val_loader = create_dataloader(
        val_data_dict, config.eval_batch_size, shuffle=False
    )
    test_loader = create_dataloader(
        test_data_dict, config.eval_batch_size, shuffle=False
    )
    
    logger.info(f"Created dataloaders: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")
    
    # Setup model
    model_manager = ModelManager(config, vocab, num_classes, device)
    
    # Load pretrained model if specified
    if config.load_model and Path(config.load_model).exists():
        logger.info(f"Loading pretrained model from: {config.load_model}")
        model = model_manager.setup_model(config.load_model)
    else:
        logger.info("Creating new model")
        model = model_manager.setup_model()
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Initialize trainer
    trainer = ClassificationTrainer(model, config, vocab, device, logger)
    
    # Training
    if config.do_train:
        logger.info("Starting training...")
        training_history = trainer.train(train_loader, val_loader, config.epochs)
        
        # Plot training history
        trainer.plot_training_history(save_dir / "training_history.png")
        
        # Save training history
        with open(save_dir / "training_history.pkl", "wb") as f:
            pickle.dump(training_history, f)
        
        # Save best model
        torch.save(trainer.get_best_model(), save_dir / "best_model.pt")
        logger.info("Training completed and model saved")
    
    # Testing
    logger.info("Starting testing...")
    test_results = trainer.test(test_loader, id_to_label)
    
    # Print results
    logger.info("=== Test Results ===")
    logger.info(f"Accuracy: {test_results['val_acc']:.4f}")
    logger.info(f"Precision: {test_results['val_precision']:.4f}")
    logger.info(f"Recall: {test_results['val_recall']:.4f}")
    logger.info(f"F1-Score: {test_results['val_f1']:.4f}")

    
    # Save test results
    results_to_save = {
        "test_metrics": {k: v for k, v in test_results.items() if k not in ["predictions", "labels"]},
        "predictions": test_results["predictions"],
        "labels": test_results["labels"],
        "id_to_label": id_to_label,
        "config": config.__dict__
    }
    
    with open(save_dir / "test_results.pkl", "wb") as f:
        pickle.dump(results_to_save, f)
    
    logger.info(f"Results saved to: {save_dir}")
    
    # Create predictions file for further analysis
    predictions_df = pd.DataFrame({
        'true_label': [id_to_label[label] for label in test_results["labels"]],
        'predicted_label': [id_to_label[pred] for pred in test_results["predictions"]],
        'true_id': test_results["labels"],
        'predicted_id': test_results["predictions"]
    })
    predictions_df.to_csv(save_dir / "predictions.csv", index=False)
    
    logger.info("Fine-tuning completed successfully!")
    
    return test_results

if __name__ == "__main__":
    results = main()