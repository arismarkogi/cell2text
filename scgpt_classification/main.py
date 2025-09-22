"""
Main script for scGPT fine-tuning on classification tasks
"""

import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

import os
import sys
import logging
import pickle
import random
import warnings
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

# Add paths (adjust according to your directory structure)
sys.path.insert(0, "../")

# Import our custom modules
from config import get_task_specific_config
from model_setup import ModelManager
from data_loader import ScGPTDataLoader, create_dataloader
from trainer import ClassificationTrainer


from torch.utils.data import DistributedSampler, DataLoader






# Suppress warnings
warnings.filterwarnings("ignore")


import torch.distributed as dist
import os



def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="scGPT Fine-tuning")
    
    # Task selection
    parser.add_argument(
        "--task", 
        type=str, 
        required=True,
        choices=["cell_type", "disease", "tissue"],
        help="Classification task to perform"
    )
    
    # Data paths
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="/home/arism/raw_data",
        help="Directory containing train/val/test subdirectories"
    )
    
    # Model configuration
    parser.add_argument(
        "--load_model", 
        type=str, 
        default=None,
        help="Path to pretrained model directory"
    )
    
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default="custom",
        help="Name of the dataset for logging"
    )
    
    # Training configuration
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--eval_batch_size", 
        type=int, 
        default=8,
        help="Evaluation batch size"
    )
    
    parser.add_argument(
        "--lr", 
        type=float, 
        default=5e-5,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--mask_ratio", 
        type=float, 
        default=0.0,
        help="Masking ratio for training"
    )
    
    # Other options
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default=None,
        help="Directory to save results (default: auto-generated)"
    )
    
    parser.add_argument(
        "--do_train", 
        action="store_true",
        default=True,
        help="Whether to run training"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode (smaller data)"
    )
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(save_dir: Path) -> logging.Logger:
    """Setup logger"""
    logger = logging.getLogger("scGPT_finetune")
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # File handler
    fh = logging.FileHandler(str(save_dir / "training.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def collect_batch_files(data_dir: str, split_prefixes: List[str]) -> Dict[str, List[Path]]:
    """Collect paths to all batch files for each split without loading them"""
    data_dir = Path(data_dir)
    split_files: Dict[str, List[Path]] = {}

    for prefix in split_prefixes:
        print(f"Collecting {prefix} file paths...")

        # Find all subdirectories starting with the prefix
        subdirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)]
        subdirs.sort()  # Sort for consistent ordering

        print(f"Found {len(subdirs)} {prefix} subdirectories: {[d.name for d in subdirs]}")

        # Collect all batch files for this split
        batch_files: List[Path] = []
        for subdir in subdirs:
            h5ad_files = list(subdir.glob("*.h5ad"))
            if h5ad_files:
                batch_files.extend(h5ad_files)
                print(f"Found {len(h5ad_files)} files in {subdir.name}")
            else:
                print(f"Warning: No .h5ad files found in {subdir}")

        if not batch_files:
            raise FileNotFoundError(f"No data files found for {prefix} split")

        split_files[prefix] = batch_files
        print(f"Collected {len(batch_files)} {prefix} files")

    return split_files


def _get_metric(results: dict, *candidates: str, default: float = 0.0) -> float:
    """Helper to fetch metric value from different possible keys"""
    for k in candidates:
        if k in results:
            return float(results[k])
    # try to find similar keys case-insensitively
    lower_keys = {kk.lower(): kk for kk in results.keys()}
    for cand in candidates:
        lk = cand.lower()
        if lk in lower_keys:
            return float(results[lower_keys[lk]])
    return default


def main(local_rank=0, world_size=1, args=None):
    use_ddp = world_size > 1
    rank = local_rank

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    ...
    print(f"[rank {rank}/{world_size}] using device: {device}")
    set_seed(args.seed + rank)
    
    # Parse command line arguments
    args = parse_arguments()

    # Get configuration and override with CLI args
    config = get_task_specific_config(args.task)
    
    # Override config with command line arguments
    config.classification_task = args.task
    config.dataset_name = args.dataset_name
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.eval_batch_size = args.eval_batch_size
    config.lr = args.lr
    config.mask_ratio = args.mask_ratio
    config.seed = args.seed
    config.do_train = args.do_train
    if args.load_model:
        config.load_model = args.load_model
    
    config.validate()

    # Create save directory
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = Path(f"./save/scGPT_finetune_{config.classification_task}_{config.dataset_name}")
    
    save_dir.mkdir(parents=True, exist_ok=True)
    # Setup logger
    logger = setup_logger(save_dir)
    
    logger.info(f"Starting fine-tuning for task: {config.classification_task}")
    logger.info(f"Save directory: {save_dir}")

    # Save configuration
    try:
        config.to_json(save_dir / "config.json")
    except Exception:
        # Fallback: dump __dict__ if no to_json method
        with open(save_dir / "config.json", "w") as f:
            json.dump(vars(config), f, default=str, indent=2)
    

    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # after args = parse_arguments()
    use_ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1  # torchrun sets WORLD_SIZE env var

    print(f"DDP enabled: {use_ddp}")

    def rebuild_dataloader_for_ddp(old_loader, batch_size, shuffle, rank, world_size, eval=False):
        dataset = old_loader.dataset
        
        if world_size > 1:  # DDP is enabled
            sampler = DistributedSampler(
                dataset, 
                num_replicas=world_size, 
                rank=rank, 
                shuffle=shuffle,
                drop_last=False  # Important: don't drop last to avoid missing data
            )
            
            # Determine num_workers
            try:
                num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)
            except AttributeError:
                num_workers = min(4, batch_size // 2)
                
            return DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                sampler=sampler,  # Use sampler instead of shuffle
                drop_last=False,
                num_workers=num_workers,
                pin_memory=True
            )
        else:
            # For non-DDP, return original or create new with proper shuffle
            return DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=False,
                num_workers=num_workers if 'num_workers' in locals() else 0,
                pin_memory=True
            )

    if use_ddp:
        # Initialize distributed process group (env:// works with torchrun)
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # Set device for this process
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[rank {rank}/{world_size}] using device: {device}")
    set_seed(args.seed + rank)

    # create logger only on rank 0 (so only rank 0 writes to file)
    if rank == 0:
        logger = setup_logger(save_dir)
    else:
        logger = setup_logger(save_dir)  # optional: or a no-op logger; but ensure console spam is minimal
        # If you want zero file logging on workers, comment file handler or set level to WARNING
        for h in list(logger.handlers):
            # keep only console handler on non-zero ranks
            if isinstance(h, logging.FileHandler):
                logger.removeHandler(h)


    # Collect batch file paths from your directory structure
    split_prefixes = ["train", "val", "test"]

    try:
        split_files = collect_batch_files(args.data_dir, split_prefixes)
        train_files = split_files["train"]
        val_files = split_files["val"]
        test_files = split_files["test"]
    except Exception as e:
        logger.error(f"Error collecting data files: {e}")
        raise

    logger.info(f"Collected files: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    # Initialize data loader with file paths
    data_loader = ScGPTDataLoader(config)
    logger.info("Setting up data processing...")

    # Load data in batches and process iteratively
    sample_adata, train_batches, val_batches, test_batches = data_loader.load_data_from_files(
        train_files, val_files, test_files
    )

    # Debug mode: limit data size
    if args.debug:
        train_batches = train_batches[:2]
        val_batches = val_batches[:1]  
        test_batches = test_batches[:1]
        logger.info("Debug mode: Using limited data")

    # Setup vocabulary using sample data
    vocab_path = None
    if getattr(config, "load_model", None):
        potential_vocab_path = Path(config.load_model) / "vocab.json"
        if potential_vocab_path.exists():
            vocab_path = str(potential_vocab_path)

    vocab = data_loader.setup_vocabulary(sample_adata, vocab_path)
    logger.info(f"Vocabulary size: {len(vocab)}")

    # Filter genes by vocabulary - this updates the gene list in data_loader
    sample_adata = data_loader.filter_genes_by_vocab(sample_adata)

    # Prepare dataset splits and labels from batches
    num_classes, id_to_label = data_loader.prepare_dataset_splits_from_batches(config.classification_task)

    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Classes: {list(id_to_label.values())}")

    # Create data loaders from processed batches
    train_loader, val_loader, test_loader = data_loader.create_batch_data_loaders(
        config.classification_task, config
    )

    train_loader = rebuild_dataloader_for_ddp(train_loader, config.batch_size, shuffle=True, rank=rank, world_size=world_size)
    val_loader   = rebuild_dataloader_for_ddp(val_loader, config.eval_batch_size, shuffle=False, rank=rank, world_size=world_size, eval=True)
    test_loader  = rebuild_dataloader_for_ddp(test_loader, config.eval_batch_size, shuffle=False, rank=rank, world_size=world_size, eval=True)

    logger.info(f"Created dataloaders: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")

    # Save label mappings
    with open(save_dir / "label_mappings.pkl", "wb") as f:
        pickle.dump({"id_to_label": id_to_label}, f)

    # Setup model manager
    model_manager = ModelManager(config, vocab, num_classes, device)

    # Load pretrained model if specified
    if getattr(config, "load_model", None) and Path(config.load_model).exists():
        logger.info(f"Loading pretrained model from: {config.load_model}")
        model = model_manager.setup_model(config.load_model)
    else:
        logger.info("Creating new model")
        model = model_manager.setup_model()
    
    if use_ddp:
        # wrap with DDP; set device_ids to local GPU index (LOCAL_RANK)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)


    # Log number of trainable parameters
    try:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {n_params:,}")
    except Exception as e:
        logger.warning(f"Could not compute model parameter count: {e}")

    trainer = ClassificationTrainer(model, config, vocab, device, logger)
    # set trainer.rank/world_size inside trainer (or assign attributes)
    if dist.is_available() and dist.is_initialized():
        trainer.rank = dist.get_rank()
        trainer.world_size = dist.get_world_size()
    else:
        trainer.rank = 0
        trainer.world_size = 1

    if getattr(config, "do_train", False):
        results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.epochs,
            save_dir=save_dir,
        )



    # Training
    if getattr(config, "do_train", False):
        logger.info("Starting training...")
        training_history = trainer.train(train_loader, val_loader, config.epochs, save_dir)

        # Plot training history
        try:
            trainer.plot_training_history(save_dir / "training_history.png")
        except Exception as e:
            logger.warning(f"Could not plot training history: {e}")

        # Save training history
        with open(save_dir / "training_history.pkl", "wb") as f:
            pickle.dump(training_history, f)

        # Save best model
        try:
            best_model = trainer.get_best_model()
            torch.save(best_model, save_dir / "best_model.pt")
        except Exception as e:
            logger.warning(f"Could not save best model: {e}")

        logger.info("Training completed and model saved")

    # Testing
    logger.info("Starting testing...")
    test_results = trainer.test(test_loader, id_to_label)

    # Process and save results
    if test_results:
        # Print results
        acc = _get_metric(test_results, "acc", "accuracy", "acc")
        prec = _get_metric(test_results, "precision", "precision", "prec")
        rec = _get_metric(test_results, "recall", "recall", "rec")
        f1 = _get_metric(test_results, "f1", "f1", "f1_score")
        weighted_f1 = _get_metric(test_results, "weighted_f1", "weighted_f1")

        logger.info("=== Test Results ===")
        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"Precision: {prec:.4f}")
        logger.info(f"Recall: {rec:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        logger.info(f"Weighted F1-Score: {weighted_f1:.4f}")

        # Save results
        preds = test_results.get("predictions", [])
        labels = test_results.get("labels", [])

        results_json = {
            "experiment_info": {
                "task": config.classification_task,
                "dataset_name": config.dataset_name,
                "model_config": {
                    "max_seq_len": getattr(config, "max_seq_len", None),
                    "batch_size": getattr(config, "batch_size", None),
                    "learning_rate": getattr(config, "lr", getattr(config, "learning_rate", None)),
                    "epochs": getattr(config, "epochs", None),
                    "mask_ratio": getattr(config, "mask_ratio", None),
                },
                "data_info": {
                    "num_classes": int(num_classes),
                    "train_files": len(train_files),
                    "val_files": len(val_files),
                    "test_files": len(test_files),
                    "train_samples": len(getattr(train_loader, "dataset", [])),
                    "val_samples": len(getattr(val_loader, "dataset", [])),
                    "test_samples": len(getattr(test_loader, "dataset", [])),
                    "num_genes": len(getattr(data_loader, "all_genes_list", [])),
                    "vocab_size": len(vocab),
                },
                "distributed_info": {
                    "ddp_enabled": False,
                    "world_size": 1,
                },
            },
            "test_metrics": {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "weighted_f1": weighted_f1,
            },
            "class_labels": {str(k): v for k, v in id_to_label.items()},
            "predictions_summary": {
                "total_predictions": len(preds),
                "correct_predictions": int(np.sum(np.array(preds) == np.array(labels))) if len(preds) == len(labels) else None,
            },
        }

        # Add training history if available
        if getattr(config, "do_train", False) and "training_history" in locals():
            history_json = {}
            for key, values in training_history.items():
                if isinstance(values, (list, np.ndarray)):
                    history_json[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in values]
                else:
                    history_json[key] = float(values) if isinstance(values, (np.floating, np.integer)) else values
            results_json["training_history"] = history_json

        # Save results as JSON
        with open(save_dir / "results.json", "w") as f:
            json.dump(results_json, f, indent=2)

        logger.info(f"Results saved to: {save_dir}/results.json")

        # Also save detailed results
        detailed_results = {
            "test_metrics": {k: v for k, v in test_results.items() if k not in ["predictions", "labels"]},
            "predictions": preds,
            "labels": labels,
            "id_to_label": id_to_label,
            "config": getattr(config, "__dict__", vars(config)),
        }

        with open(save_dir / "detailed_results.pkl", "wb") as f:
            pickle.dump(detailed_results, f)

        logger.info("Fine-tuning completed successfully!")
        
        return results_json
    
    return {}


import torch.multiprocessing as mp
import torch

def main_worker(local_rank, n_gpus, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    if n_gpus > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",  # Use env:// instead of tcp://
            world_size=n_gpus,
            rank=local_rank,
        )
    torch.cuda.set_device(local_rank)
    
    main(local_rank=local_rank, world_size=n_gpus, args=args)

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    args = parse_arguments()  # or however you parse config

    if n_gpus > 1:
        mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args))
    else:
        main(local_rank=0, world_size=1, args=args)
