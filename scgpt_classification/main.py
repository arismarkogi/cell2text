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


# Add this class definition after imports but before parse_arguments()
class SeqDataset(torch.utils.data.Dataset):
    """Dataset class for scGPT sequences"""
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}





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
    if args is None:
        args = parse_arguments()

    # Set rank/world_size
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = local_rank
        world_size = 1

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device) if torch.cuda.is_available() else None
    
    set_seed(args.seed + rank)

    # Get configuration
    config = get_task_specific_config(args.task)
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

    # Create save_dir
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = Path(f"./save/scGPT_finetune_{config.classification_task}_{config.dataset_name}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = setup_logger(save_dir)
    if rank > 0:
        # Remove file handler for non-rank-0 to avoid duplicate logs
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)

    logger.info(f"[Rank {rank}] Starting fine-tuning")

    # Save configuration (only rank 0)
    if rank == 0:
        try:
            config.to_json(save_dir / "config.json")
        except Exception:
            with open(save_dir / "config.json", "w") as f:
                json.dump(vars(config), f, default=str, indent=2)

    # Synchronize before data processing
    if world_size > 1:
        dist.barrier()

    # DATA PROCESSING: Only rank 0 does preprocessing
    processed_data_dir = save_dir / "processed_data"
    
    if rank == 0:
        logger.info("Rank 0: Starting data preprocessing...")
        processed_data_dir.mkdir(exist_ok=True)
        
        # Check if processed data already exists
        processed_flag = processed_data_dir / "processing_complete.flag"
        if processed_flag.exists():
            logger.info("Found existing processed data, skipping preprocessing")
        else:
            # Your existing data processing logic here
            split_prefixes = ["train", "val", "test"]
            split_files = collect_batch_files(args.data_dir, split_prefixes)
            train_files = split_files["train"]
            val_files = split_files["val"]
            test_files = split_files["test"]

            data_loader = ScGPTDataLoader(config)
            
            # Load and process data
            sample_adata, train_batches, val_batches, test_batches = data_loader.load_data_from_files(
                train_files, val_files, test_files
            )

            if args.debug:
                train_batches = train_batches[:2]
                val_batches = val_batches[:1]  
                test_batches = test_batches[:1]

            # Setup vocabulary
            vocab_path = None
            if getattr(config, "load_model", None):
                potential_vocab_path = Path(config.load_model) / "vocab.json"
                if potential_vocab_path.exists():
                    vocab_path = str(potential_vocab_path)

            vocab = data_loader.setup_vocabulary(sample_adata, vocab_path)
            sample_adata = data_loader.filter_genes_by_vocab(sample_adata)

            # Prepare dataset splits
            num_classes, id_to_label = data_loader.prepare_dataset_splits_from_batches(
                config.classification_task
            )

            # Create data loaders
            train_loader, val_loader, test_loader = data_loader.create_batch_data_loaders(
                config.classification_task, config
            )

            # Save processed data efficiently
            def save_processed_dataloader(loader, name):
                dataset = loader.dataset
                data_dict = dataset.data
                
                # Save as memory-mapped numpy arrays
                for key, tensor in data_dict.items():
                    if tensor.is_cuda:
                        tensor = tensor.cpu()
                    
                    # Save as numpy array
                    array = tensor.numpy()
                    np.save(processed_data_dir / f"{name}_{key}.npy", array)
                    
                    # Also save metadata
                    with open(processed_data_dir / f"{name}_{key}_meta.json", "w") as f:
                        json.dump({
                            "shape": array.shape,
                            "dtype": str(array.dtype),
                            "device": "cpu"
                        }, f)

            # Save each dataloader
            save_processed_dataloader(train_loader, "train")
            save_processed_dataloader(val_loader, "val")
            save_processed_dataloader(test_loader, "test")

            # Save vocabulary and metadata
            with open(processed_data_dir / "vocab.pkl", "wb") as f:
                pickle.dump(vocab, f)
            
            with open(processed_data_dir / "metadata.pkl", "wb") as f:
                pickle.dump({
                    "num_classes": num_classes,
                    "id_to_label": id_to_label,
                    "all_genes_list": getattr(data_loader, "all_genes_list", []),
                }, f)

            # Create completion flag
            processed_flag.touch()
            logger.info("Rank 0: Data preprocessing completed and saved")

    # Synchronize - wait for rank 0 to finish preprocessing
    if world_size > 1:
        dist.barrier()

    # ALL RANKS: Load preprocessed data
    logger.info(f"Rank {rank}: Loading preprocessed data...")
    
    def load_processed_dataloader(name, batch_size, shuffle, rank, world_size):
        # Load all arrays for this dataset
        data_dict = {}
        pattern = processed_data_dir / f"{name}_*.npy"
        
        for file_path in processed_data_dir.glob(f"{name}_*.npy"):
            if file_path.name.endswith("_meta.json"):
                continue
                
            key = file_path.stem.replace(f"{name}_", "")
            # Use memory mapping for large arrays
            array = np.load(file_path, mmap_mode='r')
            tensor = torch.from_numpy(array.copy())  # Copy to avoid mmap issues during training
            
            data_dict[key] = tensor

        # Create dataset and dataloader
        dataset = SeqDataset(data_dict)
        
        if world_size > 1:
            sampler = DistributedSampler(
                dataset, 
                num_replicas=world_size, 
                rank=rank, 
                shuffle=shuffle,
                drop_last=False
            )
            return DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=min(4, batch_size // 2),
                pin_memory=True
            )
        else:
            return DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=min(4, batch_size // 2),
                pin_memory=True
            )

    # Load dataloaders
    train_loader = load_processed_dataloader("train", config.batch_size, True, rank, world_size)
    val_loader = load_processed_dataloader("val", config.eval_batch_size, False, rank, world_size)
    test_loader = load_processed_dataloader("test", config.eval_batch_size, False, rank, world_size)

    # Load metadata
    with open(processed_data_dir / "vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    
    with open(processed_data_dir / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
        num_classes = metadata["num_classes"]
        id_to_label = metadata["id_to_label"]

    logger.info(f"Rank {rank}: Loaded preprocessed data")

    # Continue with model setup and training...
    model_manager = ModelManager(config, vocab, num_classes, device)

    if getattr(config, "load_model", None) and Path(config.load_model).exists():
        logger.info(f"Loading pretrained model from: {config.load_model}")
        model = model_manager.setup_model(config.load_model)
    else:
        logger.info("Creating new model")
        model = model_manager.setup_model()
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, 
                   find_unused_parameters=True)

    # Rest of your training code remains the same...
    trainer = ClassificationTrainer(model, config, vocab, device, logger)
    if dist.is_available() and dist.is_initialized():
        trainer.rank = dist.get_rank()
        trainer.world_size = dist.get_world_size()
    else:
        trainer.rank = 0
        trainer.world_size = 1

    # Training and testing...
    if getattr(config, "do_train", False):
        logger.info("Starting training...")
        training_history = trainer.train(train_loader, val_loader, config.epochs, save_dir)
        
        # Only rank 0 saves results
        if rank == 0:
            try:
                trainer.plot_training_history(save_dir / "training_history.png")
            except Exception as e:
                logger.warning(f"Could not plot training history: {e}")

            with open(save_dir / "training_history.pkl", "wb") as f:
                pickle.dump(training_history, f)

            try:
                best_model = trainer.get_best_model()
                torch.save(best_model, save_dir / "best_model.pt")
            except Exception as e:
                logger.warning(f"Could not save best model: {e}")

    # Testing - only rank 0 should do final evaluation and saving
    logger.info("Starting testing...")
    test_results = trainer.test(test_loader, id_to_label)

        # Testing - only rank 0 should do final evaluation and saving
    logger.info("Starting testing...")
    test_results = trainer.test(test_loader, id_to_label)

    results_json = {}  

    if rank == 0 and test_results is not None:
        # Your existing results processing and saving code...
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
                    "train_files": len(train_files) if 'train_files' in locals() else 0,
                    "val_files": len(val_files) if 'val_files' in locals() else 0,
                    "test_files": len(test_files) if 'test_files' in locals() else 0,
                    "train_samples": len(getattr(train_loader, "dataset", [])),
                    "val_samples": len(getattr(val_loader, "dataset", [])),
                    "test_samples": len(getattr(test_loader, "dataset", [])),
                    "num_genes": len(getattr(data_loader, "all_genes_list", [])) if 'data_loader' in locals() else 0,
                    "vocab_size": len(vocab),
                },
                "distributed_info": {
                    "ddp_enabled": world_size > 1,
                    "world_size": world_size,
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
    
    return results_json  # ✅ Now properly defined

import torch.multiprocessing as mp
import torch

def main_worker(local_rank, n_gpus, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    if n_gpus > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=n_gpus,
            rank=local_rank,
        )
    torch.cuda.set_device(local_rank)
    
    # Initialize distributed training
    main(local_rank=local_rank, world_size=n_gpus, args=args)
    
    # Wait for all processes to finish
    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    args = parse_arguments()  # Parse once here
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args), join=True)
    else:
        main(local_rank=0, world_size=1, args=args)