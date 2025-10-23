"""Simplified main script for scGPT fine-tuning (DDP Version)"""

import torch
import gc
import os
import sys
import logging
import pickle
import json
import argparse
import warnings
from pathlib import Path
import numpy as np
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime

sys.path.insert(0, "../")
from config import get_task_specific_config
from model_setup import ModelManager
from data_loader import ScGPTDataLoader
from trainer import ClassificationTrainer

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["cell_type", "disease", "tissue"])
    parser.add_argument("--data_dir", type=str, default="/home/arism/raw_data")
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="custom")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--cell_chunk_size", type=int, default=25000, help="Number of cells to process at a time")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--mask_ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(save_dir, rank=0):
    logger = logging.getLogger("scGPT")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(f"%(asctime)s - RANK:{rank} - %(levelname)s - %(message)s")
    
    if rank == 0:
        fh = logging.FileHandler(save_dir / "training.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def collect_files(data_dir, split_prefixes):
    data_dir = Path(data_dir)
    split_files = {}
    for prefix in split_prefixes:
        files = list(data_dir.glob(f"{prefix}*/*.h5ad"))
        if not files:
            subdirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)]
            for subdir in subdirs:
                files.extend(list(subdir.glob("*.h5ad")))
        
        if not files:
            raise FileNotFoundError(f"No .h5ad files found for prefix '{prefix}' in {data_dir}")
        
        split_files[prefix] = sorted(files)
        print(f"Found {len(files)} {prefix} files")
    return split_files


def main(local_rank=0, world_size=1, args=None):
    rank = dist.get_rank() if dist.is_initialized() else local_rank
    device = torch.device(f"cuda:{local_rank}")
    set_seed(args.seed + rank)
    
    config = get_task_specific_config(args.task)
    for key, value in vars(args).items():
        setattr(config, key, value)
    config.validate()
    
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) if args.save_dir else Path(f"./save/scGPT_{config.classification_task}_{config.dataset_name}_{timestamp}")
    
    if rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(save_dir, rank)
    logger.info(f"Starting fine-tuning on rank {rank} of {world_size}.")
    
    if rank == 0:
        with open(save_dir / "config.json", "w") as f:
            json.dump(vars(config), f, default=str, indent=2)

    logger.info("Loading data file paths...")
    split_files = collect_files(args.data_dir, ["train", "val", "test"])
    data_loader = ScGPTDataLoader(config)
    
    sample_adata, train_files, _, test_files = data_loader.load_data_from_files(
        split_files["train"], split_files["val"], split_files["test"]
    )
    
    if args.debug:
        train_files = train_files[:2]
        test_files = test_files[:1]
    
    vocab_path = Path(config.load_model) / "vocab.json" if config.load_model else None
    vocab = data_loader.setup_vocabulary(sample_adata, vocab_path)
    num_classes, id_to_label, label_to_id = data_loader.get_label_mapping(config.classification_task)
    
    model_manager = ModelManager(config, vocab, num_classes, device)
    model = model_manager.setup_model(config.load_model)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    trainer = ClassificationTrainer(model, config, vocab, device, logger)
    logger.info("Starting chunked training...")
    training_history = trainer.train_on_chunks(data_loader, config.classification_task, label_to_id, config.epochs, save_dir)
    
    if rank == 0:
        with open(save_dir / "training_history.pkl", "wb") as f:
            pickle.dump(training_history, f)
        
        best_model_state = trainer.get_best_model()
        torch.save(best_model_state, save_dir / "best_model.pt")

    logger.info("Testing...")
    all_predictions, all_labels_true = [], []
    for test_idx, test_file_path in enumerate(test_files):
        if rank == 0:
            logger.info(f"Testing on file {test_idx + 1}/{len(test_files)}: {test_file_path.name}")
        
        n_cells = data_loader.get_n_obs(test_file_path)
        if n_cells == 0:
            continue

        for i in range(0, n_cells, args.cell_chunk_size):
            cell_indices = range(i, min(i + args.cell_chunk_size, n_cells))
            
            test_loader = data_loader.process_and_create_loader(
                test_file_path, cell_indices, config.classification_task, label_to_id, 
                config.eval_batch_size, shuffle=False, rank=rank, world_size=world_size
            )
            
            chunk_results = trainer.evaluate(test_loader, return_predictions=True)
            
            if rank == 0 and chunk_results:
                all_predictions.extend(chunk_results["predictions"])
                all_labels_true.extend(chunk_results["labels"])
            
            del test_loader
            gc.collect()
            torch.cuda.empty_cache()

    if rank == 0:
        if not all_predictions:
            logger.warning("No predictions were gathered during testing. Skipping final metrics.")
            return
            
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(all_labels_true, all_predictions)
        f1 = f1_score(all_labels_true, all_predictions, average="macro", zero_division=0)
        weighted_f1 = f1_score(all_labels_true, all_predictions, average="weighted", zero_division=0)
        
        logger.info(f"Final Test Accuracy: {acc:.4f}")
        logger.info(f"Final Test F1 (Macro): {f1:.4f}")
        
        with open(save_dir / "results.json", "w") as f:
            json.dump({"accuracy": acc, "f1_macro": f1, "f1_weighted": weighted_f1}, f, indent=2)
        logger.info(f"Results saved to {save_dir}/results.json")

        logger.info("Saving raw predictions and labels to JSON...")
        try:
            # Ensure data is in standard Python lists for JSON serialization
            predictions_list = [int(p) for p in all_predictions]
            labels_list = [int(l) for l in all_labels_true]
            
            output_data = {
                "predictions": predictions_list,
                "labels": labels_list
            }
            
            with open(save_dir / "predictions_and_labels.json", "w") as f:
                json.dump(output_data, f, indent=2)
                
            logger.info(f"Predictions saved to {save_dir}/predictions_and_labels.json")
        except Exception as e:
            logger.error(f"Failed to save predictions as JSON: {e}")


def main_worker(local_rank, n_gpus, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(backend="nccl", init_method="env://", world_size=n_gpus, rank=local_rank)
    torch.cuda.set_device(local_rank)
    
    main(local_rank=local_rank, world_size=n_gpus, args=args)
    
    dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    n_gpus = torch.cuda.device_count()
    
    if n_gpus > 1:
        mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args), join=True)
    else:
        main(local_rank=0, world_size=1, args=args)

