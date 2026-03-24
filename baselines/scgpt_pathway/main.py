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
from model_setup import setup_pathway_model
from data_loader import create_pathway_dataloaders, load_pathway_mappings
from trainer import PathwayTrainer

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--preprocessed_data_dir", type=str, required=True, 
                        help="Path to preprocessed pathway data (e.g., ./preprocessed_data_pathway)")
    
    parser.add_argument("--load_model", type=str, required=True,
                        help="Path to pretrained scGPT model (e.g., /home/arism/scgpt_model/scGPT_human)")
    
    parser.add_argument("--dataset_name", type=str, default="pathway")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--k_pathways", type=int, default=2, help="Number of pathways to predict (top-k)")
    parser.add_argument("--freeze_scgpt", action="store_true", help="Freeze scGPT encoder")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default=None)
    
    return parser.parse_args()


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(save_dir, rank=0):
    logger = logging.getLogger("scGPT_pathway")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(f"%(asctime)s - RANK:{rank} - %(levelname)s - %(message)s")
    
    if rank == 0:
        fh = logging.FileHandler(save_dir / "training_pathway.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def main(local_rank=0, world_size=1, args=None):
    rank = dist.get_rank() if dist.is_initialized() else local_rank
    device = torch.device(f"cuda:{local_rank}")
    set_seed(args.seed + rank)
    
    # Use base config but adapt for pathway task
    config = get_task_specific_config("cell_type")  # Borrow settings
    config.classification_task = "pathway"
    
    # Override with args
    for key, value in vars(args).items():
        setattr(config, key, value)
    config.validate()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) if args.save_dir else Path(f"./save/scGPT_pathway_{timestamp}")
    
    if rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(save_dir, rank)
    logger.info(f"Starting pathway classification training on rank {rank} of {world_size}.")
    
    if rank == 0:
        with open(save_dir / "config.json", "w") as f:
            json.dump(vars(config), f, default=str, indent=2)

    # --- Load Preprocessed Data ---
    preprocessed_data_dir = Path(args.preprocessed_data_dir)
    logger.info("Loading preprocessed pathway data...")
    dataloaders, num_pathways, id_to_pathway, pathway_to_id = create_pathway_dataloaders(preprocessed_data_dir, config, rank, world_size, logger)

    train_loader = dataloaders.get("train")
    val_loader = dataloaders.get("val")
    test_loader = dataloaders.get("test")
    if train_loader is None or val_loader is None or test_loader is None:
        logger.error("Missing train, val, or test data. Please check your preprocessed data.")
        return
        
    # --- Load Vocab and Pathway Mappings ---
    vocab_path = Path(config.load_model) / "vocab.json"
    if not vocab_path or not vocab_path.exists():
        raise FileNotFoundError(f"vocab.json not found in {config.load_model}")
        
    from scgpt.tokenizer.gene_tokenizer import GeneVocab
    vocab = GeneVocab.from_file(vocab_path)
    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    for token in special_tokens:
        if token not in vocab:
            vocab.append_token(token)
    vocab.set_default_index(vocab["<pad>"])

    
    # --- Setup Model ---
    model = setup_pathway_model(
        config, 
        vocab, 
        num_pathways, 
        config.load_model, 
        device=str(device),
        freeze_scgpt=args.freeze_scgpt
    )
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    trainer = PathwayTrainer(model, config, vocab, device, logger, k_pathways=args.k_pathways)
    
    # --- Run Training ---
    logger.info("Starting training...")
    training_history = trainer.train(train_loader, val_loader, config.epochs, save_dir)
    
    if rank == 0:
        with open(save_dir / "training_history.pkl", "wb") as f:
            pickle.dump(training_history, f)
        
        try:
            trainer.plot_training_history(save_path=save_dir / "training_history.png")
        except Exception as e:
            logger.warning(f"Failed to plot training history: {e}")

    # --- Run Testing ---
    logger.info("Testing...")
    test_metrics = trainer.test(test_loader)
    
    # --- Save Final Results ---
    if rank == 0 and test_metrics:
        weighted_f1 = test_metrics.get("weighted_f1", 0.0)
        jaccard = test_metrics.get("jaccard_similarity", 0.0)
        subset_acc = test_metrics.get("subset_accuracy", 0.0)
            
        logger.info(f"Final Test Weighted F1: {weighted_f1:.4f}")
        logger.info(f"Final Test Jaccard Similarity: {jaccard:.4f}")
        logger.info(f"Final Test Subset Accuracy: {subset_acc:.4f}")
        
        with open(save_dir / "results.json", "w") as f:
            json.dump({
                "weighted_f1": weighted_f1,
                "jaccard_similarity": jaccard,
                "subset_accuracy": subset_acc
            }, f, indent=2)
        logger.info(f"Results saved to {save_dir}/results.json")

        logger.info("Saving raw predictions and labels...")
        try:
            predictions_list = test_metrics["predictions"].tolist()
            labels_list = test_metrics["labels"].tolist()
            
            output_data = {
                "predictions": predictions_list,
                "labels": labels_list,
                "id_to_pathway": id_to_pathway
            }
            
            with open(save_dir / "predictions_and_labels.json", "w") as f:
                json.dump(output_data, f, indent=2)
                
            logger.info(f"Predictions saved to {save_dir}/predictions_and_labels.json")
        except Exception as e:
            logger.error(f"Failed to save predictions: {e}")
            
    logger.info("--- JOB COMPLETE ---")


def main_worker(local_rank, n_gpus, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    
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