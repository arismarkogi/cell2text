"""Simplified main script for scGPT fine-tuning"""

import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
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
from torch.utils.data import Dataset, DataLoader, DistributedSampler

sys.path.insert(0, "../")
from config import get_task_specific_config
from model_setup import ModelManager
from data_loader import ScGPTDataLoader, SeqDataset
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
    
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    if rank == 0:
        fh = logging.FileHandler(save_dir / "training.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger


def collect_files(data_dir, split_prefixes):
    """Collect all .h5ad files for each split"""
    data_dir = Path(data_dir)
    split_files = {}
    
    for prefix in split_prefixes:
        subdirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)])
        files = []
        for subdir in subdirs:
            files.extend(list(subdir.glob("*.h5ad")))
        
        if not files:
            raise FileNotFoundError(f"No files found for {prefix}")
        
        split_files[prefix] = files
        print(f"Found {len(files)} {prefix} files")
    
    return split_files


def save_dataloader(loader, processed_dir, name):
    """Save dataloader data as numpy arrays"""
    data_dict = loader.dataset.data
    
    for key, tensor in data_dict.items():
        array = tensor.cpu().numpy()
        np.save(processed_dir / f"{name}_{key}.npy", array)


def load_dataloader(processed_dir, name, batch_size, shuffle, rank, world_size):
    """Load dataloader from saved numpy arrays"""
    data_dict = {}
    
    for file_path in processed_dir.glob(f"{name}_*.npy"):
        key = file_path.stem.replace(f"{name}_", "")
        array = np.load(file_path, mmap_mode='r')
        data_dict[key] = torch.from_numpy(array.copy())
    
    dataset = SeqDataset(data_dict)
    
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)


def main(local_rank=0, world_size=1, args=None):
    # Setup distributed training
    rank = dist.get_rank() if dist.is_initialized() else local_rank
    world_size = dist.get_world_size() if dist.is_initialized() else world_size
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    
    set_seed(args.seed + rank)
    
    # Setup config
    config = get_task_specific_config(args.task)
    config.classification_task = args.task
    config.dataset_name = args.dataset_name
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.eval_batch_size = args.eval_batch_size
    config.lr = args.lr
    config.mask_ratio = args.mask_ratio
    config.seed = args.seed
    if args.load_model:
        config.load_model = args.load_model
    config.validate()
    
    # # Setup directories
    # save_dir = Path(args.save_dir) if args.save_dir else Path(f"./save/scGPT_{config.classification_task}_{config.dataset_name}")
    # save_dir.mkdir(parents=True, exist_ok=True)
    
    # logger = setup_logger(save_dir, rank)
    # logger.info(f"[Rank {rank}] Starting fine-tuning")
    
    # if rank == 0:
    #     with open(save_dir / "config.json", "w") as f:
    #         json.dump(vars(config), f, default=str, indent=2)
    
    # if world_size > 1:
    #     dist.barrier()
    
    # # Data processing (rank 0 only)
    # processed_dir = save_dir / "processed_data"
    
    # if rank == 0:
    #     processed_flag = processed_dir / "complete.flag"
        
    #     if not processed_flag.exists():
    #         logger.info("Processing data...")
    #         processed_dir.mkdir(exist_ok=True)
            
    #         # Load data
    #         split_files = collect_files(args.data_dir, ["train", "val", "test"])
    #         data_loader = ScGPTDataLoader(config)
            
    #         sample_adata, train_batches, val_batches, test_batches = data_loader.load_data_from_files(
    #             split_files["train"], split_files["val"], split_files["test"]
    #         )
            
    #         if args.debug:
    #             train_batches = train_batches[:2]
    #             val_batches = val_batches[:1]
    #             test_batches = test_batches[:1]
            
    #         # Setup vocab
    #         vocab_path = None
    #         if config.load_model:
    #             potential_path = Path(config.load_model) / "vocab.json"
    #             if potential_path.exists():
    #                 vocab_path = str(potential_path)
            
    #         vocab = data_loader.setup_vocabulary(sample_adata, vocab_path)
    #         sample_adata = data_loader.filter_genes_by_vocab(sample_adata)
            
    #         # Prepare datasets
    #         num_classes, id_to_label = data_loader.prepare_dataset_splits_from_batches(config.classification_task)
    #         train_loader, val_loader, test_loader = data_loader.create_batch_data_loaders(config.classification_task, config)
            
    #         # Save processed data
    #         save_dataloader(train_loader, processed_dir, "train")
    #         save_dataloader(val_loader, processed_dir, "val")
    #         save_dataloader(test_loader, processed_dir, "test")
            
    #         with open(processed_dir / "vocab.pkl", "wb") as f:
    #             pickle.dump(vocab, f)
            
    #         with open(processed_dir / "metadata.pkl", "wb") as f:
    #             pickle.dump({
    #                 "num_classes": num_classes,
    #                 "id_to_label": id_to_label,
    #                 "all_genes_list": getattr(data_loader, "all_genes_list", []),
    #             }, f)
            
    #         processed_flag.touch()
    #         logger.info("Data processing complete")
    
    # if world_size > 1:
    #     dist.barrier()
    
    # # Load processed data (all ranks)
    # logger.info(f"[Rank {rank}] Loading processed data...")
    
    # train_loader = load_dataloader(processed_dir, "train", config.batch_size, True, rank, world_size)
    # val_loader = load_dataloader(processed_dir, "val", config.eval_batch_size, False, rank, world_size)
    # test_loader = load_dataloader(processed_dir, "test", config.eval_batch_size, False, rank, world_size)
    
    # with open(processed_dir / "vocab.pkl", "rb") as f:
    #     vocab = pickle.load(f)
    
    # with open(processed_dir / "metadata.pkl", "rb") as f:
    #     metadata = pickle.load(f)
    #     num_classes = metadata["num_classes"]
    #     id_to_label = metadata["id_to_label"]
    
    # # Setup model
    # model_manager = ModelManager(config, vocab, num_classes, device)
    
    # if config.load_model and Path(config.load_model).exists():
    #     logger.info(f"Loading pretrained model from {config.load_model}")
    #     model = model_manager.setup_model(config.load_model)
    # else:
    #     logger.info("Creating new model")
    #     model = model_manager.setup_model()
    
    # if world_size > 1:
    #     model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    # # Training
    # trainer = ClassificationTrainer(model, config, vocab, device, logger)
    # trainer.rank = rank
    # trainer.world_size = world_size
    
    # logger.info("Starting training...")
    # training_history = trainer.train(train_loader, val_loader, config.epochs, save_dir)
    
    # if rank == 0:
    #     with open(save_dir / "training_history.pkl", "wb") as f:
    #         pickle.dump(training_history, f)
        
    #     try:
    #         best_model = trainer.get_best_model()
    #         torch.save(best_model, save_dir / "best_model.pt")
    #     except Exception as e:
    #         logger.warning(f"Could not save best model: {e}")
    
    # # Testing
    # logger.info("Testing...")
    # test_results = trainer.test(test_loader, id_to_label)
    
    # if rank == 0 and test_results:
    #     acc = test_results.get("acc", 0.0)
    #     f1 = test_results.get("f1", 0.0)
    #     weighted_f1 = test_results.get("weighted_f1", 0.0)
        
    #     logger.info(f"Test Accuracy: {acc:.4f}")
    #     logger.info(f"Test F1: {f1:.4f}")
    #     logger.info(f"Test Weighted F1: {weighted_f1:.4f}")
        
    #     results_json = {
    #         "task": config.classification_task,
    #         "dataset": config.dataset_name,
    #         "test_metrics": {
    #             "accuracy": float(acc),
    #             "f1_score": float(f1),
    #             "weighted_f1": float(weighted_f1),
    #         },
    #         "predictions_count": len(test_results.get("predictions", [])),
    #     }
        
    #     with open(save_dir / "results.json", "w") as f:
    #         json.dump(results_json, f, indent=2)
        
    #     logger.info(f"Results saved to {save_dir}/results.json")
    
    # return test_results
    # Setup directories
    save_dir = Path(args.save_dir) if args.save_dir else Path(f"./save/scGPT_{config.classification_task}_{config.dataset_name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(save_dir, rank)
    logger.info(f"[Rank {rank}] Starting fine-tuning")
    
    if rank == 0:
        with open(save_dir / "config.json", "w") as f:
            json.dump(vars(config), f, default=str, indent=2)
    
    if world_size > 1:
        dist.barrier()
    
    # SIMPLIFIED DATA PROCESSING
    logger.info("Loading data files...")
    split_files = collect_files(args.data_dir, ["train", "val", "test"])
    data_loader = ScGPTDataLoader(config)
    
    sample_adata, train_batches, val_batches, test_batches = data_loader.load_data_from_files(
        split_files["train"], split_files["val"], split_files["test"]
    )
    
    if args.debug:
        train_batches = train_batches[:2]
        val_batches = val_batches[:1]
        test_batches = test_batches[:1]
    
    # Setup vocab
    vocab_path = None
    if config.load_model:
        potential_path = Path(config.load_model) / "vocab.json"
        if potential_path.exists():
            vocab_path = str(potential_path)
    
    vocab = data_loader.setup_vocabulary(sample_adata, vocab_path)
    sample_adata = data_loader.filter_genes_by_vocab(sample_adata)
    
    # Get label mappings (no preprocessing yet)
    num_classes, id_to_label, label_to_id = data_loader.get_label_mapping(config.classification_task)
    
    # Setup model
    model_manager = ModelManager(config, vocab, num_classes, device)
    
    if config.load_model and Path(config.load_model).exists():
        logger.info(f"Loading pretrained model from {config.load_model}")
        model = model_manager.setup_model(config.load_model)
    else:
        logger.info("Creating new model")
        model = model_manager.setup_model()
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    # CHUNKED TRAINING
    trainer = ClassificationTrainer(model, config, vocab, device, logger)
    trainer.rank = rank
    trainer.world_size = world_size
    
    logger.info("Starting chunked training...")
    training_history = trainer.train_on_chunks(data_loader, config.classification_task, label_to_id, config.epochs, save_dir)
    
    if rank == 0:
        with open(save_dir / "training_history.pkl", "wb") as f:
            pickle.dump(training_history, f)
        
        try:
            best_model = trainer.get_best_model()
            torch.save(best_model, save_dir / "best_model.pt")
        except Exception as e:
            logger.warning(f"Could not save best model: {e}")
    
    # Testing - process test batches one by one
    logger.info("Testing...")
    
    all_predictions = []
    all_labels_true = []
    
    for test_idx, test_batch in enumerate(test_batches):
        logger.info(f"Testing on chunk {test_idx + 1}/{len(test_batches)}")
        test_loader = data_loader.process_and_create_loader(
            test_batch, 
            config.classification_task, 
            label_to_id, 
            config.eval_batch_size, 
            shuffle=False
        )
        
        chunk_results = trainer.evaluate(test_loader, return_predictions=True)
        
        if chunk_results and rank == 0:
            all_predictions.extend(chunk_results["predictions"])
            all_labels_true.extend(chunk_results["labels"])
        
        del test_loader
        gc.collect()
        torch.cuda.empty_cache()
    
    # Calculate overall test metrics
    if rank == 0 and all_predictions:
        from sklearn.metrics import accuracy_score, f1_score
        
        acc = accuracy_score(all_labels_true, all_predictions)
        f1 = f1_score(all_labels_true, all_predictions, average="macro", zero_division=0)
        weighted_f1 = f1_score(all_labels_true, all_predictions, average="weighted", zero_division=0)
        
        logger.info(f"Test Accuracy: {acc:.4f}")
        logger.info(f"Test F1: {f1:.4f}")
        logger.info(f"Test Weighted F1: {weighted_f1:.4f}")
        
        results_json = {
            "task": config.classification_task,
            "dataset": config.dataset_name,
            "test_metrics": {
                "accuracy": float(acc),
                "f1_score": float(f1),
                "weighted_f1": float(weighted_f1),
            },
            "predictions_count": len(all_predictions),
        }
        
        with open(save_dir / "results.json", "w") as f:
            json.dump(results_json, f, indent=2)
        
        logger.info(f"Results saved to {save_dir}/results.json")
        
        return results_json
    
    return None



def main_worker(local_rank, n_gpus, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    if n_gpus > 1:
        dist.init_process_group(backend="nccl", init_method="env://", world_size=n_gpus, rank=local_rank)
    
    torch.cuda.set_device(local_rank)
    main(local_rank=local_rank, world_size=n_gpus, args=args)
    
    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    args = parse_args()
    n_gpus = torch.cuda.device_count()
    
    if n_gpus > 1:
        mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args), join=True)
    else:
        main(local_rank=0, world_size=1, args=args)