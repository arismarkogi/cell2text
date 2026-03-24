import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import argparse
from datetime import datetime
import yaml
from tqdm import tqdm
import torch.nn as nn
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

from classifier import GeneformerDiseaseClassifier   

from dataset import MultiDatasetDiseaseDataset


import json




def save_label_mapping(label_to_idx, save_path):
    """Save label-to-index mapping to JSON."""
    # Convert to regular dict if it's defaultdict or similar
    mapping_dict = dict(label_to_idx)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(mapping_dict, f, indent=2, ensure_ascii=False)
    print(f"Label mapping saved to: {save_path}")


def load_diseases_from_csv(csv_path, sort=True):
    """
    Load disease types from CSV with format:
        disease_name, count, weight
    Returns sorted list of unique disease type names.
    """
    df = pd.read_csv(csv_path, header=None, names=['disease', 'count', 'weight'])
    diseases = df['disease'].dropna().str.strip().tolist()
    
    if sort:
        diseases.sort()
    
    print(f"Loaded {len(diseases)} disease types from {csv_path}")
    print(f"First 5: {diseases[:5]}")
    
    return diseases


def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    """Load checkpoint and return epoch, best_val_f1, and label mapping"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Load model state dict
    model.module.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state dict
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state dict
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Get epoch and best metrics
    start_epoch = checkpoint.get('epoch', 0) + 1  # Start from next epoch
    best_val_f1 = checkpoint.get('best_val_f1', 0.0)
    
    # Get label mapping if available
    label_mapping = checkpoint.get('label_mapping', None)
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 0)}")
    print(f"Best validation Macro F1 so far: {best_val_f1:.4f}")
    print(f"Resuming training from epoch {start_epoch}")
    
    if label_mapping:
        print(f"Loaded label mapping with {len(label_mapping)} disease types")
    else:
        print("Warning: No label mapping found in checkpoint")
    
    return start_epoch, best_val_f1, label_mapping

def collate_fn(batch):
    max_len = max(len(item['input_ids']) for item in batch)
    batch_size = len(batch)

    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    attention_masks = torch.zeros(batch_size, max_len, dtype=torch.long)
    labels = torch.zeros(batch_size, dtype=torch.long)  # ← single label per sample

    dataset_ids = []
    disease_ids = []

    for i, item in enumerate(batch):
        seq_len = len(item['input_ids'])
        input_ids[i, :seq_len] = item['input_ids']
        attention_masks[i, :seq_len] = item['attention_mask']
        labels[i] = item['labels']
        dataset_ids.append(item['dataset_id'])
        disease_ids.append(item['disease_id'])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'labels': labels,
        'dataset_ids': dataset_ids,
        'disease_ids': disease_ids
    }

def get_topk_predictions(logits, k=1):
    """For single-label, top-1 is usually sufficient."""
    batch_size, num_classes = logits.shape
    predictions = torch.zeros_like(logits)
    _, top_indices = torch.topk(logits, k, dim=1)
    predictions.scatter_(1, top_indices, 1)
    return predictions.long()

def train_epoch(model, train_loader, criterion, optimizer, scaler, rank, epoch):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    if rank == 0:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
    else:
        pbar = train_loader
    
    for batch in pbar:
        input_ids = batch['input_ids'].to(rank, non_blocking=True)
        attention_mask = batch['attention_mask'].to(rank, non_blocking=True)
        labels = batch['labels'].to(rank, non_blocking=True)  # ← shape (batch,)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)  # ← CrossEntropy expects (N,C) and (N,)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
        
        if rank == 0:
            pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss

def validate(model, val_loader, criterion, rank, world_size, k=1):
    model.eval()
    total_loss = 0.0
    all_preds = []   # will be class indices or one-hot
    all_labels = []
    num_batches = 0
    
    if rank == 0:
        pbar = tqdm(val_loader, desc="Validation")
    else:
        pbar = val_loader
    
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch['input_ids'].to(rank)
            attention_mask = batch['attention_mask'].to(rank)
            labels = batch['labels'].to(rank)  # ← shape (batch,)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            # Get predicted class indices
            preds = torch.argmax(logits, dim=1)  # shape (batch,)
            
            total_loss += loss.item()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            num_batches += 1
            
            if rank == 0:
                pbar.set_postfix({'loss': loss.item()})
    
    # Gather results from all processes
    if world_size > 1:
        device = torch.device(f"cuda:{rank}")
        
        all_preds_tensor = torch.cat(all_preds, dim=0).to(device)      # (N_local,)
        all_labels_tensor = torch.cat(all_labels, dim=0).to(device)    # (N_local,)

        local_pred_count = torch.tensor([all_preds_tensor.shape[0]], device=device)
        local_label_count = torch.tensor([all_labels_tensor.shape[0]], device=device)

        pred_counts = [torch.zeros_like(local_pred_count) for _ in range(world_size)]
        label_counts = [torch.zeros_like(local_label_count) for _ in range(world_size)]

        dist.all_gather(pred_counts, local_pred_count)
        dist.all_gather(label_counts, local_label_count)

        pred_counts = [int(x.item()) for x in pred_counts]
        label_counts = [int(x.item()) for x in label_counts]

        max_pred_count = max(pred_counts)
        max_label_count = max(label_counts)

        if all_preds_tensor.shape[0] < max_pred_count:
            pad_size = (max_pred_count - all_preds_tensor.shape[0],)
            pad_tensor = torch.zeros(pad_size, dtype=all_preds_tensor.dtype, device=device)
            all_preds_tensor = torch.cat([all_preds_tensor, pad_tensor], dim=0)

        if all_labels_tensor.shape[0] < max_label_count:
            pad_size = (max_label_count - all_labels_tensor.shape[0],)
            pad_tensor = torch.zeros(pad_size, dtype=all_labels_tensor.dtype, device=device)
            all_labels_tensor = torch.cat([all_labels_tensor, pad_tensor], dim=0)

        gathered_preds = [torch.zeros_like(all_preds_tensor, device=device) for _ in range(world_size)]
        gathered_labels = [torch.zeros_like(all_labels_tensor, device=device) for _ in range(world_size)]

        dist.all_gather(gathered_preds, all_preds_tensor)
        dist.all_gather(gathered_labels, all_labels_tensor)

        preds_list = []
        labels_list = []
        for i in range(world_size):
            preds_list.append(gathered_preds[i][:pred_counts[i]].cpu())
            labels_list.append(gathered_labels[i][:label_counts[i]].cpu())

        all_preds = torch.cat(preds_list, dim=0).numpy()
        all_labels = torch.cat(labels_list, dim=0).numpy()
    else:
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # Calculate metrics (only on rank 0)
    if rank == 0:
        from sklearn.metrics import f1_score, accuracy_score

        acc = accuracy_score(all_labels, all_preds)
        micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        
        metrics = {
            "accuracy": acc,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1
        }
    else:
        metrics = {}
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, metrics

def save_model(model, optimizer, scheduler, epoch, metrics, best_val_f1, save_path, label_mapping=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_f1': best_val_f1,
        'metrics': metrics,
        'label_mapping': label_mapping,  # Save the label mapping
    }
    
    torch.save(checkpoint, save_path)
   


# ----------------------------
# MAIN
# ----------------------------

def main(rank, world_size, config_path, checkpoint_path=None, label_mapping_json=None):
    setup_ddp(rank, world_size)

    try:
        config = load_config(config_path)

        # -------------------------
        # Step 1: Load label mapping
        # -------------------------
        if not label_mapping_json:
            raise ValueError("You must provide --label-mapping JSON file for training")

        if rank == 0:
            print(f"Loading label mapping from {label_mapping_json}")

        with open(label_mapping_json, "r") as f:
            idx_to_label = json.load(f)
            idx_to_label = {int(k): v for k, v in idx_to_label.items()}

        target_diseases = [idx_to_label[i] for i in range(len(idx_to_label))]
        disease_names = target_diseases.copy()

        if rank == 0:
            print(f"Loaded {len(target_diseases)} disease type names")

        # -------------------------
        # Step 2: Create datasets
        # -------------------------
        train_dataset = MultiDatasetDiseaseDataset(
            config["data"]["base_data_path"],
            split="train",
            target_diseases=target_diseases,
            label_key=config["data"].get("label_key", "disease"),
        )

        val_dataset = MultiDatasetDiseaseDataset(
            config["data"]["base_data_path"],
            split="val",
            target_diseases=target_diseases,
            label_key=config["data"].get("label_key", "disease"),
        )

        if rank == 0:
            print(f"Training samples: {len(train_dataset)}")
            print(f"Validation samples: {len(val_dataset)}")
            print(f"Number of disease types: {train_dataset.num_diseases}")

        # -------------------------
        # Step 3: Create model
        # -------------------------
        model = GeneformerDiseaseClassifier(
            config["model"]["geneformer_model_path"],
            num_diseases=train_dataset.num_diseases,
            freeze_geneformer=config["model"]["freeze_geneformer"],
        ).to(rank)
        model = DDP(model, device_ids=[rank])

        # -------------------------
        # Step 4: Optimizer / Scheduler
        # -------------------------
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )
        scaler = torch.cuda.amp.GradScaler()

        start_epoch = 0
        best_val_f1 = 0.0

        # -------------------------
        # Step 5: Resume checkpoint if provided
        # -------------------------
        if checkpoint_path:
            if rank == 0:
                print(f"Resuming from checkpoint: {checkpoint_path}")
            start_epoch, best_val_f1, _ = load_checkpoint(
                checkpoint_path, model, optimizer, scheduler
            )

        # -------------------------
        # Step 6: Data loaders
        # -------------------------
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            sampler=val_sampler,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True,
        )

        # -------------------------
        # Step 7: Training loop (simplified, just one epoch shown)
        # -------------------------
        num_epochs = config["training"].get("epochs", 1)  # fallback default
        for epoch in range(start_epoch, start_epoch + num_epochs):
        
            train_sampler.set_epoch(epoch)

            train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, rank, epoch)
            val_loss, val_metrics = validate(model, val_loader, criterion, rank, world_size)

            if rank == 0:
                print(f"\nEpoch {epoch+1}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")

                if val_metrics:
                    for k, v in val_metrics.items():
                        print(f"  {k}: {v:.4f}")

                    current_f1 = val_metrics.get("macro_f1", 0.0)
                    if current_f1 > best_val_f1:
                        best_val_f1 = current_f1
                        print(f"  New best validation Macro F1: {current_f1:.4f}")

                    save_path = os.path.join(
                        "/home/arism/disease_results_fromcheckpoint",
                        f"best_model.pt"
                    )
                    save_model(model, optimizer, scheduler, epoch, val_metrics, best_val_f1, save_path, train_dataset.disease_to_idx)

                scheduler.step(val_metrics.get("macro_f1", 0.0))

        if rank == 0:
            print(f"Training completed. Best validation Macro F1: {best_val_f1:.4f}")

    finally:
        cleanup_ddp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/home/arism/cell2text/disease_classification/default.yaml', help='Config file path')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file to resume from')
    parser.add_argument('--world-size', type=int, default=torch.cuda.device_count(), 
                        help='Number of GPUs to use')
    parser.add_argument('--label-mapping', help='Path to idx_to_label.json for disease type names')

    args = parser.parse_args()
    
    if args.world_size < 2:
        print(f"Warning: Only {args.world_size} GPU(s) available. DDP works best with 2+ GPUs.")
    
    torch.multiprocessing.spawn(
        main,
        args=(args.world_size, args.config, args.checkpoint, args.label_mapping),
        nprocs=args.world_size,
        join=True
        )
