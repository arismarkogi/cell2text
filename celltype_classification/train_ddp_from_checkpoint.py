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

from classifier import GeneformerCellTypeClassifier

from dataset import MultiDatasetCellTypeDataset


import json


# Add this debugging code to your training script

def debug_batch_and_model(model, batch, criterion, rank):
    """Debug function to analyze what's happening in training"""
    print(f"\n=== DEBUGGING BATCH (Rank {rank}) ===")
    
    input_ids = batch['input_ids'].to(rank)
    attention_mask = batch['attention_mask'].to(rank)
    labels = batch['labels'].to(rank)
    
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels dtype: {labels.dtype}")
    
    print(f"Input IDs range: [{input_ids.min().item()}, {input_ids.max().item()}]")
    print(f"Labels range: [{labels.min().item()}, {labels.max().item()}]")
    print(f"Unique labels in batch: {torch.unique(labels).cpu().numpy()}")
    
    # Check for any invalid labels
    if labels.max().item() >= model.module.num_cell_types:
        print(f"ERROR: Label {labels.max().item()} >= num_classes {model.module.num_cell_types}")
    
    # Forward pass
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        print(f"Logits shape: {logits.shape}")
        print(f"Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
        print(f"Logits mean: {logits.mean().item():.4f}")
        print(f"Logits std: {logits.std().item():.4f}")
        
        # Check if logits are reasonable
        probs = torch.softmax(logits, dim=1)
        print(f"Max probability in batch: {probs.max().item():.4f}")
        print(f"Min probability in batch: {probs.min().item():.4f}")
        
        # Calculate loss
        loss = criterion(logits, labels)
        print(f"Loss: {loss.item():.4f}")
        
        # Check predictions
        preds = torch.argmax(logits, dim=1)
        print(f"Predictions: {preds.cpu().numpy()[:10]}")  # First 10
        print(f"True labels: {labels.cpu().numpy()[:10]}")  # First 10
        
        # Accuracy for this batch
        acc = (preds == labels).float().mean()
        print(f"Batch accuracy: {acc.item():.4f}")
    
    print("=== END DEBUG ===\n")

def debug_dataset_info(dataset, rank):
    """Debug dataset information"""
    if rank == 0:
        print(f"\n=== DATASET DEBUG ===")
        print(f"Dataset length: {len(dataset)}")
        print(f"Num cell types: {dataset.num_cell_types}")
        print(f"Cell type mapping size: {len(dataset.cell_type_to_idx)}")
        
        # Sample a few items
        for i in range(min(3, len(dataset))):
            item = dataset[i]
            print(f"\nSample {i}:")
            print(f"  Input IDs length: {len(item['input_ids'])}")
            print(f"  Label: {item['labels']}")
            print(f"  Dataset ID: {item['dataset_id']}")
            print(f"  Cell ID: {item['cell_id']}")
        
        # Check label distribution
        print(f"\nLabel distribution check...")
        label_counts = {}
        for i in range(min(1000, len(dataset))):  # Check first 1000 samples
            item = dataset[i]
            label = item['labels']
            if isinstance(label, torch.Tensor):
                label = label.item()
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"Found {len(label_counts)} unique labels in first {min(1000, len(dataset))} samples")
        print(f"Label range: [{min(label_counts.keys())}, {max(label_counts.keys())}]")
        if max(label_counts.keys()) >= dataset.num_cell_types:
            print(f"ERROR: Found label {max(label_counts.keys())} but num_cell_types is {dataset.num_cell_types}")
        
        # Show most common labels
        sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        print("Most common labels:", sorted_labels[:10])
        print("=== END DATASET DEBUG ===\n")

# Modified train_epoch function with debugging
def train_epoch_debug(model, train_loader, criterion, optimizer, scaler, rank, epoch):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    if rank == 0:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
    else:
        pbar = train_loader
    
    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(rank, non_blocking=True)
        attention_mask = batch['attention_mask'].to(rank, non_blocking=True)
        labels = batch['labels'].to(rank, non_blocking=True)
        
        # Debug first batch of first epoch
        if batch_idx == 0 and epoch == 0 and rank == 0:
            debug_batch_and_model(model, batch, criterion, rank)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
        
        # Check for NaN or inf loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: Invalid loss detected: {loss.item()}")
            continue
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
        
        if rank == 0:
            pbar.set_postfix({'loss': loss.item()})
        
        # Early break for testing - remove this for full training
        if batch_idx >= 10:  # Only process first 10 batches for debugging
            break
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss

# Add this to your main function before training starts:
def debug_main_additions(rank, train_dataset, val_dataset, model):
    """Add these calls to your main function"""
    if rank == 0:
        print(f"\n=== PRE-TRAINING DEBUGGING ===")
        
        # Debug datasets
        debug_dataset_info(train_dataset, rank)
        debug_dataset_info(val_dataset, rank)
        
        # Check model architecture
        print(f"Model num_cell_types: {model.module.num_cell_types}")
        print(f"Model classifier layer: {model.module.classifier}")
        
        # Check if the model's classifier matches the number of cell types
        if hasattr(model.module, 'classifier'):
            classifier_out_features = model.module.classifier.out_features
            print(f"Classifier output features: {classifier_out_features}")
            if classifier_out_features != model.module.num_cell_types:
                print(f"ERROR: Mismatch between classifier output ({classifier_out_features}) and num_cell_types ({model.module.num_cell_types})")
        
        print("=== END PRE-TRAINING DEBUGGING ===\n")




def save_label_mapping(label_to_idx, save_path):
    """Save label-to-index mapping to JSON."""
    # Convert to regular dict if it's defaultdict or similar
    mapping_dict = dict(label_to_idx)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(mapping_dict, f, indent=2, ensure_ascii=False)
    print(f"Label mapping saved to: {save_path}")


def load_cell_types_from_csv(csv_path, sort=True):
    """
    Load cell types from CSV with format:
        cell_type_name, count, weight
    Returns sorted list of unique cell type names.
    """
    df = pd.read_csv(csv_path, header=None, names=['cell_type', 'count', 'weight'])
    cell_types = df['cell_type'].dropna().str.strip().tolist()
    
    if sort:
        cell_types.sort()
    
    print(f"Loaded {len(cell_types)} cell types from {csv_path}")
    print(f"First 5: {cell_types[:5]}")
    
    return cell_types


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
        print(f"Loaded label mapping with {len(label_mapping)} cell types")
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
    cell_ids = []

    for i, item in enumerate(batch):
        seq_len = len(item['input_ids'])
        input_ids[i, :seq_len] = item['input_ids']
        attention_masks[i, :seq_len] = item['attention_mask']
        labels[i] = item['labels']
        dataset_ids.append(item['dataset_id'])
        cell_ids.append(item['cell_id'])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'labels': labels,
        'dataset_ids': dataset_ids,
        'cell_ids': cell_ids
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

def main(rank, world_size, config_path, checkpoint_path=None):
    setup_ddp(rank, world_size)
    
    try:
        config = load_config(config_path)

        # Initialize variables
        start_epoch = 0
        best_val_f1 = 0.0
        loaded_label_mapping = None
        target_cell_types = None

        # STEP 1: Determine target cell types
        if checkpoint_path:
            # Load checkpoint first to get the label mapping
            if rank == 0:
                print(f"Loading checkpoint from: {checkpoint_path}")
            
            # Create a temporary model to load checkpoint
            temp_model = GeneformerCellTypeClassifier(
                config['model']['geneformer_model_path'],
                num_cell_types=784,  
                freeze_geneformer=config['model']['freeze_geneformer']
            ).to(rank)
            temp_model = DDP(temp_model, device_ids=[rank])
            
            temp_optimizer = torch.optim.AdamW(temp_model.parameters(), lr=1e-4)
            temp_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(temp_optimizer)
            
            start_epoch, best_val_f1, loaded_label_mapping = load_checkpoint(
                checkpoint_path, temp_model, temp_optimizer, temp_scheduler
            )
            
            if loaded_label_mapping:
                target_cell_types = list(loaded_label_mapping.keys())
                if rank == 0:
                    print(f"Using label mapping from checkpoint ({len(target_cell_types)} cell types)")
            
            del temp_model, temp_optimizer, temp_scheduler
            torch.cuda.empty_cache()
        
        if target_cell_types is None:
            # No checkpoint or no mapping in checkpoint, try CSV
            cell_type_csv = config['data'].get('cell_type_csv')
            if cell_type_csv and os.path.exists(cell_type_csv):
                target_cell_types = load_cell_types_from_csv(cell_type_csv)
            else:
                if rank == 0:
                    print("Warning: No checkpoint mapping or cell_type_csv. Using default cell types.")
                target_cell_types = None
        
        # STEP 2: Create datasets ONCE with determined target cell types
        train_dataset = MultiDatasetCellTypeDataset(
            config['data']['base_data_path'], 
            split='train',
            target_cell_types=target_cell_types,
            label_key=config['data'].get('label_key', 'cell_type')
        )
        
        val_dataset = MultiDatasetCellTypeDataset(
            config['data']['base_data_path'], 
            split='val',
            target_cell_types=train_dataset.target_cell_types,  # ensure same mapping
            label_key=config['data'].get('label_key', 'cell_type')
        )
        
        # STEP 3: Override dataset mapping if we loaded from checkpoint
        if loaded_label_mapping:
            train_dataset.cell_type_to_idx = loaded_label_mapping
            train_dataset.idx_to_cell_type = {idx: cell_type for cell_type, idx in loaded_label_mapping.items()}
            train_dataset.num_cell_types = len(loaded_label_mapping)
            
            # Ensure val dataset uses same mapping
            val_dataset.cell_type_to_idx = train_dataset.cell_type_to_idx
            val_dataset.idx_to_cell_type = train_dataset.idx_to_cell_type
            val_dataset.num_cell_types = train_dataset.num_cell_types
            
            if rank == 0:
                print(f"Overrode dataset mapping with checkpoint mapping")
        
        # STEP 4: Save label mapping if this is a new training run
        if rank == 0 and not loaded_label_mapping:
            mapping_save_dir = "/home/arism/celltype_results_fromcheckpoint"
            os.makedirs(mapping_save_dir, exist_ok=True)
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            mapping_save_path = os.path.join(mapping_save_dir, f"{current_date}_label_mapping.json")
            label_to_idx = train_dataset.cell_type_to_idx
            save_label_mapping(label_to_idx, mapping_save_path)
            
            # Also save reverse mapping
            idx_to_label = {idx: label for label, idx in label_to_idx.items()}
            reverse_mapping_path = os.path.join(mapping_save_dir, f"{current_date}_idx_to_label.json")
            with open(reverse_mapping_path, 'w', encoding='utf-8') as f:
                json.dump(idx_to_label, f, indent=2, ensure_ascii=False)
            print(f"Label mappings saved")
        
        # STEP 5: Create model with correct number of cell types
        model = GeneformerCellTypeClassifier(
            config['model']['geneformer_model_path'],
            num_cell_types=train_dataset.num_cell_types,
            freeze_geneformer=config['model']['freeze_geneformer']
        )
        model = model.to(rank)
        model = DDP(model, device_ids=[rank])
        
        # STEP 6: Create data loaders
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            sampler=val_sampler,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True
        )
        
        # STEP 7: Setup training components
        criterion = nn.CrossEntropyLoss()
        
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=float(config['training']['learning_rate']), 
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        scaler = torch.cuda.amp.GradScaler()
        
        # STEP 8: Load checkpoint into actual model if resuming
        if checkpoint_path and loaded_label_mapping:
            start_epoch, best_val_f1, _ = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
        
        if rank == 0:
            print(f"Training samples: {len(train_dataset)}")
            print(f"Validation samples: {len(val_dataset)}")
            print(f"Number of cell types: {train_dataset.num_cell_types}")
            
            # Setup save path
            current_date = datetime.now().strftime("%Y-%m-%d_%H-%M")
            save_dir = "/home/arism/celltype_results_fromcheckpoint"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{current_date}_{'resumed' if checkpoint_path else 'ddp'}_best_model.pt")
            print(f"Model will be saved to: {save_path}")
        
        # STEP 9: Training loop
        end_epoch = start_epoch + 1
        
        for epoch in range(start_epoch, end_epoch):
            train_sampler.set_epoch(epoch)
            
            train_loss = train_epoch_debug(model, train_loader, criterion, optimizer, scaler, rank, epoch)
            val_loss, val_metrics = validate(model, val_loader, criterion, rank, world_size)
            
            if rank == 0:
                print(f"\nEpoch {epoch+1}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                
                if val_metrics:
                    for k, v in val_metrics.items():
                        print(f"  {k}: {v:.4f}")
                    
                    current_f1 = val_metrics["macro_f1"]
                    if current_f1 > best_val_f1:
                        best_val_f1 = current_f1
                        print(f"  New best validation Macro F1: {current_f1:.4f}")
                    
                    save_model(model, optimizer, scheduler, epoch, val_metrics, best_val_f1, save_path, train_dataset.cell_type_to_idx)
                    print(f"  Current F1: {current_f1:.4f}, Best F1: {best_val_f1:.4f}")
            
            if val_metrics and rank == 0:
                scheduler.step(val_metrics.get("macro_f1", 0.0))
        
        if rank == 0:
            print(f"Training completed. Final validation Macro F1: {val_metrics.get('macro_f1', 0.0):.4f}")
            print(f"Best validation Macro F1 overall: {best_val_f1:.4f}")
        
    finally:
        cleanup_ddp()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/home/arism/cell2text/celltype_classification/default.yaml', help='Config file path')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file to resume from')
    parser.add_argument('--world-size', type=int, default=torch.cuda.device_count(), 
                        help='Number of GPUs to use')
    args = parser.parse_args()
    
    if args.world_size < 2:
        print(f"Warning: Only {args.world_size} GPU(s) available. DDP works best with 2+ GPUs.")
    
    torch.multiprocessing.spawn(
        main,
        args=(args.world_size, args.config, args.checkpoint),
        nprocs=args.world_size,
        join=True
    )