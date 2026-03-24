import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import argparse
from datetime import datetime
import yaml

from dataset import MultiDatasetPathwayDataset
from classifier import GeneformerPathwayClassifier
from trainer import FocalLoss
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def setup_ddp(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def collate_fn(batch):
    """Collate function for batching"""
    max_len = max(len(item['input_ids']) for item in batch)
    batch_size = len(batch)

    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    attention_masks = torch.zeros(batch_size, max_len, dtype=torch.long)
    num_pathways = len(batch[0]['labels'])
    labels = torch.zeros(batch_size, num_pathways, dtype=torch.float)

    dataset_ids = []
    cell_ids = []

    for i, item in enumerate(batch):
        seq_len = len(item['input_ids'])
        input_ids[i, :seq_len] = item['input_ids']
        attention_masks[i, :seq_len] = item['attention_mask']
        labels[i, :num_pathways] = torch.tensor(item['labels'], dtype=torch.float)
        dataset_ids.append(item['dataset_id'])
        cell_ids.append(item['cell_id'])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'labels': labels,
        'dataset_ids': dataset_ids,
        'cell_ids': cell_ids
    }


def get_topk_predictions(logits, k=2):
    """Get top-k predictions"""
    batch_size, num_pathways = logits.shape
    predictions = torch.zeros_like(logits)
    _, top_indices = torch.topk(logits, k, dim=1)
    predictions.scatter_(1, top_indices, 1)
    return predictions.long()


def train_epoch(model, train_loader, criterion, optimizer, scaler, rank, epoch):
    """Train one epoch"""
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
        labels = batch['labels'].to(rank, non_blocking=True)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels.float())
        
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


def validate(model, val_loader, criterion, rank, world_size, k_pathways=2):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    all_preds_topk = []
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
            labels = batch['labels'].to(rank)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels.float())
            
            preds_topk = get_topk_predictions(logits, k_pathways)
            
            total_loss += loss.item()
            all_preds_topk.append(preds_topk.cpu())
            all_labels.append(labels.cpu())
            num_batches += 1
            
            if rank == 0:
                pbar.set_postfix({'loss': loss.item()})
    
    # Gather results from all processes
    if world_size > 1:
        # Concatenate local lists -> local tensors (on CPU currently), move to GPU device
        device = torch.device(f"cuda:{rank}")
        all_preds_tensor = torch.cat(all_preds_topk, dim=0).to(device)  # shape (N_local, C)
        all_labels_tensor = torch.cat(all_labels, dim=0).to(device)     # shape (N_local, C)

        # 1) gather sizes from all ranks so we can unpad after gather
        local_pred_count = torch.tensor([all_preds_tensor.shape[0]], device=device)
        local_label_count = torch.tensor([all_labels_tensor.shape[0]], device=device)

        pred_counts = [torch.zeros_like(local_pred_count) for _ in range(world_size)]
        label_counts = [torch.zeros_like(local_label_count) for _ in range(world_size)]

        dist.all_gather(pred_counts, local_pred_count)
        dist.all_gather(label_counts, local_label_count)

        pred_counts = [int(x.item()) for x in pred_counts]
        label_counts = [int(x.item()) for x in label_counts]

        # 2) pad local tensors to the max length so all ranks have same shape for all_gather
        max_pred_count = max(pred_counts)
        max_label_count = max(label_counts)

        if all_preds_tensor.shape[0] < max_pred_count:
            pad_size = (max_pred_count - all_preds_tensor.shape[0], all_preds_tensor.shape[1])
            pad_tensor = torch.zeros(pad_size, dtype=all_preds_tensor.dtype, device=device)
            all_preds_tensor = torch.cat([all_preds_tensor, pad_tensor], dim=0)

        if all_labels_tensor.shape[0] < max_label_count:
            pad_size = (max_label_count - all_labels_tensor.shape[0], all_labels_tensor.shape[1])
            pad_tensor = torch.zeros(pad_size, dtype=all_labels_tensor.dtype, device=device)
            all_labels_tensor = torch.cat([all_labels_tensor, pad_tensor], dim=0)

        # 3) prepare gather lists (on the same device) and all_gather
        gathered_preds = [torch.zeros_like(all_preds_tensor, device=device) for _ in range(world_size)]
        gathered_labels = [torch.zeros_like(all_labels_tensor, device=device) for _ in range(world_size)]

        dist.all_gather(gathered_preds, all_preds_tensor)
        dist.all_gather(gathered_labels, all_labels_tensor)

        # 4) unpad and concatenate, move to CPU and numpy
        preds_list = []
        labels_list = []
        for i in range(world_size):
            preds_list.append(gathered_preds[i][:pred_counts[i]].cpu())
            labels_list.append(gathered_labels[i][:label_counts[i]].cpu())

        all_preds_topk = torch.cat(preds_list, dim=0).numpy()
        all_labels = torch.cat(labels_list, dim=0).numpy()
    else:
        all_preds_topk = torch.cat(all_preds_topk, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # Calculate metrics (only on rank 0)
    if rank == 0:
        from sklearn.metrics import f1_score, accuracy_score, hamming_loss
        
        subset_acc = accuracy_score(all_labels, all_preds_topk)
        hamming_loss_val = hamming_loss(all_labels, all_preds_topk)
        micro_f1 = f1_score(all_labels, all_preds_topk, average="micro", zero_division=0)
        macro_f1 = f1_score(all_labels, all_preds_topk, average="macro", zero_division=0)
        
        metrics = {
            "subset_accuracy": subset_acc,
            "hamming_loss": hamming_loss_val,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
        }
    else:
        metrics = {}
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, metrics


def save_model(model, optimizer, scheduler, epoch, metrics, best_val_f1, k_pathways, save_path):
    """Save model checkpoint (only on rank 0)"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_f1': best_val_f1,
        'metrics': metrics,
        'k_pathways': k_pathways,
    }
    
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")


def main(rank, world_size, config_path):
    """Main training function for each process"""
    # Setup DDP
    setup_ddp(rank, world_size)
    
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Create datasets
        train_dataset = MultiDatasetPathwayDataset(
            config['data']['base_data_path'], split='train'
        )
        val_dataset = MultiDatasetPathwayDataset(
            config['data']['base_data_path'], split='val'
        )
        
        # Create model and move to GPU
        model = GeneformerPathwayClassifier(
            config['model']['geneformer_model_path'],
            num_pathways=train_dataset.num_pathways,
            freeze_geneformer=config['model']['freeze_geneformer']
        )
        model = model.to(rank)
        
        # Wrap with DDP AFTER moving to GPU
        model = DDP(model, device_ids=[rank])
        
        # Create distributed samplers
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        
        # Create data loaders
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
        
        # Setup training components
        criterion = FocalLoss(alpha=1.0, gamma=2.0)
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=float(config['training']['learning_rate']), 
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        scaler = torch.cuda.amp.GradScaler()
        k_pathways = config['training']['k_pathways']
        best_val_f1 = 0.0
        
        if rank == 0:
            print("Starting DDP training...")
            print(f"World size: {world_size}")
            print(f"Training samples: {len(train_dataset)}")
            print(f"Validation samples: {len(val_dataset)}")
            print(f"Using top-{k_pathways} predictions")
        
        # Generate save path (only rank 0 saves)
        if rank == 0:
            current_date = datetime.now()
            date_str = current_date.strftime("%Y-%m-%d")
            save_path = f"/home/arism/pathway_results/{date_str}_ddp_best_model.pt"
            print(f"Model will be saved to: {save_path}")
        
        # Training loop
        num_epochs = config['training']['num_epochs']
        for epoch in range(num_epochs):
            train_sampler.set_epoch(epoch)  # Important for proper shuffling
            
            train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, rank, epoch)
            val_loss, val_metrics = validate(model, val_loader, criterion, rank, world_size, k_pathways)
            
            if rank == 0:
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                
                if val_metrics:
                    for k, v in val_metrics.items():
                        print(f"  {k}: {v:.4f}")
                    
                    # Save best model
                    current_f1 = val_metrics["macro_f1"]
                    if current_f1 > best_val_f1:
                        best_val_f1 = current_f1
                        save_model(model, optimizer, scheduler, epoch, val_metrics, 
                                 best_val_f1, k_pathways, save_path)
                        print(f"  New best validation Macro F1: {current_f1:.4f}")
            
            # Update scheduler
            if val_metrics and rank == 0:
                scheduler.step(val_metrics.get("macro_f1", 0.0))
        
        if rank == 0:
            print(f"Training completed. Best validation Macro F1: {best_val_f1:.4f}")
        
    finally:
        # Clean up
        cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/home/arism/cell2text/pathway_classification/default.yaml', help='Config file path')
    parser.add_argument('--world-size', type=int, default=torch.cuda.device_count(), 
                        help='Number of GPUs to use')
    args = parser.parse_args()
    
    # Check if we have multiple GPUs
    if args.world_size < 2:
        print(f"Warning: Only {args.world_size} GPU(s) available. DDP works best with 2+ GPUs.")
        print("Consider using single-GPU training instead.")
    
    # Launch distributed training
    torch.multiprocessing.spawn(
        main,
        args=(args.world_size, args.config),
        nprocs=args.world_size,
        join=True
    )