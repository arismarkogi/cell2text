import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, roc_auc_score
import yaml
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')

from dataset import MultiDatasetPathwayDataset
from classifier import GeneformerPathwayClassifier


def setup_ddp(rank, world_size):
    """Initialize the distributed environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'  # Different port from training
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up the distributed environment"""
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
    """Get top-k predictions based on logits"""
    batch_size, num_pathways = logits.shape
    predictions = torch.zeros_like(logits)
    _, top_indices = torch.topk(logits, k, dim=1)
    predictions.scatter_(1, top_indices, 1)
    return predictions.long()


def gather_from_all_processes(data_tensor, world_size, rank):
    """Gather tensors from all processes"""
    if world_size == 1:
        return data_tensor.cpu().numpy()
    
    # Ensure data_tensor is on GPU for NCCL operations
    if data_tensor.device.type == 'cpu':
        data_tensor = data_tensor.cuda(rank)
    
    device = data_tensor.device
    
    # Get local size
    local_size = torch.tensor([data_tensor.shape[0]], device=device)
    size_list = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    
    # Get max size for padding
    max_size = max([int(s.item()) for s in size_list])
    sizes = [int(s.item()) for s in size_list]
    
    # Pad tensor if necessary
    if data_tensor.shape[0] < max_size:
        pad_shape = list(data_tensor.shape)
        pad_shape[0] = max_size - data_tensor.shape[0]
        pad_tensor = torch.zeros(pad_shape, dtype=data_tensor.dtype, device=device)
        padded_tensor = torch.cat([data_tensor, pad_tensor], dim=0)
    else:
        padded_tensor = data_tensor
    
    # Gather all tensors
    gathered_list = [torch.zeros_like(padded_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_list, padded_tensor)
    
    # Unpad and concatenate
    result_list = []
    for i, tensor in enumerate(gathered_list):
        result_list.append(tensor[:sizes[i]])
    
    result = torch.cat(result_list, dim=0).cpu().numpy()
    return result


def gather_lists_from_all_processes(data_list, world_size, rank):
    """Gather Python lists from all processes"""
    if world_size == 1:
        return data_list
    
    # Convert to tensor for gathering
    gathered_data = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_data, data_list)
    
    # Flatten the list of lists
    result = []
    for sublist in gathered_data:
        result.extend(sublist)
    
    return result


def evaluate_ddp(model, test_loader, rank, world_size, num_pathways, k_pathways=2):
    """Evaluate model with DDP"""
    model.eval()
    
    local_logits = []
    local_labels = []
    local_dataset_ids = []
    local_cell_ids = []
    
    # Only show progress on rank 0
    if rank == 0:
        pbar = tqdm(test_loader, desc="Evaluating")
    else:
        pbar = test_loader
    
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch['input_ids'].to(rank)
            attention_mask = batch['attention_mask'].to(rank)
            labels = batch['labels'].to(rank)
            
            logits = model(input_ids, attention_mask)
            
            # Keep tensors on GPU for gathering
            local_logits.append(logits)
            local_labels.append(labels)
            local_dataset_ids.extend(batch['dataset_ids'])
            local_cell_ids.extend(batch['cell_ids'])
    
    # Handle case where process has no data
    if len(local_logits) > 0:
        local_logits_tensor = torch.cat(local_logits, dim=0)
        local_labels_tensor = torch.cat(local_labels, dim=0)
    else:
        # Create empty tensors with correct shape for processes with no data
        local_logits_tensor = torch.empty(0, num_pathways, device=f'cuda:{rank}', dtype=torch.float32)
        local_labels_tensor = torch.empty(0, num_pathways, device=f'cuda:{rank}', dtype=torch.float32)
    
    # Gather from all processes
    all_logits = gather_from_all_processes(local_logits_tensor, world_size, rank)
    all_labels = gather_from_all_processes(local_labels_tensor, world_size, rank)
    all_dataset_ids = gather_lists_from_all_processes(local_dataset_ids, world_size, rank)
    all_cell_ids = gather_lists_from_all_processes(local_cell_ids, world_size, rank)
    
    return {
        'logits': all_logits,
        'labels': all_labels,
        'dataset_ids': all_dataset_ids,
        'cell_ids': all_cell_ids
    }


def calculate_metrics(results, k_pathways=2):
    """Calculate metrics (only on rank 0)"""
    logits = results['logits']
    labels = results['labels']
    
    # Calculate probabilities
    probs = 1 / (1 + np.exp(-logits))  # Sigmoid
    
    # Get predictions
    preds_threshold = (probs >= 0.5).astype(int)
    
    # Top-k predictions
    preds_topk = np.zeros_like(logits, dtype=int)
    for i in range(len(logits)):
        top_indices = np.argsort(logits[i])[-k_pathways:]
        preds_topk[i, top_indices] = 1
    
    metrics = {}
    
    # Threshold-based metrics
    metrics['threshold'] = {
        'subset_accuracy': accuracy_score(labels, preds_threshold),
        'hamming_loss': hamming_loss(labels, preds_threshold),
        'micro_f1': f1_score(labels, preds_threshold, average="micro", zero_division=0),
        'macro_f1': f1_score(labels, preds_threshold, average="macro", zero_division=0),
    }
    
    # Top-k metrics
    metrics[f'top{k_pathways}'] = {
        'subset_accuracy': accuracy_score(labels, preds_topk),
        'hamming_loss': hamming_loss(labels, preds_topk),
        'micro_f1': f1_score(labels, preds_topk, average="micro", zero_division=0),
        'macro_f1': f1_score(labels, preds_topk, average="macro", zero_division=0),
    }
    
    # ROC AUC
    try:
        metrics['roc_auc_macro'] = roc_auc_score(labels, probs, average="macro")
    except ValueError:
        metrics['roc_auc_macro'] = np.nan
    
    # Add predictions to results for saving
    results['probabilities'] = probs
    results['predictions_threshold'] = preds_threshold
    results['predictions_topk'] = preds_topk
    
    return metrics


def print_results(metrics, num_samples, rank):
    """Print results (only on rank 0)"""
    if rank != 0:
        return
        
    print("\n" + "="*70)
    print("DDP TEST SET EVALUATION RESULTS")
    print("="*70)
    print(f"Total samples: {num_samples}")
    
    print(f"\nTHRESHOLD (≥0.5) METRICS:")
    print("-" * 40)
    for k, v in metrics['threshold'].items():
        print(f"  {k.replace('_', ' ').title()}: {v:.4f}")
    
    k_pathways = 2
    for key in metrics.keys():
        if key.startswith('top'):
            k_pathways = int(key[3:])
            break
    
    print(f"\nTOP-{k_pathways} METRICS:")
    print("-" * 40)
    for k, v in metrics[f'top{k_pathways}'].items():
        print(f"  {k.replace('_', ' ').title()}: {v:.4f}")
    
    if not np.isnan(metrics['roc_auc_macro']):
        print(f"\nROC AUC (Macro): {metrics['roc_auc_macro']:.4f}")
    
    print("="*70)


def save_results(results, metrics, k_pathways, save_dir, rank):
    """Save results (only on rank 0)"""
    if rank != 0:
        return
        
    os.makedirs(save_dir, exist_ok=True)
    
    # Save predictions CSV
    predictions_df = pd.DataFrame({
        'cell_id': results['cell_ids'],
        'dataset_id': results['dataset_ids'],
    })
    
    num_pathways = results['labels'].shape[1]
    for i in range(num_pathways):
        predictions_df[f'pathway_{i}_true'] = results['labels'][:, i]
        predictions_df[f'pathway_{i}_prob'] = results['probabilities'][:, i]
        predictions_df[f'pathway_{i}_pred_threshold'] = results['predictions_threshold'][:, i]
        predictions_df[f'pathway_{i}_pred_top{k_pathways}'] = results['predictions_topk'][:, i]
    
    predictions_df.to_csv(f"{save_dir}/predictions.csv", index=False)
    
    # Save metrics
    with open(f"{save_dir}/metrics.txt", "w") as f:
        f.write("DDP TEST SET EVALUATION METRICS\n")
        f.write("=" * 40 + "\n\n")
        
        f.write("THRESHOLD (≥0.5) METRICS:\n")
        for k, v in metrics['threshold'].items():
            f.write(f"  {k.replace('_', ' ').title()}: {v:.4f}\n")
        
        f.write(f"\nTOP-{k_pathways} METRICS:\n")
        for k, v in metrics[f'top{k_pathways}'].items():
            f.write(f"  {k.replace('_', ' ').title()}: {v:.4f}\n")
        
        if not np.isnan(metrics['roc_auc_macro']):
            f.write(f"\nROC AUC (Macro): {metrics['roc_auc_macro']:.4f}\n")
    
    print(f"\nResults saved to {save_dir}/")
    print(f"  - predictions.csv")
    print(f"  - metrics.txt")


def main(rank, world_size, config_path, checkpoint_path, output_dir):
    """Main evaluation function for each process"""
    setup_ddp(rank, world_size)
    
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Create test dataset
        if rank == 0:
            print("Loading test dataset...")
        
        test_dataset = MultiDatasetPathwayDataset(
            config['data']['base_data_path'], 
            split='test'
        )
        
        if rank == 0:
            print(f"Test dataset size: {len(test_dataset)}")
        
        # Create model
        model = GeneformerPathwayClassifier(
            config['model']['geneformer_model_path'],
            num_pathways=test_dataset.num_pathways,
            freeze_geneformer=config['model']['freeze_geneformer']
        )
        
        # Load checkpoint
        if rank == 0:
            print("Loading model weights...")
        
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{rank}', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(rank)
        
        # Wrap with DDP (even for inference, to maintain consistency)
        model = DDP(model, device_ids=[rank])
        
        # Get k_pathways from checkpoint
        k_pathways = checkpoint.get('k_pathways', config['training']['k_pathways'])
        
        if rank == 0:
            print(f"Using k_pathways: {k_pathways}")
            print(f"Number of pathways: {test_dataset.num_pathways}")
        
        # Create distributed sampler and dataloader
        test_sampler = DistributedSampler(
            test_dataset, 
            num_replicas=world_size, 
            rank=rank, 
            shuffle=False,
            drop_last=False  # Important: don't drop last incomplete batch
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size'],
            sampler=test_sampler,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        # Run evaluation with num_pathways parameter
        results = evaluate_ddp(model, test_loader, rank, world_size, test_dataset.num_pathways, k_pathways)
        
        # Calculate metrics only on rank 0
        if rank == 0:
            metrics = calculate_metrics(results, k_pathways)
            
            # Print results
            print_results(metrics, len(test_dataset), rank)
            
            # Save results
            save_results(results, metrics, k_pathways, output_dir, rank)
            
            print("\nDDP evaluation completed successfully!")
        
        # Wait for all processes to finish
        dist.barrier()
        
    finally:
        cleanup_ddp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/home/arism/cell2text/pathway_classification/default.yaml', help='Config file path')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--output-dir', default='test_results_ddp', help='Output directory for results')
    parser.add_argument('--world-size', type=int, default=torch.cuda.device_count(), 
                        help='Number of GPUs to use')
    args = parser.parse_args()
    
    if args.world_size < 1:
        print("Error: No GPUs available")
        exit(1)
    
    # Launch distributed evaluation
    torch.multiprocessing.spawn(
        main,
        args=(args.world_size, args.config, args.checkpoint, args.output_dir),
        nprocs=args.world_size,
        join=True
    )