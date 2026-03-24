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
import json
warnings.filterwarnings('ignore')

from dataset import MultiDatasetCellTypeDataset
from classifier import GeneformerCellTypeClassifier


import pickle



def setup_ddp(rank, world_size):
    """Initialize the distributed environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'  # Different port from training
    
    # Set device first
    torch.cuda.set_device(rank)
    
    # Initialize process group with timeout
    dist.init_process_group(
        "nccl", 
        rank=rank, 
        world_size=world_size,
        timeout=torch.distributed.constants.default_pg_timeout
    )


def cleanup_ddp():
    """Clean up the distributed environment"""
    dist.destroy_process_group()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def collate_fn(batch):
    """Collate function for batching (single-label cell type)"""
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
        labels[i] = item['labels']  # scalar
        dataset_ids.append(item['dataset_id'])
        cell_ids.append(item['cell_id'])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'labels': labels,
        'dataset_ids': dataset_ids,
        'cell_ids': cell_ids
    }



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


def evaluate_ddp(model, test_loader, rank, world_size, num_cell_types):
    """Evaluate model with DDP for cell type classification"""
    model.eval()
    
    local_logits = []
    local_labels = []
    local_dataset_ids = []
    local_cell_ids = []
    
    if rank == 0:
        pbar = tqdm(test_loader, desc="Evaluating")
    else:
        pbar = test_loader
    
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch['input_ids'].to(rank)
            attention_mask = batch['attention_mask'].to(rank)
            labels = batch['labels'].to(rank)  # shape: (batch,)
            
            logits = model(input_ids, attention_mask)  # shape: (batch, num_classes)
            
            local_logits.append(logits)
            local_labels.append(labels)
            local_dataset_ids.extend(batch['dataset_ids'])
            local_cell_ids.extend(batch['cell_ids'])
    
    # Handle empty process
    if len(local_logits) > 0:
        local_logits_tensor = torch.cat(local_logits, dim=0)  # (N_local, C)
        local_labels_tensor = torch.cat(local_labels, dim=0)  # (N_local,)
    else:
        local_logits_tensor = torch.empty(0, num_cell_types, device=f'cuda:{rank}', dtype=torch.float32)
        local_labels_tensor = torch.empty(0, device=f'cuda:{rank}', dtype=torch.long)
    
    # Gather
    all_logits = gather_from_all_processes(local_logits_tensor, world_size, rank)
    all_labels = gather_from_all_processes(local_labels_tensor.unsqueeze(1), world_size, rank).squeeze(1)  # ensure 1D
    all_dataset_ids = gather_lists_from_all_processes(local_dataset_ids, world_size, rank)
    all_cell_ids = gather_lists_from_all_processes(local_cell_ids, world_size, rank)
    
    return {
        'logits': all_logits,        # (N, C)
        'labels': all_labels,        # (N,)
        'dataset_ids': all_dataset_ids,
        'cell_ids': all_cell_ids
    }


def calculate_metrics(results):
    """Calculate metrics with enhanced analysis of wrong predictions"""
    logits = results['logits']      # (N, C)
    labels = results['labels']      # (N,)
    
    # Get predictions
    pred_classes = np.argmax(logits, axis=1)  # (N,)
    
    metrics = {}
    
    # Standard metrics
    acc = accuracy_score(labels, pred_classes)
    micro_f1 = f1_score(labels, pred_classes, average="micro", zero_division=0)
    macro_f1 = f1_score(labels, pred_classes, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, pred_classes, average="weighted", zero_division=0)
    
    metrics['standard'] = {
        'accuracy': acc,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1
    }
    
   
    # Wrong prediction analysis
    wrong_mask = labels != pred_classes
    num_wrong = np.sum(wrong_mask)
    
    metrics['error_analysis'] = {
        'total_wrong_predictions': int(num_wrong),
        'wrong_prediction_rate': float(num_wrong) / len(labels),
        'confidence_wrong_predictions': np.mean(np.max(logits[wrong_mask], axis=1)) if num_wrong > 0 else 0.0,
        'confidence_correct_predictions': np.mean(np.max(logits[~wrong_mask], axis=1)) if num_wrong < len(labels) else 0.0
    }
    
    # Add to results for saving
    results['predictions'] = pred_classes
    results['probabilities'] = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    results['prediction_confidence'] = np.max(logits, axis=1)
    results['is_correct'] = labels == pred_classes
    
    return metrics

def save_results(results, metrics, save_dir, rank, cell_type_names=None):
    """Save enhanced results with more detailed analysis"""
    if rank != 0:
        return
        
    os.makedirs(save_dir, exist_ok=True)
    
    # Enhanced predictions CSV
    predictions_df = pd.DataFrame({
        'cell_id': results['cell_ids'],
        'dataset_id': results['dataset_ids'],
        'true_label_idx': results['labels'],
        'pred_label_idx': results['predictions'],
        'prediction_confidence': results['prediction_confidence'],
        'is_correct': results['is_correct']
    })
    
    # Add cell type names if available
    if cell_type_names is not None:
        predictions_df['true_cell_type'] = [cell_type_names[i] for i in results['labels']]
        predictions_df['pred_cell_type'] = [cell_type_names[i] for i in results['predictions']]
    
    
    
    predictions_df.to_csv(f"{save_dir}/predictions.csv", index=False)
    # Save wrong predictions analysis
    wrong_predictions = predictions_df[~predictions_df['is_correct']].copy()
    if len(wrong_predictions) > 0:
        # Sort by confidence (most confident wrong predictions first)
        wrong_predictions = wrong_predictions.sort_values('prediction_confidence', ascending=False)
        wrong_predictions.to_csv(f"{save_dir}/wrong_predictions.csv", index=False)
        
        # Summary of most common wrong prediction patterns
        if cell_type_names is not None:
            confusion_summary = wrong_predictions.groupby(['true_cell_type', 'pred_cell_type']).size().reset_index(name='count')
            confusion_summary = confusion_summary.sort_values('count', ascending=False)
            confusion_summary.to_csv(f"{save_dir}/confusion_patterns.csv", index=False)
    
    # Enhanced metrics file
    with open(f"{save_dir}/metrics.txt", "w") as f:
        f.write("ENHANCED CELL TYPE CLASSIFICATION METRICS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("STANDARD METRICS:\n")
        for k, v in metrics['standard'].items():
            f.write(f"  {k.replace('_', ' ').title()}: {v:.4f}\n")
        
        
        if 'error_analysis' in metrics:
            f.write("\nERROR ANALYSIS:\n")
            for k, v in metrics['error_analysis'].items():
                if isinstance(v, float):
                    f.write(f"  {k.replace('_', ' ').title()}: {v:.4f}\n")
                else:
                    f.write(f"  {k.replace('_', ' ').title()}: {v}\n")
    
    # Save detailed JSON metrics
    metrics_json = {}
    for category, values in metrics.items():
        if category != 'similarity_matrix':  # Skip the matrix from JSON
            metrics_json[category] = {}
            for k, v in values.items():
                if isinstance(v, np.ndarray):
                    metrics_json[category][k] = v.tolist()
                elif isinstance(v, (np.int64, np.int32)):
                    metrics_json[category][k] = int(v)
                elif isinstance(v, (np.float64, np.float32)):
                    metrics_json[category][k] = float(v)
                else:
                    metrics_json[category][k] = v
    
    with open(f"{save_dir}/metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)
    
    print(f"\nEnhanced results saved to {save_dir}/:")
    print(f"  - predictions.csv (all predictions)")
    print(f"  - wrong_predictions.csv (only wrong predictions)")
    print(f"  - confusion_patterns.csv (most common error patterns)")
    print(f"  - metrics.txt (human-readable metrics)")
    print(f"  - metrics.json (machine-readable metrics)")

def print_results(metrics, num_samples, rank):
    """Print enhanced results with wrong prediction analysis"""
    if rank != 0:
        return
        
    print("\n" + "="*70)
    print("ENHANCED CELL TYPE CLASSIFICATION RESULTS")
    print("="*70)
    print(f"Total samples: {num_samples}")
    
    print(f"\nSTANDARD METRICS:")
    print("-" * 40)
    for k, v in metrics['standard'].items():
        print(f"  {k.replace('_', ' ').title()}: {v:.4f}")
    

    if 'error_analysis' in metrics:
        print(f"\nERROR ANALYSIS:")
        print("-" * 40)
        err_metrics = metrics['error_analysis']
        print(f"  Wrong Predictions: {err_metrics['total_wrong_predictions']} ({err_metrics['wrong_prediction_rate']:.1%})")
        print(f"  Avg Confidence (Wrong): {err_metrics['confidence_wrong_predictions']:.4f}")
        print(f"  Avg Confidence (Correct): {err_metrics['confidence_correct_predictions']:.4f}")
    
    print("="*70)

def main(rank, world_size, config_path, checkpoint_path, output_dir, label_mapping_json=None):
    """Main evaluation function for each process"""
    try:
        setup_ddp(rank, world_size)
        
        config = load_config(config_path)
        
        # Load label mapping and create target_cell_types list
        cell_type_names = None
        target_cell_types = None
        
        if label_mapping_json:
            if rank == 0:
                print(f"Loading label mapping from {label_mapping_json}")
            
            with open(label_mapping_json, 'r') as f:
                idx_to_label = json.load(f)
                # Convert keys to int and sort by index
                idx_to_label = {int(k): v for k, v in idx_to_label.items()}
                # Create ordered list of cell types
                target_cell_types = [idx_to_label[i] for i in range(len(idx_to_label))]
                cell_type_names = target_cell_types.copy()
                
            if rank == 0:
                print(f"Loaded {len(target_cell_types)} cell type names")
        else:
            # If no label mapping provided, you might need to load from config or elsewhere
            # This is a fallback - you should ideally always provide the label mapping
            if rank == 0:
                print("Warning: No label mapping provided, this might cause issues")
            target_cell_types = config.get('target_cell_types', None)
        
        # Synchronize after loading config/labels
        dist.barrier()
        
        if rank == 0:
            print("Loading test dataset...")
        
        test_dataset = MultiDatasetCellTypeDataset(
            config['data']['base_data_path'], 
            split='test',
            target_cell_types=target_cell_types,  # ← This was missing!
            label_key=config['data'].get('label_key', 'cell_type')
        )
        
        if rank == 0:
            print(f"Test dataset size: {len(test_dataset)}")
            print(f"Number of cell types: {test_dataset.num_cell_types}")
        
        # Create model
        model = GeneformerCellTypeClassifier(
            config['model']['geneformer_model_path'],
            num_cell_types=test_dataset.num_cell_types,
            freeze_geneformer=config['model']['freeze_geneformer']
        )
        
        if rank == 0:
            print("Loading model weights...")
        
        # Load checkpoint with proper error handling
        try:
            checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{rank}', weights_only=False)
            
            # Check if we need to handle DDP wrapper keys
            state_dict = checkpoint['model_state_dict']
            
            # Remove 'module.' prefix if present (from DDP training)
            if any(key.startswith('module.') for key in state_dict.keys()):
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('module.'):
                        new_key = key[7:]  # Remove 'module.' prefix
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                state_dict = new_state_dict
            
            # Load the state dict
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if rank == 0:
                if missing_keys:
                    print(f"Warning: Missing keys in checkpoint: {missing_keys}")
                if unexpected_keys:
                    print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
                    
        except Exception as e:
            if rank == 0:
                print(f"Error loading checkpoint: {e}")
            raise e
        
        # Move model to device and wrap with DDP
        model = model.to(rank)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        
        # Synchronize after model loading
        dist.barrier()
        
        test_sampler = DistributedSampler(
            test_dataset, 
            num_replicas=world_size, 
            rank=rank, 
            shuffle=False,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size'],
            sampler=test_sampler,
            collate_fn=collate_fn,
            num_workers=2,  # Reduced from 4 to avoid potential hanging
            pin_memory=True
        )
        
        # Synchronize before evaluation
        dist.barrier()
        
        if rank == 0:
            print("Starting evaluation...")
            
        results = evaluate_ddp(model, test_loader, rank, world_size, test_dataset.num_cell_types)
        
        if rank == 0:
            print("Calculating metrics...")
            metrics = calculate_metrics(results, cell_type_names)
            
          
            print_results(metrics, len(test_dataset), rank)
            save_results(results, metrics, output_dir, rank, cell_type_names)
            
            print("\nDDP evaluation completed successfully!")
        
        # Final barrier before cleanup
        dist.barrier()
        
    except Exception as e:
        if rank == 0:
            print(f"Error in rank {rank}: {e}")
            import traceback
            traceback.print_exc()
        raise e
    finally:
        try:
            cleanup_ddp()
        except:
            pass  # Ignore cleanup errors

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/home/arism/cell2text/celltype_classification/default.yaml', help='Config file path')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--output-dir', default='test_results_ddp_celltype', help='Output directory for results')
    parser.add_argument('--label-mapping', help='Path to idx_to_label.json for cell type names')
    parser.add_argument('--world-size', type=int, default=torch.cuda.device_count(), 
                        help='Number of GPUs to use')
    args = parser.parse_args()
    
    if args.world_size < 1:
        print("Error: No GPUs available")
        exit(1)
    
    torch.multiprocessing.spawn(
        main,
        args=(args.world_size, args.config, args.checkpoint, args.output_dir,  args.label_mapping),
        nprocs=args.world_size,
        join=True
    )