import torch
import torch.distributed as dist
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
from cell2text_model.model import Cell2TextModel
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from collections import Counter
import json
from .celltype_extractor import calculate_cell_type_metrics, CellTypeExtractor

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cell2text_model.model import Cell2TextModel




def reduce_distributed_metrics(values_list, world_size, rank):
    """Reduce metrics from all DDP processes using all_reduce - returns global average and count"""
    if world_size <= 1:
        return np.mean(values_list) if values_list else 0.0, len(values_list)
    
    if not values_list:
        return 0.0, 0
    
    # Calculate local sum and count
    local_sum = sum(values_list)
    local_count = len(values_list)
    
    # Convert to tensors
    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
    sum_tensor = torch.tensor([local_sum], dtype=torch.float32, device=device)
    count_tensor = torch.tensor([local_count], dtype=torch.float32, device=device)
    
    # All-reduce sum and count
    dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
    
    # Calculate global average
    global_sum = sum_tensor.item()
    global_count = int(count_tensor.item())
    global_avg = global_sum / global_count if global_count > 0 else 0.0
    
    return global_avg, global_count


def collect_cell_type_matches(predicted_types, target_types, world_size, rank):
    """Collect cell type accuracy using all_reduce for counting matches"""
    if world_size <= 1:
        return predicted_types, target_types
    
    # Calculate local matches and total
    local_matches = sum(1 for p, t in zip(predicted_types, target_types) if p == t)
    local_total = len(predicted_types)
    
    # Convert to tensors
    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
    matches_tensor = torch.tensor([local_matches], dtype=torch.long, device=device)
    total_tensor = torch.tensor([local_total], dtype=torch.long, device=device)
    
    # All-reduce to get global counts
    dist.all_reduce(matches_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    
    global_matches = matches_tensor.item()
    global_total = total_tensor.item()
    
    # For compatibility with existing code, we'll return simplified metrics
    # This is a compromise - we lose individual predictions but avoid hanging
    return predicted_types, target_types, global_matches, global_total


def evaluate_cell2text_model(model: Cell2TextModel, 
                           val_loader: DataLoader, 
                           tokenizer: PreTrainedTokenizer, 
                           device: str,
                           print_examples: int = 10,
                           save_results: str = None,
                           use_ddp: bool = False):
    """
    Enhanced evaluation function for DDP training
    """
    
    # Check if we're in a distributed setting
    if use_ddp and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        is_main_process = rank == 0
        print(f"[Rank {rank}] Starting evaluation with world_size={world_size}")
    else:
        world_size = 1
        rank = 0
        is_main_process = True
        print("Starting evaluation without DDP")
    
    # Get the underlying model (unwrap DDP if necessary)
    if hasattr(model, 'module'):
        model = model.module
    
    model.eval()
    bleu_scores = []
    val_losses = []
    smooth = SmoothingFunction().method4
    
    # For cell type evaluation
    cell_extractor = CellTypeExtractor()
    predicted_cell_types = []
    target_cell_types = []
    
    # Store examples for printing/saving
    examples = []
    
    # Create progress bar only on main process
    if is_main_process:
        val_progress_bar = tqdm(val_loader, desc="[Validation]")
    else:
        val_progress_bar = val_loader
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_progress_bar):
            # Move batch to device
            expression_tokens = batch["expression_tokens"].to(device)
            expression_token_lengths = batch["expression_token_lengths"].to(device)
            text_input_ids = batch["input_ids"].to(device)
            text_attention_mask = batch["attention_mask"].to(device)
            
            # Calculate loss if we have target descriptions
            val_loss = None
            if "description_input_ids" in batch and batch["description_input_ids"] is not None:
                description_ids = batch["description_input_ids"].to(device)
                
                # Create combined input and labels for loss calculation
                combined_input_ids = torch.cat([text_input_ids, description_ids], dim=1)
                combined_attention_mask = torch.cat([
                    text_attention_mask, 
                    torch.ones_like(description_ids, dtype=torch.bool)
                ], dim=1)
                
                # Create labels: ignore prompt tokens (-100), use description tokens for loss
                prompt_labels = torch.full_like(text_input_ids, fill_value=-100)
                combined_labels = torch.cat([prompt_labels, description_ids], dim=1)
                
                # Forward pass with labels for loss calculation
                try:
                    outputs = model(
                        expression_tokens=expression_tokens,
                        expression_token_lengths=expression_token_lengths,
                        input_ids=combined_input_ids,
                        attention_mask=combined_attention_mask,
                        labels=combined_labels,
                        return_dict=True
                    )
                    val_loss = outputs.loss.item()
                    val_losses.append(val_loss)
                except Exception as e:
                    if is_main_process:
                        print(f"Warning: Could not calculate loss - {e}")
            
            # Generate descriptions
            generated = model.generate_cell_description(
                expression_tokens=expression_tokens,
                expression_token_lengths=expression_token_lengths,
                inputs=text_input_ids,
                attention_mask=text_attention_mask,
                device=device
            )
            
            # Handle both single string and list of strings return
            if isinstance(generated, str):
                generated = [generated]
            
            # Process each sample in the batch
            for j, gen in enumerate(generated):
                decoded_pred = gen
                
                # Decode target
                target = ""
                if "description_input_ids" in batch and batch["description_input_ids"] is not None:
                    target_ids = batch["description_input_ids"][j]
                    target_ids = target_ids[target_ids != tokenizer.pad_token_id]
                    target = tokenizer.decode(target_ids, skip_special_tokens=True)
                else:
                    if is_main_process:
                        print(f"Warning: No target text available for sample {j}")
                    continue
                
                # Calculate BLEU score
                bleu = sentence_bleu(
                    [target.split()],
                    decoded_pred.split(),
                    smoothing_function=smooth
                )
                bleu_scores.append(bleu)
                
                # Extract cell types
                pred_cell_type = cell_extractor.extract_cell_type(decoded_pred)
                target_cell_type = cell_extractor.extract_cell_type(target)
                
                # Normalize cell types
                pred_cell_type = cell_extractor.normalize_cell_type(pred_cell_type)
                target_cell_type = cell_extractor.normalize_cell_type(target_cell_type)
                
                predicted_cell_types.append(pred_cell_type)
                target_cell_types.append(target_cell_type)
                
                # Store example (only on main process to avoid duplicates)
                if is_main_process:
                    example = {
                        'batch_idx': batch_idx,
                        'sample_idx': j,
                        'generated': decoded_pred,
                        'target': target,
                        'bleu_score': bleu,
                        'predicted_cell_type': pred_cell_type,
                        'target_cell_type': target_cell_type,
                        'cell_type_match': pred_cell_type == target_cell_type,
                        'loss': val_loss
                    }
                    examples.append(example)
    
    # Reduce metrics from all processes if using DDP
    global_matches = None
    global_total = None


    print(f"[Rank {rank if use_ddp else 0}] Finished forward passes, starting metric reduction")

    
    if use_ddp and world_size > 1:
        # Reduce BLEU scores - get global average and total count
        avg_bleu, total_bleu_samples = reduce_distributed_metrics(bleu_scores, world_size, rank)
        
        # Reduce validation losses
        if val_losses:
            avg_loss, total_loss_samples = reduce_distributed_metrics(val_losses, world_size, rank)
        else:
            avg_loss, total_loss_samples = None, 0
        
        # For cell types, use the existing function but get the counts
        if predicted_cell_types and target_cell_types:
            _, _, global_matches, global_total = collect_cell_type_matches(
                predicted_cell_types, target_cell_types, world_size, rank
            )
        else:
            global_matches, global_total = 0, 0
            
        # Print sample counts for debugging
        if is_main_process:
            print(f"Total BLEU samples across all processes: {total_bleu_samples}")
            print(f"Total cell type samples across all processes: {global_total}")
    else:
        # Single process - calculate normally
        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
        avg_loss = np.mean(val_losses) if val_losses else None
        total_bleu_samples = len(bleu_scores)
        global_matches = sum(1 for p, t in zip(predicted_cell_types, target_cell_types) if p == t)
        global_total = len(predicted_cell_types)
    
    # Calculate cell type metrics
    cell_type_metrics = calculate_cell_type_metrics(
        predicted_cell_types, target_cell_types, global_matches, global_total
    )
    
    # Print results only on main process
    if is_main_process:
        print(f"\n{'='*60}")
        print(f"VALIDATION RESULTS")
        print(f"{'='*60}")
        print(f"BLEU Score: {avg_bleu:.4f}")
        print(f"Total samples evaluated: {total_bleu_samples if use_ddp and world_size > 1 else len(bleu_scores)}")
        if avg_loss is not None:
            print(f"Validation Loss: {avg_loss:.4f}")
        else:
            print("Validation Loss: N/A (could not calculate)")
        print(f"\nCell Type Extraction Metrics:")
        print(f"  Accuracy: {cell_type_metrics['accuracy']:.4f}")
        print(f"  F1 Score: {cell_type_metrics['f1']:.4f}")
        print(f"  Precision: {cell_type_metrics['precision']:.4f}")
        print(f"  Recall: {cell_type_metrics['recall']:.4f}")
        print(f"  Total Samples: {cell_type_metrics['total_samples']}")
    
    # Print results only on main process
    if is_main_process:
        # Print overall results
        print(f"\n{'='*60}")
        print(f"VALIDATION RESULTS")
        print(f"{'='*60}")
        print(f"BLEU Score: {avg_bleu:.4f}")
        if avg_loss is not None:
            print(f"Validation Loss: {avg_loss:.4f}")
        else:
            print("Validation Loss: N/A (could not calculate)")
        print(f"\nCell Type Extraction Metrics:")
        print(f"  Accuracy: {cell_type_metrics['accuracy']:.4f}")
        print(f"  F1 Score: {cell_type_metrics['f1']:.4f}")
        print(f"  Precision: {cell_type_metrics['precision']:.4f}")
        print(f"  Recall: {cell_type_metrics['recall']:.4f}")
        print(f"  Total Samples: {cell_type_metrics['total_samples']}")
        
        # Print example predictions
        print(f"\n{'='*60}")
        print(f"EXAMPLE PREDICTIONS (showing first {min(print_examples, len(examples))})")
        print(f"{'='*60}")
        
        for i, example in enumerate(examples[:print_examples]):
            print(f"\n--- Example {i+1} ---")
            print(f"TARGET: {example['target']}")
            print(f"GENERATED: {example['generated']}")
            print(f"BLEU: {example['bleu_score']:.4f}")
            print(f"Target Cell Type: '{example['target_cell_type']}'")
            print(f"Predicted Cell Type: '{example['predicted_cell_type']}'")
            print(f"Cell Type Match: {'✓' if example['cell_type_match'] else '✗'}")
            if example['loss'] is not None:
                print(f"Loss: {example['loss']:.4f}")
        
        # Print cell type distribution analysis
        print(f"\n{'='*60}")
        print(f"CELL TYPE ANALYSIS")
        print(f"{'='*60}")
        
        target_counter = Counter(target_cell_types)
        pred_counter = Counter(predicted_cell_types)
        
        print(f"\nTop 10 Target Cell Types:")
        for cell_type, count in target_counter.most_common(10):
            print(f"  {cell_type}: {count}")
        
        print(f"\nTop 10 Predicted Cell Types:")
        for cell_type, count in pred_counter.most_common(10):
            print(f"  {cell_type}: {count}")
        
        # Save detailed results if requested
        if save_results:
            results = {
                'overall_metrics': {
                    'bleu_score': avg_bleu,
                    'validation_loss': avg_loss,
                    'cell_type_metrics': cell_type_metrics
                },
                'examples': examples,
                'cell_type_distribution': {
                    'target': dict(target_counter),
                    'predicted': dict(pred_counter)
                }
            }
            
            with open(save_results, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nDetailed results saved to: {save_results}")
    
    return {
        'bleu': avg_bleu,
        'validation_loss': avg_loss,
        'cell_type_accuracy': cell_type_metrics['accuracy'],
        'cell_type_f1': cell_type_metrics['f1'],
        'cell_type_precision': cell_type_metrics['precision'],
        'cell_type_recall': cell_type_metrics['recall'],
        'total_samples': total_bleu_samples if use_ddp and world_size > 1 else len(bleu_scores)
    }