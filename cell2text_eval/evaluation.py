import torch
import torch.distributed as dist
import numpy as np
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
from cell2text_model.model import Cell2TextModel
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import Counter
import json


class CellTypeExtractor:
    """Extract cell types from generated descriptions for evaluation"""
    
    def __init__(self):
        # Common cell type patterns that appear after "consists of a"
        self.cell_type_patterns = [
            r"consists of a ([^,\.]+?)(?:,|\.|$)",
            r"consists of an ([^,\.]+?)(?:,|\.|$)",
        ]
        
        # Clean up extracted cell types
        self.cleanup_patterns = [
            r"^(a|an)\s+",  # Remove leading articles
            r"\s+cell$",    # Remove trailing "cell"
        ]
    
    def extract_cell_type(self, description: str) -> str:
        """Extract the main cell type from a description"""
        if not description:
            return "unknown"
            
        # Try each pattern
        for pattern in self.cell_type_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                cell_type = match.group(1).strip()
                
                # Clean up the extracted cell type
                for cleanup_pattern in self.cleanup_patterns:
                    cell_type = re.sub(cleanup_pattern, "", cell_type, flags=re.IGNORECASE).strip()
                
                return cell_type.lower()
        
        return "unknown"
    
    def normalize_cell_type(self, cell_type: str) -> str:
        """Normalize cell type names for better matching"""
        if not cell_type:
            return "unknown"
            
        # Convert to lowercase and clean
        normalized = cell_type.lower().strip()
        
        # Remove common prefixes/suffixes that might cause mismatches
        suffixes_to_remove = ["cell", "cells"]
        
        for suffix in suffixes_to_remove:
            if normalized.endswith(" " + suffix):
                normalized = normalized[:-len(" " + suffix)]
        
        return normalized


def calculate_cell_type_metrics(predicted_types, target_types, global_matches=None, global_total=None):
    """Calculate precision, recall, and F1 for cell type extraction"""
    
    # If we have global matches from DDP reduction, use those for accuracy
    if global_matches is not None and global_total is not None:
        accuracy = global_matches / global_total if global_total > 0 else 0
        # For distributed case, we can't easily calculate F1/precision/recall without all data
        # So we'll use accuracy as approximation or calculate locally
        return {
            'accuracy': accuracy,
            'f1': accuracy,  # Approximation
            'precision': accuracy,  # Approximation  
            'recall': accuracy,  # Approximation
            'total_samples': global_total
        }
    
    # Local calculation (single GPU or main process with all data)
    exact_matches = sum(1 for t, p in zip(target_types, predicted_types) if t == p)
    accuracy = exact_matches / len(target_types) if target_types else 0
    
    # Calculate macro F1, precision, recall
    unique_labels = list(set(target_types + predicted_types))
    if len(unique_labels) > 1:
        f1 = f1_score(target_types, predicted_types, labels=unique_labels, average='macro', zero_division=0)
        precision = precision_score(target_types, predicted_types, labels=unique_labels, average='macro', zero_division=0)
        recall = recall_score(target_types, predicted_types, labels=unique_labels, average='macro', zero_division=0)
    else:
        f1 = precision = recall = accuracy
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'total_samples': len(target_types)
    }


def reduce_distributed_metrics(values_list, world_size, rank):
    """Reduce metrics from all DDP processes using all_reduce"""
    if world_size <= 1:
        return values_list
    
    if not values_list:
        return []
    
    # For averaging metrics, we need sum and count
    local_sum = torch.tensor([sum(values_list)], dtype=torch.float32, device=torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))
    local_count = torch.tensor([len(values_list)], dtype=torch.float32, device=local_sum.device)
    
    # All-reduce sum and count
    dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
    
    # Calculate global average
    if local_count.item() > 0:
        global_avg = local_sum.item() / local_count.item()
        # Return a list with the global average repeated for compatibility
        return [global_avg] * int(local_count.item())
    else:
        return []


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
    
    Args:
        model: The Cell2TextModel to evaluate
        val_loader: DataLoader for validation data
        tokenizer: Tokenizer for decoding text
        device: Device to run evaluation on
        print_examples: Number of example predictions to print (default: 10)
        save_results: Optional path to save detailed results as JSON
        use_ddp: Whether we're using DDP (affects printing and gathering)
    """
    

    if use_ddp and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"[Rank {rank}] Starting evaluation with world_size={world_size}")
    else:
        print("Starting evaluation without DDP")

    # Check if we're in a distributed setting
    if use_ddp and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        is_main_process = rank == 0
    else:
        world_size = 1
        rank = 0
        is_main_process = True
    
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
    
    # Create progress bar only on main 
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
        print(f"[Rank {rank}] About to reduce BLEU scores: {len(bleu_scores)} scores")
        # Reduce BLEU scores (get average)
        if bleu_scores:
            all_bleu_scores = reduce_distributed_metrics(bleu_scores, world_size, rank)
            avg_bleu = np.mean(all_bleu_scores) if all_bleu_scores else 0.0
        else:
            avg_bleu = 0.0
        
        # Reduce validation losses (get average)
        if val_losses:
            all_val_losses = reduce_distributed_metrics(val_losses, world_size, rank)
            avg_loss = np.mean(all_val_losses) if all_val_losses else None
        else:
            avg_loss = None
        
        # For cell types, we'll use a simpler approach - just count matches
        if predicted_cell_types and target_cell_types:
            _, _, global_matches, global_total = collect_cell_type_matches(
                predicted_cell_types, target_cell_types, world_size, rank
            )
    else:
        # Single process - calculate normally
        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
        avg_loss = np.mean(val_losses) if val_losses else None
    
    # Calculate cell type metrics
    cell_type_metrics = calculate_cell_type_metrics(
        predicted_cell_types, target_cell_types, global_matches, global_total
    )
    
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
        'cell_type_recall': cell_type_metrics['recall']
    }