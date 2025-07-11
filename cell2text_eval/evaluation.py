import torch
import torch.distributed as dist
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import BERTScorer
from tqdm import tqdm
import evaluate
from transformers import AutoTokenizer  
from cell2text_model.model import Cell2TextModel
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from collections import Counter, defaultdict
import json
from .celltype_extractor import calculate_cell_type_metrics, CellTypeExtractor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cell2text_model.model import Cell2TextModel

def convert_json_compat(obj):
    if isinstance(obj, dict):
        return {k: convert_json_compat(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_json_compat(v) for v in obj]
    elif isinstance(obj, (np.float32, np.float64, np.floating)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.integer)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

def compute_biomedical_bert_score(predictions, references, model_name="dmis-lab/biobert-large-cased-v1.1"):
    """
    Compute BERT score using the biomedical BERT model (biobert-large only)
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        model_name: Name of the biomedical BERT model to use
    Returns:
        dict: Dictionary with precision, recall, and f1 scores
    """
    # Load the tokenizer for the biomedical model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    # Truncate predictions to fit model's max position embeddings (usually 512)
    # Use 495 to leave room for special tokens
    retokenized_predictions = tokenizer(
        predictions, 
        padding="max_length", 
        truncation=True, 
        max_length=495, 
        return_tensors="pt"
    )["input_ids"]
    truncated_predictions = tokenizer.batch_decode(retokenized_predictions, skip_special_tokens=True)
        
    # Truncate references similarly
    retokenized_references = tokenizer(
        references, 
        padding="max_length", 
        truncation=True, 
        max_length=495, 
        return_tensors="pt"
    )["input_ids"]
    truncated_references = tokenizer.batch_decode(retokenized_references, skip_special_tokens=True)
        
    # Load BERTScore evaluator
    bert_scorer = evaluate.load("bertscore")
       
    # Compute BERTScore with the biomedical model
    results = bert_scorer.compute(
        predictions=truncated_predictions,
        references=truncated_references,
        model_type=model_name,
        num_layers=24,
        lang="en",
        verbose=False  # Reduce verbosity to avoid token-related warnings
    )
        
    # Calculate averages
    avg_precision = sum(results["precision"]) / len(results["precision"])
    avg_recall = sum(results["recall"]) / len(results["recall"])
    avg_f1 = sum(results["f1"]) / len(results["f1"])
      
    return {
       "precision": avg_precision,
       "recall": avg_recall,
       "f1": avg_f1,
       "individual_scores": {
           "precision": results["precision"],
           "recall": results["recall"],
            "f1": results["f1"]
        }
    }
        
   

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


def analyze_wrong_predictions(predicted_types, target_types, top_k=20):
    """
    Analyze wrong predictions and return the most common target-predicted pairs
    
    Args:
        predicted_types: List of predicted cell types
        target_types: List of target cell types
        top_k: Number of top wrong prediction pairs to return
        
    Returns:
        dict: Analysis of wrong predictions
    """
    wrong_pairs = []
    correct_predictions = 0
    
    for pred, target in zip(predicted_types, target_types):
        if pred == target:
            correct_predictions += 1
        else:
            wrong_pairs.append((target, pred))
    
    # Count frequency of wrong pairs
    wrong_pair_counts = Counter(wrong_pairs)
    
    # Analyze by target type (what was confused)
    target_confusion = defaultdict(list)
    for (target, pred), count in wrong_pair_counts.items():
        target_confusion[target].append((pred, count))
    
    # Sort confusions for each target type
    for target in target_confusion:
        target_confusion[target].sort(key=lambda x: x[1], reverse=True)
    
    # Analyze by predicted type (what it was confused as)
    pred_confusion = defaultdict(list)
    for (target, pred), count in wrong_pair_counts.items():
        pred_confusion[pred].append((target, count))
    
    # Sort confusions for each predicted type
    for pred in pred_confusion:
        pred_confusion[pred].sort(key=lambda x: x[1], reverse=True)
    
    analysis = {
        'total_samples': len(predicted_types),
        'correct_predictions': correct_predictions,
        'wrong_predictions': len(predicted_types) - correct_predictions,
        'accuracy': correct_predictions / len(predicted_types) if predicted_types else 0,
        'most_common_wrong_pairs': wrong_pair_counts.most_common(top_k),
        'target_confusion': dict(target_confusion),
        'pred_confusion': dict(pred_confusion),
        'unique_wrong_pairs': len(wrong_pair_counts)
    }
    
    return analysis


def create_confusion_matrix(predicted_types, target_types, save_path=None, top_k=20):
    """
    Create and save confusion matrix for cell type predictions
    
    Args:
        predicted_types: List of predicted cell types
        target_types: List of target cell types
        save_path: Path to save the confusion matrix plot
        top_k: Number of most common cell types to include in the matrix
    """
    if not predicted_types or not target_types:
        print("Warning: No predictions or targets available for confusion matrix")
        return None, None
    
    # Get the most common cell types
    all_types = set(predicted_types + target_types)
    type_counts = Counter(target_types)
    
    if len(all_types) > top_k:
        # Use top_k most common types plus "Other" category
        top_types = [ct for ct, _ in type_counts.most_common(top_k)]
        
        # Map less common types to "Other"
        mapped_predicted = []
        mapped_target = []
        
        for pred, target in zip(predicted_types, target_types):
            mapped_pred = pred if pred in top_types else "Other"
            mapped_target = target if target in top_types else "Other"
            mapped_predicted.append(mapped_pred)
            mapped_target.append(mapped_target)
        
        labels = top_types + ["Other"]
    else:
        # Use all types
        labels = sorted(list(all_types))
        mapped_predicted = predicted_types
        mapped_target = target_types
    
    # Create confusion matrix
    cm = confusion_matrix(mapped_target, mapped_predicted, labels=labels)
    
    # Create DataFrame for better visualization
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    
    # Create the plot
    if save_path:
        plt.figure(figsize=(max(12, len(labels) * 0.8), max(10, len(labels) * 0.7)))
        
        # Create heatmap
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True,
                   square=True, linewidths=0.5, 
                   xticklabels=True, yticklabels=True)
        
        plt.title(f'Cell Type Prediction Confusion Matrix\n(Top {top_k} most common types)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Cell Type', fontsize=12)
        plt.ylabel('True Cell Type', fontsize=12)
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to: {save_path}")
    
    # Generate classification report
    report = classification_report(mapped_target, mapped_predicted, 
                                 target_names=labels, output_dict=True)
    
    return cm_df, report


def save_wrong_predictions_report(wrong_pred_analysis, save_path):
    """
    Save detailed wrong predictions analysis to file
    
    Args:
        wrong_pred_analysis: Analysis from analyze_wrong_predictions
        save_path: Path to save the report
    """
    report_lines = []
    
    # Summary
    report_lines.append("WRONG PREDICTIONS ANALYSIS REPORT")
    report_lines.append("=" * 50)
    report_lines.append(f"Total Samples: {wrong_pred_analysis['total_samples']}")
    report_lines.append(f"Correct Predictions: {wrong_pred_analysis['correct_predictions']}")
    report_lines.append(f"Wrong Predictions: {wrong_pred_analysis['wrong_predictions']}")
    report_lines.append(f"Accuracy: {wrong_pred_analysis['accuracy']:.4f}")
    report_lines.append(f"Unique Wrong Pairs: {wrong_pred_analysis['unique_wrong_pairs']}")
    report_lines.append("")
    
    # Most common wrong pairs
    report_lines.append("MOST COMMON WRONG PREDICTION PAIRS")
    report_lines.append("-" * 40)
    report_lines.append("Format: (Target -> Predicted) Count")
    report_lines.append("")
    
    for i, ((target, pred), count) in enumerate(wrong_pred_analysis['most_common_wrong_pairs'], 1):
        report_lines.append(f"{i:2d}. ({target} -> {pred}) {count}")
    
    report_lines.append("")
    
    # Target confusion analysis
    report_lines.append("TARGET CONFUSION ANALYSIS")
    report_lines.append("-" * 30)
    report_lines.append("What each true cell type was confused as:")
    report_lines.append("")
    
    for target, confusions in sorted(wrong_pred_analysis['target_confusion'].items()):
        report_lines.append(f"TRUE: {target}")
        for pred, count in confusions[:5]:  # Show top 5 confusions
            report_lines.append(f"  -> {pred}: {count}")
        report_lines.append("")
    
    # Prediction confusion analysis
    report_lines.append("PREDICTION CONFUSION ANALYSIS")
    report_lines.append("-" * 32)
    report_lines.append("What was confused as each predicted cell type:")
    report_lines.append("")
    
    for pred, confusions in sorted(wrong_pred_analysis['pred_confusion'].items()):
        report_lines.append(f"PREDICTED: {pred}")
        for target, count in confusions[:5]:  # Show top 5 confusions
            report_lines.append(f"  <- {target}: {count}")
        report_lines.append("")
    
    # Write to file
    with open(save_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Wrong predictions report saved to: {save_path}")


def evaluate_cell2text_model(model: Cell2TextModel, 
                           val_loader: DataLoader, 
                           tokenizer: PreTrainedTokenizer, 
                           device: str,
                           print_examples: int = 10,
                           save_results: str = None,
                           use_ddp: bool = False,
                           use_bertscore: bool = True,
                           bertscore_model: str = "dmis-lab/biobert-v1.1",
                           create_confusion_matrix_plot: bool = True,
                           confusion_matrix_path: str = None,
                           wrong_predictions_report_path: str = None):
    """
    Enhanced evaluation function for DDP training with BERTScore support, confusion matrix, and wrong predictions analysis
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
    bert_scores_precision = []
    bert_scores_recall = []
    bert_scores_f1 = []
    val_losses = []
    smooth = SmoothingFunction().method4
    
    # For cell type evaluation
    cell_extractor = CellTypeExtractor()
    predicted_cell_types = []
    target_cell_types = []
    
    # Store examples for printing/saving
    examples = []
    
    # Store predictions and targets for batch BERTScore computation
    batch_predictions = []
    batch_targets = []
    
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
                
                # Store for batch BERTScore computation
                if use_bertscore:
                    batch_predictions.append(decoded_pred)
                    batch_targets.append(target)
                
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
    
    # Compute BERTScore
    if use_bertscore and batch_predictions:
        try:
            if is_main_process:
                print(f"Computing BERTScore for {len(batch_predictions)} samples using {bertscore_model}...")
            
            # Use the new biomedical BERTScore function
            bert_results = compute_biomedical_bert_score(
                batch_predictions, 
                batch_targets, 
                model_name=bertscore_model
            )
            
            if bert_results is not None:
                # Extract individual scores for compatibility
                bert_scores_precision = bert_results['individual_scores']['precision']
                bert_scores_recall = bert_results['individual_scores']['recall']
                bert_scores_f1 = bert_results['individual_scores']['f1']
                
                # Add BERTScore to examples
                if is_main_process:
                    for i, example in enumerate(examples):
                        if i < len(bert_scores_f1):
                            example['bert_precision'] = bert_scores_precision[i]
                            example['bert_recall'] = bert_scores_recall[i]
                            example['bert_f1'] = bert_scores_f1[i]
            else:
                if is_main_process:
                    print("Warning: BERTScore computation failed")
                use_bertscore = False
                    
        except Exception as e:
            if is_main_process:
                print(f"Warning: BERTScore computation failed - {e}")
            use_bertscore = False
    
    # Reduce metrics from all processes if using DDP
    global_matches = None
    global_total = None

    print(f"[Rank {rank if use_ddp else 0}] Finished forward passes, starting metric reduction")
    
    if use_ddp and world_size > 1:
        # Reduce BLEU scores - get global average and total count
        avg_bleu, total_bleu_samples = reduce_distributed_metrics(bleu_scores, world_size, rank)
        
       # Reduce BERTScore metrics if available
        if use_bertscore and bert_scores_f1:
            avg_bert_precision, total_bert_samples = reduce_distributed_metrics(bert_scores_precision, world_size, rank)
            avg_bert_recall, _ = reduce_distributed_metrics(bert_scores_recall, world_size, rank)
            avg_bert_f1, _ = reduce_distributed_metrics(bert_scores_f1, world_size, rank)
        else:
            avg_bert_precision = avg_bert_recall = avg_bert_f1 = None
            total_bert_samples = 0
        
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
            print(f"Total BERTScore samples across all processes: {total_bert_samples}")
            print(f"Total cell type samples across all processes: {global_total}")
    else:
        # Single process - calculate normally
        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
        avg_loss = np.mean(val_losses) if val_losses else None
        total_bleu_samples = len(bleu_scores)
        
        # BERTScore averages
        if use_bertscore and bert_scores_f1:
            avg_bert_precision = np.mean(bert_scores_precision)
            avg_bert_recall = np.mean(bert_scores_recall)
            avg_bert_f1 = np.mean(bert_scores_f1)
            total_bert_samples = len(bert_scores_f1)
        else:
            avg_bert_precision = avg_bert_recall = avg_bert_f1 = None
            total_bert_samples = 0
        
        global_matches = sum(1 for p, t in zip(predicted_cell_types, target_cell_types) if p == t)
        global_total = len(predicted_cell_types)
    
    # Calculate cell type metrics
    cell_type_metrics = calculate_cell_type_metrics(
        predicted_cell_types, target_cell_types, global_matches, global_total
    )
    
    # Analyze wrong predictions (only on main process)
    wrong_pred_analysis = None
    if is_main_process and predicted_cell_types and target_cell_types:
        wrong_pred_analysis = analyze_wrong_predictions(predicted_cell_types, target_cell_types)
    
    # Create confusion matrix (only on main process)
    confusion_matrix_df = None
    classification_report_dict = None
    if is_main_process and predicted_cell_types and target_cell_types and create_confusion_matrix_plot:
        cm_path = confusion_matrix_path or (save_results.replace('.json', '_confusion_matrix.png') if save_results else 'confusion_matrix.png')
        confusion_matrix_df, classification_report_dict = create_confusion_matrix(
            predicted_cell_types, target_cell_types, save_path=cm_path
        )
    
    # Print results only on main process
    if is_main_process:
        print(f"\n{'='*60}")
        print(f"VALIDATION RESULTS")
        print(f"{'='*60}")
        print(f"BLEU Score: {avg_bleu:.4f}")
        print(f"Total samples evaluated: {total_bleu_samples if use_ddp and world_size > 1 else len(bleu_scores)}")
        
        if use_bertscore and avg_bert_f1 is not None:
            print(f"\nBERTScore Results:")
            print(f"  Precision: {avg_bert_precision:.4f}")
            print(f"  Recall: {avg_bert_recall:.4f}")
            print(f"  F1: {avg_bert_f1:.4f}")
            print(f"  Total BERTScore samples: {total_bert_samples}")
        
        if avg_loss is not None:
            print(f"\nValidation Loss: {avg_loss:.4f}")
        else:
            print("\nValidation Loss: N/A (could not calculate)")
        
        print(f"\nCell Type Extraction Metrics:")
        print(f"  Accuracy: {cell_type_metrics['accuracy']:.4f}")
        print(f"  F1 Score: {cell_type_metrics['f1']:.4f}")
        print(f"  Precision: {cell_type_metrics['precision']:.4f}")
        print(f"  Recall: {cell_type_metrics['recall']:.4f}")
        print(f"  Total Samples: {cell_type_metrics['total_samples']}")
        
        # Print wrong predictions analysis
        if wrong_pred_analysis:
            print(f"\n{'='*60}")
            print(f"WRONG PREDICTIONS ANALYSIS")
            print(f"{'='*60}")
            print(f"Total Wrong Predictions: {wrong_pred_analysis['wrong_predictions']}")
            print(f"Unique Wrong Pairs: {wrong_pred_analysis['unique_wrong_pairs']}")
            
            print(f"\nTop 10 Most Common Wrong Prediction Pairs:")
            for i, ((target, pred), count) in enumerate(wrong_pred_analysis['most_common_wrong_pairs'][:10], 1):
                print(f"  {i:2d}. {target} -> {pred} ({count} times)")
            
            # Save detailed wrong predictions report
            if wrong_predictions_report_path or save_results:
                report_path = wrong_predictions_report_path or (save_results.replace('.json', '_wrong_predictions.txt') if save_results else 'wrong_predictions_report.txt')
                save_wrong_predictions_report(wrong_pred_analysis, report_path)
        
        # Print example predictions
        print(f"\n{'='*60}")
        print(f"EXAMPLE PREDICTIONS (showing first {min(print_examples, len(examples))})")
        print(f"{'='*60}")
        
        for i, example in enumerate(examples[:print_examples]):
            print(f"\n--- Example {i+1} ---")
            print(f"TARGET: {example['target']}")
            print(f"GENERATED: {example['generated']}")
            print(f"BLEU: {example['bleu_score']:.4f}")
            
            if use_bertscore and 'bert_f1' in example:
                print(f"BERTScore - P: {example['bert_precision']:.4f}, R: {example['bert_recall']:.4f}, F1: {example['bert_f1']:.4f}")
            
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
                    'bleu_score': convert_json_compat(avg_bleu),
                    'bert_score_precision': convert_json_compat(avg_bert_precision),
                    'bert_score_recall': convert_json_compat(avg_bert_recall),
                    'bert_score_f1': convert_json_compat(avg_bert_f1),
                    'validation_loss': convert_json_compat(avg_loss),
                    'cell_type_metrics': convert_json_compat(cell_type_metrics)
                },
                'examples': convert_json_compat(examples),
                'cell_type_distribution': {
                    'target': convert_json_compat(dict(target_counter)),
                    'predicted': convert_json_compat(dict(pred_counter))
                },
                'wrong_predictions_analysis': convert_json_compat(wrong_pred_analysis),
                'confusion_matrix': convert_json_compat(confusion_matrix_df.to_dict()) if confusion_matrix_df is not None else None,
                'classification_report': convert_json_compat(classification_report_dict)
            }
            
            with open(save_results, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nDetailed results saved to: {save_results}")
    
    return convert_json_compat({
        'bleu': avg_bleu,
        'bert_score_precision': avg_bert_precision,
        'bert_score_recall': avg_bert_recall,
        'bert_score_f1': avg_bert_f1,
        'validation_loss': avg_loss,
        'cell_type_accuracy': cell_type_metrics['accuracy'],
        'cell_type_f1': cell_type_metrics['f1'],
        'cell_type_precision': cell_type_metrics['precision'],
        'cell_type_recall': cell_type_metrics['recall'],
        'total_samples': total_bleu_samples if use_ddp and world_size > 1 else len(bleu_scores),
        'wrong_predictions_analysis': wrong_pred_analysis,
        'confusion_matrix': confusion_matrix_df.to_dict() if confusion_matrix_df is not None else None,
        'classification_report': classification_report_dict
    })