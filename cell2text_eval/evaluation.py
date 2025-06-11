import torch
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

def calculate_cell_type_metrics(predicted_types, target_types):
    """Calculate precision, recall, and F1 for cell type extraction"""
    
    # Get unique cell types
    all_types = list(set(predicted_types + target_types))
    
    # Convert to binary classification for each cell type
    y_true = []
    y_pred = []
    
    for target, pred in zip(target_types, predicted_types):
        y_true.append(target)
        y_pred.append(pred)
    
    # Calculate exact match accuracy
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

def evaluate_cell2text_model(model: Cell2TextModel, 
                           val_loader: DataLoader, 
                           tokenizer: PreTrainedTokenizer, 
                           device: str,
                           print_examples: int = 10,
                           save_results: str = None):
    """
    Enhanced evaluation function with text printing and cell type accuracy
    
    Args:
        model: The Cell2TextModel to evaluate
        val_loader: DataLoader for validation data
        tokenizer: Tokenizer for decoding text
        device: Device to run evaluation on
        print_examples: Number of example predictions to print (default: 10)
        save_results: Optional path to save detailed results as JSON
    """
    
    model.eval()
    val_loss = 0
    bleu_scores = []
    smooth = SmoothingFunction().method4
    
    # For cell type evaluation
    cell_extractor = CellTypeExtractor()
    predicted_cell_types = []
    target_cell_types = []
    
    # Store examples for printing/saving
    examples = []
    
    val_progress_bar = tqdm(val_loader, desc="[Validation]")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_progress_bar):
            # Move batch to device
            expression_tokens = batch["expression_tokens"].to(device)
            expression_token_lengths = batch["expression_token_lengths"].to(device)
            text_input_ids = batch["input_ids"].to(device)
            text_attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device) if batch["labels"] is not None else None
            
            # Forward pass to get loss
            outputs = model(
                expression_tokens=expression_tokens,
                expression_token_lengths=expression_token_lengths,
                input_ids=text_input_ids,
                attention_mask=text_attention_mask,
                labels=labels,
                return_dict=True
            )
            
            if outputs.loss is not None:
                loss = outputs.loss
                val_loss += loss.item()
            else:
                loss = torch.tensor(0.0)
            
            # Generate descriptions
            generated = model.generate_cell_description(
                expression_tokens=expression_tokens,
                expression_token_lengths=expression_token_lengths,
                prompt_template=None,
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
                if "decoder_input_ids" in batch:
                    target = tokenizer.decode(batch["decoder_input_ids"][j], skip_special_tokens=True)
                elif "labels" in batch and batch["labels"] is not None:
                    target_ids = batch["labels"][j]
                    target_ids = target_ids[target_ids != -100]
                    target = tokenizer.decode(target_ids, skip_special_tokens=True)
                else:
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
                
                # Normalize cell types for better matching
                pred_cell_type = cell_extractor.normalize_cell_type(pred_cell_type)
                target_cell_type = cell_extractor.normalize_cell_type(target_cell_type)
                
                predicted_cell_types.append(pred_cell_type)
                target_cell_types.append(target_cell_type)
                
                # Store example for printing/saving
                example = {
                    'batch_idx': batch_idx,
                    'sample_idx': j,
                    'generated': decoded_pred,
                    'target': target,
                    'bleu_score': bleu,
                    'predicted_cell_type': pred_cell_type,
                    'target_cell_type': target_cell_type,
                    'cell_type_match': pred_cell_type == target_cell_type
                }
                examples.append(example)
            
            val_progress_bar.set_postfix({"loss": loss.item()})
    
    # Calculate overall metrics
    avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
    
    # Calculate cell type metrics
    cell_type_metrics = calculate_cell_type_metrics(predicted_cell_types, target_cell_types)
    
    # Print overall results
    print(f"\n{'='*60}")
    print(f"VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Loss: {avg_val_loss:.4f}")
    print(f"BLEU Score: {avg_bleu:.4f}")
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
                'loss': avg_val_loss,
                'bleu_score': avg_bleu,
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
        'loss': avg_val_loss,
        'bleu': avg_bleu,
        'cell_type_accuracy': cell_type_metrics['accuracy'],
        'cell_type_f1': cell_type_metrics['f1'],
        'cell_type_precision': cell_type_metrics['precision'],
        'cell_type_recall': cell_type_metrics['recall']
    }