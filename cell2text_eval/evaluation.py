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

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def compute_additional_metrics(predictions, references):
    """Compute BLEU-2, ROUGE-2, METEOR, MMD, and EMD metrics"""
    smooth = SmoothingFunction().method4
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
    
    bleu2_scores = []
    rouge2_scores = []
    meteor_scores = []
    
    for pred, ref in zip(predictions, references):
        # BLEU-2
        bleu2 = sentence_bleu(
            [ref.split()], 
            pred.split(), 
            weights=(0.5, 0.5, 0, 0),
            smoothing_function=smooth
        )
        bleu2_scores.append(bleu2)
        
        # ROUGE-2
        rouge_scores = rouge_scorer_obj.score(ref, pred)
        rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
        
        # METEOR
        try:
            meteor = meteor_score([ref.split()], pred.split())
            meteor_scores.append(meteor)
        except:
            meteor_scores.append(0.0)
    
    # Compute sentence embeddings for MMD and EMD
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    try:
        all_texts = predictions + references
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        all_vectors = vectorizer.fit_transform(all_texts)
        
        pred_vectors = all_vectors[:len(predictions)].toarray()
        ref_vectors = all_vectors[len(predictions):].toarray()
        
        mmd = compute_mmd(pred_vectors, ref_vectors)
        emd = compute_average_emd(pred_vectors, ref_vectors)
        
    except Exception as e:
        print(f"Warning: Could not compute MMD/EMD: {e}")
        mmd = 0.0
        emd = 0.0
    
    return {
        'bleu2': bleu2_scores,
        'rouge2': rouge2_scores,
        'meteor': meteor_scores,
        'mmd': mmd,
        'emd': emd
    }

def compute_mmd(X, Y, kernel='rbf', gamma=1.0):
    """Compute Maximum Mean Discrepancy between two distributions"""
    try:
        if kernel == 'rbf':
            XX = np.exp(-gamma * cdist(X, X, 'sqeuclidean'))
            XY = np.exp(-gamma * cdist(X, Y, 'sqeuclidean'))
            YY = np.exp(-gamma * cdist(Y, Y, 'sqeuclidean'))
        else:
            XX = np.dot(X, X.T)
            XY = np.dot(X, Y.T)
            YY = np.dot(Y, Y.T)
        
        mmd = XX.mean() + YY.mean() - 2 * XY.mean()
        return max(0.0, mmd)
        
    except Exception as e:
        print(f"Warning: MMD computation failed: {e}")
        return 0.0

def compute_average_emd(X, Y):
    """Compute average Earth Mover's Distance across all dimensions"""
    try:
        emds = []
        for i in range(X.shape[1]):
            emd = wasserstein_distance(X[:, i], Y[:, i])
            emds.append(emd)
        return np.mean(emds)
    except Exception as e:
        print(f"Warning: EMD computation failed: {e}")
        return 0.0

def compute_biomedical_bert_score(predictions, references):
    """Compute BERT score using biomedical BERT model"""
    model_name = "dmis-lab/biobert-large-cased-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    # Truncate predictions and references
    retokenized_predictions = tokenizer(
        predictions, 
        padding="max_length", 
        truncation=True, 
        max_length=495, 
        return_tensors="pt"
    )["input_ids"]
    truncated_predictions = tokenizer.batch_decode(retokenized_predictions, skip_special_tokens=True)
        
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
       
    # Compute BERTScore
    results = bert_scorer.compute(
        predictions=truncated_predictions,
        references=truncated_references,
        model_type=model_name,
        num_layers=24,
        lang="en",
        verbose=False
    )
      
    return {
        "precision": results["precision"],
        "recall": results["recall"],
        "f1": results["f1"]
    }

def reduce_all_metrics(local_metrics, world_size, rank):
    """
    Reduce all metrics from all processes at once
    
    Args:
        local_metrics: Dict containing all local metric lists
        world_size: Number of processes  
        rank: Current process rank
        
    Returns:
        dict: Reduced global metrics (only valid on rank 0)
    """
    if world_size <= 1:
        # Single process - just compute averages
        return {
            'bleu': np.mean(local_metrics['bleu']) if local_metrics['bleu'] else 0.0,
            'bleu2': np.mean(local_metrics['bleu2']) if local_metrics['bleu2'] else 0.0,
            'rouge2': np.mean(local_metrics['rouge2']) if local_metrics['rouge2'] else 0.0,
            'meteor': np.mean(local_metrics['meteor']) if local_metrics['meteor'] else 0.0,
            'bert_precision': np.mean(local_metrics['bert_precision']) if local_metrics['bert_precision'] else 0.0,
            'bert_recall': np.mean(local_metrics['bert_recall']) if local_metrics['bert_recall'] else 0.0,
            'bert_f1': np.mean(local_metrics['bert_f1']) if local_metrics['bert_f1'] else 0.0,
            'mmd': local_metrics['mmd'],
            'emd': local_metrics['emd'],
            'ontology_similarity': np.mean(local_metrics['ontology_similarities']) if local_metrics['ontology_similarities'] else 0.0,
            'cell_type_accuracy': local_metrics['cell_type_matches'] / local_metrics['cell_type_total'] if local_metrics['cell_type_total'] > 0 else 0.0,
            'total_samples': local_metrics['cell_type_total']
        }
    
    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
    
    # Prepare data to reduce
    metrics_to_reduce = ['bleu', 'bleu2', 'rouge2', 'meteor', 'bert_precision', 'bert_recall', 'bert_f1', 'ontology_similarities']
    scalar_metrics = ['mmd', 'emd', 'cell_type_matches', 'cell_type_total']
    
    # Calculate local sums and counts for list metrics
    local_sums = []
    local_counts = []
    
    for metric in metrics_to_reduce:
        if local_metrics[metric]:
            local_sums.append(sum(local_metrics[metric]))
            local_counts.append(len(local_metrics[metric]))
        else:
            local_sums.append(0.0)
            local_counts.append(0)
    
    # Add scalar metrics
    for metric in scalar_metrics:
        local_sums.append(local_metrics[metric])
        local_counts.append(1)  # Scalars have count 1
    
    # Convert to tensors
    sums_tensor = torch.tensor(local_sums, dtype=torch.float32, device=device)
    counts_tensor = torch.tensor(local_counts, dtype=torch.long, device=device)
    
    # All-reduce sums and counts
    dist.all_reduce(sums_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)
    
    # Only rank 0 computes final metrics
    if rank == 0:
        global_sums = sums_tensor.cpu().numpy()
        global_counts = counts_tensor.cpu().numpy()
        
        # Calculate averages for list metrics
        global_metrics = {}
        for i, metric in enumerate(metrics_to_reduce):
            if global_counts[i] > 0:
                global_metrics[metric] = global_sums[i] / global_counts[i]
            else:
                global_metrics[metric] = 0.0
        
        # Handle scalar metrics
        scalar_start_idx = len(metrics_to_reduce)
        global_metrics['mmd'] = global_sums[scalar_start_idx]
        global_metrics['emd'] = global_sums[scalar_start_idx + 1]
        
        # Cell type accuracy
        total_matches = global_sums[scalar_start_idx + 2]
        total_samples = global_sums[scalar_start_idx + 3]
        global_metrics['cell_type_accuracy'] = total_matches / total_samples if total_samples > 0 else 0.0
        global_metrics['total_samples'] = int(total_samples)
        
        # Rename ontology_similarities to ontology_similarity for consistency
        global_metrics['ontology_similarity'] = global_metrics.pop('ontology_similarities')
        
        return global_metrics
    
    return None

def convert_json_compat(obj):
    """Convert numpy types to JSON-compatible types"""
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

def evaluate_cell2text_model(model: Cell2TextModel, 
                           val_loader: DataLoader, 
                           tokenizer: PreTrainedTokenizer, 
                           device: str,
                           print_examples: int = 10,
                           save_results: str = None,
                           use_ddp: bool = False,
                           use_bertscore: bool = True,
                           similarity_file_path: str = "/home/arism/datasets/cell_type_similarities.pkl"):
    """
    Simplified evaluation function with proper DDP metric reduction
    """
    
    # Setup DDP
    if use_ddp and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        is_main_process = rank == 0
    else:
        world_size = 1
        rank = 0
        is_main_process = True
        use_ddp = False
    
    if is_main_process:
        print(f"Starting evaluation - World size: {world_size}, Rank: {rank}")
    
    # Get the underlying model
    if hasattr(model, 'module'):
        model = model.module
    
    model.eval()
    
    # Initialize metric collectors
    local_metrics = {
        'bleu': [],
        'bleu2': [],
        'rouge2': [],
        'meteor': [],
        'bert_precision': [],
        'bert_recall': [],
        'bert_f1': [],
        'mmd': 0.0,
        'emd': 0.0,
        'cell_type_matches': 0,
        'cell_type_total': 0,
        'ontology_similarities': []
    }
    
    # Store all predictions and targets for batch processing
    all_predictions = []
    all_targets = []
    all_pred_cell_types = []
    all_target_cell_types = []
    examples = []
    
    smooth = SmoothingFunction().method4
    cell_extractor = CellTypeExtractor(similarity_file_path)
    
    # Progress bar only on main process
    if is_main_process:
        progress_bar = tqdm(val_loader, desc="[Validation]")
    else:
        progress_bar = val_loader
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            expression_tokens = batch["expression_tokens"].to(device)
            expression_token_lengths = batch["expression_token_lengths"].to(device)
            text_input_ids = batch["input_ids"].to(device)
            text_attention_mask = batch["attention_mask"].to(device)
            
            # Generate descriptions
            generated = model.generate_cell_description(
                expression_tokens=expression_tokens,
                expression_token_lengths=expression_token_lengths,
                inputs=text_input_ids,
                attention_mask=text_attention_mask,
                device=device
            )
            
            if isinstance(generated, str):
                generated = [generated]
            
            # Process each sample in the batch
            for j, gen in enumerate(generated):
                decoded_pred = gen
                
                # Decode target
                if "description_input_ids" in batch and batch["description_input_ids"] is not None:
                    target_ids = batch["description_input_ids"][j]
                    target_ids = target_ids[target_ids != tokenizer.pad_token_id]
                    target = tokenizer.decode(target_ids, skip_special_tokens=True)
                else:
                    continue
                
                # Calculate BLEU score
                bleu = sentence_bleu(
                    [target.split()],
                    decoded_pred.split(),
                    smoothing_function=smooth
                )
                local_metrics['bleu'].append(bleu)
                
                # Store for batch processing
                all_predictions.append(decoded_pred)
                all_targets.append(target)
                
                # Extract and compare cell types
                pred_cell_type = cell_extractor.extract_cell_type(decoded_pred)
                target_cell_type = cell_extractor.extract_cell_type(target)
                
                pred_cell_type = cell_extractor.normalize_cell_type(pred_cell_type)
                target_cell_type = cell_extractor.normalize_cell_type(target_cell_type)
                
                all_pred_cell_types.append(pred_cell_type)
                all_target_cell_types.append(target_cell_type)
                
                # Calculate ontology similarity
                ont_similarity = cell_extractor.get_ontology_similarity(pred_cell_type, target_cell_type)
                local_metrics['ontology_similarities'].append(ont_similarity)
                
                # Count cell type matches
                if pred_cell_type == target_cell_type:
                    local_metrics['cell_type_matches'] += 1
                local_metrics['cell_type_total'] += 1
                
                # Store examples (only on main process)
                if is_main_process and len(examples) < print_examples:
                    examples.append({
                        'generated': decoded_pred,
                        'target': target,
                        'bleu_score': bleu,
                        'predicted_cell_type': pred_cell_type,
                        'target_cell_type': target_cell_type,
                        'cell_type_match': pred_cell_type == target_cell_type,
                        'ontology_similarity': ont_similarity
                    })
    
    # Compute additional metrics on collected data
    if all_predictions and all_targets:
        if is_main_process:
            print("Computing additional metrics...")
        
        additional_metrics = compute_additional_metrics(all_predictions, all_targets)
        local_metrics['bleu2'] = additional_metrics['bleu2']
        local_metrics['rouge2'] = additional_metrics['rouge2'] 
        local_metrics['meteor'] = additional_metrics['meteor']
        local_metrics['mmd'] = additional_metrics['mmd']
        local_metrics['emd'] = additional_metrics['emd']
        
        # Compute BERTScore if requested
        if use_bertscore:
            try:
                if is_main_process:
                    print("Computing BERTScore...")
                
                bert_results = compute_biomedical_bert_score(all_predictions, all_targets)
                local_metrics['bert_precision'] = bert_results['precision']
                local_metrics['bert_recall'] = bert_results['recall']
                local_metrics['bert_f1'] = bert_results['f1']
                
            except Exception as e:
                if is_main_process:
                    print(f"Warning: BERTScore computation failed - {e}")
                use_bertscore = False
    
    # Reduce all metrics from all processes
    if is_main_process:
        print("Reducing metrics across all processes...")
    
    global_metrics = reduce_all_metrics(local_metrics, world_size, rank)
    
    # Only main process prints results and saves
    if is_main_process and global_metrics:
        print(f"\n{'='*60}")
        print(f"VALIDATION RESULTS")
        print(f"{'='*60}")
        print(f"Total samples: {global_metrics['total_samples']}")
        print(f"BLEU Score: {global_metrics['bleu']:.4f}")
        print(f"BLEU-2 Score: {global_metrics['bleu2']:.4f}")
        print(f"ROUGE-2 Score: {global_metrics['rouge2']:.4f}")
        print(f"METEOR Score: {global_metrics['meteor']:.4f}")
        print(f"MMD Score (↓): {global_metrics['mmd']:.4f}")
        print(f"EMD Score (↓): {global_metrics['emd']:.4f}")
        
        if use_bertscore:
            print(f"BERTScore Precision: {global_metrics['bert_precision']:.4f}")
            print(f"BERTScore Recall: {global_metrics['bert_recall']:.4f}")
            print(f"BERTScore F1: {global_metrics['bert_f1']:.4f}")
        
        print(f"Cell Type Accuracy: {global_metrics['cell_type_accuracy']:.4f}")
        print(f"Ontology Similarity Score: {global_metrics['ontology_similarity']:.4f}")
        
        # Print examples
        print(f"\n{'='*60}")
        print(f"EXAMPLE PREDICTIONS")
        print(f"{'='*60}")
        
        for i, example in enumerate(examples):
            print(f"\n--- Example {i+1} ---")
            print(f"TARGET: {example['target']}")
            print(f"GENERATED: {example['generated']}")
            print(f"BLEU: {example['bleu_score']:.4f}")
            print(f"Target Cell Type: '{example['target_cell_type']}'")
            print(f"Predicted Cell Type: '{example['predicted_cell_type']}'")
            print(f"Cell Type Match: {'✓' if example['cell_type_match'] else '✗'}")
            print(f"Ontology Similarity: {example['ontology_similarity']:.4f}")
        
        # Save results if requested
        if save_results:
            results = {
                'global_metrics': convert_json_compat(global_metrics),
                'examples': convert_json_compat(examples),
                'cell_type_distribution': {
                    'target': convert_json_compat(dict(Counter(all_target_cell_types))),
                    'predicted': convert_json_compat(dict(Counter(all_pred_cell_types)))
                }
            }
            
            with open(save_results, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {save_results}")
        
        # Return in the format expected by your run_evaluation.py script
        return convert_json_compat({
            'bleu': global_metrics['bleu'],
            'bleu2': global_metrics['bleu2'],
            'rouge2': global_metrics['rouge2'],
            'meteor': global_metrics['meteor'],
            'mmd': global_metrics['mmd'],
            'emd': global_metrics['emd'],
            'bert_score_precision': global_metrics['bert_precision'],
            'bert_score_recall': global_metrics['bert_recall'],
            'bert_score_f1': global_metrics['bert_f1'],
            'cell_type_accuracy': global_metrics['cell_type_accuracy'],
            'cell_type_f1': global_metrics.get('cell_type_f1', 0.0),
            'cell_type_precision': global_metrics.get('cell_type_precision', 0.0),
            'cell_type_recall': global_metrics.get('cell_type_recall', 0.0),
            'ontology_similarity_score': global_metrics['ontology_similarity'],
            'total_samples': global_metrics['total_samples'],
            'validation_loss': None,
            'wrong_predictions_analysis': None,
            'confusion_matrix': None,
            'classification_report': None
        })
    
    return None