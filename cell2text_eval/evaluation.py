import torch
import torch.distributed as dist
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
import evaluate
from transformers import AutoTokenizer
from collections import Counter
import json
import os
from datetime import datetime
from .celltype_extractor import CellTypeExtractor, calculate_comprehensive_metrics

from rouge_score import rouge_scorer
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
import nltk

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



# Download required NLTK data
for resource in ['tokenizers/punkt', 'corpora/wordnet']:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split('/')[-1])


def create_safe_cell_extractor(similarity_file_path=None, cell_type_csv_path=None,
                              disease_csv_path=None, tissue_csv_path=None,
                              pathway_descriptions_path=None):
    """Create CellTypeExtractor with safe file handling"""
    default_paths = {
        'similarity_file_path': "/home/arism/datasets/cell_type_similarities.pkl",
        'cell_type_csv_path': "/home/arism/analysis_output/final_combined/final_combined_cell_type_top_values.csv",
        'disease_csv_path': "/home/arism/analysis_output/final_combined/final_combined_disease_top_values.csv",
        'tissue_csv_path': "/home/arism/analysis_output/final_combined/final_combined_tissue_top_values.csv",
        'pathway_descriptions_path': "pathway_descriptions.json"
    }
    
    safe_paths = {}
    for key, default_path in default_paths.items():
        provided_path = locals().get(key)
        path = provided_path or default_path
        safe_paths[key] = path if path and os.path.exists(path) else None
        if safe_paths[key] is None:
            print(f"Warning: {key} not found, using None")
    
    try:
        return CellTypeExtractor(**safe_paths)
    except Exception as e:
        print(f"Warning: CellTypeExtractor creation failed: {e}")
        return CellTypeExtractor(**{k: None for k in safe_paths})


def compute_text_metrics(predictions, references):
    """Compute BLEU-2, BLEU-4, ROUGE-1, ROUGE-2, ROUGE-L, """
    smooth = SmoothingFunction().method4
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    metrics = {'bleu': [], 'bleu2': [], 'bleu4': [], 
               'rouge1': [], 'rouge2': [], 'rougeL': [], }
    
    for pred, ref in zip(predictions, references):
        # BLEU-1 (default)
        metrics['bleu'].append(sentence_bleu(
            [ref.split()], pred.split(), smoothing_function=smooth
        ))
        # BLEU-2
        metrics['bleu2'].append(sentence_bleu(
            [ref.split()], pred.split(), weights=(0.5, 0.5, 0, 0), smoothing_function=smooth
        ))
        # BLEU-4
        metrics['bleu4'].append(sentence_bleu(
            [ref.split()], pred.split(), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth
        ))
        
        # ROUGE scores
        rouge_scores = rouge_scorer_obj.score(ref, pred)
        metrics['rouge1'].append(rouge_scores['rouge1'].fmeasure)
        metrics['rouge2'].append(rouge_scores['rouge2'].fmeasure)
        metrics['rougeL'].append(rouge_scores['rougeL'].fmeasure)
        
        
    
    return metrics

def compute_roberta_bertscore(predictions, references):
    """Compute BERTScore using RoBERTa"""
    bert_scorer = evaluate.load("bertscore")
    results = bert_scorer.compute(
        predictions=predictions,
        references=references,
        model_type="roberta-large",
        lang="en",
        verbose=False
    )
    
    return {
        'precision': results["precision"],
        'recall': results["recall"],
        'f1': results["f1"]
    }






def compute_bert_score(predictions, references):
    """Compute BERTScore using biomedical BERT"""
    model_name = "dmis-lab/biobert-large-cased-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Truncate
    truncate = lambda texts: tokenizer.batch_decode(
        tokenizer(texts, padding="max_length", truncation=True, 
                 max_length=495, return_tensors="pt")["input_ids"],
        skip_special_tokens=True
    )
    
    bert_scorer = evaluate.load("bertscore")
    results = bert_scorer.compute(
        predictions=truncate(predictions),
        references=truncate(references),
        model_type=model_name,
        num_layers=24,
        lang="en",
        verbose=False
    )
    
    return {
        'precision': results["precision"],
        'recall': results["recall"],
        'f1': results["f1"]
    }


def gather_predictions_ddp(local_data, world_size, rank):
    """Gather predictions from all processes to rank 0"""
    if world_size <= 1:
        return local_data
    
    # Serialize to JSON
    local_json = json.dumps(local_data)
    local_size = torch.tensor([len(local_json)], dtype=torch.long).cuda()
    
    # Gather sizes
    size_list = [torch.zeros(1, dtype=torch.long).cuda() for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    
    # Gather data
    max_size = max(size_list).item()
    local_json_padded = local_json + ' ' * (max_size - len(local_json))
    local_tensor = torch.ByteTensor(list(local_json_padded.encode('utf-8'))).cuda()
    
    gathered = [torch.zeros(max_size, dtype=torch.uint8).cuda() 
                for _ in range(world_size)] if rank == 0 else None
    
    dist.gather(local_tensor, gathered, dst=0)
    
    if rank == 0:
        all_data = {'predictions': [], 'targets': [], 
                   'pred_cells': [], 'target_cells': []}
        for i, tensor in enumerate(gathered):
            data_str = tensor.cpu().numpy().tobytes().decode('utf-8')[:size_list[i].item()]
            data = json.loads(data_str)
            for key in all_data:
                all_data[key].extend(data[key])
        return all_data
    
    return None




def save_results_json(predictions, targets, pred_cells, target_cells, 
                     metrics, examples, save_path):
    """Save comprehensive results to JSON"""
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(predictions)
        },
        'metrics': convert_json_compat(metrics),
        'examples': convert_json_compat(examples),
        'predictions': {
            'text': predictions,
            'cell_types': pred_cells
        },
        'targets': {
            'text': targets,
            'cell_types': target_cells
        },
        'distributions': {
            'target_cells': convert_json_compat(dict(Counter(target_cells))),
            'pred_cells': convert_json_compat(dict(Counter(pred_cells)))
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {save_path}")


def reduce_metrics_ddp(local_metrics, world_size, rank):
    """Reduce metrics from all processes"""
    if world_size <= 1:
        return {k: np.mean(v) if isinstance(v, list) else v 
                for k, v in local_metrics.items()}
    
    # Prepare tensors - ALL metrics that are lists in local_metrics
    list_metrics = [
        'bleu', 'bleu2', 'bleu4', 
        'rouge1', 'rouge2', 'rougeL',
        'biobert_precision', 'biobert_recall', 'biobert_f1',
        'roberta_precision', 'roberta_recall', 'roberta_f1',
        'ontology_sim'
    ]
    scalar_metrics = ['cell_matches', 'cell_total']
    
    sums = []
    counts = []
    
    for key in list_metrics:
        vals = local_metrics.get(key, [])
        sums.append(sum(vals) if vals else 0.0)
        counts.append(len(vals) if vals else 0)
    
    for key in scalar_metrics:
        sums.append(local_metrics.get(key, 0.0))
        counts.append(1)
    
    sums_t = torch.tensor(sums, dtype=torch.float32).cuda()
    counts_t = torch.tensor(counts, dtype=torch.long).cuda()
    
    dist.all_reduce(sums_t, op=dist.ReduceOp.SUM)
    dist.all_reduce(counts_t, op=dist.ReduceOp.SUM)
    
    if rank == 0:
        sums_np = sums_t.cpu().numpy()
        counts_np = counts_t.cpu().numpy()
        
        result = {}
        for i, key in enumerate(list_metrics):
            result[key] = sums_np[i] / counts_np[i] if counts_np[i] > 0 else 0.0
        
        idx = len(list_metrics)
        result['cell_accuracy'] = (sums_np[idx] / sums_np[idx + 1] 
                                   if sums_np[idx + 1] > 0 else 0.0)
        result['total_samples'] = int(sums_np[idx + 1])
        
        return result
    
    return None


def evaluate_cell2text_model(model, val_loader, tokenizer, device,
                            print_examples=10, save_results=None,
                            save_detailed_json=None, use_ddp=False,
                            use_bertscore=True, use_comprehensive_metrics=False,
                            similarity_file_path=None, cell_type_csv_path=None,
                            disease_csv_path=None, tissue_csv_path=None,
                            pathway_descriptions_path=None):
    
    # Setup DDP
    if use_ddp and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = 1
        rank = 0
    
    is_main = rank == 0
    
    if is_main:
        print(f"Starting evaluation (World size: {world_size})")
    
    # Get model
    model = model.module if hasattr(model, 'module') else model
    model.eval()
    
    local_metrics = {
        'bleu': [], 'bleu2': [], 'bleu4': [],
        'rouge1': [], 'rouge2': [], 'rougeL': [], 
        'biobert_precision': [], 'biobert_recall': [], 'biobert_f1': [],
        'roberta_precision': [], 'roberta_recall': [], 'roberta_f1': [],
        'ontology_sim': [], 
        'cell_matches': 0, 'cell_total': 0
    }
    
    local_predictions = []
    local_targets = []
    local_pred_cells = []
    local_target_cells = []
    examples = []
    
    cell_extractor = create_safe_cell_extractor(
        similarity_file_path, cell_type_csv_path, disease_csv_path,
        tissue_csv_path, pathway_descriptions_path
    )
    
    smooth = SmoothingFunction().method4
    progress = tqdm(val_loader, desc="[Validation]") if is_main else val_loader
    
    # Generate predictions
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress):
            expr_tokens = batch["expression_tokens"].to(device)
            expr_lengths = batch["expression_token_lengths"].to(device)
            text_ids = batch["input_ids"].to(device)
            text_mask = batch["attention_mask"].to(device)
            
            try:
                generated = model.generate_cell_description(
                    expression_tokens=expr_tokens,
                    expression_token_lengths=expr_lengths,
                    inputs=text_ids,
                    attention_mask=text_mask,
                    device=device
                )
                if isinstance(generated, str):
                    generated = [generated]
            except Exception as e:
                if is_main:
                    print(f"Generation failed for batch {batch_idx}: {e}")
                continue
            
            # Process each sample
            for j, pred in enumerate(generated):
                if "description_input_ids" not in batch:
                    continue
                
                target_ids = batch["description_input_ids"][j]
                target_ids = target_ids[target_ids != tokenizer.pad_token_id]
                target = tokenizer.decode(target_ids, skip_special_tokens=True)
                
                # BLEU
                try:
                    bleu = sentence_bleu([target.split()], pred.split(), 
                                        smoothing_function=smooth)
                    local_metrics['bleu'].append(bleu)
                except:
                    local_metrics['bleu'].append(0.0)
                
                local_predictions.append(pred)
                local_targets.append(target)
                
                # Cell type extraction
                try:
                    pred_cell = cell_extractor.normalize_cell_type(
                        cell_extractor.extract_cell_type(pred)
                    )
                    target_cell = cell_extractor.normalize_cell_type(
                        cell_extractor.extract_cell_type(target)
                    )
                    
                    local_pred_cells.append(pred_cell)
                    local_target_cells.append(target_cell)
                    
                    ont_sim = cell_extractor.get_ontology_similarity(pred_cell, target_cell)
                    local_metrics['ontology_sim'].append(ont_sim)
                    
                    if pred_cell == target_cell:
                        local_metrics['cell_matches'] += 1
                    local_metrics['cell_total'] += 1
                    
                except Exception as e:
                    local_pred_cells.append("unknown")
                    local_target_cells.append("unknown")
                    local_metrics['ontology_sim'].append(0.0)
                    local_metrics['cell_total'] += 1
                
                # Store examples
                if is_main and len(examples) < print_examples:
                    examples.append({
                        'generated': pred,
                        'target': target,
                        'bleu': local_metrics['bleu'][-1],
                        'pred_cell': local_pred_cells[-1],
                        'target_cell': local_target_cells[-1],
                        'match': local_pred_cells[-1] == local_target_cells[-1],
                        'ont_sim': local_metrics['ontology_sim'][-1]
                    })
    
    # Compute additional local metrics
    if local_predictions:
        if is_main:
            print("Computing text metrics...")
        
        text_metrics = compute_text_metrics(local_predictions, local_targets)
        local_metrics['bleu2'] = text_metrics['bleu2']
        local_metrics['bleu4'] = text_metrics['bleu4']
        local_metrics['rouge1'] = text_metrics['rouge1']
        local_metrics['rouge2'] = text_metrics['rouge2']
        local_metrics['rougeL'] = text_metrics['rougeL']

        if use_bertscore:
            try:
                if is_main:
                    print("Computing BioBERT BERTScore...")
                biobert_results = compute_bert_score(local_predictions, local_targets)
                local_metrics['biobert_precision'] = biobert_results['precision']
                local_metrics['biobert_recall'] = biobert_results['recall']
                local_metrics['biobert_f1'] = biobert_results['f1']
                
                if is_main:
                    print("Computing RoBERTa BERTScore...")
                roberta_results = compute_roberta_bertscore(local_predictions, local_targets)
                local_metrics['roberta_precision'] = roberta_results['precision']
                local_metrics['roberta_recall'] = roberta_results['recall']
                local_metrics['roberta_f1'] = roberta_results['f1']
            except Exception as e:
                if is_main:
                    print(f"BERTScore failed: {e}")
                # Ensure these keys exist even if BERTScore fails
                for key in ['biobert_precision', 'biobert_recall', 'biobert_f1',
                           'roberta_precision', 'roberta_recall', 'roberta_f1']:
                    if key not in local_metrics or not local_metrics[key]:
                        local_metrics[key] = [0.0] * len(local_predictions)
    
    # Reduce metrics across processes
    if is_main:
        print("Reducing metrics...")
    global_metrics = reduce_metrics_ddp(local_metrics, world_size, rank)
    
    # Gather all predictions to rank 0
    if is_main:
        print("Gathering predictions...")
    
    gathered = gather_predictions_ddp({
        'predictions': local_predictions,
        'targets': local_targets,
        'pred_cells': local_pred_cells,
        'target_cells': local_target_cells
    }, world_size, rank)
    
    # Compute comprehensive metrics on rank 0
    if is_main and gathered and use_comprehensive_metrics:
        try:
            print("Computing comprehensive metrics...")
            comp_metrics = calculate_comprehensive_metrics(
                gathered['predictions'], gathered['targets'],
                similarity_file_path, cell_type_csv_path, disease_csv_path,
                tissue_csv_path, pathway_descriptions_path
            )
            global_metrics.update(comp_metrics)
        except Exception as e:
            print(f"Comprehensive metrics failed: {e}")
    
    # Print and save results (rank 0 only)
    if is_main and global_metrics:
        print(f"\n{'='*60}")
        print("VALIDATION RESULTS")
        print(f"{'='*60}")
        print(f"Total: {global_metrics.get('total_samples', 0)}")
        print(f"BLEU: {global_metrics.get('bleu', 0.0):.4f}")
        print(f"BLEU-2: {global_metrics.get('bleu2', 0.0):.4f}")
        print(f"BLEU-4: {global_metrics.get('bleu4', 0.0):.4f}")
        print(f"ROUGE-1: {global_metrics.get('rouge1', 0.0):.4f}")
        print(f"ROUGE-2: {global_metrics.get('rouge2', 0.0):.4f}")
        print(f"ROUGE-L: {global_metrics.get('rougeL', 0.0):.4f}")
        
        if use_bertscore:
            print(f"\nBioBERT BERTScore:")
            print(f"  Precision: {global_metrics.get('biobert_precision', 0.0):.4f}")
            print(f"  Recall: {global_metrics.get('biobert_recall', 0.0):.4f}")
            print(f"  F1: {global_metrics.get('biobert_f1', 0.0):.4f}")
            
            print(f"\nRoBERTa BERTScore:")
            print(f"  Precision: {global_metrics.get('roberta_precision', 0.0):.4f}")
            print(f"  Recall: {global_metrics.get('roberta_recall', 0.0):.4f}")
            print(f"  F1: {global_metrics.get('roberta_f1', 0.0):.4f}")
        
        print(f"\nCell Type Metrics:")
        print(f"  Accuracy: {global_metrics.get('cell_accuracy', 0.0):.4f}")
        print(f"  Ontology Similarity: {global_metrics.get('ontology_sim', 0.0):.4f}")
        
        print(f"\n{'='*60}")
        print("EXAMPLES")
        print(f"{'='*60}")
        for i, ex in enumerate(examples):
            print(f"\n[{i+1}]")
            print(f"Target: {ex['target']}")
            print(f"Generated: {ex['generated']}")
            print(f"BLEU: {ex['bleu']:.4f} | Match: {ex['match']} | Ont Sim: {ex.get('ont_sim', 0.0):.4f}")
        
        # Save results
        if gathered:
            if save_detailed_json:
                save_results_json(
                    gathered['predictions'], gathered['targets'],
                    gathered['pred_cells'], gathered['target_cells'],
                    global_metrics, examples, save_detailed_json
                )
            
            if save_results:
                with open(save_results, 'w') as f:
                    json.dump({
                        'metrics': convert_json_compat(global_metrics),
                        'examples': convert_json_compat(examples),
                        'distributions': {
                            'target': convert_json_compat(dict(Counter(gathered['target_cells']))),
                            'predicted': convert_json_compat(dict(Counter(gathered['pred_cells'])))
                        }
                    }, f, indent=2)
                print(f"Standard results saved to: {save_results}")
        
        # Return formatted results with safe .get() access
        return {
            'bleu': global_metrics.get('bleu', 0.0),
            'bleu2': global_metrics.get('bleu2', 0.0),
            'bleu4': global_metrics.get('bleu4', 0.0),
            'rouge1': global_metrics.get('rouge1', 0.0),
            'rouge2': global_metrics.get('rouge2', 0.0),
            'rougeL': global_metrics.get('rougeL', 0.0),            
            'biobert_precision': global_metrics.get('biobert_precision', 0.0),
            'biobert_recall': global_metrics.get('biobert_recall', 0.0),
            'biobert_f1': global_metrics.get('biobert_f1', 0.0),
            'roberta_precision': global_metrics.get('roberta_precision', 0.0),
            'roberta_recall': global_metrics.get('roberta_recall', 0.0),
            'roberta_f1': global_metrics.get('roberta_f1', 0.0),
            'cell_type_accuracy': global_metrics.get('cell_accuracy', 0.0),
            'ontology_similarity_score': global_metrics.get('ontology_sim', 0.0),
            'total_samples': global_metrics.get('total_samples', 0),
            **{k: v for k, v in global_metrics.items() 
               if k.startswith(('disease_', 'tissue_', 'pathway_', 'cell_type_precision', 'cell_type_recall', 'cell_type_f1'))}
        }
    return None