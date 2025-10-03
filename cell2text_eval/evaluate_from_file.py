import json
import numpy as np
from celltype_extractor import calculate_comprehensive_metrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from bert_score import BERTScorer
import evaluate
from transformers import AutoTokenizer

def load_predictions_from_json(json_path):
    """Load predictions and targets from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if 'all_predictions' in data and 'all_targets' in data:
        predictions = data['all_predictions']['text'] if isinstance(data['all_predictions'], dict) else data['all_predictions']
        targets = data['all_targets']['text'] if isinstance(data['all_targets'], dict) else data['all_targets']
    elif 'detailed_comparisons' in data:
        predictions = [item['predicted_text'] for item in data['detailed_comparisons']]
        targets = [item['target_text'] for item in data['detailed_comparisons']]
    else:
        raise ValueError("JSON format not recognized. Expected 'all_predictions'/'all_targets' or 'detailed_comparisons'")
    
    return predictions, targets

def compute_basic_metrics(predictions, targets):
    """Compute BLEU, BLEU-2, ROUGE-2, METEOR"""
    smooth = SmoothingFunction().method4
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
    
    bleu_scores = []
    bleu2_scores = []
    rouge2_scores = []
    meteor_scores = []
    
    print("Computing basic metrics...")
    for pred, ref in zip(predictions, targets):
        # BLEU-1
        bleu = sentence_bleu([ref.split()], pred.split(), smoothing_function=smooth)
        bleu_scores.append(bleu)
        
        # BLEU-2
        bleu2 = sentence_bleu([ref.split()], pred.split(), weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
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
    
    return {
        'bleu': np.mean(bleu_scores),
        'bleu2': np.mean(bleu2_scores),
        'rouge2': np.mean(rouge2_scores),
        'meteor': np.mean(meteor_scores)
    }

def compute_bertscore(predictions, targets):
    """Compute BERTScore using biomedical BERT"""
    print("Computing BERTScore (this may take a while)...")
    model_name = "dmis-lab/biobert-large-cased-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Truncate texts
    truncated_predictions = tokenizer(predictions, padding="max_length", truncation=True, 
                                     max_length=495, return_tensors="pt")["input_ids"]
    truncated_predictions = tokenizer.batch_decode(truncated_predictions, skip_special_tokens=True)
    
    truncated_targets = tokenizer(targets, padding="max_length", truncation=True,
                                  max_length=495, return_tensors="pt")["input_ids"]
    truncated_targets = tokenizer.batch_decode(truncated_targets, skip_special_tokens=True)
    
    # Compute BERTScore
    bert_scorer = evaluate.load("bertscore")
    results = bert_scorer.compute(
        predictions=truncated_predictions,
        references=truncated_targets,
        model_type=model_name,
        num_layers=24,
        lang="en",
        verbose=False
    )
    
    return {
        'bert_precision': np.mean(results['precision']),
        'bert_recall': np.mean(results['recall']),
        'bert_f1': np.mean(results['f1'])
    }

def evaluate_from_file(json_path, 
                      use_bertscore=True,
                      use_comprehensive=True,
                      similarity_file_path="/home/arism/datasets/cell_type_similarities.pkl",
                      cell_type_csv_path="/home/arism/analysis_output/final_combined/final_combined_cell_type_top_values.csv",
                      disease_csv_path="/home/arism/analysis_output/final_combined/final_combined_disease_top_values.csv",
                      tissue_csv_path="/home/arism/analysis_output/final_combined/final_combined_tissue_top_values.csv",
                      pathway_descriptions_path="pathway_descriptions.json"):
    """
    Evaluate predictions from a JSON file
    
    Args:
        json_path: Path to JSON file with predictions and targets
        use_bertscore: Whether to compute BERTScore (slow)
        use_comprehensive: Whether to compute comprehensive metrics
        similarity_file_path: Path to cell type similarities
        cell_type_csv_path: Path to cell types CSV
        disease_csv_path: Path to diseases CSV
        tissue_csv_path: Path to tissues CSV
        pathway_descriptions_path: Path to pathway descriptions
    """
    
    # Load predictions and targets
    print(f"Loading predictions from {json_path}...")
    predictions, targets = load_predictions_from_json(json_path)
    print(f"Loaded {len(predictions)} predictions and {len(targets)} targets")
    
    # Compute basic metrics
    basic_metrics = compute_basic_metrics(predictions, targets)
    
    # Compute BERTScore
    if use_bertscore:
        bert_metrics = compute_bertscore(predictions, targets)
        basic_metrics.update(bert_metrics)
    
    # Compute comprehensive metrics
    if use_comprehensive:
        print("Computing comprehensive metrics...")
        comprehensive_metrics = calculate_comprehensive_metrics(
            predictions, targets,
            similarity_file_path,
            cell_type_csv_path,
            disease_csv_path,
            tissue_csv_path,
            pathway_descriptions_path
        )
        basic_metrics.update(comprehensive_metrics)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total samples: {len(predictions)}")
    print(f"\nText Generation Metrics:")
    print(f"  BLEU:    {basic_metrics['bleu']:.4f}")
    print(f"  BLEU-2:  {basic_metrics['bleu2']:.4f}")
    print(f"  ROUGE-2: {basic_metrics['rouge2']:.4f}")
    print(f"  METEOR:  {basic_metrics['meteor']:.4f}")
    
    if use_bertscore:
        print(f"\nBERTScore:")
        print(f"  Precision: {basic_metrics['bert_precision']:.4f}")
        print(f"  Recall:    {basic_metrics['bert_recall']:.4f}")
        print(f"  F1:        {basic_metrics['bert_f1']:.4f}")
    
    if use_comprehensive:
        print(f"\nCell Type Metrics:")
        print(f"  Accuracy:   {basic_metrics.get('cell_type_accuracy', 0):.4f}")
        print(f"  F1:         {basic_metrics.get('cell_type_f1', 0):.4f}")
        print(f"  Precision:  {basic_metrics.get('cell_type_precision', 0):.4f}")
        print(f"  Recall:     {basic_metrics.get('cell_type_recall', 0):.4f}")
        print(f"  Ontology Similarity: {basic_metrics.get('ontology_similarity_score', 0):.4f}")
        
        print(f"\nDisease Metrics:")
        print(f"  Accuracy:   {basic_metrics.get('disease_accuracy', 0):.4f}")
        print(f"  F1:         {basic_metrics.get('disease_f1', 0):.4f}")
        
        print(f"\nTissue Metrics:")
        print(f"  Accuracy:   {basic_metrics.get('tissue_accuracy', 0):.4f}")
        print(f"  F1:         {basic_metrics.get('tissue_f1', 0):.4f}")
        
        if 'pathway_subset_accuracy' in basic_metrics:
            print(f"\nPathway Metrics:")
            print(f"  Subset Accuracy: {basic_metrics['pathway_subset_accuracy']:.4f}")
            print(f"  Jaccard:         {basic_metrics.get('pathway_jaccard_accuracy', 0):.4f}")
            print(f"  Macro F1:        {basic_metrics.get('pathway_macro_f1', 0):.4f}")
    
    # Save results
    output_path = json_path.replace('.json', '_evaluation_results.json')
    with open(output_path, 'w') as f:
        json.dump(basic_metrics, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    return basic_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate predictions from JSON file")
    parser.add_argument("--json_path", type=str, help="Path to JSON file with predictions")
    parser.add_argument("--no-bertscore", action="store_true", help="Skip BERTScore computation")
    parser.add_argument("--no-comprehensive", action="store_true", help="Skip comprehensive metrics")
    parser.add_argument("--similarity-file", type=str, default="/home/arism/datasets/cell_type_similarities.pkl")
    parser.add_argument("--cell-type-csv", type=str, default="/home/arism/analysis_output/final_combined/final_combined_cell_type_top_values.csv")
    parser.add_argument("--disease-csv", type=str, default="/home/arism/analysis_output/final_combined/final_combined_disease_top_values.csv")
    parser.add_argument("--tissue-csv", type=str, default="/home/arism/analysis_output/final_combined/final_combined_tissue_top_values.csv")
    parser.add_argument("--pathway-json", type=str, default="pathway_descriptions.json")
    
    args = parser.parse_args()
    
    evaluate_from_file(
        args.json_path,
        use_bertscore=not args.no_bertscore,
        use_comprehensive=not args.no_comprehensive,
        similarity_file_path=args.similarity_file,
        cell_type_csv_path=args.cell_type_csv,
        disease_csv_path=args.disease_csv,
        tissue_csv_path=args.tissue_csv,
        pathway_descriptions_path=args.pathway_json
    )