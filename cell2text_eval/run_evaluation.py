import torch
import os
import argparse
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import PretrainedConfig
import json
import sys

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cell2text_dataset.dataset import Cell2TextDataset
from cell2text_model.model import Cell2TextModel
from cell2text_model.util import load_model
from cell2text_eval.evaluation import evaluate_cell2text_model

def setup_ddp(rank, world_size):
    """Setup DDP"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """Cleanup DDP"""
    dist.destroy_process_group()

def create_model_args(args):
    """Create model arguments dictionary from command line arguments"""
    model_args = {
        # Required paths
        "geneformer_path": args.geneformer_path,
        "llama_path": args.llama_path,
        
        # Model loading paths
        "load_model_checkpoint_path": args.checkpoint_path,
        "load_adapter_checkpoint_dir": args.adapter_checkpoint_dir,
        
        # Model architecture
        "projector": args.projector,
        "cell_encoder_hidden_size": args.cell_encoder_hidden_size,
        "decoder_hidden_size": args.decoder_hidden_size,
        "mlp_hidden_size": args.mlp_hidden_size,
        "mlp_dropout": args.mlp_dropout,
        "top_k": args.top_k,

        #QFormer arguments
        #"qformer_bert_model": args.qformer_bert_model,
        "qformer_cross_attention_freq": args.qformer_cross_attention_freq,
        "qformer_use_flash_attn": args.qformer_use_flash_attn,
        #"qformer_freeze": args.qformer_freeze,
        
        # Geneformer settings
        "emb_mode": args.emb_mode,
        "max_ncells": args.max_ncells,
        "emb_layer": args.emb_layer,
        "emb_label": args.emb_label,
        "nproc": args.nproc,
        "forward_batch_size": args.forward_batch_size,
        "summary_stat": args.summary_stat,
        "token_dictionary_path": args.token_dictionary_path,
        
        # Generation settings
        "max_length": args.max_length,
        "num_beams": args.num_beams,
        "early_stopping": args.early_stopping,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        
        # Training settings
        "lora_rank": args.lora_rank,
    }

    
    # Add Perceiver-specific settings if using perceiver projector
    model_args.update({
            "num_latents": args.num_latents,
            "perceiver_cross_attn_layers": args.perceiver_cross_attn_layers,
            "perceiver_num_heads": args.perceiver_num_heads,
            "ff_mult": args.ff_mult,
            "perceiver_dropout": args.perceiver_dropout,
            "use_position_encoding": args.use_position_encoding,
    })
    
    return model_args

def run_evaluation(rank, world_size, args):
    """Run evaluation on one GPU"""
    setup_ddp(rank, world_size)
    
    # Load tokenizer
    if rank == 0:
        print(f"Loading tokenizer from: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create model arguments
    model_args = create_model_args(args)
    
    # Load model using the proper load_model function
    if rank == 0:
        print(f"Loading model from: {args.checkpoint_path}")
        if args.adapter_checkpoint_dir:
            print(f"Loading LoRA adapter from: {args.adapter_checkpoint_dir}")
    
    model = load_model(model_args)
    model.to(rank)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])
    
    # Load test dataset
    if rank == 0:
        print(f"Loading test dataset from: {args.test_data_path}")
    test_dataset = Cell2TextDataset(
        data_path=args.test_data_path,
        tokenizer=tokenizer,
        system_message=args.system_message,
        placeholder_token=args.placeholder_token,
        top_k=args.top_k,
        projector=args.projector,
        num_latents=args.num_latents if args.projector == "perceiver" else None
    )
    
    # Create distributed sampler
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    
    # Create data loader
    collate_fn = test_dataset.collate_fn(geneformer_pad_token_id=0, mode="inference")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    if rank == 0:
        print(f"Test dataset size: {len(test_dataset)}")
        print(f"Number of test batches: {len(test_loader)}")
        print("Starting evaluation...")
    
    # Run evaluation with BERTScore
    results = evaluate_cell2text_model(
        model=model,
        val_loader=test_loader,
        tokenizer=tokenizer,
        device=rank,
        print_examples=args.print_examples if rank == 0 else 0,
        save_results=args.save_results if rank == 0 else None,
        save_detailed_json=args.save_detailed_json if rank == 0 else None,  # NEW
        use_ddp=True,
        use_bertscore=args.use_bertscore,
        use_comprehensive_metrics=args.use_comprehensive_metrics,  # NEW
        similarity_file_path=args.similarity_file_path,
        cell_type_csv_path=args.cell_type_csv_path,  # NEW
        disease_csv_path=args.disease_csv_path,  # NEW
        tissue_csv_path=args.tissue_csv_path,  # NEW
        pathway_descriptions_path=args.pathway_descriptions_path  # NEW
    )
    
    # Expand the results printing to include comprehensive metrics:
    if rank == 0 and results is not None:
        print(f"\n{'='*60}")
        print(f"FINAL EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total Samples: {results['total_samples']}")
        print(f"BLEU: {results['bleu']:.4f}")
        print(f"BLEU-2: {results['bleu2']:.4f}")
        print(f"ROUGE-2: {results['rouge2']:.4f}")
        print(f"METEOR: {results['meteor']:.4f}")
        print(f"MMD (↓): {results['mmd']:.4f}")
        print(f"EMD (↓): {results['emd']:.4f}")
            
        if results['bert_score_f1'] is not None:
            print(f"BERTScore Precision: {results['bert_score_precision']:.4f}")
            print(f"BERTScore Recall: {results['bert_score_recall']:.4f}")
            print(f"BERTScore F1: {results['bert_score_f1']:.4f}")
            
        print(f"Cell Type Accuracy: {results['cell_type_accuracy']:.4f}")
        print(f"Ontology Similarity: {results['ontology_similarity_score']:.4f}")
        
        # ADD: Print comprehensive metrics if available
        if args.use_comprehensive_metrics:
            print(f"\nCOMPREHENSIVE METRICS:")
            for key, value in results.items():
                if key.startswith(('disease_', 'tissue_', 'pathway_')):
                    print(f"{key}: {value:.4f}")
        
        print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Cell2Text model on test set with DDP")
    
    # Required arguments
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to pretrained model checkpoint")
    parser.add_argument("--test_data_path", type=str, required=True,
                        help="Path to test dataset")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to tokenizer")
    parser.add_argument("--geneformer_path", type=str, required=True,
                        help="Path to pretrained Geneformer model")
    parser.add_argument("--llama_path", type=str, required=True,
                        help="Path to pretrained LLaMA model")
    
    # Optional model loading arguments
    parser.add_argument("--adapter_checkpoint_dir", type=str, default=None,
                        help="Path to LoRA adapter checkpoint directory")
    
    # Model architecture arguments
    parser.add_argument("--cell_encoder_hidden_size", type=int, default=1152,
                        help="Hidden size of cell encoder")
    parser.add_argument("--decoder_hidden_size", type=int, default=3072,
                        help="Hidden size of decoder")
    parser.add_argument("--mlp_hidden_size", type=int, default=2048,
                        help="Hidden size of MLP projector")
    parser.add_argument("--mlp_dropout", type=float, default=0.1,
                        help="Dropout rate for MLP projector")
    parser.add_argument("--top_k", type=int, default=32,
                        help="Top k gene expression tokens to use")
    parser.add_argument("--projector", type=str, default="mlp", choices=["mlp", "perceiver", "qformer"],
                        help="Type of projector to use")
    
    # Geneformer arguments
    parser.add_argument("--emb_mode", type=str, default="gene",
                        help="Geneformer embedding mode")
    parser.add_argument("--max_ncells", type=int, default=1000,
                        help="Maximum number of cells")
    parser.add_argument("--emb_layer", type=int, default=-1,
                        help="Geneformer embedding layer")
    parser.add_argument("--emb_label", type=str, default=None,
                        help="Geneformer embedding label")
    parser.add_argument("--nproc", type=int, default=-1,
                        help="Number of processes")
    parser.add_argument("--forward_batch_size", type=int, default=100,
                        help="Forward batch size")
    parser.add_argument("--summary_stat", type=str, default=None,
                        help="Summary statistic")
    parser.add_argument("--token_dictionary_path", type=str, 
                        default="/home/arism/cell2text/Geneformer/geneformer/token_dictionary_gc104M.pkl",
                        help="Path to token dictionary")
    
    # Generation arguments
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum generation length")
    parser.add_argument("--num_beams", type=int, default=1,
                        help="Number of beams for beam search")
    parser.add_argument("--early_stopping", type=bool, default=True,
                        help="Use early stopping")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3,
                        help="No repeat ngram size")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top p for nucleus sampling")
    
    # Training/Model arguments
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank")
    
    # Perceiver arguments (only used if projector == "perceiver")
    parser.add_argument("--num_latents", type=int, default=128,
                        help="Number of latent tokens for Perceiver")
    parser.add_argument("--perceiver_cross_attn_layers", type=int, default=1,
                        help="Number of cross attention layers in Perceiver")
    parser.add_argument("--perceiver_num_heads", type=int, default=2,
                        help="Number of attention heads in Perceiver")
    parser.add_argument("--ff_mult", type=float, default=4,
                        help="Feed forward multiplier in Perceiver")
    parser.add_argument("--perceiver_dropout", type=float, default=0.1,
                        help="Dropout rate in Perceiver")
    parser.add_argument("--use_position_encoding", type=bool, default=False,
                        help="Use position encoding in Perceiver")
    
     # QFormer arguments
    # parser.add_argument("--qformer_bert_model", type=str, default="dmis-lab/biobert-base-cased-v1.2", 
    #                    help="BERT model to use in QFormer")
    parser.add_argument("--qformer_cross_attention_freq", type=int, default=2,
                       help="Cross-attention frequency in QFormer")
    parser.add_argument("--qformer_use_flash_attn", type=bool, default=True,
                       help="Use flash attention in QFormer")
    # parser.add_argument("--qformer_freeze", type=bool, default=True,
    #                    help="Freeze QFormer parameters")

    
    # Evaluation arguments
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for evaluation")
    parser.add_argument("--print_examples", type=int, default=10,
                        help="Number of examples to print")
    parser.add_argument("--save_results", type=str, default=None,
                        help="Path to save detailed results JSON")
    parser.add_argument("--system_message", type=str, 
                        default="You are a scientific assistant specialized in analyzing single-cell gene expression data. Given the gene expression profile, describe the cell type and its characteristics clearly and concisely in professional language.",
                        help="System message for the model")
    parser.add_argument("--placeholder_token", type=str, default="<|reserved_special_token_1|>",
                        help="Placeholder token for expression embeddings")
    
    parser.add_argument("--use_bertscore", type=bool, default=True,
                        help="Whether to compute BERTScore")
    parser.add_argument("--similarity_file_path", type=str, 
                    default="/home/arism/datasets/cell_type_similarities.pkl",
                    help="Path to precomputed cell type similarities file")
    

    # Add these new arguments for enhanced functionality:
    parser.add_argument("--save_detailed_json", type=str, default=None,
                        help="Path to save detailed predictions and targets JSON")
    parser.add_argument("--use_comprehensive_metrics", type=bool, default=False,
                        help="Whether to compute comprehensive metrics (disease, tissue, pathways)")
    parser.add_argument("--cell_type_csv_path", type=str, 
                        default="/home/arism/analysis_output/final_combined/final_combined_cell_type_top_values.csv",
                        help="Path to cell type CSV file")
    parser.add_argument("--disease_csv_path", type=str, 
                        default="/home/arism/analysis_output/final_combined/final_combined_disease_top_values.csv",
                        help="Path to disease CSV file")
    parser.add_argument("--tissue_csv_path", type=str, 
                        default="/home/arism/analysis_output/final_combined/final_combined_tissue_top_values.csv",
                        help="Path to tissue CSV file")
    parser.add_argument("--pathway_descriptions_path", type=str, default="pathway_descriptions.json",
                        help="Path to pathway descriptions JSON file")


    args = parser.parse_args()
    
    # Check GPU availability
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print(f"Warning: Only {world_size} GPU(s) available. DDP requires at least 2 GPUs.")
        print("Running on single GPU...")
        world_size = 1
    
    print(f"Using {world_size} GPUs for evaluation")
    
    # Spawn processes for DDP
    mp.spawn(run_evaluation, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()