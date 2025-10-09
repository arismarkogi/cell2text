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

def load_from_fsdp_checkpoint(model, checkpoint_path, rank=0):
    """
    Load model weights from FSDP checkpoint
    
    Args:
        model: The model instance to load weights into
        checkpoint_path: Path to the FSDP checkpoint directory or file
        rank: Current process rank
    
    Returns:
        Additional checkpoint state (optimizer, scheduler states, etc.)
    """
    if rank == 0:
        print(f"Loading FSDP checkpoint from: {checkpoint_path}")
    
    # Determine checkpoint file path
    if os.path.isdir(checkpoint_path):
        checkpoint_file = os.path.join(checkpoint_path, "checkpoint.pt")
    else:
        checkpoint_file = checkpoint_path
    
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
    
    # Load checkpoint on CPU first to avoid OOM
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    
    if rank == 0:
        print("Checkpoint keys:", checkpoint.keys())
    
    # Extract model state dict
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
    else:
        # Assume the checkpoint is the state dict itself
        model_state_dict = checkpoint
    
    # Check if checkpoint contains LoRA weights
    has_lora = any('lora' in key.lower() for key in model_state_dict.keys())
    if rank == 0 and has_lora:
        print("✓ Detected LoRA weights in FSDP checkpoint")
    
    # Load state dict into model
    # Handle potential key mismatches (FSDP might have different prefixes)
    try:
        # Try direct loading first
        missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
        
        if rank == 0:
            if missing_keys:
                print(f"Warning: Missing keys in checkpoint: {missing_keys[:10]}...")  # Show first 10
            if unexpected_keys:
                print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys[:10]}...")  # Show first 10
            
            if not missing_keys and not unexpected_keys:
                print("✓ All model weights loaded successfully from FSDP checkpoint")
            else:
                print(f"✓ Model weights loaded with {len(missing_keys)} missing and {len(unexpected_keys)} unexpected keys")
    
    except Exception as e:
        if rank == 0:
            print(f"Error loading state dict directly: {e}")
            print("Attempting to handle key mismatches...")
        
        # Try to handle common FSDP wrapper prefixes
        new_state_dict = {}
        for key, value in model_state_dict.items():
            # Remove common FSDP prefixes
            new_key = key
            if key.startswith('_fsdp_wrapped_module.'):
                new_key = key.replace('_fsdp_wrapped_module.', '')
            elif key.startswith('module.'):
                new_key = key.replace('module.', '')
            new_state_dict[new_key] = value
        
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        if rank == 0:
            if missing_keys:
                print(f"Warning: Missing keys after prefix removal: {missing_keys[:10]}...")
            if unexpected_keys:
                print(f"Warning: Unexpected keys after prefix removal: {unexpected_keys[:10]}...")
            print("✓ Model weights loaded after handling key mismatches")
    
    # Return additional checkpoint information
    additional_state = {
        'global_step': checkpoint.get('global_step', None),
        'epoch': checkpoint.get('epoch', None),
        'has_lora': has_lora,
    }
    
    if rank == 0 and additional_state['global_step'] is not None:
        print(f"Checkpoint was saved at global step: {additional_state['global_step']}")
    
    return additional_state

def create_model_args(args):
    """Create model arguments dictionary from command line arguments"""
    model_args = {
        # Required paths
        "geneformer_path": args.geneformer_path,
        "llama_path": args.llama_path,

        "skip_lora_init": args.load_from_fsdp,
        
        # Model loading paths
        "load_model_checkpoint_path": args.checkpoint_path if not args.load_from_fsdp else None,
        "load_adapter_checkpoint_dir": args.adapter_checkpoint_dir,
        
        # Model architecture
        "projector": args.projector,
        "cell_encoder_hidden_size": args.cell_encoder_hidden_size,
        "decoder_hidden_size": args.decoder_hidden_size,
        "mlp_hidden_size": args.mlp_hidden_size,
        "mlp_dropout": args.mlp_dropout,
        "top_k": args.top_k,

        # QFormer arguments
        "qformer_cross_attention_freq": args.qformer_cross_attention_freq,
        "qformer_use_flash_attn": args.qformer_use_flash_attn,
        
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
    
    # Load model
    if rank == 0:
        if args.load_from_fsdp:
            print(f"Loading model architecture and then FSDP checkpoint from: {args.checkpoint_path}")
        else:
            print(f"Loading model from: {args.checkpoint_path}")
            if args.adapter_checkpoint_dir:
                print(f"Loading LoRA adapter from: {args.adapter_checkpoint_dir}")
    
    # Initialize model
    model = load_model(model_args)
    
    # Load FSDP checkpoint if specified
    if args.load_from_fsdp:
        checkpoint_info = load_from_fsdp_checkpoint(model, args.checkpoint_path, rank=rank)
        if rank == 0 and checkpoint_info['global_step']:
            print(f"Loaded checkpoint from training step: {checkpoint_info['global_step']}")
    
    # Move model to device and wrap with DDP
    model.to(rank)
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
    
    # Run evaluation
    results = evaluate_cell2text_model(
        model=model,
        val_loader=test_loader,
        tokenizer=tokenizer,
        device=rank,
        print_examples=args.print_examples if rank == 0 else 0,
        save_results=args.save_results if rank == 0 else None,
        save_detailed_json=args.save_detailed_json if rank == 0 else None,
        use_ddp=True,
        use_bertscore=args.use_bertscore,
        use_comprehensive_metrics=args.use_comprehensive_metrics,
        similarity_file_path=args.similarity_file_path,
        cell_type_csv_path=args.cell_type_csv_path,
        disease_csv_path=args.disease_csv_path,
        tissue_csv_path=args.tissue_csv_path,
        pathway_descriptions_path=args.pathway_descriptions_path
    )
    
    # Print results
    if rank == 0 and results is not None:
        print(f"\n{'='*60}")
        print(f"FINAL EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total Samples: {results['total_samples']}")
        print(f"\nText Generation Metrics:")
        print(f"  BLEU-1: {results['bleu']:.4f}")
        print(f"  BLEU-2 (B-2): {results['bleu2']:.4f}")
        print(f"  BLEU-4 (B-4): {results['bleu4']:.4f}")
        print(f"  ROUGE-1 (R-1): {results['rouge1']:.4f}")
        print(f"  ROUGE-2 (R-2): {results['rouge2']:.4f}")
        print(f"  ROUGE-L (R-L): {results['rougeL']:.4f}")
        
        
        if results.get('biobert_f1') is not None:
            print(f"\nBERTScore Metrics:")
            print(f"  BioBERT (BBT-f1):")
            print(f"    Precision: {results['biobert_precision']:.4f}")
            print(f"    Recall: {results['biobert_recall']:.4f}")
            print(f"    F1: {results['biobert_f1']:.4f}")
            print(f"  RoBERTa (RBT-f1):")
            print(f"    Precision: {results['roberta_precision']:.4f}")
            print(f"    Recall: {results['roberta_recall']:.4f}")
            print(f"    F1: {results['roberta_f1']:.4f}")
        
        print(f"\nCell Type Metrics:")
        print(f"  Cell Type Accuracy: {results['cell_type_accuracy']:.4f}")
        print(f"  Ontology Similarity: {results['ontology_similarity_score']:.4f}")
        
        if args.use_comprehensive_metrics:
            print(f"\nComprehensive Metrics:")
            for key, value in results.items():
                if key.startswith(('disease_', 'tissue_', 'pathway_', 'cell_type_precision', 'cell_type_recall', 'cell_type_f1')):
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.4f}")
        
        print(f"{'='*60}")
    
    cleanup_ddp()

def main():
    parser = argparse.ArgumentParser(description="Evaluate Cell2Text model on test set with DDP")
    
    # Required arguments
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint (standard checkpoint or FSDP checkpoint dir)")
    parser.add_argument("--test_data_path", type=str, required=True,
                        help="Path to test dataset")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to tokenizer")
    parser.add_argument("--geneformer_path", type=str, required=True,
                        help="Path to pretrained Geneformer model")
    parser.add_argument("--llama_path", type=str, required=True,
                        help="Path to pretrained LLaMA model")
    
    # Checkpoint loading options
    parser.add_argument("--load_from_fsdp", action="store_true",
                        help="Load checkpoint from FSDP training (handles FSDP-specific state dict)")
    parser.add_argument("--adapter_checkpoint_dir", type=str, default=None,
                        help="Path to LoRA adapter checkpoint directory (not used with FSDP checkpoints)")
    
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
    
    # Perceiver arguments
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
    parser.add_argument("--qformer_cross_attention_freq", type=int, default=2,
                        help="Cross-attention frequency in QFormer")
    parser.add_argument("--qformer_use_flash_attn", type=bool, default=True,
                        help="Use flash attention in QFormer")
    
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
    
    # Comprehensive metrics arguments
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
    
    if args.load_from_fsdp:
        print("Loading from FSDP checkpoint format")
    
    # Spawn processes for DDP
    mp.spawn(run_evaluation, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()