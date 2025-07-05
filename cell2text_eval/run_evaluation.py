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

def load_config_from_args(args):
    """Create config from command line arguments"""
    config = PretrainedConfig()
    
    # Model architecture settings
    config.cell_encoder_hidden_size = args.cell_encoder_hidden_size
    config.decoder_hidden_size = args.decoder_hidden_size
    config.mlp_hidden_size = args.mlp_hidden_size
    config.mlp_dropout = args.mlp_dropout
    config.top_k = args.top_k
    config.projector = args.projector
    
    # Geneformer settings
    config.emb_mode = args.emb_mode
    config.max_ncells = args.max_ncells
    config.emb_layer = args.emb_layer
    config.emb_label = args.emb_label
    config.nproc = args.nproc
    config.forward_batch_size = args.forward_batch_size
    config.summary_stat = args.summary_stat
    config.token_dictionary_path = args.token_dictionary_path
    
    # Generation settings
    config.max_length = args.max_length
    config.num_beams = args.num_beams
    config.early_stopping = args.early_stopping
    config.no_repeat_ngram_size = args.no_repeat_ngram_size
    config.temperature = args.temperature
    config.top_p = args.top_p
    
    # Perceiver settings (if using perceiver projector)
    if args.projector == "perceiver":
        config.num_latents = args.num_latents
        config.perceiver_cross_attn_layers = args.perceiver_cross_attn_layers
        config.perceiver_num_heads = args.perceiver_num_heads
        config.ff_mult = args.ff_mult
        config.perceiver_dropout = args.perceiver_dropout
        config.use_position_encoding = args.use_position_encoding
    
    return config

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
    
    # Create config
    config = load_config_from_args(args)
    
    # Load model
    if rank == 0:
        print(f"Loading model from: {args.checkpoint_path}")
    model = Cell2TextModel.from_pretrained(args.checkpoint_path, config=config)
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
        num_workers=4
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
        use_ddp=True
    )
    
    # Only print results on rank 0
    if rank == 0:
        print("\n" + "="*60)
        print("FINAL EVALUATION RESULTS")
        print("="*60)
        print(f"BLEU Score: {results['bleu']:.4f}")
        if results['validation_loss'] is not None:
            print(f"Validation Loss: {results['validation_loss']:.4f}")
        print(f"Cell Type Accuracy: {results['cell_type_accuracy']:.4f}")
        print(f"Cell Type F1 Score: {results['cell_type_f1']:.4f}")
        print(f"Cell Type Precision: {results['cell_type_precision']:.4f}")
        print(f"Cell Type Recall: {results['cell_type_recall']:.4f}")
        
        # Save summary results
        if args.save_results:
            summary_path = args.save_results.replace('.json', '_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Summary results saved to: {summary_path}")
        
        print("Evaluation completed successfully!")
    
    cleanup_ddp()

def main():
    parser = argparse.ArgumentParser(description="Evaluate Cell2Text model on test set with DDP")
    
    # Required arguments
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to pretrained model checkpoint directory")
    parser.add_argument("--test_data_path", type=str, required=True,
                        help="Path to test dataset")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to tokenizer ")
    
    # Model architecture arguments
    parser.add_argument("--cell_encoder_hidden_size", type=int, default=1152,
                        help="Hidden size of cell encoder")
    parser.add_argument("--decoder_hidden_size", type=int, default=3072,
                        help="Hidden size of decoder")
    parser.add_argument("--mlp_hidden_size", type=int, default=2048,
                        help="Hidden size of MLP projector")
    parser.add_argument("--mlp_dropout", type=float, default=0.1,
                        help="Dropout rate for MLP projector")
    parser.add_argument("--top_k", type=int, default=256,
                        help="Top k gene expression tokens to use")
    parser.add_argument("--projector", type=str, default="mlp", choices=["mlp", "perceiver"],
                        help="Type of projector to use")
    
    # Geneformer arguments
    parser.add_argument("--emb_mode", type=str, default="cell",
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
                        default="/home/arismarkog/Desktop/cell2text/Geneformer/geneformer/token_dictionary_gc104M.pkl",
                        help="Path to token dictionary")
    
    # Generation arguments
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum generation length")
    parser.add_argument("--num_beams", type=int, default=4,
                        help="Number of beams for beam search")
    parser.add_argument("--early_stopping", type=bool, default=True,
                        help="Use early stopping")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3,
                        help="No repeat ngram size")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top p for nucleus sampling")
    
    # Perceiver arguments (only used if projector == "perceiver")
    parser.add_argument("--num_latents", type=int, default=128,
                        help="Number of latent tokens for Perceiver")
    parser.add_argument("--perceiver_cross_attn_layers", type=int, default=2,
                        help="Number of cross attention layers in Perceiver")
    parser.add_argument("--perceiver_num_heads", type=int, default=8,
                        help="Number of attention heads in Perceiver")
    parser.add_argument("--ff_mult", type=int, default=4,
                        help="Feed forward multiplier in Perceiver")
    parser.add_argument("--perceiver_dropout", type=float, default=0.1,
                        help="Dropout rate in Perceiver")
    parser.add_argument("--use_position_encoding", type=bool, default=True,
                        help="Use position encoding in Perceiver")
    
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