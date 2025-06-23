#!/usr/bin/env python3
"""
Cell2Text Model Test and Evaluation Script

This script loads a trained Cell2Text model and runs comprehensive evaluation
including BLEU scores and cell type extraction metrics.
"""

import os
import sys
import torch
import argparse
import json
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PretrainedConfig
import logging

# Import your model and evaluation functions
from cell2text_model.model import Cell2TextModel
from evaluation import evaluate_cell2text_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_device():
    """Setup and return the appropriate device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device

def load_model_and_tokenizer(model_path, config_path=None):
    """
    Load the trained Cell2Text model and tokenizer
    
    Args:
        model_path: Path to the trained model directory
        config_path: Optional path to config file (if different from model_path)
    
    Returns:
        model: Loaded Cell2TextModel
        tokenizer: Loaded tokenizer
    """
    logger.info(f"Loading model from: {model_path}")
    
    # Load config
    if config_path:
        config = PretrainedConfig.from_json_file(config_path)
    else:
        config_file = os.path.join(model_path, "config.json")
        if os.path.exists(config_file):
            config = PretrainedConfig.from_json_file(config_file)
        else:
            raise FileNotFoundError(f"No config.json found in {model_path}")
    
    logger.info("Config loaded successfully")
    
    # Load model
    model = Cell2TextModel.from_pretrained(
        pretrained_model_name_or_path=model_path,
        config=config
    )
    
    logger.info("Model loaded successfully")
    
    # Load tokenizer - assuming it's stored with the model or using a standard one
    tokenizer_path = os.path.join(model_path, "tokenizer")
    if os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logger.info(f"Tokenizer loaded from: {tokenizer_path}")
    else:
        # Fallback to a standard tokenizer - adjust as needed
        tokenizer_name = getattr(config, 'tokenizer_name', 'meta-llama/Llama-2-7b-hf')
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        logger.info(f"Using fallback tokenizer: {tokenizer_name}")
    
    return model, tokenizer

def create_test_dataloader(test_data_path, batch_size=8, num_workers=2):
    """
    Create test dataloader from your test dataset
    
    Args:
        test_data_path: Path to test dataset
        batch_size: Batch size for evaluation
        num_workers: Number of workers for data loading
    
    Returns:
        DataLoader for test data
    """
    # You'll need to implement this based on your dataset format
    # This is a placeholder - replace with your actual dataset loading logic
    
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data not found at: {test_data_path}")
    
    # Example implementation - replace with your actual dataset class
    from your_dataset_module import Cell2TextDataset  # Replace with actual import
    
    test_dataset = Cell2TextDataset(
        data_path=test_data_path,
        mode='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Test dataset loaded: {len(test_dataset)} samples")
    return test_loader

def run_evaluation(model, tokenizer, test_loader, device, args):
    """
    Run comprehensive evaluation of the model
    
    Args:
        model: Cell2TextModel
        tokenizer: Tokenizer
        test_loader: Test data loader
        device: Device to run on
        args: Command line arguments
    
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Starting evaluation...")
    
    # Move model to device
    model = model.to(device)
    
    # Run evaluation
    results = evaluate_cell2text_model(
        model=model,
        val_loader=test_loader,
        tokenizer=tokenizer,
        device=device,
        print_examples=args.print_examples,
        save_results=args.save_results
    )
    
    return results

def save_test_results(results, output_path):
    """Save test results to file"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Test and evaluate Cell2Text model")
    
    # Model and data paths
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model directory")
    parser.add_argument("--test_data_path", type=str, required=True,
                       help="Path to test dataset")
    parser.add_argument("--config_path", type=str, default=None,
                       help="Optional path to config file")
    
    # Evaluation settings
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for evaluation")
    parser.add_argument("--print_examples", type=int, default=10,
                       help="Number of example predictions to print")
    parser.add_argument("--num_workers", type=int, default=2,
                       help="Number of workers for data loading")
    
    # Output settings
    parser.add_argument("--save_results", type=str, default=None,
                       help="Path to save detailed results JSON")
    parser.add_argument("--output_dir", type=str, default="./test_results",
                       help="Directory to save all outputs")
    
    # Generation settings
    parser.add_argument("--max_length", type=int, default=100,
                       help="Maximum generation length")
    parser.add_argument("--num_beams", type=int, default=4,
                       help="Number of beams for beam search")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Temperature for generation")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = setup_device()
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(
            args.model_path, 
            args.config_path
        )
        
        # Create test dataloader
        test_loader = create_test_dataloader(
            args.test_data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # Set default save path if not provided
        if args.save_results is None:
            args.save_results = os.path.join(args.output_dir, "detailed_results.json")
        
        # Run evaluation
        results = run_evaluation(model, tokenizer, test_loader, device, args)
        
        # Print final summary
        print(f"\n{'='*60}")
        print(f"FINAL EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"BLEU Score: {results['bleu']:.4f}")
        print(f"Cell Type Accuracy: {results['cell_type_accuracy']:.4f}")
        print(f"Cell Type F1: {results['cell_type_f1']:.4f}")
        print(f"Cell Type Precision: {results['cell_type_precision']:.4f}")
        print(f"Cell Type Recall: {results['cell_type_recall']:.4f}")
        
        # Save summary results
        summary_path = os.path.join(args.output_dir, "summary_results.json")
        save_test_results(results, summary_path)
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()


# Additional utility functions for different testing scenarios

def quick_test(model_path, sample_expression_tokens=None):
    """
    Quick test function for basic model loading and single prediction
    
    Args:
        model_path: Path to model
        sample_expression_tokens: Optional sample data for testing
    """
    device = setup_device()
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(model_path)
    model = model.to(device)
    model.eval()
    
    # Create dummy data if not provided
    if sample_expression_tokens is None:
        batch_size = 1
        seq_length = 100
        vocab_size = 30000  # Adjust based on your token dictionary
        
        sample_expression_tokens = torch.randint(0, vocab_size, (batch_size, seq_length))
        sample_lengths = torch.tensor([seq_length])
        
        # Create a simple prompt
        prompt = "This cell"
        prompt_tokens = tokenizer(prompt, return_tensors="pt")
        
        print("Using dummy data for quick test...")
    
    # Move to device
    sample_expression_tokens = sample_expression_tokens.to(device)
    sample_lengths = sample_lengths.to(device)
    prompt_tokens = {k: v.to(device) for k, v in prompt_tokens.items()}
    
    # Generate description
    with torch.no_grad():
        generated = model.generate_cell_description(
            expression_tokens=sample_expression_tokens,
            expression_token_lengths=sample_lengths,
            inputs=prompt_tokens['input_ids'],
            attention_mask=prompt_tokens['attention_mask'],
            device=device
        )
    
    print(f"Generated description: {generated}")
    print("Quick test completed successfully!")

def benchmark_model(model_path, num_samples=100):
    """
    Benchmark model inference speed
    
    Args:
        model_path: Path to model
        num_samples: Number of samples to benchmark
    """
    import time
    
    device = setup_device()
    model, tokenizer = load_model_and_tokenizer(model_path)
    model = model.to(device)
    model.eval()
    
    # Create dummy data
    batch_size = 1
    seq_length = 100
    vocab_size = 30000
    
    times = []
    
    print(f"Benchmarking model with {num_samples} samples...")
    
    for i in range(num_samples):
        # Generate random data
        expression_tokens = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
        lengths = torch.tensor([seq_length]).to(device)
        
        prompt = "This cell"
        prompt_tokens = tokenizer(prompt, return_tensors="pt")
        prompt_tokens = {k: v.to(device) for k, v in prompt_tokens.items()}
        
        # Time the generation
        start_time = time.time()
        
        with torch.no_grad():
            _ = model.generate_cell_description(
                expression_tokens=expression_tokens,
                expression_token_lengths=lengths,
                inputs=prompt_tokens['input_ids'],
                attention_mask=prompt_tokens['attention_mask'],
                device=device
            )
        
        end_time = time.time()
        times.append(end_time - start_time)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{num_samples} samples")
    
    # Print benchmark results
    avg_time = sum(times) / len(times)
    print(f"\nBenchmark Results:")
    print(f"Average inference time: {avg_time:.4f} seconds")
    print(f"Samples per second: {1/avg_time:.2f}")
    print(f"Total time: {sum(times):.2f} seconds")