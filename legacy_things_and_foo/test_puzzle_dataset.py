import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import numpy as np
from cellpuzzle_dataset import CellPuzzlesDataset

def test_cellpuzzles_dataset():
    """Comprehensive test of CellPuzzlesDataset functionality"""
    
    print("="*80)
    print("CELLPUZZLES DATASET COMPREHENSIVE TEST")
    print("="*80)
    
    # Initialize tokenizer
    print("1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    print(f"   Tokenizer loaded: {tokenizer.__class__.__name__}")
    print(f"   Vocab size: {tokenizer.vocab_size}")
    print(f"   Pad token ID: {tokenizer.pad_token_id}")
    print(f"   EOS token: {tokenizer.eos_token}")
    # Fix the tokenizer pad token issue

    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    print(f"Pad token ID: {tokenizer.pad_token_id}")


    
    # Initialize dataset
    print("\n2. Loading dataset...")
    try:
        dataset = CellPuzzlesDataset(
            data_path="/home/arism/datasets/puzzle_datasets/train_geneformer.dataset",
            tokenizer=tokenizer,
            projector="qformer",
            num_query_tokens=32,
            pad_cells=True
        )
        print(f"   Dataset loaded successfully with {len(dataset)} samples")
    except Exception as e:
        print(f"   ERROR loading dataset: {e}")
        return
    
    # Test single sample
    print("\n3. Testing single sample...")
    try:
        sample_idx = min(42, len(dataset) - 1)  # Use valid index
        sample = dataset[sample_idx]
        
        print(f"   Sample {sample_idx} structure:")
        for key, value in sample.items():
            if key == "cell_expressions":
                print(f"     {key}: List of {len(value)} tensors")
                for i, expr in enumerate(value):
                    print(f"       Cell {i}: shape {expr.shape}, dtype {expr.dtype}")
            elif isinstance(value, torch.Tensor):
                print(f"     {key}: shape {value.shape}, dtype {value.dtype}")
            elif isinstance(value, str):
                print(f"     {key}: str (length {len(value)})")
                print(f"       Preview: {value}...")
            else:
                print(f"     {key}: {type(value)} = {value}")
        
    except Exception as e:
        print(f"   ERROR accessing sample: {e}")
        return
    
    # Test collate function
    print("\n4. Testing collate function...")
    try:
        collate_fn = dataset.collate_fn(
            geneformer_pad_token_id=0,
            mode="train"
        )
        
        # Test with small batch
        batch_size = 3
        indices = list(range(min(batch_size, len(dataset))))
        batch_samples = [dataset[i] for i in indices]
        
        print(f"   Creating batch of {len(batch_samples)} samples...")
        batch = collate_fn(batch_samples)
        
        print(f"   Batch structure:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"     {key}: shape {value.shape}, dtype {value.dtype}")
                # Show some statistics
                if value.dtype in [torch.float32, torch.float16, torch.long, torch.int]:
                    print(f"       min: {value.min().item()}, max: {value.max().item()}")
            elif isinstance(value, list):
                print(f"     {key}: list of {len(value)} items")
            else:
                print(f"     {key}: {type(value)}")
                
    except Exception as e:
        print(f"   ERROR with collate function: {e}")
        return
    
    # Test DataLoader
    print("\n5. Testing DataLoader...")
    try:
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0  # Avoid multiprocessing issues
        )
        
        print(f"   DataLoader created with {len(dataloader)} batches")
        
        # Test first batch
        first_batch = next(iter(dataloader))
        print(f"   First batch loaded successfully:")
        for key, value in first_batch.items():
            if isinstance(value, torch.Tensor):
                print(f"     {key}: {value.shape}")
            else:
                print(f"     {key}: {type(value)}")
        
        # Test memory usage
        expression_tokens = first_batch["expression_tokens"]
        memory_mb = expression_tokens.numel() * expression_tokens.element_size() / 1024 / 1024
        print(f"   Expression tokens memory usage: {memory_mb:.2f} MB")
        
    except Exception as e:
        print(f"   ERROR with DataLoader: {e}")
        return
    
    # Test inference mode
    print("\n6. Testing inference mode...")
    try:
        inference_collate = dataset.collate_fn(
            geneformer_pad_token_id=0,
            mode="inference"
        )
        
        inference_batch = inference_collate(batch_samples)
        print(f"   Inference batch structure:")
        for key, value in inference_batch.items():
            if isinstance(value, torch.Tensor):
                print(f"     {key}: {value.shape}")
            elif isinstance(value, list):
                print(f"     {key}: list of {len(value)} items")
            else:
                print(f"     {key}: {type(value)}")
                
    except Exception as e:
        print(f"   ERROR with inference mode: {e}")
        return
    
    # Test placeholder replacement
    print("\n7. Testing placeholder replacement...")
    try:
        sample = dataset[0]
        raw_user_msg = sample["raw_user_msg"]
        processed_msg = sample  # The processed message is in the prompt
        
        print(f"   Raw message length: {len(raw_user_msg)}")
        print(f"   Placeholder token: {dataset.placeholder_token}")
        print(f"   Number of query tokens: {dataset.num_query_tokens}")
        
        # Decode the prompt to see placeholder replacement
        prompt_ids = sample["prompt_input_ids"][0]
        decoded_prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        placeholder_count = decoded_prompt.count(dataset.placeholder_token)
        print(f"   Placeholders in prompt: {placeholder_count}")
        print(f"   Expected placeholders: {sample['num_cells'] * dataset.num_query_tokens}")
        
    except Exception as e:
        print(f"   ERROR testing placeholder replacement: {e}")
    
    # Test SanityDataset
    print("\n8. Testing SanityDataset...")
    try:
        sanity_dataset = SanityDataset(dataset, num_samples=4, seed=42)
        print(f"   Sanity dataset created with {len(sanity_dataset)} samples")
        
        sanity_sample = sanity_dataset[0]
        print(f"   Sanity sample structure matches main dataset: {set(sample.keys()) == set(sanity_sample.keys())}")
        
    except Exception as e:
        print(f"   ERROR with SanityDataset: {e}")
    
    # Performance test
    print("\n9. Performance test...")
    try:
        import time
        
        # Time single sample access
        start_time = time.time()
        for i in range(min(10, len(dataset))):
            _ = dataset[i]
        single_sample_time = (time.time() - start_time) / min(10, len(dataset))
        print(f"   Average time per sample: {single_sample_time*1000:.2f} ms")
        
        # Time batch creation
        start_time = time.time()
        batch_samples = [dataset[i] for i in range(min(4, len(dataset)))]
        batch = collate_fn(batch_samples)
        batch_time = time.time() - start_time
        print(f"   Batch creation time: {batch_time*1000:.2f} ms")
        
    except Exception as e:
        print(f"   ERROR in performance test: {e}")
    
    print("\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY! 🎉")
    print("="*80)

if __name__ == "__main__":
    test_cellpuzzles_dataset()