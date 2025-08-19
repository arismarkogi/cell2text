import torch
from collections import defaultdict
from typing import Dict, Any

def explore_checkpoint(checkpoint_path: str):
    """
    Loads a PyTorch checkpoint and provides a structured overview of its parameters.
    
    Args:
        checkpoint_path (str): The path to the PyTorch checkpoint file.
    """
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        print("❌ Error: Invalid checkpoint path provided.")
        return
        
    try:
        print(f"🔬 Exploring checkpoint at: {checkpoint_path}\n")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        
        # Group parameters by their top-level module
        param_groups = defaultdict(list)
        for name, param in state_dict.items():
            # Get the top-level module name
            module_name = name.split('.')[0]
            param_groups[module_name].append({
                "name": name, 
                "shape": tuple(param.shape), 
                "dtype": str(param.dtype)
            })

        print(f"✔️ Found {len(state_dict)} parameters in total, grouped by module:")
        print("-" * 50)

        # Print the overview
        for module, params in param_groups.items():
            print(f"📦 Module: {module} ({len(params)} parameters)")
            for param in params:
                print(f"  - {param['name']:<50} | Shape: {str(param['shape']):<25} | Dtype: {param['dtype']}")
            print("-" * 50)
            
    except FileNotFoundError:
        print(f"❌ Error: Checkpoint file not found at {checkpoint_path}")
    except Exception as e:
        print(f"❌ An error occurred while loading the checkpoint: {e}")

# Example usage:
explore_checkpoint("/home/arism/deepspeed_training/qformer_queries32_biobertbase_crossattn2_freezebertTrue_lorar32/qformer_queries32_biobertbase_crossattn2_freezebertTrue_lorar32_best_model/training_state.pt")