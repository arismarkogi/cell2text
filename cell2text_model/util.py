import torch
from peft import PeftModel, LoraConfig, get_peft_model
from typing import Dict, Any
from transformers import PretrainedConfig
import os
import sys
from peft import AutoPeftModelForCausalLM

# Import your model classes
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cell2text_model.model import Cell2TextModel
from cell2text_model.geneformer_encoder import GeneformerModel, GeneformerConfig
from cell2text_model.llama_decoder import Cell2TextLlamaModel, Cell2TextLlamaConfig
from cell2text_model.projectors import MLPProjectionLayer, PerceiverIO


import torch
from safetensors.torch import load_file, save_file
import json
import os
import shutil

import os
import torch
from safetensors.torch import load_file, save_file
import shutil
from peft import PeftModel

def fix_and_load_adapter(model, adapter_dir: str, is_trainable: bool = True) -> PeftModel:
    """
    Checks for key mismatches in a LoRA adapter, fixes them by creating a new 
    corrected adapter directory, and loads it onto the base model.

    Args:
        model: The base PyTorch model (non-PEFT).
        adapter_dir: Path to the original LoRA adapter directory.
        is_trainable: Whether the loaded adapter should be trainable.

    Returns:
        A PeftModel with the correctly loaded adapter.
    """
    # Determine the path for the new, fixed adapter directory
    dir_name = os.path.basename(os.path.normpath(adapter_dir))
    parent_dir = os.path.dirname(os.path.normpath(adapter_dir))
    fixed_adapter_dir = os.path.join(parent_dir, f"{dir_name}_fixed")

    print(f"Searching for adapter config in: {adapter_dir}")
    print(f"Will save/load fixed adapter from: {fixed_adapter_dir}")
    
    # Create the fixed adapter only if it doesn't already exist
    if not os.path.exists(fixed_adapter_dir):
        print("Fixed adapter not found. Creating a new one...")
        os.makedirs(fixed_adapter_dir, exist_ok=True)
        
        original_adapter_path = os.path.join(adapter_dir, "adapter_model.safetensors")
        if not os.path.exists(original_adapter_path):
            raise FileNotFoundError(f"'adapter_model.safetensors' not found in '{adapter_dir}'")
            
        state_dict = load_file(original_adapter_path)
        new_state_dict = {}
        print("\nTransforming adapter keys...")

        for old_key, tensor in state_dict.items():
            new_key = old_key
            
            # Transformation 1: Add the 'decoder.' path component.
            # Turns 'base_model.model.llama...' into 'base_model.model.decoder.llama...'
            if old_key.startswith("base_model.model.llama.model."):
                new_key = old_key.replace("base_model.model.llama.model.", "base_model.model.decoder.llama.model.", 1)
            
            # Transformation 2: Add '.default' to LoRA weight names.
            # Turns '...lora_A.weight' into '...lora_A.default.weight'
            if ".lora_A.weight" in new_key or ".lora_B.weight" in new_key:
                new_key = new_key.replace(".weight", ".default.weight")

            if old_key != new_key:
                print(f"  - Remapped: {old_key}\n    -> TO:     {new_key}")
            
            new_state_dict[new_key] = tensor
            
        # Save the new state dict
        fixed_adapter_model_path = os.path.join(fixed_adapter_dir, "adapter_model.safetensors")
        save_file(new_state_dict, fixed_adapter_model_path)
        
        # Copy other essential files
        for filename in ["adapter_config.json", "README.md"]:
            src = os.path.join(adapter_dir, filename)
            dst = os.path.join(fixed_adapter_dir, filename)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        
        print(f"\n✅ Successfully created fixed adapter at: {fixed_adapter_dir}")
    
    else:
        print(f"✅ Using existing fixed adapter from: {fixed_adapter_dir}")

    # Load the adapter from the FIXED directory
    model = PeftModel.from_pretrained(
        model,
        fixed_adapter_dir,
        is_trainable=is_trainable
    )
    print("\n🎉 Adapter loaded successfully onto the model!")
    return model



def fix_adapter_keys_exact(adapter_dir, output_dir=None):
    """
    Fixes the exact key mismatch issue. This function is correct.
    It transforms saved keys to the format the current model expects.
    """
    if output_dir is None:
        output_dir = adapter_dir + "_fixed"
    
    # Don't re-run if it already exists
    if os.path.exists(output_dir):
        print(f"Fixed adapter directory already exists: {output_dir}")
        return output_dir

    os.makedirs(output_dir, exist_ok=True)
    
    adapter_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"adapter_model.safetensors not found in {adapter_dir}")
        
    state_dict = load_file(adapter_path)
    print(f"Loaded {len(state_dict)} keys from original adapter.")
    
    new_state_dict = {}
    for old_key, tensor in state_dict.items():
        new_key = old_key
        # Check 1: Add the '.decoder.' path segment
        if old_key.startswith("base_model.model.llama.model.layers"):
            new_key = old_key.replace(
                "base_model.model.llama.model.layers",
                "base_model.model.decoder.llama.model.layers"
            )
        
        # Check 2: Add the '.default.' segment for LoRA weights
        if ".lora_A.weight" in new_key or ".lora_B.weight" in new_key:
            new_key = new_key.replace(".weight", ".default.weight")
        
        if new_key != old_key:
            print(f"Remapped: {old_key} -> {new_key}")
        
        new_state_dict[new_key] = tensor

    new_adapter_path = os.path.join(output_dir, "adapter_model.safetensors")
    save_file(new_state_dict, new_adapter_path)
    print(f"\nSaved fixed adapter to {new_adapter_path}")
    
    # Copy other essential files
    for filename in ["adapter_config.json", "README.md", "training_args.bin"]:
        src_path = os.path.join(adapter_dir, filename)
        if os.path.exists(src_path):
            dst_path = os.path.join(output_dir, filename)
            shutil.copy2(src_path, dst_path)
            print(f"Copied {filename}")
            
    return output_dir


def load_model(args: Dict[str, Any]) -> PeftModel:
    """
    Standard API for Cell2Text model. Used in both `train` and `generate`.
    Load base model components, and load weights from the checkpoint path 
    if provided.
    
    Args:
        args: Dictionary containing model configuration with keys:
            - geneformer_path: Path to pretrained Geneformer model
            - llama_path: Path to pretrained LLaMA model
            - projector: Type of projector ("mlp" or "perceiver")
            - cell_encoder_hidden_size: Hidden size of cell encoder
            - decoder_hidden_size: Hidden size of decoder
            - mlp_hidden_size: Hidden size of MLP projector (if using MLP)
            - mlp_dropout: Dropout for MLP projector
            - top_k: Number of top tokens to keep from encoder
            - load_model_checkpoint_path: Path to pretrained Cell2Text model checkpoint
            - load_adapter_checkpoint_dir: Path to pretrained LoRA adapter
            - lora_rank: Rank for LoRA adaptation
            - fix_modality_adapter: Whether to freeze the projector/adapter
            - Additional projector-specific args for Perceiver
    """
    
    # Create configuration for the Cell2Text model
    config = create_cell2text_config(args)
    
    # Load Geneformer encoder (frozen)
    geneformer_config = GeneformerConfig(
        emb_mode=args.get("emb_mode", "gene"),
        max_ncells=args.get("max_ncells", 1000),
        emb_layer=args.get("emb_layer", -1),
        emb_label=args.get("emb_label", None),
        nproc=args.get("nproc", -1),
        forward_batch_size=args.get("forward_batch_size", 100),
        summary_stat=args.get("summary_stat", None),
        token_dictionary_path=args.get("token_dictionary_path", 
            "/home/arism/cell2text/Geneformer/geneformer/token_dictionary_gc104M.pkl")
    )
    
    geneformer_encoder = GeneformerModel.from_pretrained(
        args["geneformer_path"],
        config=geneformer_config,
    )
    
    # Load LLaMA decoder
    llama_config = Cell2TextLlamaConfig(
        max_length=args.get("max_length", 100),
        num_beams=args.get("num_beams", 1),
        early_stopping=args.get("early_stopping", True),
        no_repeat_ngram_size=args.get("no_repeat_ngram_size", 3),
        temperature=args.get("temperature", 1.0),
        top_p=args.get("top_p", 1.0)
    )
    
    llama_decoder = Cell2TextLlamaModel.from_pretrained(
        args["llama_path"],
        config=llama_config,
    )
    
    # Create projector/adapter
    if args["projector"] == "mlp":
        adapter = MLPProjectionLayer(
            input_dim=args["cell_encoder_hidden_size"],
            hidden_dim=args["mlp_hidden_size"],
            output_dim=args["decoder_hidden_size"],
            dropout_prob=args["mlp_dropout"],
            bias=True
        )
    elif args["projector"] == "perceiver":
        adapter = PerceiverIO(
            input_dim=args["cell_encoder_hidden_size"],
            output_dim=args["decoder_hidden_size"],
            num_latents=args["num_latents"],
            num_cross_attn_layers=args["perceiver_cross_attn_layers"],
            num_heads=args["perceiver_num_heads"],
            ff_mult=args["ff_mult"],
            dropout=args["perceiver_dropout"],
            use_position_encoding=args["use_position_encoding"]
        )
    else:
        raise ValueError(f"Unknown projector type: {args['projector']}")
        
    # 1. Create the BASE model (NOT a PeftModel yet)
    model = Cell2TextModel(config)
    model.cell_encoder = geneformer_encoder
    model.decoder = llama_decoder
    model.cell_to_embedding = adapter
    
    for param in model.cell_encoder.parameters():
        param.requires_grad = False
    
    if args.get("load_model_checkpoint_path"):
        print(f"Loading base model weights from {args['load_model_checkpoint_path']}")
        state_dict = torch.load(args["load_model_checkpoint_path"], map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    
    
    # Set up LoRA adaptation
    if args.get("load_adapter_checkpoint_dir"):
        # print("--- Loading and Fixing LoRA Adapter ---")
        
        # # 2a. Run the fixing script to create a corrected adapter directory
        # original_adapter_dir = args['load_adapter_checkpoint_dir']
        # fixed_adapter_dir = fix_adapter_keys_exact(original_adapter_dir)

        # # 2b. Load the adapter from the FIXED directory onto the BASE model
        # model = PeftModel.from_pretrained(
        #     model,
        #     fixed_adapter_dir,
        #     is_trainable=True
        # )

        model = fix_and_load_adapter(
            model=model, 
            adapter_dir=args['load_adapter_checkpoint_dir']
        )
        print("\nSuccessfully loaded fixed LoRA adapter.")

    else:
        print("Initializing LoRA adapter")
        
        # Get the correct target modules by inspecting the model structure
        target_modules = get_target_modules_from_model(model)
        print(f"Target modules found: {target_modules}")
        
        # Define modules to save (projector/adapter parameters)
        modules_to_save = None
        if not args.get("fix_modality_adapter", False):
            if args["projector"] == "mlp":
                modules_to_save = get_mlp_modules_to_save(args)
            elif args["projector"] == "perceiver":
                modules_to_save = [
                    "cell_to_embedding.latents",
                    "cell_to_embedding.input_projection",
                    "cell_to_embedding.cross_attention_layers",
                    "cell_to_embedding.self_attention_layers",
                    "cell_to_embedding.final_norm",
                    "cell_to_embedding.position_encoding"
                ]
        
        lora_config = LoraConfig(
            r=args["lora_rank"],
            lora_alpha=args["lora_rank"] * 2,
            lora_dropout=0.1,
            bias="none",
            init_lora_weights=True,
            target_modules=target_modules,
            modules_to_save=modules_to_save
        )
        
        model = get_peft_model(model, lora_config)
    
    return model


def get_target_modules_from_model(model):
    """
    Helper function to dynamically find the correct target modules
    by inspecting the model structure.
    """
    target_modules = []
    
    # First, let's debug what we have
    print("=== Model structure analysis ===")
    for name, module in model.named_modules():
        if "decoder" in name and any(target in name for target in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
            print(f"Found potential target: {name}")
    
    # Check if we have a nested decoder structure
    has_llama_decoder = any("decoder.llama" in name for name, _ in model.named_modules())
    
    if has_llama_decoder:
        # Structure: model.decoder.llama.model.layers.X.self_attn.{q,k,v,o}_proj
        print("Detected LLaMA decoder structure")
        target_modules = [
            "decoder.llama.model.layers.*.self_attn.q_proj",
            "decoder.llama.model.layers.*.self_attn.k_proj", 
            "decoder.llama.model.layers.*.self_attn.v_proj",
            "decoder.llama.model.layers.*.self_attn.o_proj",
            "decoder.llama.model.layers.*.mlp.gate_proj",
            "decoder.llama.model.layers.*.mlp.up_proj",
            "decoder.llama.model.layers.*.mlp.down_proj"
        ]
    else:
        # Fallback: look for actual module names
        attention_modules = set()
        mlp_modules = set()
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and "decoder" in name:
                if any(proj in name for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]):
                    # Extract the pattern up to the projection layer
                    pattern = name.replace(name.split(".")[-1], "*")
                    if pattern not in attention_modules:
                        attention_modules.add(pattern[:-1])  # Remove the trailing *
                        
                elif any(proj in name for proj in ["gate_proj", "up_proj", "down_proj"]):
                    pattern = name.replace(name.split(".")[-1], "*")
                    if pattern not in mlp_modules:
                        mlp_modules.add(pattern[:-1])  # Remove the trailing *
        
        # Convert to specific target modules
        for base_pattern in attention_modules:
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                target_modules.append(f"{base_pattern}.{proj}")
                
        for base_pattern in mlp_modules:
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                target_modules.append(f"{base_pattern}.{proj}")
    
    print(f"Final target modules: {target_modules}")
    return target_modules


def debug_model_structure_detailed(model):
    """
    Enhanced debug function to understand the exact model structure
    """
    print("=== DETAILED MODEL STRUCTURE ===")
    decoder_modules = []
    
    for name, module in model.named_modules():
        if "decoder" in name:
            decoder_modules.append((name, type(module).__name__))
            
    # Sort by depth and name
    decoder_modules.sort(key=lambda x: (x[0].count('.'), x[0]))
    
    for name, module_type in decoder_modules:
        depth = name.count('.')
        indent = "  " * depth
        print(f"{indent}{name}: {module_type}")
        
        # Highlight projection layers
        if any(proj in name for proj in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
            print(f"{indent}  ⭐ TARGET CANDIDATE")
    
    print("=== END STRUCTURE ===")

def debug_model_structure(model, max_depth=3):
    """
    Debug function to print the model structure and help identify correct target modules.
    """
    print("Model structure:")
    for name, module in model.named_modules():
        depth = name.count('.')
        if depth <= max_depth:
            indent = "  " * depth
            print(f"{indent}{name}: {type(module).__name__}")
            
            # Specifically look for attention and MLP components
            if any(target in name for target in ["q_proj", "k_proj", "v_proj", "o_proj", 
                                               "gate_proj", "up_proj", "down_proj"]):
                print(f"{indent}  -> TARGET MODULE FOUND")


def get_mlp_modules_to_save(args: Dict[str, Any]) -> list:
    """
    Get the correct module names for MLP projector based on dropout configuration.
    
    MLP structure:
    - Linear(input_dim, hidden_dim)     # index 0
    - LayerNorm(hidden_dim)            # index 1  
    - GELU()                           # index 2
    - [Dropout(p=dropout_prob)]        # index 3 (if dropout > 0)
    - Linear(hidden_dim, output_dim)   # index 3 or 4
    - LayerNorm(output_dim)            # index 4 or 5
    """
    dropout_prob = args.get("mlp_dropout", 0.0)
    
    if dropout_prob > 0:
        # With dropout: Linear(0), LayerNorm(1), GELU(2), Dropout(3), Linear(4), LayerNorm(5)
        return [
            "cell_to_embedding.projection.0",  # First Linear layer
            "cell_to_embedding.projection.1",  # First LayerNorm
            "cell_to_embedding.projection.4",  # Second Linear layer
            "cell_to_embedding.projection.5",  # Final LayerNorm
        ]
    else:
        # Without dropout: Linear(0), LayerNorm(1), GELU(2), Linear(3), LayerNorm(4)
        return [
            "cell_to_embedding.projection.0",  # First Linear layer
            "cell_to_embedding.projection.1",  # First LayerNorm
            "cell_to_embedding.projection.3",  # Second Linear layer
            "cell_to_embedding.projection.4",  # Final LayerNorm
        ]


def create_cell2text_config(args: Dict[str, Any]) -> PretrainedConfig:
    """
    Create a configuration object for Cell2Text model from args dictionary.
    """
    config = PretrainedConfig()
    
    # Basic model configuration
    config.projector = args["projector"]
    config.cell_encoder_hidden_size = args["cell_encoder_hidden_size"]
    config.decoder_hidden_size = args["decoder_hidden_size"]
    config.top_k = args["top_k"]
    
    # MLP projector configuration
    if args["projector"] == "mlp":
        config.mlp_hidden_size = args["mlp_hidden_size"]
        config.mlp_dropout = args["mlp_dropout"]
    
    # Perceiver projector configuration
    elif args["projector"] == "perceiver":
        config.num_latents = args["num_latents"]
        config.perceiver_cross_attn_layers = args["perceiver_cross_attn_layers"]
        config.perceiver_num_heads = args["perceiver_num_heads"]
        config.ff_mult = args["ff_mult"]
        config.perceiver_dropout = args["perceiver_dropout"]
        config.use_position_encoding = args["use_position_encoding"]
    
    # Geneformer configuration
    config.emb_mode = args.get("emb_mode", "gene")
    config.max_ncells = args.get("max_ncells", 1000)
    config.emb_layer = args.get("emb_layer", -1)
    config.emb_label = args.get("emb_label", None)
    config.nproc = args.get("nproc", -1)
    config.forward_batch_size = args.get("forward_batch_size", 100)
    config.summary_stat = args.get("summary_stat", None)
    config.token_dictionary_path = args.get("token_dictionary_path",
        "/home/arism/cell2text/Geneformer/geneformer/token_dictionary_gc104M.pkl")
    
    # LLaMA decoder configuration
    config.max_length = args.get("max_length", 100)
    config.num_beams = args.get("num_beams", 4)
    config.early_stopping = args.get("early_stopping", True)
    config.no_repeat_ngram_size = args.get("no_repeat_ngram_size", 3)
    config.temperature = args.get("temperature", 1.0)
    config.top_p = args.get("top_p", 1.0)
    
    return config