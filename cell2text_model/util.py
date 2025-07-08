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


def load_model(args: Dict[str, Any]) -> PeftModel:
    """
    Standard API for Cell2Text model. Used in both `train` and `generate`.
    Load base model components, and load weights from the checkpoint path 
    if provided.
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
        
    # Create the full Cell2Text model
    model = Cell2TextModel(config)
    model.cell_encoder = geneformer_encoder
    model.decoder = llama_decoder
    model.cell_to_embedding = adapter
    
    # Freeze the Geneformer encoder
    for param in model.cell_encoder.parameters():
        param.requires_grad = False
    
    debug_model_structure_detailed(model)
    
    # Overwrite weights of base model if checkpoint path is provided
    if args.get("load_model_checkpoint_path"):
        print(f"Loading {args['load_model_checkpoint_path']}")
        model_state_dict = torch.load(
            args["load_model_checkpoint_path"],
            weights_only=True,
            map_location="cpu"
        )
        # Handle different checkpoint formats
        if "model_state_dict" in model_state_dict:
            state_dict = model_state_dict["model_state_dict"]
        elif "state_dict" in model_state_dict:
            state_dict = model_state_dict["state_dict"]
        else:
            state_dict = model_state_dict
        
        model.load_state_dict(state_dict, strict=False)
    
    # Set up LoRA adaptation
    if args.get("load_adapter_checkpoint_dir"):
        print(f"Loading LoRA adapter from {args['load_adapter_checkpoint_dir']}")
        model = PeftModel.from_pretrained(
            model,
            args["load_adapter_checkpoint_dir"],
            is_trainable=True
        )
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
    
    # Find all relevant projection layers
    projection_layers = []
    for name, module in model.named_modules():
        if "decoder" in name and any(target in name for target in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
            projection_layers.append(name)
            print(f"Found projection layer: {name}")
    
    # Extract the exact module names (no wildcards)
    target_modules = projection_layers
    
    # If we found specific layers, use them directly
    if target_modules:
        print(f"Using specific target modules: {target_modules}")
        return target_modules
    
    # Fallback: use pattern matching
    print("No specific modules found, using pattern matching")
    
    # Check if we have a nested decoder structure
    has_llama_decoder = any("decoder.llama" in name for name, _ in model.named_modules())
    
    if has_llama_decoder:
        print("Detected LLaMA decoder structure")
        # Find the actual layer structure
        layer_pattern = None
        for name, module in model.named_modules():
            if "decoder.llama.model.layers" in name and "self_attn.q_proj" in name:
                # Extract pattern like "decoder.llama.model.layers.0.self_attn.q_proj"
                layer_num = name.split("layers.")[1].split(".")[0]
                layer_pattern = f"decoder.llama.model.layers.{layer_num}"
                break
        
        if layer_pattern:
            # Get all layer numbers
            layer_nums = set()
            for name, module in model.named_modules():
                if "decoder.llama.model.layers" in name:
                    try:
                        layer_num = name.split("layers.")[1].split(".")[0]
                        layer_nums.add(int(layer_num))
                    except:
                        continue
            
            # Create target modules for all layers
            for layer_num in sorted(layer_nums):
                base_path = f"decoder.llama.model.layers.{layer_num}"
                target_modules.extend([
                    f"{base_path}.self_attn.q_proj",
                    f"{base_path}.self_attn.k_proj", 
                    f"{base_path}.self_attn.v_proj",
                    f"{base_path}.self_attn.o_proj",
                    f"{base_path}.mlp.gate_proj",
                    f"{base_path}.mlp.up_proj",
                    f"{base_path}.mlp.down_proj"
                ])
    
    print(f"Final target modules: {target_modules}")
    return target_modules


def debug_model_structure_detailed(model):
    """
    Enhanced debug function to understand the exact model structure
    """
    print("=== DETAILED MODEL STRUCTURE ===")
    decoder_modules = []
    
    for name, module in model.named_modules():
        if "decoder" in name or "cell_to_embedding" in name:
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


def fix_adapter_keys(adapter_checkpoint_dir: str, output_dir: str = None):
    """
    Fix the adapter keys to match the expected format.
    This function converts keys from the old format to the new format.
    
    Args:
        adapter_checkpoint_dir: Directory containing the adapter checkpoint
        output_dir: Directory to save the fixed adapter (if None, overwrites original)
    """
    import safetensors
    from safetensors.torch import load_file, save_file
    import json
    
    if output_dir is None:
        output_dir = adapter_checkpoint_dir
    
    # Load the adapter model
    adapter_path = os.path.join(adapter_checkpoint_dir, "adapter_model.safetensors")
    if not os.path.exists(adapter_path):
        print(f"No adapter_model.safetensors found at {adapter_path}")
        return
    
    print(f"Loading adapter from {adapter_path}")
    state_dict = load_file(adapter_path)
    
    # Create key mapping
    new_state_dict = {}
    for old_key, tensor in state_dict.items():
        # Convert keys from old format to new format
        if old_key.startswith("base_model.model.llama.model.layers"):
            # Add decoder prefix and .default suffix
            new_key = old_key.replace("base_model.model.llama.model.layers", 
                                    "base_model.model.decoder.llama.model.layers")
            # Add .default before .weight
            new_key = new_key.replace(".weight", ".default.weight")
            new_state_dict[new_key] = tensor
            print(f"Converted: {old_key} -> {new_key}")
        else:
            # Keep other keys as-is (like cell_to_embedding keys)
            new_state_dict[old_key] = tensor
            print(f"Kept: {old_key}")
    
    # Save the fixed adapter
    output_path = os.path.join(output_dir, "adapter_model.safetensors")
    save_file(new_state_dict, output_path)
    print(f"Fixed adapter saved to {output_path}")
    
    # Copy other files
    for filename in ["adapter_config.json", "README.md"]:
        src_path = os.path.join(adapter_checkpoint_dir, filename)
        if os.path.exists(src_path):
            dst_path = os.path.join(output_dir, filename)
            if src_path != dst_path:
                import shutil
                shutil.copy2(src_path, dst_path)
                print(f"Copied {filename}")


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
    """
    dropout_prob = args.get("mlp_dropout", 0.0)
    
    if dropout_prob > 0:
        return [
            "cell_to_embedding.projection.0",  # First Linear layer
            "cell_to_embedding.projection.1",  # First LayerNorm
            "cell_to_embedding.projection.4",  # Second Linear layer
            "cell_to_embedding.projection.5",  # Final LayerNorm
        ]
    else:
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

