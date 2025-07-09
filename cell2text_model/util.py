# import torch
# from peft import PeftModel, LoraConfig, get_peft_model
# from typing import Dict, Any
# from transformers import PretrainedConfig
# import os
# import sys
# from peft import AutoPeftModelForCausalLM

# # Import your model classes
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from cell2text_model.model import Cell2TextModel
# from cell2text_model.geneformer_encoder import GeneformerModel, GeneformerConfig
# from cell2text_model.llama_decoder import Cell2TextLlamaModel, Cell2TextLlamaConfig
# from cell2text_model.projectors import MLPProjectionLayer, PerceiverIO

# import torch
# from safetensors.torch import load_file, save_file
# import json
# import os
# import shutil
# from pathlib import Path


# def print_lora_discrepancies_fixed(model: torch.nn.Module, adapter_dir: str):
#     """
#     Compare LoRA parameters expected by PEFT (based on adapter config) 
#     with those found in adapter_model.safetensors, and print a diff.
#     """
#     adapter_path = Path(adapter_dir) / "adapter_model.safetensors"
#     config_path = Path(adapter_dir) / "adapter_config.json"
    
#     if not adapter_path.exists():
#         raise FileNotFoundError(f"Expected adapter file not found: {adapter_path}")
#     if not config_path.exists():
#         raise FileNotFoundError(f"Expected config file not found: {config_path}")

#     # Load adapter config
#     with open(config_path, 'r') as f:
#         adapter_config = json.load(f)
    
#     # Get target modules from config
#     target_modules = adapter_config.get('target_modules', [])
    
#     # Generate expected parameter names based on model structure and config
#     expected = set()
    
#     # Walk through the model and find modules that match target patterns
#     for name, module in model.named_modules():
#         # Check if this module matches any target pattern
#         module_matches = False
#         for target in target_modules:
#             if target.replace('*', '') in name or name.endswith(target.replace('*', '')):
#                 module_matches = True
#                 break
        
#         if module_matches and hasattr(module, 'weight'):
#             # For each matching module, expect lora_A and lora_B parameters
#             base_name = f"base_model.model.{name}"
#             expected.add(f"{base_name}.lora_A.default.weight")
#             expected.add(f"{base_name}.lora_B.default.weight")
#             # Also add scaling parameter if it exists
#             if adapter_config.get('use_rslora', False):
#                 expected.add(f"{base_name}.lora_magnitude_vector.default.weight")

#     # Load what's actually provided
#     provided = set(load_file(adapter_path).keys())

#     missing = sorted(expected - provided)
#     unexpected = sorted(provided - expected)

#     print("\n┌──────────────────────────────────────────┐")
#     print("│           LoRA PARAMETER DIFF (FIXED)    │")
#     print("└──────────────────────────────────────────┘")
#     print(f"Expected by PEFT config: {len(expected)} tensors")
#     print(f"Provided by file       : {len(provided)} tensors\n")

#     if missing:
#         print(f"❌  MISSING ({len(missing)}):")
#         for k in missing[:10]:  # Show first 10 to avoid spam
#             print(f"   - {k}")
#         if len(missing) > 10:
#             print(f"   ... and {len(missing) - 10} more")
#     else:
#         print("✅  No missing tensors")

#     print()

#     if unexpected:
#         print(f"⚠️  UNEXPECTED ({len(unexpected)}):")
#         for k in unexpected[:10]:  # Show first 10 to avoid spam
#             print(f"   - {k}")
#         if len(unexpected) > 10:
#             print(f"   ... and {len(unexpected) - 10} more")
#     else:
#         print("✅  No unexpected tensors")

#     print("────────────────────────────────────────────\n")


# def fix_adapter_keys_correct(adapter_dir, output_dir=None):
#     """
#     Fixed version that handles the nested PEFT structure correctly.
#     The issue is that the model expects 'base_model.model.base_model.model.' 
#     but the saved adapter has 'base_model.model.'.
#     """
#     if output_dir is None:
#         output_dir = adapter_dir + "_fixed"
    
#     # Don't re-run if it already exists
#     if os.path.exists(output_dir):
#         print(f"Fixed adapter directory already exists: {output_dir}")
#         return output_dir

#     os.makedirs(output_dir, exist_ok=True)
    
#     adapter_path = os.path.join(adapter_dir, "adapter_model.safetensors")
#     if not os.path.exists(adapter_path):
#         raise FileNotFoundError(f"adapter_model.safetensors not found in {adapter_dir}")
        
#     state_dict = load_file(adapter_path)
#     print(f"Loaded {len(state_dict)} keys from original adapter.")
    
#     new_state_dict = {}
#     for old_key, tensor in state_dict.items():
#         new_key = old_key
        
#         # MAIN FIX: Handle the nested PEFT structure
#         # Transform: base_model.model.decoder.llama.model.layers...
#         # To:        base_model.model.base_model.model.decoder.llama.model.layers...
#         if old_key.startswith("base_model.model.decoder.llama.model.layers"):
#             new_key = old_key.replace(
#                 "base_model.model.decoder.llama.model.layers",
#                 "base_model.model.base_model.model.decoder.llama.model.layers",
#                 1
#             )
        
#         # Handle other potential patterns that might exist
#         elif old_key.startswith("base_model.model.decoder.") and "base_model.model.base_model.model." not in old_key:
#             new_key = old_key.replace(
#                 "base_model.model.decoder.",
#                 "base_model.model.base_model.model.decoder.",
#                 1
#             )
        
#         if new_key != old_key:
#             print(f"Remapped: {old_key}")
#             print(f"    -> TO: {new_key}")
        
#         new_state_dict[new_key] = tensor

#     new_adapter_path = os.path.join(output_dir, "adapter_model.safetensors")
#     save_file(new_state_dict, new_adapter_path)
#     print(f"\nSaved fixed adapter to {new_adapter_path}")
    
#     # Copy other essential files
#     for filename in ["adapter_config.json", "README.md", "training_args.bin"]:
#         src_path = os.path.join(adapter_dir, filename)
#         if os.path.exists(src_path):
#             dst_path = os.path.join(output_dir, filename)
#             shutil.copy2(src_path, dst_path)
#             print(f"Copied {filename}")
            
#     return output_dir


# def fix_and_load_adapter_correct(model, adapter_dir: str, is_trainable: bool = True) -> PeftModel:
#     """
#     Corrected version that handles the nested PEFT structure issue.
#     """
#     # Determine the path for the new, fixed adapter directory
#     dir_name = os.path.basename(os.path.normpath(adapter_dir))
#     parent_dir = os.path.dirname(os.path.normpath(adapter_dir))
#     fixed_adapter_dir = os.path.join(parent_dir, f"{dir_name}_fixed")

#     print(f"Searching for adapter config in: {adapter_dir}")
#     print(f"Will save/load fixed adapter from: {fixed_adapter_dir}")
    
#     # Create the fixed adapter only if it doesn't already exist
#     if not os.path.exists(fixed_adapter_dir):
#         print("Fixed adapter not found. Creating a new one...")
#         os.makedirs(fixed_adapter_dir, exist_ok=True)
        
#         original_adapter_path = os.path.join(adapter_dir, "adapter_model.safetensors")
#         if not os.path.exists(original_adapter_path):
#             raise FileNotFoundError(f"'adapter_model.safetensors' not found in '{adapter_dir}'")
            
#         state_dict = load_file(original_adapter_path)
#         new_state_dict = {}
#         print("\nTransforming adapter keys...")

#         for old_key, tensor in state_dict.items():
#             new_key = old_key
            
#             # MAIN FIX: Handle the nested PEFT structure
#             # The issue is that saved keys have: base_model.model.decoder.llama.model.layers...
#             # But expected keys have: base_model.model.base_model.model.decoder.llama.model.layers...
#             if old_key.startswith("base_model.model.decoder.llama.model.layers"):
#                 new_key = old_key.replace(
#                     "base_model.model.decoder.llama.model.layers",
#                     "base_model.model.base_model.model.decoder.llama.model.layers",
#                     1
#                 )
            
#             # Handle other potential decoder patterns
#             elif old_key.startswith("base_model.model.decoder.") and "base_model.model.base_model.model." not in old_key:
#                 new_key = old_key.replace(
#                     "base_model.model.decoder.",
#                     "base_model.model.base_model.model.decoder.",
#                     1
#                 )

#             if old_key != new_key:
#                 print(f"  - Remapped: {old_key}")
#                 print(f"    -> TO:     {new_key}")
            
#             new_state_dict[new_key] = tensor
            
#         # Save the new state dict
#         fixed_adapter_model_path = os.path.join(fixed_adapter_dir, "adapter_model.safetensors")
#         save_file(new_state_dict, fixed_adapter_model_path)
        
#         # Copy other essential files
#         for filename in ["adapter_config.json", "README.md"]:
#             src = os.path.join(adapter_dir, filename)
#             dst = os.path.join(fixed_adapter_dir, filename)
#             if os.path.exists(src):
#                 shutil.copy2(src, dst)
        
#         print(f"\n✅ Successfully created fixed adapter at: {fixed_adapter_dir}")
    
#     else:
#         print(f"✅ Using existing fixed adapter from: {fixed_adapter_dir}")

#     # Load the adapter from the FIXED directory
#     model = PeftModel.from_pretrained(
#         model,
#         fixed_adapter_dir,
#         is_trainable=is_trainable
#     )

#     print_lora_discrepancies_fixed(model, fixed_adapter_dir)
    
#     print("\n🎉 Adapter loaded successfully onto the model!")
#     return model


# def get_target_modules_from_model(model):
#     """
#     Helper function to dynamically find the correct target modules
#     by inspecting the model structure.
#     """
#     target_modules = []
    
#     # First, let's debug what we have
#     print("=== Model structure analysis ===")
#     decoder_modules = []
#     for name, module in model.named_modules():
#         if "decoder" in name and any(target in name for target in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
#             decoder_modules.append(name)
    
#     # Show some examples
#     for name in sorted(decoder_modules)[:5]:
#         print(f"Found potential target: {name}")
#     if len(decoder_modules) > 5:
#         print(f"... and {len(decoder_modules) - 5} more")
    
#     # Check if we have a nested decoder structure
#     has_llama_decoder = any("decoder.llama" in name for name, _ in model.named_modules())
    
#     if has_llama_decoder:
#         # Structure: model.decoder.llama.model.layers.X.self_attn.{q,k,v,o}_proj
#         print("Detected LLaMA decoder structure")
#         target_modules = [
#             "decoder.llama.model.layers.*.self_attn.q_proj",
#             "decoder.llama.model.layers.*.self_attn.k_proj", 
#             "decoder.llama.model.layers.*.self_attn.v_proj",
#             "decoder.llama.model.layers.*.self_attn.o_proj",
#             "decoder.llama.model.layers.*.mlp.gate_proj",
#             "decoder.llama.model.layers.*.mlp.up_proj",
#             "decoder.llama.model.layers.*.mlp.down_proj"
#         ]
#     else:
#         # Fallback: look for actual module names
#         attention_modules = set()
#         mlp_modules = set()
        
#         for name, module in model.named_modules():
#             if hasattr(module, 'weight') and "decoder" in name:
#                 if any(proj in name for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]):
#                     # Extract the pattern up to the projection layer
#                     pattern = name.replace(name.split(".")[-1], "*")
#                     if pattern not in attention_modules:
#                         attention_modules.add(pattern[:-1])  # Remove the trailing *
                        
#                 elif any(proj in name for proj in ["gate_proj", "up_proj", "down_proj"]):
#                     pattern = name.replace(name.split(".")[-1], "*")
#                     if pattern not in mlp_modules:
#                         mlp_modules.add(pattern[:-1])  # Remove the trailing *
        
#         # Convert to specific target modules
#         for base_pattern in attention_modules:
#             for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
#                 target_modules.append(f"{base_pattern}.{proj}")
                
#         for base_pattern in mlp_modules:
#             for proj in ["gate_proj", "up_proj", "down_proj"]:
#                 target_modules.append(f"{base_pattern}.{proj}")
    
#     print(f"Final target modules: {target_modules}")
#     return target_modules


# def load_model(args: Dict[str, Any]) -> PeftModel:
#     """
#     Standard API for Cell2Text model. Used in both `train` and `generate`.
#     Load base model components, and load weights from the checkpoint path 
#     if provided.
#     """
    
#     # Create configuration for the Cell2Text model
#     config = create_cell2text_config(args)
    
#     # Load Geneformer encoder (frozen)
#     geneformer_config = GeneformerConfig(
#         emb_mode=args.get("emb_mode", "gene"),
#         max_ncells=args.get("max_ncells", 1000),
#         emb_layer=args.get("emb_layer", -1),
#         emb_label=args.get("emb_label", None),
#         nproc=args.get("nproc", -1),
#         forward_batch_size=args.get("forward_batch_size", 100),
#         summary_stat=args.get("summary_stat", None),
#         token_dictionary_path=args.get("token_dictionary_path", 
#             "/home/arism/cell2text/Geneformer/geneformer/token_dictionary_gc104M.pkl")
#     )
    
#     geneformer_encoder = GeneformerModel.from_pretrained(
#         args["geneformer_path"],
#         config=geneformer_config,
#     )
    
#     # Load LLaMA decoder
#     llama_config = Cell2TextLlamaConfig(
#         max_length=args.get("max_length", 100),
#         num_beams=args.get("num_beams", 1),
#         early_stopping=args.get("early_stopping", True),
#         no_repeat_ngram_size=args.get("no_repeat_ngram_size", 3),
#         temperature=args.get("temperature", 1.0),
#         top_p=args.get("top_p", 1.0)
#     )
    
#     llama_decoder = Cell2TextLlamaModel.from_pretrained(
#         args["llama_path"],
#         config=llama_config,
#     )
    
#     # Create projector/adapter
#     if args["projector"] == "mlp":
#         adapter = MLPProjectionLayer(
#             input_dim=args["cell_encoder_hidden_size"],
#             hidden_dim=args["mlp_hidden_size"],
#             output_dim=args["decoder_hidden_size"],
#             dropout_prob=args["mlp_dropout"],
#             bias=True
#         )
#     elif args["projector"] == "perceiver":
#         adapter = PerceiverIO(
#             input_dim=args["cell_encoder_hidden_size"],
#             output_dim=args["decoder_hidden_size"],
#             num_latents=args["num_latents"],
#             num_cross_attn_layers=args["perceiver_cross_attn_layers"],
#             num_heads=args["perceiver_num_heads"],
#             ff_mult=args["ff_mult"],
#             dropout=args["perceiver_dropout"],
#             use_position_encoding=args["use_position_encoding"]
#         )
#     else:
#         raise ValueError(f"Unknown projector type: {args['projector']}")
        
#     # 1. Create the BASE model (NOT a PeftModel yet)
#     model = Cell2TextModel(config)
#     model.cell_encoder = geneformer_encoder
#     model.decoder = llama_decoder
#     model.cell_to_embedding = adapter
    
#     for param in model.cell_encoder.parameters():
#         param.requires_grad = False
    
#     if args.get("load_model_checkpoint_path"):
#         ckpt_path = args["load_model_checkpoint_path"]
#         print(f"Loading *ONLY* projector weights from {ckpt_path}")

#         full_sd = torch.load(ckpt_path, map_location="cpu")

#         # 1) keep only the projector tensors
#         proj_sd = {k[len("cell_to_embedding."):] : v          # strip the prefix
#                 for k, v in full_sd.items()
#                 if k.startswith("cell_to_embedding.")}

#         print(f"  ↳ {len(proj_sd)} tensors will be loaded into cell_to_embedding")

#         # 2) hand the sub‑dict to the projector module itself
#         missing, unexpected = model.cell_to_embedding.load_state_dict(proj_sd, strict=False)

#         if unexpected:
#             print(f"The missing keys {missing}")
#             raise ValueError(f"Still have unexpected keys: {unexpected}")
#         print(f"  ✓ loaded {len(proj_sd) - len(missing)} tensors ("
#             f"{len(missing)} params left at init values)")

    
#     # Set up LoRA adaptation
#     if args.get("load_adapter_checkpoint_dir"):
#         print("--- Loading and Fixing LoRA Adapter ---")
        
#         # Use the corrected function
#         model.decoder = fix_and_load_adapter_correct(
#             model=model.decoder, 
#             adapter_dir=args['load_adapter_checkpoint_dir']
#         )
#         print("\nSuccessfully loaded fixed LoRA adapter.")

#     else:
#         print("Initializing LoRA adapter")
        
#         # Get the correct target modules by inspecting the model structure
#         target_modules = get_target_modules_from_model(model)
#         print(f"Target modules found: {target_modules}")
        
#         # Define modules to save (projector/adapter parameters)
#         modules_to_save = None
#         if not args.get("fix_modality_adapter", False):
#             if args["projector"] == "mlp":
#                 modules_to_save = get_mlp_modules_to_save(args)
#             elif args["projector"] == "perceiver":
#                 modules_to_save = [
#                     "cell_to_embedding.latents",
#                     "cell_to_embedding.input_projection",
#                     "cell_to_embedding.cross_attention_layers",
#                     "cell_to_embedding.self_attention_layers",
#                     "cell_to_embedding.final_norm",
#                     "cell_to_embedding.position_encoding"
#                 ]
        
#         lora_config = LoraConfig(
#             r=args["lora_rank"],
#             lora_alpha=args["lora_rank"] * 2,
#             lora_dropout=0.1,
#             bias="none",
#             init_lora_weights=True,
#             target_modules=target_modules,
#             modules_to_save=modules_to_save
#         )
        
#         model = get_peft_model(model, lora_config)
    
#     return model


# def get_mlp_modules_to_save(args: Dict[str, Any]) -> list:
#     """
#     Get the correct module names for MLP projector based on dropout configuration.
#     """
#     dropout_prob = args.get("mlp_dropout", 0.0)
    
#     if dropout_prob > 0:
#         return [
#             "cell_to_embedding.projection.0",  # First Linear layer
#             "cell_to_embedding.projection.1",  # First LayerNorm
#             "cell_to_embedding.projection.4",  # Second Linear layer
#             "cell_to_embedding.projection.5",  # Final LayerNorm
#         ]
#     else:
#         return [
#             "cell_to_embedding.projection.0",  # First Linear layer
#             "cell_to_embedding.projection.1",  # First LayerNorm
#             "cell_to_embedding.projection.3",  # Second Linear layer
#             "cell_to_embedding.projection.4",  # Final LayerNorm
#         ]


# def create_cell2text_config(args: Dict[str, Any]) -> PretrainedConfig:
#     """
#     Create a configuration object for Cell2Text model from args dictionary.
#     """
#     config = PretrainedConfig()
    
#     # Basic model configuration
#     config.projector = args["projector"]
#     config.cell_encoder_hidden_size = args["cell_encoder_hidden_size"]
#     config.decoder_hidden_size = args["decoder_hidden_size"]
#     config.top_k = args["top_k"]
    
#     # MLP projector configuration
#     config.mlp_hidden_size = args["mlp_hidden_size"]
#     config.mlp_dropout = args["mlp_dropout"]
    
#     # Perceiver projector configuration
#     config.num_latents = args["num_latents"]
#     config.perceiver_cross_attn_layers = args["perceiver_cross_attn_layers"]
#     config.perceiver_num_heads = args["perceiver_num_heads"]
#     config.ff_mult = args["ff_mult"]
#     config.perceiver_dropout = args["perceiver_dropout"]
#     config.use_position_encoding = args["use_position_encoding"]
    
#     # Geneformer configuration
#     config.emb_mode = args.get("emb_mode", "gene")
#     config.max_ncells = args.get("max_ncells", 1000)
#     config.emb_layer = args.get("emb_layer", -1)
#     config.emb_label = args.get("emb_label", None)
#     config.nproc = args.get("nproc", -1)
#     config.forward_batch_size = args.get("forward_batch_size", 100)
#     config.summary_stat = args.get("summary_stat", None)
#     config.token_dictionary_path = args.get("token_dictionary_path",
#         "/home/arism/cell2text/Geneformer/geneformer/token_dictionary_gc104M.pkl")
    
#     # LLaMA decoder configuration
#     config.max_length = args.get("max_length", 100)
#     config.num_beams = args.get("num_beams", 4)
#     config.early_stopping = args.get("early_stopping", True)
#     config.no_repeat_ngram_size = args.get("no_repeat_ngram_size", 3)
#     config.temperature = args.get("temperature", 1.0)
#     config.top_p = args.get("top_p", 1.0)
    
#     return config

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


def load_model(args: Dict[str, Any]) -> Cell2TextModel:
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
        print(f'USE_POSITION_ENCODINGS: {args["use_position_encoding"]}')
    else:
        raise ValueError(f"Unknown projector type: {args['projector']}")
        
    # Create the base model
    model = Cell2TextModel(config)
    model.cell_encoder = geneformer_encoder
    model.decoder = llama_decoder
    model.cell_to_embedding = adapter
    
    # Freeze encoder
    for param in model.cell_encoder.parameters():
        param.requires_grad = False
    
    # Load projector weights if provided
    if args.get("load_model_checkpoint_path"):
        load_projector_weights(model, args["load_model_checkpoint_path"])
    
    # Apply LoRA to decoder
    if args.get("load_adapter_checkpoint_dir"):
        print("Loading existing LoRA adapter...")
        model.decoder = PeftModel.from_pretrained(
            model.decoder, 
            args["load_adapter_checkpoint_dir"]
        )
    else:
        print("Initializing new LoRA adapter...")
        model.decoder = create_lora_adapter(model.decoder, args)
    
    return model


def load_projector_weights(model: Cell2TextModel, checkpoint_path: str):
    """Load only the projector weights from checkpoint"""
    print(f"Loading projector weights from {checkpoint_path}")
    
    full_state_dict = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract only projector weights
    projector_state_dict = {
        k[len("cell_to_embedding."):]: v 
        for k, v in full_state_dict.items() 
        if k.startswith("cell_to_embedding.")
    }
    
    print(f"Loading {len(projector_state_dict)} projector parameters")
    
    # Load into projector
    missing, unexpected = model.cell_to_embedding.load_state_dict(
        projector_state_dict, strict=False
    )
    
    # Debug: print the missing parameter
    if missing:
        print(f"Missing parameter(s): {missing}")
    
    if unexpected:
        raise ValueError(f"Unexpected keys in projector state dict: {unexpected}")
    
    print(f"✓ Loaded {len(projector_state_dict) - len(missing)} parameters")


def create_lora_adapter(decoder_model, args: Dict[str, Any]) -> PeftModel:
    """Create a new LoRA adapter for the decoder"""
    
    # Get target modules from args or use defaults
    target_modules = args.get("lora_target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=args.get("lora_rank", 8),
        lora_alpha=args.get("lora_alpha", 16),
        lora_dropout=args.get("lora_dropout", 0.1),
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA to decoder
    peft_model = get_peft_model(decoder_model, lora_config)
    
    print(f"LoRA applied to decoder with rank={lora_config.r}")
    print_trainable_parameters(peft_model)
    
    return peft_model


def print_trainable_parameters(model):
    """Print the number of trainable parameters in the model"""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")


def create_cell2text_config(args: Dict[str, Any]) -> PretrainedConfig:
    """Create a configuration object for Cell2Text model from args dictionary"""
    config = PretrainedConfig()
    
    # Basic model configuration
    config.projector = args["projector"]
    config.cell_encoder_hidden_size = args["cell_encoder_hidden_size"]
    config.decoder_hidden_size = args["decoder_hidden_size"]
    config.top_k = args["top_k"]
    
    # MLP projector configuration
    config.mlp_hidden_size = args["mlp_hidden_size"]
    config.mlp_dropout = args["mlp_dropout"]
    
    # Perceiver projector configuration
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