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
#     # Apply LoRA to decoder only
#     if args.get("load_adapter_checkpoint_dir"):
#         print("--- Loading LoRA Adapter for Decoder Only ---")
        
#         # First apply LoRA to decoder
#         target_modules = get_target_modules_from_model(model.decoder)
        
#         lora_config = LoraConfig(
#             r=args["lora_rank"],
#             lora_alpha=args["lora_rank"] * 2,
#             lora_dropout=0.1,
#             bias="none",
#             init_lora_weights=True,
#             target_modules=target_modules,
#         )
        
#         # Apply LoRA to decoder
#         model.decoder = get_peft_model(model.decoder, lora_config)
        
#         # Now load the adapter weights
#         model.decoder = PeftModel.from_pretrained(
#             model.decoder.get_base_model(),  # Get the base model
#             args['load_adapter_checkpoint_dir'],
#             is_trainable=True
#         )
        
#         print("✅ LoRA adapter loaded successfully to decoder only!")

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


"""
Utilities for loading Cell2Text model with optional LoRA adapters.

Key features
------------
* `load_model` – end‑to‑end factory that builds the encoder/decoder, applies
  projector and LoRA adapter and returns a `PeftModel`.
* `fix_and_load_adapter` – rewrites a mismatched LoRA checkpoint (nested
  PEFT bug) **once** and then loads it.
* `print_lora_discrepancies` – helper to inspect which tensors are missing /
  unexpected inside an adapter checkpoint.

The public surface is intentionally small; everything else is private.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file, save_file
from transformers import PretrainedConfig

# Ensure local project root is on sys.path BEFORE third‑party imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from cell2text_model.model import Cell2TextModel
from cell2text_model.geneformer_encoder import GeneformerModel, GeneformerConfig
from cell2text_model.llama_decoder import Cell2TextLlamaModel, Cell2TextLlamaConfig
from cell2text_model.projectors import MLPProjectionLayer, PerceiverIO

log = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
)

# -----------------------------------------------------------------------------#
# LoRA helpers
# -----------------------------------------------------------------------------#

def _print_table(title: str, rows: Sequence[str]) -> None:
    """Internal helper for pretty console output."""
    border = "─" * 42
    log.info("\n%s\n│ %s │\n%s", border, title.center(38), border)
    for row in rows:
        log.info(row)
    log.info("%s\n", border)


def print_lora_discrepancies(model: torch.nn.Module, adapter_dir: str) -> None:
    """Compare LoRA parameters expected by PEFT with those stored in the
    adapter checkpoint and log a diff.
    """
    adapter_path = Path(adapter_dir, "adapter_model.safetensors")
    cfg_path = Path(adapter_dir, "adapter_config.json")

    if not adapter_path.exists() or not cfg_path.exists():
        raise FileNotFoundError("LoRA adapter checkpoint is incomplete.")

    with cfg_path.open() as f:
        target_modules: List[str] = json.load(f).get("target_modules", [])

    expected: Set[str] = set()
    for name, module in model.named_modules():
        if any(name.endswith(t.replace("*", "")) or t.replace("*", "") in name
               for t in target_modules):
            if hasattr(module, "weight"):
                base = f"base_model.model.{name}"
                expected.update(
                    {f"{base}.lora_{s}.default.weight" for s in ("A", "B")}
                )

    provided = set(load_file(adapter_path))

    missing = sorted(expected - provided)
    unexpected = sorted(provided - expected)

    rows = [
        f"expected tensors : {len(expected)}",
        f"provided tensors : {len(provided)}",
        "",
        f"missing    ({len(missing):>4}) : {missing[:5]}{' …' if len(missing) > 5 else ''}",
        f"unexpected ({len(unexpected):>4}) : {unexpected[:5]}{' …' if len(unexpected) > 5 else ''}",
    ]
    _print_table("LoRA parameter diff", rows)


def _rewrite_lora_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Fix the nested PEFT bug (missing *base_model.model.* in keys)."""
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("base_model.model.decoder.llama.model.layers"):
            k = k.replace(
                "base_model.model.decoder.llama.model.layers",
                "base_model.model.base_model.model.decoder.llama.model.layers",
                1,
            )
        elif k.startswith("base_model.model.decoder."):
            k = k.replace(
                "base_model.model.decoder.",
                "base_model.model.base_model.model.decoder.",
                1,
            )
        new_sd[k] = v
    return new_sd


def fix_and_load_adapter(
    model: torch.nn.Module,
    adapter_dir: str,
    is_trainable: bool = True,
) -> PeftModel:
    """Rewrite LoRA checkpoint keys if necessary then load it onto *model*."""
    adapter_dir = Path(adapter_dir)
    fixed_dir = adapter_dir.with_name(f"{adapter_dir.name}_fixed")

    if not fixed_dir.exists():
        log.info("Fixing LoRA keys in %s → %s", adapter_dir, fixed_dir)
        state_dict = load_file(adapter_dir / "adapter_model.safetensors")
        fixed_dir.mkdir(parents=True, exist_ok=True)
        save_file(_rewrite_lora_keys(state_dict), fixed_dir / "adapter_model.safetensors")

        for fname in ("adapter_config.json", "README.md"):
            src = adapter_dir / fname
            if src.exists():
                shutil.copy2(src, fixed_dir / fname)
    else:
        log.info("Using cached fixed adapter at %s", fixed_dir)

    peft = PeftModel.from_pretrained(model, fixed_dir, is_trainable=is_trainable)
    print_lora_discrepancies(peft, fixed_dir)
    return peft


# -----------------------------------------------------------------------------#
# Target‑module discovery
# -----------------------------------------------------------------------------#
PROJ_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def discover_target_modules(model: torch.nn.Module, scope: str | None = None) -> List[str]:
    """Return LoRA‑adaptable parameter patterns discovered in *model*."""
    patterns: Set[str] = set()

    for name, module in model.named_modules():
        if (scope and scope not in name) or not hasattr(module, "weight"):
            continue
        if any(p in name for p in PROJ_NAMES):
            root = ".".join(name.split(".")[:-1])
            patterns.add(root)

    targets = [f"{base}.{p}" for base in sorted(patterns) for p in PROJ_NAMES]
    return targets


# -----------------------------------------------------------------------------#
# Model factory
# -----------------------------------------------------------------------------#

def load_model(args: Dict[str, Any]) -> PeftModel:
    """Return a **train‑ready** Cell2Text model configured via *args*."""
    cfg = create_cell2text_config(args)

    # ---------- Encoder ---------- #
    gene_cfg = GeneformerConfig(
        emb_mode=args.get("emb_mode", "gene"),
        max_ncells=args.get("max_ncells", 1000),
        emb_layer=args.get("emb_layer", -1),
        emb_label=args.get("emb_label"),
        nproc=args.get("nproc", -1),
        forward_batch_size=args.get("forward_batch_size", 100),
        summary_stat=args.get("summary_stat"),
        token_dictionary_path=args.get(
            "token_dictionary_path",
            "/home/arism/cell2text/Geneformer/geneformer/token_dictionary_gc104M.pkl",
        ),
    )
    encoder = GeneformerModel.from_pretrained(args["geneformer_path"], config=gene_cfg)
    for p in encoder.parameters():
        p.requires_grad = False

    # ---------- Decoder ---------- #
    llama_cfg = Cell2TextLlamaConfig(
        max_length=args.get("max_length", 100),
        num_beams=args.get("num_beams", 1),
        early_stopping=args.get("early_stopping", True),
        no_repeat_ngram_size=args.get("no_repeat_ngram_size", 3),
        temperature=args.get("temperature", 1.0),
        top_p=args.get("top_p", 1.0),
    )
    decoder = Cell2TextLlamaModel.from_pretrained(args["llama_path"], config=llama_cfg)

    # ---------- Projector ---------- #
    if args["projector"] == "mlp":
        projector = MLPProjectionLayer(
            input_dim=args["cell_encoder_hidden_size"],
            hidden_dim=args["mlp_hidden_size"],
            output_dim=args["decoder_hidden_size"],
            dropout_prob=args["mlp_dropout"],
            bias=True,
        )
    elif args["projector"] == "perceiver":
        projector = PerceiverIO(
            input_dim=args["cell_encoder_hidden_size"],
            output_dim=args["decoder_hidden_size"],
            num_latents=args["num_latents"],
            num_cross_attn_layers=args["perceiver_cross_attn_layers"],
            num_heads=args["perceiver_num_heads"],
            ff_mult=args["ff_mult"],
            dropout=args["perceiver_dropout"],
            use_position_encoding=args["use_position_encoding"],
        )
    else:
        raise ValueError(f"Unknown projector type: {args['projector']}")

    # ---------- Assemble ---------- #
    model = Cell2TextModel(cfg)
    model.cell_encoder = encoder
    model.decoder = decoder
    model.cell_to_embedding = projector

    # ---------- Projector checkpoint ---------- #
    if ckpt := args.get("load_model_checkpoint_path"):
        full_sd = torch.load(ckpt, map_location="cpu")
        proj_sd = {
            k[len("cell_to_embedding.") :]: v
            for k, v in full_sd.items()
            if k.startswith("cell_to_embedding.")
        }
        log.info("Loading %d projector weights from %s", len(proj_sd), ckpt)
        missing, unexpected = projector.load_state_dict(proj_sd, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected projector keys: {unexpected}")
        log.info("Projector: %d tensors left at init values", len(missing))

    # ---------- LoRA ---------- #
    if "load_adapter_checkpoint_dir" in args:
        log.info("Loading pre‑trained LoRA adapter")
        decoder = get_peft_model(
            decoder,
            LoraConfig(
                r=args["lora_rank"],
                lora_alpha=args["lora_rank"] * 2,
                target_modules=discover_target_modules(decoder, "decoder"),
                lora_dropout=0.1,
                bias="none",
            ),
        )
        model.decoder = fix_and_load_adapter(
            decoder.get_base_model(), args["load_adapter_checkpoint_dir"]
        )
    else:
        log.info("Initialising new LoRA adapter")
        modules_to_save = None
        if not args.get("fix_modality_adapter", False):
            if args["projector"] == "mlp":
                modules_to_save = _get_mlp_modules_to_save(args["mlp_dropout"])
            elif args["projector"] == "perceiver":
                modules_to_save = [
                    "cell_to_embedding.latents",
                    "cell_to_embedding.input_projection",
                    "cell_to_embedding.cross_attention_layers",
                    "cell_to_embedding.self_attention_layers",
                    "cell_to_embedding.final_norm",
                    "cell_to_embedding.position_encoding",
                ]

        model = get_peft_model(
            model,
            LoraConfig(
                r=args["lora_rank"],
                lora_alpha=args["lora_rank"] * 2,
                lora_dropout=0.1,
                bias="none",
                target_modules=discover_target_modules(model, "decoder"),
                modules_to_save=modules_to_save,
            ),
        )

    return model


# -----------------------------------------------------------------------------#
# Misc helpers
# -----------------------------------------------------------------------------#

def _get_mlp_modules_to_save(dropout: float) -> List[str]:
    base = [
        "cell_to_embedding.projection.0",  # Linear
        "cell_to_embedding.projection.1",  # LayerNorm
    ]
    base.append("cell_to_embedding.projection.4" if dropout > 0 else "cell_to_embedding.projection.3")
    base.append("cell_to_embedding.projection.5" if dropout > 0 else "cell_to_embedding.projection.4")
    return base


def create_cell2text_config(args: Dict[str, Any]) -> PretrainedConfig:
    """Return a PretrainedConfig capturing hyper‑parameters from *args*."""
    cfg = PretrainedConfig()

    # core architecture
    cfg.projector = args["projector"]
    cfg.cell_encoder_hidden_size = args["cell_encoder_hidden_size"]
    cfg.decoder_hidden_size = args["decoder_hidden_size"]
    cfg.top_k = args["top_k"]

    # MLP
    cfg.mlp_hidden_size = args["mlp_hidden_size"]
    cfg.mlp_dropout = args["mlp_dropout"]

    # Perceiver
    cfg.num_latents = args["num_latents"]
    cfg.perceiver_cross_attn_layers = args["perceiver_cross_attn_layers"]
    cfg.perceiver_num_heads = args["perceiver_num_heads"]
    cfg.ff_mult = args["ff_mult"]
    cfg.perceiver_dropout = args["perceiver_dropout"]
    cfg.use_position_encoding = args["use_position_encoding"]

    # Geneformer
    for key in (
        "emb_mode",
        "max_ncells",
        "emb_layer",
        "emb_label",
        "nproc",
        "forward_batch_size",
        "summary_stat",
        "token_dictionary_path",
    ):
        cfg.__dict__[key] = args.get(key)

    # decoding
    for key in (
        "max_length",
        "num_beams",
        "early_stopping",
        "no_repeat_ngram_size",
        "temperature",
        "top_p",
    ):
        cfg.__dict__[key] = args.get(key)

    return cfg
