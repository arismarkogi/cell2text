import torch
from peft import PeftModel, LoraConfig, get_peft_model
from typing import Dict, Any
from transformers import PretrainedConfig
import os
import sys
from peft import AutoPeftModelForCausalLM
from transformers import AutoModel

# Import your model classes
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cell2text_model.model import Cell2TextModel
from cell2text_model.geneformer_encoder import GeneformerModel, GeneformerConfig
from cell2text_model.llama_decoder import Cell2TextLlamaModel, Cell2TextLlamaConfig
from cell2text_model.projectors import MLPProjectionLayer, PerceiverIO, QFormerProjector


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
    elif args["projector"] == "qformer":
        adapter = QFormerProjector(
            #bert_model_name=args["qformer_bert_model"],
            cross_attention_freq=args["qformer_cross_attention_freq"],
            use_flash_attn=args["qformer_use_flash_attn"],
            #freeze_qformer=args["qformer_freeze"],
            input_dim=args["cell_encoder_hidden_size"],
            output_dim=args["decoder_hidden_size"],
        )
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
        load_projector_weights(model, args["load_model_checkpoint_path"],bert_model_name=args.get("qformer_bert_model"))
    
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



import torch

def is_trainable_parameter_qformer(param_name):
    """Check if a QFormer parameter should be trainable (and thus loaded)"""
    # Query tokens
    if param_name == "query_tokens":
        return True
    
    # Projection layers
    if any(comp in param_name for comp in ["input_projection", "output_projection", "ln_cell"]):
        return True
    
    # Cross-attention (any layer)
    if ".crossattention." in param_name:
        return True
    
    # Query-specific components (any layer)
    if ".output_query" in param_name or ".intermediate_query." in param_name:
        return True
    
    return False

def load_projector_weights(model, checkpoint_path: str, bert_model_name: str = None, load_only_trainable: bool = False):
    """Load projector weights with proper trainable parameter filtering for QFormer."""
    print(f"\n[DEBUG] Loading projector weights from {checkpoint_path}")
    
    full_state_dict = torch.load(checkpoint_path, map_location="cpu")
    print(f"[DEBUG] Full checkpoint keys: {len(full_state_dict)} parameters")
    
    # Extract projector-related weights
    raw_projector_state_dict = {
        k: v for k, v in full_state_dict.items()
        if k.startswith("cell_to_embedding.")
    }
    print(f"[DEBUG] Extracted {len(raw_projector_state_dict)} projector-related parameters")
    
    if hasattr(model.cell_to_embedding, '__class__') and 'QFormer' in model.cell_to_embedding.__class__.__name__:
        print("[DEBUG] Detected QFormer projector")
        
        if load_only_trainable:
            print("[DEBUG] → Loading only trainable components")
            trainable_components = {}
            frozen_components = {}
            
            for k, v in raw_projector_state_dict.items():
                new_key = k[len("cell_to_embedding."):]
                
                if is_trainable_parameter_qformer(new_key):
                    trainable_components[new_key] = v
                else:
                    frozen_components[new_key] = v
            
            print(f"[DEBUG] Trainable components: {len(trainable_components)} parameters")
            print(f"[DEBUG] → Loading: {list(trainable_components.keys())[:5]}{' ...' if len(trainable_components) > 5 else ''}")
            print(f"[DEBUG] Frozen components: {len(frozen_components)} parameters (skipped)")
            print(f"[DEBUG] → Skipping: {list(frozen_components.keys())[:5]}{' ...' if len(frozen_components) > 5 else ''}")
            
            # Load only trainable components
            missing, unexpected = model.cell_to_embedding.load_state_dict(
                trainable_components, strict=False
            )
            
        else:
            print("[DEBUG] → Loading all QFormer components")
            # Load everything except BERT base weights (your original logic)
            custom_components = {}
            skipped_keys = []
            
            for k, v in raw_projector_state_dict.items():
                new_key = k[len("cell_to_embedding."):]    
                custom_components[new_key] = v
               
            print(f"[DEBUG] Loading {len(custom_components)} custom components")
            print(f"[DEBUG] Skipped {len(skipped_keys)} base BERT components")
            
            missing, unexpected = model.cell_to_embedding.load_state_dict(
                custom_components, strict=False
            )
    
    else:
        print("[DEBUG] Detected non-QFormer projector → loading all projector weights")
        projector_state_dict = {}
        for k, v in raw_projector_state_dict.items():
            new_key = k[len("cell_to_embedding."):]
            projector_state_dict[new_key] = v
        
        print(f"[DEBUG] Loading {len(projector_state_dict)} projector weights")
        missing, unexpected = model.cell_to_embedding.load_state_dict(
            projector_state_dict, strict=False
        )
    
    print(f"[DEBUG] Load results → Missing: {missing}")
    print(f"[DEBUG] Load results → Unexpected: {unexpected}")
    print(f"[DEBUG] ✓ Projector weights loaded successfully\n")
    
    return missing, unexpected



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