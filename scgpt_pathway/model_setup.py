import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, "../")
from scgpt.model import TransformerModel

class ScGPTPathwayClassifier(nn.Module):
    """
    scGPT model adapted for multi-label pathway classification.
    Similar architecture to GeneformerPathwayClassifier.
    """
    def __init__(self, config, vocab, num_pathways: int, device: str = "cuda", freeze_scgpt: bool = True):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.num_pathways = num_pathways
        self.device = device
        self.freeze_scgpt = freeze_scgpt
        
        # Set model parameters (same as your existing code)
        self.pad_token = "<pad>"
        self.pad_value = -2 if config.input_emb_style != "category" else config.n_bins
        self.n_input_bins = config.n_bins + 2 if config.input_emb_style == "category" else config.n_bins
        
        # Build base model
        self.model = None
        self.model_configs = {}
        
    def load_pretrained_model(self, model_path: str):
        """Load pre-trained scGPT model"""
        model_dir = Path(model_path)
        model_config_file = model_dir / "args.json"
        model_file = model_dir / "best_model.pt"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        # Load model configuration
        if model_config_file.exists():
            with open(model_config_file, "r") as f:
                self.model_configs = json.load(f)
            print(f"Loaded model config from {model_config_file}")
            
            self.embsize = self.model_configs["embsize"]
            self.nhead = self.model_configs["nheads"]
            self.d_hid = self.model_configs["d_hid"]
            self.nlayers = self.model_configs["nlayers"]
            self.n_layers_cls = self.model_configs.get("n_layers_cls", 3)
            
            # Match architecture to checkpoint
            self.use_fast_transformer = True
            self.fast_transformer_backend = "flash"
            self.pre_norm = self.model_configs.get("pre_norm", False)
        else:
            raise FileNotFoundError("Pretrained model config is required")
        
        # Create model with EXACT same architecture
        ntokens = len(self.vocab)
        self.model = TransformerModel(
            ntokens,
            self.embsize,
            self.nhead,
            self.d_hid,
            self.nlayers,
            nlayers_cls=self.n_layers_cls,
            n_cls=self.num_pathways,  # Multi-label output
            vocab=self.vocab,
            dropout=self.config.dropout,
            pad_token=self.pad_token,
            pad_value=self.pad_value,
            do_mvc=False,
            do_dab=False,
            use_batch_labels=False,
            num_batch_labels=1,
            domain_spec_batchnorm=self.config.DSBN,
            input_emb_style=self.config.input_emb_style,
            n_input_bins=self.n_input_bins,
            cell_emb_style=self.config.cell_emb_style,
            mvc_decoder_style="inner product",
            ecs_threshold=0.0,
            explicit_zero_prob=False,
            use_fast_transformer=self.use_fast_transformer,
            fast_transformer_backend=self.fast_transformer_backend,
            pre_norm=self.pre_norm,
        )
        
        # Load pretrained weights
        state_dict = torch.load(model_file, map_location=self.device)
        model_dict = self.model.state_dict()
        
        # Load compatible weights
        pretrained_dict = {k: v for k, v in state_dict.items() 
                        if k in model_dict and v.shape == model_dict[k].shape}
        
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict, strict=False)
        
        print(f"✅ Successfully loaded {len(pretrained_dict)}/{len(state_dict)} parameters")
        
        # Freeze scGPT if requested
        if self.freeze_scgpt:
            print("❄️ Freezing all non-decoder parameters...")
            frozen_count = 0
            for name, para in self.model.named_parameters():
                if "decoder" not in name:  # Freeze everything except classification head
                    para.requires_grad = False
                    frozen_count += 1
            print(f"--- Frozen {frozen_count} parameters. ---")
        
        return self.model
    
    def forward(self, gene_ids, values, src_key_padding_mask=None):
        """
        Forward pass through scGPT model.
        Returns logits for multi-label classification.
        """
        output_dict = self.model(
            gene_ids,
            values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
            CLS=True,
            CCE=False,
            MVC=False,
            ECS=False,
            do_sample=False,
        )
        # Return raw logits (no sigmoid here - done in loss function)
        return output_dict["cls_output"]  # Shape: (batch_size, num_pathways)


def setup_pathway_model(config, vocab, num_pathways, pretrained_path, device="cuda", freeze_scgpt=True):
    """
    Convenience function to setup the pathway classification model.
    """
    model = ScGPTPathwayClassifier(config, vocab, num_pathways, device, freeze_scgpt)
    model.load_pretrained_model(pretrained_path)
    model = model.to(device)
    
    # Log parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Model Parameters:     {total_params:,}")
    print(f"Total Trainable Parameters: {trainable_params:,}")
    
    return model