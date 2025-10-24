"""
Model setup and loading utilities for scGPT fine-tuning
"""
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any
import warnings

import sys
sys.path.insert(0, "../")
from scgpt.model import TransformerModel


class ModelManager:
    def __init__(self, config, vocab, num_classes: int, device: str = "cuda"):
        self.config = config
        self.vocab = vocab
        self.num_classes = num_classes
        self.device = device
        self.model = None
        self.model_configs = {}
        
        # Set model parameters
        self.pad_token = "<pad>"
        self.pad_value = -2 if config.input_emb_style != "category" else config.n_bins
        self.n_input_bins = config.n_bins + 2 if config.input_emb_style == "category" else config.n_bins
    
    def load_pretrained_model(self, model_path: str) -> TransformerModel:
        """Load pre-trained scGPT model with exact architectural compatibility"""
        model_dir = Path(model_path)
        model_config_file = model_dir / "args.json"
        model_file = model_dir / "best_model.pt"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        # Load model configuration FIRST
        if model_config_file.exists():
            with open(model_config_file, "r") as f:
                self.model_configs = json.load(f)
            print(f"Loaded model config from {model_config_file}")
            
            # 🚨 CRITICAL: Use EXACT same architecture as pretrained model
            self.embsize = self.model_configs["embsize"]
            self.nhead = self.model_configs["nheads"]
            self.d_hid = self.model_configs["d_hid"]
            self.nlayers = self.model_configs["nlayers"]
            self.n_layers_cls = self.model_configs.get("n_layers_cls", 3)
            
            # -----------------------------------------------------------------
            # --- FIX 1: Force architecture to match checkpoint keys ---
            # The 'Wqkv' keys in your log mean the checkpoint uses 
            # a fused QKV layer, which corresponds to the 'flash' backend.
            # We will IGNORE the args.json and force this architecture.
            # -----------------------------------------------------------------
            self.use_fast_transformer = True
            self.fast_transformer_backend = "flash" # This backend uses Wqkv
            self.pre_norm = self.model_configs.get("pre_norm", False) # pre_norm is probably fine
            
            print(f"⚠️ FORCING pretrained model architecture: "
                  f"fast_transformer={self.use_fast_transformer}, "
                  f"backend={self.fast_transformer_backend}")
        else:
            raise FileNotFoundError("Pretrained model config is required for compatibility")
        
        # Create model with EXACT same architecture
        ntokens = len(self.vocab)
        self.model = TransformerModel(
            ntokens,
            self.embsize,
            self.nhead,
            self.d_hid,
            self.nlayers,
            nlayers_cls=self.n_layers_cls,
            n_cls=self.num_classes,
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
            use_fast_transformer=self.use_fast_transformer,  # ← MUST MATCH
            fast_transformer_backend=self.fast_transformer_backend,  # ← MUST MATCH
            pre_norm=self.pre_norm,  # ← MUST MATCH
        )
        
        # Load pretrained weights
        # This (non-strict) loading logic is what the tutorial's 'except' block
        # was doing, but we do it by default to be safe.
        state_dict = torch.load(model_file, map_location=self.device)
        model_dict = self.model.state_dict()
        
        ignored_keys = []
        compatible_keys = []
        
        for k, v in state_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                compatible_keys.append(k)
            else:
                ignored_keys.append(k)
        
        print("=" * 60)
        print("COMPATIBLE KEYS (will be loaded):")
        for k in compatible_keys:
            print(f"✅ {k}")
        
        print("\n" + "=" * 60)
        print("IGNORED KEYS (architecture mismatch):")
        for k in ignored_keys:
            print(f"❌ {k}")
        print("=" * 60)
        
        # Load only compatible weights
        pretrained_dict = {k: v for k, v in state_dict.items() 
                        if k in model_dict and v.shape == model_dict[k].shape}
        
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict, strict=False)
        
        print(f"✅ Successfully loaded {len(compatible_keys)}/{len(state_dict)} parameters")
        
        return self.model
    
    # --------------------------------------------------------------------
    # --- FIX 2: CORRECTED FREEZING LOGIC ---
    # This logic correctly freezes everything EXCEPT the classifier head.
    # This is what gives the correct ~593k trainable parameter count.
    # --------------------------------------------------------------------
    def freeze_encoder(self):
        """Freeze encoder parameters for fine-tuning"""
        if not self.model:
            raise ValueError("Model not loaded yet")
        
        pre_freeze_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Pre-freeze trainable parameters: {pre_freeze_params:,}")

        print("--- Freezing Model Layers ---")
        for name, param in self.model.named_parameters():
            # Freeze everything that is NOT part of the classification head
            # The classifier layers have "cls" in their name
            if self.config.freeze and "cls" not in name:
                param.requires_grad = False
            else:
                # This will print the layers you ARE training
                print(f"✅ TRAINING: {name}")

        post_freeze_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Post-freeze trainable parameters: {post_freeze_params:,}")
        
        return pre_freeze_params, post_freeze_params
    
    def setup_model(self, pretrained_path: Optional[str] = None) -> TransformerModel:
        """Setup model with optional pretrained weights"""
        if pretrained_path:
            self.load_pretrained_model(pretrained_path)
        else:
            # Create fresh model
            ntokens = len(self.vocab)
            self.model = TransformerModel(
                ntokens,
                self.config.layer_size,
                self.config.nhead,
                self.config.layer_size,
                self.config.nlayers,
                nlayers_cls=3,
                n_cls=self.num_classes,
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
                use_fast_transformer=self.config.fast_transformer,
                fast_transformer_backend="linear",
                pre_norm=self.config.pre_norm,
            )
        
        # Apply freezing if specified
        if self.config.freeze:
            self.freeze_encoder()
        
        # -----------------------------------------------------------------
        # --- FIX 3: BATCHNORM FIX ---
        # This prevents contamination of BatchNorm running statistics 
        # during chunked training. This is correct.
        # -----------------------------------------------------------------
        
        def set_bn_eval(module):
            """Recursively set all BatchNorm layers to eval mode."""
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()

        # Apply the fix to the entire model
        self.model.apply(set_bn_eval)
        print("✅ Applied BatchNorm fix: All BatchNorm layers set to eval mode.")
        # -----------------------------------------------------------------
        # --- END FIX ---
        # -----------------------------------------------------------------
        
        # Move to device
        self.model.to(self.device)
        
        return self.model
    
    def get_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler"""
        if not self.model:
            raise ValueError("Model not setup yet")
        
        # Get only parameters that require gradients
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())

        optimizer = torch.optim.Adam(
            trainable_params,  # Only pass trainable parameters to the optimizer
            lr=self.config.lr,
            eps=1e-4 if self.config.amp else 1e-8
        )
        
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1,
            gamma=self.config.schedule_ratio
        )
        
        return optimizer, scheduler
    
    def get_loss_function(self):
        """Get loss function for classification"""
        return nn.CrossEntropyLoss()
    
    def save_model(self, save_path: str, additional_info: Optional[Dict[str, Any]] = None):
        """Save model state and configuration"""
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save(self.model.state_dict(), save_dir / "model.pt")
        
        # Save configuration
        config_dict = {k: v for k, v in self.config.__dict__.items() 
                      if not k.startswith('_')}
        if additional_info:
            config_dict.update(additional_info)
        
        with open(save_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # Save vocabulary
        if hasattr(self.vocab, 'save'):
            self.vocab.save(save_dir / "vocab.json")
        
        print(f"Model saved to {save_dir}")
