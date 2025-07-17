import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from transformers import (
    PretrainedConfig,
    get_linear_schedule_with_warmup,
    AutoTokenizer
)
import pickle
from tqdm import tqdm
import sys
import json
from torch.utils.data.distributed import DistributedSampler


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import os

# LoRA imports
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    PeftModel,
)


from util import create_argument_parser, create_progress_bar, compute_enhanced_training_summary, convert_json_compat, save_simple_training_history

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cell2text_model.model import Cell2TextModel
from cell2text_model.configuration import Cell2TextConfig
from cell2text_dataset.dataset import Cell2TextDataset, SanityDataset


class Cell2TextDDPTrainer:
    """DDP-enabled trainer for Cell2Text model"""
    
    def __init__(self, args, rank=0, world_size=1):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = (rank == 0)
        self.experiment_name = getattr(args, 'experiment_name', None)
        self.device = None
        self.model = None
        self.tokenizer = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.lr_scheduler = None
        self.global_step = 0
        self.losses = []

        # Setup distributed training
        if world_size > 1:
            self.setup_distributed()
        
        # Set device
        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        torch.cuda.set_device(self.device)
         
        
    def setup_distributed(self):
        """Initialize distributed training"""
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '12355'
            
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            rank=self.rank,
            world_size=self.world_size
        )

        
    def load_tokenizer(self):
        """Load tokenizer for text descriptions"""
        print("Loading tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.decoder_path, 
                pad_token='<|reserved_special_token_0|>'
            )
            print("Tokenizer loaded successfully.")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            print("Proceeding without tokenizer. This may cause issues in training.")
            self.tokenizer = None
            
    def setup_datasets(self):
        """Setup training and validation datasets with DDP samplers"""
        if self.is_main_process:
            print(f"Loading dataset...")
        
        # Load full training dataset
        if self.args.projector == "mlp":
            top_k = self.args.top_k
        else:  # perceiver
            top_k = None
            
        full_train_dataset = Cell2TextDataset(self.args.train_data_path,
                                               self.tokenizer, top_k=top_k, 
                                               projector=self.args.projector, 
                                               num_latents=self.args.num_latents,
                                               sort_by_depth=self.args.sort_by_depth)
        
        if self.args.mode == "sanity":
            # Create small subset for sanity check
            sanity_dataset = SanityDataset(full_train_dataset, num_samples=self.args.num_samples, seed=self.args.seed)
            
            if self.world_size > 1:
                train_sampler = DistributedSampler(full_train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
                
                # 👇 Log the assigned indices for this rank
                if self.is_main_process:
                    print(f"[Rank {self.rank}] Train sampler total size: {train_sampler.total_size}")
                    print(f"[Rank {self.rank}] First 10 train indices: {train_sampler.indices[:10]}")
                    print(f"[Rank {self.rank}] First 10 train indices: {train_sampler["cl_depth"][:10]}")
                else:
                    train_sampler = None

                val_sampler = DistributedSampler(sanity_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
            else:
                train_sampler = None
                val_sampler = None
            
            self.train_loader = DataLoader(
                sanity_dataset, 
                batch_size=self.args.batch_size_per_device, 
                shuffle=False,
                sampler=train_sampler,
                num_workers=0,
                collate_fn=full_train_dataset.collate_fn(mode="train")
            )
            
            self.val_loader = DataLoader(
                sanity_dataset, 
                batch_size=self.args.batch_size_per_device, 
                shuffle=False,
                sampler=val_sampler,
                num_workers=0,
                collate_fn=full_train_dataset.collate_fn(mode="train")
            )
            
            if self.is_main_process:
                print(f"Sanity dataset loaded. Size: {len(sanity_dataset)}")
        else:
            # Full training mode
            if self.world_size > 1:
                train_sampler = DistributedSampler(full_train_dataset, num_replicas=self.world_size, rank=self.rank)
            else:
                train_sampler = None
                
            self.train_loader = DataLoader(
                full_train_dataset, 
                batch_size=self.args.batch_size_per_device, 
                shuffle=False,
                sampler=train_sampler,
                num_workers=0,
                collate_fn=full_train_dataset.collate_fn(mode="train")
            )
            
            if self.is_main_process:
                print(f"Full training dataset loaded. Size: {len(full_train_dataset)}")
            
            # Load validation dataset if provided
            if self.args.val_data_path:
                val_dataset = Cell2TextDataset(self.args.val_data_path, 
                                               self.tokenizer,
                                                 top_k=top_k, 
                                                 projector=self.args.projector, 
                                                 num_latents=self.args.num_latents)
                
                if self.world_size > 1:
                    val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
                else:
                    val_sampler = None
                
                self.val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.args.batch_size_per_device, 
                    shuffle=False,
                    sampler=val_sampler,
                    num_workers=0,
                    collate_fn=val_dataset.collate_fn(mode="train")
                )
                
                if self.is_main_process:
                    print(f"Validation dataset loaded. Size: {len(val_dataset)}")
            else:
                if self.is_main_process:
                    print("No validation dataset provided.")
                self.val_loader = None
        
    def initialize_model(self):
        """Initialize the Cell2Text model with configuration"""
        print("Initializing model configuration...")
        config = Cell2TextConfig()
        
        # Set required configuration parameters
        config.cell_encoder_hidden_size = self.args.encoder_hidden_size
        config.decoder_hidden_size = self.args.decoder_hidden_size
        config.geneformer_path = self.args.geneformer_path
        config.decoder_model_name_or_path = self.args.decoder_path
        config.token_dictionary_path = self.args.token_dictionary_path
        
        # Projector configuration
        config.projector = self.args.projector
        
        if self.args.projector == "mlp":
            config.mlp_hidden_size = self.args.mlp_hidden_size
            config.mlp_dropout = self.args.mlp_dropout
            config.top_k = self.args.top_k
        elif self.args.projector == "perceiver":
            config.num_latents = self.args.num_latents
            config.num_heads = self.args.num_heads
            config.ff_mult = self.args.ff_mult
            config.perceiver_dropout = self.args.perceiver_dropout
        
        # Additional configuration parameters
        config.max_ncells = self.args.max_ncells
        config.max_new_tokens = self.args.max_length
        config.num_beams = self.args.num_beams
        
        # Initialize the model
        print("Initializing model...")
        self.model = Cell2TextModel(config=config)
        self.model.warm_up()  # Load pretrained weights
        
    def freeze_model_components(self):
        """Freeze specified model components"""
        print("Freezing cell encoder parameters...")
        for param in self.model.cell_encoder.parameters():
            param.requires_grad = False
        print("Cell encoder parameters frozen.")
        
        if not self.args.use_lora_decoder and self.args.freeze_decoder:
            print("Freezing decoder parameters...")
            for param in self.model.decoder.parameters():
                param.requires_grad = False
            print("Decoder parameters frozen.")
            
    def apply_lora_to_model(self):
        """Apply LoRA to the specified components of the model"""
        if not self.args.use_lora_decoder:
            return
            
        print("Applying LoRA to decoder...")
                
        decoder_lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.args.lora_r_decoder,
            lora_alpha=self.args.lora_alpha_decoder,
            lora_dropout=self.args.lora_dropout_decoder,
            target_modules=self.args.lora_target_modules_decoder,
            bias=self.args.lora_bias_decoder,
            modules_to_save=self.args.lora_modules_to_save_decoder if self.args.lora_modules_to_save_decoder else None,
        )
        
        # Apply LoRA to decoder
        self.model.decoder = get_peft_model(self.model.decoder, decoder_lora_config)
        
        # Add peft_config to the main model for easier access during saving
        self.model.peft_config = self.model.decoder.peft_config
        self.model.is_peft_model = True
        
        # Print trainable parameters for decoder
        trainable_decoder_params = sum(p.numel() for p in self.model.decoder.parameters() if p.requires_grad)
        total_decoder_params = sum(p.numel() for p in self.model.decoder.parameters())
        print(f"Decoder LoRA applied. Trainable parameters: {trainable_decoder_params:,} / {total_decoder_params:,} ({100 * trainable_decoder_params / total_decoder_params:.2f}%)")

    def print_model_parameters(self):
        """Print detailed information about model parameters"""
        trainable_params, all_params = self.get_trainable_parameters()
        print(f"\nParameter Summary:")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"All parameters: {all_params:,}")
        print(f"Trainable %: {100 * trainable_params / all_params:.2f}%")
        
    def get_trainable_parameters(self):
        """Get the number of trainable parameters"""
        trainable_params = 0
        all_params = 0
        
        for param in self.model.parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        return trainable_params, all_params
    
    def setup_model_and_optimizer(self):
        """Setup model, optimizer with DDP wrapping"""
        if self.is_main_process:
            print("Setting up optimizer...")
        
        # Move model to device first
        self.model = self.model.to(self.device)
        
        # Wrap with DDP
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=False)
            model_for_params = self.model.module
        else:
            model_for_params = self.model
        
        # Create optimizer with different parameter groups
        parameters = []
        
        # Projector parameters
        projector_name = "cell_to_embedding" if self.args.projector == "mlp" else "perceiver_projector"
        projector_module = getattr(model_for_params, projector_name, None)
        if projector_module:
            for name, param in projector_module.named_parameters():
                if param.requires_grad:
                    parameters.append({
                        "params": [param],
                        "lr": self.args.projector_lr
                    })
        
        # Decoder parameters
        for name, param in model_for_params.decoder.named_parameters():
            if param.requires_grad:
                parameters.append({
                    "params": [param],
                    "lr": self.args.decoder_lr
                })
        
        if self.is_main_process:
            print(f"Created {len(parameters)} parameter groups")
        
        # Create optimizer
        self.optimizer = AdamW(
            parameters,
            lr=self.args.decoder_lr,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Create scheduler
        total_steps = self.args.epochs * len(self.train_loader) // self.args.gradient_accumulation_steps
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=total_steps
        )
        
        if self.is_main_process:
            print("Optimizer and scheduler created")
        
        
    def train_step(self, batch):
        # Move batch to device
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

        # Are we at the last accumulation step?
        is_last_step = (self.global_step + 1) % self.args.gradient_accumulation_steps == 0

        # Forward + backward
        loss = self.model(
            expression_tokens=batch["expression_tokens"],
            expression_token_lengths=batch["expression_token_lengths"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            return_dict=True
        ).loss

        loss = loss / self.args.gradient_accumulation_steps  # scale loss

        if self.world_size > 1 and not is_last_step:
            with self.model.no_sync():
                loss.backward()
        else:
            loss.backward()

        # Optimizer step on last accumulation step
        if is_last_step:
            if self.args.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters() if self.world_size == 1 else self.model.module.parameters(),
                    self.args.max_grad_norm
                )
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

        self.global_step += 1
        return loss.item() * self.args.gradient_accumulation_steps

    def validate(self):
        """Simplified validation - just compute loss"""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                
                # Forward pass
                loss = self.model(
                    expression_tokens=batch["expression_tokens"],
                    expression_token_lengths=batch["expression_token_lengths"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    return_dict=True
                ).loss
                
                total_loss += loss.item()
                num_batches += 1
        
        # Average loss across batches
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Average across all processes if using DDP
        if self.world_size > 1:
            avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss_tensor.item() / self.world_size
        
        return avg_loss
    

    def save_checkpoint_separate_adapters(
        self,
        checkpoint_dir: str,
        additional_state: dict | None = None,
        save_full: bool = False,          # set True only if you *really* want the old behaviour
    ):
        """
        Save a minimal checkpoint:
        • LoRA / projector adapter (PEFT format)
        • Slim model state‑dict (trainable params only)
        • Training state (optimizer, scheduler, global_step …)
        • LoRA config JSON
        """
        if not self.is_main_process:
            return

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Unwrap DDP if needed
        model = self.model.module if isinstance(self.model, DDP) else self.model

        # -------- 1. Detect PEFT ------------------------------------------------
        has_peft = (
            hasattr(model, "peft_config") and model.peft_config
        ) or (
            hasattr(model, "decoder")
            and hasattr(model.decoder, "peft_config")
            and model.decoder.peft_config
        )

        if not has_peft:
            # Plain model => only save trainable params (usually projector) + state
            slim_sd = {
                n: p.detach().clone()
                for n, p in model.named_parameters()
                if p.requires_grad
            }
            torch.save(slim_sd, os.path.join(checkpoint_dir, "pytorch_model.bin"))
            print(f"[checkpoint] saved slim state‑dict with {len(slim_sd)} tensors")

        else:
            # ---------------- 2. Save adapter -----------------------------------
            print("[checkpoint] PEFT model detected – saving LoRA adapter only")
            peft_model = model.decoder if hasattr(model.decoder, "peft_config") else model

            adapter_dir = os.path.join(checkpoint_dir, "adapter")
            peft_model.save_pretrained(adapter_dir)
            print(f"    • adapter → {adapter_dir}")

            # ---------------- 3. Save LoRA config (optional but tiny) ----------
            lora_cfg = peft_model.peft_config[next(iter(peft_model.peft_config))]
            lora_cfg_dict = {
                "peft_type": str(lora_cfg.peft_type),
                "task_type": str(lora_cfg.task_type),
                "r": lora_cfg.r,
                "lora_alpha": lora_cfg.lora_alpha,
                "lora_dropout": lora_cfg.lora_dropout,
                "target_modules": (list(lora_cfg.target_modules)
                                if isinstance(lora_cfg.target_modules, set)
                                else lora_cfg.target_modules),
                "bias": str(lora_cfg.bias),
                "modules_to_save": (list(lora_cfg.modules_to_save)
                                    if isinstance(lora_cfg.modules_to_save, set)
                                    else lora_cfg.modules_to_save),
                "init_lora_weights": lora_cfg.init_lora_weights,
                "applies_to": "decoder",
            }
            with open(os.path.join(checkpoint_dir, "lora_config.json"), "w") as f:
                json.dump(lora_cfg_dict, f, indent=2)
            print("    • lora_config.json written")

            # ---------------- 4. Slim state‑dict -------------------------------
            # Only things that were updated (requires_grad=True) – much smaller
            slim_sd = {
                n: p.detach().clone()
                for n, p in model.named_parameters()
                if p.requires_grad
            }
            torch.save(slim_sd, os.path.join(checkpoint_dir, "pytorch_model.bin"))
            print(f"    • slim state‑dict ({len(slim_sd)} tensors)")

            # (optional) also save the huge full model if explicitly requested
            if save_full:
                torch.save(model.state_dict(),
                        os.path.join(checkpoint_dir, "pytorch_model_full.bin"))
                print("    • full merged model saved (requested)")

        # -------- 5. Training‑state --------------------------------------------
        train_state = {
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "global_step": self.global_step,
            "losses": self.losses,
            "has_peft": has_peft,
        }
        if additional_state:
            train_state.update(additional_state)
        torch.save(train_state, os.path.join(checkpoint_dir, "training_state.pt"))
        print("    • training_state.pt")

        print(f"[checkpoint] done → {checkpoint_dir}")



    def save_checkpoint(self, checkpoint_dir, additional_state=None):
        """
        Backward compatibility - redirect to the new saving method
        """
        return self.save_checkpoint_separate_adapters(checkpoint_dir, additional_state)

    # For compatibility with your existing load_model function
    def load_checkpoint_training_state(checkpoint_dir):
        """Load training state from checkpoint"""
        training_state_path = os.path.join(checkpoint_dir, "training_state.pt")
        if os.path.exists(training_state_path):
            return torch.load(training_state_path, map_location="cpu")
        return None

    def cleanup_distributed(self):
        """Cleanup distributed training"""
        if self.world_size > 1:
            dist.destroy_process_group()

    def sanity_train(self):
        """Sanity training loop - overfit on small dataset"""
        print("="*60)
        print("STARTING DDP SANITY CHECK TRAINING")
        print("="*60)
        print(f"Target loss: {self.args.target_loss}")
        print(f"Epochs: {self.args.epochs}")
        print(f"Number of samples: {self.args.num_samples}")
        print(f"Batch size per device: {self.args.batch_size_per_device}")
        print(f"Gradient accumulation steps: {self.args.gradient_accumulation_steps}")
        print(f"Projector type: {self.args.projector}")
        print("="*60)
        
        # Training loop - overfit until target loss
        self.model.train()
        
        print("\nStarting overfitting training...")
        progress_bar = tqdm(
            desc="🚀 Sanity Training", 
            total=self.args.epochs, 
            position=0,
            leave=True,
            file=sys.stdout,
            disable=not self.is_main_process
        )
        
        epoch_loss = 0.0
        epoch = 0

        while epoch < self.args.epochs:
            epoch_losses = []
            
            for batch in self.train_loader:
                loss = self.train_step(batch)
                epoch_losses.append(loss)
                self.losses.append(loss)

            # After completing one epoch
            epoch_loss = np.mean(epoch_losses)
            
            # Only update progress bar on main process
            if self.is_main_process:
                progress_bar.update(1)
                progress_bar.set_description(
                    f"🚀 Sanity Training | Epoch: {epoch+1}/{self.args.epochs} | "
                    f"Loss: {epoch_loss:.4f} | Target: {self.args.target_loss:.4f}"
                )

            epoch += 1

            if epoch_loss <= self.args.target_loss:
                if self.is_main_process:
                    print(f"\n🎉 Target loss {self.args.target_loss:.4f} reached at epoch {epoch}!")
                    print(f"Final epoch loss: {epoch_loss:.4f}")
                break
                
        if progress_bar is not None:
            progress_bar.close()
        
        final_loss = epoch_loss
        if self.is_main_process:
            print(f"\nSanity training completed!")
            print(f"Final loss: {final_loss:.4f}")
            print(f"Target reached: {'✓' if final_loss <= self.args.target_loss else '✗'}")
        
        if self.args.save_model:
            checkpoint_name = "overfitted_sanity_model"
            if self.experiment_name:
                checkpoint_name = f"{self.experiment_name}_overfitted_sanity_model"
            
            checkpoint_dir = os.path.join(self.args.output_dir, checkpoint_name)
            self.save_checkpoint_separate_adapters(checkpoint_dir)
            
            # Save training info (only on main process)
            if self.is_main_process:
                info_path = os.path.join(checkpoint_dir, "training_info.json")
                info_dict = {
                    "experiment_name": self.experiment_name,
                    "final_loss": final_loss,
                    "epochs_taken": epoch,
                    "target_reached": final_loss <= self.args.target_loss,
                    "losses": self.losses,
                    "training_mode": self.args.mode,
                    "projector_type": self.args.projector
                }

                with open(info_path, 'w') as f:
                    json.dump(convert_json_compat(info_dict), f, indent=2)
                            
                print(f"Overfitted model saved to: {checkpoint_dir}")
        
        return final_loss <= self.args.target_loss
    
    def full_train(self):
        """Full training loop with DDP"""
        if self.is_main_process:
            print("="*60)
            print("STARTING DDP FULL TRAINING")
            print("="*60)
            print(f"World size: {self.world_size}")
            print(f"Rank: {self.rank}")
            print(f"Epochs: {self.args.epochs}")
            print(f"Training samples: {len(self.train_loader.dataset)}")
            print(f"Batch size per device: {self.args.batch_size_per_device}")
            print(f"Gradient accumulation steps: {self.args.gradient_accumulation_steps}")
            print(f"Projector type: {self.args.projector}")
            print(f"Validation every: {self.args.eval_steps} steps")
            print("="*60)
        
        self.model.train()
        
        # Initialize validation tracking
        validation_history = []
        training_history = []
        
        total_steps = self.args.epochs * len(self.train_loader)
        progress_bar = create_progress_bar("🚀 DDP Full Training", total_steps, self.is_main_process)
        
        best_val_loss = float('inf')
        steps_since_improvement = 0
        
        for epoch in range(self.args.epochs):
            # Set epoch for distributed sampler
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            
            epoch_losses = []
            
            for step, batch in enumerate(self.train_loader):
                loss = self.train_step(batch)
                epoch_losses.append(loss)
                self.losses.append(loss)
                
                # Store training step info (only on main process)
                if self.is_main_process:
                    training_step_info = {
                        'epoch': epoch + 1,
                        'step': step + 1,
                        'global_step': self.global_step,
                        'loss': loss,
                        'learning_rate': self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else self.args.decoder_lr
                    }
                    training_history.append(training_step_info)
                
                # Update progress bar
                if progress_bar:
                    progress_bar.update(1)
                    progress_bar.set_description(
                        f"🚀 DDP Training | Epoch: {epoch+1}/{self.args.epochs} | "
                        f"Step: {step+1}/{len(self.train_loader)} | Loss: {loss:.4f}"
                    )
                
                # Validation 
                if self.global_step % self.args.eval_steps == 0 and self.val_loader is not None:
                    if self.world_size > 1:
                        dist.barrier()

                    self.model.eval()
                    val_loss = self.validate()
                    
                    # Store validation results
                    validation_record = {
                        'epoch': epoch + 1,
                        'step': step + 1,
                        'global_step': self.global_step,
                        'validation_loss': val_loss,
                        'training_loss': np.mean(self.losses[-self.args.eval_steps:]) if len(self.losses) >= self.args.eval_steps else np.mean(self.losses)
                    }
                    validation_history.append(validation_record)
                    
                    if self.is_main_process:
                        print(f"\nValidation at step {self.global_step}:")
                        print(f"  Validation Loss: {val_loss:.4f}")
                    
                    # Early stopping and best model tracking
                    improved = val_loss < best_val_loss
                    
                    if improved:
                        best_val_loss = val_loss
                        steps_since_improvement = 0
                        
                        if self.args.save_model:
                            checkpoint_name = "best_model"
                            if self.experiment_name:
                                checkpoint_name = f"{self.experiment_name}_best_model"
                            
                            checkpoint_dir = os.path.join(self.args.output_dir, checkpoint_name)
                            self.save_checkpoint_separate_adapters(checkpoint_dir)
                            
                            if self.is_main_process:
                                print(f"Best model checkpoint saved to: {checkpoint_dir}")
                    else:
                        steps_since_improvement += self.args.eval_steps
                        
                        if self.args.early_stopping > 0 and steps_since_improvement >= self.args.early_stopping:
                            if self.is_main_process:
                                print(f"\nEarly stopping triggered after {steps_since_improvement} steps without improvement")
                            if progress_bar:
                                progress_bar.close()
                            
                            # Save final training and validation history
                            save_simple_training_history(self, training_history, validation_history)
                            return
                    
                    self.model.train()  # Switch back to training mode
                
                if self.world_size > 1:
                    dist.barrier()
                    
            # End of epoch summary
            epoch_loss = np.mean(epoch_losses)
            if self.is_main_process:
                print(f"\nEpoch {epoch+1} completed. Average loss: {epoch_loss:.4f}")
        
        if progress_bar is not None:
            progress_bar.close()
        
        # Final evaluation
        if self.val_loader is not None:
            self.model.eval()
            final_val_loss = self.validate()
            
            # Add final validation to history
            final_validation_record = {
                'epoch': self.args.epochs,
                'step': 'final',
                'global_step': self.global_step,
                'validation_loss': final_val_loss,
                'training_loss': np.mean(self.losses[-100:]) if len(self.losses) >= 100 else np.mean(self.losses),
                'is_final': True
            }
            validation_history.append(final_validation_record)
            
            if self.is_main_process:
                print(f"\nFinal Validation Loss: {final_val_loss:.4f}")

            # Check if final model is better than best saved model
            if final_val_loss < best_val_loss and self.args.save_model:
                checkpoint_name = "best_model"
                if self.experiment_name:
                    checkpoint_name = f"{self.experiment_name}_best_model"
                
                checkpoint_dir = os.path.join(self.args.output_dir, checkpoint_name)
                self.save_checkpoint(checkpoint_dir)
                
                if self.is_main_process:
                    print(f"New best model saved after final evaluation to: {checkpoint_dir}")
        
        # Save training history
        if self.is_main_process:
            save_simple_training_history(self, training_history, validation_history)
        
        
def run_ddp(rank, world_size, args):
    """Run training with DDP on specific rank"""
    trainer = Cell2TextDDPTrainer(args, rank, world_size)
        
    try:
        trainer.load_tokenizer()
        trainer.setup_datasets()
        trainer.initialize_model()
        trainer.freeze_model_components()
        trainer.apply_lora_to_model()
        if trainer.is_main_process:
            trainer.print_model_parameters()
            
        trainer.setup_model_and_optimizer()
            
        if args.mode == "sanity":
            target_reached = trainer.sanity_train()
        else:
            trainer.full_train()
            target_reached = True
        
        
        print(f"I am proces with rank {rank}")
        # dist.barrier()                       # sync after printing
        # results = trainer.validate()
        # dist.barrier() 

        results = []   
        return target_reached, results
            
    finally:
        trainer.cleanup_distributed()
    


def main():
    """Main function for DeepSpeed training"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Handle experiment naming
    if args.experiment_name:
        # Create experiment-specific output directory
        base_output_dir = args.output_dir
        args.output_dir = os.path.join(base_output_dir, args.experiment_name)
        print(f"Using experiment name: {args.experiment_name}")
        print(f"Output directory: {args.output_dir}")
    else:
        # Generate default experiment name with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"{args.mode}_{args.projector}_{timestamp}"
        args.output_dir = os.path.join(args.output_dir, default_name)
        print(f"No experiment name provided. Using default: {default_name}")
        print(f"Output directory: {args.output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    
    # Determine world size
    world_size = torch.cuda.device_count() if torch.cuda.is_available() and not args.no_cuda else 1

    # Create trainer and run training
    trainer = Cell2TextDDPTrainer(args)
    
    if world_size > 1:
        # Multi-GPU training with DDP
        mp.spawn(run_ddp, args=(world_size, args), nprocs=world_size, join=True)
    else:
        # Single GPU/CPU training
        target_reached, results = run_ddp(0, 1, args)
    
    
    
    # Final summary (only on rank 0)
    if not trainer.is_main_process:
        print("\n" + "="*60)
        print(f"FINAL DEEPSPEED {args.mode.upper()} TRAINING RESULTS")
        print("="*60)
        print(f"\nKey metrics:")
        if trainer.losses:
            print(f" Final training loss: {trainer.losses[-1]:.4f}")
        if args.mode == "sanity":
            print(f" Target loss: {args.target_loss:.4f}")
            print(f" Target reached: {'✓' if target_reached else '✗'}")
        print(f" BLEU score: {results.get('bleu', 0):.4f}")
        print(f" Cell type accuracy: {results.get('cell_type_accuracy', 0):.4f}")
        print(f" Cell type F1: {results.get('cell_type_f1', 0):.4f}")
        
        # Save final results summary
        if args.save_results:
            summary_path = os.path.join(args.output_dir, f"{args.mode}_training_summary.json")
            summary = {
                "mode": args.mode,
                "target_reached": target_reached,
                "final_training_loss": trainer.losses[-1] if trainer.losses else None,
                "target_loss": args.target_loss if args.mode == "sanity" else None,
                "evaluation_results": results,
                "training_args": vars(args)
            }
            with open(summary_path, 'w') as f:
                json.dump(convert_json_compat(summary), f, indent=2)
           
            print(f"\nTraining summary saved to: {summary_path}")


if __name__ == "__main__":
    main()