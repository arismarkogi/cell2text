import torch
import pandas as pd
import numpy as np
import os
import argparse
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
import random
import json
from torch.utils.data.distributed import DistributedSampler


# Accelerate imports
from accelerate import Accelerator
from accelerate.utils import set_seed

# LoRA imports
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    PeftModel,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cell2text_model.model import Cell2TextModel
from cell2text_model.configuration import Cell2TextConfig
from cell2text_dataset.dataset import Cell2TextDataset
from cell2text_eval.evaluation import evaluate_cell2text_model


class SanityDataset(Dataset):
    """Wrapper to create a small subset of data for sanity check"""
    
    def __init__(self, full_dataset, num_samples=8, seed=42):
        self.full_dataset = full_dataset
        self.num_samples = min(num_samples, len(full_dataset))
        
        # Set seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        # Select random indices
        self.indices = random.sample(range(len(full_dataset)), self.num_samples)
        print(f"Selected {self.num_samples} samples for sanity check: {self.indices}")
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.full_dataset[self.indices[idx]]


class Cell2TextDeepSpeedTrainer:
    """DeepSpeed-enabled trainer for Cell2Text model with support for both sanity check and full training"""
    
    def __init__(self, args):
        self.args = args
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

        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision='fp16' if args.fp16 else ('bf16' if args.bf16 else 'no'),
            log_with=None,  # Add logging if needed
        )
        self.device = self.accelerator.device
         
        
    def setup_device(self):
        """Setup device for training with DeepSpeed"""
        # DeepSpeed handles device placement automatically
        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")
        print(f"DeepSpeed will handle device placement. Local device: {self.device}")
        
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
        """Setup training and validation datasets"""
        print(f"Loading dataset...")
        
        # Load full training dataset
        if self.args.projector == "mlp":
            top_k = self.args.top_k
        else:  # perceiver
            top_k = None  # Perceiver doesn't need top_k selection
            
        full_train_dataset = Cell2TextDataset(self.args.train_data_path,
                                               self.tokenizer, top_k=top_k, 
                                               projector=self.args.projector, 
                                               num_latents=self.args.num_latents)
        
        use_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
        train_sampler = DistributedSampler(full_train_dataset, shuffle=True) if use_ddp else None
        
        if self.args.mode == "sanity":
            # Create small subset for sanity check
            sanity_dataset = SanityDataset(full_train_dataset, num_samples=self.args.num_samples, seed=self.args.seed)
            
            self.train_loader = DataLoader(
                sanity_dataset, 
                batch_size=self.args.batch_size_per_device, 
                shuffle=(train_sampler is None),      
                sampler=train_sampler,                
                collate_fn=full_train_dataset.collate_fn(mode="train")
            )
            print(f"Sanity dataset loaded. Size: {len(sanity_dataset)}")
            
            # Use the same small dataset for validation
            self.val_loader = DataLoader(
                sanity_dataset, 
                batch_size=self.args.batch_size_per_device, 
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                collate_fn=full_train_dataset.collate_fn(mode="inference")
            )
        else:
            # Full training mode
            self.train_loader = DataLoader(
                full_train_dataset, 
                batch_size=self.args.batch_size_per_device, 
                shuffle=True,
                collate_fn=full_train_dataset.collate_fn(mode="train")
            )
            print(f"Full training dataset loaded. Size: {len(full_train_dataset)}")
            
            # Load validation dataset if provided
            if self.args.val_data_path:
                val_dataset = Cell2TextDataset(self.args.val_data_path, 
                                               self.tokenizer,
                                                 top_k=top_k, 
                                                 projector=self.args.projector, 
                                                 num_latents=self.args.num_latents)
                
                val_sampler = DistributedSampler(val_dataset, shuffle=False) if use_ddp else None
                self.val_loader = DataLoader(
                    val_dataset, 
                    batch_size=self.args.batch_size_per_device, 
                    shuffle=False,
                    collate_fn=val_dataset.collate_fn(mode="inference")
                )
                print(f"Validation dataset loaded. Size: {len(val_dataset)}")
            else:
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
            config.perceiver_depth = self.args.perceiver_depth
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
        """Setup model, optimizer, and prepare with Accelerate"""
        # Create optimizer with different parameter groups
        parameters = []
        
        # Projector parameters
        projector_name = "cell_to_embedding" if self.args.projector == "mlp" else "perceiver_projector"
        projector_module = getattr(self.model, projector_name, None)
        
        if projector_module:
            for name, param in projector_module.named_parameters():
                if param.requires_grad:
                    parameters.append({
                        "params": [param],
                        "lr": self.args.projector_lr
                    })
        
        # Decoder parameters
        for name, param in self.model.decoder.named_parameters():
            if param.requires_grad:
                parameters.append({
                    "params": [param],
                    "lr": self.args.decoder_lr
                })
        
        # Create optimizer
        self.optimizer = AdamW(
            parameters,
            lr=self.args.decoder_lr,  # Default LR
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
        
        # Prepare everything with Accelerate
        self.model, self.optimizer, self.train_loader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.lr_scheduler
        )
        
        if self.val_loader:
            self.val_loader = self.accelerator.prepare(self.val_loader)
        
        
    def train_step(self, batch):
        """Perform a single training step with Accelerate"""
        with self.accelerator.accumulate(self.model):
            # Forward pass
            outputs = self.model(
                expression_tokens=batch["expression_tokens"],
                expression_token_lengths=batch["expression_token_lengths"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                return_dict=True
            )
            
            loss = outputs.loss
            
            # Backward pass
            self.accelerator.backward(loss)
            
            # Gradient clipping
            if self.args.max_grad_norm > 0:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            
            # Optimizer step
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
        
        self.global_step += 1
        return loss.item()
    
    def validate(self):
        """Run validation using the full evaluation function"""
        if self.val_loader is None:
            return None
        
        results = evaluate_cell2text_model(
            model=self.model,  
            val_loader=self.val_loader,
            tokenizer=self.tokenizer,
            device=self.accelerator.device,
            accelerator=self.accelerator,  
            print_examples=10,
            save_results=None
        )
        
        return results.get('validation_loss', None)
        
    def sanity_train(self):
        """Sanity training loop - overfit on small dataset with DeepSpeed"""
        print("="*60)
        print("STARTING DEEPSPEED SANITY CHECK TRAINING")
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
        
        print("\nStarting overfitting training with DeepSpeed...")
        progress_bar = tqdm(
            desc="🚀 DeepSpeed Sanity Training", 
            total=self.args.epochs, 
            position=0,
            leave=True,
            file=sys.stdout,
            disable=not (torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True)
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
            
            # Only update progress bar on rank 0
            if not self.accelerator.is_main_process:
                progress_bar.update(1)
                progress_bar.set_description(
                    f"🚀 DeepSpeed Sanity Training | 📊 Epoch: {epoch+1}/{self.args.epochs} | "
                    f"📉 Loss: {epoch_loss:.4f} | 🎯 Target: {self.args.target_loss:.4f}"
                )

            epoch += 1

            if epoch_loss <= self.args.target_loss:
                if not self.accelerator.is_main_process:
                    print(f"\n🎉 Target loss {self.args.target_loss:.4f} reached at epoch {epoch}!")
                    print(f"Final epoch loss: {epoch_loss:.4f}")
                break
        
        progress_bar.close()
        
        final_loss = epoch_loss
        if not self.accelerator.is_main_process:
            print(f"\nDeepSpeed sanity training completed!")
            print(f"Final loss: {final_loss:.4f}")
            print(f"Target reached: {'✓' if final_loss <= self.args.target_loss else '✗'}")
        
        if self.args.save_model:
            checkpoint_name = "overfitted_sanity_model"
            if self.experiment_name:
                checkpoint_name = f"{self.experiment_name}_overfitted_sanity_model"
            
            checkpoint_dir = os.path.join(self.args.output_dir, checkpoint_name)
            
            self.accelerator.save_model(self.model, checkpoint_dir)

            checkpoint_state = {
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'global_step': self.global_step,
                'losses': self.losses
            }
            self.accelerator.save(checkpoint_state, os.path.join(checkpoint_dir, "training_state.pt"))
            
            # Save training info with experiment context (only on rank 0)
            if not self.accelerator.is_main_process:
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

                def convert_json_compat(obj):
                    if isinstance(obj, dict):
                        return {k: convert_json_compat(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_json_compat(v) for v in obj]
                    elif isinstance(obj, (np.float32, np.float64, np.floating)):
                        return float(obj)
                    elif isinstance(obj, (np.int32, np.int64, np.integer)):
                        return int(obj)
                    elif isinstance(obj, np.bool_):
                        return bool(obj)
                    else:
                        return obj

                with open(info_path, 'w') as f:
                    json.dump(convert_json_compat(info_dict), f, indent=2)
                            
                print(f"Overfitted model saved to: {checkpoint_dir}")
        
        return final_loss <= self.args.target_loss
    
    def full_train(self):
        """Full training loop with DeepSpeed and validation score storage"""
        print("="*60)
        print("STARTING DEEPSPEED FULL TRAINING")
        print("="*60)
        print(f"Epochs: {self.args.epochs}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Batch size per device: {self.args.batch_size_per_device}")
        print(f"Gradient accumulation steps: {self.args.gradient_accumulation_steps}")
        print(f"Projector type: {self.args.projector}")
        print(f"Validation every: {self.args.eval_steps} steps")
        print("="*60)
        
        self.model.train()
        

        # put this once at the top of full_train()
        is_distributed = torch.distributed.is_initialized()
        rank          = torch.distributed.get_rank() if is_distributed else 0
        is_main       = (rank == 0)


        # Initialize validation tracking
        validation_history = []
        training_history = []
        
        total_steps = self.args.epochs * len(self.train_loader)
        progress_bar = tqdm(
            desc="🚀 DeepSpeed Full Training", 
            total=total_steps,
            position=0,
            leave=True,
            file=sys.stdout,
            disable=not (torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True)
        )
        
        best_val_loss = float('inf')
        best_val_bleu = 0.0
        best_val_cell_type_acc = 0.0
        best_val_cell_type_f1 = 0.0
        steps_since_improvement = 0
        
        for epoch in range(self.args.epochs):

            if isinstance(self.train_loader.sampler, DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)
            
            epoch_losses = []
            
            for step, batch in enumerate(self.train_loader):
                loss = self.train_step(batch)
                epoch_losses.append(loss)
                self.losses.append(loss)
                
                # Store training step info
                training_step_info = {
                    'epoch': epoch + 1,
                    'step': step + 1,
                    'global_step': self.global_step,
                    'loss': loss,
                    'learning_rate': self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else self.args.decoder_lr
                }
                training_history.append(training_step_info)
                
                # Update progress bar
                if not self.accelerator.is_main_process:
                    progress_bar.update(1)
                    progress_bar.set_description(
                        f"🚀 Full Training | Epoch: {epoch+1}/{self.args.epochs} | "
                        f"Step: {step+1}/{len(self.train_loader)} | Loss: {loss:.4f}"
                    )
                
                # Validation
                if self.global_step % self.args.eval_steps == 0 and self.val_loader is not None:
                    

                    if is_main:                          # ← only rank-0 runs it
                        self.model.eval()
                        val_results = self.validate()
                    else:
                        val_results = None
                    
                   
                    if not is_main:
                        continue
                    # Extract validation metrics
                    val_loss = val_results if isinstance(val_results, float) else val_results.get('validation_loss', None)
                    val_bleu = val_results.get('bleu', 0.0) if isinstance(val_results, dict) else 0.0
                    val_cell_type_acc = val_results.get('cell_type_accuracy', 0.0) if isinstance(val_results, dict) else 0.0
                    val_cell_type_f1 = val_results.get('cell_type_f1', 0.0) if isinstance(val_results, dict) else 0.0
                    val_cell_type_precision = val_results.get('cell_type_precision', 0.0) if isinstance(val_results, dict) else 0.0
                    val_cell_type_recall = val_results.get('cell_type_recall', 0.0) if isinstance(val_results, dict) else 0.0
                    
                    # Store validation results
                    validation_record = {
                        'epoch': epoch + 1,
                        'step': step + 1,
                        'global_step': self.global_step,
                        'validation_loss': val_loss,
                        'bleu_score': val_bleu,
                        'cell_type_accuracy': val_cell_type_acc,
                        'cell_type_f1': val_cell_type_f1,
                        'cell_type_precision': val_cell_type_precision,
                        'cell_type_recall': val_cell_type_recall,
                        'training_loss': np.mean(self.losses[-self.args.eval_steps:]) if len(self.losses) >= self.args.eval_steps else np.mean(self.losses)
                    }
                    validation_history.append(validation_record)
                    
                    if not self.accelerator.is_main_process:
                        print(f"\nValidation at step {self.global_step}:")
                        if val_loss is not None:
                            print(f"  Loss: {val_loss:.4f}")
                        print(f"  BLEU: {val_bleu:.4f}")
                        print(f"  Cell Type Acc: {val_cell_type_acc:.4f}")
                        print(f"  Cell Type F1: {val_cell_type_f1:.4f}")
                    
                    # Early stopping and best model tracking
                    improved = False
                    
                    # Use validation loss as primary metric if available, otherwise use BLEU
                    if val_loss is not None and val_loss < best_val_loss:
                        best_val_loss = val_loss
                        improved = True
                    elif val_loss is None and val_bleu > best_val_bleu:
                        best_val_bleu = val_bleu
                        improved = True
                    
                    # Also track best scores for other metrics
                    if val_bleu > best_val_bleu:
                        best_val_bleu = val_bleu
                    if val_cell_type_acc > best_val_cell_type_acc:
                        best_val_cell_type_acc = val_cell_type_acc
                    if val_cell_type_f1 > best_val_cell_type_f1:
                        best_val_cell_type_f1 = val_cell_type_f1
                    

                    
                    if improved:
                        steps_since_improvement = 0
                        
                        if self.args.save_model:
                            checkpoint_name = "best_model"
                            if self.experiment_name:
                                checkpoint_name = f"{self.experiment_name}_best_model"
                            
                            checkpoint_dir = os.path.join(self.args.output_dir, checkpoint_name)
                            
                            self.accelerator.save_model(self.model, checkpoint_dir)

                            # For additional state:
                            checkpoint_state = {
                                'optimizer': self.optimizer.state_dict(),
                                'lr_scheduler': self.lr_scheduler.state_dict(),
                                'global_step': self.global_step,
                                'losses': self.losses
                            }
                            self.accelerator.save(checkpoint_state, os.path.join(checkpoint_dir, "training_state.pt"))
                            
                            # On Rank 0, save supplementary files
                            if not self.accelerator.is_main_process:
                                # Save validation history
                                validation_history_path = os.path.join(checkpoint_dir, "validation_history.json")
                                with open(validation_history_path, 'w') as f:
                                    json.dump(self._convert_json_compat(validation_history), f, indent=2)
                                
                                print(f"Best model checkpoint saved to: {checkpoint_dir}")
                                print(f"Validation history saved to: {validation_history_path}")


                    else:
                        steps_since_improvement += self.args.eval_steps
                        
                        if self.args.early_stopping > 0 and steps_since_improvement >= self.args.early_stopping:
                            if not self.accelerator.is_main_process:
                                print(f"\nEarly stopping triggered after {steps_since_improvement} steps without improvement")
                            progress_bar.close()
                            
                            # Save final training and validation history
                            self._save_training_history(training_history, validation_history)
                            return
                    
                    self.model.train()  # Switch back to training mode
            
            # End of epoch summary
            epoch_loss = np.mean(epoch_losses)
            if not self.accelerator.is_main_process:
                print(f"\nEpoch {epoch+1} completed. Average loss: {epoch_loss:.4f}")
        
        progress_bar.close()
        
        # Final evaluation before closing
        final_val_results = None
        if self.val_loader is not None:
            self.model.eval()
            final_val_results = self.validate()
            
            # Extract final validation metrics
            final_val_loss = final_val_results if isinstance(final_val_results, float) else final_val_results.get('validation_loss', None)
            final_val_bleu = final_val_results.get('bleu', 0.0) if isinstance(final_val_results, dict) else 0.0
            final_val_cell_type_acc = final_val_results.get('cell_type_accuracy', 0.0) if isinstance(final_val_results, dict) else 0.0
            final_val_cell_type_f1 = final_val_results.get('cell_type_f1', 0.0) if isinstance(final_val_results, dict) else 0.0
            final_val_cell_type_precision = final_val_results.get('cell_type_precision', 0.0) if isinstance(final_val_results, dict) else 0.0
            final_val_cell_type_recall = final_val_results.get('cell_type_recall', 0.0) if isinstance(final_val_results, dict) else 0.0
            
            # Add final validation to history
            final_validation_record = {
                'epoch': self.args.epochs,
                'step': 'final',
                'global_step': self.global_step,
                'validation_loss': final_val_loss,
                'bleu_score': final_val_bleu,
                'cell_type_accuracy': final_val_cell_type_acc,
                'cell_type_f1': final_val_cell_type_f1,
                'cell_type_precision': final_val_cell_type_precision,
                'cell_type_recall': final_val_cell_type_recall,
                'training_loss': np.mean(self.losses[-100:]) if len(self.losses) >= 100 else np.mean(self.losses),
                'is_final': True
            }
            validation_history.append(final_validation_record)
            
            if not self.accelerator.is_main_process:
                print(f"\nFinal Validation Results:")
                if final_val_loss is not None:
                    print(f"  Loss: {final_val_loss:.4f}")
                print(f"  BLEU: {final_val_bleu:.4f}")
                print(f"  Cell Type Acc: {final_val_cell_type_acc:.4f}")
                print(f"  Cell Type F1: {final_val_cell_type_f1:.4f}")

            # Check if final model is better than the best saved model
            final_improved = False
            if final_val_loss is not None and final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                final_improved = True
            elif final_val_loss is None and final_val_bleu > best_val_bleu:
                best_val_bleu = final_val_bleu
                final_improved = True
                
            if final_improved and self.args.save_model:
                checkpoint_name = "best_model"
                if self.experiment_name:
                    checkpoint_name = f"{self.experiment_name}_best_model"
                
                checkpoint_dir = os.path.join(self.args.output_dir, checkpoint_name)
                
                self.accelerator.save_model(self.model, checkpoint_dir)

                # For additional state:
                checkpoint_state = {
                    'optimizer': self.optimizer.state_dict(),
                    'lr_scheduler': self.lr_scheduler.state_dict(),
                    'global_step': self.global_step,
                    'losses': self.losses
                }
                self.accelerator.save(checkpoint_state, os.path.join(checkpoint_dir, "training_state.pt"))
                
                if not self.accelerator.is_main_process:
                    print(f"New best model saved after final evaluation to: {checkpoint_dir}")
            elif not final_improved:
                if not self.accelerator.is_main_process:
                    print("Final model did not outperform the best model. No new best model saved.")
        else:
            print("No validation loader available, skipping final evaluation.")
        
        # Save complete training and validation history
        self._save_training_history(training_history, validation_history)
        
        if not self.accelerator.is_main_process:
            print(f"\nFull training completed!")
            print(f"Best validation loss: {best_val_loss:.4f}")
            print(f"Best BLEU score: {best_val_bleu:.4f}")
            print(f"Best cell type accuracy: {best_val_cell_type_acc:.4f}")
            print(f"Best cell type F1: {best_val_cell_type_f1:.4f}")

    def _save_training_history(self, training_history, validation_history):
        """Save training and validation history to files with enhanced metrics tracking"""
        if not self.accelerator.is_main_process:
            # Enhanced training history with loss progression
            enhanced_training_history = {
                'training_steps': training_history,
                'loss_progression': {
                    'all_losses': self.losses,
                    'loss_statistics': {
                        'initial_loss': self.losses[0] if self.losses else None,
                        'final_loss': self.losses[-1] if self.losses else None,
                        'min_loss': min(self.losses) if self.losses else None,
                        'max_loss': max(self.losses) if self.losses else None,
                        'avg_loss': np.mean(self.losses) if self.losses else None,
                        'loss_std': np.std(self.losses) if self.losses else None,
                        'total_steps': len(self.losses)
                    }
                }
            }
            
            # Enhanced validation history with metric progressions
            enhanced_validation_history = {
                'validation_steps': validation_history,
                'metric_progressions': {}
            }
            
            if validation_history:
                # Extract metric progressions
                metrics_to_track = [
                    'validation_loss', 'bleu_score', 'cell_type_accuracy', 
                    'cell_type_f1', 'cell_type_precision', 'cell_type_recall'
                ]
                
                for metric in metrics_to_track:
                    values = [entry.get(metric) for entry in validation_history if entry.get(metric) is not None]
                    if values:
                        enhanced_validation_history['metric_progressions'][metric] = {
                            'all_values': values,
                            'statistics': {
                                'initial_value': values[0],
                                'final_value': values[-1],
                                'best_value': min(values) if 'loss' in metric else max(values),
                                'worst_value': max(values) if 'loss' in metric else min(values),
                                'avg_value': np.mean(values),
                                'std_value': np.std(values),
                                'total_evaluations': len(values),
                                'improvement': values[-1] - values[0] if len(values) > 1 else 0
                            }
                        }
            
            # Save enhanced histories
            training_history_path = os.path.join(self.args.output_dir, "enhanced_training_history.json")
            with open(training_history_path, 'w') as f:
                json.dump(self._convert_json_compat(enhanced_training_history), f, indent=2)
            
            validation_history_path = os.path.join(self.args.output_dir, "enhanced_validation_history.json")
            with open(validation_history_path, 'w') as f:
                json.dump(self._convert_json_compat(enhanced_validation_history), f, indent=2)
            
            # Save summary statistics (enhanced version)
            summary_stats = self._compute_enhanced_training_summary(enhanced_training_history, enhanced_validation_history)
            summary_path = os.path.join(self.args.output_dir, "enhanced_training_summary_stats.json")
            with open(summary_path, 'w') as f:
                json.dump(self._convert_json_compat(summary_stats), f, indent=2)
            
            print(f"Enhanced training history saved to: {training_history_path}")
            print(f"Enhanced validation history saved to: {validation_history_path}")
            print(f"Enhanced training summary stats saved to: {summary_path}")


    def _compute_enhanced_training_summary(self, enhanced_training_history, enhanced_validation_history):
        """Compute enhanced summary statistics from training and validation history"""
        summary = {
            'training_stats': {},
            'validation_stats': {},
            'best_metrics': {},
            'training_progression': {},
            'validation_progression': {}
        }
        
        # Enhanced training stats
        if enhanced_training_history.get('loss_progression'):
            loss_stats = enhanced_training_history['loss_progression']['loss_statistics']
            summary['training_stats'] = loss_stats.copy()
            
            # Add progression analysis
            if enhanced_training_history['loss_progression']['all_losses']:
                losses = enhanced_training_history['loss_progression']['all_losses']
                summary['training_progression'] = {
                    'loss_trend': 'decreasing' if losses[-1] < losses[0] else 'increasing',
                    'loss_reduction_percentage': ((losses[0] - losses[-1]) / losses[0] * 100) if losses[0] != 0 else 0,
                    'convergence_point': self._find_convergence_point(losses)
                }
        
        # Enhanced validation stats
        if enhanced_validation_history.get('metric_progressions'):
            metric_progs = enhanced_validation_history['metric_progressions']
            
            summary['validation_stats'] = {}
            summary['validation_progression'] = {}
            summary['best_metrics'] = {}
            
            for metric, data in metric_progs.items():
                stats = data['statistics']
                summary['validation_stats'][f'{metric}_stats'] = stats
                summary['best_metrics'][f'best_{metric}'] = stats['best_value']
                
                # Progression analysis
                if len(data['all_values']) > 1:
                    values = data['all_values']
                    summary['validation_progression'][f'{metric}_trend'] = {
                        'direction': 'improving' if stats['improvement'] > 0 and 'loss' not in metric else 
                                   'improving' if stats['improvement'] < 0 and 'loss' in metric else 'degrading',
                        'improvement_percentage': abs(stats['improvement'] / values[0] * 100) if values[0] != 0 else 0,
                        'stability': 'stable' if stats['std_value'] < (stats['avg_value'] * 0.1) else 'unstable'
                    }
        
        return summary
    
    def _find_convergence_point(self, losses, window_size=100, threshold=0.01):
        """Find the point where loss starts to converge"""
        if len(losses) < window_size * 2:
            return None
        
        for i in range(window_size, len(losses) - window_size):
            window1 = losses[i-window_size:i]
            window2 = losses[i:i+window_size]
            
            avg1 = np.mean(window1)
            avg2 = np.mean(window2)
            
            if abs(avg1 - avg2) / avg1 < threshold:
                return i
        
        return None
        
    def run_evaluation(self):
        """Run evaluation"""
        if not self.accelerator.is_main_process:
            print("\n" + "="*60)
            print("RUNNING EVALUATION")
            print("="*60)
        
        if self.val_loader is None:
            print("No validation dataset available for evaluation.")
            return {}
        
        results_filename = f"{self.args.mode}_evaluation_results.json"
        
        if self.experiment_name:
            results_filename = f"{self.experiment_name}_{results_filename}"
        
        results = evaluate_cell2text_model(
            model=self.model_engine.module,
            val_loader=self.val_loader,
            tokenizer=self.tokenizer,
            device=self.model_engine.device,
            print_examples=self.args.num_samples if self.args.mode == "sanity" and (not self.accelerator.is_main_process) else 8,
            save_results=os.path.join(self.args.output_dir, results_filename) if self.args.save_results else None
        )
        
        print("\n" + "="*60)
        print(f"DEEPSPEED {self.args.mode.upper()} TRAINING SUMMARY")
        print("="*60)
        print(f"Final training loss: {self.losses[-1] if self.losses else 'N/A':.4f}")
        if self.args.mode == "sanity":
            print(f"Target loss: {self.args.target_loss:.4f}")
            print(f"Target reached: {'✓' if self.losses and self.losses[-1] <= self.args.target_loss else '✗'}")
        print(f"BLEU score: {results.get('bleu', 0):.4f}")
        print(f"Cell type accuracy: {results.get('cell_type_accuracy', 0):.4f}")
        print(f"Cell type F1: {results.get('cell_type_f1', 0):.4f}")
        
        return results
        
    def run(self):
        """Run the complete training pipeline with DeepSpeed"""
        if not self.accelerator.is_main_process:
            print(f"Starting DeepSpeed {self.args.mode} training pipeline...")
        
            """Main training method"""
            self.setup_device()  # This now just sets up accelerator
            self.load_tokenizer()
            self.setup_datasets()
            self.initialize_model()
            self.freeze_model_components()
            self.apply_lora_to_model()
            self.print_model_parameters()
            
            # Replace setup_deepspeed_model() with:
            self.setup_model_and_optimizer()
            
            if self.args.mode == "sanity":
                target_reached= self.sanity_train()
            else:
                self.full_train()
                target_reached = True
        
        # Evaluate
        results = self.run_evaluation()
        
        return target_reached, results
    
    def _convert_json_compat(self, obj):
        if isinstance(obj, dict):
            return {k: self._convert_json_compat(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_json_compat(v) for v in obj]
        elif isinstance(obj, (np.float32, np.float64, np.floating)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64, np.integer)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj



def create_argument_parser():
    """Create and return the argument parser for DeepSpeed training"""
    parser = argparse.ArgumentParser(description="DeepSpeed training for Cell2Text model")

     # Add experiment naming parameter
    parser.add_argument("--experiment_name", type=str, required=True,
                        help="Name for the experiment (will be used in output directory and file names)")
    
    # Mode selection
    parser.add_argument("--mode", type=str, choices=["sanity", "full"], default="sanity",
                        help="Training mode: 'sanity' for sanity check or 'full' for full training")
    
    # Data parameters
    parser.add_argument("--train_data_path", type=str, required=True,
                        help="Path to the parquet file containing training data")
    parser.add_argument("--val_data_path", type=str, default=None,
                        help="Path to the parquet file containing validation data (for full training)")
    parser.add_argument("--output_dir", type=str, default="./deepspeed_training",
                        help="Directory to save training results")
    
    # Sanity check specific parameters
    parser.add_argument("--num_samples", type=int, default=8,
                        help="Number of samples to overfit on")
    parser.add_argument("--target_loss", type=float, default=0.01,
                        help="Target loss to reach (default: 0.01)")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Training epochs")
    parser.add_argument("--save_overfitted_model", type=bool, default=True,
                        help="Save the overfitted model")
    parser.add_argument("--save_results", type=bool, default=True,
                        help="Save evaluation results to JSON")
    
    # Model parameters
    parser.add_argument("--encoder_hidden_size", type=int, default=1152,
                        help="Hidden size of the cell encoder")
    parser.add_argument("--mlp_hidden_size", type=int, default=1024,
                        help="Hidden size of the 2-layer MLP cell-to-embedding projector")
    parser.add_argument("--mlp_dropout", type=float, default=0.1, 
                        help="Dropout probability at MLP projector")
    parser.add_argument("--decoder_hidden_size", type=int, default=2048,
                        help="Hidden size of the decoder")
    parser.add_argument("--top_k", type=int, default=384,
                        help="select the top_k most expressed genes after the Geneformer encoder")
    parser.add_argument("--geneformer_path", type=str, required=True,
                        help="Path to pretrained Geneformer model")
    parser.add_argument("--decoder_path", type=str, required=True,
                        help="Path to pretrained Llama decoder model")
    parser.add_argument("--token_dictionary_path", type=str, required=True,
                        help="Path to geneformer Dictionary file")
    parser.add_argument("--max_ncells", type=int, default=1000,
                        help="Maximum number of cells")
    parser.add_argument("--max_length", type=int, default=500,
                        help="Maximum length of generated text")
    parser.add_argument("--num_beams", type=int, default=1,
                        help="Number of beams for beam search")
    
    # LoRA parameters
    parser.add_argument("--use_lora_decoder", type=bool, default=True,
                        help="Use LoRA for the text decoder (LLaMA)")
    
    # LoRA parameters for decoder
    parser.add_argument("--lora_r_decoder", type=int, default=16,
                        help="LoRA rank for decoder")
    parser.add_argument("--lora_alpha_decoder", type=int, default=32,
                        help="LoRA alpha for decoder")
    parser.add_argument("--lora_dropout_decoder", type=float, default=0.1,
                        help="LoRA dropout for decoder")
    parser.add_argument("--lora_target_modules_decoder", type=str, nargs="+", 
                        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                        help="Target modules for LoRA in decoder (LLaMA modules)")
    parser.add_argument("--lora_bias_decoder", type=str, default="none",
                        choices=["none", "all", "lora_only"],
                        help="LoRA bias type for decoder")
    parser.add_argument("--lora_modules_to_save_decoder", type=str, nargs="*", default=None,
                        help="Additional modules to save for decoder LoRA")
    
    # Freezing parameters
    parser.add_argument("--freeze_decoder", type=bool, default=True,
                        help="Freeze decoder parameters (only applies if not using LoRA for decoder)")
    
    # DeepSpeed specific parameters
    parser.add_argument("--batch_size_per_device", type=int, default=8,
                        help="Batch size per device/GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--zero_stage", type=int, default=2, choices=[0, 1, 2, 3],
                        help="DeepSpeed ZeRO optimization stage")
    parser.add_argument("--fp16", type=bool, default=False,
                        help="Enable FP16 mixed precision training")
    parser.add_argument("--bf16", type=bool, default=False,
                        help="Enable BF16 mixed precision training")
    
    # Training parameters (optimized for sanity check)
    parser.add_argument("--decoder_lr", type=float, default=1e-4,
                        help="Learning rate for the decoder (higher for faster overfitting)")
    parser.add_argument("--projector_lr", type=float, default=1e-3,
                        help="Learning rate for the projection layer (higher for faster overfitting)")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay for AdamW optimizer (0 for easier overfitting)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for gradient clipping")
    parser.add_argument("--warmup_steps", type=int, default=10,
                        help="Number of warmup steps for learning rate schedule")

    # Misc parameters
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA even if available")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (automatically set by DeepSpeed)")
    
    # Projector type selection
    parser.add_argument("--projector", type=str, choices=["mlp", "perceiver"], default="mlp",
                        help="Type of projector to use: 'mlp' or 'perceiver'")
    
    # Perceiver-specific parameters
    parser.add_argument("--num_latents", type=int, default=64,
                        help="Number of latent vectors for Perceiver")
    parser.add_argument("--perceiver_depth", type=int, default=6,
                        help="Number of layers in Perceiver")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads in Perceiver")
    parser.add_argument("--ff_mult", type=float, default=4,
                        help="Feedforward multiplier in Perceiver")
    parser.add_argument("--perceiver_dropout", type=float, default=0.1,
                        help="Dropout probability in Perceiver")
    
    # Full training specific parameters
    parser.add_argument("--eval_steps", type=int, default=2000,
                        help="Number of steps between evaluations")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Number of steps between saving checkpoints")
    parser.add_argument("--early_stopping", type=int, default=0,
                        help="Number of steps without improvement before early stopping (0 to disable)")
    parser.add_argument("--save_model", type=bool, default=True,
                        help="Save model checkpoints")
    
    return parser


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
    
    # Create trainer and run training
    trainer = Cell2TextDeepSpeedTrainer(args)
    target_reached, results = trainer.run()
    
    # Final summary (only on rank 0)
    if not trainer.accelerator.is_main_process:
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
            def convert_json_compat(obj):
                if isinstance(obj, dict):
                    return {k: convert_json_compat(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_json_compat(v) for v in obj]
                elif isinstance(obj, (np.float32, np.float64, np.floating)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64, np.integer)):
                    return int(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                else:
                    return obj

            with open(summary_path, 'w') as f:
                json.dump(convert_json_compat(summary), f, indent=2)
           
            print(f"\nTraining summary saved to: {summary_path}")


if __name__ == "__main__":
    main()