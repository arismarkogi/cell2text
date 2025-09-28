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
from torch.distributed.checkpoint.state_dict import StateDictType
import pickle
from tqdm import tqdm
import sys
import json
from torch.utils.data.distributed import DistributedSampler

from torch.distributed.fsdp import (
    FullyShardedDataParallel, 
    FullStateDictConfig, 
    StateDictType
)

# FSDP imports 
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, BackwardPrefetch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import CPUOffload

from util import create_argument_parser, create_progress_bar, save_simple_training_history

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cell2text_model.model import Cell2TextModel
from cell2text_model.configuration import Cell2TextConfig
from cell2text_dataset.dataset import Cell2TextDataset, SanityDataset


class Cell2TextFSDPTrainer:
    """Simple FSDP-enabled trainer for Cell2Text model"""
    
    def __init__(self, args, rank=0, world_size=1):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = (rank == 0)
        self.experiment_name = getattr(args, 'experiment_name', None)
        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        
        # Initialize training state
        self.model = None
        self.tokenizer = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.lr_scheduler = None
        self.global_step = 0
        self.losses = []

        # Setup distributed training if needed
        if world_size > 1:
            self._setup_distributed()
        
    def _setup_distributed(self):
        """Simple distributed setup"""
        os.environ.setdefault('MASTER_ADDR', 'localhost')
        os.environ.setdefault('MASTER_PORT', '12355')
        
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            rank=self.rank,
            world_size=self.world_size
        )
        torch.cuda.set_device(self.rank)
        
    def load_tokenizer(self):
        """Load tokenizer for text descriptions"""
        if self.is_main_process:
            print("Loading tokenizer...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.decoder_path, 
                pad_token='<|reserved_special_token_0|>'
            )
            if self.is_main_process:
                print("Tokenizer loaded successfully.")
        except Exception as e:
            if self.is_main_process:
                print(f"Error loading tokenizer: {e}")
            self.tokenizer = None
            
    def setup_datasets(self):
        """Setup training and validation datasets"""
        if self.is_main_process:
            print(f"Loading dataset...")
        
        # Determine top_k based on projector type
        top_k = self.args.top_k if self.args.projector in ["mlp", "qformer"] else None
            
        # Load training dataset
        full_train_dataset = Cell2TextDataset(
            self.args.train_data_path,
            self.tokenizer, 
            top_k=top_k, 
            projector=self.args.projector, 
            num_latents=self.args.num_latents,
            sort_by_depth=self.args.sort_by_depth
        )
        
        if self.args.mode == "sanity":
            # Create small subset for sanity check
            dataset = SanityDataset(full_train_dataset, num_samples=self.args.num_samples, seed=self.args.seed)
            val_dataset = dataset  # Use same dataset for validation in sanity mode
        else:
            dataset = full_train_dataset
            # Load validation dataset if provided
            val_dataset = None
            if self.args.val_data_path:
                val_dataset = Cell2TextDataset(
                    self.args.val_data_path, 
                    self.tokenizer,
                    top_k=top_k, 
                    projector=self.args.projector, 
                    num_latents=self.args.num_latents
                )
        
        # Setup data loaders with distributed samplers
        train_sampler = None
        val_sampler = None
        
        if self.world_size > 1:
            train_sampler = DistributedSampler(
                dataset, 
                num_replicas=self.world_size, 
                rank=self.rank,
                shuffle=True
            )
            if val_dataset is not None:
                val_sampler = DistributedSampler(
                    val_dataset, 
                    num_replicas=self.world_size, 
                    rank=self.rank,
                    shuffle=False
                )
        
        self.train_loader = DataLoader(
            dataset, 
            batch_size=self.args.batch_size_per_device, 
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=0,
            collate_fn=dataset.collate_fn(mode="train")
        )
        
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset, 
                batch_size=self.args.batch_size_per_device, 
                shuffle=False,
                sampler=val_sampler,
                num_workers=0,
                collate_fn=val_dataset.collate_fn(mode="train")
            )
        else:
            self.val_loader = None
            
        if self.is_main_process:
            print(f"Training dataset size: {len(dataset)}")
            if val_dataset:
                print(f"Validation dataset size: {len(val_dataset)}")
        
    def initialize_model(self):
        """Initialize the Cell2Text model with configuration"""
        if self.is_main_process:
            print("Initializing model...")
        
        config = Cell2TextConfig()
        
        # Set configuration parameters
        config.cell_encoder_hidden_size = self.args.encoder_hidden_size
        config.decoder_hidden_size = self.args.decoder_hidden_size
        config.geneformer_path = self.args.geneformer_path
        config.decoder_model_name_or_path = self.args.decoder_path
        config.token_dictionary_path = self.args.token_dictionary_path
        config.projector = self.args.projector
        config.max_ncells = self.args.max_ncells
        config.max_new_tokens = self.args.max_length
        config.num_beams = self.args.num_beams
        
        # Projector-specific configuration
        if self.args.projector == "mlp":
            config.mlp_hidden_size = self.args.mlp_hidden_size
            config.mlp_dropout = self.args.mlp_dropout
            config.top_k = self.args.top_k
        elif self.args.projector == "perceiver":
            config.num_latents = self.args.num_latents
            config.num_heads = self.args.num_heads
            config.ff_mult = self.args.ff_mult
            config.perceiver_dropout = self.args.perceiver_dropout
        

        self.model = Cell2TextModel(config=config)
        # If you have a checkpoint, load it directly like the reference
        if hasattr(self.args, 'checkpoint_path') and self.args.checkpoint_path:
            checkpoint = torch.load(self.args.checkpoint_path, map_location="cpu")
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            self.model.load_state_dict(state_dict, strict=False)
        else:
            # Otherwise use warm_up
            self.model.warm_up()

        # Check dtypes before conversion
        print("Checking dtypes before conversion:")
        for name, param in self.model.named_parameters():
            if param.dtype != torch.float16:
                print(f"  {name}: {param.dtype}")
        
        self.model = self.model.to(torch.float32)
        
        # Check again after conversion
        print("Checking dtypes after conversion:")
        for name, param in self.model.named_parameters():
            if param.dtype != torch.float16:
                print(f"  {name}: {param.dtype}")
        
        #self.model = self.model.to(torch.float16)


    def setup_model_for_training(self):
        """Setup model with freezing and FSDP wrapping"""
        if self.is_main_process:
            print("Setting up model for training...")
        
        # 1. Freeze cell encoder
        for param in self.model.cell_encoder.parameters():
            param.requires_grad = False
        
        # DEBUG: Check decoder before FSDP
        if self.is_main_process:
            print(f"Before FSDP - decoder.llama is None: {self.model.decoder.llama is None}")
            if self.model.decoder.llama is not None:
                llama_params = sum(p.numel() for p in self.model.decoder.llama.parameters())
                print(f"Before FSDP - LLaMA parameters: {llama_params:,}")
        
        # 3. Move to device first
        self.model = self.model.to(self.device)

        if self.is_main_process:
            print("Parameter breakdown:")
            
            # Direct LLaMA access
            if hasattr(self.model.decoder, 'llama') and self.model.decoder.llama:
                direct_llama = sum(p.numel() for p in self.model.decoder.llama.parameters())
                print(f"  Direct decoder.llama: {direct_llama:,}")
            
            # Through wrapper
            wrapper_decoder = sum(p.numel() for p in self.model.decoder.parameters())
            print(f"  Through decoder wrapper: {wrapper_decoder:,}")
            
            # Full model
            total_model = sum(p.numel() for p in self.model.parameters())
            print(f"  Total model: {total_model:,}")
        
        # 4. Wrap with FSDP
        if self.world_size > 1:
            self.model = FSDP(
                self.model,
                mixed_precision=None,
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                device_id=self.rank,
                sync_module_states=True,
                use_orig_params=True,
            )
            
            # DEBUG: Check decoder after FSDP
            if not self.is_main_process:
                print(f"After FSDP - type of model: {type(self.model)}")
                print(f"After FSDP - decoder type: {type(self.model.decoder)}")
                
                # Check if we can access llama through FSDP
                try:
                    llama_accessible = self.model.decoder.llama is not None
                    print(f"After FSDP - decoder.llama accessible: {llama_accessible}")
                    if llama_accessible:
                        llama_params = sum(p.numel() for p in self.model.decoder.llama.parameters())
                        print(f"After FSDP - LLaMA parameters: {llama_params:,}")
                except Exception as e:
                    print(f"After FSDP - Error accessing decoder.llama: {e}")
        
       

    def _print_model_parameters(self):
        """Prints the names, shapes, and grad requirements of all model parameters, with a focus on the decoder."""
        print("="*60)
        print("MODEL PARAMETERS")
        print("="*60)
        
        decoder_params = []
        decoder_trainable_count = 0
        total_decoder_count = 0
        
       
      
            
        print("\n📊 **Decoder Parameter Summary**")
        print(f"  - Total Decoder Parameters: {total_decoder_count:,}")
        print(f"  - Trainable Decoder Parameters: {decoder_trainable_count:,}")
        if total_decoder_count > 0:
            print(f"  - Trainable %: {100 * decoder_trainable_count / total_decoder_count:.2f}%")
        print("="*60)

        for name, _ in self.model.named_parameters():
            print(name)
            

        

        
    
    def _print_trainable_parameters(self):
        """Print trainable parameter information"""
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"All parameters: {all_params:,}")
        print(f"Trainable %: {100 * trainable_params / all_params:.2f}%")
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        if self.is_main_process:
            print("Setting up optimizer...")
        
        model_for_params = self.model
        
        # Create parameter groups
        projector_params = []
        decoder_params = []
        
        if hasattr(model_for_params, "cell_to_embedding"):
            for param in getattr(model_for_params, "cell_to_embedding").parameters():
                if param.requires_grad:
                    projector_params.append(param)
        
        for param in model_for_params.decoder.parameters():
            if param.requires_grad:
                decoder_params.append(param)
        
        param_groups = []
        if projector_params:
            param_groups.append({"params": projector_params})
        if decoder_params:
            param_groups.append({"params": decoder_params, "lr": self.args.decoder_lr})
        
        self.optimizer = AdamW(
            param_groups if param_groups else self.model.parameters(),
            lr=1e-7,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        total_steps = self.args.epochs * len(self.train_loader) // self.args.gradient_accumulation_steps
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=total_steps
        )
        
        if self.is_main_process:
            print(f"Optimizer created with {len(param_groups)} parameter groups")

    
    def train_step(self, batch):
        """Single training step"""
        # Move batch to device and ensure correct dtypes
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

       
    
        if "expression_token_lengths" in batch and torch.is_tensor(batch["expression_token_lengths"]):
            batch["expression_token_lengths"] = batch["expression_token_lengths"].long()
        
        # Ensure expression_tokens are properly cast to long integers
        if "expression_tokens" in batch and torch.is_tensor(batch["expression_tokens"]):
            batch["expression_tokens"] = batch["expression_tokens"].long()
        
        # Forward pass
        outputs = self.model(
            expression_tokens=batch["expression_tokens"],
            expression_token_lengths=batch["expression_token_lengths"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            return_dict=True
        )
        
        loss = outputs.loss / self.args.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        
        # Optimizer step (handle gradient accumulation)
        if (self.global_step + 1) % self.args.gradient_accumulation_steps == 0:
            # Clip gradients BEFORE checking for NaN
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)

            
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
        
        self.global_step += 1
        return loss.item() * self.args.gradient_accumulation_steps

    def validate(self):
        """Validation step"""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                
                # Ensure expression_tokens are properly cast to long integers
                if "expression_tokens" in batch and torch.is_tensor(batch["expression_tokens"]):
                    batch["expression_tokens"] = batch["expression_tokens"].long()
                
                outputs = self.model(
                    expression_tokens=batch["expression_tokens"],
                    expression_token_lengths=batch["expression_token_lengths"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    return_dict=True
                )
                
                total_loss += outputs.loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Average across all processes if using FSDP
        if self.world_size > 1:
            avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss_tensor.item() / self.world_size
        
        return avg_loss
    
    def save_checkpoint(self, checkpoint_dir, additional_state=None):
        if not self.is_main_process:
            return
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # FSDP state dict handling like reference
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
            model_state_dict = self.model.state_dict()
        
        # Save checkpoint
        checkpoint = {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "global_step": self.global_step,
        }
        if additional_state:
            checkpoint.update(additional_state)
        
        torch.save(checkpoint, os.path.join(checkpoint_dir, "checkpoint.pt"))
    
    def sanity_train(self):
        """Sanity training - overfit on small dataset"""
        if self.is_main_process:
            print("="*60)
            print("STARTING FSDP SANITY TRAINING")
            print("="*60)
            print(f"Target loss: {self.args.target_loss}")
            print(f"Epochs: {self.args.epochs}")
            print(f"Samples: {self.args.num_samples}")
        
        self.model.train()
        progress_bar = create_progress_bar("🚀 Sanity Training", self.args.epochs, self.is_main_process)
        
        for epoch in range(self.args.epochs):
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            
            epoch_losses = []
            for batch in self.train_loader:
                loss = self.train_step(batch)
                epoch_losses.append(loss)
                self.losses.append(loss)
            
            epoch_loss = np.mean(epoch_losses)
            
            if progress_bar:
                progress_bar.update(1)
                progress_bar.set_description(
                    f"🚀 Sanity Training | Epoch: {epoch+1} | Loss: {epoch_loss:.4f}"
                )
            
            # Check if target reached
            if epoch_loss <= self.args.target_loss:
                if self.is_main_process:
                    print(f"\n🎉 Target loss {self.args.target_loss:.4f} reached!")
                break
        
        if progress_bar:
            progress_bar.close()
        
        final_loss = epoch_loss
        target_reached = final_loss <= self.args.target_loss
        
        if self.is_main_process:
            print(f"\nSanity training completed!")
            print(f"Final loss: {final_loss:.4f}")
            print(f"Target reached: {'✓' if target_reached else '✗'}")
        
        # Save model if requested
        if self.args.save_model:
            checkpoint_name = f"{self.experiment_name}_sanity" if self.experiment_name else "sanity_checkpoint"
            checkpoint_dir = os.path.join(self.args.output_dir, checkpoint_name)
            self.save_checkpoint(checkpoint_dir)
        
        return target_reached
    
    def full_train(self):
        """Full training loop"""
        if self.is_main_process:
            print("="*60)
            print("STARTING FSDP FULL TRAINING")
            print("="*60)
            print(f"Epochs: {self.args.epochs}")
            print(f"Training samples: {len(self.train_loader.dataset)}")
        
        self.model.train()
        training_history = []
        validation_history = []
        
        total_steps = self.args.epochs * len(self.train_loader)
        progress_bar = create_progress_bar("🚀 Full Training", total_steps, self.is_main_process)
        
        for epoch in range(self.args.epochs):
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            
            epoch_losses = []
            
            for step, batch in enumerate(self.train_loader):
                loss = self.train_step(batch)
                epoch_losses.append(loss)
                self.losses.append(loss)
                
                if self.is_main_process:
                    training_history.append({
                        'epoch': epoch + 1,
                        'step': step + 1,
                        'global_step': self.global_step,
                        'loss': loss,
                        'learning_rate': self.lr_scheduler.get_last_lr()[0]
                    })
                
                if progress_bar:
                    progress_bar.update(1)
                    progress_bar.set_description(
                        f"🚀 Training | Epoch: {epoch+1} | Step: {step+1} | Loss: {loss:.4f}"
                    )
            
            epoch_loss = np.mean(epoch_losses)
            
            # Validation
            val_loss = None
            if self.val_loader is not None:
                if self.world_size > 1:
                    dist.barrier()
                val_loss = self.validate()
                
                if self.is_main_process:
                    validation_history.append({
                        'epoch': epoch + 1,
                        'validation_loss': val_loss,
                        'training_loss': epoch_loss
                    })
                    print(f"\nEpoch {epoch+1}: Train Loss = {epoch_loss:.4f}, Val Loss = {val_loss:.4f}")
                
                self.model.train()
                if self.world_size > 1:
                    dist.barrier()
            
            # Save checkpoint after each epoch
            if self.args.save_model:
                checkpoint_name = f"{self.experiment_name}_epoch{epoch+1}" if self.experiment_name else f"epoch{epoch+1}"
                checkpoint_dir = os.path.join(self.args.output_dir, checkpoint_name)
                self.save_checkpoint(checkpoint_dir)
        
        if progress_bar:
            progress_bar.close()
        
        # Save training history
        if self.is_main_process:
            save_simple_training_history(self, training_history, validation_history)
    
    def cleanup(self):
        """Cleanup distributed resources"""
        if self.world_size > 1:
            dist.destroy_process_group()


def run_training(rank, world_size, args):
    """Main training function for each process"""
    trainer = Cell2TextFSDPTrainer(args, rank, world_size)
    
    try:
        # Setup training pipeline
        trainer.load_tokenizer()
        trainer.setup_datasets()
        trainer.initialize_model()
        trainer.setup_model_for_training()
        trainer.setup_optimizer()
        
        # Run training based on mode
        if args.mode == "sanity":
            success = trainer.sanity_train()
        else:
            trainer.full_train()
            success = True
        
        return success, {}
        
    except Exception as e:
        if trainer.is_main_process:
            print(f"Training failed with error: {e}")
        raise
    finally:
        trainer.cleanup()


def main():
    """Main function"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Handle experiment naming
    if not args.experiment_name:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"{args.mode}_{args.projector}_{timestamp}"
    
    args.output_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Experiment: {args.experiment_name}")
    print(f"Output directory: {args.output_dir}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Determine world size
    world_size = torch.cuda.device_count() if torch.cuda.is_available() and not args.no_cuda else 1
    
    if world_size > 1:
        print(f"Starting FSDP training with {world_size} GPUs")
        mp.spawn(run_training, args=(world_size, args), nprocs=world_size, join=True)
    else:
        print("Starting single GPU/CPU training")
        success, results = run_training(0, 1, args)
        print(f"Training completed successfully: {success}")


if __name__ == "__main__":
    main()