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

# DeepSpeed imports
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam

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


class Cell2TextDeepSpeedSanityTrainer:
    """DeepSpeed-enabled sanity check trainer that overfits on a small dataset"""
    
    def __init__(self, args):
        self.args = args
        self.device = None
        self.model = None
        self.tokenizer = None
        self.train_loader = None
        self.model_engine = None
        self.optimizer = None
        self.lr_scheduler = None
        self.global_step = 0
        self.losses = []
        
        # Initialize DeepSpeed
        deepspeed.init_distributed()
        
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
        """Setup small training dataset for sanity check"""
        print(f"Loading dataset for sanity check...")
        
        # Load full dataset
        full_dataset = Cell2TextDataset(self.args.train_data_path, self.tokenizer, top_k=self.args.top_k)
        
        # Create small subset
        sanity_dataset = SanityDataset(full_dataset, num_samples=self.args.num_samples, seed=self.args.seed)
        
        self.train_loader = DataLoader(
            sanity_dataset, 
            batch_size=self.args.batch_size_per_device, 
            shuffle=True,
            collate_fn=full_dataset.collate_fn(mode="train")
        )
        print(f"Sanity dataset loaded. Size: {len(sanity_dataset)}")
        
        # Use the same small dataset for validation
        self.val_loader = DataLoader(
            sanity_dataset, 
            batch_size=self.args.batch_size_per_device, 
            shuffle=False,
            collate_fn=full_dataset.collate_fn(mode="inference")
        )
        
    def initialize_model(self):
        """Initialize the Cell2Text model with configuration"""
        print("Initializing model configuration...")
        config = Cell2TextConfig()
        
        # Set required configuration parameters
        config.cell_encoder_hidden_size = self.args.encoder_hidden_size
        config.mlp_hidden_size = self.args.mlp_hidden_size
        config.mlp_dropout = self.args.mlp_dropout
        config.decoder_hidden_size = self.args.decoder_hidden_size
        config.geneformer_path = self.args.geneformer_path
        config.decoder_model_name_or_path = self.args.decoder_path
        config.top_k = self.args.top_k
        config.token_dictionary_path = self.args.token_dictionary_path
        
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
        
    def create_deepspeed_config(self):
        """Create DeepSpeed configuration"""
        ds_config = {
            "train_batch_size": self.args.batch_size_per_device * self.args.gradient_accumulation_steps * torch.distributed.get_world_size() if torch.distributed.is_initialized() else self.args.batch_size_per_device * self.args.gradient_accumulation_steps,
            "train_micro_batch_size_per_gpu": self.args.batch_size_per_device,
            "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
            
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.args.decoder_lr,  # Will be overridden by parameter groups
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": self.args.weight_decay
                }
            },
            
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": self.args.decoder_lr,
                    "warmup_num_steps": self.args.warmup_steps,
                    "total_num_steps": self.args.epochs * len(self.train_loader) // self.args.gradient_accumulation_steps
                }
            },
            
            "fp16": {
                "enabled": self.args.fp16,
                "auto_cast": False,
                "loss_scale": 0,
                "initial_scale_power": 16,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            
            "bf16": {
                "enabled": self.args.bf16
            },
            
            "zero_optimization": {
                "stage": self.args.zero_stage,
                "offload_optimizer": {
                    "device": "cpu" if self.args.zero_stage >= 2 else "none"
                },
                "offload_param": {
                    "device": "cpu" if self.args.zero_stage == 3 else "none"
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": 5e8,
                "stage3_prefetch_bucket_size": 5e7,
                "stage3_param_persistence_threshold": 1e5,
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "gather_16bit_weights_on_model_save": True
            },
            
            "gradient_clipping": self.args.max_grad_norm,
            "steps_per_print": 10,
            "wall_clock_breakdown": False
        }
        
        return ds_config
        
    def setup_deepspeed_model(self):
        """Setup model with DeepSpeed"""
        print("Setting up DeepSpeed model...")
        
        # Create DeepSpeed config
        ds_config = self.create_deepspeed_config()
        
        # Save DeepSpeed config
        config_path = os.path.join(self.args.output_dir, "deepspeed_config.json")
        with open(config_path, 'w') as f:
            json.dump(ds_config, f, indent=2)
        print(f"DeepSpeed config saved to: {config_path}")
        
        # Setup parameter groups for different learning rates
        parameters = []
        
        # Projector parameters
        for name, param in self.model.cell_to_embedding.named_parameters():
            if param.requires_grad:
                parameters.append({
                    "params": [param],
                    "lr": self.args.projector_lr,
                    "name": name
                })
                print(f"Adding projector parameter: {name} ({param.numel():,} params) with lr={self.args.projector_lr}")
        
        # Decoder parameters (including LoRA parameters)
        for name, param in self.model.decoder.named_parameters():
            if param.requires_grad:
                parameters.append({
                    "params": [param],
                    "lr": self.args.decoder_lr,
                    "name": name
                })
                print(f"Adding decoder parameter: {name} ({param.numel():,} params) with lr={self.args.decoder_lr}")
        
        if not parameters:
            raise ValueError("No trainable parameters found! Check your LoRA configuration and parameter freezing.")
        
        print(f"Total parameter groups for DeepSpeed: {len(parameters)}")
        
        # Initialize DeepSpeed
        self.model_engine, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
            model=self.model,
            model_parameters=parameters,
            config=ds_config
        )
        
        print("DeepSpeed model initialized successfully!")
        
    def train_step(self, batch):
        """Perform a single training step with DeepSpeed"""
        # Move data to device (DeepSpeed handles this automatically)
        expression_tokens = batch["expression_tokens"]
        expression_token_lengths = batch["expression_token_lengths"]
        text_input_ids = batch["input_ids"]
        text_input_attention_mask = batch["attention_mask"]
        labels = batch["labels"] if batch["labels"] is not None else None
        
        # Forward pass
        outputs = self.model_engine(
            expression_tokens=expression_tokens,
            expression_token_lengths=expression_token_lengths,
            input_ids=text_input_ids,
            attention_mask=text_input_attention_mask,
            labels=labels,
            return_dict=True
        )
        
        loss = outputs.loss
        
        # DeepSpeed handles backward pass and optimization
        self.model_engine.backward(loss)
        self.model_engine.step()
        
        self.global_step += 1
        
        return loss.item()
        
    def sanity_train(self):
        """Main sanity training loop - overfit on small dataset with DeepSpeed"""
        print("="*60)
        print("STARTING DEEPSPEED SANITY CHECK TRAINING")
        print("="*60)
        print(f"Target loss: {self.args.target_loss}")
        print(f"Epochs: {self.args.epochs}")
        print(f"Number of samples: {self.args.num_samples}")
        print(f"Batch size per device: {self.args.batch_size_per_device}")
        print(f"Gradient accumulation steps: {self.args.gradient_accumulation_steps}")
        print("="*60)
        
        # Setup everything
        self.setup_device()
        self.load_tokenizer()
        self.setup_datasets()
        self.initialize_model()
        self.freeze_model_components()
        self.apply_lora_to_model()
        self.print_model_parameters()
        
        # Setup DeepSpeed
        self.setup_deepspeed_model()
        
        # Training loop - overfit until target loss
        self.model_engine.train()
        
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
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                progress_bar.update(1)
                progress_bar.set_description(
                    f"🚀 DeepSpeed Sanity Training | 📊 Epoch: {epoch+1}/{self.args.epochs} | "
                    f"📉 Loss: {epoch_loss:.4f} | 🎯 Target: {self.args.target_loss:.4f}"
                )

            epoch += 1

            if epoch_loss <= self.args.target_loss:
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    print(f"\n🎉 Target loss {self.args.target_loss:.4f} reached at epoch {epoch}!")
                    print(f"Final epoch loss: {epoch_loss:.4f}")
                break
        
        progress_bar.close()
        
        final_loss = epoch_loss
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"\nDeepSpeed sanity training completed!")
            print(f"Final loss: {final_loss:.4f}")
            print(f"Target reached: {'✓' if final_loss <= self.args.target_loss else '✗'}")
        
        # Save the overfitted model (only on rank 0)
        if self.args.save_overfitted_model and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            checkpoint_dir = os.path.join(self.args.output_dir, "overfitted_sanity_model")
            self.model_engine.save_checkpoint(checkpoint_dir)
            
            # Also save additional info
            info_path = os.path.join(checkpoint_dir, "training_info.json")
            info_dict = {
                "final_loss": final_loss,
                "epochs_taken": epoch,
                "target_reached": final_loss <= self.args.target_loss,
                "losses": self.losses
            }
            with open(info_path, 'w') as f:
                json.dump(info_dict, f, indent=2)
            
            print(f"Overfitted model saved to: {checkpoint_dir}")
        
        return final_loss <= self.args.target_loss
        
    def run_evaluation(self):
        """Run evaluation on the same 8 samples"""
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print("\n" + "="*60)
            print("RUNNING EVALUATION ON SANITY SAMPLES")
            print("="*60)
        
        # Use the enhanced evaluation function
        results = evaluate_cell2text_model(
            model=self.model_engine.module,  # Get the actual model from DeepSpeed
            val_loader=self.val_loader,
            tokenizer=self.tokenizer,
            device=self.model_engine.device,
            print_examples=self.args.num_samples if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0) else 0,
            save_results=os.path.join(self.args.output_dir, "sanity_evaluation_results.json") if self.args.save_results and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0) else None
        )
        
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print("\n" + "="*60)
            print("DEEPSPEED SANITY CHECK SUMMARY")
            print("="*60)
            print(f"Final training loss: {self.losses[-1]:.4f}")
            print(f"Target loss: {self.args.target_loss:.4f}")
            print(f"Target reached: {'✓' if self.losses[-1] <= self.args.target_loss else '✗'}")
            print(f"BLEU score: {results['bleu']:.4f}")
            print(f"Cell type accuracy: {results['cell_type_accuracy']:.4f}")
            print(f"Cell type F1: {results['cell_type_f1']:.4f}")
        
        return results
        
    def run_sanity_check(self):
        """Run the complete sanity check pipeline with DeepSpeed"""
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print("Starting complete DeepSpeed sanity check pipeline...")
        
        # Train to overfit
        target_reached = self.sanity_train()
        
        # Evaluate
        results = self.run_evaluation()
        
        return target_reached, results


def create_argument_parser():
    """Create and return the argument parser for DeepSpeed sanity check"""
    parser = argparse.ArgumentParser(description="DeepSpeed sanity check training for Cell2Text model")
    
    # Data parameters
    parser.add_argument("--train_data_path", type=str, required=True,
                        help="Path to the parquet file containing training data")
    parser.add_argument("--output_dir", type=str, default="./deepspeed_sanity_check",
                        help="Directory to save sanity check results")
    
    # Sanity check specific parameters
    parser.add_argument("--num_samples", type=int, default=8,
                        help="Number of samples to overfit on")
    parser.add_argument("--target_loss", type=float, default=0.01,
                        help="Target loss to reach (default: 0.01)")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Training epochs")
    parser.add_argument("--save_overfitted_model", action="store_true", default=True,
                        help="Save the overfitted model")
    parser.add_argument("--save_results", action="store_true", default=True,
                        help="Save evaluation results to JSON")
    
    # Model parameters
    parser.add_argument("--encoder_hidden_size", type=int, default=512,
                        help="Hidden size of the cell encoder")
    parser.add_argument("--mlp_hidden_size", type=int, default=256,
                        help="Hidden size of the 2-layer MLP cell-to-embedding projector")
    parser.add_argument("--mlp_dropout", type=float, default=0.1, 
                        help="Dropout probability at MLP projector")
    parser.add_argument("--decoder_hidden_size", type=int, default=2048,
                        help="Hidden size of the decoder")
    parser.add_argument("--top_k", type=int, default=512,
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
    parser.add_argument("--num_beams", type=int, default=4,
                        help="Number of beams for beam search")
    
    # LoRA parameters
    parser.add_argument("--use_lora_decoder", action="store_true", default=True,
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
    parser.add_argument("--freeze_decoder", action="store_true", default=True,
                        help="Freeze decoder parameters (only applies if not using LoRA for decoder)")
    
    # DeepSpeed specific parameters
    parser.add_argument("--batch_size_per_device", type=int, default=2,
                        help="Batch size per device/GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--zero_stage", type=int, default=2, choices=[0, 1, 2, 3],
                        help="DeepSpeed ZeRO optimization stage")
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="Enable FP16 mixed precision training")
    parser.add_argument("--bf16", action="store_true", default=False,
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
    
    return parser


def validate_args(args):
    """Validate command line arguments"""
    if args.use_lora_decoder:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            print("PEFT library found. LoRA support enabled.")
        except ImportError:
            raise ImportError("PEFT library not found. Please install it with: pip install peft")
    
    try:
        import deepspeed
        print(f"DeepSpeed version: {deepspeed.__version__}")
    except ImportError:
        raise ImportError("DeepSpeed library not found. Please install it with: pip install deepspeed")
    
    if args.fp16 and args.bf16:
        raise ValueError("Cannot enable both FP16 and BF16 at the same time")


def main():
    """Main function for DeepSpeed sanity check"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Validate arguments
    validate_args(args)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create trainer and run sanity check
    trainer = Cell2TextDeepSpeedSanityTrainer(args)
    target_reached, results = trainer.run_sanity_check()
    
    # Final summary (only on rank 0)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print("\n" + "="*60)
        print("FINAL DEEPSPEED SANITY CHECK RESULTS")
        print("="*60)
        
        print(f"\nKey metrics:")
        print(f"  Training loss: {trainer.losses[-1]:.4f}")
        print(f"  Validation loss: {results['loss']:.4f}")
        print(f"  BLEU score: {results['bleu']:.4f}")
        print(f"  Cell type accuracy: {results['cell_type_accuracy']:.4f}")


if __name__ == "__main__":
    main()