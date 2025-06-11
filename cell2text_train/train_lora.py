import torch
import pandas as pd
import numpy as np
import os
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    PretrainedConfig,
    get_linear_schedule_with_warmup,
    AutoTokenizer
)
import pickle
from tqdm import tqdm
import sys

# LoRA imports
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cell2text_model.model import Cell2TextModel
from cell2text_model.configuration import Cell2TextConfig
from cell2text_dataset.dataset import Cell2TextDataset
from cell2text_eval.evaluation import evaluate_cell2text_model


class Cell2TextTrainer:
    """Main trainer class for Cell2Text model with LoRA support"""
    
    def __init__(self, args):
        self.args = args
        self.device = None
        self.model = None
        self.tokenizer = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.best_val_loss = float('inf')
        self.no_improvement_epochs = 0
        self.global_step = 0
        
    def setup_device(self):
        """Setup device for training"""
        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")
        print(f"Using device: {self.device}")
        
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
        print(f"Loading datasets...")
        
        
        
        
        # Training dataset
        train_dataset = Cell2TextDataset(self.args.train_data_path, self.tokenizer, top_k=self.args.top_k)
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=True,
            collate_fn=train_dataset.collate_fn()
        )
        print(f"Training dataset loaded. Size: {len(train_dataset)}")
        
        # Validation dataset
        if self.args.val_data_path:
            val_dataset = Cell2TextDataset(self.args.val_data_path, self.tokenizer, top_k=self.args.top_k)
            self.val_loader = DataLoader(
                val_dataset, 
                batch_size=self.args.batch_size, 
                shuffle=False,
                collate_fn=train_dataset.collate_fn()
            )
            print(f"Validation dataset loaded. Size: {len(val_dataset)}")
        else:
            self.val_loader = None
            
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
        
        # First, prepare the model for k-bit training if needed
        self.model.decoder = prepare_model_for_kbit_training(self.model.decoder)
        
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
        
        # Print LoRA module names to verify they were created
        print("LoRA modules in decoder:")
        for name, module in self.model.decoder.named_modules():
            if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
                print(f"  - {name}")
                
    def print_model_parameters(self):
        """Print detailed information about model parameters"""
        trainable_params, all_params = self.get_trainable_parameters()
        print(f"\nParameter Summary:")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"All parameters: {all_params:,}")
        print(f"Trainable %: {100 * trainable_params / all_params:.2f}%")
        
        if self.args.verbose_params:
            self.print_parameter_details()
            
    def get_trainable_parameters(self):
        """Get the number of trainable parameters"""
        trainable_params = 0
        all_params = 0
        
        for param in self.model.parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        return trainable_params, all_params
        
    def print_parameter_details(self):
        """Print detailed information about model parameters"""
        print("\nDetailed parameter information:")
        print("-" * 50)
        
        total_trainable = 0
        total_frozen = 0
        
        for name, param in self.model.named_parameters():
            param_count = param.numel()
            status = "TRAINABLE" if param.requires_grad else "FROZEN"
            print(f"{name:60} | {param_count:>10,} | {status}")
            
            if param.requires_grad:
                total_trainable += param_count
            else:
                total_frozen += param_count
        
        print("-" * 50)
        print(f"{'TOTAL TRAINABLE':60} | {total_trainable:>10,}")
        print(f"{'TOTAL FROZEN':60} | {total_frozen:>10,}")
        print(f"{'TOTAL PARAMETERS':60} | {total_trainable + total_frozen:>10,}")
        print(f"{'TRAINABLE PERCENTAGE':60} | {100 * total_trainable / (total_trainable + total_frozen):>9.2f}%")
        
    def load_checkpoint(self):
        """Load model from checkpoint if specified"""
        if not self.args.resume_from_checkpoint:
            return
            
        print(f"Loading checkpoint from {self.args.resume_from_checkpoint}")
        checkpoint = torch.load(self.args.resume_from_checkpoint, map_location=self.device)
        
        # Handle LoRA checkpoint loading
        if self.args.use_lora_decoder and "lora_config" in checkpoint:
            # Model should already have LoRA applied, just load state dict
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            
    def setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler"""
        # Collect trainable parameters with different learning rates
        optimizer_grouped_parameters = []
        
        # Projector parameters
        projector_params = []
        for name, param in self.model.cell_to_embedding.named_parameters():
            if param.requires_grad:
                projector_params.append(param)
                print(f"Adding projector parameter to optimizer: {name} ({param.numel():,} params)")
        
        if projector_params:
            optimizer_grouped_parameters.append({"params": projector_params, "lr": self.args.projector_lr})
            print(f"Projector learning rate: {self.args.projector_lr}")
        
        # Decoder parameters (including LoRA parameters)
        decoder_params = []
        for name, param in self.model.decoder.named_parameters():
            if param.requires_grad:
                decoder_params.append(param)
                print(f"Adding decoder parameter to optimizer: {name} ({param.numel():,} params)")
        
        if decoder_params:
            optimizer_grouped_parameters.append({"params": decoder_params, "lr": self.args.decoder_lr})
            print(f"Decoder learning rate: {self.args.decoder_lr}")
        
        if not optimizer_grouped_parameters:
            raise ValueError("No trainable parameters found! Check your LoRA configuration and parameter freezing.")
        
        print(f"Total parameter groups for optimizer: {len(optimizer_grouped_parameters)}")
        
        self.optimizer = AdamW(optimizer_grouped_parameters, weight_decay=self.args.weight_decay)
        
        # Calculate total training steps
        total_steps = len(self.train_loader) * self.args.num_epochs
        warmup_steps = int(total_steps * self.args.warmup_ratio)
        
        # Learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
    def train_epoch(self, epoch):
        """Train the model for one epoch"""
        self.model.train()
        epoch_loss = 0
        train_progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.num_epochs} [Train]")
        
        for batch in train_progress_bar:
            loss = self.train_step(batch)
            epoch_loss += loss
            
            # Update progress bar
            train_progress_bar.set_postfix({"loss": loss})
            
            # Log every n steps
            if self.global_step % self.args.logging_steps == 0:
                print(f"Step {self.global_step}: loss = {loss:.4f}")
        
        avg_train_loss = epoch_loss / len(self.train_loader)
        print(f"Epoch {epoch+1} average training loss: {avg_train_loss:.4f}")
        return avg_train_loss
        
    def train_step(self, batch):
        """Perform a single training step"""
        # Move data to device
        expression_tokens = batch["expression_tokens"].to(self.device)
        expression_token_lengths = batch["expression_token_lengths"].to(self.device)
        text_input_ids = batch["input_ids"].to(self.device)
        text_input_attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device) if batch["labels"] is not None else None
        
        # Forward pass
        outputs = self.model(
            expression_tokens=expression_tokens,
            expression_token_lengths=expression_token_lengths,
            input_ids=text_input_ids,
            input_attetnion_mask=text_input_attention_mask,
            labels=labels,
            return_dict=True
        )
        
        if self.args.debug_outputs:
            print(f"Model outputs keys: {list(outputs.keys()) if hasattr(outputs, 'keys') else 'Not a dict'}")
        
        loss = outputs.loss
        
        # Backward pass
        loss.backward()

        print(outputs)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        self.global_step += 1
        
        return loss.item()
        
    def validate_epoch(self, epoch):
        """Validate the model for one epoch"""
        if not self.val_loader:
            return None, None
            
        avg_val_loss, avg_bleu = evaluate_cell2text_model(
            self.model, self.val_loader, self.tokenizer, self.device, self.args
        )
        
        print(f"Epoch {epoch+1} average validation loss: {avg_val_loss:.4f}, average BLEU score: {avg_bleu:.4f}")
        return avg_val_loss, avg_bleu
        
    def save_checkpoint(self, epoch, train_loss, val_loss=None, is_best=False):
        """Save model checkpoint"""
        suffix = "best_model" if is_best else f"checkpoint_epoch_{epoch+1}"
        checkpoint_path = os.path.join(self.args.output_dir, f"{suffix}.pt")
        
        save_dict = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "global_step": self.global_step
        }
        
        # Save LoRA configuration in checkpoint
        if self.args.use_lora_decoder:
            save_dict["lora_config"] = {
                "use_lora_decoder": self.args.use_lora_decoder,
                "lora_args": vars(self.args)
            }
        
        torch.save(save_dict, checkpoint_path)
        print(f"{'Best model' if is_best else 'Checkpoint'} saved to {checkpoint_path}")
        
        # Save LoRA adapters separately if using LoRA and this is the best model
        if is_best and self.args.use_lora_decoder:
            decoder_adapter_path = os.path.join(self.args.output_dir, "best_decoder_lora")
            self.model.decoder.save_pretrained(decoder_adapter_path)
            print(f"Best decoder LoRA adapter saved to {decoder_adapter_path}")
            
    def check_early_stopping(self, val_loss):
        """Check if early stopping should be triggered"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.no_improvement_epochs = 0
            print(f"New best validation loss: {self.best_val_loss:.4f}")
            return False, True  # not early stop, is best
        else:
            self.no_improvement_epochs += 1
            print(f"No improvement for {self.no_improvement_epochs} epoch(s).")
            if self.no_improvement_epochs >= self.args.patience:
                print("Early stopping triggered.")
                return True, False  # early stop, not best
        return False, False  # not early stop, not best
        
    def train(self):
        """Main training loop"""
        print("Starting Cell2Text model training with LoRA...")
        
        # Setup everything
        self.setup_device()
        self.load_tokenizer()
        self.setup_datasets()
        self.initialize_model()
        self.freeze_model_components()
        self.apply_lora_to_model()
        self.print_model_parameters()
        self.load_checkpoint()
        
        self.model.to(self.device)
        self.setup_optimizer_and_scheduler()
        
        # Training loop
        print(f"Starting training for {self.args.num_epochs} epochs...")
        
        for epoch in range(self.args.num_epochs):
            # Training
            avg_train_loss = self.train_epoch(epoch)
            
            # Validation
            avg_val_loss, avg_bleu = self.validate_epoch(epoch)
            
            # Check for early stopping and best model
            if avg_val_loss is not None:
                should_early_stop, is_best = self.check_early_stopping(avg_val_loss)
                
                if is_best:
                    self.save_checkpoint(epoch, avg_train_loss, avg_val_loss, is_best=True)
                
                if should_early_stop:
                    return
            
            # Save regular checkpoint
            self.save_checkpoint(epoch, avg_train_loss, avg_val_loss, is_best=False)
        
        print("Training completed.")


def create_argument_parser():
    """Create and return the argument parser"""
    parser = argparse.ArgumentParser(description="Train Cell2Text model with LoRA support")
    
    # Data parameters
    parser.add_argument("--train_data_path", type=str, required=True,
                        help="Path to the parquet file containing training data")
    parser.add_argument("--val_data_path", type=str, default=None,
                        help="Path to the parquet file containing validation data")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Directory to save model checkpoints")
    
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
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--decoder_lr", type=float, default=5e-5,
                        help="Learning rate for the decoder")
    parser.add_argument("--projector_lr", type=float, default=1e-4,
                        help="Learning rate for the projection layer")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW optimizer")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Ratio of total training steps used for warmup")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for gradient clipping")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log training stats every X steps")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")

    # Debug parameters
    parser.add_argument("--verbose_params", action="store_true",
                        help="Print detailed parameter information")
    parser.add_argument("--debug_outputs", action="store_true",
                        help="Print debug information about model outputs")

    # Misc parameters
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA even if available")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    return parser


def validate_args(args):
    """Validate command line arguments"""
    if args.use_lora_decoder:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            print("PEFT library found. LoRA support enabled.")
        except ImportError:
            raise ImportError("PEFT library not found. Please install it with: pip install peft")


def main():
    """Main function"""
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
    
    # Create trainer and start training
    trainer = Cell2TextTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()