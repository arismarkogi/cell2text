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


class Cell2TextSanityTrainer:
    """Sanity check trainer that overfits on a small dataset"""
    
    def __init__(self, args):
        self.args = args
        self.device = None
        self.model = None
        self.tokenizer = None
        self.train_loader = None
        self.optimizer = None
        self.global_step = 0
        self.losses = []
        
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
        """Setup small training dataset for sanity check"""
        print(f"Loading dataset for sanity check...")
        
        # Load full dataset
        full_dataset = Cell2TextDataset(self.args.train_data_path, self.tokenizer, top_k=self.args.top_k)
        
        # Create small subset
        sanity_dataset = SanityDataset(full_dataset, num_samples=self.args.num_samples, seed=self.args.seed)
        
        self.train_loader = DataLoader(
            sanity_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=True,
            collate_fn=full_dataset.collate_fn()
        )
        print(f"Sanity dataset loaded. Size: {len(sanity_dataset)}")
        
        # Use the same small dataset for validation
        self.val_loader = DataLoader(
            sanity_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=False,
            collate_fn=full_dataset.collate_fn()
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
        
    def setup_optimizer(self):
        """Setup optimizer"""
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
        
        
        
    def train_step(self, batch):
        """Perform a single training step"""
        # Move data to device
        expression_tokens = batch["expression_tokens"].to(self.device)
        expression_token_lengths = batch["expression_token_lengths"].to(self.device)
        text_input_ids = batch["input_ids"].to(self.device)
        text_input_attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device) if batch["labels"] is not None else None
        
        # Forward pass - Fix the typo in parameter name
        outputs = self.model(
            expression_tokens=expression_tokens,
            expression_token_lengths=expression_token_lengths,
            input_ids=text_input_ids,
            attention_mask=text_input_attention_mask,  # Fixed typo
            labels=labels,
            return_dict=True
        )
        
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        self.global_step += 1
        
        return loss.item()
        
    def sanity_train(self):
        """Main sanity training loop - overfit on small dataset"""
        print("="*60)
        print("STARTING SANITY CHECK TRAINING")
        print("="*60)
        print(f"Target loss: {self.args.target_loss}")
        print(f"Epochs: {self.args.epochs}")
        print(f"Number of samples: {self.args.num_samples}")
        print("="*60)
        
        # Setup everything
        self.setup_device()
        self.load_tokenizer()
        self.setup_datasets()
        self.initialize_model()
        self.freeze_model_components()
        self.apply_lora_to_model()
        self.print_model_parameters()
        
        self.model.to(self.device)
        self.setup_optimizer()
        
        # Training loop - overfit until target loss
        self.model.train()
        step = 0
        
        
        print("\nStarting overfitting training...")
        progress_bar = tqdm(desc="Sanity Training", total=self.args.epochs)
        
        epoch_loss = 0.0
        epoch = 0

        while epoch < self.args.epochs:
            epoch_losses = []
            
            for batch in self.train_loader:
                
                    
                loss = self.train_step(batch)
                epoch_losses.append(loss)
                self.losses.append(loss)
                
                step += 1
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss": f"{loss:.4f}",
                    "avg_loss": f"{np.mean(self.losses[-10:]):.4f}",
                    "target": f"{self.args.target_loss:.4f}"
                })
                

            # After completing one epoch
            epoch_loss = np.mean(epoch_losses)

            print(f"Epoch {epoch}: avg_loss = {epoch_loss:.4f}")
            epoch += 1

            if epoch_loss <= self.args.target_loss:
                print(f"\n🎉 Target loss {self.args.target_loss:.4f} reached at epoch {epoch}!")
                print(f"Final epoch loss: {epoch_loss:.4f}")
                break
        
        progress_bar.close()
        
        final_loss = epoch_loss
        print(f"\nSanity training completed!")
        print(f"Steps taken: {step}")
        print(f"Final loss: {final_loss:.4f}")
        print(f"Target reached: {'✓' if final_loss <= self.args.target_loss else '✗'}")
        
        # Save the overfitted model
        if self.args.save_overfitted_model:
            checkpoint_path = os.path.join(self.args.output_dir, "overfitted_sanity_model.pt")
            save_dict = {
                "model_state_dict": self.model.state_dict(),
                "final_loss": final_loss,
                "steps_taken": step,
                "target_reached": final_loss <= self.args.target_loss,
                "losses": self.losses
            }
            torch.save(save_dict, checkpoint_path)
            print(f"Overfitted model saved to: {checkpoint_path}")
        
        return final_loss <= self.args.target_loss
        
    def run_evaluation(self):
        """Run evaluation on the same 8 samples"""
        print("\n" + "="*60)
        print("RUNNING EVALUATION ON SANITY SAMPLES")
        print("="*60)
        
        # Use the enhanced evaluation function
        results = evaluate_cell2text_model(
            model=self.model,
            val_loader=self.val_loader,
            tokenizer=self.tokenizer,
            device=self.device,
            print_examples=self.args.num_samples,  # Print all samples
            save_results=os.path.join(self.args.output_dir, "sanity_evaluation_results.json") if self.args.save_results else None
        )
        
        print("\n" + "="*60)
        print("SANITY CHECK SUMMARY")
        print("="*60)
        print(f"Final training loss: {self.losses[-1]:.4f}")
        print(f"Target loss: {self.args.target_loss:.4f}")
        print(f"Target reached: {'✓' if self.losses[-1] <= self.args.target_loss else '✗'}")
        print(f"Validation loss: {results['loss']:.4f}")
        print(f"BLEU score: {results['bleu']:.4f}")
        print(f"Cell type accuracy: {results['cell_type_accuracy']:.4f}")
        print(f"Cell type F1: {results['cell_type_f1']:.4f}")
        
        # Check if overfitting worked (low validation loss on same samples)
        if results['loss'] < 0.1:
            print("✅ SANITY CHECK PASSED - Model successfully overfitted to small dataset!")
        else:
            print("❌ SANITY CHECK FAILED - Model did not overfit properly")
        
        return results
        
    def run_sanity_check(self):
        """Run the complete sanity check pipeline"""
        print("Starting complete sanity check pipeline...")
        
        # Train to overfit
        target_reached = self.sanity_train()
        
        # Evaluate
        results = self.run_evaluation()
        
        return target_reached, results


def create_argument_parser():
    """Create and return the argument parser for sanity check"""
    parser = argparse.ArgumentParser(description="Sanity check training for Cell2Text model")
    
    # Data parameters
    parser.add_argument("--train_data_path", type=str, required=True,
                        help="Path to the parquet file containing training data")
    parser.add_argument("--output_dir", type=str, default="./sanity_check",
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
    
    # Training parameters (optimized for sanity check)
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for training (small for sanity check)")
    parser.add_argument("--decoder_lr", type=float, default=1e-4,
                        help="Learning rate for the decoder (higher for faster overfitting)")
    parser.add_argument("--projector_lr", type=float, default=1e-3,
                        help="Learning rate for the projection layer (higher for faster overfitting)")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay for AdamW optimizer (0 for easier overfitting)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for gradient clipping")

    # Misc parameters
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
    """Main function for sanity check"""
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
    trainer = Cell2TextSanityTrainer(args)
    target_reached, results = trainer.run_sanity_check()
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SANITY CHECK RESULTS")
    print("="*60)
    
    print(f"\nKey metrics:")
    print(f"  Training loss: {trainer.losses[-1]:.4f}")
    print(f"  Validation loss: {results['loss']:.4f}")
    print(f"  BLEU score: {results['bleu']:.4f}")
    print(f"  Cell type accuracy: {results['cell_type_accuracy']:.4f}")


if __name__ == "__main__":
    main()