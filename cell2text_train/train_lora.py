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


def apply_lora_to_model(model, args):
    """Apply LoRA to the specified components of the model"""
    
    # LoRA configuration for the decoder (LLaMA)
    if args.use_lora_decoder:
        print("Applying LoRA to decoder...")
        decoder_lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r_decoder,
            lora_alpha=args.lora_alpha_decoder,
            lora_dropout=args.lora_dropout_decoder,
            target_modules=args.lora_target_modules_decoder,
            bias=args.lora_bias_decoder,
            modules_to_save=args.lora_modules_to_save_decoder if args.lora_modules_to_save_decoder else None,
        )
        
        # Apply LoRA to decoder
        model.decoder = get_peft_model(model.decoder, decoder_lora_config)
        print(f"LoRA applied to decoder. Trainable parameters: {model.decoder.num_parameters()}")
    
    
    return model


def get_trainable_parameters(model):
    """Get the number of trainable parameters"""
    trainable_params = 0
    all_params = 0
    
    for param in model.parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    return trainable_params, all_params


def train_cell2text_model(args):

    # Early stopping variables
    patience = args.patience
    no_improvement_epochs = 0
    
    print("Starting Cell2Text model training with LoRA...")
    
    # 1. Set up device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # 2. Load the tokenizer for text descriptions
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.decoder_path, pad_token='<|reserved_special_token_0|>')
        
        print("Tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Proceeding without tokenizer. This may cause issues in training.")
        tokenizer = None
    
    # 3. Load the datasets
    print(f"Loading datasets...")
    print(tokenizer)

    # Find the pad_token_id of geneformer
    token_dictionary_file = "/home/arismarkog/Desktop/cell2text/Geneformer/geneformer/token_dictionary_gc95M.pkl"

    with open(token_dictionary_file, "rb") as f:
        gene_token_dict = pickle.load(f)
        geneformer_pad_token_id = gene_token_dict["<pad>"]

    print(f"geneformer_pad_token_id: {geneformer_pad_token_id}")

    train_dataset = Cell2TextDataset(args.train_data_path, tokenizer, top_k=args.top_k)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=train_dataset.collate_fn(geneformer_pad_token_id=geneformer_pad_token_id)
    )
    
    if args.val_data_path:
        val_dataset = Cell2TextDataset(args.val_data_path, tokenizer, top_k=args.top_k)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            collate_fn=train_dataset.collate_fn(geneformer_pad_token_id=geneformer_pad_token_id)
        )
        print(f"Validation dataset loaded. Size: {len(val_dataset)}")
    else:
        val_loader = None
        
    print(f"Training dataset loaded. Size: {len(train_dataset)}")
    
    # 4. Initialize model configuration
    print("Initializing model configuration...")
    config = Cell2TextConfig()
    
    # Set required configuration parameters
    config.cell_encoder_hidden_size = args.encoder_hidden_size
    config.mlp_hidden_size = args.mlp_hidden_size
    config.mlp_dropout = args.mlp_dropout
    config.decoder_hidden_size = args.decoder_hidden_size
    config.geneformer_path = args.geneformer_path
    config.decoder_model_name_or_path = args.decoder_path
    config.top_k = args.top_k
    
    # Additional configuration parameters
    config.max_ncells = args.max_ncells
    config.max_new_tokens = args.max_length
    config.num_beams = args.num_beams
    
    # 5. Initialize the model
    print("Initializing model...")
    model = Cell2TextModel(config=config)
    model.warm_up()  # Load pretrained weights
    
    print("I got here")
    # Apply LoRA if specified
    if args.use_lora_decoder:
        model = apply_lora_to_model(model, args)
    
    # Get parameter counts
    trainable_params, all_params = get_trainable_parameters(model)
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"All parameters: {all_params:,}")
    print(f"Trainable %: {100 * trainable_params / all_params:.2f}%")

    print("Trainable parameters using named_parameters():")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel()}")

    # Freeze parameters based on LoRA settings
    if not args.use_lora_decoder and args.freeze_decoder:
        for param in model.decoder.parameters():
            param.requires_grad = False
        print("LLaMA decoder parameters frozen.")
    
    for param in model.cell_encoder.parameters():
        param.requires_grad = False
        print("Cell encoder parameters frozen.")
    
    # If resuming from checkpoint
    if args.resume_from_checkpoint:
        print(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model.to(device)
    
    # 6. Setup optimizer and scheduler
    # Collect trainable parameters with different learning rates
    optimizer_grouped_parameters = []
    
    
    # Projector parameters
    projector_params = []
    for name, param in model.cell_to_embedding.named_parameters():
        if param.requires_grad:
            projector_params.append(param)
    if projector_params:
        optimizer_grouped_parameters.append({"params": projector_params, "lr": args.projector_lr})
    
    # Decoder parameters
    decoder_params = []
    for name, param in model.decoder.named_parameters():
        if param.requires_grad:
            decoder_params.append(param)
    if decoder_params:
        optimizer_grouped_parameters.append({"params": decoder_params, "lr": args.decoder_lr})
    
    if not optimizer_grouped_parameters:
        raise ValueError("No trainable parameters found!")
    
    optimizer = AdamW(optimizer_grouped_parameters, weight_decay=args.weight_decay)
    
    # Calculate total training steps
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    # Learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 7. Training loop
    print(f"Starting training for {args.num_epochs} epochs...")
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(args.num_epochs):
        # Training
        model.train()
        epoch_loss = 0
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]")
        
        for batch in train_progress_bar:
            # Move data to device
            expression_tokens = batch["expression_tokens"].to(device)
            expression_token_lengths = batch["expression_token_lengths"].to(device)

            text_input_ids = batch["input_ids"].to(device)
            text_input_attention_mask = batch["attention_mask"].to(device)
           
            labels = batch["labels"].to(device) if batch["labels"] is not None else None
            
            outputs = model(
                    expression_tokens=expression_tokens,
                    expression_token_lengths=expression_token_lengths,
                    input_ids=text_input_ids,
                    input_attetnion_mask=text_input_attention_mask,
                    labels=labels,
                    return_dict=True
            )

            print(outputs)
            
            loss = outputs.loss
            epoch_loss += loss.item()

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Update progress bar
            train_progress_bar.set_postfix({"loss": loss.item()})
            global_step += 1
            
            # Log every n steps
            if global_step % args.logging_steps == 0:
                print(f"Step {global_step}: loss = {loss.item():.4f}")
        
        # Calculate average epoch loss
        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} average training loss: {avg_train_loss:.4f}")
        
        # Validation
        if val_loader:
            avg_val_loss, avg_bleu = evaluate_cell2text_model(model, val_loader, tokenizer, device, args)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improvement_epochs = 0
                print(f"New best validation loss: {best_val_loss:.4f}")
                
                # Save best model
                save_path = os.path.join(args.output_dir, "best_model.pt")
                save_dict = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "global_step": global_step
                }
                
                # Save LoRA adapter if using LoRA
                if args.use_lora_decoder:
                    save_dict["lora_config"] = {
                        "use_lora_decoder": args.use_lora_decoder,
                        "lora_args": vars(args)
                    }
                
                torch.save(save_dict, save_path)
                print(f"Best model saved to {save_path}")
                
                # Save LoRA adapters separately if using LoRA                
                if args.use_lora_decoder:
                    decoder_adapter_path = os.path.join(args.output_dir, "best_decoder_lora")
                    model.decoder.save_pretrained(decoder_adapter_path)
                    print(f"Best decoder LoRA adapter saved to {decoder_adapter_path}")
                    
            else:
                no_improvement_epochs += 1
                print(f"No improvement for {no_improvement_epochs} epoch(s).")
                if no_improvement_epochs >= patience:
                    print("Early stopping triggered.")
                    return

            # Calculate average validation loss
            print(f"Epoch {epoch+1} average validation loss: {avg_val_loss:.4f}, average BLEU score: {avg_bleu:.4f}")

        # Save checkpoint after each epoch
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
        save_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss if val_loader else None,
            "global_step": global_step
        }
        
        # Save LoRA configuration in checkpoint
        if  args.use_lora_decoder:
            save_dict["lora_config"] = {
                "use_lora_decoder": args.use_lora_decoder,
                "lora_args": vars(args)
            }
        
        torch.save(save_dict, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    print("Training completed.")

if __name__ == "__main__":
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
    parser.add_argument("--max_ncells", type=int, default=1000,
                        help="Maximum number of cells")
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum length of generated text")
    parser.add_argument("--num_beams", type=int, default=4,
                        help="Number of beams for beam search")
    
    # LoRA parameters
    parser.add_argument("--use_lora_decoder", action="store_true",
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

    # Misc parameters
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA even if available")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Validate LoRA arguments
    if args.use_lora_decoder:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            print("PEFT library found. LoRA support enabled.")
        except ImportError:
            raise ImportError("PEFT library not found. Please install it with: pip install peft")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    train_cell2text_model(args)