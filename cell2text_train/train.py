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


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cell2text_model.model import Cell2TextModel
from cell2text_model.configuration import Cell2TextConfig
from cell2text_dataset.dataset import Cell2TextDataset
from cell2text_eval.evaluation import evaluate_cell2text_model


def train_cell2text_model(args):

    # Early stopping variables
    patience = args.patience
    no_improvement_epochs = 0
    
    print("Starting Cell2Text model training...")
    
    # 1. Set up device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # 2. Load the tokenizer for text descriptions
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.decoder_path)
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
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



    train_dataset = Cell2TextDataset(args.train_data_path, tokenizer)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=train_dataset.collate_fn(geneformer_pad_token_id=geneformer_pad_token_id)
    )
    
    if args.val_data_path:
        val_dataset = Cell2TextDataset(args.val_data_path, tokenizer)
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
    
    # Additional configuration parameters
    config.max_ncells = args.max_ncells
    config.max_length = args.max_length
    config.num_beams = args.num_beams


    
    # 5. Initialize the model
    print("Initializing model...")
    model = Cell2TextModel(config=config)
    model.warm_up()  # Load pretrained weights

    print("Trainable parameters using named_parameters():")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel()}")

    # Freeze LLaMA decoder weights in order to do a loss.backward() my computer's RAM
    for param in model.decoder.parameters():
        param.requires_grad = False
    print("LLaMA decoder parameters frozen.")
    
    # # Freeze cell encoder weights
    # for param in model.cell_encoder.parameters():
    #     param.requires_grad = False
    # print("Cell encoder parameters frozen.")

    
    # If resuming from checkpoint
    if args.resume_from_checkpoint:
        print(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model.to(device)
    
    # 6. Setup optimizer and scheduler
    # We'll use different learning rates for encoder and decoder
    encoder_params = list(model.cell_encoder.parameters())
    projector_params = list(model.cell_to_embedding.parameters())
    decoder_params = list(model.decoder.parameters())
    
    # Combine parameters with different learning rates
    optimizer_grouped_parameters = [
        {"params": encoder_params, "lr": args.encoder_lr},
        {"params": projector_params, "lr": args.projector_lr}, 
        {"params": decoder_params, "lr": args.decoder_lr}
    ]
    
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

            text_input_ids = batch["text_input_ids"].to(device)
            text_input_attetnion_mask = batch["text_attention_mask"].to(device)
           
            labels = batch["labels"].to(device) if batch["labels"] is not None else None
            
            outputs = model(
                    expression_tokens=expression_tokens,
                    expression_token_lengths=expression_token_lengths,
                    text_input_ids=text_input_ids,
                    text_input_attetnion_mask=text_input_attetnion_mask,
                    labels=labels
            )
            print(type(outputs))
            print(outputs)

            loss = outputs.loss


            print("Before loss.backward()")
            loss.backward()
            print("After loss.backward()")

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
                
                save_path = os.path.join(args.output_dir, "best_model.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "global_step": global_step
                }, save_path)
                print(f"Best model saved to {save_path}")
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
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss if val_loader else None,
            "global_step": global_step
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    print("Training completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cell2Text model")
    
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
    parser.add_argument("--mlp_hidden_size", type=int, default=1024,
                        help="Hidden size of the 2-layer MLP cell-to-embedding projector")
    parser.add_argument("--mlp_dropout", type=float, default=0.1, 
                        help="Dropout probability at MLP projector")
    parser.add_argument("--decoder_hidden_size", type=int, default=2048,
                        help="Hidden size of the decoder")
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
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--encoder_lr", type=float, default=1e-5,
                        help="Learning rate for the encoder")
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
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    train_cell2text_model(args)