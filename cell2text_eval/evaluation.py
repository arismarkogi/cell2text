import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
from cell2text_model.model import Cell2TextModel
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

def evaluate_cell2text_model(model: Cell2TextModel, val_loader: DataLoader, tokenizer: PreTrainedTokenizer, device: str):
    model.eval()
    val_loss = 0
    bleu_scores = []
    smooth = SmoothingFunction().method4
    val_progress_bar = tqdm(val_loader, desc="[Validation]")
    
    with torch.no_grad():
        for batch in val_progress_bar:
            # Move batch to device
            expression_tokens = batch["expression_tokens"].to(device)
            expression_token_lengths = batch["expression_token_lengths"].to(device)
            text_input_ids = batch["input_ids"].to(device)
            text_attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device) if batch["labels"] is not None else None
            
            # Forward pass to get loss - Fixed parameter names
            outputs = model(
                expression_tokens=expression_tokens,
                expression_token_lengths=expression_token_lengths,
                input_ids=text_input_ids,  # Correct parameter name
                attention_mask=text_attention_mask,  # Fixed typo: was input_attetnion_mask
                labels=labels,
                return_dict=True
            )
            
            loss = outputs.loss
            val_loss += loss.item()
            
            # Generate descriptions - Fixed to use prompt_template parameter
            generated = model.generate_cell_description(
                expression_tokens=expression_tokens,
                expression_token_lengths=expression_token_lengths,
                prompt_template=None,  # Added prompt_template parameter (can be None for default)
                device=device  # Added device parameter
            )
            
            # Handle both single string and list of strings return
            if isinstance(generated, str):
                generated = [generated]
            
            # Calculate BLEU scores
            for j, gen in enumerate(generated):
                # generated is already decoded text from generate_cell_description
                decoded_pred = gen  # No need to decode again
                
                # Decode target - assuming you have decoder_input_ids in batch
                if "decoder_input_ids" in batch:
                    target = tokenizer.decode(batch["decoder_input_ids"][j], skip_special_tokens=True)
                elif "labels" in batch and batch["labels"] is not None:
                    # Use labels as target if decoder_input_ids not available
                    target_ids = batch["labels"][j]
                    # Remove padding tokens (-100) if present
                    target_ids = target_ids[target_ids != -100]
                    target = tokenizer.decode(target_ids, skip_special_tokens=True)
                else:
                    # If no target available, skip BLEU calculation for this sample
                    print(f"Warning: No target text available for sample {j}")
                    continue
                
                bleu = sentence_bleu(
                    [target.split()],
                    decoded_pred.split(),
                    smoothing_function=smooth
                )
                bleu_scores.append(bleu)
            
            val_progress_bar.set_postfix({"loss": loss.item()})
    
    avg_val_loss = val_loss / len(val_loader)
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
    
    print(f"\nValidation Results — Loss: {avg_val_loss:.4f}, BLEU: {avg_bleu:.4f}")
    return avg_val_loss, avg_bleu