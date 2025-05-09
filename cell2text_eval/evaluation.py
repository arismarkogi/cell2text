# evaluation.py

import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
from cell2text_model.model import Cell2TextModel
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer


def evaluate_cell2text_model(model : Cell2TextModel, val_loader: DataLoader, tokenizer: PreTrainedTokenizer, device: str):
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

            text_input_ids = batch["text_input_ids"].to(device)
            text_attention_mask = batch["text_attention_mask"].to(device)

            labels = batch["labels"].to(device) if batch["labels"] is not None else None

            # Forward pass to get loss
            outputs = model(
                expression_tokens=expression_tokens,
                expression_token_lengths=expression_token_lengths,
                text_input_ids=text_input_ids,
                text_input_attetnion_mask=text_attention_mask,
                labels=labels
            )
            loss = outputs.loss
            val_loss += loss.item()

            # Decode predictions
            generated = model.generate_cell_description(
                expression_tokens=expression_tokens,
                expression_token_lengths=expression_token_lengths,
            )

            for j, gen in enumerate(generated):
                decoded_pred = tokenizer.decode(gen, skip_special_tokens=True)
                target = tokenizer.decode(batch["decoder_input_ids"][j], skip_special_tokens=True)

                bleu = sentence_bleu(
                    [target.split()],
                    decoded_pred.split(),
                    smoothing_function=smooth
                )
                bleu_scores.append(bleu)

            val_progress_bar.set_postfix({"loss": loss.item()})

    avg_val_loss = val_loss / len(val_loader)
    avg_bleu = np.mean(bleu_scores)

    print(f"\nValidation Results — Loss: {avg_val_loss:.4f}, BLEU: {avg_bleu:.4f}")
    return avg_val_loss, avg_bleu
