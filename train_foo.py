import os
import torch
from transformers import AutoTokenizer

from cell2text_model.model import Cell2TextModel
from cell2text_model.configuration import Cell2TextConfig
from train.train_encoder import EncoderTrainingArguments, CellTextDataset, EncoderTrainer  # Assuming your main code is in train_script.py

def main():
    # Set up training arguments
    args = EncoderTrainingArguments(
        data_dir="datasets",  # path to your dataset
        output_dir="output/debug_run",
        cell_data_path="cell_data.pkl",
        text_data_path="text_data.pkl",
        model_type="cell2text",
        #pretrained_cell_encoder=None,  # Set path if needed
        pretrained_text_encoder="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",  # example
        embedding_fusion_method="attention",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=3e-5,
        logging_steps=10,
        eval_steps=50,
        save_steps=100,
        cell_dropout_rate=0.2,
        text_dropout_rate=0.1,
        seed=42,
        fp16=False  # Turn on if you want mixed precision and have a GPU
    )

    # Load tokenizer for text
    text_tokenizer = AutoTokenizer.from_pretrained(args.pretrained_text_encoder)

    # Load datasets
    train_dataset = CellTextDataset(
        data_path=args.data_dir,
        text_tokenizer=text_tokenizer,
        augment_cell=True,
        augment_text=True,
        cell_dropout_rate=args.cell_dropout_rate,
        text_dropout_rate=args.text_dropout_rate
    )

    # You can split your dataset for evaluation (e.g. 80/20)
    train_size = int(0.8 * len(train_dataset))
    eval_size = len(train_dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(train_dataset, [train_size, eval_size])

    # Create model config and model
    config = Cell2TextConfig(
        embedding_fusion_method=args.embedding_fusion_method,
        # You can add other config options here
    )
    model = Cell2TextModel(config)

    # Trainer
    trainer = EncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        text_tokenizer=text_tokenizer
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
