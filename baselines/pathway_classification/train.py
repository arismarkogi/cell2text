import argparse
from dataset import MultiDatasetPathwayDataset
from classifier import GeneformerPathwayClassifier
from trainer import PathwayTrainer
from datetime import datetime
import os



def load_config(config_path):
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/home/arism/cell2text/pathway_classification/default.yaml')
    args = parser.parse_args()
    config = load_config(args.config)

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # After creating datasets, before creating trainer:
    train_dataset = MultiDatasetPathwayDataset(config['data']['base_data_path'], split='train')
    val_dataset = MultiDatasetPathwayDataset(config['data']['base_data_path'], split='val')
    
    # Create model
    model = GeneformerPathwayClassifier(
        config['model']['geneformer_model_path'],
        num_pathways=train_dataset.num_pathways,
        freeze_geneformer=config['model']['freeze_geneformer']
    )
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
    # Create trainer
    trainer = PathwayTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset 
    )
    current_date = datetime.now()
    date_str = current_date.strftime("%Y-%m-%d")
    save_path = f"/home/arism/pathway_results/{date_str}_best_model.pt"
    print(f"Model will be saved to: {save_path}")
    # Train
    history = trainer.train(save_path=save_path)
    
    return history

if __name__ == "__main__":
    main()