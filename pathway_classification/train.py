import argparse
from dataset import MultiDatasetPathwayDataset
from classifier import GeneformerPathwayClassifier
from trainer import PathwayTrainer

def load_config(config_path):
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Create datasets
    train_dataset = MultiDatasetPathwayDataset(config['data']['base_data_path'], split='train')
    val_dataset = MultiDatasetPathwayDataset(config['data']['base_data_path'], split='val')
    
    # Create model
    model = GeneformerPathwayClassifier(
        config['model']['geneformer_model_path'],
        num_pathways=train_dataset.num_pathways,
        freeze_geneformer=config['model']['freeze_geneformer']
    )

   
    
    # Create trainer
    trainer = PathwayTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        **config['training']  # unpack training config
    )
    
    # Train
    history = trainer.train(save_path=f"results/best_model.pt")

if __name__ == "__main__":
    main()