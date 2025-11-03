import torch
import sys
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from datasets import load_from_disk
import json

class PreprocessedPathwayDataset(Dataset):
    """
    Dataset wrapper for pathway classification (multi-label).
    Converts pathway1_id and pathway2_id to multi-hot labels on-the-fly.
    """
    def __init__(self, dataset_dir: Path, num_pathways: int):
        self.dataset = load_from_disk(str(dataset_dir))
        self.dataset.set_format(type='torch', columns=['gene_ids', 'values', 'pathway1_id', 'pathway2_id'])
        self.n_obs = len(self.dataset)
        self.num_pathways = num_pathways
    
    def __len__(self):
        return self.n_obs
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Convert pathway IDs to multi-hot label on-the-fly
        label = torch.zeros(self.num_pathways, dtype=torch.float32)
        
        p1 = item['pathway1_id'].item()
        p2 = item['pathway2_id'].item()
        
        if p1 >= 0:  # Valid pathway
            label[p1] = 1.0
        if p2 >= 0:  # Valid pathway
            label[p2] = 1.0
        
        return {
            'gene_ids': item['gene_ids'],
            'values': item['values'],
            'labels': label  # Multi-hot vector
        }

def load_pathway_mappings(data_dir: Path):
    """Loads the saved pathway mappings."""
    mapping_file = data_dir / "pathway_mappings.json"
    if not mapping_file.exists():
        raise FileNotFoundError(f"pathway_mappings.json not found in {data_dir}")
    
    with open(mapping_file, 'r') as f:
        mappings = json.load(f)
    
    id_to_pathway = mappings['id_to_pathway']
    id_to_pathway = {int(k): v for k, v in id_to_pathway.items()}
    pathway_to_id = mappings['pathway_to_id']
    num_pathways = len(id_to_pathway)
    
    return num_pathways, id_to_pathway, pathway_to_id

def create_pathway_dataloaders(preprocessed_data_dir: Path, config, rank, world_size, logger):
    """
    Creates DataLoaders for train, val, and test splits for pathway classification.
    """
    # Load pathway mappings first to get num_pathways
    num_pathways, id_to_pathway, pathway_to_id = load_pathway_mappings(preprocessed_data_dir)
    logger.info(f"Loaded {num_pathways} pathways from mappings.")
    
    loaders = {}
    for split in ["train", "val", "test"]:
        split_dir = preprocessed_data_dir / "hf_dataset" / split
        if not split_dir.exists():
            logger.warning(f"No preprocessed data found for {split} split. Skipping.")
            continue
        
        logger.info(f"Loading {split} dataset from {split_dir}...")
        dataset = PreprocessedPathwayDataset(split_dir, num_pathways)
        
        is_train = (split == "train")
        sampler = None
        if world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=is_train
            )
        
        batch_size = config.batch_size if is_train else config.eval_batch_size
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(is_train and sampler is None),
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"Created {split} loader with {len(dataset)} samples.")
        loaders[split] = loader
    
    return loaders, num_pathways, id_to_pathway, pathway_to_id