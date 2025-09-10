import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

from datasets import load_from_disk
import os
import warnings
warnings.filterwarnings('ignore')

class MultiDatasetPathwayDataset(Dataset):
    """Dataset class that combines multiple tokenized datasets"""
    
    def __init__(self, base_path: str, split: str = 'train', target_pathways=None):
        """
        Args:
            base_path: Base path containing split folders (train/validation/test)
            split: 'train', 'validation', or 'test'
            target_pathways: List of target pathway names
        """
        self.base_path = base_path
        self.split = split
        self.data = []
        self.labels = []
        self.dataset_ids = []
        self.cell_ids = []
        
        # Define target pathways
        if target_pathways is None:
            self.target_pathways = [
                "HALLMARK_ANDROGEN_RESPONSE",
                "HALLMARK_APOPTOSIS",
                "HALLMARK_UV_RESPONSE_DN",
                "HALLMARK_INTERFERON_GAMMA_RESPONSE",
                "HALLMARK_HEDGEHOG_SIGNALING",
                "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
                "HALLMARK_ALLOGRAFT_REJECTION",
                "HALLMARK_INTERFERON_ALPHA_RESPONSE",
                "HALLMARK_CHOLESTEROL_HOMEOSTASIS",
                "HALLMARK_ANGIOGENESIS",
                "HALLMARK_NOTCH_SIGNALING",
                "HALLMARK_MYC_TARGETS_V2",
                "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
                "HALLMARK_P53_PATHWAY",
                "HALLMARK_PANCREAS_BETA_CELLS",
                "HALLMARK_HYPOXIA",
                "HALLMARK_WNT_BETA_CATENIN_SIGNALING",
                "HALLMARK_APICAL_SURFACE",
                "HALLMARK_IL6_JAK_STAT3_SIGNALING",
                "HALLMARK_MYOGENESIS",
                "HALLMARK_COMPLEMENT",
                "HALLMARK_ESTROGEN_RESPONSE_LATE",
                "HALLMARK_HEME_METABOLISM",
                "HALLMARK_ESTROGEN_RESPONSE_EARLY",
                "HALLMARK_APICAL_JUNCTION",
                "HALLMARK_XENOBIOTIC_METABOLISM",
                "HALLMARK_COAGULATION",
                "HALLMARK_INFLAMMATORY_RESPONSE",
                "HALLMARK_GLYCOLYSIS",
                "HALLMARK_BILE_ACID_METABOLISM",
                "HALLMARK_KRAS_SIGNALING_UP",
                "HALLMARK_SPERMATOGENESIS",
                "HALLMARK_IL2_STAT5_SIGNALING",
                "HALLMARK_KRAS_SIGNALING_DN"
            ]
        else:
            self.target_pathways = target_pathways
        
        # Create pathway name to index mapping
        self.pathway_to_idx = {pathway: idx for idx, pathway in enumerate(self.target_pathways)}
        self.num_pathways = len(self.target_pathways)
        
        print(f"Target pathways ({self.num_pathways}): {self.target_pathways[:5]}... (showing first 5)")
        
        self._load_all_datasets()
        
    def _load_all_datasets(self):
        """Load all datasets from the base path"""
        # Construct the path to the split directory
        split_path = os.path.join(self.base_path, self.split)
        
        if not os.path.exists(split_path):
            print(f"Warning: Split path '{split_path}' does not exist")
            return
            
        dataset_folders = [d for d in os.listdir(split_path) 
                          if d.startswith('dataset_') and 'descriptions' in d]
        dataset_folders.sort()  # Ensure consistent ordering
        
        print(f"Found {len(dataset_folders)} datasets: {dataset_folders}")
        
        for dataset_folder in dataset_folders:
            dataset_path = os.path.join(split_path, dataset_folder)
            self._load_single_dataset(dataset_path, dataset_folder)
    
    def _load_single_dataset(self, dataset_path: str, dataset_name: str):
        """Load a single dataset"""
        try:
            # Load the tokenized dataset directly (it's already a single split)
            dataset = load_from_disk(dataset_path)
            
            print(f"Loading {dataset_name}: {len(dataset)} samples")
            
            # Track unknown pathways
            unknown_pathway1 = set()
            unknown_pathway2 = set()
            valid_samples = 0
            
            for idx, sample in enumerate(dataset):

                if idx > 1000:
                    break  # Limit to first 10 samples for debugging

               
                # Extract data
                input_ids = sample['input_ids']
                attention_mask = sample.get('attention_mask', [1] * len(input_ids))
                
                # Extract pathway names
                pathway1_name = sample.get('pathway1', '').strip()
                pathway2_name = sample.get('pathway2', '').strip()
                
                # Convert pathway names to indices
                pathway1_idx = self.pathway_to_idx.get(pathway1_name, -1)
                pathway2_idx = self.pathway_to_idx.get(pathway2_name, -1)
                
                # Track unknown pathways
                if pathway1_name and pathway1_idx == -1:
                    unknown_pathway1.add(pathway1_name)
                if pathway2_name and pathway2_idx == -1:
                    unknown_pathway2.add(pathway2_name)
                
                # Only include samples where both pathways are in our target list
                if pathway1_idx != -1 and pathway2_idx != -1:
                    self.data.append({
                        'input_ids': input_ids,
                        'attention_mask': attention_mask
                    })
                    
                    self.labels.append([pathway1_idx, pathway2_idx])
                    self.dataset_ids.append(dataset_name)
                    self.cell_ids.append(f"{dataset_name}_{idx}")
                    valid_samples += 1
                    
                    # Debug: Print first few samples
                    if valid_samples <= 3:
                        print(f"  Sample {valid_samples}: '{pathway1_name}' -> {pathway1_idx}, '{pathway2_name}' -> {pathway2_idx}")
            
            print(f"  Valid samples (both pathways in target list): {valid_samples}/{len(dataset)}")
            
            if unknown_pathway1:
                print(f"  Unknown pathway1 names: {sorted(list(unknown_pathway1))[:5]}{'...' if len(unknown_pathway1) > 5 else ''}")
            if unknown_pathway2:
                print(f"  Unknown pathway2 names: {sorted(list(unknown_pathway2))[:5]}{'...' if len(unknown_pathway2) > 5 else ''}")
                
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Create binary vector instead of indices
        binary_labels = torch.zeros(self.num_pathways, dtype=torch.float32)
        pathway1_idx, pathway2_idx = self.labels[idx]
        binary_labels[pathway1_idx] = 1.0
        binary_labels[pathway2_idx] = 1.0
        
        return {
            'input_ids': torch.tensor(self.data[idx]['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(self.data[idx]['attention_mask'], dtype=torch.long),
            'labels': binary_labels,  # Now shape: (34,)
            'dataset_id': self.dataset_ids[idx],
            'cell_id': self.cell_ids[idx]
        }
    
    def get_label_statistics(self):
        """Get statistics about the labels"""
        if len(self.labels) == 0:
            return {
                'total_samples': 0,
                'pathway1_distribution': {},
                'pathway2_distribution': {},
                'pathway_pair_distribution': {}
            }
        
        labels_array = np.array(self.labels, dtype=np.int32)
        
        # Calculate distributions
        pathway1_counts = np.bincount(labels_array[:, 0], minlength=self.num_pathways)
        pathway2_counts = np.bincount(labels_array[:, 1], minlength=self.num_pathways)
        
        # Create distribution dictionaries
        pathway1_dist = {self.target_pathways[i]: int(pathway1_counts[i]) for i in range(self.num_pathways)}
        pathway2_dist = {self.target_pathways[i]: int(pathway2_counts[i]) for i in range(self.num_pathways)}
        
        # Most common pathway pairs
        from collections import Counter
        pairs = [(self.target_pathways[p1], self.target_pathways[p2]) for p1, p2 in labels_array]
        pair_counts = Counter(pairs)
        
        stats = {
            'total_samples': len(self.labels),
            'num_pathways': self.num_pathways,
            'pathway1_distribution': pathway1_dist,
            'pathway2_distribution': pathway2_dist,
            'most_common_pairs': dict(pair_counts.most_common(10)),
            'unique_pathway1_count': int(np.sum(pathway1_counts > 0)),
            'unique_pathway2_count': int(np.sum(pathway2_counts > 0))
        }
        
        return stats