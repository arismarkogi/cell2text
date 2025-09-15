import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from datasets import load_from_disk
import os
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class CellTypeDataset(Dataset):
    """Dataset class for cell type classification"""
    
    def __init__(self, base_path: str, split: str = 'train', min_samples_per_type=10):
        """
        Args:
            base_path: Base path containing split folders (train/validation/test)
            split: 'train', 'validation', or 'test'
            min_samples_per_type: Minimum number of samples required per cell type
        """
        self.base_path = base_path
        self.split = split
        self.min_samples_per_type = min_samples_per_type
        
        self.data = []
        self.labels = []
        self.dataset_ids = []
        self.cell_ids = []
        self.cell_type_names = []
        
        # These will be populated after loading data
        self.cell_type_to_idx = {}
        self.idx_to_cell_type = {}
        self.num_cell_types = 0
        
        self._load_all_datasets()
        self._create_cell_type_mapping()
        self._filter_by_frequency()
        
    def _load_all_datasets(self):
        """Load all datasets and collect cell type information"""
        split_path = os.path.join(self.base_path, self.split)
        
        if not os.path.exists(split_path):
            print(f"Warning: Split path '{split_path}' does not exist")
            return
            
        dataset_folders = [d for d in os.listdir(split_path) 
                          if d.startswith('dataset_') and 'descriptions' in d]
        dataset_folders.sort()
        
        print(f"Found {len(dataset_folders)} datasets: {dataset_folders}")
        
        # First pass: collect all cell types
        all_cell_types = []
        
        for dataset_folder in dataset_folders:
            dataset_path = os.path.join(split_path, dataset_folder)
            try:
                dataset = load_from_disk(dataset_path)
                print(f"Loading {dataset_folder}: {len(dataset)} samples")
                
                for idx, sample in enumerate(dataset):
                    cell_type = sample.get('cell_type', '').strip()
                    if cell_type:  # Only include samples with valid cell types
                        all_cell_types.append(cell_type)
                        
                        # Store the data
                        input_ids = sample['input_ids']
                        attention_mask = sample.get('attention_mask', [1] * len(input_ids))
                        
                        self.data.append({
                            'input_ids': input_ids,
                            'attention_mask': attention_mask
                        })
                        
                        self.cell_type_names.append(cell_type)
                        self.dataset_ids.append(dataset_folder)
                        self.cell_ids.append(f"{dataset_folder}_{idx}")
                        
            except Exception as e:
                print(f"Error loading {dataset_folder}: {e}")
        
        print(f"Total samples loaded: {len(self.data)}")
        
        # Analyze cell type distribution
        cell_type_counts = Counter(all_cell_types)
        print(f"Found {len(cell_type_counts)} unique cell types")
        print("Most common cell types:")
        for cell_type, count in cell_type_counts.most_common(10):
            print(f"  {cell_type}: {count} samples")
        
        return cell_type_counts
    
    def _create_cell_type_mapping(self):
        """Create mapping between cell type names and indices"""
        # Count frequencies first
        cell_type_counts = Counter(self.cell_type_names)
        
        # Only include cell types with sufficient samples
        valid_cell_types = [ct for ct, count in cell_type_counts.items() 
                           if count >= self.min_samples_per_type]
        valid_cell_types.sort()  # Consistent ordering
        
        print(f"Cell types with >= {self.min_samples_per_type} samples: {len(valid_cell_types)}")
        
        self.cell_type_to_idx = {cell_type: idx for idx, cell_type in enumerate(valid_cell_types)}
        self.idx_to_cell_type = {idx: cell_type for cell_type, idx in self.cell_type_to_idx.items()}
        self.num_cell_types = len(valid_cell_types)
        
        print(f"Final number of cell types: {self.num_cell_types}")
        print("Cell type mapping (first 10):")
        for i, (cell_type, idx) in enumerate(list(self.cell_type_to_idx.items())[:10]):
            print(f"  {idx}: {cell_type}")
    
    def _filter_by_frequency(self):
        """Filter data to only include valid cell types"""
        filtered_data = []
        filtered_labels = []
        filtered_dataset_ids = []
        filtered_cell_ids = []
        
        for i in range(len(self.data)):
            cell_type = self.cell_type_names[i]
            if cell_type in self.cell_type_to_idx:
                filtered_data.append(self.data[i])
                filtered_labels.append(self.cell_type_to_idx[cell_type])
                filtered_dataset_ids.append(self.dataset_ids[i])
                filtered_cell_ids.append(self.cell_ids[i])
        
        self.data = filtered_data
        self.labels = filtered_labels
        self.dataset_ids = filtered_dataset_ids
        self.cell_ids = filtered_cell_ids
        
        print(f"Samples after filtering: {len(self.data)}")
        
        # Final distribution
        final_counts = Counter(self.labels)
        print("Final cell type distribution:")
        for idx in sorted(final_counts.keys())[:10]:  # Show first 10
            cell_type = self.idx_to_cell_type[idx]
            count = final_counts[idx]
            print(f"  {idx} ({cell_type}): {count} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.data[idx]['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(self.data[idx]['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),  # Single class label
            'dataset_id': self.dataset_ids[idx],
            'cell_id': self.cell_ids[idx]
        }
    
    def get_label_statistics(self):
        """Get statistics about the labels"""
        if len(self.labels) == 0:
            return {'total_samples': 0, 'num_classes': 0}
        
        labels_array = np.array(self.labels)
        class_counts = np.bincount(labels_array, minlength=self.num_cell_types)
        
        stats = {
            'total_samples': len(self.labels),
            'num_classes': self.num_cell_types,
            'class_distribution': {self.idx_to_cell_type[i]: int(class_counts[i]) 
                                 for i in range(self.num_cell_types)},
            'min_samples_per_class': int(np.min(class_counts[class_counts > 0])),
            'max_samples_per_class': int(np.max(class_counts)),
            'mean_samples_per_class': float(np.mean(class_counts[class_counts > 0]))
        }
        
        return stats
    
    def get_class_weights(self):
        """Calculate class weights for handling imbalanced data"""
        if len(self.labels) == 0:
            return None
        
        labels_array = np.array(self.labels)
        class_counts = np.bincount(labels_array, minlength=self.num_cell_types)
        
        # Calculate inverse frequency weights
        total_samples = len(labels_array)
        weights = total_samples / (self.num_cell_types * class_counts)
        
        # Handle zero counts
        weights[class_counts == 0] = 0
        
        return torch.FloatTensor(weights)