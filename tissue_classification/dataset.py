import torch
from torch.utils.data import Dataset
import numpy as np
from datasets import load_from_disk
import os
from collections import Counter

import pandas as pd



class MultiDatasetTissueDataset(Dataset):
    def __init__(self, base_path: str, split: str = 'train', target_tissues=None, max_samples_per_dataset=None, label_key='tissue'):
        self.base_path = base_path
        self.split = split
        self.datasets = []         # hold Arrow dataset objects
        self.index_map = []        # list of (dataset_idx, local_idx)
        self.dataset_names = []    # parallel list of dataset folder names
        self.label_key = label_key

        
        self.target_tissues = target_tissues

        self.tissue_to_idx = {ct: i for i, ct in enumerate(self.target_tissues)}
        self.num_tissues = len(self.target_tissues)

        print(f"Target tissue types ({self.num_tissues}): {self.target_tissues[:5]}... (showing first 5)")
        self._discover_and_index(max_samples_per_dataset)

    def _discover_and_index(self, max_samples_per_dataset=None):
        split_path = os.path.join(self.base_path, self.split)
        if not os.path.exists(split_path):
            print(f"Warning: Split path '{split_path}' does not exist")
            return

        dataset_folders = [d for d in os.listdir(split_path) if d.startswith('dataset_') and 'descriptions' in d]
        dataset_folders.sort()
        print(f"Found {len(dataset_folders)} datasets: {dataset_folders}")

        for ds_idx, dataset_folder in enumerate(dataset_folders):
            dataset_path = os.path.join(split_path, dataset_folder)
            try:
                ds = load_from_disk(dataset_path)
            except Exception as e:
                print(f"Error loading {dataset_folder}: {e}")
                continue

            self.datasets.append(ds)
            self.dataset_names.append(dataset_folder)
            valid_in_dataset = 0

            total = len(ds)
            scan_limit = total if max_samples_per_dataset is None else min(total, max_samples_per_dataset)

            for local_idx in range(scan_limit):
                sample = ds[local_idx]
                tissue = sample.get(self.label_key, '').strip()
                if tissue in self.tissue_to_idx:
                    self.index_map.append((ds_idx, local_idx))
                    valid_in_dataset += 1

            print(f"  {dataset_folder}: scanned {scan_limit}/{total}, valid samples: {valid_in_dataset}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        ds_idx, local_idx = self.index_map[idx]
        ds = self.datasets[ds_idx]
        sample = ds[local_idx]

        input_ids = torch.tensor(sample['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(sample.get('attention_mask', [1] * len(input_ids)), dtype=torch.long)

        tissue = sample.get(self.label_key, '').strip()
        label_idx = self.tissue_to_idx.get(tissue, -1)

        if label_idx == -1:
            # Should not happen if _discover_and_index worked correctly
            label_idx = 0

        # RETURN CLASS INDEX (for CrossEntropyLoss)
        labels = torch.tensor(label_idx, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'dataset_id': self.dataset_names[ds_idx],
            'tissue_id': f"{self.dataset_names[ds_idx]}_{local_idx}"
        }

    