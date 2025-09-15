import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

from datasets import load_from_disk
import os
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import Dataset
import numpy as np
from datasets import load_from_disk
import os
import warnings
warnings.filterwarnings('ignore')

class MultiDatasetPathwayDataset(Dataset):
    def __init__(self, base_path: str, split: str = 'train', target_pathways=None, max_samples_per_dataset=None):
        self.base_path = base_path
        self.split = split
        self.datasets = []         # hold Arrow dataset objects (do NOT materialize into python lists)
        self.index_map = []        # list of (dataset_idx, local_idx) for valid samples
        self.dataset_names = []    # parallel list of dataset folder names

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

        self.pathway_to_idx = {p: i for i, p in enumerate(self.target_pathways)}
        self.num_pathways = len(self.target_pathways)

        print(f"Target pathways ({self.num_pathways}): {self.target_pathways[:5]}... (showing first 5)")
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

            # iterate dataset to find valid samples but do NOT copy token arrays into python lists
            total = len(ds)
            # optionally limit how many samples to scan per dataset for indexing (useful while debugging)
            scan_limit = total if max_samples_per_dataset is None else min(total, max_samples_per_dataset)

            for local_idx in range(scan_limit):
                sample = ds[local_idx]
                pathway1 = sample.get('pathway1', '').strip()
                pathway2 = sample.get('pathway2', '').strip()
                p1_idx = self.pathway_to_idx.get(pathway1, -1)
                p2_idx = self.pathway_to_idx.get(pathway2, -1)
                if p1_idx != -1 and p2_idx != -1:
                    # append index mapping only — don't materialize input arrays
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

        pathway1 = sample.get('pathway1', '').strip()
        pathway2 = sample.get('pathway2', '').strip()
        p1_idx = self.pathway_to_idx.get(pathway1, -1)
        p2_idx = self.pathway_to_idx.get(pathway2, -1)

        labels = torch.zeros(self.num_pathways, dtype=torch.float32)
        if p1_idx != -1:
            labels[p1_idx] = 1.0
        if p2_idx != -1:
            labels[p2_idx] = 1.0

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'dataset_id': self.dataset_names[ds_idx],
            'cell_id': f"{self.dataset_names[ds_idx]}_{local_idx}"
        }

    def get_label_statistics(self):
        if len(self.index_map) == 0:
            return {
                'total_samples': 0,
                'pathway1_distribution': {},
                'pathway2_distribution': {},
                'pathway_pair_distribution': {}
            }

        pairs = []
        for ds_idx, local_idx in self.index_map:
            sample = self.datasets[ds_idx][local_idx]
            pairs.append((sample.get('pathway1','').strip(), sample.get('pathway2','').strip()))

        from collections import Counter
        pair_counts = Counter(pairs)
        stats = {
            'total_samples': len(self.index_map),
            'num_pathways': self.num_pathways,
            'most_common_pairs': dict(pair_counts.most_common(10))
        }
        return stats
