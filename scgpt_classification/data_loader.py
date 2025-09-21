"""
Data loading and preprocessing utilities for scGPT fine-tuning
"""
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from typing import Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, "../")
from scgpt.preprocess import Preprocessor
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt import SubsetsBatchSampler


class ScGPTDataLoader:
    def __init__(self, config, vocab=None):
        self.config = config
        self.vocab = vocab
        self.preprocessor = None
        self.special_tokens = ["<pad>", "<cls>", "<eoc>"]
        
        # Set preprocessing parameters
        self.pad_token = "<pad>"
        self.mask_value = -1 if config.input_emb_style != "category" else config.n_bins + 1
        self.pad_value = -2 if config.input_emb_style != "category" else config.n_bins
        
    def load_data(self, train_path: str, val_path: str, test_path: str) -> Tuple:
        """Load train, validation, and test datasets"""
        print("Loading datasets...")
        
        # Load h5ad files
        adata_train = sc.read(train_path)
        adata_val = sc.read(val_path) 
        adata_test = sc.read(test_path)
        
        # Add batch information
        adata_train.obs["str_batch"] = "train"
        adata_val.obs["str_batch"] = "val"
        adata_test.obs["str_batch"] = "test"
        
        # Concatenate all data for consistent preprocessing
        adata_all = adata_train.concatenate([adata_val, adata_test], batch_key="str_batch")
        
        return adata_all, adata_train, adata_val, adata_test
    
    def setup_vocabulary(self, adata, pretrained_vocab_path: Optional[str] = None):
        """Setup gene vocabulary"""
        if pretrained_vocab_path:
            self.vocab = GeneVocab.from_file(pretrained_vocab_path)
            for token in self.special_tokens:
                if token not in self.vocab:
                    self.vocab.append_token(token)
        else:
            from torchtext.vocab import Vocab
            from torchtext._torchtext import Vocab as VocabPybind
            genes = adata.var_names.tolist()
            self.vocab = Vocab(VocabPybind(genes + self.special_tokens, None))
        
        self.vocab.set_default_index(self.vocab["<pad>"])
        return self.vocab
    
    def filter_genes_by_vocab(self, adata):
        """Filter genes based on vocabulary"""
        if self.vocab is None:
            return adata
            
        adata.var["id_in_vocab"] = [
            1 if gene in self.vocab else -1 for gene in adata.var_names
        ]
        gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
        print(f"Matched {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes in vocabulary")
        
        return adata[:, adata.var["id_in_vocab"] >= 0]
    
    def preprocess_data(self, adata, is_raw_data: bool = True):
        """Preprocess the data"""
        self.preprocessor = Preprocessor(
            use_key="X",
            filter_gene_by_counts=False,
            filter_cell_by_counts=False,
            normalize_total=1e4,
            result_normed_key="X_normed",
            log1p=is_raw_data,
            result_log1p_key="X_log1p",
            subset_hvg=False,
            binning=self.config.n_bins,
            result_binned_key="X_binned",
        )
        
        self.preprocessor(adata, batch_key=None)
        return adata
    
    def prepare_labels(self, adata, task: str):
        """Prepare labels for classification task"""
        if task == "celltype":
            label_key = "celltype"
        elif task == "disease":
            label_key = "disease"
        elif task == "tissue":
            label_key = "tissue"
        else:
            raise ValueError(f"Unknown task: {task}")
        
        if label_key not in adata.obs.columns:
            raise ValueError(f"Label '{label_key}' not found in adata.obs")
        
        # Convert to categorical and get integer labels
        adata.obs[f"{label_key}_cat"] = adata.obs[label_key].astype("category")
        adata.obs[f"{label_key}_id"] = adata.obs[f"{label_key}_cat"].cat.codes.values
        
        # Create label mapping
        label_to_id = dict(enumerate(adata.obs[f"{label_key}_cat"].cat.categories))
        id_to_label = {v: k for k, v in label_to_id.items()}
        
        return adata, len(label_to_id), id_to_label
    
    def prepare_dataset_splits(self, adata_all, task: str):
        """Prepare train/val/test splits"""
        # Prepare labels
        adata_all, num_classes, id_to_label = self.prepare_labels(adata_all, task)
        
        # Split back into train/val/test
        adata_train = adata_all[adata_all.obs["str_batch"] == "train"].copy()
        adata_val = adata_all[adata_all.obs["str_batch"] == "val"].copy()
        adata_test = adata_all[adata_all.obs["str_batch"] == "test"].copy()
        
        return adata_train, adata_val, adata_test, num_classes, id_to_label
    
    def tokenize_data(self, adata, subset_name: str = ""):
        """Tokenize and prepare data for model input"""
        input_layer_key = {
            "normed_raw": "X_normed",
            "log1p": "X_normed", 
            "binned": "X_binned",
        }[self.config.input_style]
        
        # Get expression data
        if issparse(adata.layers[input_layer_key]):
            expression_data = adata.layers[input_layer_key].toarray()
        else:
            expression_data = adata.layers[input_layer_key]
        
        # Get gene IDs
        genes = adata.var_names.tolist()
        gene_ids = np.array(self.vocab(genes), dtype=int)
        
        # Tokenize
        tokenized = tokenize_and_pad_batch(
            expression_data,
            gene_ids,
            max_len=self.config.max_seq_len,
            vocab=self.vocab,
            pad_token=self.pad_token,
            pad_value=self.pad_value,
            append_cls=True,
            include_zero_gene=self.config.include_zero_gene,
        )
        
        print(f"{subset_name} set: {tokenized['genes'].shape[0]} samples, "
              f"feature length: {tokenized['genes'].shape[1]}")
        
        return tokenized, expression_data, gene_ids
    
    def create_data_dict(self, tokenized, adata, task: str, mask_ratio: float = 0.0):
        """Create data dictionary for training"""
        # Apply masking
        masked_values = random_mask_value(
            tokenized["values"],
            mask_ratio=mask_ratio,
            mask_value=self.mask_value,
            pad_value=self.pad_value,
        )
        
        # Get labels
        labels = adata.obs[f"{task}_id"].values
        batch_labels = np.zeros(len(labels))  # Single batch for now
        
        data_dict = {
            "gene_ids": tokenized["genes"],
            "values": masked_values,
            "target_values": tokenized["values"],
            "batch_labels": torch.from_numpy(batch_labels).long(),
            "labels": torch.from_numpy(labels).long(),
        }
        
        return data_dict


class SeqDataset(Dataset):
    """Dataset class for scGPT sequences"""
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def create_dataloader(data_dict: Dict[str, torch.Tensor], 
                     batch_size: int,
                     shuffle: bool = False,
                     num_workers: int = 0) -> DataLoader:
    """Create DataLoader from data dictionary"""
    if num_workers == 0:
        import os
        num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)
    
    dataset = SeqDataset(data_dict)
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return dataloader