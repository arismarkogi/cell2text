"""
Data loading and preprocessing utilities for scGPT fine-tuning
"""
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from typing import Dict, Tuple, Optional, List
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
    
    def create_data_dict(self, tokenized_data, adata, task, mask_ratio=0.0):
        data_dict = {
            "gene_ids": tokenized_data["genes"],
            "values": tokenized_data["values"],
            "labels": torch.tensor(adata.obs[f"{task}_id"].values, dtype=torch.long),  # ✅ changed to "labels"
        }

        if mask_ratio > 0:
            masked_values = random_mask_value(
                tokenized_data["values"],
                mask_ratio=mask_ratio,
                mask_value=self.mask_value,
                pad_value=self.pad_value,
            )
            data_dict["masked_values"] = masked_values
            data_dict["input_values"] = tokenized_data["values"]
        else:
            data_dict["masked_values"] = tokenized_data["values"]

        if "batch_id" in adata.obs:
            batch_ids = adata.obs["batch_id"].astype("category").cat.codes.values
            data_dict["batch_labels"] = torch.tensor(batch_ids, dtype=torch.long)

        return data_dict
    
    def load_data_from_files(self, train_files: List[Path], val_files: List[Path], test_files: List[Path]) -> Tuple:
        """Load datasets from file lists without concatenation"""
        print("Processing datasets in batches...")
        
        # First pass: collect gene names and basic stats
        all_genes = set()
        total_cells = {"train": 0, "val": 0, "test": 0}
        
        file_splits = {
            "train": train_files,
            "val": val_files, 
            "test": test_files
        }
        
        # Collect gene universe and cell counts
        for split_name, files in file_splits.items():
            for file_path in files:
                print(f"Scanning {file_path} for genes...")
                adata_temp = sc.read(file_path)
                all_genes.update(adata_temp.var_names.tolist())
                total_cells[split_name] += adata_temp.shape[0]
                del adata_temp  # Free memory immediately
        
        print(f"Found {len(all_genes)} unique genes across all files")
        print(f"Cell counts: Train={total_cells['train']}, Val={total_cells['val']}, Test={total_cells['test']}")
        
        # Create a reference AnnData with all genes for consistent structure
        all_genes_list = sorted(list(all_genes))
        
        # Load and process each split separately
        def load_split_data(files, split_name):
            split_data = []
            
            for i, file_path in enumerate(files):
                print(f"Loading {split_name} batch {i+1}/{len(files)}: {file_path}")
                adata_batch = sc.read(file_path)

                #limit to 100 samples for debugging
                # if adata_batch.n_obs > 100:
                #     adata_batch = adata_batch[:100, :].copy()
                


                # --- 🔑 Handle duplicate genes by summing counts ---
                if adata_batch.var_names.has_duplicates:
                    print(f"Found {sum(adata_batch.var_names.duplicated())} duplicate genes, aggregating...")

                    var_names = adata_batch.var_names
                    unique_genes, inverse_indices = np.unique(var_names, return_inverse=True)
                    
                    # Create a sparse matrix where each row is a cell, each column is a unique gene
                    # We'll sum columns that map to the same gene
                    from scipy import sparse
                    
                    X = adata_batch.X  # should be sparse (csr or csc)
                    if not sparse.issparse(X):
                        X = sparse.csr_matrix(X)
                    
                    # Build a "grouping" matrix: shape (n_original_genes, n_unique_genes)
                    # Each column has 1s where original genes map to that unique gene
                    grouping_matrix = sparse.coo_matrix(
                        (np.ones(len(inverse_indices)), (np.arange(len(inverse_indices)), inverse_indices)),
                        shape=(len(var_names), len(unique_genes))
                    ).tocsr()
                    
                    # Multiply: X @ grouping_matrix → sums duplicate gene columns
                    X_dedup = X @ grouping_matrix  # still sparse!
                    
                    # Create new var DataFrame with unique genes
                    new_var = pd.DataFrame(index=unique_genes)
                    # Optional: preserve any var metadata by aggregating (e.g., mean, first, etc.)
                    # For now, we just keep index since metadata may not be consistent
                    
                    # Recreate AnnData with deduplicated data
                    adata_batch = sc.AnnData(
                        X=X_dedup,
                        obs=adata_batch.obs.copy(),
                        var=new_var,
                        uns=adata_batch.uns.copy() if hasattr(adata_batch, 'uns') else {},
                        obsm=adata_batch.obsm.copy() if hasattr(adata_batch, 'obsm') else {},
                    )
                
                # Add batch and split identifiers
                adata_batch.obs['batch_id'] = f"{split_name}_{i}"
                adata_batch.obs['str_batch'] = split_name
                adata_batch.obs['file_path'] = str(file_path)

                # Ensure consistent gene ordering with the global gene list
                missing_genes = [g for g in all_genes_list if g not in adata_batch.var_names]
                if missing_genes:
                    print(f"Adding {len(missing_genes)} missing genes to batch")
                    import scipy.sparse as sp
                    n_cells = adata_batch.shape[0]
                    n_missing = len(missing_genes)

                    if sp.issparse(adata_batch.X):
                        missing_data = sp.csr_matrix((n_cells, n_missing))
                        adata_batch.X = sp.hstack([adata_batch.X, missing_data])
                    else:
                        missing_data = np.zeros((n_cells, n_missing))
                        adata_batch.X = np.hstack([adata_batch.X, missing_data])

                    new_var = pd.DataFrame(index=missing_genes)
                    adata_batch.var = pd.concat([adata_batch.var, new_var])

                # Final reordering of genes
                adata_batch = adata_batch[:, all_genes_list].copy()
                split_data.append(adata_batch)
            
            return split_data


        
        # Load each split
        train_batches = load_split_data(train_files, "train")
        val_batches = load_split_data(val_files, "val")
        test_batches = load_split_data(test_files, "test")
        
        # For vocabulary setup, use first batch from each split
        sample_adata = train_batches[0].copy()
        sample_adata.obs["str_batch"] = "train"
        
        # Store batch data in the data loader for later use
        self.train_batches = train_batches
        self.val_batches = val_batches
        self.test_batches = test_batches
        self.all_genes_list = all_genes_list
        
        return sample_adata, train_batches, val_batches, test_batches
        
    def load_data(self, train_path: str, val_path: str, test_path: str) -> Tuple:
        """Load train, validation, and test datasets - now supports directory structure"""
        print("Loading datasets...")
        
        # Check if paths are directories (your case) or files (original case)
        train_path = Path(train_path)
        val_path = Path(val_path)
        test_path = Path(test_path)
        
        if train_path.is_dir() or not train_path.exists():
            # Assume directory structure like yours
            # Extract parent directory
            if train_path.parent.name == "raw_data":
                data_dir = train_path.parent
            else:
                data_dir = Path("raw_data")  # Default
                
            split_data = self.load_batch_data_from_directory(str(data_dir), ["train", "val", "test"])
            adata_train = split_data["train"]
            adata_val = split_data["val"]
            adata_test = split_data["test"]
        else:
            # Original file-based loading
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
        if task == "cell_type":
            label_key = "cell_type"
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
    
    def prepare_dataset_splits_from_batches(self, task: str):
        """Prepare train/val/test splits from batch lists"""
        print("Preparing dataset splits from batches...")
        
        # Process each split separately to avoid memory issues
        def process_split_batches(batch_list, split_name):
            print(f"Processing {split_name} batches...")
            processed_batches = []
            
            for i, adata_batch in enumerate(batch_list):
                print(f"Processing {split_name} batch {i+1}/{len(batch_list)}")
                
                # Preprocess this batch
                adata_processed = self.preprocess_data(adata_batch.copy(), is_raw_data=True)
                processed_batches.append(adata_processed)
            
            return processed_batches
        
        # Process all splits
        train_processed = process_split_batches(self.train_batches, "train")
        val_processed = process_split_batches(self.val_batches, "val") 
        test_processed = process_split_batches(self.test_batches, "test")
        
        # Collect all unique labels across all batches
        all_labels = set()
        for batch_list in [train_processed, val_processed, test_processed]:
            for adata_batch in batch_list:
                if task == "cell_type":
                    label_key = "cell_type"
                elif task == "disease":
                    label_key = "disease"
                elif task == "tissue":
                    label_key = "tissue"
                else:
                    raise ValueError(f"Unknown task: {task}")
                
                if label_key in adata_batch.obs.columns:
                    all_labels.update(adata_batch.obs[label_key].unique())
        
        all_labels = sorted(list(all_labels))
        id_to_label = {i: label for i, label in enumerate(all_labels)}
        label_to_id = {label: i for i, label in enumerate(all_labels)}
        
        print(f"Found {len(all_labels)} unique labels: {all_labels}")
        
        def add_label_ids(batch_list, split_name):
            for adata_batch in batch_list:
                if task in adata_batch.obs.columns:  # Check if task column exists (e.g., "celltype")
                    # Convert labels to IDs
                    labels = adata_batch.obs[task].values
                    label_ids = []
                    
                    for label in labels:
                        if label in label_to_id:
                            label_ids.append(label_to_id[label])
                        else:
                            print(f"❌ WARNING: Label '{label}' not found in label mapping!")
                            # Assign to first class as fallback
                            label_ids.append(0)
                    
                    adata_batch.obs[f"{task}_id"] = label_ids
                else:
                    print(f"❌ ERROR: No '{task}' column found in batch!")
                    print(f"Available columns: {adata_batch.obs.columns.tolist()}")
                    # Don't assign -1, this will cause the error!
                    raise ValueError(f"Missing '{task}' column in batch")
        
        add_label_ids(train_processed, "train")
        add_label_ids(val_processed, "val")
        add_label_ids(test_processed, "test")
        
        # Store processed batches
        self.train_processed = train_processed
        self.val_processed = val_processed  
        self.test_processed = test_processed
        
        return len(all_labels), id_to_label
    
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
    
    def create_batch_data_loaders(self, task: str, config):
        """Create data loaders from processed batches"""
        print("Creating data loaders from batches...")
        
        def process_batches_to_dataloader(batch_list, split_name, mask_ratio=0.0):
            all_data_dicts = []
            
            for i, adata_batch in enumerate(batch_list):
                print(f"Tokenizing {split_name} batch {i+1}/{len(batch_list)}")
                
                # Tokenize this batch
                tokenized, _, _ = self.tokenize_data(adata_batch, f"{split_name}_batch_{i}")
                
                # Create data dict for this batch
                data_dict = self.create_data_dict(
                    tokenized, adata_batch, task, mask_ratio=mask_ratio
                )
                
                all_data_dicts.append(data_dict)
            
            # Combine all data dicts
            combined_dict = {}
            for key in all_data_dicts[0].keys():
                combined_dict[key] = torch.cat([d[key] for d in all_data_dicts], dim=0)
            
            return combined_dict
        
        # Process each split
        train_data_dict = process_batches_to_dataloader(
            self.train_processed, "train", mask_ratio=config.mask_ratio
        )
        val_data_dict = process_batches_to_dataloader(
            self.val_processed, "val", mask_ratio=0.0
        )
        test_data_dict = process_batches_to_dataloader(
            self.test_processed, "test", mask_ratio=0.0
        )
        
        # Create data loaders
        train_loader = create_dataloader(
            train_data_dict, config.batch_size, shuffle=True
        )
        val_loader = create_dataloader(
            val_data_dict, config.eval_batch_size, shuffle=False
        )
        test_loader = create_dataloader(
            test_data_dict, config.eval_batch_size, shuffle=False
        )
        
        return train_loader, val_loader, test_loader


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
        try:
            num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)
        except AttributeError:
            # Windows doesn't have sched_getaffinity
            num_workers = min(4, batch_size // 2)
    
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