# """Simplified data loading for scGPT fine-tuning"""

# import numpy as np
# import pandas as pd
# import scanpy as sc
# import torch
# import gc
# import sys
# from pathlib import Path
# from scipy.sparse import issparse
# from torch.utils.data import Dataset, DataLoader

# sys.path.insert(0, "../")
# from scgpt.preprocess import Preprocessor
# from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
# from scgpt.tokenizer.gene_tokenizer import GeneVocab


# class SeqDataset(Dataset):
#     def __init__(self, data):
#         self.data = data
    
#     def __len__(self):
#         return self.data["gene_ids"].shape[0]
    
#     def __getitem__(self, idx):
#         return {k: v[idx] for k, v in self.data.items()}


# class ScGPTDataLoader:
#     def __init__(self, config, vocab=None):
#         self.config = config
#         self.vocab = vocab
#         self.preprocessor = None
#         self.special_tokens = ["<pad>", "<cls>", "<eoc>"]
        
#         self.pad_token = "<pad>"
#         self.mask_value = -1 if config.input_emb_style != "category" else config.n_bins + 1
#         self.pad_value = -2 if config.input_emb_style != "category" else config.n_bins
    
#     def load_data_from_files(self, train_files, val_files, test_files):
#         """Load and align all data files"""
#         print("Loading data files...")
        
#         # Collect all unique genes
#         all_genes = set()
#         for files in [train_files, val_files, test_files]:
#             for f in files:
#                 adata = sc.read(f)
#                 all_genes.update(adata.var_names.tolist())
#                 del adata
        
#         all_genes_list = sorted(list(all_genes))
#         print(f"Found {len(all_genes_list)} unique genes")
        
#         def load_split(files, split_name):
#             batches = []
#             for i, fpath in enumerate(files):
#                 print(f"Loading {split_name} batch {i+1}/{len(files)}")
#                 adata = sc.read(fpath)
                
#                 # Handle duplicate genes by summing
#                 if adata.var_names.has_duplicates:
#                     print(f"Aggregating {sum(adata.var_names.duplicated())} duplicate genes")
#                     unique_genes, inverse = np.unique(adata.var_names, return_inverse=True)
                    
#                     from scipy import sparse
#                     X = adata.X if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
                    
#                     grouping = sparse.coo_matrix(
#                         (np.ones(len(inverse)), (np.arange(len(inverse)), inverse)),
#                         shape=(len(adata.var_names), len(unique_genes))
#                     ).tocsr()
                    
#                     X_dedup = X @ grouping
#                     adata = sc.AnnData(X=X_dedup, obs=adata.obs.copy(), var=pd.DataFrame(index=unique_genes))
                
#                 # Add batch info
#                 adata.obs['batch_id'] = f"{split_name}_{i}"
#                 adata.obs['str_batch'] = split_name
                
#                 # Add missing genes
#                 missing = [g for g in all_genes_list if g not in adata.var_names]
#                 if missing:
#                     from scipy import sparse
#                     n_cells, n_missing = adata.shape[0], len(missing)
                    
#                     if sparse.issparse(adata.X):
#                         adata.X = sparse.hstack([adata.X, sparse.csr_matrix((n_cells, n_missing))])
#                     else:
#                         adata.X = np.hstack([adata.X, np.zeros((n_cells, n_missing))])
                    
#                     adata.var = pd.concat([adata.var, pd.DataFrame(index=missing)])
                
#                 # Reorder genes
#                 adata = adata[:, all_genes_list].copy()
#                 batches.append(adata)
            
#             return batches
        
#         train_batches = load_split(train_files, "train")
#         val_batches = load_split(val_files, "val")
#         test_batches = load_split(test_files, "test")
        
#         self.train_batches = train_batches
#         self.val_batches = val_batches
#         self.test_batches = test_batches
#         self.all_genes_list = all_genes_list
        
#         return train_batches[0].copy(), train_batches, val_batches, test_batches
    
#     def setup_vocabulary(self, adata, pretrained_vocab_path=None):
#         """Setup gene vocabulary"""
#         if pretrained_vocab_path:
#             self.vocab = GeneVocab.from_file(pretrained_vocab_path)
#             for token in self.special_tokens:
#                 if token not in self.vocab:
#                     self.vocab.append_token(token)
#         else:
#             from torchtext.vocab import Vocab
#             from torchtext._torchtext import Vocab as VocabPybind
#             genes = adata.var_names.tolist()
#             self.vocab = Vocab(VocabPybind(genes + self.special_tokens, None))
        
#         self.vocab.set_default_index(self.vocab["<pad>"])
#         return self.vocab
    
#     def filter_genes_by_vocab(self, adata):
#         """Filter genes by vocabulary"""
#         if not self.vocab:
#             return adata
        
#         adata.var["id_in_vocab"] = [1 if gene in self.vocab else -1 for gene in adata.var_names]
#         gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
#         print(f"Matched {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes")
        
#         return adata[:, adata.var["id_in_vocab"] >= 0]
    
#     def preprocess_data(self, adata, is_raw=True, chunk_size=100000):
#         """Preprocess data with chunking"""
#         print(f"Preprocessing {adata.n_obs} cells...")
        
#         self.preprocessor = Preprocessor(
#             use_key="X",
#             filter_gene_by_counts=False,
#             filter_cell_by_counts=False,
#             normalize_total=1e4,
#             result_normed_key="X_normed",
#             log1p=is_raw,
#             result_log1p_key="X_log1p",
#             subset_hvg=False,
#             binning=self.config.n_bins,
#             result_binned_key="X_binned",
#         )
        
#         if adata.n_obs <= chunk_size:
#             self.preprocessor(adata, batch_key=None)
#             return adata
        
#         # Process in chunks
#         import tempfile
#         temp_dir = tempfile.mkdtemp(prefix="scgpt_")
#         temp_files = []
        
#         for start in range(0, adata.n_obs, chunk_size):
#             end = min(start + chunk_size, adata.n_obs)
#             chunk = adata[start:end, :].copy()
#             self.preprocessor(chunk, batch_key=None)
            
#             temp_file = f"{temp_dir}/chunk_{start//chunk_size}.h5ad"
#             chunk.write(temp_file)
#             temp_files.append(temp_file)
            
#             del chunk
#             gc.collect()
        
#         return temp_files
    
#     def prepare_dataset_splits_from_batches(self, task, chunk_size=1000):
#         """Prepare splits with labels"""
#         print("Preparing dataset splits...")
        
#         def process_split(batch_list, split_name):
#             processed = []
#             for i, batch in enumerate(batch_list):
#                 print(f"Processing {split_name} batch {i+1}/{len(batch_list)}")
#                 result = self.preprocess_data(batch.copy(), is_raw=True, chunk_size=chunk_size)
                
#                 if isinstance(result, list):
#                     processed.extend([sc.read(f) for f in result])
#                 else:
#                     processed.append(result)
                
#                 del batch
#                 gc.collect()
            
#             return processed
        
#         train_processed = process_split(self.train_batches, "train")
#         val_processed = process_split(self.val_batches, "val")
#         test_processed = process_split(self.test_batches, "test")
        
#         # Collect all labels
#         all_labels = set()
#         for batches in [train_processed, val_processed, test_processed]:
#             for batch in batches:
#                 if task in batch.obs.columns:
#                     all_labels.update(batch.obs[task].unique())
        
#         all_labels = sorted(list(all_labels))
#         id_to_label = {i: label for i, label in enumerate(all_labels)}
#         label_to_id = {label: i for i, label in enumerate(all_labels)}
        
#         print(f"Found {len(all_labels)} unique labels")
        
#         # Add label IDs
#         for batches in [train_processed, val_processed, test_processed]:
#             for batch in batches:
#                 if task not in batch.obs.columns:
#                     raise ValueError(f"Missing '{task}' column")
                
#                 batch.obs[f"{task}_id"] = [label_to_id[label] for label in batch.obs[task].values]
        
#         self.train_processed = train_processed
#         self.val_processed = val_processed
#         self.test_processed = test_processed
        
#         return len(all_labels), id_to_label
    
#     def tokenize_data(self, adata, subset_name=""):
#         """Tokenize data"""
#         input_layer = {
#             "normed_raw": "X_normed",
#             "log1p": "X_normed",
#             "binned": "X_binned",
#         }[self.config.input_style]
        
#         expr_data = adata.layers[input_layer].toarray() if issparse(adata.layers[input_layer]) else adata.layers[input_layer]
#         genes = adata.var_names.tolist()
#         gene_ids = np.array(self.vocab(genes), dtype=int)
        
#         tokenized = tokenize_and_pad_batch(
#             expr_data,
#             gene_ids,
#             max_len=self.config.max_seq_len,
#             vocab=self.vocab,
#             pad_token=self.pad_token,
#             pad_value=self.pad_value,
#             append_cls=True,
#             include_zero_gene=self.config.include_zero_gene,
#         )
        
#         print(f"{subset_name}: {tokenized['genes'].shape[0]} samples")
#         return tokenized, expr_data, gene_ids
    
#     def create_data_dict(self, tokenized, adata, task, mask_ratio=0.0):
#         """Create data dictionary"""
#         data_dict = {
#             "gene_ids": tokenized["genes"],
#             "values": tokenized["values"],
#             "labels": torch.tensor(adata.obs[f"{task}_id"].values, dtype=torch.long),
#         }
        
#         if mask_ratio > 0:
#             masked = random_mask_value(tokenized["values"], mask_ratio, self.mask_value, self.pad_value)
#             data_dict["masked_values"] = masked
#             data_dict["input_values"] = tokenized["values"]
#         else:
#             data_dict["masked_values"] = tokenized["values"]
        
#         if "batch_id" in adata.obs:
#             batch_ids = adata.obs["batch_id"].astype("category").cat.codes.values
#             data_dict["batch_labels"] = torch.tensor(batch_ids, dtype=torch.long)
        
#         return data_dict
    
#     def create_batch_data_loaders(self, task, config):
#         """Create data loaders from batches"""
#         print("Creating data loaders...")
        
#         def process_batches(batch_list, split_name, mask_ratio=0.0):
#             all_dicts = []
            
#             for i, batch in enumerate(batch_list):
#                 print(f"Tokenizing {split_name} batch {i+1}/{len(batch_list)}")
#                 tokenized, _, _ = self.tokenize_data(batch, f"{split_name} batch {i+1}")
#                 data_dict = self.create_data_dict(tokenized, batch, task, mask_ratio)
#                 all_dicts.append(data_dict)
                
#                 del tokenized, batch
#                 gc.collect()
            
#             # Combine all batches
#             combined = {k: torch.cat([d[k] for d in all_dicts], dim=0) for k in all_dicts[0].keys()}
#             del all_dicts
#             gc.collect()
            
#             return combined
        
#         train_data = process_batches(self.train_processed, "train", config.mask_ratio)
#         val_data = process_batches(self.val_processed, "val", 0.0)
#         test_data = process_batches(self.test_processed, "test", 0.0)
        
#         train_loader = self.create_dataloader(train_data, config.batch_size, shuffle=True)
#         val_loader = self.create_dataloader(val_data, config.eval_batch_size, shuffle=False)
#         test_loader = self.create_dataloader(test_data, config.eval_batch_size, shuffle=False)
        
#         return train_loader, val_loader, test_loader
    
#     def create_dataloader(self, data_dict, batch_size, shuffle=False):
#         """Create DataLoader"""
#         dataset = SeqDataset(data_dict)
#         return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)



"""Simplified data loading for scGPT fine-tuning with chunked processing"""

"""Simplified data loading for scGPT fine-tuning with chunked processing"""

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import gc
import sys
from pathlib import Path
from scipy.sparse import issparse
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "../")
from scgpt.preprocess import Preprocessor
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.tokenizer.gene_tokenizer import GeneVocab


class SeqDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return self.data["gene_ids"].shape[0]
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


class ScGPTDataLoader:
    def __init__(self, config, vocab=None):
        self.config = config
        self.vocab = vocab
        self.preprocessor = None
        self.special_tokens = ["<pad>", "<cls>", "<eoc>"]
        
        self.pad_token = "<pad>"
        self.mask_value = -1 if config.input_emb_style != "category" else config.n_bins + 1
        self.pad_value = -2 if config.input_emb_style != "category" else config.n_bins
    
    def load_data_from_files(self, train_files, val_files, test_files):
        """Load and align all data files"""
        print("Loading data files...")
        
        # Collect all unique genes
        all_genes = set()
        for files in [train_files, val_files, test_files]:
            for f in files:
                adata = sc.read(f)
                all_genes.update(adata.var_names.tolist())
                del adata
        
        all_genes_list = sorted(list(all_genes))
        print(f"Found {len(all_genes_list)} unique genes")
        
        def load_split(files, split_name):
            batches = []
            for i, fpath in enumerate(files):
                print(f"Loading {split_name} batch {i+1}/{len(files)}")
                adata = sc.read(fpath)
                
                # Handle duplicate genes by summing
                if adata.var_names.has_duplicates:
                    print(f"Aggregating {sum(adata.var_names.duplicated())} duplicate genes")
                    unique_genes, inverse = np.unique(adata.var_names, return_inverse=True)
                    
                    from scipy import sparse
                    X = adata.X if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
                    
                    grouping = sparse.coo_matrix(
                        (np.ones(len(inverse)), (np.arange(len(inverse)), inverse)),
                        shape=(len(adata.var_names), len(unique_genes))
                    ).tocsr()
                    
                    X_dedup = X @ grouping
                    adata = sc.AnnData(X=X_dedup, obs=adata.obs.copy(), var=pd.DataFrame(index=unique_genes))
                
                # Add batch info
                adata.obs['batch_id'] = f"{split_name}_{i}"
                adata.obs['str_batch'] = split_name
                
                # Add missing genes
                missing = [g for g in all_genes_list if g not in adata.var_names]
                if missing:
                    from scipy import sparse
                    n_cells, n_missing = adata.shape[0], len(missing)
                    
                    if sparse.issparse(adata.X):
                        adata.X = sparse.hstack([adata.X, sparse.csr_matrix((n_cells, n_missing))])
                    else:
                        adata.X = np.hstack([adata.X, np.zeros((n_cells, n_missing))])
                    
                    adata.var = pd.concat([adata.var, pd.DataFrame(index=missing)])
                
                # Reorder genes
                adata = adata[:, all_genes_list].copy()
                batches.append(adata)
            
            return batches
        
        train_batches = load_split(train_files, "train")
        val_batches = load_split(val_files, "val")
        test_batches = load_split(test_files, "test")
        
        self.train_batches = train_batches
        self.val_batches = val_batches
        self.test_batches = test_batches
        self.all_genes_list = all_genes_list
        
        return train_batches[0].copy(), train_batches, val_batches, test_batches
    
    def setup_vocabulary(self, adata, pretrained_vocab_path=None):
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
        """Filter genes by vocabulary"""
        if not self.vocab:
            return adata
        
        adata.var["id_in_vocab"] = [1 if gene in self.vocab else -1 for gene in adata.var_names]
        gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
        print(f"Matched {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes")
        
        return adata[:, adata.var["id_in_vocab"] >= 0]
    
    def preprocess_data(self, adata, is_raw=True):
        """Preprocess single adata object"""
        print(f"Preprocessing {adata.n_obs} cells...")
        
        if self.preprocessor is None:
            self.preprocessor = Preprocessor(
                use_key="X",
                filter_gene_by_counts=False,
                filter_cell_by_counts=False,
                normalize_total=1e4,
                result_normed_key="X_normed",
                log1p=is_raw,
                result_log1p_key="X_log1p",
                subset_hvg=False,
                binning=self.config.n_bins,
                result_binned_key="X_binned",
            )
        
        self.preprocessor(adata, batch_key=None)
        return adata
    
    def get_label_mapping(self, task):
        """Get label mappings from all batches"""
        all_labels = set()
        for batches in [self.train_batches, self.val_batches, self.test_batches]:
            for batch in batches:
                if task in batch.obs.columns:
                    all_labels.update(batch.obs[task].unique())
        
        all_labels = sorted(list(all_labels))
        id_to_label = {i: label for i, label in enumerate(all_labels)}
        label_to_id = {label: i for i, label in enumerate(all_labels)}
        
        print(f"Found {len(all_labels)} unique labels")
        return len(all_labels), id_to_label, label_to_id
    
    def process_and_create_loader(self, batch, task, label_to_id, batch_size, shuffle=False):
        """Process a single batch and create dataloader"""
        # Preprocess
        batch_copy = batch.copy()
        batch_processed = self.preprocess_data(batch_copy, is_raw=True)
        
        # Add label IDs
        if task not in batch_processed.obs.columns:
            raise ValueError(f"Missing '{task}' column")
        batch_processed.obs[f"{task}_id"] = [label_to_id[label] for label in batch_processed.obs[task].values]
        
        # Tokenize
        tokenized, _, _ = self.tokenize_data(batch_processed, "")
        
        # Create data dict
        mask_ratio = self.config.mask_ratio if shuffle else 0.0  # Only mask training data
        data_dict = self.create_data_dict(tokenized, batch_processed, task, mask_ratio)
        
        # Create loader
        dataset = SeqDataset(data_dict)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
        
        del batch_copy, batch_processed, tokenized, data_dict, dataset
        gc.collect()
        
        return loader
    
    def tokenize_data(self, adata, subset_name=""):
        """Tokenize data"""
        input_layer = {
            "normed_raw": "X_normed",
            "log1p": "X_normed",
            "binned": "X_binned",
        }[self.config.input_style]
        
        expr_data = adata.layers[input_layer].toarray() if issparse(adata.layers[input_layer]) else adata.layers[input_layer]
        genes = adata.var_names.tolist()
        gene_ids = np.array(self.vocab(genes), dtype=int)
        
        tokenized = tokenize_and_pad_batch(
            expr_data,
            gene_ids,
            max_len=self.config.max_seq_len,
            vocab=self.vocab,
            pad_token=self.pad_token,
            pad_value=self.pad_value,
            append_cls=True,
            include_zero_gene=self.config.include_zero_gene,
        )
        
        print(f"{subset_name}: {tokenized['genes'].shape[0]} samples")
        return tokenized, expr_data, gene_ids
    
    def create_data_dict(self, tokenized, adata, task, mask_ratio=0.0):
        """Create data dictionary"""
        data_dict = {
            "gene_ids": tokenized["genes"],
            "values": tokenized["values"],
            "labels": torch.tensor(adata.obs[f"{task}_id"].values, dtype=torch.long),
        }
        
        if mask_ratio > 0:
            masked = random_mask_value(tokenized["values"], mask_ratio, self.mask_value, self.pad_value)
            data_dict["masked_values"] = masked
            data_dict["input_values"] = tokenized["values"]
        else:
            data_dict["masked_values"] = tokenized["values"]
        
        if "batch_id" in adata.obs:
            batch_ids = adata.obs["batch_id"].astype("category").cat.codes.values
            data_dict["batch_labels"] = torch.tensor(batch_ids, dtype=torch.long)
        
        return data_dict