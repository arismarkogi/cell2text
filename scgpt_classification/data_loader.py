import pandas as pd
import scanpy as sc
import torch
import gc
import sys
import numpy as np
from pathlib import Path
from scipy.sparse import issparse
from torch.utils.data import Dataset, DataLoader, DistributedSampler

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
        
        # ADDED: Debug tracking
        self._debug_label_checks = []
    
    def get_n_obs(self, file_path: Path) -> int:
        """Reads the number of observations (cells) from an h5ad file without loading it."""
        try:
            return sc.read_h5ad(file_path, backed='r').n_obs
        except Exception as e:
            print(f"Warning: Could not read n_obs from {file_path}. {e}")
            return 0

    def get_all_genes(self, all_files):
        """Scans all files for the global gene list (low memory)."""
        print("Scanning all files for gene list...")
        all_genes = set()
        for f in all_files:
            try:
                adata_var = sc.read_h5ad(f, backed='r').var_names
                all_genes.update(adata_var.tolist())
            except Exception as e:
                print(f"Warning: Could not read {f} for genes. {e}")
        return sorted(list(all_genes))

    def align_adata_to_genes(self, adata, all_genes_list):
        """Aligns a single loaded adata object to the global gene list."""
        if adata.var_names.has_duplicates:
            adata.var_names_make_unique()
        
        current_genes = set(adata.var_names)
        missing_genes = set(all_genes_list) - current_genes
        
        if missing_genes:
            from scipy import sparse
            n_cells = adata.shape[0]
            missing_data = sparse.csr_matrix((n_cells, len(missing_genes)), dtype=adata.X.dtype)
            
            new_var = pd.DataFrame(index=list(missing_genes))
            new_adata = sc.AnnData(X=missing_data, obs=adata.obs, var=new_var)
            
            adata = sc.concat([adata, new_adata], axis=1, join='outer', index_unique=None, fill_value=0)
        
        return adata[:, all_genes_list].copy()

    def load_data_from_files(self, train_files, val_files, test_files):
        """Scans all files and returns file paths for chunked processing."""
        print("Scanning data files...")
        all_files_flat = train_files + val_files + test_files
        self.all_genes_list = self.get_all_genes(all_files_flat)
        print(f"Found {len(self.all_genes_list)} unique genes")

        print("Loading sample adata for vocab setup...")
        sample_adata = sc.read(train_files[0])
        sample_adata_aligned = self.align_adata_to_genes(sample_adata, self.all_genes_list)
        
        self.train_files = train_files
        self.val_files = val_files
        self.test_files = test_files
        
        # Build train_batches: list of (file_path, cell_indices_range) tuples used for chunked training
        self.train_batches = []
        chunk_size = getattr(self.config, "cell_chunk_size", None) or 1000
        for f in self.train_files:
            n_cells = self.get_n_obs(f)
            if n_cells == 0:
                # skip files we couldn't read metadata for
                continue
            for i in range(0, n_cells, chunk_size):
                cell_range = range(i, min(i + chunk_size, n_cells))
                self.train_batches.append((f, cell_range))
        print(f"Prepared {len(self.train_batches)} training chunks (chunk_size={chunk_size})")
        
        return sample_adata_aligned, train_files, val_files, test_files
    
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
    
    def preprocess_data(self, adata, is_raw=True):
        """Preprocess single adata object"""
        if self.preprocessor is None:
            self.preprocessor = Preprocessor(
                use_key="X", filter_gene_by_counts=False, filter_cell_by_counts=False,
                normalize_total=1e4, result_normed_key="X_normed",
                log1p=is_raw, result_log1p_key="X_log1p",
                subset_hvg=False, binning=self.config.n_bins, result_binned_key="X_binned",
            )
        self.preprocessor(adata, batch_key=None)
        return adata
    
    def get_label_mapping(self, task):
        """
        Get label mappings from pre-defined CSV files.
        This ensures a consistent label mapping across all chunks and runs.
        """
        # Base path provided by the user
        base_path = "/home/arism/analysis_output/final_combined/"
        
        # Map task name to the specific CSV file
        task_to_file = {
            "cell_type": "final_combined_cell_type_top_values.csv",
            "disease": "final_combined_disease_top_values.csv",
            "tissue": "final_combined_tissue_top_values.csv"
        }
        
        if task not in task_to_file:
            print(f"Warning: No pre-defined label CSV for task '{task}'. "
                  "Falling back to scanning H5AD files. This may be slow or inconsistent.")
            # Call the original function (which we will rename)
            return self.get_label_mapping_from_scan(task)

        csv_file_path = Path(base_path) / task_to_file[task]
        
        if not csv_file_path.exists():
            print(f"ERROR: Label file not found: {csv_file_path}")
            print("Falling back to scanning H5AD files.")
            return self.get_label_mapping_from_scan(task)
        
        print(f"Loading consistent label mapping from: {csv_file_path}")
        try:
            df = pd.read_csv(csv_file_path)
            
            if 'value' not in df.columns:
                raise ValueError(f"'value' column not in {csv_file_path}")
                
            # Get all labels from the 'value' column
            all_labels = df['value'].tolist()
            
            # Ensure they are unique and sorted
            all_labels = sorted(list(set([l for l in all_labels if pd.notna(l)])))
            
            id_to_label = {i: label for i, label in enumerate(all_labels)}
            label_to_id = {label: i for i, label in enumerate(all_labels)}
            
            print(f"Found {len(all_labels)} unique labels for task '{task}' from CSV.")
            print(f"Label to ID mapping (first 5): {dict(list(label_to_id.items())[:5])}")
            
            return len(all_labels), id_to_label, label_to_id
        
        except Exception as e:
            print(f"ERROR: Could not read labels from {csv_file_path}. {e}")
            print("Falling back to scanning H5AD files.")
            return self.get_label_mapping_from_scan(task)
    
    def process_and_create_loader(self, file_path, cell_indices, task, label_to_id, batch_size, 
                                    shuffle=False, rank=0, world_size=1, debug=False):
        """Process a slice of cells from a single file and create a DDP-aware dataloader"""
        adata_full = sc.read(file_path)
        batch = adata_full[cell_indices, :].copy()
        del adata_full

        batch_aligned = self.align_adata_to_genes(batch, self.all_genes_list)
        del batch
        
        batch_processed = self.preprocess_data(batch_aligned, is_raw=True)
        del batch_aligned
        
        if task not in batch_processed.obs.columns:
            raise ValueError(f"Missing '{task}' column in {file_path}")
        
        # CRITICAL: Check for missing labels BEFORE mapping
        original_labels = batch_processed.obs[task].values
        unique_original = np.unique(original_labels)
        
        # Check if any labels are missing from label_to_id
        missing_labels = [l for l in unique_original if pd.notna(l) and l not in label_to_id]
        if missing_labels:
            print(f"ERROR: Found labels not in label_to_id mapping: {missing_labels}")
            print(f"Available labels in mapping: {sorted(label_to_id.keys())}")
            raise ValueError(f"Label mismatch! Found unknown labels: {missing_labels}")
        
        # Map labels to IDs
        batch_processed.obs[f"{task}_id"] = batch_processed.obs[task].map(label_to_id)
        
        # CRITICAL: Check for NaN values after mapping (indicates unmapped labels)
        if batch_processed.obs[f"{task}_id"].isna().any():
            nan_labels = batch_processed.obs[task][batch_processed.obs[f"{task}_id"].isna()].unique()
            print(f"ERROR: Found NaN after label mapping! Original labels: {nan_labels}")
            raise ValueError(f"Label mapping failed for: {nan_labels}")
        
        # DEBUG: Track label statistics
        label_ids = batch_processed.obs[f"{task}_id"].values
        unique_ids = np.unique(label_ids[~pd.isna(label_ids)])
        
        if debug or rank == 0:
            print(f"\n[DEBUG] File: {file_path.name}")
            print(f"  Cells in chunk: {len(cell_indices)}")
            print(f"  Unique original labels: {unique_original.tolist()}")
            print(f"  Unique label IDs: {unique_ids.tolist()}")
            print(f"  Expected ID range: 0 to {len(label_to_id)-1}")
            print(f"  Label ID stats - min: {unique_ids.min()}, max: {unique_ids.max()}")
            
            # Check label distribution
            label_counts = pd.Series(label_ids).value_counts().sort_index()
            print(f"  Label distribution:\n{label_counts}")
        
        tokenized, _, _ = self.tokenize_data(batch_processed)
        data_dict = self.create_data_dict(tokenized, batch_processed, task, 
                                          self.config.mask_ratio if shuffle else 0.0)
        
        # FINAL CHECK: Verify labels in data_dict
        final_labels = data_dict["labels"]
        if torch.isnan(final_labels).any() or torch.isinf(final_labels).any():
            print(f"ERROR: Invalid labels in data_dict!")
            raise ValueError("Data dict contains NaN or Inf labels!")
        
        if final_labels.min() < 0 or final_labels.max() >= len(label_to_id):
            print(f"ERROR: Label IDs out of range! Min: {final_labels.min()}, Max: {final_labels.max()}")
            print(f"Expected range: 0 to {len(label_to_id)-1}")
            raise ValueError("Label IDs out of expected range!")
        
        dataset = SeqDataset(data_dict)
        
        sampler = None
        if world_size > 1:
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
            
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(shuffle and sampler is None), 
            sampler=sampler, num_workers=4, pin_memory=True
        )
        
        del batch_processed, tokenized, data_dict, dataset
        gc.collect()
        
        return loader
    
    def tokenize_data(self, adata):
        """Tokenize data"""
        input_layer_key = "X_binned" if self.config.input_style == "binned" else "X_normed"
        
        expr_data = adata.layers[input_layer_key].toarray() if issparse(adata.layers[input_layer_key]) else adata.layers[input_layer_key]
        genes = adata.var_names.tolist()
        gene_ids = np.array(self.vocab(genes), dtype=int)
        
        tokenized = tokenize_and_pad_batch(
            expr_data, gene_ids, max_len=self.config.max_seq_len,
            vocab=self.vocab, pad_token=self.pad_token, pad_value=self.pad_value,
            append_cls=True, include_zero_gene=self.config.include_zero_gene,
        )
        return tokenized, expr_data, gene_ids
    
    def create_data_dict(self, tokenized, adata, task, mask_ratio=0.0):
        """Create data dictionary for the dataloader"""
        # CRITICAL FIX: Convert to long tensor and validate
        label_values = adata.obs[f"{task}_id"].values
        
        # Remove any NaN values (shouldn't happen if previous checks passed)
        if pd.isna(label_values).any():
            print(f"WARNING: Found NaN in labels before tensor conversion!")
            # Option 1: Raise error
            raise ValueError("Cannot create data dict with NaN labels")
            # Option 2: Filter them out (not recommended)
            # valid_mask = ~pd.isna(label_values)
            # label_values = label_values[valid_mask]
        
        data_dict = {
            "gene_ids": tokenized["genes"],
            "values": tokenized["values"],
            "labels": torch.tensor(label_values, dtype=torch.long),
        }
        
        if mask_ratio > 0:
            masked = random_mask_value(tokenized["values"], mask_ratio, self.mask_value, self.pad_value)
            data_dict["masked_values"] = masked
            data_dict["input_values"] = tokenized["values"]
        else:
            data_dict["masked_values"] = tokenized["values"]
        
        return data_dict
    
    def verify_label_consistency(self, loader, label_to_id, rank=0):
        """Helper function to verify label consistency in a loader"""
        if rank != 0:
            return
        
        print("\n[LABEL CONSISTENCY CHECK]")
        all_labels = []
        for batch_idx, batch in enumerate(loader):
            labels = batch["labels"].numpy()
            all_labels.extend(labels.tolist())
            if batch_idx >= 2:  # Check first 3 batches
                break
        
        unique_labels = np.unique(all_labels)
        print(f"  Labels in first batches: {unique_labels}")
        print(f"  Expected range: 0 to {len(label_to_id)-1}")
        print(f"  Valid: {unique_labels.min() >= 0 and unique_labels.max() < len(label_to_id)}")
        print()