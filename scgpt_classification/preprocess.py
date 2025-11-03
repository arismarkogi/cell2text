import torch
import gc
import os
import sys
import logging
import json
import argparse
import warnings
from pathlib import Path
import numpy as np
import scanpy as sc
import pandas as pd
from scipy.sparse import issparse
from tqdm import tqdm
# MODIFIED: Import concatenate_datasets and load_from_disk
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

# We need to import the preprocessing and vocab code from your project
sys.path.insert(0, "../")
from scgpt.preprocess import Preprocessor
from scgpt.tokenizer import tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab

warnings.filterwarnings("ignore")

def setup_logger(save_dir):
    logger = logging.getLogger("scGPT_preprocess")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(f"%(asctime)s - %(levelname)s - %(message)s")
    
    fh = logging.FileHandler(save_dir / "preprocessing.log")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def get_all_genes(all_files, logger):
    """Scans all files for the global gene list (low memory)."""
    logger.info("Scanning all files for global gene list...")
    all_genes = set()
    for f in tqdm(all_files, desc="Scanning for genes"):
        try:
            adata_var = sc.read_h5ad(f, backed='r').var_names
            all_genes.update(adata_var.tolist())
        except Exception as e:
            logger.warning(f"Could not read {f} for genes. {e}")
    return sorted(list(all_genes))

def align_adata_to_genes(adata, all_genes_list):
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

def get_label_mapping_from_csv(base_path, task, logger):
    """
    Get label mappings from pre-defined CSV files.
    """
    task_to_file = {
        "cell_type": "final_combined_cell_type_top_values.csv",
        "disease": "final_combined_disease_top_values.csv",
        "tissue": "final_combined_tissue_top_values.csv"
    }
    
    if task not in task_to_file:
        raise ValueError(f"Task '{task}' not in predefined CSV file map.")

    csv_file_path = Path(base_path) / task_to_file[task]
    
    if not csv_file_path.exists():
        raise FileNotFoundError(f"Label file not found: {csv_file_path}")
    
    logger.info(f"Loading consistent label mapping from: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    
    if 'value' not in df.columns:
        raise ValueError(f"'value' column not in {csv_file_path}")
        
    all_labels = sorted(list(set(l for l in df['value'].tolist() if pd.notna(l))))
    
    id_to_label = {i: label for i, label in enumerate(all_labels)}
    label_to_id = {label: i for i, label in enumerate(all_labels)}
    
    logger.info(f"Found {len(all_labels)} unique labels for task '{task}' from CSV.")
    return len(all_labels), id_to_label, label_to_id

def collect_files(data_dir, split_prefixes):
    data_dir = Path(data_dir)
    split_files = {}
    all_files = []
    for prefix in split_prefixes:
        files = list(data_dir.glob(f"{prefix}*/*.h5ad"))
        if not files:
            subdirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)]
            for subdir in subdirs:
                files.extend(list(subdir.glob("*.h5ad")))
        
        if not files:
            raise FileNotFoundError(f"No .h5ad files found for prefix '{prefix}' in {data_dir}")
        
        split_files[prefix] = sorted(files)
        all_files.extend(files)
        print(f"Found {len(files)} {prefix} files")
    return split_files, all_files

# --- MODIFIED FUNCTION ---
# This function now saves each file as a separate dataset part
# instead of returning one giant list.
def process_and_save(files, split_name, preprocessor, all_genes_list, vocab, label_to_id, task, config, logger, split_output_dir):
    """
    Processes a list of .h5ad files and saves them as individual
    Hugging Face Dataset parts in the split_output_dir.
    """
    
    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    pad_token = "<pad>"
    pad_value = -2 if config.input_emb_style != "category" else config.n_bins
    
    # We no longer accumulate in a giant list
    # all_processed_data = []

    logger.info(f"Saving parts to {split_output_dir}")
    split_output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    for f in tqdm(files, desc=f"Processing {split_name} files"):
        try:
            # This list will only hold data for *one* file
            file_processed_data = []
            
            adata = sc.read(f)
            
            adata_aligned = align_adata_to_genes(adata, all_genes_list)
            del adata
            
            preprocessor(adata_aligned, batch_key=None)
            
            if task not in adata_aligned.obs.columns:
                logger.warning(f"Missing '{task}' column in {f}. Skipping.")
                del adata_aligned
                gc.collect()
                continue
            
            adata_aligned.obs[f"{task}_id"] = adata_aligned.obs[task].map(label_to_id)
            
            valid_cells_mask = adata_aligned.obs[f"{task}_id"].notna()
            if not valid_cells_mask.all():
                logger.warning(f"Found {valid_cells_mask.sum()} valid cells out of {len(valid_cells_mask)} in {f}")
                adata_aligned = adata_aligned[valid_cells_mask, :].copy()
                
            if adata_aligned.n_obs == 0:
                logger.warning(f"No valid cells in {f} after label mapping. Skipping.")
                del adata_aligned
                gc.collect()
                continue
            
            input_layer_key = "X_binned" if config.input_style == "binned" else "X_normed"
            expr_data = adata_aligned.layers[input_layer_key].toarray() if issparse(adata_aligned.layers[input_layer_key]) else adata_aligned.layers[input_layer_key]
            genes = adata_aligned.var_names.tolist()
            gene_ids = np.array(vocab(genes), dtype=int)
            
            tokenized = tokenize_and_pad_batch(
                expr_data, gene_ids, max_len=config.max_seq_len,
                vocab=vocab, pad_token=pad_token, pad_value=pad_value,
                append_cls=True, include_zero_gene=config.include_zero_gene,
            )
            
            labels = torch.tensor(adata_aligned.obs[f"{task}_id"].values, dtype=torch.long)
            
            for i in range(adata_aligned.n_obs):
                # Add to the *file-local* list
                file_processed_data.append({
                    "gene_ids": tokenized["genes"][i].numpy(),
                    "values": tokenized["values"][i].numpy(),
                    "labels": labels[i].item(),
                })
            
            # --- NEW: Save this file's data as a separate part ---
            if file_processed_data:
                temp_dataset = Dataset.from_list(file_processed_data)
                part_path = split_output_dir / f.stem
                temp_dataset.save_to_disk(part_path)
                processed_count += 1

            del adata_aligned, tokenized, labels, expr_data, file_processed_data, temp_dataset
            gc.collect()
            
        except Exception as e:
            logger.error(f"Failed to process file {f}: {e}")
            gc.collect()

    # This function no longer returns anything
    logger.info(f"Finished processing split. Saved {processed_count} parts.")
    return
# --- END MODIFIED FUNCTION ---


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["cell_type", "disease", "tissue"])
    parser.add_argument("--data_dir", type=str, default="/home/arism/raw_data")
    parser.add_argument("--load_model", type=str, required=True, help="Path to pretrained model dir (e.g., /home/arism/scgpt_model/scGPT_human)")
    parser.add_argument("--output_dir", type=str, default="./preprocessed_data")
    parser.add_argument("--label_csv_dir", type=str, default="/home/arism/analysis_output/final_combined/", help="Dir containing label CSVs")
    
    # Import config settings from config.py
    from config import Config
    config = Config()
    
    args = parser.parse_args()
    
    # --- Setup ---
    output_dir = Path(args.output_dir) / args.task
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)
    logger.info(f"Starting preprocessing for task: {args.task}")
    logger.info(f"Output directory: {output_dir}")
    
    # --- 1. Collect Files ---
    split_files, all_files = collect_files(args.data_dir, ["train", "val", "test"])
    
    # --- 2. Get Global Gene List ---
    all_genes_list = get_all_genes(all_files, logger)
    logger.info(f"Found {len(all_genes_list)} unique genes across all files.")
    
    # --- 3. Setup Vocab ---
    logger.info("Setting up vocabulary...")
    vocab_path = Path(args.load_model) / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(f"vocab.json not found in {args.load_model}")
    
    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    vocab = GeneVocab.from_file(vocab_path)
    for token in special_tokens:
        if token not in vocab:
            vocab.append_token(token)
    vocab.set_default_index(vocab["<pad>"])
    logger.info(f"Vocabulary loaded from {vocab_path}")
    
    # --- 4. Get Label Mappings ---
    num_classes, id_to_label, label_to_id = get_label_mapping_from_csv(args.label_csv_dir, args.task, logger)
    
    # Save mappings
    mappings = {"id_to_label": id_to_label, "label_to_id": label_to_id}
    with open(output_dir / "label_mappings.json", "w") as f:
        json.dump(mappings, f, indent=2)
    logger.info(f"Label mappings saved to {output_dir / 'label_mappings.json'}")

    # --- 5. Setup Preprocessor ---
    logger.info("Setting up preprocessor...")
    preprocessor = Preprocessor(
        use_key="X", filter_gene_by_counts=False, filter_cell_by_counts=False,
        normalize_total=1e4, result_normed_key="X_normed",
        log1p=True, result_log1p_key="X_log1p",
        subset_hvg=False, binning=config.n_bins, result_binned_key="X_binned",
    )
    
    # --- 6. Process and Save Datasets ---
    logger.info("Starting dataset processing...")
    
    # --- MODIFIED: This section now saves parts, then combines them ---
    hf_dataset_dict = DatasetDict()
    
    # Define a directory to store the temporary parts
    parts_base_dir = output_dir / "hf_dataset_parts"
    
    for split in ["train", "val", "test"]:
        logger.info(f"--- Processing {split} split ---")
        if not split_files[split]:
            logger.warning(f"No files found for {split} split. Skipping.")
            continue
        
        # Define a specific output dir for this split's parts
        split_parts_dir = parts_base_dir / split
        
        # Run processing. This will save parts to disk and return None.
        process_and_save(
            files=split_files[split],
            split_name=split,
            preprocessor=preprocessor,
            all_genes_list=all_genes_list,
            vocab=vocab,
            label_to_id=label_to_id,
            task=args.task,
            config=config,
            logger=logger,
            split_output_dir=split_parts_dir # Pass the new path
        )
        
        # Now, load all the parts from disk and combine them
        logger.info(f"Combining processed parts for {split} split...")
        try:
            part_dirs = [d for d in split_parts_dir.iterdir() if d.is_dir()]
            if not part_dirs:
                logger.error(f"No dataset parts found in {split_parts_dir}. Preprocessing may have failed.")
                continue
            
            datasets_to_combine = [load_from_disk(str(d)) for d in part_dirs]
            
            # Combine all parts into one final dataset for this split
            final_split_dataset = concatenate_datasets(datasets_to_combine)
            hf_dataset_dict[split] = final_split_dataset
            
            logger.info(f"Finished combining {split} split. Total cells: {len(final_split_dataset)}")
        except Exception as e:
            logger.error(f"Failed to combine dataset parts for {split}: {e}")

    # --- 7. Save to Disk ---
    logger.info("Saving final combined datasets to disk...")
    # This saves the *final* combined DatasetDict
    hf_dataset_dict.save_to_disk(output_dir / "hf_dataset")
    logger.info(f"All processed data saved to {output_dir / 'hf_dataset'}")
    logger.info("You can now optionally delete the temporary parts directory:")
    logger.info(f"rm -rf {parts_base_dir}")
    logger.info("--- PREPROCESSING COMPLETE ---")
    # --- END MODIFIED SECTION ---

if __name__ == "__main__":
    main()

