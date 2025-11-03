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
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

sys.path.insert(0, "../")
from scgpt.preprocess import Preprocessor
from scgpt.tokenizer import tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab

warnings.filterwarnings("ignore")

def setup_logger(save_dir):
    logger = logging.getLogger("scGPT_preprocess_pathway")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(f"%(asctime)s - %(levelname)s - %(message)s")
    
    fh = logging.FileHandler(save_dir / "preprocessing_pathway.log")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def get_all_genes(all_files, logger):
    """Scans all files for the global gene list"""
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
    """Aligns a single loaded adata object to the global gene list"""
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

def get_pathway_mapping(base_path, logger):
    """
    Get pathway mappings from separate CSV files for pathway1 and pathway2.
    CSVs should have 'value' column containing pathway names.
    """
    pathway1_csv = Path(base_path) 
    pathway2_csv = Path(base_path) 
    
    if not pathway1_csv.exists():
        raise FileNotFoundError(f"Pathway1 file not found: {pathway1_csv}")
    if not pathway2_csv.exists():
        raise FileNotFoundError(f"Pathway2 file not found: {pathway2_csv}")
    
    logger.info(f"Loading pathway mappings from CSV files...")
    
    # Read both CSVs
    df1 = pd.read_csv(pathway1_csv)
    df2 = pd.read_csv(pathway2_csv)
    
    # Collect all unique pathways from 'value' column
    all_pathways = set()
    
    if 'value' in df1.columns:
        pathways1 = df1['value'].dropna().unique()
        all_pathways.update(pathways1)
        logger.info(f"  Loaded {len(pathways1)} pathways from pathway1 CSV")
    else:
        raise ValueError(f"'value' column not found in {pathway1_csv}")
    
    if 'value' in df2.columns:
        pathways2 = df2['value'].dropna().unique()
        all_pathways.update(pathways2)
        logger.info(f"  Loaded {len(pathways2)} pathways from pathway2 CSV")
    else:
        raise ValueError(f"'value' column not found in {pathway2_csv}")
    
    # Sort for consistent ordering
    all_pathways = sorted(list(all_pathways))
    
    id_to_pathway = {i: pathway for i, pathway in enumerate(all_pathways)}
    pathway_to_id = {pathway: i for i, pathway in enumerate(all_pathways)}
    
    logger.info(f"Found {len(all_pathways)} unique pathways total")
    logger.info(f"  First 5 pathways: {all_pathways[:5]}")
    
    return len(all_pathways), id_to_pathway, pathway_to_id

def collect_files(data_dir, split_prefixes):
    """Collect .h5ad files for each split"""
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

def process_and_save(files, split_name, preprocessor, all_genes_list, vocab, pathway_to_id, 
                     config, logger, split_output_dir):
    """
    Process .h5ad files with multi-label pathway annotations.
    MODIFIED: Uses Document 2's memory-efficient pattern throughout
    """
    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    pad_token = "<pad>"
    pad_value = -2 if config.input_emb_style != "category" else config.n_bins
    
    logger.info(f"Saving parts to {split_output_dir}")
    split_output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    
    for f in tqdm(files, desc=f"Processing {split_name} files"):
        try:
            file_processed_data = []
            
            adata = sc.read(f)
            adata_aligned = align_adata_to_genes(adata, all_genes_list)
            del adata
            
            preprocessor(adata_aligned, batch_key=None)
            
            # Check for pathway columns
            if 'pathway1' not in adata_aligned.obs.columns or 'pathway2' not in adata_aligned.obs.columns:
                logger.warning(f"Missing pathway columns in {f}. Skipping.")
                del adata_aligned
                gc.collect()
                continue
            
            # EFFICIENT: Map pathways to IDs using pandas (no loop, vectorized operation)
            # Convert to object dtype first to avoid Categorical issues
            pathway1_series = adata_aligned.obs['pathway1'].astype(str).map(pathway_to_id)
            pathway2_series = adata_aligned.obs['pathway2'].astype(str).map(pathway_to_id)
            
            # EFFICIENT: Filter immediately to reduce memory footprint
            valid_cells_mask = (pathway1_series.notna() | pathway2_series.notna())
            
            if not valid_cells_mask.all():
                logger.warning(f"Found {valid_cells_mask.sum()} valid cells out of {len(valid_cells_mask)} in {f}")
                adata_aligned = adata_aligned[valid_cells_mask, :].copy()
                pathway1_series = pathway1_series[valid_cells_mask]
                pathway2_series = pathway2_series[valid_cells_mask]
            
            if adata_aligned.n_obs == 0:
                logger.warning(f"No valid cells in {f} after pathway mapping. Skipping.")
                del adata_aligned
                gc.collect()
                continue
            
            # EFFICIENT: Convert to dense only once, after filtering
            input_layer_key = "X_binned" if config.input_style == "binned" else "X_normed"
            expr_data = adata_aligned.layers[input_layer_key].toarray() if issparse(adata_aligned.layers[input_layer_key]) else adata_aligned.layers[input_layer_key]
            
            genes = adata_aligned.var_names.tolist()
            gene_ids = np.array(vocab(genes), dtype=int)
            
            tokenized = tokenize_and_pad_batch(
                expr_data, gene_ids, max_len=config.max_seq_len,
                vocab=vocab, pad_token=pad_token, pad_value=pad_value,
                append_cls=True, include_zero_gene=config.include_zero_gene,
            )
            
            # EFFICIENT: Extract pathway IDs as numpy arrays (already filtered)
            # Replace NaN with -1 for missing pathways
            pathway1_ids = pathway1_series.fillna(-1).astype(int).values
            pathway2_ids = pathway2_series.fillna(-1).astype(int).values
            
            # EFFICIENT: Build the list with pre-extracted arrays (no pd.notna checks in loop)
            for i in range(adata_aligned.n_obs):
                file_processed_data.append({
                    "gene_ids": tokenized["genes"][i].numpy(),
                    "values": tokenized["values"][i].numpy(),
                    "pathway1_id": int(pathway1_ids[i]),  # Already cleaned
                    "pathway2_id": int(pathway2_ids[i]),  # Already cleaned
                })
            
            # Save this file's data as a separate part
            if file_processed_data:
                temp_dataset = Dataset.from_list(file_processed_data)
                part_path = split_output_dir / f.stem
                temp_dataset.save_to_disk(part_path)
                processed_count += 1

            del adata_aligned, tokenized, expr_data, file_processed_data, temp_dataset, pathway1_ids, pathway2_ids
            gc.collect()
            
        except Exception as e:
            logger.error(f"Failed to process file {f}: {e}")
            gc.collect()

    logger.info(f"Finished processing split. Saved {processed_count} parts.")
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with .h5ad files")
    parser.add_argument("--load_model", type=str, required=True, help="Path to pretrained model dir")
    parser.add_argument("--output_dir", type=str, default="./preprocessed_data_pathway")
    parser.add_argument("--pathway_csv_dir", type=str, required=True, 
                        help="Dir containing final_combined_pathway1_top_values.csv and final_combined_pathway2_top_values.csv")
    
    from config import Config
    config = Config()
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)
    logger.info("Starting pathway preprocessing (MEMORY OPTIMIZED)")
    logger.info(f"Output directory: {output_dir}")
    
    # Collect files
    split_files, all_files = collect_files(args.data_dir, ["train", "val", "test"])
    
    # Get global gene list
    all_genes_list = get_all_genes(all_files, logger)
    logger.info(f"Found {len(all_genes_list)} unique genes across all files.")
    
    # Setup vocabulary
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
    
    # Get pathway mappings
    num_pathways, id_to_pathway, pathway_to_id = get_pathway_mapping(args.pathway_csv_dir, logger)
    
    # Save mappings
    mappings = {"id_to_pathway": id_to_pathway, "pathway_to_id": pathway_to_id}
    with open(output_dir / "pathway_mappings.json", "w") as f:
        json.dump(mappings, f, indent=2)
    logger.info(f"Pathway mappings saved to {output_dir / 'pathway_mappings.json'}")

    # Setup preprocessor
    logger.info("Setting up preprocessor...")
    preprocessor = Preprocessor(
        use_key="X", filter_gene_by_counts=False, filter_cell_by_counts=False,
        normalize_total=1e4, result_normed_key="X_normed",
        log1p=True, result_log1p_key="X_log1p",
        subset_hvg=False, binning=config.n_bins, result_binned_key="X_binned",
    )
    
    # Process and save datasets
    logger.info("Starting dataset processing...")
    
    hf_dataset_dict = DatasetDict()
    parts_base_dir = output_dir / "hf_dataset_parts"
    
    for split in ["train", "val", "test"]:
        logger.info(f"--- Processing {split} split ---")
        if not split_files[split]:
            logger.warning(f"No files found for {split} split. Skipping.")
            continue
        
        split_parts_dir = parts_base_dir / split
        
        process_and_save(
            files=split_files[split],
            split_name=split,
            preprocessor=preprocessor,
            all_genes_list=all_genes_list,
            vocab=vocab,
            pathway_to_id=pathway_to_id,
            config=config,
            logger=logger,
            split_output_dir=split_parts_dir
        )
        
        logger.info(f"Combining processed parts for {split} split...")
        try:
            part_dirs = [d for d in split_parts_dir.iterdir() if d.is_dir()]
            if not part_dirs:
                logger.error(f"No dataset parts found in {split_parts_dir}.")
                continue
            
            datasets_to_combine = [load_from_disk(str(d)) for d in part_dirs]
            final_split_dataset = concatenate_datasets(datasets_to_combine)
            hf_dataset_dict[split] = final_split_dataset
            
            logger.info(f"Finished combining {split} split. Total cells: {len(final_split_dataset)}")
        except Exception as e:
            logger.error(f"Failed to combine dataset parts for {split}: {e}")

    logger.info("Saving final combined datasets to disk...")
    hf_dataset_dict.save_to_disk(output_dir / "hf_dataset")
    logger.info(f"All processed data saved to {output_dir / 'hf_dataset'}")
    logger.info("--- PREPROCESSING COMPLETE ---")

if __name__ == "__main__":
    main()