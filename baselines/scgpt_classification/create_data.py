import scanpy as sc
import pandas as pd
import os
from pathlib import Path

def load_gene_mapping(csv_path):
    """Load gene mapping from CSV file."""
    print(f"Loading gene mapping from {csv_path}...")
    mapping_df = pd.read_csv(csv_path)
    print(f"CSV columns: {mapping_df.columns.tolist()}")
    print(f"First 5 rows of mapping:")
    print(mapping_df.head())
    
    # Create mapping dictionary: feature_id (Ensembl ID) -> feature_name (gene name)
    gene_mapping = dict(zip(mapping_df['feature_id'], mapping_df['feature_name']))
    print(f"Loaded {len(gene_mapping)} gene mappings")
    print(f"First 5 mappings: {dict(list(gene_mapping.items())[:5])}")
    return gene_mapping

def map_gene_names(adata, gene_mapping):
    """Map Ensembl IDs to gene names."""
    print("Mapping Ensembl IDs to gene names...")
    print(f"First 5 var index values: {adata.var.index[:5].tolist()}")
    print(f"Var index dtype: {adata.var.index.dtype}")
    
    # Map the var index (Ensembl IDs) to gene names
    adata.var['gene_name'] = adata.var.index.map(gene_mapping)
    
    # Count successful mappings
    successful = adata.var['gene_name'].notna().sum()
    print(f"Successfully mapped {successful} out of {len(adata.var)} genes")
    
    if successful == 0:
        print("No mappings found! Keeping original index...")
        adata.var = adata.var.drop(columns=['gene_name'])
        return adata
    
    # Handle duplicates and NaNs
    gene_names = adata.var['gene_name'].fillna('UNKNOWN')
    
    # Make gene names unique
    gene_names = pd.Series(gene_names).astype(str)
    duplicated_mask = gene_names.duplicated(keep=False)
    if duplicated_mask.any():
        print(f"Found {duplicated_mask.sum()} duplicate gene names, making unique...")
        for i, is_dup in enumerate(duplicated_mask):
            if is_dup:
                gene_names.iloc[i] = f"{gene_names.iloc[i]}_{i}"
    
    # Set gene names as new index
    adata.var.index = gene_names
    adata.var = adata.var.drop(columns=['gene_name'])
    
    return adata

import scanpy as sc
import pandas as pd
import numpy as np
from collections import Counter

def debug_duplicates(adata, dataset_name=""):
    """Debug and print duplicate gene names."""
    print(f"\n=== Debugging duplicates for {dataset_name} ===")
    var_names = adata.var.index.tolist()
    
    # Count occurrences
    name_counts = Counter(var_names)
    duplicates = {name: count for name, count in name_counts.items() if count > 1}
    
    if duplicates:
        print(f"Found {len(duplicates)} duplicate gene names:")
        for name, count in sorted(duplicates.items()):
            print(f"  '{name}': appears {count} times")
            # Show indices where this name appears
            indices = [i for i, x in enumerate(var_names) if x == name]
            print(f"    at positions: {indices}")
    else:
        print("No duplicates found!")
    
    return duplicates

def sum_duplicate_genes(adata, dataset_name=""):
    """Sum expression values for genes with duplicate names."""
    print(f"\n=== Processing duplicates for {dataset_name} ===")
    
    # First debug what duplicates we have
    duplicates = debug_duplicates(adata, dataset_name)
    
    if not duplicates:
        print("No duplicates to process")
        return adata
    
    print(f"Original shape: {adata.shape}")
    
    # Create a DataFrame from the expression matrix
    # adata.X might be sparse, so convert to dense if needed
    if hasattr(adata.X, 'toarray'):
        expr_df = pd.DataFrame(adata.X.toarray(), 
                              columns=adata.var.index, 
                              index=adata.obs.index)
    else:
        expr_df = pd.DataFrame(adata.X, 
                              columns=adata.var.index, 
                              index=adata.obs.index)
    
    # Group by gene name and sum
    print("Summing duplicate genes...")
    expr_summed = expr_df.groupby(expr_df.columns, axis=1).sum()
    
    print(f"After summing duplicates: {expr_summed.shape}")
    print(f"Reduced from {len(adata.var)} to {len(expr_summed.columns)} genes")
    
    # Create new AnnData object
    new_adata = sc.AnnData(X=expr_summed.values)
    new_adata.obs = adata.obs.copy()
    
    # Create new var DataFrame
    unique_genes = expr_summed.columns.tolist()
    new_var = pd.DataFrame(index=unique_genes)
    
    # Try to preserve some var information by taking the first occurrence
    if not adata.var.empty and len(adata.var.columns) > 0:
        for col in adata.var.columns:
            new_var[col] = [adata.var.loc[adata.var.index == gene, col].iloc[0] 
                           if gene in adata.var.index else np.nan 
                           for gene in unique_genes]
    
    new_adata.var = new_var
    
    # Verify no duplicates remain
    remaining_duplicates = debug_duplicates(new_adata, f"{dataset_name}_processed")
    
    return new_adata

def map_gene_names_with_sum(adata, gene_mapping, dataset_name=""):
    """Map Ensembl IDs to gene names and sum duplicates."""
    print(f"Mapping Ensembl IDs to gene names for {dataset_name}...")
    print(f"First 5 var index values: {adata.var.index[:5].tolist()}")
    print(f"Var index dtype: {adata.var.index.dtype}")
    
    # Debug original state
    debug_duplicates(adata, f"{dataset_name}_original")
    
    # Map the var index (Ensembl IDs) to gene names
    adata.var['gene_name'] = adata.var.index.map(gene_mapping)
    
    # Count successful mappings
    successful = adata.var['gene_name'].notna().sum()
    print(f"Successfully mapped {successful} out of {len(adata.var)} genes")
    
    if successful == 0:
        print("No mappings found! Keeping original index...")
        adata.var = adata.var.drop(columns=['gene_name'])
        return adata
    
    # Handle unmapped genes
    gene_names = adata.var['gene_name'].fillna('UNKNOWN').astype(str)
    
    # Set gene names as index temporarily
    adata.var.index = gene_names
    adata.var = adata.var.drop(columns=['gene_name'])
    
    # Debug after mapping
    debug_duplicates(adata, f"{dataset_name}_after_mapping")
    
    # Sum duplicates
    adata = sum_duplicate_genes(adata, dataset_name)
    
    return adata

def merge_datasets(data_dir, split_name, gene_mapping):
    """Merge all h5ad files for a given split."""
    split_path = Path(data_dir) / split_name
    h5ad_files = list(split_path.glob("*.h5ad"))
    
    if not h5ad_files:
        print(f"No h5ad files found in {split_path}")
        return None
    
    print(f"\nProcessing {split_name} split with {len(h5ad_files)} files...")
    
    # Load and process first file
    first_file = h5ad_files[0]
    print(f"Loading {first_file.name}...")
    adata = sc.read_h5ad(first_file)
    adata = map_gene_names_with_sum(adata, gene_mapping, f"{split_name}_{first_file.stem}")
    
    # Concatenate with remaining files
    for h5ad_file in h5ad_files[1:]:
        print(f"Loading and merging {h5ad_file.name}...")
        temp_adata = sc.read_h5ad(h5ad_file)
        temp_adata = map_gene_names_with_sum(temp_adata, gene_mapping, f"{split_name}_{h5ad_file.stem}")
        
        # Debug before concatenation
        print(f"\nBefore concatenation:")
        print(f"  Main dataset genes: {len(adata.var)}")
        print(f"  New dataset genes: {len(temp_adata.var)}")
        
        debug_duplicates(adata, "main_before_concat")
        debug_duplicates(temp_adata, "temp_before_concat")
        
        # Concatenate
        try:
            adata = sc.concat([adata, temp_adata], axis=0, join='outer')
            print(f"Concatenation successful. New shape: {adata.shape}")
        except Exception as e:
            print(f"Concatenation failed: {e}")
            # Debug the combined gene names that would be created
            combined_genes = list(set(adata.var.index.tolist() + temp_adata.var.index.tolist()))
            print(f"Combined unique genes would be: {len(combined_genes)}")
            
            # Check for duplicates in the union
            all_genes = adata.var.index.tolist() + temp_adata.var.index.tolist()
            duplicate_check = Counter(all_genes)
            cross_duplicates = {name: count for name, count in duplicate_check.items() if count > 1}
            if cross_duplicates:
                print(f"Cross-dataset duplicates found: {list(cross_duplicates.keys())[:10]}...")
            
            raise e
    
    print(f"Final {split_name} shape: {adata.shape}")
    return adata

def inspect_h5ad_file(filepath):
    """Inspect the structure of an h5ad file."""
    print(f"\n--- Inspecting {filepath} ---")
    adata = sc.read_h5ad(filepath)
    print(f"Shape: {adata.shape}")
    print(f"Var index dtype: {adata.var.index.dtype}")
    print(f"First 10 var index values: {adata.var.index[:10].tolist()}")
    print(f"Var columns: {adata.var.columns.tolist()}")
    if not adata.var.empty:
        print(f"First few var rows:")
        print(adata.var.head())
    return adata

def main():
    # Paths
    data_dir = "/home/arism/raw_data/"  
    gene_csv = "/home/arism/cell2text/scgpt_classification/gene_info.csv"
    
    # First, let's inspect one file to understand the structure
    train_folders = [d for d in os.listdir(data_dir) if d.startswith('train_')]
    if train_folders:
        sample_folder = sorted(train_folders)[0]
        sample_path = Path(data_dir) / sample_folder
        h5ad_files = list(sample_path.glob("*.h5ad"))
        if h5ad_files:
            inspect_h5ad_file(h5ad_files[0])
    
    # Load gene mapping
    gene_mapping = load_gene_mapping(gene_csv)
    
    # Process each split
    splits = []
    
    # Find all train folders
    train_folders = [d for d in os.listdir(data_dir) if d.startswith('train_')]
    if train_folders:
        print(f"Found train folders: {train_folders}")
        train_data = []
        for folder in sorted(train_folders):
            split_data = merge_datasets(data_dir, folder, gene_mapping)
            if split_data is not None:
                train_data.append(split_data)
        
        if train_data:
            print("\n=== MERGING ALL TRAIN DATASETS ===")
            print("Merging all train datasets...")
            
            # Debug each dataset before final merge
            for i, data in enumerate(train_data):
                debug_duplicates(data, f"train_dataset_{i}")
            
            # Final concatenation
            train_adata = sc.concat(train_data, axis=0, join='outer')
            train_adata.write("/home/arism/raw_data/train.h5ad")
            print(f"Saved train.h5ad with shape: {train_adata.shape}")
    
    # Similar updates for val and test folders...
    # Find all val folders
    val_folders = [d for d in os.listdir(data_dir) if d.startswith('val_')]
    if val_folders:
        print(f"Found val folders: {val_folders}")
        val_data = []
        for folder in sorted(val_folders):
            split_data = merge_datasets(data_dir, folder, gene_mapping)
            if split_data is not None:
                val_data.append(split_data)
        
        if val_data:
            print("\n=== MERGING ALL VAL DATASETS ===")
            print("Merging all val datasets...")
            val_adata = sc.concat(val_data, axis=0, join='outer')
            val_adata.write("/home/arism/raw_data/val.h5ad")
            print(f"Saved val.h5ad with shape: {val_adata.shape}")
    
    # Find all test folders
    test_folders = [d for d in os.listdir(data_dir) if d.startswith('test_')]
    if test_folders:
        print(f"Found test folders: {test_folders}")
        test_data = []
        for folder in sorted(test_folders):
            split_data = merge_datasets(data_dir, folder, gene_mapping)
            if split_data is not None:
                test_data.append(split_data)
        
        if test_data:
            print("\n=== MERGING ALL TEST DATASETS ===")
            print("Merging all test datasets...")
            test_adata = sc.concat(test_data, axis=0, join='outer')
            test_adata.write("/home/arism/raw_data/test.h5ad")
            print(f"Saved test.h5ad with shape: {test_adata.shape}")
    
    print("\nAll merging complete!")

    
if __name__ == "__main__":
    main()