import resource
import os
import re
import gc
import sys
import time
import json
import torch
import h5py
import argparse
import random
import requests
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import mygene
import shutil
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional
from bs4 import BeautifulSoup
from datasets import Dataset, load_from_disk
from anndata import AnnData
import anndata as ad

from scipy.sparse import csr_matrix, vstack
import scipy.sparse as sparse

from pyscenic.aucell import GeneSignature, create_rankings, enrichment
from geneformer import TranscriptomeTokenizer
import cellxgene_census

ROOT = Path(__file__).resolve().parent.parent
# ROOT = Path("/datadisk3/cell2text")


DATA_DIR = ROOT / 'data_preprocess'

# argParser = argparse.ArgumentParser()
# argParser.add_argument("-N", type=int, required=True, help="Sample size targeted")
# argParser.add_argument("-BS", type=int, required=True, help="Batch size")
# argParser.add_argument("-CS", type=int, required=True, help="Chunck size")
# args = argParser.parse_args()

# target_total = args.N

# print("--- Starting Data Pre-processing and Sampling ---")
# print("Connecting to CELLxGENE census...")

# # Connect to CELLxGENE API
# census = cellxgene_census.open_soma(census_version="2025-01-30")
# print("Connection successful.")

# def map_dev_stage(stage: str) -> str:
#     """
#     Map developmental stage to standardized categories.
    
#     Categories:
#     - prenatal: embryonic/fetal development
#     - infant: 0-2 years
#     - child: 2-15 years  
#     - young_adult: 15-30 years
#     - adult: 30-60 years
#     - aged: 60+ years
#     - unknown: cannot determine
#     """
    
#     if not isinstance(stage, str) or not stage.strip():
#         return "unknown"
    
#     stage = stage.lower().strip()
    
#     # Handle empty or unknown cases
#     if stage in ['unknown', '', 'na', 'n/a', 'null']:
#         return "unknown"
    
#     # Prenatal/Embryonic stages
#     prenatal_patterns = [
#         r'blastula', r'gastrula', r'embryonic', r'organogenesis',
#         r'post-fertilization', r'carnegie stage', r'lmp month',
#         r'fetal', r'prenatal', r'gestation'
#     ]
    
#     if any(re.search(pattern, stage) for pattern in prenatal_patterns):
#         return "prenatal"
    
#     # Extract numeric age with units
#     # Handle formats like "5-year-old", "3 years", "12 months", "8 weeks"
#     age_match = re.search(r'(\d+)[-\s]*(year|month|week|day)s?[-\s]*old|\b(\d+)[-\s]*(year|month|week|day)s?\b', stage)
    
#     if age_match:
#         # Get the numeric value and unit
#         value = int(age_match.group(1) or age_match.group(3))
#         unit = (age_match.group(2) or age_match.group(4)).lower()
        
#         # Convert everything to years for easier comparison
#         if unit == "day":
#             age_years = value / 365.25
#         elif unit == "week":
#             age_years = value / 52.18
#         elif unit == "month":
#             age_years = value / 12
#         elif unit == "year":
#             age_years = value
#         else:
#             age_years = None
            
#         if age_years is not None:
#             if age_years < 2:
#                 return "infant"
#             elif age_years < 15:
#                 return "child"
#             elif age_years < 30:
#                 return "young_adult"
#             elif age_years < 60:
#                 return "adult"
#             else:
#                 return "aged"
    
#     # Handle decade-based descriptions
#     decade_mappings = {
#         r'first decade|0-10': 'child',
#         r'second decade|10-20': 'young_adult',
#         r'third decade|20-30': 'young_adult',
#         r'fourth decade|30-40': 'adult',
#         r'fifth decade|40-50': 'adult',
#         r'sixth decade|50-60': 'adult',
#         r'seventh decade|60-70': 'aged',
#         r'eighth decade|70-80': 'aged',
#         r'ninth decade|80-90': 'aged'
#     }
    
#     for pattern, category in decade_mappings.items():
#         if re.search(pattern, stage):
#             return category
    
#     # Handle stage-based descriptions
#     stage_mappings = {
#         # Infant stages
#         r'newborn|neonatal|infant(?!ile)': 'infant',
        
#         # Child stages
#         r'child(?!birth)|juvenile|pediatric|toddler': 'child',
#         r'adolescent|teenage|teen': 'child',  # Late childhood/adolescence
        
#         # Adult stages
#         r'young adult': 'young_adult',
#         r'prime adult|adult(?!escence)': 'adult',
#         r'middle.?aged': 'adult',
        
#         # Elderly stages
#         r'elderly|aged|senior|geriatric': 'aged',
#         r'late adult': 'aged',
#         r'very elderly|advanced age': 'aged'
#     }
    
#     for pattern, category in stage_mappings.items():
#         if re.search(pattern, stage):
#             return category
    
#     # Handle specific age ranges (e.g., "60-79 year-old", "80+")
#     range_match = re.search(r'(\d+)[-–](\d+)|(\d+)\+', stage)
#     if range_match:
#         if range_match.group(3):  # "80+" format
#             start_age = int(range_match.group(3))
#         else:  # "60-79" format
#             start_age = int(range_match.group(1))
            
#         if start_age < 2:
#             return "infant"
#         elif start_age < 15:
#             return "child"
#         elif start_age < 30:
#             return "young_adult"
#         elif start_age < 60:
#             return "adult"
#         else:
#             return "aged"
    
#     # Handle postnatal (general term for after birth)
#     if 'postnatal' in stage:
#         return "infant"  # Default to infant for general postnatal
    
#     return "unknown"


# # Get only the metadata of cell from the database to
# obs_df = cellxgene_census.get_obs(census, "homo_sapiens", column_names=["soma_joinid", "development_stage", "disease", "assay","dataset_id", "donor_id", "sex", "tissue", "tissue_general", "cell_type", "is_primary_data"])

# # Apply binning at development stage
# obs_df["dev_stage_group"] = obs_df["development_stage"].apply(map_dev_stage)

# filtered = obs_df[
#     obs_df["tissue_general"].notna() &
#     obs_df["tissue_general"].ne("unknown") &
#     obs_df["disease"].notna() &
#     obs_df["disease"].ne("unknown") &
#     obs_df["cell_type"].ne("unknown") &
#     obs_df["cell_type"].notna() &
#     obs_df["is_primary_data"] == True
# ]

# del obs_df

# excluded_dataset_ids = [
#     "1b9d8702-5af8-4142-85ed-020eb06ec4f6", # Dominguez
#     "fe52003e-1460-4a65-a213-2bb1a508332f", # Dominguez
#     "b6579ac6-2298-4a9e-8bbe-bdf70b9bb303", # Dominguez
#     "e47f2480-6493-4b42-a83e-a2df2e1a6bb4", # Dominguez

# ]

# excluded_assays = [
#     "Smart-seq", "Smart-seq2", "Smart-seq3", "Smart-seq v4",   # Full-length protocols
#     "Quartz-seq", "MARS-seq", "SORT-seq",                      # Rare and/or niche protocols
#     "GEXSCOPE technology",                                     # Proprietary, very low usage
#     "BD Rhapsody Targeted mRNA",                               # Targeted, not full transcriptome
#     "10x gene expression flex"                                 # Low prevalence + differences
# ]


# filtered = filtered[
#     ~filtered["dataset_id"].isin(excluded_dataset_ids) & 
#     ~filtered["assay"].isin(excluded_assays)
# ]
# print(f"Filtered down to {len(filtered)} cells after applying initial criteria.")

# # Define our sampling strategy percentages with adjustments
# pct_distribution = 0.30
# pct_cell_type = 0.25
# pct_disease = 0.20
# pct_donor = 0.15
# pct_rare = 0.10

# # Calculate target counts for each strategy
# n_distribution = int(target_total * pct_distribution)
# n_cell_type = int(target_total * pct_cell_type)
# n_disease = int(target_total * pct_disease)
# n_donor = int(target_total * pct_donor)
# n_rare = target_total - n_distribution - n_cell_type - n_disease - n_donor

# # Part 1: Distribution-based sampling with tissue adjustment
# # Let's check if brain/blood are overrepresented in the original data
# tissue_counts = filtered['tissue_general'].value_counts(normalize=True)

# # Adjust our sampling to reflect more realistic proportions
# tissue_adjustment = {
#     'brain': 0.20,  # Cap brain at 15%
#     'blood': 0.15   # Cap blood at 15%
# }

# # Create adjusted weights for distribution sampling
# weights = np.ones(len(filtered))

# for tissue, cap in tissue_adjustment.items():
#     # Calculate how much to downweight these tissues
#     current_prop = tissue_counts.get(tissue, 0)
#     if current_prop > cap:
#         downweight_factor = cap / current_prop
#         # Apply downweighting to these tissues
#         weights[filtered['tissue_general'] == tissue] = downweight_factor

# # Sample with adjusted weights
# distribution_sample = filtered.sample(
#     n=n_distribution,
#     weights=weights,
#     random_state=42
# )

# # Part 2: Cell type representation with improved balance
# # Get counts of each cell type
# cell_type_counts = filtered['cell_type'].value_counts()

# # Calculate the max cells per type (cap at 3% of the cell type sample)
# max_per_cell_type = int(n_cell_type * 0.03)

# # Initialize list to hold samples from each cell type
# cell_samples = []

# # Sample from each cell type, capping at the max per type
# for cell_type in cell_type_counts.index:
#     cell_subset = filtered[filtered['cell_type'] == cell_type]

#     # Determine how many to sample (capped at our max)
#     n_to_sample = min(len(cell_subset), max_per_cell_type)

#     if n_to_sample > 0:
#         sample = cell_subset.sample(n=n_to_sample, random_state=42)
#         cell_samples.append(sample)

# # Combine all the cell type samples
# cell_type_sample = pd.concat(cell_samples, ignore_index=True)

# # If we have more than needed, take a random subsample
# if len(cell_type_sample) > n_cell_type:
#     cell_type_sample = cell_type_sample.sample(n=n_cell_type, random_state=42)
# # If we have less than needed, sample more from the general population
# elif len(cell_type_sample) < n_cell_type:
#     remaining = n_cell_type - len(cell_type_sample)
#     remaining_cells = filtered[~filtered.index.isin(cell_type_sample.index)]
#     additional = remaining_cells.sample(n=remaining, random_state=42)
#     cell_type_sample = pd.concat([cell_type_sample, additional], ignore_index=True)

# # Part 3: Disease representation with improved balance
# # Get disease counts
# disease_counts = filtered['disease'].value_counts()

# # Cap COVID-19 representation
# max_covid = int(n_disease * 0.05)  # Cap at 5% of disease sample
# max_normal = int(n_disease * 0.70)  # Allow up to 70% normal cells
# max_per_disease = int(n_disease * 0.03)  # Cap other diseases at 3% each

# # Initialize list to hold disease samples
# disease_samples = []

# # Handle COVID-19 separately
# covid_cells = filtered[filtered['disease'] == 'COVID-19']
# if len(covid_cells) > 0:
#     n_covid = min(len(covid_cells), max_covid)
#     covid_sample = covid_cells.sample(n=n_covid, random_state=42)
#     disease_samples.append(covid_sample)

# # Handle normal cells separately
# normal_cells = filtered[filtered['disease'] == 'normal']
# if len(normal_cells) > 0:
#     n_normal = min(len(normal_cells), max_normal)
#     normal_sample = normal_cells.sample(n=n_normal, random_state=42)
#     disease_samples.append(normal_sample)

# # Sample from each other disease, capping at the max per disease
# for disease in disease_counts.index:
#     if disease not in ['COVID-19', 'normal', 'unknown', np.nan]:
#         disease_subset = filtered[filtered['disease'] == disease]

#         # Determine how many to sample (capped at our max)
#         n_to_sample = min(len(disease_subset), max_per_disease)

#         if n_to_sample > 0:
#             sample = disease_subset.sample(n=n_to_sample, random_state=42)
#             disease_samples.append(sample)

# # Combine all the disease samples
# disease_sample = pd.concat(disease_samples, ignore_index=True)

# # If we have more than needed, take a random subsample
# if len(disease_sample) > n_disease:
#     disease_sample = disease_sample.sample(n=n_disease, random_state=42)
# # If we have less than needed, sample more from the general population
# elif len(disease_sample) < n_disease:
#     remaining = n_disease - len(disease_sample)
#     remaining_cells = filtered[~filtered.index.isin(disease_sample.index)]
#     additional = remaining_cells.sample(n=remaining, random_state=42)
#     disease_sample = pd.concat([disease_sample, additional], ignore_index=True)

# # Part 4: Donor diversity sampling
# # Get donor counts and ensure representation from many donors
# donor_counts = filtered['donor_id'].value_counts()

# # Cap cells per donor to ensure diversity (max 2% of donor sample per donor)
# max_per_donor = max(1, int(n_donor * 0.02))

# # Initialize list to hold donor samples
# donor_samples = []

# # Sample from each donor, capping at the max per donor
# for donor_id in donor_counts.index:
#     donor_subset = filtered[filtered['donor_id'] == donor_id]

#     # Determine how many to sample (capped at our max)
#     n_to_sample = min(len(donor_subset), max_per_donor)

#     if n_to_sample > 0:
#         sample = donor_subset.sample(n=n_to_sample, random_state=42)
#         donor_samples.append(sample)

# # Combine all the donor samples
# donor_sample = pd.concat(donor_samples, ignore_index=True)

# # If we have more than needed, take a random subsample
# if len(donor_sample) > n_donor:
#     donor_sample = donor_sample.sample(n=n_donor, random_state=42)
# # If we have less than needed, sample more from the general population
# elif len(donor_sample) < n_donor:
#     remaining = n_donor - len(donor_sample)
#     remaining_cells = filtered[~filtered.index.isin(donor_sample.index)]
#     additional = remaining_cells.sample(n=remaining, random_state=42)
#     donor_sample = pd.concat([donor_sample, additional], ignore_index=True)

# # Part 5: Rare tissue and developmental stage representation
# # Define rare tissues (bottom 10% by frequency)
# tissue_counts = filtered['tissue_general'].value_counts()
# rare_threshold = tissue_counts.quantile(0.1)
# rare_tissues = tissue_counts[tissue_counts <= rare_threshold].index


# # Get cells from rare tissues or rare developmental stages
# rare_tissue_cells = filtered[filtered['tissue_general'].isin(rare_tissues)]

# # Combine rare tissue and rare developmental stage cells
# rare_cells = pd.concat([rare_tissue_cells]).drop_duplicates()

# if len(rare_cells) > n_rare:
#     n_actual_rare = min(len(rare_cells), n_rare)
#     rare_sample = rare_cells.sample(n=n_actual_rare, random_state=42)
# else:
#     # If we don't have enough rare cells, take from the general population
#     rare_sample = filtered.sample(n=n_rare, random_state=42)

# # Add is_disease column for later analysis
# filtered['is_disease'] = ~filtered['disease'].isin(['normal', 'unknown', np.nan])

# # Combine all samples
# final_samples = [distribution_sample, cell_type_sample, disease_sample, donor_sample, rare_sample]
# final_sample = pd.concat(final_samples, ignore_index=True)

# # Add is_disease column for analysis
# final_sample['is_disease'] = ~final_sample['disease'].isin(['normal', 'unknown', np.nan])

# # Remove potential duplicates
# final_sample = final_sample.drop_duplicates()

# # Adjust to target size
# if len(final_sample) > target_total:
#     final_sample = final_sample.sample(n=target_total, random_state=42)
# elif len(final_sample) < target_total:
#     remaining = target_total - len(final_sample)
#     remaining_cells = filtered[~filtered.index.isin(final_sample.index)]
#     if len(remaining_cells) >= remaining:
#         additional = remaining_cells.sample(n=remaining, random_state=42)
#         final_sample = pd.concat([final_sample, additional], ignore_index=True)
#     else:
#         additional = filtered.sample(n=remaining, replace=True, random_state=42)
#         final_sample = pd.concat([final_sample, additional], ignore_index=True)

# # Final analysis of our sample
# print(f"Final dataset size: {len(final_sample)} cells")

# # Measure diversity
# variables = ['tissue_general', 'dev_stage_group', 'cell_type', 'sex', 'is_disease', 'disease', 'donor_id']

# def shannon_diversity(series):
#     proportions = series.value_counts(normalize = True)
#     shannon_index = - (proportions * np.log2(proportions)).sum()
#     return shannon_index

# def normalized_shannon_diversity(series):
#     num_categories = series.nunique()
#     if num_categories <= 1:
#         return 0.0
    
#     raw_shannon = shannon_diversity(series)
#     max_shannon = np.log2(num_categories)
    
#     return raw_shannon / max_shannon

# diversity_scores = {}
# old_diversity_scores = {}
# for col in variables:
#     diversity_scores[col] = normalized_shannon_diversity(final_sample[col])
#     old_diversity_scores[col] = normalized_shannon_diversity(filtered[col])
#     print(f"Normalized Shannon for '{col}': {old_diversity_scores[col]:.4f} -> {diversity_scores[col]:.4f} ")

# # Calculate the overall average diversity
# average_diversity = np.mean(list(diversity_scores.values()))
# old_average_diversity = np.mean(list(old_diversity_scores.values()))
# print(f"\nAverage Normalized Diversity: {old_average_diversity:.4f} -> {average_diversity:.4f}")


# join_ids = final_sample["soma_joinid"].tolist()
# sorted_join_ids = sorted(final_sample['soma_joinid'].unique())

# print("\n--- Preparing for Robust 95/2.5/2.5 Split ---")

# # Define the columns for which we must ensure category representation
# stratification_cols = ['disease', 'cell_type', 'tissue_general', 'dev_stage_group']

# # Get cell counts for each donor, which helps in making smart choices
# donor_cell_counts = final_sample['donor_id'].value_counts()

# # --- Step 1: Guarantee representation for every category in the training set ---

# train_donors = set()
# unassigned_donors = set(final_sample['donor_id'].unique())

# print("Ensuring all metadata categories are represented in the training set...")

# for col in stratification_cols:
#     # Group donors by the categories they belong to in the current column
#     category_to_donors = final_sample.groupby(col)['donor_id'].unique().apply(set)
    
#     for category, donors_with_category in category_to_donors.items():
#         # Check if this category is already represented by a donor in our training set
#         if train_donors.isdisjoint(donors_with_category):
#             # This category is NOT represented yet. We must add one of its donors.
            
#             # Find which of the potential donors are still available to be assigned
#             available_donors = donors_with_category.intersection(unassigned_donors)
            
#             if not available_donors:
#                 # This should not happen if logic is correct, but as a safeguard:
#                 # It means all donors for this category were already assigned to train_donors
#                 # in a previous step. We can continue.
#                 continue

#             # Heuristic: Pick the smallest available donor to fulfill the requirement.
#             # This leaves larger donors for the flexible greedy assignment later.
#             chosen_donor = min(available_donors, key=lambda d: donor_cell_counts[d])
            
#             # Reserve this donor for the training set
#             train_donors.add(chosen_donor)
#             unassigned_donors.remove(chosen_donor)

# print(f"Reserved {len(train_donors)} donors to guarantee category coverage in the training set.")


# # --- Step 2: Greedily assign remaining donors to validation and test sets ---

# # Calculate target cell counts
# total_cells = len(final_sample)
# val_target_size = int(total_cells * 0.025)
# test_target_size = int(total_cells * 0.025)

# # The pool for assignment is the set of donors not already reserved for training
# assignment_pool = list(unassigned_donors)
# np.random.seed(42)
# np.random.shuffle(assignment_pool)

# val_donors = set()
# test_donors = set()
# val_current_size = 0
# test_current_size = 0

# for donor_id in assignment_pool:
#     donor_size = donor_cell_counts[donor_id]
    
#     # Decide which set is "emptier" proportionally
#     val_fullness = val_current_size / val_target_size if val_target_size > 0 else 1
#     test_fullness = test_current_size / test_target_size if test_target_size > 0 else 1
    
#     # Assign to the less full set, as long as it's not over its target
#     if val_fullness <= test_fullness and val_current_size + donor_size < val_target_size * 1.001: # Allow going slightly over
#         val_donors.add(donor_id)
#         val_current_size += donor_size
#     elif test_current_size + donor_size < test_target_size * 1.001:
#         test_donors.add(donor_id)
#         test_current_size += donor_size
#     else:
#         # If both are full enough, add the remainder to the training set pool
#         pass 

# # --- Step 3: Finalize the splits ---

# # The training set consists of the initially reserved donors plus any unassigned ones
# final_train_donors = train_donors.union(set(assignment_pool) - val_donors - test_donors)

# # Create the final DataFrames
# train_df = final_sample[final_sample['donor_id'].isin(final_train_donors)]
# val_df = final_sample[final_sample['donor_id'].isin(val_donors)]
# test_df = final_sample[final_sample['donor_id'].isin(test_donors)]


# # --- Step 4: Verification and Analysis ---

# print("\n--- Final Split Analysis ---")

# # Check for donor overlap
# assert len(final_train_donors.intersection(val_donors)) == 0, "FATAL: Overlap between train and val!"
# assert len(final_train_donors.intersection(test_donors)) == 0, "FATAL: Overlap between train and test!"
# assert len(val_donors.intersection(test_donors)) == 0, "FATAL: Overlap between val and test!"
# print("✅ No donor overlap between sets.")

# # Check for category coverage
# for col in stratification_cols:
#     train_categories = set(train_df[col].unique())
#     val_categories = set(val_df[col].unique())
#     test_categories = set(test_df[col].unique())
    
#     assert val_categories.issubset(train_categories), f"FATAL: Column '{col}' in validation has categories not in train!"
#     assert test_categories.issubset(train_categories), f"FATAL: Column '{col}' in test has categories not in train!"
# print(f"✅ All categories in {stratification_cols} for val/test are present in train.")

# # Report on sizes
# print("\n--- Dataset Sizes ---")
# print(f"{'Set':<12} | {'Donors':>10} | {'Cells':>12} | {'% of Total Cells':>18}")
# print("-" * 68)
# print(f"{'Training':<12} | {len(final_train_donors):>10} | {len(train_df):>12,} | {len(train_df)/total_cells:>17.2%}")
# print(f"{'Validation':<12} | {len(val_donors):>10} | {len(val_df):>12,} | {len(val_df)/total_cells:>17.2%}")
# print(f"{'Test':<12} | {len(test_donors):>10} | {len(test_df):>12,} | {len(test_df)/total_cells:>17.2%}")

# # Report on diversity preservation
# print("\n--- Diversity Preservation (Normalized Shannon Index) ---")
# diversity_results = []
# for col in ['tissue_general', 'dev_stage_group', 'cell_type', 'is_disease']:
#     train_div = normalized_shannon_diversity(train_df[col])
#     val_div = normalized_shannon_diversity(val_df[col])
#     test_div = normalized_shannon_diversity(test_df[col])
#     diversity_results.append({
#         "Variable": col,
#         "Train": f"{train_div:.3f}",
#         "Validation": f"{val_div:.3f}",
#         "Test": f"{test_div:.3f}"
#     })

# print(pd.DataFrame(diversity_results).to_string(index=False))

# join_ids = train_df["soma_joinid"].tolist()
# train_join_ids = sorted(train_df['soma_joinid'].unique())
# test_join_ids = sorted(test_df['soma_joinid'].unique())
# val_join_ids = sorted(val_df['soma_joinid'].unique())


# # Clear variables from memory

# for name in list(globals()):
#     if name in ('filtered', 'final_sample', 'remaining_cells', 'normal_cells', 'weights', 'covid_cells', 'rare_cells', 'join_ids', 'train_df', 'test_df', 'val_df') :
#         globals().pop(name)

# # Force garbage collection
# gc.collect()

# def _save_adata_in_batches(adata, dataset_type, base_output_dir, batch_size):
#     """Helper function to save a large AnnData object in smaller batches."""
#     if adata.n_obs == 0:
#         print(f"Skipping save for '{dataset_type}' as it has 0 cells.")
#         return

#     print(f"--- Saving {dataset_type} data in batches ---")
#     pbar = tqdm(range(0, adata.n_obs, batch_size), desc=f"Saving {dataset_type} batches")
#     for i in pbar:
#         batch_slice = adata[i:i+batch_size, :].copy()  # Use .copy() to avoid modifying the original adata object
#         batch_num = i // batch_size + 1
        
#         # Define save directory and create it
#         save_dir = os.path.join(base_output_dir, f"{dataset_type}_{batch_num}")
#         os.makedirs(save_dir, exist_ok=True)
        
#         # Define the output path
#         output_path = os.path.join(save_dir, f"batch_{batch_num}.h5ad")
        
#         pbar.set_description(f"Saving {dataset_type} batch {batch_num}")

#         # --- MODIFICATION TO HANDLE GENE SYMBOL CONFLICT ---
#         if batch_slice.var.index.name == 'gene_symbol' and 'gene_symbol' in batch_slice.var.columns:
#             # Check if the index values are identical to the column values
#             if all(batch_slice.var.index == batch_slice.var['gene_symbol']):
#                 print(f"  Batch {batch_num}: Index and 'gene_symbol' column are identical. Dropping the column.")
#                 batch_slice.var.drop(columns=['gene_symbol'], inplace=True)
#             else:
#                 print(f"  Batch {batch_num}: Index and 'gene_symbol' column have different values. Renaming the column.")
#                 batch_slice.var.rename(columns={'gene_symbol': 'gene_symbol_col'}, inplace=True)
            
#             # Remove the index name to prevent further conflicts during writing
#             batch_slice.var.index.name = None
#             print(f"  Batch {batch_num}: Removed index name.")
        
#         batch_slice.write(output_path)
        
#     print(f"Finished saving {dataset_type} data.")

# def process_anndata_and_split(
#     sorted_join_ids,
#     train_ids,
#     test_ids,
#     val_ids,
#     batch_size,
#     data_dir,
#     census,
#     organism="Homo sapiens",
#     gmt_path=None,
#     output_dir=None,
#     chunk_size=1000,
# ):
#     """
#     Fetches a full dataset, processes it globally with HVG selection and normalization,
#     then calculates pathway scores in batches and splits into train/test/val sets.

#     Args:
#         sorted_join_ids (list): A sorted list of ALL soma_joinids to process.
#         train_ids (list): A list of soma_joinids for the training set.
#         test_ids (list): A list of soma_joinids for the test set.
#         val_ids (list): A list of soma_joinids for the validation set.
#         batch_size (int): The number of cells to save in each output file (batch).
#         data_dir (pathlib.Path or str): The base directory for data.
#         census: The cellxgene_census object.
#         organism (str, optional): The organism to query. Defaults to "Homo sapiens".
#         gmt_path (str, optional): Path to the GMT file for gene signatures.
#         output_dir (str, optional): Directory to save the output h5ad files.
#         chunk_size (int, optional): Chunk size for AUCell calculation.
#     """
#     if gmt_path is None:
#         gmt_path = os.path.join(data_dir,"h.all.v2025.1.Hs.symbols.gmt")
#     if output_dir is None:
#         output_dir = os.path.join(data_dir, 'input_data')

#     os.makedirs(output_dir, exist_ok=True)

#     # --- 1. Fetch ALL data at once ---
#     print("--- Starting to fetch the entire AnnData object ---")
#     if not sorted_join_ids:
#         print("Warning: sorted_join_ids is empty. Nothing to do.")
#         return
        
#     join_ids_str = ",".join(map(str, sorted_join_ids))

#     full_data = cellxgene_census.get_anndata(
#         census=census,
#         organism=organism,
#         obs_value_filter=f"soma_joinid in [{join_ids_str}]",
#         obs_column_names=[
#             "soma_joinid", "sex", "tissue", "donor_id", "dataset_id",
#             "tissue_ontology_term_id", "tissue_general", 'tissue_general_ontology_term_id',
#             "cell_type", "cell_type_ontology_term_id", "disease_ontology_term_id",
#             "assay", "assay_ontology_term_id", "disease", "development_stage"
#         ],
#         X_name="raw"
#     )
#     print(f"Successfully fetched {full_data.n_obs} cells.")

#     # --- 2. Initial processing and gene annotation ---
#     print("--- Adding basic annotations ---")
    
#     if 'ensembl_id' not in full_data.var.columns:
#         full_data.var['ensembl_id'] = full_data.var['feature_id']

#     if scipy.sparse.issparse(full_data.X):
#         full_data.obs["n_counts"] = np.array(full_data.X.sum(axis=1)).flatten()
#     else:
#         full_data.obs["n_counts"] = full_data.X.sum(axis=1)
#     print("Added n_counts and ensembl_id.")

#     mg = mygene.MyGeneInfo()
#     ensembl_ids = full_data.var['ensembl_id'].tolist()
#     print("Querying MyGene.info for gene symbols...")
#     results = mg.querymany(ensembl_ids, scopes='ensembl.gene', fields='symbol', species='human')
#     id_to_symbol = {res['query']: res.get('symbol', '') for res in results}
#     full_data.var['gene_symbol'] = full_data.var['ensembl_id'].map(id_to_symbol)
#     full_data.var.index = full_data.var['gene_symbol']
#     print("Gene symbols added.")



#     # --- 4. Global normalization and HVG selection ---
#     print("--- Global normalization and HVG selection ---")
    
#     # Create a copy for global processing
#     global_data = full_data.copy()
    
#     # Global normalization and HVG selection
#     sc.pp.normalize_total(global_data, target_sum=1e4)
#     sc.pp.log1p(global_data)
#     sc.pp.highly_variable_genes(global_data, n_top_genes=40000, flavor="seurat")
    
#     # Get the global HVG genes
#     hvg_genes = global_data.var_names[global_data.var.highly_variable]
#     print(f"Selected {len(hvg_genes)} highly variable genes globally.")
    
#     # Keep only HVGs in the normalized data
#     global_data_hvg = global_data[:, global_data.var.highly_variable].copy()
#     print(f"Global normalized data shape: {global_data_hvg.shape}")
    
#     # Clean up the full global_data (keep only HVG version)
#     del global_data
#     gc.collect()

#     # --- 5. Load gene signatures ---
#     print("--- Loading gene signatures ---")
#     signatures = GeneSignature.from_gmt(str(gmt_path), field_separator="\t")
#     signature_names = [sig.name for sig in signatures]
#     print(f"Loaded {len(signatures)} gene signatures.")

#     # --- 6. Calculate AUCell scores in batches ---
#     print("--- Calculating AUCell scores in batches ---")
    
#     # Convert to CSR for efficient row slicing
#     ex_matrix = global_data_hvg.X.tocsr() if sparse.issparse(global_data_hvg.X) else global_data_hvg.X
#     n_chunks = int(np.ceil(global_data_hvg.n_obs / chunk_size))
    
#     # Initialize storage for all AUC scores
#     all_aucs = []
    
#     for chunk_idx in range(n_chunks):
#         start_idx = chunk_idx * chunk_size
#         end_idx = min((chunk_idx + 1) * chunk_size, global_data_hvg.n_obs)
        
#         print(f"Processing chunk {chunk_idx + 1}/{n_chunks} (cells {start_idx}:{end_idx})")
        
#         # Get chunk data
#         chunk_slice = slice(start_idx, end_idx)
#         chunk_matrix = ex_matrix[chunk_slice]
        
#         # Convert to dense for AUCell
#         if sparse.issparse(chunk_matrix):
#             chunk_dense = np.array(chunk_matrix.todense(), dtype=np.float16)
#         else:
#             chunk_dense = chunk_matrix.astype(np.float16)
        
#         # Create DataFrame for this chunk
#         chunk_df = pd.DataFrame(
#             chunk_dense,
#             index=global_data_hvg.obs_names[chunk_slice],
#             columns=global_data_hvg.var_names
#         )
        
#         # Create rankings for this chunk
#         rnk_chunk = create_rankings(chunk_df)
        
#         # Calculate AUC scores for all signatures
#         chunk_aucs = []
#         for signature in signatures:
#             auc_scores = enrichment(rnk_chunk, signature)
#             chunk_aucs.append(auc_scores)
        
#         # Stack AUC scores for this chunk
#         chunk_aucs = np.column_stack(chunk_aucs)
#         all_aucs.append(chunk_aucs)
        
#         # Clean up chunk data
#         del chunk_df, rnk_chunk, chunk_dense
#         gc.collect()

#     # --- 7. Combine all AUC scores ---
#     print("--- Combining AUCell results ---")
#     final_auc_array = np.vstack(all_aucs)
#     aucs_df = pd.DataFrame(
#         final_auc_array,
#         index=global_data_hvg.obs_names,
#         columns=signature_names
#     )
    
#     # Clean up
#     del all_aucs, final_auc_array, ex_matrix, global_data_hvg
#     gc.collect()

#     # --- 8. Filter pathways and calculate top pathways ---
#     print("--- Filtering pathways and calculating top pathways ---")
    
#     # Pathway filtering (keep pathways active in >0.5% of cells)
#     top5_percentile_mask = aucs_df.apply(lambda row: row >= np.percentile(row, 95), axis=1)
#     pathway_frequencies = top5_percentile_mask.sum(axis=0) / top5_percentile_mask.shape[0]
#     selected_pathways = pathway_frequencies[pathway_frequencies > 0.005].index
#     filtered_aucs = aucs_df[selected_pathways]
    
#     print(f"Kept {len(selected_pathways)} pathways out of {len(signature_names)} after filtering.")
    
#     # Calculate top 2 pathways for each cell
#     pathway1 = aucs_df.idxmax(axis=1)
    
#     def get_second_largest_pathway_name(row):
#         sorted_pathways = row.sort_values(ascending=False).index
#         return sorted_pathways[1] if len(sorted_pathways) >= 2 else np.nan
    
#     pathway2 = aucs_df.apply(get_second_largest_pathway_name, axis=1).fillna("No_Second_Pathway")
    
#     # --- 9. Add pathway information to original data ---
#     print("--- Adding pathway information to original data ---")
    
#     # Ensure the indices match between aucs results and original data
#     common_cells = full_data.obs_names.intersection(aucs_df.index)
#     print(f"Found {len(common_cells)} common cells between original and processed data.")
    
#     # Initialize pathway columns
#     full_data.obs["pathway1"] = "Unknown"
#     full_data.obs["pathway2"] = "Unknown"
    
#     # Add pathway information for cells that were processed
#     full_data.obs.loc[common_cells, "pathway1"] = pathway1.loc[common_cells].values
#     full_data.obs.loc[common_cells, "pathway2"] = pathway2.loc[common_cells].values
    
    
#     # Clean up AUC data
#     del aucs_df, filtered_aucs, pathway1, pathway2
#     gc.collect()
    
#     print("--- Completed pathway enrichment for all data ---")

#     # --- 10. Split the data ---
#     print("--- Splitting data into train, test, and validation sets ---")
    
#     train_mask = full_data.obs['soma_joinid'].isin(set(train_ids))
#     test_mask = full_data.obs['soma_joinid'].isin(set(test_ids))
#     val_mask = full_data.obs['soma_joinid'].isin(set(val_ids))

#     train_adata = full_data[train_mask, :].copy()
#     test_adata = full_data[test_mask, :].copy()
#     val_adata = full_data[val_mask, :].copy()
    
#     print(f"Split complete. Train: {train_adata.n_obs}, Test: {test_adata.n_obs}, Val: {val_adata.n_obs} cells.")
    
#     del full_data
#     gc.collect()
    
#     # --- 11. Save each split in batches ---
#     _save_adata_in_batches(train_adata, 'train', output_dir, batch_size)
#     _save_adata_in_batches(test_adata, 'test', output_dir, batch_size)
#     _save_adata_in_batches(val_adata, 'val', output_dir, batch_size)
    
#     print("\n--- All processing and saving complete. ---")

# process_anndata_and_split(
#     sorted_join_ids,
#     train_join_ids,
#     test_join_ids,
#     val_join_ids,
#     args.BS,
#     DATA_DIR,
#     census,
#     organism="Homo sapiens",
#     chunk_size=args.CS,
# )


def get_webpage_content(url):
    """
    Fetches the HTML content of a given URL and extracts the brief description.

    Args:
        url (str): The URL of the webpage to fetch.

    Returns:
        str: The extracted brief description, or an error message if the request or parsing fails.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url, timeout=10) # Added timeout
        # Added a small delay to be polite to the server and avoid being blocked
        time.sleep(0.1)

        # Raise an HTTPError for bad responses (4xx or 5xx)
        response.raise_for_status()

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the <th> tag with the text "Brief description"
        # Then find the next sibling <td> tag to get its text
        brief_description_th = soup.find('th', string='Brief description')
        if brief_description_th:
            brief_description_td = brief_description_th.find_next_sibling('td')
            if brief_description_td:
                return brief_description_td.get_text(strip=True)
            else:
                return "Error: Could not find the <td> tag after 'Brief description'."
        else:
            return "Error: Could not find 'Brief description' header on the page."

    except requests.exceptions.Timeout:
        return f"Error: Request timed out for {url}"
    except requests.exceptions.RequestException as e:
        # Handle any request-related errors (e.g., network issues, invalid URL)
        return f"Error fetching {url}: {e}"
    except Exception as e:
        # Handle any other unexpected errors during parsing
        return f"Error parsing content from {url}: {e}"

def process_gmt_content(gmt_data):
    """
    Reads GMT-like content, extracts URLs, and fetches brief descriptions for each.

    Args:
        gmt_data (str): A string containing the content of a .gmt file.
                        Each line should have gene set name, URL, and then genes,
                        separated by tabs or multiple spaces.
    """
    results = []
    lines = gmt_data.strip().split('\n')
    for line in lines:
        parts = line.strip().split('\t') # Split by tab
        if len(parts) >= 2: # Ensure at least name and URL are present
            gene_set_name = parts[0].strip()
            url = parts[1].strip()
            print(f"Fetching description for: {gene_set_name} (URL: {url})")
            description = get_webpage_content(url)
            results.append(f"{gene_set_name}: {description}")
        else:
            results.append(f"Skipping malformed line: {line}")
    return "\n".join(results)



def parse_pathway_descriptions(pathway_text):
    """
    Parse pathway descriptions from the format:
    PATHWAY_NAME: Description text.

    Args:
        pathway_text: String containing pathway descriptions

    Returns:
        Dictionary mapping pathway names to descriptions
    """
    descriptions = {}

    for line in pathway_text.strip().split('\n'):
        if ':' in line:
            pathway_name, description = line.split(':', 1)
            descriptions[pathway_name.strip()] = description.strip()

    return descriptions


# Specify the name of your .gmt file
gmt_filename = DATA_DIR /  "h.all.v2025.1.Hs.symbols.gmt"

if os.path.exists(gmt_filename):
    with open(gmt_filename, 'r') as f:
        gmt_file_content = f.read()

    print(f"Starting to process GMT content from {gmt_filename}...")
    processed_results = process_gmt_content(gmt_file_content)
    print("\n--- Processing Complete ---")
    print(processed_results)
    pathway_descriptions = parse_pathway_descriptions(processed_results)
    print("\n--- Pathway Descriptions ---")
    print(pathway_descriptions)

# Parse the metadata text from the ontologies
def parse_cell_ontology(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    def extract_info(cell_id, entry):
        """ Extracts relevant fields while handling nested structures. """
        return {
            "id": cell_id,  # Extract the outer dictionary key as the ID
            "name": entry.get("name", ""),
            "definition": entry.get("def", ""),
            "synonyms": entry.get("synonym", []) if isinstance(entry.get("synonym"), list) else [entry.get("synonym")] if entry.get("synonym") else [],
        }

    parsed_data = [extract_info(cell_id, details) for cell_id, details in data.items()]

    return parsed_data



def tokenize_with_geneformer(input_dir, output_dir, dataset_name):
    """
    Tokenize an h5ad file using Geneformer's TranscriptomeTokenizer and save as an Arrow file.
    Includes preprocessing steps to add required ensembl_id and n_counts columns.

    Parameters:
    -----------
    input_dir : str
        Path to the directory with .h5ad files.
    output_dir : str
        Directory to save the tokenized dataset.
    dataset_name : str
        Name of the dataset.

    Returns:
    --------
    str
        Path to the saved tokenized dataset.
    """
    # Default custom attributes
    custom_attrs = {
        "soma_joinid": "soma_joinid",
        "cell_type": "cell_type",
        "cell_type_ontology_term_id": "cell_type_ontology_term_id",
        "development_stage": "development_stage",
        "disease": "disease",
        "disease_ontology_term_id": "disease_ontology_term_id",
        "sex": "sex",
        "assay": "assay",
        "assay_ontology_term_id": "assay_ontology_term_id",
        "tissue": "tissue",
        "tissue_ontology_term_id": "tissue_ontology_term_id",
        "tissue_general": "tissue_general",
        "tissue_general_ontology_term_id": "tissue_general_ontology_term_id",
        "pathway1": "pathway1",
        "pathway2": "pathway2",
    }

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Tokenizer
    tk = TranscriptomeTokenizer(custom_attrs,
                                gene_median_file= ROOT / "Geneformer" / "geneformer" / "gene_median_dictionary_gc104M.pkl", ######################## to change location #############################
                                token_dictionary_file= ROOT / "Geneformer" / "geneformer" / "token_dictionary_gc104M.pkl",
                                gene_mapping_file= ROOT / "Geneformer" / "geneformer" / "ensembl_mapping_dict_gc104M.pkl"
                                )

    # Tokenize the data
    tk.tokenize_data(
        input_dir,
        output_dir,
        dataset_name,
        file_format="h5ad",
        
    )

    return os.path.join(output_dir, dataset_name)



class CellDescriptionGenerator:
    def __init__(self):
        # Age bucketing for consistent groupings
        self.age_buckets = {
            # Embryonic stages (post-fertilization weeks)
            'embryonic_early': list(range(9, 17)),  # 9th-16th week post-fertilization + Carnegie stages
            'embryonic_late': list(range(17, 32)),   # 17th-31st week post-fertilization

            # Postnatal age buckets
            'infant': list(range(0, 2)),           # 0-1 years (newborn, infant stages)
            'toddler': list(range(2, 5)),          # 2-4 years (child stage 1-4 yo)
            'child': list(range(5, 15)),           # 5-14 years (juvenile stage 5-14 yo)
            'adolescent': list(range(15, 20)),     # 15-19 years
            'young_adult': list(range(20, 30)),    # 20-29 years (young adult, third decade)
            'adult': list(range(30, 50)),          # 30-49 years (fourth, fifth decade)
            'middle_aged': list(range(50, 65)),    # 50-64 years (sixth decade)
            'elderly': list(range(65, 80)),        # 65-79 years (seventh, eighth decade)
            'very_elderly': list(range(80, 120))   # 80+ years (ninth decade, 80 year-old and over)
        }

        # Special stage mappings
        self.special_stages = {
            'blastula stage': 'embryonic_early',
            'embryonic stage': 'embryonic_early',
            'organogenesis stage': 'embryonic_early',
            'newborn stage (0-28 days)': 'infant',
            'infant stage': 'infant',
            'child stage (1-4 yo)': 'toddler',
            'juvenile stage (5-14 yo)': 'child',
            'pediatric stage': 'child',
            'young adult stage': 'young_adult',
            'adult stage': 'adult',
            'prime adult stage': 'adult',
            'middle aged stage': 'middle_aged',
            'late adult stage': 'elderly',
            'postnatal stage': 'infant',  # Generic postnatal
            # Decade stages
            'third decade stage': 'young_adult',
            'fourth decade stage': 'adult',
            'fifth decade stage': 'adult',
            'sixth decade stage': 'middle_aged',
            'seventh decade stage': 'elderly',
            'eighth decade stage': 'elderly',
            'ninth decade stage': 'very_elderly',
            # LMP stages (prenatal)
            'fourth LMP month stage': 'embryonic_early',
            'fifth LMP month stage': 'embryonic_early',
            'sixth LMP month stage': 'embryonic_late',
            'seventh LMP month stage': 'embryonic_late',
            'eighth LMP month stage': 'embryonic_late',
            'ninth LMP month stage': 'embryonic_late',
            # Range stage
            '60-79 year-old stage': 'elderly',
            '80 year-old and over stage': 'very_elderly'
        }

        # Bucket descriptions for natural language
        self.bucket_descriptions = {
            'embryonic_early': 'early embryonic stage',
            'embryonic_late': 'late embryonic stage',
            'infant': 'infancy',
            'toddler': 'early childhood',
            'child': 'childhood',
            'adolescent': 'adolescence',
            'young_adult': 'young adulthood',
            'adult': 'adulthood',
            'middle_aged': 'middle age',
            'elderly': 'elderly stage',
            'very_elderly': 'advanced elderly stage'
        }

    def _extract_cell_info(self, cell_type: str) -> tuple:
        """Extract cell name and description from cell_type string"""
        if ',' in cell_type:
            parts = cell_type.split(',', 1)
            cell_name = parts[0].split(':')[-1].strip()
            cell_desc = parts[1].strip()

            # Clean up the description
            cell_desc = re.sub(r'^[Aa]\s+', '', cell_desc)  # Remove leading "A "
            cell_desc = re.sub(r'\s*\.\s*$', '', cell_desc)  # Remove trailing period

            return cell_name, cell_desc
        else:
            cell_name = cell_type.split(':')[-1].strip()
            return cell_name, None

    def _bucket_age(self, age_str: str) -> str:
        """Convert age to bucket category"""
        if not age_str or age_str.lower() == 'unknown':
            return None

        age_str = age_str.lower().strip()

        # Check special stages first
        if age_str in self.special_stages:
            return self.bucket_descriptions[self.special_stages[age_str]]

        # Extract numeric age for year-old stages
        year_match = re.search(r'(\d+)-year-old stage', age_str)
        if year_match:
            age_num = int(year_match.group(1))
            for bucket, age_range in self.age_buckets.items():
                if age_num in age_range:
                    return self.bucket_descriptions[bucket]

        # Extract numeric age for month-old stages (convert to years)
        month_match = re.search(r'(\d+)-month-old stage', age_str)
        if month_match:
            months = int(month_match.group(1))
            age_years = months / 12
            for bucket, age_range in self.age_buckets.items():
                if int(age_years) in age_range:
                    return self.bucket_descriptions[bucket]

        # Handle post-fertilization weeks
        pf_match = re.search(r'(\d+)(?:st|nd|rd|th)?\s*week post-fertilization', age_str)
        if pf_match:
            week = int(pf_match.group(1))
            if week in self.age_buckets['embryonic_early']:
                return self.bucket_descriptions['embryonic_early']
            elif week in self.age_buckets['embryonic_late']:
                return self.bucket_descriptions['embryonic_late']

        # Handle Carnegie stages (all early embryonic)
        if 'carnegie stage' in age_str:
            return self.bucket_descriptions['embryonic_early']

        # Default fallback
        return age_str.replace('stage', '').strip()

    def _should_combine_tissues(self, tissue: str, tissue_general: str) -> bool:
        """Determine if tissue and tissue_general should be combined"""
        if not tissue or not tissue_general:
            return False

        tissue_clean = tissue.lower().strip()
        tissue_general_clean = tissue_general.lower().strip()

        return (tissue_clean == tissue_general_clean or
                tissue_clean in tissue_general_clean or
                tissue_general_clean in tissue_clean)

    def generate_description(self,
                           cell_type: str,
                           tissue: str = "",
                           tissue_general: str = "",
                           disease: str = "normal",
                           sex: Optional[str] = None,
                           age: Optional[str] = None) -> str:
        """Generate natural language description from structured cell data"""

        # Extract cell information (keep ontology info)
        cell_name, cell_description = self._extract_cell_info(cell_type)

        # Extract tissue names - prefer specific tissue, fallback to general
        tissue_name = ""
        if tissue:
            tissue_name = tissue.split(',')[0].split(':')[-1].strip()
        elif tissue_general:
            tissue_name = tissue_general.split(',')[0].split(':')[-1].strip()

        disease_name = disease.split(',')[0].split(':')[-1].strip() if disease else "normal"

        # Start building description
        description = f"This sample consists of a {cell_name}"

        # Add cell type description if available
        if cell_description:
            description += f", {cell_description}"

        description += ". It originates from"

        # Add tissue
        if tissue_name:
            description += f" the {tissue_name} of"

        # Handle disease and sex
        if sex and sex.lower() != 'unknown':
            sex_part = f" {sex.lower()}"
        else:
            sex_part = ""

        if disease_name.lower() not in ['normal', 'healthy', '']:
            # For diseased samples: "of a [sex] with [disease]"
            description += f" a{sex_part} with {disease_name}"
        else:
            # For normal samples: "of a normal [sex]" or "of a [sex]"
            if sex_part:
                description += f" a normal{sex_part}"
            else:
                description += " a normal individual"

        # Add age/developmental stage (bucketed)
        if age and age.lower() != 'unknown':
            bucketed_age = self._bucket_age(age)
            if bucketed_age:
                description += f" during {bucketed_age}"

        return description + "."


def add_natural_description(dataset: Dataset, ontology_dict: dict, pathway_dict: dict = None, batch_size: int = 1000) -> Dataset:
    """Add a natural language description column to a Hugging Face dataset."""

    generator = CellDescriptionGenerator()

    # Identify all ontology columns
    ontology_columns = [col for col in dataset.column_names if col.endswith('_ontology_term_id')]
    base_columns = {col: col.replace('_ontology_term_id', '') for col in ontology_columns}

    def process_batch(examples):
        batch_descriptions = []
        batch_size_actual = len(examples[list(examples.keys())[0]])

        for i in range(batch_size_actual):
            # Get values for this example
            example = {key: values[i] for key, values in examples.items()}

            # Extract ontology-based information
            ontology_info = {}
            for onto_col, base_col in base_columns.items():
                ontology_id = str(example.get(onto_col, '')).upper() if example.get(onto_col) else ""
                base_value = example.get(base_col, '')

                if base_value:  # If we have a base value
                    if ontology_id and ontology_id in ontology_dict:
                        # Use ontology definition if available
                        definition = ontology_dict[ontology_id].get('definition', '')
                        if definition:
                            ontology_info[base_col] = f"{base_value}, {definition}"
                        else:
                            ontology_info[base_col] = base_value
                    else:
                        # Fallback to just the base name
                        ontology_info[base_col] = base_value

            # Generate basic cell description
            description = generator.generate_description(
                cell_type=ontology_info.get('cell_type', ''),
                tissue=ontology_info.get('tissue', ''),
                tissue_general=ontology_info.get('tissue_general', ''),
                disease=ontology_info.get('disease', 'normal'),
                sex=example.get('sex', ''),
                age=example.get('development_stage', '')
            )

            # Add pathway information if available
            if pathway_dict:
                pathways = []

                # Check for pathway1
                pathway1 = example.get('pathway1', '')
                if pathway1 and pathway1 in pathway_dict:
                    pathways.append(pathway_dict[pathway1])

                # Check for pathway2
                pathway2 = example.get('pathway2', '')
                if pathway2 and pathway2 in pathway_dict:
                    pathways.append(pathway_dict[pathway2])

                # Add pathway info naturally
                if len(pathways) == 1:
                    description = description.rstrip('.') + f". This cell is associated with {pathways[0]}"
                elif len(pathways) == 2:
                    description = description.rstrip('.') + f". This cell is associated with {pathways[0]} Additionally, it involves {pathways[1]}"

            batch_descriptions.append(description)

        examples["natural_desc"] = batch_descriptions
        return examples

    print("Adding natural language descriptions to all examples...")
    return dataset.map(
        process_batch,
        batched=True,
        batch_size=batch_size,
        desc="Adding natural descriptions"
    )

def process_dataset(
    input_dir="/content/drive/MyDrive/cell2text_dataset",
    output_dir="/content/drive/MyDrive/cell2text_dataset_final",
    dataset_name="final_dataset",
    ontology_filepath="/home/arism/cell2text/data_preprocess/obo.json",
    batch_size=1000
):
    print("Tokenizing dataset...")

    #Tokenize the dataset
    tokenize_with_geneformer(input_dir, output_dir, dataset_name)

    print("Loading tokenized dataset...")
    # Load the tokenized dataset
    hf_dataset = load_from_disk(f"{output_dir}/{dataset_name}.dataset")

    print(f"Dataset info: {hf_dataset}")
    print(f"Columns: {hf_dataset.column_names}")
    print(f"Shape: {hf_dataset.shape}")
    print(f"Total number of examples: {len(hf_dataset)}")

    print("Adding structured descriptions...")
    #Load ontology dictionary
    parsed_cells = parse_cell_ontology(ontology_filepath)
    ontology_dict = {
        cell["id"]: {
            "name": cell["name"],
            "definition": cell["definition"],
            "synonym": cell["synonyms"]
        }
        for cell in parsed_cells
    }

    #Add structured descriptions
    hf_dataset = add_natural_description(hf_dataset, ontology_dict, pathway_dict=pathway_descriptions, batch_size=batch_size)

    # Verify all examples have descriptions
    print(f"Examples with struct_desc: {sum(1 for ex in hf_dataset if 'struct_desc' in ex and ex['struct_desc'])}")

    print("Saving final dataset and deleting the old one...")

    # --- Deletion of the old dataset ---
    old_dataset_path = os.path.join(output_dir, dataset_name, ".dataset")

    # Check if the old directory exists before trying to delete it
    if os.path.exists(old_dataset_path) and os.path.isdir(old_dataset_path):
        print(f"Deleting old dataset directory: {old_dataset_path}")
        shutil.rmtree(old_dataset_path)
        print("Old dataset successfully deleted.")
    else:
        print(f"Old dataset directory not found at {old_dataset_path}. Nothing to delete.")
    #Save the final dataset
    final_save_path = os.path.join(output_dir, dataset_name + "_with_descriptions")
    hf_dataset.save_to_disk(final_save_path)
    print(f"Final dataset saved to {final_save_path}")
    print(f"Final dataset size: {len(hf_dataset)} examples")

    return hf_dataset


######################## to change location #############################
os.makedirs(os.path.join(DATA_DIR,'output_data', 'train'), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR,'output_data', 'test'), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR,'output_data', 'val'), exist_ok=True)
path = DATA_DIR / 'input_data'
#path = DATA_DIR / 'output_data'  ######################## to change location #############################

i_train = 1
i_test = 1
i_val = 1
in_dirs = [os.path.join(path, name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

for directo in tqdm(in_dirs, desc='final step'):
    if 'train' in directo.split('/')[-1]:
        hf_dataset = process_dataset(input_dir=directo, output_dir=DATA_DIR / 'output_data' / 'train', ontology_filepath=DATA_DIR  / "obo.json" ,dataset_name=f"dataset_{i_train}")
        i_train+=1
    elif 'test' in directo.split('/')[-1]:
        hf_dataset = process_dataset(input_dir=directo, output_dir=DATA_DIR / 'output_data' / 'test', ontology_filepath=DATA_DIR  / "obo.json" ,dataset_name=f"dataset_{i_test}")
        i_test+=1
    elif 'val' in directo.split('/')[-1]:
        hf_dataset = process_dataset(input_dir=directo, output_dir=DATA_DIR / 'output_data' / 'val', ontology_filepath=DATA_DIR  / "obo.json" ,dataset_name=f"dataset_{i_val}")
        i_val+=1


usage = resource.getrusage(resource.RUSAGE_SELF)
print(f"Maximum RAM used: {usage.ru_maxrss / (1024*1024):.2f} MB")