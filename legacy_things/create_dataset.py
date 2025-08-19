import pandas as pd
import numpy as np
import scanpy as sc
import os
import json
import random
import cellxgene_census
import re
import os
import gc
import scipy
import anndata
import mygene
from pyscenic.aucell import GeneSignature, create_rankings, enrichment
import pandas as pd
import numpy as np
import mygene
import pandas as pd
import numpy as np
import scanpy as sc
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import re
import os
import scipy.sparse
from anndata import AnnData
import gc
import h5py
import scipy.sparse as sparse
from scipy.sparse import csr_matrix, vstack
import anndata as ad
import anndata
import torch
import sys
from datasets import  load_from_disk
from Geneformer.geneformer import TranscriptomeTokenizer



# Connect to CELLxGENE API
census = cellxgene_census.open_soma()

import re
from typing import Optional

def map_dev_stage(stage: str) -> str:
    """
    Map developmental stage to standardized categories.
    
    Categories:
    - prenatal: embryonic/fetal development
    - infant: 0-2 years
    - child: 2-15 years  
    - young_adult: 15-30 years
    - adult: 30-60 years
    - aged: 60+ years
    - unknown: cannot determine
    """
    
    if not isinstance(stage, str) or not stage.strip():
        return "unknown"
    
    stage = stage.lower().strip()
    
    # Handle empty or unknown cases
    if stage in ['unknown', '', 'na', 'n/a', 'null']:
        return "unknown"
    
    # Prenatal/Embryonic stages
    prenatal_patterns = [
        r'blastula', r'gastrula', r'embryonic', r'organogenesis',
        r'post-fertilization', r'carnegie stage', r'lmp month',
        r'fetal', r'prenatal', r'gestation'
    ]
    
    if any(re.search(pattern, stage) for pattern in prenatal_patterns):
        return "prenatal"
    
    # Extract numeric age with units
    # Handle formats like "5-year-old", "3 years", "12 months", "8 weeks"
    age_match = re.search(r'(\d+)[-\s]*(year|month|week|day)s?[-\s]*old|\b(\d+)[-\s]*(year|month|week|day)s?\b', stage)
    
    if age_match:
        # Get the numeric value and unit
        value = int(age_match.group(1) or age_match.group(3))
        unit = (age_match.group(2) or age_match.group(4)).lower()
        
        # Convert everything to years for easier comparison
        if unit == "day":
            age_years = value / 365.25
        elif unit == "week":
            age_years = value / 52.18
        elif unit == "month":
            age_years = value / 12
        elif unit == "year":
            age_years = value
        else:
            age_years = None
            
        if age_years is not None:
            if age_years < 2:
                return "infant"
            elif age_years < 15:
                return "child"
            elif age_years < 30:
                return "young_adult"
            elif age_years < 60:
                return "adult"
            else:
                return "aged"
    
    # Handle decade-based descriptions
    decade_mappings = {
        r'first decade|0-10': 'child',
        r'second decade|10-20': 'young_adult',
        r'third decade|20-30': 'young_adult',
        r'fourth decade|30-40': 'adult',
        r'fifth decade|40-50': 'adult',
        r'sixth decade|50-60': 'adult',
        r'seventh decade|60-70': 'aged',
        r'eighth decade|70-80': 'aged',
        r'ninth decade|80-90': 'aged'
    }
    
    for pattern, category in decade_mappings.items():
        if re.search(pattern, stage):
            return category
    
    # Handle stage-based descriptions
    stage_mappings = {
        # Infant stages
        r'newborn|neonatal|infant(?!ile)': 'infant',
        
        # Child stages
        r'child(?!birth)|juvenile|pediatric|toddler': 'child',
        r'adolescent|teenage|teen': 'child',  # Late childhood/adolescence
        
        # Adult stages
        r'young adult': 'young_adult',
        r'prime adult|adult(?!escence)': 'adult',
        r'middle.?aged': 'adult',
        
        # Elderly stages
        r'elderly|aged|senior|geriatric': 'aged',
        r'late adult': 'aged',
        r'very elderly|advanced age': 'aged'
    }
    
    for pattern, category in stage_mappings.items():
        if re.search(pattern, stage):
            return category
    
    # Handle specific age ranges (e.g., "60-79 year-old", "80+")
    range_match = re.search(r'(\d+)[-–](\d+)|(\d+)\+', stage)
    if range_match:
        if range_match.group(3):  # "80+" format
            start_age = int(range_match.group(3))
        else:  # "60-79" format
            start_age = int(range_match.group(1))
            
        if start_age < 2:
            return "infant"
        elif start_age < 15:
            return "child"
        elif start_age < 30:
            return "young_adult"
        elif start_age < 60:
            return "adult"
        else:
            return "aged"
    
    # Handle postnatal (general term for after birth)
    if 'postnatal' in stage:
        return "infant"  # Default to infant for general postnatal
    
    return "unknown"


# Get only the metadata of cell from the database to
obs_df = cellxgene_census.get_obs(census, "homo_sapiens", column_names=["soma_joinid", "development_stage", "disease", "assay","dataset_id", "donor_id", "sex", "tissue", "tissue_general", "cell_type", "is_primary_data"])

# Apply binning at development stage
obs_df["dev_stage_group"] = obs_df["development_stage"].apply(map_dev_stage)

excluded_dataset_ids = [
    "1b9d8702-5af8-4142-85ed-020eb06ec4f6", # Dominguez
    "fe52003e-1460-4a65-a213-2bb1a508332f", # Dominguez
    "b6579ac6-2298-4a9e-8bbe-bdf70b9bb303", # Dominguez
    "e47f2480-6493-4b42-a83e-a2df2e1a6bb4", # Dominguez

]

excluded_assays = [
    "Smart-seq", "Smart-seq2", "Smart-seq3", "Smart-seq v4",  # Full-length protocols
    "CEL-seq2", "Quartz-seq", "MARS-seq", "SORT-seq",         # Rare and/or niche protocols
    "GEXSCOPE technology",                                     # Proprietary, very low usage
    "BD Rhapsody Targeted mRNA",                               # Targeted, not full transcriptome
    "10x gene expression flex"                                 # Low prevalence + differences
]



# Apply any filtering & sampling logic here
filtered = obs_df[
    #obs_df["sex"].isin(["male", "female"]) &
    #obs_df["dev_stage_group"].ne("unknown") &
    obs_df["tissue_general"].notna() &
    obs_df["tissue_general"].ne("unknown") &
    obs_df["disease"].notna() &
    obs_df["disease"].ne("unknown") &
    obs_df["cell_type"].ne("unknown") &
    obs_df["cell_type"].notna() &
    obs_df["is_primary_data"] == True & # removes duplicate entries from the dataset
    ~obs_df["dataset_id"].isin(excluded_dataset_ids) & 
    ~obs_df["assay"].isin(excluded_assays)
]


del obs_df


# Target number of cells
target_total = 510000

# Define our sampling strategy percentages with adjustments
pct_distribution = 0.30
pct_cell_type = 0.25
pct_disease = 0.20
pct_donor = 0.15
pct_rare = 0.10

# Calculate target counts for each strategy
n_distribution = int(target_total * pct_distribution)
n_cell_type = int(target_total * pct_cell_type)
n_disease = int(target_total * pct_disease)
n_donor = int(target_total * pct_donor)
n_rare = target_total - n_distribution - n_cell_type - n_disease - n_donor

# Part 1: Distribution-based sampling with tissue adjustment
# Let's check if brain/blood are overrepresented in the original data
tissue_counts = filtered['tissue_general'].value_counts(normalize=True)

# Adjust our sampling to reflect more realistic proportions
tissue_adjustment = {
    'brain': 0.20,  # Cap brain at 15%
    'blood': 0.15   # Cap blood at 15%
}

# Create adjusted weights for distribution sampling
weights = np.ones(len(filtered))

for tissue, cap in tissue_adjustment.items():
    # Calculate how much to downweight these tissues
    current_prop = tissue_counts.get(tissue, 0)
    if current_prop > cap:
        downweight_factor = cap / current_prop
        # Apply downweighting to these tissues
        weights[filtered['tissue_general'] == tissue] = downweight_factor

# Sample with adjusted weights
distribution_sample = filtered.sample(
    n=n_distribution,
    weights=weights,
    random_state=42
)

# Part 2: Cell type representation with improved balance
# Get counts of each cell type
cell_type_counts = filtered['cell_type'].value_counts()

# Calculate the max cells per type (cap at 3% of the cell type sample)
max_per_cell_type = int(n_cell_type * 0.03)

# Initialize list to hold samples from each cell type
cell_samples = []

# Sample from each cell type, capping at the max per type
for cell_type in cell_type_counts.index:
    cell_subset = filtered[filtered['cell_type'] == cell_type]

    # Determine how many to sample (capped at our max)
    n_to_sample = min(len(cell_subset), max_per_cell_type)

    if n_to_sample > 0:
        sample = cell_subset.sample(n=n_to_sample, random_state=42)
        cell_samples.append(sample)

# Combine all the cell type samples
cell_type_sample = pd.concat(cell_samples, ignore_index=True)

# If we have more than needed, take a random subsample
if len(cell_type_sample) > n_cell_type:
    cell_type_sample = cell_type_sample.sample(n=n_cell_type, random_state=42)
# If we have less than needed, sample more from the general population
elif len(cell_type_sample) < n_cell_type:
    remaining = n_cell_type - len(cell_type_sample)
    remaining_cells = filtered[~filtered.index.isin(cell_type_sample.index)]
    additional = remaining_cells.sample(n=remaining, random_state=42)
    cell_type_sample = pd.concat([cell_type_sample, additional], ignore_index=True)

# Part 3: Disease representation with improved balance
# Get disease counts
disease_counts = filtered['disease'].value_counts()

# Cap COVID-19 representation
max_covid = int(n_disease * 0.05)  # Cap at 5% of disease sample
max_normal = int(n_disease * 0.70)  # Allow up to 70% normal cells
max_per_disease = int(n_disease * 0.03)  # Cap other diseases at 3% each

# Initialize list to hold disease samples
disease_samples = []

# Handle COVID-19 separately
covid_cells = filtered[filtered['disease'] == 'COVID-19']
if len(covid_cells) > 0:
    n_covid = min(len(covid_cells), max_covid)
    covid_sample = covid_cells.sample(n=n_covid, random_state=42)
    disease_samples.append(covid_sample)

# Handle normal cells separately
normal_cells = filtered[filtered['disease'] == 'normal']
if len(normal_cells) > 0:
    n_normal = min(len(normal_cells), max_normal)
    normal_sample = normal_cells.sample(n=n_normal, random_state=42)
    disease_samples.append(normal_sample)

# Sample from each other disease, capping at the max per disease
for disease in disease_counts.index:
    if disease not in ['COVID-19', 'normal', 'unknown', np.nan]:
        disease_subset = filtered[filtered['disease'] == disease]

        # Determine how many to sample (capped at our max)
        n_to_sample = min(len(disease_subset), max_per_disease)

        if n_to_sample > 0:
            sample = disease_subset.sample(n=n_to_sample, random_state=42)
            disease_samples.append(sample)

# Combine all the disease samples
disease_sample = pd.concat(disease_samples, ignore_index=True)

# If we have more than needed, take a random subsample
if len(disease_sample) > n_disease:
    disease_sample = disease_sample.sample(n=n_disease, random_state=42)
# If we have less than needed, sample more from the general population
elif len(disease_sample) < n_disease:
    remaining = n_disease - len(disease_sample)
    remaining_cells = filtered[~filtered.index.isin(disease_sample.index)]
    additional = remaining_cells.sample(n=remaining, random_state=42)
    disease_sample = pd.concat([disease_sample, additional], ignore_index=True)

# Part 4: Donor diversity sampling
# Get donor counts and ensure representation from many donors
donor_counts = filtered['donor_id'].value_counts()

# Cap cells per donor to ensure diversity (max 2% of donor sample per donor)
max_per_donor = max(1, int(n_donor * 0.02))

# Initialize list to hold donor samples
donor_samples = []

# Sample from each donor, capping at the max per donor
for donor_id in donor_counts.index:
    donor_subset = filtered[filtered['donor_id'] == donor_id]

    # Determine how many to sample (capped at our max)
    n_to_sample = min(len(donor_subset), max_per_donor)

    if n_to_sample > 0:
        sample = donor_subset.sample(n=n_to_sample, random_state=42)
        donor_samples.append(sample)

# Combine all the donor samples
donor_sample = pd.concat(donor_samples, ignore_index=True)

# If we have more than needed, take a random subsample
if len(donor_sample) > n_donor:
    donor_sample = donor_sample.sample(n=n_donor, random_state=42)
# If we have less than needed, sample more from the general population
elif len(donor_sample) < n_donor:
    remaining = n_donor - len(donor_sample)
    remaining_cells = filtered[~filtered.index.isin(donor_sample.index)]
    additional = remaining_cells.sample(n=remaining, random_state=42)
    donor_sample = pd.concat([donor_sample, additional], ignore_index=True)

# Part 5: Rare tissue and developmental stage representation
# Define rare tissues (bottom 10% by frequency)
tissue_counts = filtered['tissue_general'].value_counts()
rare_threshold = tissue_counts.quantile(0.1)
rare_tissues = tissue_counts[tissue_counts <= rare_threshold].index


# Get cells from rare tissues or rare developmental stages
rare_tissue_cells = filtered[filtered['tissue_general'].isin(rare_tissues)]

# Combine rare tissue and rare developmental stage cells
rare_cells = pd.concat([rare_tissue_cells]).drop_duplicates()

if len(rare_cells) > n_rare:
    n_actual_rare = min(len(rare_cells), n_rare)
    rare_sample = rare_cells.sample(n=n_actual_rare, random_state=42)
else:
    # If we don't have enough rare cells, take from the general population
    rare_sample = filtered.sample(n=n_rare, random_state=42)

# Add is_disease column for later analysis
filtered['is_disease'] = ~filtered['disease'].isin(['normal', 'unknown', np.nan])

# Combine all samples
final_samples = [distribution_sample, cell_type_sample, disease_sample, donor_sample, rare_sample]
final_sample = pd.concat(final_samples, ignore_index=True)

# Add is_disease column for analysis
final_sample['is_disease'] = ~final_sample['disease'].isin(['normal', 'unknown', np.nan])

# Remove potential duplicates
final_sample = final_sample.drop_duplicates()

# Adjust to target size
if len(final_sample) > target_total:
    final_sample = final_sample.sample(n=target_total, random_state=42)
elif len(final_sample) < target_total:
    remaining = target_total - len(final_sample)
    remaining_cells = filtered[~filtered.index.isin(final_sample.index)]
    if len(remaining_cells) >= remaining:
        additional = remaining_cells.sample(n=remaining, random_state=42)
        final_sample = pd.concat([final_sample, additional], ignore_index=True)
    else:
        additional = filtered.sample(n=remaining, replace=True, random_state=42)
        final_sample = pd.concat([final_sample, additional], ignore_index=True)

# Final analysis of our sample
print(f"Final dataset size: {len(final_sample)} cells")

# Compare distributions of key variables
variables = ['tissue_general', 'dev_stage_group', 'cell_type', 'sex', 'is_disease', 'disease', 'donor_id']

for var in variables:
    if var in filtered.columns:
        orig_dist = filtered[var].value_counts(normalize=True).head(10)
        sample_dist = final_sample[var].value_counts(normalize=True).head(10)

        print(f"\nTop 10 {var} distribution (original):")
        print(orig_dist)

        print(f"\nTop 10 {var} distribution (sampled):")
        print(sample_dist)

        # Calculate percentage change for key categories
        if var in ['tissue_general']:
            print("\nKey tissue changes (original → sampled):")
            for tissue in ['brain', 'blood', 'lung', 'eye', 'breast']:
                orig_pct = orig_dist.get(tissue, 0) * 100
                sample_pct = sample_dist.get(tissue, 0) * 100
                change = sample_pct - orig_pct
                print(f"{tissue}: {orig_pct:.1f}% → {sample_pct:.1f}% ({change:+.1f}%)")

        if var == 'disease':
            print("\nKey disease changes (original → sampled):")
            for disease in ['normal', 'COVID-19', 'Parkinson disease', 'lung adenocarcinoma']:
                if disease in filtered['disease'].values:
                    orig_count = filtered[filtered['disease'] == disease].shape[0]
                    orig_pct = (orig_count / len(filtered)) * 100

                    sample_count = final_sample[final_sample['disease'] == disease].shape[0]
                    sample_pct = (sample_count / len(final_sample)) * 100

                    change = sample_pct - orig_pct
                    print(f"{disease}: {orig_pct:.1f}% → {sample_pct:.1f}% ({change:+.1f}%)")

# Check disease representation
disease_orig_pct = filtered[filtered['is_disease']].shape[0] / len(filtered) * 100
disease_sample_pct = final_sample[final_sample['is_disease']].shape[0] / len(final_sample) * 100
change = disease_sample_pct - disease_orig_pct

print(f"\nDisease samples: {disease_orig_pct:.1f}% → {disease_sample_pct:.1f}% ({change:+.1f}%)")

# Check donor diversity
orig_unique_donors = filtered['donor_id'].nunique()
sample_unique_donors = final_sample['donor_id'].nunique()
donor_coverage = (sample_unique_donors / orig_unique_donors) * 100

print(f"\nDonor diversity:")
print(f"Original dataset: {orig_unique_donors} unique donors")
print(f"Sampled dataset: {sample_unique_donors} unique donors ({donor_coverage:.1f}% coverage)")

# Check average cells per donor in sample
avg_cells_per_donor = len(final_sample) / sample_unique_donors
max_cells_from_donor = final_sample['donor_id'].value_counts().iloc[0]
print(f"Average cells per donor in sample: {avg_cells_per_donor:.1f}")
print(f"Maximum cells from any single donor: {max_cells_from_donor}")

# Check rare category representation
rare_tissue_orig_pct = filtered[filtered['tissue_general'].isin(rare_tissues)].shape[0] / len(filtered) * 100
rare_tissue_sample_pct = final_sample[final_sample['tissue_general'].isin(rare_tissues)].shape[0] / len(final_sample) * 100


print(f"\nRare tissues: {rare_tissue_orig_pct:.2f}% → {rare_tissue_sample_pct:.2f}% ({rare_tissue_sample_pct - rare_tissue_orig_pct:+.2f}%)")

# Get cell type distribution for further analysis
print("\nDetailed cell type distribution in sampled dataset:")
cell_type_dist = final_sample['cell_type'].value_counts().head(15)
print(cell_type_dist)
print(f"Number of unique cell types in sample: {final_sample['cell_type'].nunique()}")

# Get disease distribution for further analysis
print("\nDetailed disease distribution in sampled dataset:")
disease_dist = final_sample['disease'].value_counts().head(15)
print(disease_dist)
print(f"Number of unique diseases in sample: {final_sample['disease'].nunique()}")

# Save the final sampled dataset
final_sample.to_csv('balanced_cell_sample.csv', index=False)

join_ids = final_sample["soma_joinid"].tolist()
sorted_join_ids = sorted(final_sample['soma_joinid'].unique())


# Clear variables from memory

for name in list(globals()):
    if name in ('filtered', 'final_sample', 'remaining_cells', 'normal_cells', 'weights', 'covid_cells', 'rare_cells', 'join_ids') :
        globals().pop(name)

# Force garbage collection
gc.collect()


# Define batch size
batch_size = 100000000000000
all_data = []

# Process in batches using ID ranges
for i in range(0, len(sorted_join_ids), batch_size):
    batch_ids = sorted_join_ids[i:i+batch_size]
    join_ids_str = ",".join(map(str, batch_ids))
    if not batch_ids:
        continue


    print(f"Processing batch {i//batch_size + 1}")

    # Query using range instead of membership
    batch_data = cellxgene_census.get_anndata(
        census=census,
        organism="Homo sapiens",
        obs_value_filter=f"soma_joinid  in [{join_ids_str}]",
        obs_column_names=["soma_joinid", "sex", "tissue", "donor_id", "dataset_id", "tissue_ontology_term_id", "tissue_general", 'tissue_general_ontology_term_id',
                          "cell_type", "cell_type_ontology_term_id", "disease_ontology_term_id", "assay", "assay_ontology_term_id",
                          "disease", "development_stage"],
        X_name="raw"
    )

    # these are necesary for the tokenization with Geneformer
    if 'ensembl_id' not in batch_data.var.columns:
        batch_data.var['ensembl_id'] = batch_data.var['feature_id']


    if scipy.sparse.issparse(batch_data.X):
        batch_data.obs["n_counts"] = np.array(batch_data.X.sum(axis=1)).flatten()
    else:
        batch_data.obs["n_counts"] = batch_data.X.sum(axis=1)

    # CELLxGENE return the raw values at adata.X and Geneforemr expect the raw values at adata.raw.X
    if batch_data.raw is None:
      batch_data.raw = batch_data.copy()



    # Initialize MyGeneInfo
    mg = mygene.MyGeneInfo()

    # Convert Ensembl IDs to gene symbols
    ensembl_ids = batch_data.var['ensembl_id'].tolist()

    # Query mygene
    results = mg.querymany(ensembl_ids, scopes='ensembl.gene', fields='symbol', species='human')

    # Build a mapping from Ensembl ID to gene symbol
    id_to_symbol = {res['query']: res.get('symbol', '') for res in results}

    # Create new column in adata.var with gene symbols
    batch_data.var['gene_symbol'] = batch_data.var['ensembl_id'].map(id_to_symbol)

    batch_data.var.index = batch_data.var['gene_symbol']

    sc.pp.normalize_total(batch_data, target_sum=1e4)
    sc.pp.log1p(batch_data)
    sc.pp.highly_variable_genes(batch_data, n_top_genes=30000, flavor="seurat")
    batch_data = batch_data[:, batch_data.var.highly_variable]


    import gc
    gc.collect()

    # Load gene signatures
    GMT_FNAME = "/home/arism/cell2text/data_preprocess/h.all.v2025.1.Hs.symbols.gmt"
    signatures = GeneSignature.from_gmt(GMT_FNAME, field_separator="\t")

    # Convert adata.X to float32 dense matrix (saves 50% memory)
    print("Creating expression matrix...")
    ex_matrix = pd.DataFrame(
        batch_data.X.astype(np.float32).toarray(),
        index=batch_data.obs_names,
        columns=batch_data.var_names
    )

    # Parameters
    chunk_size = 20000  # Adjust based on your memory budget
    n_chunks = int(np.ceil(ex_matrix.shape[0] / chunk_size))

    # Prepare result container
    all_aucs = []
    signature_names = [sig.name for sig in signatures]

    print("Processing in chunks...")
    for i in range(n_chunks):
        print(f"Processing chunk {i+1}/{n_chunks}")
        chunk = ex_matrix.iloc[i*chunk_size:(i+1)*chunk_size]

        # Create rankings
        rnk_chunk = create_rankings(chunk)

        # Calculate AUCs for this chunk
        chunk_aucs = []
        for signature in signatures:
            auc_scores = enrichment(rnk_chunk, signature)
            chunk_aucs.append(auc_scores)
        
        # Stack into array and add to list
        chunk_aucs = np.column_stack(chunk_aucs)  # shape: (n_cells_chunk, n_signatures)
        all_aucs.append(chunk_aucs)

    # Combine all chunks into one big array
    print("Combining results...")
    final_auc_array = np.vstack(all_aucs)  # shape: (n_total_cells, n_signatures)

    # Create final DataFrame
    aucs_df = pd.DataFrame(
        final_auc_array,
        index=ex_matrix.index,
        columns=signature_names
    )

    # Filter pathways that appear in top 5% of >0.5% of cells
    top5_percentile_mask = aucs_df.apply(lambda row: row >= np.percentile(row, 95), axis=1)
    pathway_frequencies = top5_percentile_mask.sum(axis=0) / top5_percentile_mask.shape[0]
    selected_pathways = pathway_frequencies[pathway_frequencies > 0.005].index

    # Final filtered AUCell matrix
    filtered_aucs = aucs_df[selected_pathways]

    # Store in AnnData object
    batch_data.obsm["AUCell_scores_filtered"] = filtered_aucs.loc[batch_data.obs_names]

    # Find the pathway with the maximum score for each cell
    batch_data.obs["pathway1"] = aucs_df.idxmax(axis=1)

    # --- Add "pathway2" (second most active) to adata.obs ---
    # Define a function to get the second largest pathway name (index) for a given row (cell)
    def get_second_largest_pathway_name(row):
        # Sort the row values in descending order and get the index (pathway names)
        sorted_pathways = row.sort_values(ascending=False).index

        # If there are at least two pathways, return the second one
        if len(sorted_pathways) >= 2:
            return sorted_pathways[1]
        else:
            # Return a placeholder for cells that don't have a second pathway
            # (e.g., if a cell only has one non-zero pathway score, or fewer than 2 pathways overall)
            return np.nan

    # Apply this function across each row (cell) to get the second most active pathway
    batch_data.obs["pathway2"] = aucs_df.apply(get_second_largest_pathway_name, axis=1)

    # --- Optional: Fill NaN values if some cells didn't have a second pathway ---
    # If you prefer 'N/A' or another string instead of NaN for better readability
    batch_data.obs["pathway2"] = batch_data.obs["pathway2"].fillna("No_Second_Pathway")

    # Save results
    batch_data.write(f"/home/arism/datasets/raw_data/batch_{i}.h5ad")

    del batch_data
    gc.collect()


import requests
from bs4 import BeautifulSoup
import time # Import time for rate limiting
import os # Import os for file path handling

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
gmt_filename = "/home/arism/cell2text/data_preprocess/h.all.v2025.1.Hs.symbols.gmt" # You can change this to your file's name

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
                                gene_median_file="/home/arism/cell2text/data_preprocess/Geneformer/geneformer/gene_median_dictionary_gc104M.pkl",
                                token_dictionary_file="/home/arism/cell2text/data_preprocess/Geneformer/geneformer/token_dictionary_gc104M.pkl",
                                gene_mapping_file="/home/arism/cell2text/data_preprocess/Geneformer/geneformer/ensembl_mapping_dict_gc104M.pkl"
                                )

    # Tokenize the data
    tk.tokenize_data(
        input_dir,
        output_dir,
        dataset_name,
        file_format="h5ad",
        
    )

    return os.path.join(output_dir, dataset_name)

import re
from typing import Dict, Optional
from datasets import Dataset

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

    print("Saving final dataset...")
    #Save the final dataset
    final_save_path = os.path.join(output_dir, dataset_name + "_with_descriptions")
    hf_dataset.save_to_disk(final_save_path)
    print(f"Final dataset saved to {final_save_path}")
    print(f"Final dataset size: {len(hf_dataset)} examples")

    return hf_dataset

hf_dataset = process_dataset(input_dir="/home/arism/datasets/raw_data/", output_dir="/home/arism/datasets", ontology_filepath="/home/arism/cell2text/data_preprocess/obo.json" ,dataset_name="dataset_510k")


