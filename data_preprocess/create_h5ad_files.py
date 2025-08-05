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

    # Save results
    batch_data.write(f"/home/arism/datasets/raw_data/batch_{i}.h5ad")


    del batch_data
    gc.collect()


