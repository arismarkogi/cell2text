import os
import requests
import scanpy as sc
import h5py
import json
import re
import h5py
import pandas as pd
import numpy as np
import scipy.sparse
from anndata import AnnData
import gc
import h5py
import pandas as pd
import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csr_matrix, vstack
from sklearn.model_selection import train_test_split
import anndata as ad
import anndata
import torch
from datasets import Dataset
from geneformer import TranscriptomeTokenizer
import gc



# Create a dictionary that stores the text descritpiosns of the obo.json (OBO Foundry) Ontologies
# Key: ontology_term_id, Value: the description
def parse_obo_ontologies(file_path):
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


# Ger the ID's of the .h5ad files that we will donwload
def get_ids_from_csv(file_path):
    
    try:
        # Read the CSV fle
        df = pd.read_csv(file_path)

        # Extract the 'id' column as a list
        if 'id' in df.columns:
            return df['id'].tolist()
        else:
            print("Column 'id' not found in the CSV.")
            return []

    except Exception as e:
        print(f"An error occurred: {e}")
        return []


# Downloads datasets from CELL x GENE API
def download_datasets(dataset_names):
    os.makedirs("my_datasets", exist_ok=True)  # Ensure directory exists

    for dataset in dataset_names:
        url = f"https://datasets.cellxgene.cziscience.com/{dataset}.h5ad"
        filename = f"my_datasets/{dataset}.h5ad"

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded_size = 0

            with open(filename, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024*1024*10):  # 1MB chunks
                    if chunk:
                        file.write(chunk)
                        downloaded_size += len(chunk)
                        print(f"\rDownloading {dataset}: {downloaded_size / total_size:.2%} complete", end="")

            print(f"\nDataset downloaded successfully to: {os.path.abspath(filename)}")

        except requests.exceptions.RequestException as e:
            print(f"Error downloading dataset {dataset}: {e}")
        except IOError as e:
            print(f"Error writing to file {dataset}: {e}")

# Fucntion to show some basic things about an h5ad file
def inspect_h5ad(dataset):

    file_path = f"{dataset}"  # Update with your dataset path

    try:
        with h5py.File(file_path, "r") as f:
            print(f"\nInspecting {file_path}...")

            # Check if 'X' exists and is a dataset
            shape = "Unknown"
            if "X" in f:
                if isinstance(f["X"], h5py.Dataset):
                    shape = f["X"].shape
                else:
                    print("Warning: 'X' is a group, not a dataset.")

            print("Shape (cells x genes):", shape)

            # Observations (obs metadata)
            obs_keys = list(f["obs"].keys()) if "obs" in f else []
            print("\n--- Observations (adata.obs) ---")
            print(f"Metadata columns in obs: {obs_keys}")

            # Variables (var metadata)
            var_keys = list(f["var"].keys()) if "var" in f else []
            print("\n--- Variables (adata.var) ---")
            print(f"Metadata columns in var: {var_keys}")

            # Layers (additional data matrices)
            layers = list(f["layers"].keys()) if "layers" in f else []
            print("\n--- Layers ---")
            print(f"Available layers: {layers}")

            # Embeddings (obsm)
            obsm_keys = list(f["obsm"].keys()) if "obsm" in f else []
            print("\n--- Embeddings (obsm) ---")
            print(f"Available embeddings: {obsm_keys}")

            # Unstructured data (uns)
            uns_keys = list(f["uns"].keys()) if "uns" in f else []
            print("\n--- Unstructured Data (uns) ---")
            print(f"Keys in uns: {uns_keys}")

            print(f"First 5 rows of .var_names (index):")
            print(f["var"].keys()[:5])

            print("-" * 100)

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred while reading {file_path}: {e}")


import anndata as ad

def print_first_n_rows_h5ad(h5ad_file, n=5):
    """
    Prints the first n rows of the .obs dataframe of an h5ad file.

    Args:
        h5ad_file (str): Path to the h5ad file.
        n (int): Number of rows to print. Defaults to 5.
    """
    try:
        adata = ad.read_h5ad(h5ad_file, backed = "r")
        print(adata.obs.head(n))  # Print the first n rows of the .obs dataframe.
        row_index = adata.obs.index[0] #gets the first row index label.
        print(adata.obs.loc[row_index]['text_desc']) #prints the whole row.
    except FileNotFoundError:
        print(f"Error: File '{h5ad_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def add_text_description(adata, ontology_dict):
    """More memory-efficient version that avoids iterrows()"""
    # Identify all ontology columns in obs
    ontology_columns = [col for col in adata.obs.columns if col.endswith('_ontology_term_id')]
    base_columns = {col: col.replace('_ontology_term_id', '') for col in ontology_columns}

    # Create a function to process a single row
    def process_row(row):
        cell_desc_parts = []
        for onto_col, base_col in base_columns.items():
            if pd.isnull(row[onto_col]) or row[onto_col] == "":
                continue

            ontology_id = str(row[onto_col]).upper() if not pd.isnull(row[onto_col]) else ""
            base_value = row[base_col] if base_col in row and not pd.isnull(row[base_col]) else ""

            definition = ""
            if ontology_id in ontology_dict:
                definition = ontology_dict[ontology_id].get('definition', '')

            if base_value and definition:
                desc_part = f"{base_col.replace('_', ' ').title()}: {base_value}, {definition}"
                cell_desc_parts.append(desc_part)
            elif base_value:
                desc_part = f"{base_col.replace('_', ' ').title()}: {base_value}"
                cell_desc_parts.append(desc_part)

        return "; ".join(cell_desc_parts) + "."

    # Apply the function to each row without using iterrows()
    adata.obs['text_desc'] = adata.obs.apply(process_row, axis=1)

    return adata

def tokenize_with_geneformer(h5ad_path, output_dir):
    """
    Tokenize an h5ad file using Geneformer's TranscriptomeTokenizer and save as an Arrow file.

    Parameters:
    -----------
    h5ad_path : str
        Path to the input .h5ad file.
    output_dir : str
        Directory to save the tokenized dataset.

    Returns:
    --------
    str
        Path to the saved tokenized dataset.
    """
    # Default custom attributes
    custom_attrs = {
        "assay": "assay",
        "cell_type": "cell_type",
        "development_stage": "development_stage",
        "disease": "disease",
        "donor_id": "donor_id",
        "sex": "sex",
        "tissue": "tissue",
        "text_desc": "text_desc"
    }

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get filename without extension for output naming
    filename = os.path.splitext(os.path.basename(h5ad_path))[0]

    # Initialize Tokenizer
    tk = TranscriptomeTokenizer(custom_attrs)

    # Tokenize the data
    tk.tokenize_data(
        h5ad_path,
        output_dir,
        filename,
        file_format="h5ad"
    )

    # Locate tokenized dataset files
    tokenized_files = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.startswith(filename)
    ]

    if not tokenized_files:
        raise ValueError("No tokenized dataset files found")

    # Load the tokenized dataset
    hf_dataset = Dataset.load_from_disk(tokenized_files[0])

    # Define the output Arrow file path
    arrow_file_path = os.path.join(output_dir, f"{filename}.arrow")

    # Save the dataset in Apache Arrow format
    hf_dataset.save_to_disk(arrow_file_path)

    print(f"Tokenized dataset saved at: {arrow_file_path}")

    return hf_dataset

def process_single_file(input_filename, output_filename, ontology_dict=None,
                      obs_columns_to_keep=None, clean=True, add_description=False):
    """Memory-optimized version of process_single_file"""
    print(f"Processing {input_filename}...")

    try:
        print("Processing dataset...")
        adata = sc.read_h5ad(input_filename)

        # Get the filename without extension for output naming
        filename = os.path.splitext(os.path.basename(input_filename))[0]

        # Filter obs columns if requested
        print("Applying filtering")
        existing_columns = list(adata.obs.keys())
        columns_to_keep = [col for col in obs_columns_to_keep if col in existing_columns]
        adata.obs = adata.obs[columns_to_keep]

        # Aggressively clean up unneeded data
        adata.layers = {}
        adata.obsm = {}
        adata.uns = {}
        adata.var = pd.DataFrame(index=adata.var.index)
        adata.var['ensembl_id'] = adata.var.index

        # Force garbage collection after large operation
        gc.collect()

        if "n_counts" not in adata.obs:
          print( "'n_counts' column is missing! we will fix it")
        adata.obs["n_counts"] = np.array(adata.X.sum(axis=1)).flatten()

        print(f"Initial shape {adata.shape}")
        print(f"Number of genes {adata.n_vars}")
        print(f"Number of cells {adata.n_obs}")


        # Add text descriptions
        if add_description and ontology_dict:
            adata = add_text_description(adata, ontology_dict)
            gc.collect()
        
        print(f"Final shape {adata.shape}")
        print(f"Number of genes {adata.n_vars}")
        print(f"Number of cells {adata.n_obs}")

        

        adata.write(output_filename, compression="gzip")
        
        inspect_h5ad(output_filename)
        print_first_n_rows_h5ad(output_filename)

        # Clear the AnnData object from memory
        del adata
        gc.collect()

        os.remove(input_filename)  # Remove the original file

        # Create tokenized dataset directly
        hf_dataset = tokenize_with_geneformer("processed", "tokenized")

        # Clear the AnnData object from memory
        os.remove(output_filename)  
        gc.collect()

        return hf_dataset

    except Exception as e:
        print(f"Error processing {input_filename}: {e}")
        return False



# # Example usage pipeline
# if __name__ == "__main__":
#     # Define commonly needed metadata columns
#     obs_columns_to_keep = [
#         'assay', 'assay_ontology_term_id', 'cell_type', 'cell_type_ontology_term_id',
#         'development_stage', 'development_stage_ontology_term_id', 'disease',
#         'disease_ontology_term_id', 'donor_id', 'self_reported_ethnicity',
#         'self_reported_ethnicity_ontology_term_id', 'sex', 'sex_ontology_term_id',
#         'tissue', 'tissue_ontology_term_id'
#     ]

#     os.makedirs("processed", exist_ok=True)

#     os.makedirs("tokenized", exist_ok=True)

#     for dataset in dataset_ids:
#         cur_dataset = process_single_file(
#           f"my_datasets/{dataset}.h5ad",
#           f"processed/{dataset}.h5ad",
#           ontology_dict=parsed_ontology_dict,
#           obs_columns_to_keep=obs_columns_to_keep,
#           clean=True,
#           add_description=True
#         )
#         print(f"Processed {dataset}.h5ad")
#         print(cur_dataset)

#         del cur_dataset

#         gc.collect()





