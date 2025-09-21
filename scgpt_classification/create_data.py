import scanpy as sc
import pandas as pd
import requests
import os
import io # Import the io module to handle string data as a file

def download_h5ad_file(url, filename):
    """
    Downloads a single .h5ad file from a given URL if it doesn't already exist.

    Args:
        url (str): The URL of the file to download.
        filename (str): The local filename to save the file as.
    """
    if not os.path.exists(filename):
        print(f"Downloading {filename} from {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded {filename} successfully.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {filename}: {e}")
            return False
    else:
        print(f"{filename} already exists, skipping download.")
    return True

def map_gene_info(adata, mapping_df):
    """
    Maps gene IDs in the AnnData object using a provided mapping DataFrame.

    This function assumes that the AnnData .var index contains the IDs to be mapped (e.g., soma_joinid),
    and the mapping_df contains the corresponding gene symbols.

    Args:
        adata (sc.AnnData): The AnnData object to update.
        mapping_df (pd.DataFrame): A DataFrame with at least 'soma_joinid' and
                                   'feature_name' columns.

    Returns:
        sc.AnnData: The updated AnnData object.
    """
    print("Mapping gene IDs to gene symbols using the provided CSV data...")
    
    # Create a dictionary from the mapping DataFrame for quick lookups
    # We set the index of the mapping DataFrame to the soma_joinid
    # for easy alignment with the adata.var index.
    mapping_df.set_index('soma_joinid', inplace=True)

    # Add the 'feature_name' column from the mapping DataFrame to the adata.var DataFrame
    adata.var['gene_symbol'] = adata.var.index.map(mapping_df['feature_name'])

    # Count how many genes were successfully mapped
    successful_mappings = adata.var['gene_symbol'].notna().sum()
    print(f"Found symbols for {successful_mappings} genes.")

    # Set the gene symbols as the new index
    # We use a copy to avoid a SettingWithCopyWarning
    adata.var = adata.var.copy()
    adata.var.index = adata.var['gene_symbol']

    # Drop the temporary column and the original index to clean up
    adata.var = adata.var.drop(columns=['gene_symbol'])

    print("Gene symbols added and set as index.")
    return adata

def process_single_dataset(url, dataset_id, mapping_df):
    """
    Main function to process a single H5AD dataset from a URL.

    Args:
        url (str): The URL of the dataset.
        dataset_id (int): A unique identifier for the dataset (e.g., 1, 2, ...).
        mapping_df (pd.DataFrame): The DataFrame containing gene mapping information.
    """
    filename = f"dataset_{dataset_id}.h5ad"
    output_filename = f"processed_dataset_{dataset_id}.h5ad"

    if not download_h5ad_file(url, filename):
        return

    print(f"\n--- Processing {filename} ---")
    
    # Load the dataset
    print(f"Loading {filename}...")
    try:
        adata = sc.read_h5ad(filename)
        print(f"Loaded {filename}: {adata.shape}")
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return

    # Map gene IDs using the provided mapping DataFrame
    adata = map_gene_info(adata, mapping_df.copy()) # Pass a copy to avoid modifying the original DataFrame

    # Save the processed dataset
    adata.write(output_filename)
    print(f"Saved processed dataset as {output_filename}")
    print(f"Final shape for {output_filename}: {adata.shape}")
    print(f"Number of cells: {adata.n_obs}")
    print(f"Number of genes: {adata.n_vars}")
    print(f"--- Finished processing {filename} ---")

# --- Main Script Execution ---
if __name__ == "__main__":
    # Dataset URLs
    urls = [
        "https://datasets.cellxgene.cziscience.com/7939849e-eb3d-4a05-b59d-89645c391242.h5ad",
        "https://datasets.cellxgene.cziscience.com/acc75828-6325-4ccc-9f2a-2861c0edd00e.h5ad",
    ]

    # Your gene mapping data as a string
    gene_mapping_data = """soma_joinid,feature_id,feature_name,feature_length
0,0,ENSG00000121410,A1BG,3999
1,1,ENSG00000268895,A1BG-AS1,3374
2,2,ENSG00000148584,A1CF,9603
3,3,ENSG00000175899,A2M,6318
4,4,ENSG00000245105,A2M-AS1,2948
5,5,ENSG00000166535,A2ML1,7156
6,6,ENSG00000256661,A2ML1-AS1,452
7,7,ENSG00000184389,A3GALT2,1023
8,8,ENSG00000128274,A4GALT,3358
9,9,ENSG00000118017,A4GNT,1779
10,10,ENSG00000094914,AAAS,4727"""
    
    # Load the gene mapping data into a pandas DataFrame
    mapping_df = pd.read_csv(io.StringIO(gene_mapping_data))

    print("Starting dataset processing...")

    # Process each file individually, passing the mapping DataFrame
    for i, url in enumerate(urls):
        process_single_dataset(url, i + 1, mapping_df)

    print("\nAll tasks are complete!")
