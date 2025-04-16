from data.utils import get_ids_from_csv, download_datasets, inspect_h5ad

# Example usage:
file_path = "/datasets.csv"
dataset_ids = get_ids_from_csv(file_path)


# this is for debugging
dataset_ids = ['26f36ff7-17b6-4285-8b35-9512dcae307b']


download_datasets(dataset_ids)

# you can remove this if you want
for dataset in dataset_ids:
    inspect_h5ad(f"my_datasets/{dataset}.h5ad")(f"my_datasets/{dataset}.h5ad")