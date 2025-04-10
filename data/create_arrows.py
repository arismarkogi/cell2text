from dataset_util import parse_obo_ontologies, process_single_file, get_ids_from_csv
import os
import gc

file_path = "/datasets.csv"
dataset_ids = get_ids_from_csv(file_path)


# this is for debugging
dataset_ids = ['26f36ff7-17b6-4285-8b35-9512dcae307b']

os.makedirs("processed", exist_ok=True)

os.makedirs("tokenized", exist_ok=True)


parsed_terms = parse_obo_ontologies("obo.json")


parsed_ontology_dict = {
    cell["id"]: {
        "name": cell["name"],
        "definition": cell["definition"],
        "synonym": cell["synonyms"]
    }
    for cell in parsed_terms
}

obs_columns_to_keep = [
        'assay', 'assay_ontology_term_id', 'cell_type', 'cell_type_ontology_term_id',
        'development_stage', 'development_stage_ontology_term_id', 'disease',
        'disease_ontology_term_id', 'donor_id', 'self_reported_ethnicity',
        'self_reported_ethnicity_ontology_term_id', 'sex', 'sex_ontology_term_id',
        'tissue', 'tissue_ontology_term_id'
]

for dataset in dataset_ids:
    cur_dataset = process_single_file(
          f"my_datasets/{dataset}.h5ad",
          f"processed/{dataset}.h5ad",
          ontology_dict=parsed_ontology_dict,
          obs_columns_to_keep=obs_columns_to_keep,
          clean=True,
          add_description=True
        )
    

    print(f"Processed {dataset}.h5ad")
    
    
    # This is for debugging
    print(cur_dataset)

    del cur_dataset

    gc.collect()
