from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, jaccard_score, roc_auc_score, average_precision_score
import re
import pickle
import numpy as np
import pandas as pd
import json
import difflib
from typing import Optional, Tuple, List


class CellTypeExtractor:
    """Extract cell types, diseases, tissues, pathways and calculate metrics"""
    
    def __init__(self, 
                 similarity_file_path="/home/arism/datasets/cell_type_similarities.pkl", 
                 cell_type_csv_path="/home/arism/analysis_output/final_combined/final_combined_cell_type_top_values.csv",
                 disease_csv_path="/home/arism/analysis_output/final_combined/final_combined_disease_top_values.csv",
                 tissue_csv_path="/home/arism/analysis_output/final_combined/final_combined_tissue_top_values.csv",
                 pathway_descriptions_path="pathway_descriptions.json"):
        
        # Extraction patterns
        self.cell_type_patterns = [
            r"consists of a ([^,\.]+?)(?:,|\.|$)",
            r"consists of an ([^,\.]+?)(?:,|\.|$)",
        ]
        
        self.tissue_patterns = [
            r"originates from the ([^,\.]+?) of",
            r"located in the ([^,\.]+?)(?:,|\.|$)",
        ]
        
        self.disease_patterns = [
            r"of a ([^,\.]*(?:normal|healthy|diseased|tumor|cancer|carcinoma|lymphoma|leukemia)[^,\.]*)",
            r"from a ([^,\.]*(?:normal|healthy|diseased|tumor|cancer|carcinoma|lymphoma|leukemia)[^,\.]*)",
        ]
        
        self.cleanup_patterns = [
            r"^(a|an)\s+",
            r"\s+cell$",
        ]
        
        # Load lists from CSV files
        self.cell_type_list = self.load_csv_values(cell_type_csv_path, "cell types")
        self.disease_list = self.load_csv_values(disease_csv_path, "diseases") 
        self.tissue_list = self.load_csv_values(tissue_csv_path, "tissues")
        
        # Load pathway descriptions and setup
        self.pathway_data = self.load_pathway_data(pathway_descriptions_path)
        
        # Load ontology similarities
        self.similarities = None
        if similarity_file_path:
            self.load_similarities(similarity_file_path)
    
    def load_csv_values(self, csv_path, data_type):
        """Load values from CSV file"""
        try:
            df = pd.read_csv(csv_path)
            values = df['value'].tolist()
            print(f"Loaded {len(values)} {data_type} from {csv_path}")
            return values
        except Exception as e:
            print(f"Warning: Could not load {data_type} from {csv_path}: {e}")
            return []
    
    def load_pathway_data(self, pathway_descriptions_path):
        """Load pathway descriptions and setup mapping"""
        try:
            with open(pathway_descriptions_path, "r") as f:
                descriptions = json.load(f)
            
            # Use all pathways from the file
            target_pathways = list(descriptions.keys())
            
            # Build description → key mapping
            desc_to_key = {}
            for key, desc in descriptions.items():
                norm_desc = " ".join(desc.lower().split())
                desc_to_key[norm_desc] = key
            
            pathway_to_index = {key: i for i, key in enumerate(target_pathways)}
            
            print(f"Loaded {len(descriptions)} pathway descriptions")
            return {
                'descriptions': descriptions,
                'desc_to_key': desc_to_key,
                'target_pathways': target_pathways,
                'pathway_to_index': pathway_to_index,
                'n_pathways': len(target_pathways)
            }
        except Exception as e:
            print(f"Warning: Could not load pathway data: {e}")
        return None

    def load_similarities(self, similarity_file_path):
        """Load precomputed ontology similarities"""
        try:
            with open(similarity_file_path, 'rb') as f:
                self.similarities = pickle.load(f)
            print(f"Loaded similarities for {len(self.similarities)} cell types")
        except Exception as e:
            print(f"Warning: Could not load similarities: {e}")
            self.similarities = None
    
    def extract_with_candidates(self, text, patterns, candidates_list):
        """Generic extraction with candidate matching"""
        if not candidates_list:
            return self.extract_basic(text, patterns)
        
        match = None
        for pattern in patterns:
            regex_match = re.search(pattern, text, re.IGNORECASE)
            if regex_match:
                match = regex_match
                break
        
        if not match:
            return "unknown"
        
        partial_text = match.group(1).strip()
        start_index = match.start(1)
        search_key = partial_text
        
        # Clean up the search key
        for cleanup_pattern in [r"^(a|an)\s+", r"\s+(cell|cells)$"]:
            search_key = re.sub(cleanup_pattern, "", search_key, flags=re.IGNORECASE).strip()
        
        if not search_key:
            return "unknown"
        
        # Find candidates
        candidates = [
            label for label in candidates_list
            if label.lower().startswith(search_key.lower())
        ]
        
        if not candidates:
            return "unknown"
        
        # Sort by length (longest first)
        candidates.sort(key=len, reverse=True)
        
        # Find exact match
        for candidate in candidates:
            end_index = start_index + len(candidate)
            if end_index <= len(text):
                text_slice = text[start_index:end_index]
                if text_slice.lower() == candidate.lower():
                    return candidate
        
        return "unknown"
    
    def extract_basic(self, text, patterns):
        """Basic extraction without candidates"""
        if not text:
            return "unknown"
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result = match.group(1).strip()
                for cleanup_pattern in self.cleanup_patterns:
                    result = re.sub(cleanup_pattern, "", result, flags=re.IGNORECASE).strip()
                return result.lower()
        
        return "unknown"
    
    def extract_cell_type(self, description):
        """Extract cell type from description"""
        return self.extract_with_candidates(description, self.cell_type_patterns, self.cell_type_list)
    
    def extract_disease(self, description):
        """Extract disease from description"""
        return self.extract_with_candidates(description, self.disease_patterns, self.disease_list)
    
    def extract_tissue(self, description):
        """Extract tissue from description"""
        return self.extract_with_candidates(description, self.tissue_patterns, self.tissue_list)
    
    def extract_pathways(self, text):
        """Extract pathway descriptions from text"""
        part1_start = ". This cell is associated with "
        part2_start = ". Additionally, it involves "

        idx1 = text.find(part1_start)
        if idx1 == -1:
            return None, None

        start_p1 = idx1 + len(part1_start)
        idx2 = text.find(part2_start, start_p1)
        if idx2 == -1:
            return None, None

        pathway1 = text[start_p1:idx2].strip() + '.'
        start_p2 = idx2 + len(part2_start)
        pathway2 = text[start_p2:].strip()
        
        if pathway2.endswith('.'):
            pathway2 = pathway2[:-1].strip()

        return pathway1, pathway2
    
    def find_best_matching_pathway(self, description, threshold=0.9):
        """Find best matching pathway key"""
        if not self.pathway_data:
            return None
        
        norm_desc = " ".join(description.lower().split())
        best_match = None
        best_score = 0.0

        for ref_desc, key in self.pathway_data['desc_to_key'].items():
            score = difflib.SequenceMatcher(None, norm_desc, ref_desc).ratio()
            if score > best_score:
                best_score = score
                best_match = key

        return best_match if best_score >= threshold else None
    
    def pathways_to_vector(self, pathways):
        """Convert pathway keys to binary vector"""
        if not self.pathway_data:
            return np.zeros(0)
        
        vec = np.zeros(self.pathway_data['n_pathways'], dtype=int)
        for p in pathways:
            if p in self.pathway_data['pathway_to_index']:
                vec[self.pathway_data['pathway_to_index'][p]] = 1
        return vec
    
    def normalize_cell_type(self, cell_type):
        """Normalize cell type names for better matching"""
        if not cell_type:
            return "unknown"
        
        normalized = cell_type.lower().strip()
        suffixes_to_remove = ["cell", "cells"]
        for suffix in suffixes_to_remove:
            if normalized.endswith(" " + suffix):
                normalized = normalized[:-len(" " + suffix)]
        
        return normalized
    
    def get_ontology_similarity(self, type1, type2):
        """Get ontology similarity between two cell types"""
        if self.similarities is None:
            return 0.0
        
        type1_norm = self.normalize_cell_type(type1)
        type2_norm = self.normalize_cell_type(type2)
        
        if type1_norm in self.similarities and type2_norm in self.similarities[type1_norm]:
            return self.similarities[type1_norm][type2_norm]
        
        return 0.0
    
    def calculate_ontology_similarity_score(self, predicted_types, target_types):
        """Calculate average ontology similarity between predictions and targets"""
        if not predicted_types or not target_types or self.similarities is None:
            return 0.0
        
        similarities = []
        for pred, target in zip(predicted_types, target_types):
            sim = self.get_ontology_similarity(pred, target)
            similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    

def calculate_comprehensive_metrics(predicted_texts, target_texts, 
                                  similarity_file_path="/home/arism/datasets/cell_type_similarities.pkl",
                                  cell_type_csv_path="/home/arism/analysis_output/final_combined/final_combined_cell_type_top_values.csv",
                                  disease_csv_path="/home/arism/analysis_output/final_combined/final_combined_disease_top_values.csv", 
                                  tissue_csv_path="/home/arism/analysis_output/final_combined/final_combined_tissue_top_values.csv",
                                  pathway_descriptions_path="pathway_descriptions.json"):
    """Calculate comprehensive metrics for all extracted components"""
    
    extractor = CellTypeExtractor(similarity_file_path, cell_type_csv_path, disease_csv_path, tissue_csv_path, pathway_descriptions_path)
    
    # Storage for all extractions
    pred_cell_types, target_cell_types = [], []
    pred_diseases, target_diseases = [], []
    pred_tissues, target_tissues = [], []
    pred_pathway_vectors, target_pathway_vectors = [], []
    
    for pred_text, target_text in zip(predicted_texts, target_texts):
        # Extract cell types
        pred_cell = extractor.extract_cell_type(pred_text)
        target_cell = extractor.extract_cell_type(target_text)
        pred_cell_types.append(pred_cell)
        target_cell_types.append(target_cell)
        
        # Extract diseases
        pred_disease = extractor.extract_disease(pred_text)
        target_disease = extractor.extract_disease(target_text)
        pred_diseases.append(pred_disease)
        target_diseases.append(target_disease)
        
        # Extract tissues
        pred_tissue = extractor.extract_tissue(pred_text)
        target_tissue = extractor.extract_tissue(target_text)
        pred_tissues.append(pred_tissue)
        target_tissues.append(target_tissue)
        
        # Extract pathways
        if extractor.pathway_data:
            pred_p1, pred_p2 = extractor.extract_pathways(pred_text)
            target_p1, target_p2 = extractor.extract_pathways(target_text)
            
            # Map to pathway keys
            pred_keys = []
            if pred_p1:
                key = extractor.find_best_matching_pathway(pred_p1)
                if key: pred_keys.append(key)
            if pred_p2:
                key = extractor.find_best_matching_pathway(pred_p2)
                if key: pred_keys.append(key)
            
            target_keys = []
            if target_p1:
                key = extractor.find_best_matching_pathway(target_p1)
                if key: target_keys.append(key)
            if target_p2:
                key = extractor.find_best_matching_pathway(target_p2)
                if key: target_keys.append(key)
            
            # Convert to vectors
            pred_vec = extractor.pathways_to_vector(pred_keys)
            target_vec = extractor.pathways_to_vector(target_keys)
            pred_pathway_vectors.append(pred_vec)
            target_pathway_vectors.append(target_vec)
    
    # Calculate metrics for each component
    def calc_basic_metrics(pred_list, target_list, component_name):
        if not pred_list or not target_list:
            return {}
        
        exact_matches = sum(1 for t, p in zip(target_list, pred_list) if t == p)
        accuracy = exact_matches / len(target_list)
        
        unique_labels = list(set(target_list + pred_list))
        if len(unique_labels) > 1:
            f1 = f1_score(target_list, pred_list, labels=unique_labels, average='weighted', zero_division=0)
            precision = precision_score(target_list, pred_list, labels=unique_labels, average='macro', zero_division=0)
            recall = recall_score(target_list, pred_list, labels=unique_labels, average='macro', zero_division=0)
        else:
            f1 = precision = recall = accuracy
        
        return {
            f'{component_name}_accuracy': accuracy,
            f'{component_name}_f1': f1,
            f'{component_name}_precision': precision,
            f'{component_name}_recall': recall
        }
    
    # Compile all metrics
    metrics = {}
    metrics.update(calc_basic_metrics(pred_cell_types, target_cell_types, 'cell_type'))
    metrics.update(calc_basic_metrics(pred_diseases, target_diseases, 'disease'))
    metrics.update(calc_basic_metrics(pred_tissues, target_tissues, 'tissue'))
    
    # Ontology similarity for cell types
    similarity_score = extractor.calculate_ontology_similarity_score(pred_cell_types, target_cell_types)
    metrics['ontology_similarity_score'] = similarity_score
    
    # Pathway metrics
    if pred_pathway_vectors and target_pathway_vectors and extractor.pathway_data:
        pred_path_array = np.array(pred_pathway_vectors)
        target_path_array = np.array(target_pathway_vectors)
        
        # Subset accuracy (exact match)
        subset_acc = accuracy_score(target_path_array, pred_path_array)
        jaccard_acc = jaccard_score(target_path_array, pred_path_array, average='samples', zero_division=0)
        
        # F1 scores
        weighted_f1 = f1_score(target_path_array, pred_path_array, average="weighted", zero_division=0)
        macro_f1 = f1_score(target_path_array, pred_path_array, average="macro", zero_division=0)
        micro_f1 = f1_score(target_path_array, pred_path_array, average="micro", zero_division=0)
        
        
        metrics.update({
            'pathway_subset_accuracy': subset_acc,
            'pathway_jaccard_accuracy': jaccard_acc,
            'pathway_weighted_f1': weighted_f1,
            'pathway_macro_f1': macro_f1,
            'pathway_micro_f1': micro_f1,
        })
    
    metrics['total_samples'] = len(predicted_texts)
    
    return metrics