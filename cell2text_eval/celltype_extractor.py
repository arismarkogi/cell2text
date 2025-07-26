from sklearn.metrics import f1_score, precision_score, recall_score
import re
import pickle
import numpy as np


class CellTypeExtractor:
    """Extract cell types and calculate ontology-aware metrics"""
    
    def __init__(self, similarity_file_path="/home/arism/datasets/cell_type_similarities.pkl"):
        # Original patterns
        self.cell_type_patterns = [
            r"consists of a ([^,\.]+?)(?:,|\.|$)",
            r"consists of an ([^,\.]+?)(?:,|\.|$)",
        ]
        
        self.cleanup_patterns = [
            r"^(a|an)\s+",
            r"\s+cell$",
        ]
        
        # Load ontology similarities
        self.similarities = None
        if similarity_file_path:
            self.load_similarities(similarity_file_path)
    
    def load_similarities(self, similarity_file_path):
        """Load precomputed ontology similarities"""
        try:
            with open(similarity_file_path, 'rb') as f:
                self.similarities = pickle.load(f)
            print(f"Loaded similarities for {len(self.similarities)} cell types")
        except Exception as e:
            print(f"Warning: Could not load similarities: {e}")
            self.similarities = None
    
    def extract_cell_type(self, description):
        """Extract the main cell type from a description"""
        if not description:
            return "unknown"
            
        for pattern in self.cell_type_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                cell_type = match.group(1).strip()
                
                for cleanup_pattern in self.cleanup_patterns:
                    cell_type = re.sub(cleanup_pattern, "", cell_type, flags=re.IGNORECASE).strip()
                
                return cell_type.lower()
        
        return "unknown"
    
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
    
    def calculate_ontology_aware_accuracy(self, predicted_types, target_types, similarity_threshold=2.0):
        """Calculate accuracy considering ontology similarities"""
        if not predicted_types or not target_types:
            return 0.0
        
        if self.similarities is None:
            # Fallback to exact match
            return sum(1 for p, t in zip(predicted_types, target_types) if p == t) / len(target_types)
        
        correct = 0
        for pred, target in zip(predicted_types, target_types):
            if pred == target:
                correct += 1
            else:
                # Check if they're similar enough in ontology
                similarity = self.get_ontology_similarity(pred, target)
                if similarity >= similarity_threshold:
                    correct += 1
        
        return correct / len(target_types)
    
    def calculate_ontology_similarity_score(self, predicted_types, target_types):
        """Calculate average ontology similarity between predictions and targets"""
        if not predicted_types or not target_types or self.similarities is None:
            return 0.0
        
        similarities = []
        for pred, target in zip(predicted_types, target_types):
            sim = self.get_ontology_similarity(pred, target)
            similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0


def calculate_cell_type_metrics(predicted_types, target_types, similarity_file_path="/home/arism/datasets/cell_type_similarities.pkl", 
                                       global_matches=None, global_total=None):
    """Calculate both traditional and ontology-aware metrics"""
    
    extractor = CellTypeExtractor(similarity_file_path)
    
    # Handle distributed case
    if global_matches is not None and global_total is not None:
        accuracy = global_matches / global_total if global_total > 0 else 0
        return {
            'accuracy': accuracy,
            'f1': accuracy,
            'precision': accuracy,
            'recall': accuracy,
            'ontology_aware_accuracy': accuracy,  # Approximation
            'ontology_similarity_score': 0.0,
            'total_samples': global_total
        }
    
    # Traditional metrics
    exact_matches = sum(1 for t, p in zip(target_types, predicted_types) if t == p)
    accuracy = exact_matches / len(target_types) if target_types else 0
    
    unique_labels = list(set(target_types + predicted_types))
    if len(unique_labels) > 1:
        f1 = f1_score(target_types, predicted_types, labels=unique_labels, average='macro', zero_division=0)
        precision = precision_score(target_types, predicted_types, labels=unique_labels, average='macro', zero_division=0)
        recall = recall_score(target_types, predicted_types, labels=unique_labels, average='macro', zero_division=0)
    else:
        f1 = precision = recall = accuracy
    
    # Ontology-aware metrics
    ontology_accuracy = extractor.calculate_ontology_aware_accuracy(predicted_types, target_types)
    similarity_score = extractor.calculate_ontology_similarity_score(predicted_types, target_types)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'ontology_aware_accuracy': ontology_accuracy,
        'ontology_similarity_score': similarity_score,
        'total_samples': len(target_types)
    }
