from sklearn.metrics import f1_score, precision_score, recall_score
import re


class CellTypeExtractor:
    """Extract cell types from generated descriptions for evaluation"""
    
    def __init__(self):
        # Common cell type patterns that appear after "consists of a"
        self.cell_type_patterns = [
            r"consists of a ([^,\.]+?)(?:,|\.|$)",
            r"consists of an ([^,\.]+?)(?:,|\.|$)",
        ]
        
        # Clean up extracted cell types
        self.cleanup_patterns = [
            r"^(a|an)\s+",  # Remove leading articles
            r"\s+cell$",    # Remove trailing "cell"
        ]
    
    def extract_cell_type(self, description: str) -> str:
        """Extract the main cell type from a description"""
        if not description:
            return "unknown"
            
        # Try each pattern
        for pattern in self.cell_type_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                cell_type = match.group(1).strip()
                
                # Clean up the extracted cell type
                for cleanup_pattern in self.cleanup_patterns:
                    cell_type = re.sub(cleanup_pattern, "", cell_type, flags=re.IGNORECASE).strip()
                
                return cell_type.lower()
        
        return "unknown"
    
    def normalize_cell_type(self, cell_type: str) -> str:
        """Normalize cell type names for better matching"""
        if not cell_type:
            return "unknown"
            
        # Convert to lowercase and clean
        normalized = cell_type.lower().strip()
        
        # Remove common prefixes/suffixes that might cause mismatches
        suffixes_to_remove = ["cell", "cells"]
        
        for suffix in suffixes_to_remove:
            if normalized.endswith(" " + suffix):
                normalized = normalized[:-len(" " + suffix)]
        
        return normalized


def calculate_cell_type_metrics(predicted_types, target_types, global_matches=None, global_total=None):
    """Calculate precision, recall, and F1 for cell type extraction"""
    
    # If we have global matches from DDP reduction, use those for accuracy
    if global_matches is not None and global_total is not None:
        accuracy = global_matches / global_total if global_total > 0 else 0
        # For distributed case, we can't easily calculate F1/precision/recall without all data
        # So we'll use accuracy as approximation or calculate locally
        return {
            'accuracy': accuracy,
            'f1': accuracy,  # Approximation
            'precision': accuracy,  # Approximation  
            'recall': accuracy,  # Approximation
            'total_samples': global_total
        }
    
    # Local calculation (single GPU or main process with all data)
    exact_matches = sum(1 for t, p in zip(target_types, predicted_types) if t == p)
    accuracy = exact_matches / len(target_types) if target_types else 0
    
    # Calculate macro F1, precision, recall
    unique_labels = list(set(target_types + predicted_types))
    if len(unique_labels) > 1:
        f1 = f1_score(target_types, predicted_types, labels=unique_labels, average='macro', zero_division=0)
        precision = precision_score(target_types, predicted_types, labels=unique_labels, average='macro', zero_division=0)
        recall = recall_score(target_types, predicted_types, labels=unique_labels, average='macro', zero_division=0)
    else:
        f1 = precision = recall = accuracy
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'total_samples': len(target_types)
    }