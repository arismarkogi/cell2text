import torch.nn as nn
from transformers import BertForSequenceClassification
import warnings
import torch
warnings.filterwarnings('ignore')

class GeneformerCellTypeClassifier(nn.Module):
    """Geneformer-based cell type classifier for single-label classification"""
    
    def __init__(self, geneformer_model_path, num_cell_types, freeze_geneformer=True, dropout_rate=0.1):
        super().__init__()
        self.freeze_geneformer = freeze_geneformer
        self.num_cell_types = num_cell_types
        self.geneformer_model_path = geneformer_model_path
        
        # Load Geneformer model for single-label classification
        self.model = BertForSequenceClassification.from_pretrained(
            geneformer_model_path,
            num_labels=num_cell_types,
            problem_type="single_label_classification"  # Changed from multi_label
        )
        
        # Optionally freeze Geneformer parameters
        if freeze_geneformer:
            for param in self.model.bert.parameters():
                param.requires_grad = False
                
        # Add dropout to the classifier head for regularization
        if hasattr(self.model, 'classifier'):
            # Replace the classifier with one that includes dropout
            hidden_size = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, num_cell_types)
            )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits  # Shape: (batch_size, num_cell_types)
    
    def get_embeddings(self, input_ids, attention_mask):
        """Get hidden embeddings from the model (useful for analysis)"""
        with torch.no_grad():
            outputs = self.model.bert(input_ids=input_ids, attention_mask=attention_mask)
            # Return the [CLS] token embedding
            return outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)