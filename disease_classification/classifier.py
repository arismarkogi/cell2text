import torch.nn as nn
from transformers import BertForSequenceClassification

class GeneformerDiseaseClassifier(nn.Module):
    def __init__(self, geneformer_model_path, num_diseases, freeze_geneformer=True):
        super().__init__()
        self.freeze_geneformer = freeze_geneformer
        self.num_diseases = num_diseases
        self.geneformer_model_path = geneformer_model_path
        
        self.model = BertForSequenceClassification.from_pretrained(
            geneformer_model_path,
            num_labels=num_diseases,
            problem_type="single_label_classification"  # ← CHANGED
        )
            
        if freeze_geneformer:
            for param in self.model.bert.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits  # Shape: (batch_size, num_diseases)
