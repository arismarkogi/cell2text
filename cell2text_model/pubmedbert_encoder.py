import torch
import logging
from transformers import AutoModel, AutoConfig
from typing import Optional, Union, Tuple
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

logger = logging.getLogger(__name__)

# Set as constants here, so they are available throughout the file
PAD_TOKEN_ID = 0
MODEL_INPUT_SIZE = 512  # PubMedBERT typically uses smaller sequence lengths than Geneformer

class PubMedBertEncoder(PreTrainedModel):

    def __init__(
        self,
        num_classes=0,
        emb_mode="cls",
        hidden_size=768,  # PubMedBERT default hidden size
        max_texts=200,
        emb_layer=-1,
        emb_label=["article_id", "publication_type", "medical_domain"],
        forward_batch_size=-1,
        nproc=4,
        summary_stat=None,
        **kwargs
    ):
        # Validate emb_mode
        valid_emb_modes = ["cls", "mean", "token", "pooled"]
        assert emb_mode in valid_emb_modes, f"Invalid emb_mode: {emb_mode}, must be one of {valid_emb_modes}"
        
        # Initialize with direct configuration
        config_dict = {
            "num_classes": num_classes,
            "emb_mode": emb_mode,
            "hidden_size": hidden_size,
            "max_texts": max_texts,
            "emb_layer": emb_layer,
            "emb_label": emb_label,
            "forward_batch_size": forward_batch_size,
            "nproc": nproc,
            "summary_stat": summary_stat,
        }
        config_dict.update(kwargs)

        # Create a configuration object
        pubmed_config = AutoConfig.from_pretrained(
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            output_hidden_states=True,
            output_attentions=False
        )
        
        # Pass configuration to parent
        super().__init__(pubmed_config)

        # Store configuration attributes directly on the model
        for key, value in config_dict.items():
            setattr(self, key, value)
        
        # Initialize the PubMedBERT model
        self.pubmedbert_model = AutoModel.from_pretrained(
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            config=pubmed_config
        )
        
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_tokens: torch.Tensor,
        input_token_lengths: torch.Tensor,
        text_data=None,  # ignored, but kept for API compatibility
        expression_value=None,  # ignored, but kept for API compatibility
        attention_padding_mask=None,  # ignored, but kept for API compatibility
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        """
        Process text data through the PubMedBERT model to generate embeddings.
        """
        assert not return_dict, f"No support for return_dict={return_dict}"
        
        # Create attention mask from token lengths
        batch_size, seq_len = input_tokens.shape
        attention_mask = torch.zeros(batch_size, seq_len, device=input_tokens.device)
        
        for i, length in enumerate(input_token_lengths):
            attention_mask[i, :length] = 1
        
        # Run PubMedBERT model
        outputs = self.pubmedbert_model(
            input_ids=input_tokens,
            attention_mask=attention_mask.bool(),
            return_dict=True
        )
        
        # Get hidden states from the specified layer
        hidden_states = outputs.hidden_states
        layer_idx = self.emb_layer
        if layer_idx < 0:
            layer_idx = len(hidden_states) + layer_idx
        
        target_hidden_state = hidden_states[layer_idx]
        
        # Get embeddings based on emb_mode
        if self.emb_mode == "cls":
            # Get CLS token embedding (first token)
            embs = target_hidden_state[:, 0]
        elif self.emb_mode == "pooled":
            # Use the pooled output directly
            embs = outputs.pooler_output
        elif self.emb_mode == "mean":
            # Mean of non-padding tokens
            non_padding_mask = (input_tokens != PAD_TOKEN_ID).float().unsqueeze(-1)
            sum_embeddings = torch.sum(target_hidden_state * non_padding_mask, dim=1)
            token_counts = torch.sum(non_padding_mask, dim=1)
            embs = sum_embeddings / token_counts
        elif self.emb_mode == "token":
            # Return all token embeddings (may need further processing downstream)
            embs = target_hidden_state
        else:
            raise ValueError(f"Unsupported emb_mode: {self.emb_mode}")
        
        return (outputs, embs)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, *args, **kwargs
    ) -> PreTrainedModel:
        # Extract configuration parameters from kwargs or use defaults
        config_params = {}
        config_keys = [
            "num_classes", "emb_mode", "hidden_size", "max_texts",
            "emb_layer", "emb_label", "forward_batch_size", "nproc", "summary_stat"
        ]
        
        for key in config_keys:
            if key in kwargs:
                config_params[key] = kwargs.pop(key)
        
        # Create the model with extracted or default configuration
        model = cls(**config_params)
        
        # If a custom model path is provided, load it instead of the default PubMedBERT
        if pretrained_model_name_or_path != "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext":
            model.pubmedbert_model = AutoModel.from_pretrained(
                pretrained_model_name_or_path,
                output_hidden_states=True,
                output_attentions=False,
            )
            
        return model