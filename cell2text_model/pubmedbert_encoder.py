import torch
import logging
from transformers import AutoModel, AutoConfig
from typing import Optional, Union, Tuple
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from cell2text_model.configuration import Cell2TextConfig

logger = logging.getLogger(__name__)

# Set as constants here, so they are available throughout the file
PAD_TOKEN_ID = 0
MODEL_INPUT_SIZE = 512

class PubMedBertEncoder(PreTrainedModel):

    def __init__(
        self,
        num_classes=0,
        emb_mode="cls",
        emb_layer=-1,
        forward_batch_size=-1,
        nproc=4,
        **kwargs
    ):
        # Validate emb_mode
        valid_emb_modes = ["cls", "mean", "token", "pooled"]
        assert emb_mode in valid_emb_modes, f"Invalid emb_mode: {emb_mode}, must be one of {valid_emb_modes}"
        
        # Initialize with direct configuration
        config_dict = {
            "num_classes": num_classes,
            "emb_mode": emb_mode,
            "emb_layer": emb_layer,
            "forward_batch_size": forward_batch_size,
            "nproc": nproc,
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

        self.hidden_size = self.pubmedbert_model.config.hidden_size
        
        if kwargs:
            logger.warning(f"Unused kwargs in from_pretrained: {list(kwargs.keys())}")


        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_tokens: torch.Tensor,
        input_token_lengths: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        """
        Forward pass through PubMedBERT to obtain embeddings.
        """
        # If no attention_mask is provided, create it using input_token_lengths
        if attention_mask is None:
            attention_mask = torch.arange(input_tokens.size(1), device=input_tokens.device).unsqueeze(0) < input_token_lengths.unsqueeze(1)

        # Run PubMedBERT
        outputs = self.pubmedbert_model(
            input_ids=input_tokens,
            attention_mask=attention_mask,
            return_dict=return_dict
        )

        hidden_states = outputs.hidden_states
        layer_idx = self.emb_layer if self.emb_layer >= 0 else len(hidden_states) + self.emb_layer
        target_hidden = hidden_states[layer_idx]

        # Choose embedding strategy
        if self.emb_mode == "cls":
            embs = target_hidden[:, 0]
        elif self.emb_mode == "pooled":
            embs = outputs.pooler_output
        elif self.emb_mode == "mean":
            # Use the attention mask to average over valid tokens
            mask = attention_mask.unsqueeze(-1).float()
            sum_hidden = (target_hidden * mask).sum(dim=1)
            lengths = mask.sum(dim=1)
            embs = sum_hidden / lengths
        elif self.emb_mode == "token":
            embs = target_hidden
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
            "num_classes", "emb_mode", "hidden_size", 
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