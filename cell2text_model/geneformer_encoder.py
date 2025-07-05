import torch
import logging

from transformers import BertForMaskedLM, BertConfig
from typing import Optional, Union, Tuple
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.configuration_utils import PretrainedConfig
from Geneformer.geneformer.emb_extractor import get_embs
from Geneformer.geneformer import perturber_utils as pu
import pickle
from datasets import Dataset
import numpy as np



logger = logging.getLogger(__name__)

# Set as constants here, so they are available later
PAD_TOKEN_ID = 0
MODEL_INPUT_SIZE = 4096



class GeneformerConfig(PretrainedConfig):

    def __init__(
        self,
        emb_mode="gene",
        max_ncells=1000,  
        emb_layer=-1,
        emb_label=None,
        summary_stat=None,
        forward_batch_size=100,
        nproc=4,
        token_dictionary_path="/home/arismarkog/Desktop/cell2text/Geneformer/geneformer/token_dictionary_gc95M.pkl",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.emb_mode = emb_mode
        self.max_ncells = max_ncells
        self.emb_layer = emb_layer
        self.emb_label = emb_label
        self.summary_stat = summary_stat
        self.forward_batch_size = forward_batch_size
        self.nproc = nproc
        self.token_dictionary_path = token_dictionary_path



class GeneformerModel(
    PreTrainedModel
):  
    config_class = GeneformerConfig
    base_model_prefix = "geneformer_model"
    is_parallelizable = False  
    main_input_name = "expression_tokens"


    def __init__(self, config: GeneformerConfig):
        super().__init__(config)
        self.config = config

    
        token_dictionary_file = config.token_dictionary_path
        self.token_dictionary_file = token_dictionary_file

        with open(token_dictionary_file, "rb") as f:
            self.gene_token_dict = pickle.load(f)
            self.pad_token_id = self.gene_token_dict.get("<pad>")

        self.token_gene_dict = {v: k for k, v in self.gene_token_dict.items()}
        

        # Don't initialize geneformer_model in __init__ - let it be None
        # It will be properly initialized either via from_pretrained() or load_state_dict()
        self.geneformer_model = None

    def forward(
        self,
        expression_tokens: torch.Tensor,
        expression_token_lengths: torch.Tensor,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        
        # Check if geneformer_model is loaded
        if self.geneformer_model is None:
            raise RuntimeError("Model not loaded. Please use GeneformerModel.from_pretrained() or load_from_state_dict() to load the model.")

        layer_to_quant = pu.quant_layers(self.geneformer_model) + self.config.emb_layer
        
        # Create dataset from numpy arrays directly
        filtered_input_data = Dataset.from_dict({
            'input_ids': expression_tokens,
            'length': expression_token_lengths
        })
        
        embs = get_embs(
            model=self.geneformer_model,
            filtered_input_data=filtered_input_data,
            emb_mode=self.config.emb_mode,
            layer_to_quant=layer_to_quant,
            pad_token_id=PAD_TOKEN_ID,
            forward_batch_size=self.config.forward_batch_size,
            token_gene_dict=self.token_gene_dict,
            summary_stat=self.config.summary_stat
        )


        return embs


    @classmethod
    def from_pretrained(
          cls, pretrained_model_name_or_path: str, *args, **kwargs
    ) -> PreTrainedModel:
        
        
        if "config" in kwargs:
            config = kwargs.pop("config")
            if isinstance(config, dict):
                config = GeneformerConfig(**config)
            elif not isinstance(config, GeneformerConfig):
                raise ValueError(
                    "Parameter `config` must be a dictionary or an instance of `GeneformerConfig`."
                )
        else:
            config = GeneformerConfig()
            logger.warning(
                "No configuration provided. Using default configuration from checkpoint."
            )

        model = cls(config, *args, **kwargs)




        bert_config = BertConfig.from_pretrained(pretrained_model_name_or_path)
        bert_config.output_hidden_states = True 
        
        model.geneformer_model = BertForMaskedLM.from_pretrained(
            pretrained_model_name_or_path,
            config=bert_config,
        )

        # Print model architecture
        print("Geneformer Architecture:")
        print(model.geneformer_model)


        return model
    
    def load_from_state_dict(self, state_dict, strict=True):
        """
        Load geneformer model from a state dict (for loading from complete checkpoints)
        """
        # Initialize the BERT model if it doesn't exist
        if self.geneformer_model is None:
            from transformers import BertConfig
            bert_config = BertConfig()
            bert_config.output_hidden_states = True
            self.geneformer_model = BertForMaskedLM(bert_config)
        
        # Extract only the geneformer-related keys from the state dict
        geneformer_state_dict = {}
        prefix = "cell_encoder.geneformer_model."
        
        for key, value in state_dict.items():
            if key.startswith(prefix):
                # Remove the prefix to get the actual model key
                model_key = key[len(prefix):]
                geneformer_state_dict[model_key] = value
        
        if geneformer_state_dict:
            # Load the state dict into the geneformer model
            missing_keys, unexpected_keys = self.geneformer_model.load_state_dict(
                geneformer_state_dict, strict=strict
            )
            
            if missing_keys:
                print(f"Missing keys in geneformer model: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys in geneformer model: {unexpected_keys}")
                
            print("Geneformer model loaded from state dict successfully!")
        else:
            print("Warning: No geneformer weights found in state dict")