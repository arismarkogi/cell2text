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
        

        layer_to_quant = pu.quant_layers(self.geneformer_model) + self.config.emb_layer
        



        # Might need to comment this out when run on GPU

        # Convert to numpy arrays instead of lists (more efficient)
        if expression_tokens.is_cuda:
            tokens_np = expression_tokens.cpu().numpy()
            lengths_np = expression_token_lengths.cpu().numpy()
        else:
            tokens_np = expression_tokens.numpy()
            lengths_np = expression_token_lengths.numpy()

        
        # this might not be efficient
        
        # Create dataset from numpy arrays directly
        filtered_input_data = Dataset.from_dict({
            'input_ids': tokens_np,
            'length': lengths_np
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



        print("Shape of embs:", embs.shape)

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
    
    