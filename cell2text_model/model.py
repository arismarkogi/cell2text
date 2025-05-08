import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from typing import Optional, Dict, Any

from .projectors import LinearProjectionLayer, MLPProjectionLayer
from .geneformer_encoder import GeneformerModel, GeneformerConfig
from .llama_decoder import Cell2TextLlamaModel, Cell2TextLlamaConfig


class Cell2TextModel(PreTrainedModel):
    config_class = PretrainedConfig
    base_model_prefix = "cell2text"
    
    def __init__(self, config):
        super().__init__(config)
        
        self.geneformer_config = GeneformerConfig(
            emb_mode=config.emb_mode if hasattr(config, "emb_mode") else "cell",
            max_ncells=config.max_ncells if hasattr(config, "max_ncells") else 1000,
            emb_layer=config.emb_layer if hasattr(config, "emb_layer") else -1,
            emb_label=config.emb_label if hasattr(config, "emb_label") else None,
            nproc=config.nproc if hasattr(config, "nproc") else -1,
            forward_batch_size=config.forward_batch_size if hasattr(config, "forward_batch_size") else 100,
            summary_stat=config.summary_stat if hasattr(config, "summary_stat") else None,
            token_dictionary_path=config.token_dictionary_path if hasattr(config, "token_dictionary_path") else "/home/arismarkog/Desktop/cell2text/Geneformer/geneformer/token_dictionary_gc95M.pkl"
        )

        self.cell2text_llama_config = Cell2TextLlamaConfig(
            max_length=config.max_length if hasattr(config, "max_length") else 100,
            num_beams=config.num_beams if hasattr(config, "num_beams") else 4,
            early_stopping=config.early_stopping if hasattr(config, "early_stopping") else True,
            no_repeat_ngram_size=config.no_repeat_ngram_size if hasattr(config, "no_repeat_ngram_size") else 3,
            temperature=config.temperature if hasattr(config, "temperature") else 1.0,
            top_p=config.top_p if hasattr(config, "top_p") else 1.0
        )
        
       
         
        # Store cell encoder hidden size for embedding projection
        self.cell_encoder_hidden_size = config.cell_encoder_hidden_size
        self.decoder_hidden_size = config.decoder_hidden_size
        self.mlp_hidden_size = config.mlp_hidden_size
        self.mlp_dropout = config.mlp_dropout

        # Initialize models
        self.cell_encoder = None  # Will be loaded in warm_up
        self.decoder = None       # Will be loaded in warm_up

        self.cell_to_embedding = MLPProjectionLayer(input_dim=self.cell_encoder_hidden_size, hidden_dim=self.mlp_hidden_size, output_dim=self.decoder_hidden_size,dropout_prob=self.mlp_dropout, bias=True)
        
        self.config = config
        
        # print("Trainable parameters using named_parameters():")
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(f"{name}: {param.numel()}")
    
    def get_cell_encoder(self):
        return self.cell_encoder
    
    def get_decoder(self):
        return self.decoder
    
    def warm_up(self):
        """
        Load pre-trained weights for the model components
        """

        
        # Load cell encoder
        self.cell_encoder = GeneformerModel.from_pretrained(
            pretrained_model_name_or_path=self.config.geneformer_path, 
            config=self.geneformer_config
        )
        
        # Create Llama decoder with custom config
        self.decoder = Cell2TextLlamaModel.from_pretrained(
            pretrained_model_name_or_path=self.config.decoder_model_name_or_path,
            config=self.cell2text_llama_config
        )
    
    def forward(
        self,
        expression_tokens: Optional[torch.LongTensor] = None,
        expression_token_lengths: Optional[torch.LongTensor] = None,
        text_input_ids: Optional[torch.LongTensor] = None,
        text_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = False,
        **kwargs
    ):
        """
        Forward pass through the entire model
        """
        # Process cell expression data with Geneformer
        cell_embeddings = self.cell_encoder(
            expression_tokens=expression_tokens,
            expression_token_lengths=expression_token_lengths,
            return_dict=True
        )
        
         
        
        cell_embeddings = self.cell_to_embedding(cell_embeddings)

        # Forward pass through decoder
        decoder_outputs = self.decoder(
            cell_embeddings=cell_embeddings,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        
        return decoder_outputs
    
    @torch.no_grad()
    def generate_cell_description(
        self,
        expression_tokens: Optional[torch.LongTensor] = None,
        expression_token_lengths: Optional[torch.LongTensor] = None,
        device='cpu',
        **generate_kwargs
    ):
        """
        Generate text description for cell expression data
        """
        if expression_tokens is None:
            raise ValueError("You need to provide expression_tokens")
        
        # Process inputs
        expression_tokens = expression_tokens.to(device)
        expression_token_lengths = expression_token_lengths.to(device)
        
        # Get cell embeddings from Geneformer
        cell_embeddings = self.cell_encoder(
            expression_tokens=expression_tokens,
            expression_token_lengths=expression_token_lengths,
            return_dict=True
        )
        

        cell_embeddings = self.cell_to_embedding(cell_embeddings)
        
        # Generate text description using the decoder
        return self.decoder.generate_cell_description(
            cell_embeddings=cell_embeddings,
            device=device,
            **generate_kwargs
        )


