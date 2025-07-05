import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from typing import Optional, Dict, Any
import os
from safetensors.torch import load_file

from .projectors import  MLPProjectionLayer, PerceiverIO
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

        
        # Store configuration parameters
        self.cell_encoder_hidden_size = config.cell_encoder_hidden_size
        self.decoder_hidden_size = config.decoder_hidden_size
        self.mlp_hidden_size = config.mlp_hidden_size
        self.mlp_dropout = config.mlp_dropout
        self.top_k = config.top_k
        

        # Initialize all components in __init__
        self.cell_encoder = GeneformerModel(self.geneformer_config)

        self.decoder = Cell2TextLlamaModel(self.cell2text_llama_config)
        
        self.projector = config.projector

        if self.projector == "mlp":

            self.cell_to_embedding = MLPProjectionLayer(
                input_dim=self.cell_encoder_hidden_size, 
                hidden_dim=self.mlp_hidden_size, 
                output_dim=self.decoder_hidden_size,
                dropout_prob=self.mlp_dropout, 
                bias=True
            )

        elif self.projector == "perceiver":
            self.cell_to_embedding = PerceiverIO(
                input_dim=self.cell_encoder_hidden_size,
                output_dim=self.decoder_hidden_size,
                num_latents=config.num_latents,
                num_cross_attn_layers=config.perceiver_cross_attn_layers,
                num_heads=config.perceiver_num_heads,
                ff_mult=config.ff_mult,
                dropout=config.perceiver_dropout,
                use_position_encoding=config.use_position_encoding
            )
        
        self.config = config
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, config=None, **kwargs):
        """
        Simplified loading for complete model checkpoints stored as .bin files.
        """
        # Load config
        if config is None:
            config_path = os.path.join(pretrained_model_name_or_path, "config.json")
            if os.path.exists(config_path):
                config = PretrainedConfig.from_json_file(config_path)
            else:
                raise ValueError("No config provided and no config.json found")
        
        # Initialize model
        model = cls(config)
        
        # Load checkpoint 
        checkpoint_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        if os.path.exists(checkpoint_path):
            print(f"Loading complete model from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            
            # Handle both direct state_dict and nested checkpoint formats
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            
            # Load the state dict
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Warning: Missing keys in checkpoint: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
                
            print("Model loaded successfully!")
        else:
            raise FileNotFoundError(f"No pytorch_model.bin found at {checkpoint_path}")
        
        return model
    
    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Save the complete model as a single .bin file.
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save config
        self.config.save_pretrained(save_directory)
        
        # Save complete model state dict
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        print(f"Model saved to {save_directory}")
    

    def process_encoder_outputs(self, cell_embeddings):
        """
        Utility to process encoder output: truncate top_k and project to decoder hidden size.
        """
        if self.projector == "mlp":
            # Truncate to top_k tokens
            if self.top_k < cell_embeddings.shape[1]:
                cell_embeddings = cell_embeddings[:, :self.top_k, :]
            
        # Project to decoder hidden size
        cell_embeddings = self.cell_to_embedding(cell_embeddings)
        
        return cell_embeddings

    def warm_up(self):
        """
        DEPRECATED: Use from_pretrained() or load_pretrained_weights() instead.
        This method is kept for backward compatibility.
        """
        print("Warning: warm_up() is deprecated. Use from_pretrained() or load_pretrained_weights() instead.")
        if hasattr(self.config, 'pretrained_model_path'):
            self.load_pretrained_weights(self.config.pretrained_model_path)
        else:
            self._load_individual_components("")
    
    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Save the model to a directory.
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save config
        self.config.save_pretrained(save_directory)
        
        # Save model weights
        model_path = os.path.join(save_directory, "model.safetensors")
        
        # Collect all state dicts with proper prefixes
        full_state_dict = {}
        
        # Add cell encoder state
        for key, value in self.cell_encoder.state_dict().items():
            full_state_dict[f"cell_encoder.{key}"] = value
        
        # Add decoder state
        for key, value in self.decoder.state_dict().items():
            full_state_dict[f"decoder.{key}"] = value
        
        # Add projector state
        for key, value in self.cell_to_embedding.state_dict().items():
            full_state_dict[f"cell_to_embedding.{key}"] = value
        
        # Save as safetensors
        from safetensors.torch import save_file
        save_file(full_state_dict, model_path)
        
        print(f"Model saved to {save_directory}")
    
    def get_cell_encoder(self):
        return self.cell_encoder
    
    def get_decoder(self):
        return self.decoder
    
    def forward(
        self,
        expression_tokens: Optional[torch.LongTensor] = None,
        expression_token_lengths: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,  
        attention_mask: Optional[torch.FloatTensor] = None,  
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

        
                    
        cell_embeddings = self.process_encoder_outputs(cell_embeddings)

        # Forward pass through decoder - using correct parameter names
        decoder_outputs = self.decoder(
            input_ids=input_ids,  
            attention_mask=attention_mask, 
            cell_embeddings=cell_embeddings,
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
        inputs: Optional[torch.LongTensor] = None,  # tokenized prompt (input_ids)
        attention_mask: Optional[torch.LongTensor] = None,  # attention mask for prompt
        device='cpu',
        **generate_kwargs
    ):
        """
        Generate text description for cell expression data using tokenized prompt.
        
        Args:
            expression_tokens: Gene expression tokens
            expression_token_lengths: Lengths of expression token sequences
            inputs: Tokenized prompt (input_ids)
            attention_mask: Attention mask for the tokenized prompt
            device: Device to run inference on
            **generate_kwargs: Additional generation parameters
        """
        if expression_tokens is None:
            raise ValueError("You need to provide expression_tokens")
        
        # Process inputs
        # expression_tokens = expression_tokens.to(device)
        # expression_token_lengths = expression_token_lengths.to(device)
        
        # Get cell embeddings from Geneformer
        cell_embeddings = self.cell_encoder(
            expression_tokens=expression_tokens,
            expression_token_lengths=expression_token_lengths,
            return_dict=True
        )
        cell_embeddings = self.process_encoder_outputs(cell_embeddings)
                
        
        # Generate text description using the decoder with tokenized prompt
        return self.decoder.generate_cell_description(
            cell_embeddings=cell_embeddings,
            inputs=inputs,  # tokenized prompt
            attention_mask=attention_mask,
            device=device,
            **generate_kwargs
        )