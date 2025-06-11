import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from typing import Optional, Dict, Any
import os
from safetensors.torch import load_file

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

        
        # Store configuration parameters
        self.cell_encoder_hidden_size = config.cell_encoder_hidden_size
        self.decoder_hidden_size = config.decoder_hidden_size
        self.mlp_hidden_size = config.mlp_hidden_size
        self.mlp_dropout = config.mlp_dropout
        self.top_k = config.top_k
        

        # Initialize all components in __init__
        self.cell_encoder = GeneformerModel(self.geneformer_config)

        self.decoder = Cell2TextLlamaModel(self.cell2text_llama_config)
        
        self.cell_to_embedding = MLPProjectionLayer(
            input_dim=self.cell_encoder_hidden_size, 
            hidden_dim=self.mlp_hidden_size, 
            output_dim=self.decoder_hidden_size,
            dropout_prob=self.mlp_dropout, 
            bias=True
        )

        
        self.config = config
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, config=None, **kwargs):
        """
        Load the complete model from pretrained weights.
        Handles both individual component loading and full model safetensors.
        """
        if config is None:
            # Try to load config from the pretrained path
            config_path = os.path.join(pretrained_model_name_or_path, "config.json")
            if os.path.exists(config_path):
                config = PretrainedConfig.from_json_file(config_path)
            else:
                raise ValueError("No config provided and no config.json found in pretrained path")
        
        # Initialize model with config
        model = cls(config)
        
        # Load weights
        model.load_pretrained_weights(pretrained_model_name_or_path, **kwargs)
        
        return model
    
    def load_pretrained_weights(self, pretrained_model_name_or_path: str, **kwargs):
        """
        Load pretrained weights for all components.
        Supports both individual component loading and full model safetensors.
        """
        # Check if we have a full model safetensors file
        full_model_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        pytorch_model_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        
        if os.path.exists(full_model_path):
            # Load from safetensors
            print(f"Loading full model from safetensors: {full_model_path}")
            self._load_from_safetensors(full_model_path)
        elif os.path.exists(pytorch_model_path):
            # Load from pytorch bin
            print(f"Loading full model from pytorch: {pytorch_model_path}")
            self._load_from_pytorch(pytorch_model_path)
        else:
            # Load individual components
            print("Loading individual components...")
            self._load_individual_components(pretrained_model_name_or_path, **kwargs)
    
    def _load_from_safetensors(self, safetensors_path: str):
        """Load weights from a safetensors file."""
        state_dict = load_file(safetensors_path)
        self._load_state_dict_with_prefix_handling(state_dict)
    
    def _load_from_pytorch(self, pytorch_path: str):
        """Load weights from a pytorch .bin file."""
        state_dict = torch.load(pytorch_path, map_location="cpu")
        self._load_state_dict_with_prefix_handling(state_dict)
    
    
    def _load_state_dict_with_prefix_handling(self, state_dict: Dict[str, torch.Tensor]):
        """
        Load state dict with proper prefix handling for different components.
        """
        # Separate state dict by component
        cell_encoder_state = {}
        decoder_state = {}
        projector_state = {}
        
        for key, value in state_dict.items():
            if key.startswith("cell_encoder."):
                new_key = key.replace("cell_encoder.", "")
                cell_encoder_state[new_key] = value
            elif key.startswith("decoder."):
                new_key = key.replace("decoder.", "")
                decoder_state[new_key] = value
            elif key.startswith("cell_to_embedding."):
                new_key = key.replace("cell_to_embedding.", "")
                projector_state[new_key] = value
        
        # Load into components
        if cell_encoder_state:
            print(f"Loading {len(cell_encoder_state)} cell encoder parameters")
            self.cell_encoder.load_state_dict(cell_encoder_state, strict=False)
        
        if decoder_state:
            print(f"Loading {len(decoder_state)} decoder parameters")
            self.decoder.load_state_dict(decoder_state, strict=False)
        
        if projector_state:
            print(f"Loading {len(projector_state)} projector parameters")
            self.cell_to_embedding.load_state_dict(projector_state, strict=False)
    
    def _load_individual_components(self, base_path: str, **kwargs):
        """
        Load individual components from separate directories/files.
        """
        # Load cell encoder
        geneformer_path = getattr(self.config, 'geneformer_path', None)
        if geneformer_path:
            print(f"Loading Geneformer from: {geneformer_path}")
            # Load cell encoder
            self.cell_encoder = GeneformerModel.from_pretrained(
                pretrained_model_name_or_path=self.config.geneformer_path, 
                config=self.geneformer_config
            )
        
        # Load decoder
        decoder_path = getattr(self.config, 'decoder_model_name_or_path', None)
        if decoder_path:
            print(f"Loading LLaMA decoder from: {decoder_path}")
            self.decoder = Cell2TextLlamaModel.from_pretrained(
                pretrained_model_name_or_path=self.config.decoder_model_name_or_path,
                config=self.cell2text_llama_config
            )
        
        # Projector weights will be randomly initialized unless loaded from full model
        print("Projector weights randomly initialized (unless loaded from full model)")
    

    def process_encoder_outputs(self, cell_embeddings):
        """
        Utility to process encoder output: truncate top_k and project to decoder hidden size.
        """
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
        input_ids: Optional[torch.LongTensor] = None,  # Changed from text_input_ids
        attention_mask: Optional[torch.FloatTensor] = None,  # Changed from text_attention_mask
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

        
                    
        print(f"self.top_k: {self.top_k}")

        cell_embeddings = self.process_encoder_outputs(cell_embeddings)

        # Forward pass through decoder - using correct parameter names
        decoder_outputs = self.decoder(
            input_ids=input_ids,  # Changed from text_input_ids
            attention_mask=attention_mask,  # Changed from text_attention_mask
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