import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM, AutoConfig, AutoTokenizer
from typing import Optional, Tuple, Union, List, Dict, Any
from transformers import PretrainedConfig



class Cell2TextLlamaConfig(PretrainedConfig):

    def __init__(
        self,
        num_beams=4,
        max_length=100,
        early_stopping=True,
        no_repeat_ngram_size=3,
        temperature=1.0,
        top_p=1.0,

        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_beams=num_beams
        self.max_length=max_length
        self.early_stopping=early_stopping
        self.no_repeat_ngram_size=no_repeat_ngram_size
        self.temperature=temperature
        self.top_p=top_p




class Cell2TextLlamaModel(nn.Module):
    
    config_class = Cell2TextLlamaConfig
    
    
    def __init__(self, config):
        super().__init__()
            
        # Initialize the Llama model
        self.decoder = None  # Will be loaded in warm_up
        
        self.config = config or Cell2TextLlamaConfig()
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, config=None, torch_dtype=None, **kwargs):
        """
        Instantiate model and load pre-trained LLaMA weights from a given path or model name
        """

        if "config" in kwargs:
            config = kwargs.pop("config")
            if isinstance(config, dict):
                config = Cell2TextLlamaConfig(**config)
            elif not isinstance(config, Cell2TextLlamaConfig):
                raise ValueError(
                    "Parameter `config` must be a dictionary or an instance of `GeneformerConfig`."
                )
        else:
            config = Cell2TextLlamaConfig()
            

        model = cls(config, **kwargs)
        
        model = cls(config)
        
        # Use appropriate dtype
        if torch_dtype is None:
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Load decoder config and model
        llama_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        model.decoder = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            config=llama_config,
            **kwargs
        )

        # Attach config explicitly if needed
        model.decoder.config = llama_config

        model.tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        
        return model
    
    def forward(
        self,
        cell_embeddings: torch.FloatTensor,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        """
        Forward pass that prepends cell embeddings to input_ids
        """
        batch_size = cell_embeddings.shape[0]
        
        
        # Get Llama embeddings for text tokens
        text_embeddings = self.decoder.get_input_embeddings()(decoder_input_ids)
        
        # Reshape cell embeddings to match token embeddings shape
        cell_embeddings = cell_embeddings.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # Concatenate cell embeddings with text embeddings
        combined_embeddings = torch.cat([cell_embeddings, text_embeddings], dim=1)
        
        # Update attention mask to account for the prepended embedding
        if decoder_attention_mask is not None:
            combined_attention_mask = torch.ones(
                (batch_size, 1), device=decoder_attention_mask.device
            )
            combined_attention_mask = torch.cat(
                [combined_attention_mask, decoder_attention_mask], dim=1
            )
        else:
            combined_attention_mask = None
        
        # If labels are provided, shift them to account for prepended embeddings
        combined_labels = None
        if labels is not None:
            # Create ignore index (-100) for the prepended cell embeddings position
            prepend_labels = torch.full(
                (batch_size, 1), fill_value=-100, dtype=labels.dtype, device=labels.device
            )
            combined_labels = torch.cat([prepend_labels, labels], dim=1)

        
        # Forward pass through the decoder with combined embeddings
        outputs = self.decoder(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
            labels=combined_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        
        return outputs
    
    @torch.no_grad()
    def generate_cell_description(
        self,
        cell_embeddings: torch.FloatTensor,
        device='cpu',
        **generate_kwargs
    ):
        """
        Generate text description for cell expression data by prepending
        cell embeddings to the input sequence
        """
 
        
        batch_size = cell_embeddings.shape[0]
        
        # Project cell embeddings to match Llama hidden size
        cell_embeddings = cell_embeddings.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # Create a starting input_ids with just the bos token
        input_ids = torch.ones((batch_size, 1), dtype=torch.long, device=device) * self.tokenizer.bos_token_id
        
        # Get the embeddings for the BOS token
        bos_embeddings = self.decoder.get_input_embeddings()(input_ids)
        
        # Concatenate cell embeddings with BOS embeddings
        combined_embeddings = torch.cat([cell_embeddings, bos_embeddings], dim=1)
        
        # Create attention mask for the combined embeddings
        attention_mask = torch.ones((batch_size, combined_embeddings.size(1)), device=device)
        
        my_params = {
            "max_length": self.config.max_length,
            "num_beams": self.config.num_beams,
            "early_stopping": self.config.early_stopping,
            "no_repeat_ngram_size": self.config.no_repeat_ngram_size,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p
        }
        
        # Update with user-provided parameters
        for key, value in my_params.items():
            if key not in generate_kwargs:
                generate_kwargs[key] = value
        
        # Generate text
        outputs = self.decoder.generate(
            inputs_embeds=combined_embeddings,
            attention_mask=attention_mask,
            **generate_kwargs
        )
        
        # Decode generated tokens
        generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Clean up any special tokens that might remain
        if hasattr(self.tokenizer, "additional_special_tokens"):
            for token in self.tokenizer.additional_special_tokens:
                generated_text = [text.replace(token, "") for text in generated_text]
        
        return generated_text[0] if len(generated_text) == 1 else generated_text


 