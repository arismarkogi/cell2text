import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM, AutoConfig, AutoTokenizer
from typing import Optional, Tuple, Union, List, Dict, Any
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin




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




class Cell2TextLlamaModel(PreTrainedModel, GenerationMixin):
    
    config_class = Cell2TextLlamaConfig
    
    
    def __init__(self, config):
        super().__init__(config)
            
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
        text_input_ids: Optional[torch.LongTensor] = None,
        text_attention_mask: Optional[torch.FloatTensor] = None,
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
        
        if text_input_ids is None:
            print("TEXT_INPUT_IDS is None\nWe set it to BOS")
            bos_token_id = self.tokenizer.bos_token_id  # Ensure `tokenizer` is available
            text_input_ids = torch.full(
                (cell_embeddings.shape[0], 1), bos_token_id, dtype=torch.long, device=cell_embeddings.device
            )


        # Get Llama embeddings for text tokens
        text_embeddings = self.decoder.get_input_embeddings()(text_input_ids)
        print(f"text_embeddings_shape{text_embeddings.shape}")

        # Concatenate cell embeddings with text embeddings
        combined_embeddings = torch.cat([cell_embeddings, text_embeddings], dim=1)

        print(f"combined_embeddings_shape: {combined_embeddings.shape}")


        num_cell_tokens = cell_embeddings.shape[1]

        # Update attention mask
        if text_attention_mask is not None:
            prepend_mask = torch.ones((batch_size, 1), device=text_attention_mask.device)
            combined_attention_mask = torch.cat([prepend_mask, text_attention_mask], dim=1)
        else:
            combined_attention_mask = None

        # Handle labels (shift for added cell embedding)
        if labels is not None:
            prepend_labels = torch.full((batch_size, num_cell_tokens), -100, dtype=labels.dtype, device=labels.device)
            combined_labels = torch.cat([prepend_labels, labels], dim=1)
        else:
            combined_labels = None


        return self.decoder(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
            labels=combined_labels,
            **kwargs
        )
        
        
        
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple[Any]] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        cell_embeddings: Optional[torch.FloatTensor] = None,
        **kwargs
    ):
        # If past is not None, we are in generation mode past the first step
        if past_key_values is not None:
            return {
                "input_ids": input_ids,
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
            }

        # First generation step — prepend cell embeddings
        batch_size = input_ids.shape[0]
        text_embeddings = self.decoder.decoder.get_input_embeddings()(input_ids)

        if cell_embeddings is None:
            raise ValueError("`cell_embeddings` must be provided at generation start")

        
        
        inputs_embeds = torch.cat([cell_embeddings, text_embeddings], dim=1)

        # Update attention mask
        if attention_mask is not None:
            prefix = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([prefix, attention_mask], dim=1)

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
        }

    
    

    
    @torch.no_grad()
    def generate_cell_description(
        self,
        cell_embeddings: torch.FloatTensor,
        device: str = 'cpu',
        **generate_kwargs
    ):
        """
        Generate text description from cell embeddings.
        """
        batch_size = cell_embeddings.shape[0]
        #cell_embeddings = cell_embeddings.to(device).unsqueeze(1)  # [B, 1, H]


        # BOS token input IDs and embeddings
        bos_input_ids = torch.full((batch_size, 1), self.tokenizer.bos_token_id, dtype=torch.long, device=device)
        bos_embedding = self.decoder.get_input_embeddings()(bos_input_ids)  # [B, 1, H]


        # Concatenate: [BOS] + [cell_embedding] 
        combined_embeddings = torch.cat([bos_embedding, cell_embeddings], dim=1)

        # Attention mask
        attention_mask = torch.ones((batch_size, combined_embeddings.size(1)), dtype=torch.long, device=device)

        # Default generation parameters (merged with kwargs)
        default_generate_args = {
            "max_new_tokens": self.config.max_length,
            "num_beams": self.config.num_beams,
            "early_stopping": self.config.early_stopping,
            "no_repeat_ngram_size": self.config.no_repeat_ngram_size,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }

        print(default_generate_args)
        for k, v in default_generate_args.items():
            generate_kwargs.setdefault(k, v)

        # Generate output
        output_ids = self.decoder.generate(
            inputs_embeds=combined_embeddings,
            attention_mask=attention_mask,
            input_ids=bos_input_ids,
            **generate_kwargs
        )

        # Decode and clean
        decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        if hasattr(self.tokenizer, "additional_special_tokens"):
            for token in self.tokenizer.additional_special_tokens:
                decoded = [text.replace(token, "") for text in decoded]

        return decoded[0] if batch_size == 1 else decoded



 