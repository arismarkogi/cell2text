#https://github.com/ColinFX/Prot2Text-V2 This code was really helpful

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM, AutoConfig, AutoTokenizer
from typing import Optional, Tuple, Union, List, Dict, Any
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Cache


class Cell2TextLlamaConfig(PretrainedConfig):
    def __init__(
        self,
        num_beams=4,
        max_length=100,
        early_stopping=True,
        no_repeat_ngram_size=3,
        temperature=1.0,
        top_p=1.0,
        placeholder_id=128003,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_beams = num_beams
        self.max_length = max_length
        self.early_stopping = early_stopping
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.temperature = temperature
        self.top_p = top_p
        self.placeholder_id = placeholder_id


class Cell2TextLlamaModel(PreTrainedModel, GenerationMixin):
    
    config_class = Cell2TextLlamaConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.llama = None  # Will be loaded in from_pretrained
        self.tokenizer = None
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, config=None, torch_dtype=None, **kwargs):
        if "config" in kwargs:
            config = kwargs.pop("config")
            if isinstance(config, dict):
                config = Cell2TextLlamaConfig(**config)
            elif not isinstance(config, Cell2TextLlamaConfig):
                raise ValueError("Parameter `config` must be a dictionary or an instance of `Cell2TextLlamaConfig`.")
        else:
            config = Cell2TextLlamaConfig()

        model = cls(config, **kwargs)
                
        if torch_dtype is None:
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Load the LLaMA model
        model.llama = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            **kwargs
        )
        model.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        
        
        return model
    
    def prepare_decoder_inputs(
        self,
        input_ids: torch.LongTensor,         # Tokenized text prompt (batch_size, seq_len)
        cell_embeddings: torch.FloatTensor,  # Gene expression embeddings (batch_size, cell_seq_len, hidden_dim)
        attention_mask: Optional[torch.LongTensor] = None,    # Attention mask for input_ids (batch_size, seq_len)
        cell_attention_mask: Optional[torch.LongTensor] = None, # Attention mask for cell_embeddings (batch_size, cell_seq_len)
    ):
        batch_size, seq_len = input_ids.size()
        _, cell_seq_len_dim, _ = cell_embeddings.size() # Note: Renamed cell_seq_len to avoid conflict if it's a class member

        # Default attention masks if not provided (all ones, meaning attend to all tokens)
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=input_ids.device)
        if cell_attention_mask is None:
            
            cell_attention_mask = torch.ones((batch_size, cell_seq_len_dim), dtype=torch.long, device=cell_embeddings.device)

        
        # 1. Get text embeddings from the language model's embedding layer
        # input_ids are token indices. This converts them to dense vectors.
        # Shape: (batch_size, seq_len, hidden_dim)
        inputs_embeds = self.llama.get_input_embeddings()(input_ids)

        # 2. Replace placeholders with cell embeddings
        # placeholder_mask = input_ids == self.config.placeholder_id # (batch_size, seq_len)
        # cell_mask = cell_attention_mask.bool() # (batch_size, cell_seq_len_dim)
        # inputs_embeds[placeholder_mask] = cell_embeddings[cell_mask]


        # Iterate over each sample in the batch for robust replacement
        for i in range(batch_size):
            # Find indices of placeholder tokens in the current sample's input_ids
            # .nonzero() returns a tuple of tensors, one for each dimension. We want the first one.
            placeholder_indices_in_sample = (input_ids[i] == self.config.placeholder_id).nonzero(as_tuple=True)[0]
            num_placeholders_sample = len(placeholder_indices_in_sample)

            if num_placeholders_sample == 0:
                continue # No placeholders to replace in this sample

            # Select the active cell embeddings for the current sample using its cell_attention_mask
            # active_cell_embeddings_for_sample shape: (num_active_genes_in_sample, hidden_dim)
            active_cell_embeddings_for_sample = cell_embeddings[i][cell_attention_mask[i].bool()]

            # The number of placeholders in the text prompt (num_placeholders_sample)
            # dictates how many gene embeddings we need to insert. This count comes from
            # `placeholder_length` in `__getitem__` (which considers `top_k`).
            # We must ensure we have enough cell embeddings and select the correct ones.
            if active_cell_embeddings_for_sample.shape[0] < num_placeholders_sample:
                raise ValueError(
                    f"Sample {i}: Not enough cell embeddings ({active_cell_embeddings_for_sample.shape[0]}) "
                    f"to fill placeholders ({num_placeholders_sample}). "
                    f"Ensure cell_embeddings provide at least `top_k` (or actual gene count if less than `top_k`) embeddings per sample."
                )

            # Select the subset of cell embeddings to insert. We assume the first
            # `num_placeholders_sample` active cell embeddings are the ones to use.
            # This aligns with using `top_k` (or min(len, top_k)) genes.
            embeddings_to_insert = active_cell_embeddings_for_sample[:num_placeholders_sample]

            # Perform the replacement for the current sample
            inputs_embeds[i, placeholder_indices_in_sample] = embeddings_to_insert

        return inputs_embeds, attention_mask
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        cell_embeddings: Optional[torch.FloatTensor] = None,
        cell_attention_mask: Optional[torch.LongTensor] = None,
        return_decoder_inputs: bool = False,
        **kwargs  # All other LlamaForCausalLM arguments
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        if cell_embeddings is None:
            # Standard LLaMA forward without cell embeddings
            return self.llama(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
        
        if input_ids is None:
            batch_size = cell_embeddings.shape[0]
            input_ids = torch.full(
                (batch_size, 1), 
                self.tokenizer.bos_token_id, 
                dtype=torch.long, 
                device=cell_embeddings.device
            )

        print(f"Inside llama decoder, input_ids.shape: {input_ids.shape}")
        print(f"Inside llama decoder, cell_embeddings.shape: {cell_embeddings.shape}")

        # Prepare inputs with placeholder replacement
        inputs_embeds, attention_mask = self.prepare_decoder_inputs(
            input_ids=input_ids, 
            cell_embeddings=cell_embeddings, 
            attention_mask=attention_mask, 
            cell_attention_mask=cell_attention_mask, 
        )
        
        if return_decoder_inputs:
            return inputs_embeds, attention_mask
        
        print(f"Inside llama decoder, input_embeds.shape: {inputs_embeds.shape}")

        # Remove inputs_embeds from kwargs to avoid duplicate argument error, i dont know what's happemning
        kwargs.pop('inputs_embeds', None)

        print(f"Inside llama decoder, input_embeds.shape: {inputs_embeds.shape}")

        # Forward through LLaMA with prepared embeddings
        return self.llama(
            input_ids=None,  # We use inputs_embeds instead
            attention_mask=attention_mask, 
            inputs_embeds=inputs_embeds,
            **kwargs
        )
    
   # Updated generate method in Cell2TextLlamaModel class
    def generate(
        self,
        inputs: torch.LongTensor,  # alias of `input_ids` - tokenized prompt
        attention_mask: Optional[torch.LongTensor] = None,
        cell_embeddings: Optional[torch.FloatTensor] = None,
        cell_attention_mask: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
        Do inference based on given input prompt. 
        `inputs` is expected to be tokenized [prompt] only. 
        Output will not keep the input prompt due to input in form of embeds.
        Generation behavior can be controlled by `kwargs`, read 
        `GenerationMixin.generate` for more info. 
        """
        if cell_embeddings is not None:
            # Get decoder inputs
            prompt_inputs_embeds, prompt_attention_mask = self(
                input_ids=inputs, 
                attention_mask=attention_mask,
                cell_embeddings=cell_embeddings,
                cell_attention_mask=cell_attention_mask,
                return_decoder_inputs=True
            )
            
            # Generate with prepared embeddings
            return self.llama.generate(
                inputs_embeds=prompt_inputs_embeds, 
                attention_mask=prompt_attention_mask, 
                **kwargs
            )
        else:
            # Standard generation without cell embeddings
            return self.llama.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                **kwargs
            )
    
    # Delegate essential properties and methods to LLaMA
    def get_input_embeddings(self):
        return self.llama.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        self.llama.set_input_embeddings(value)
    
    def get_output_embeddings(self):
        return self.llama.get_output_embeddings()
    
    def set_output_embeddings(self, new_embeddings):
        self.llama.set_output_embeddings(new_embeddings)
    
    def tie_weights(self):
        self.llama.tie_weights()
    
    @property
    def device(self):
        return self.llama.device
    
    @torch.no_grad()
    def generate_cell_description(
        self,
        cell_embeddings: torch.FloatTensor,
        inputs: Optional[torch.LongTensor] = None,  # tokenized prompt
        attention_mask: Optional[torch.LongTensor] = None,
        device: str = 'cpu',
        **generate_kwargs
    ):
        """
        Generate text description for cell embeddings using tokenized prompt.
        
        Args:
            cell_embeddings: Cell embeddings tensor
            inputs: Tokenized prompt (input_ids)
            attention_mask: Attention mask for the tokenized prompt
            device: Device to run inference on
            **generate_kwargs: Additional generation parameters
        """
        batch_size = cell_embeddings.shape[0]
        cell_embeddings = cell_embeddings.to(device)
        
        if inputs is not None:
            # Use provided tokenized prompt
            inputs = inputs.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            else:
                # Create attention mask if not provided
                attention_mask = torch.ones_like(inputs)
            
            # Expand to match batch size if needed
            if batch_size > 1 and inputs.shape[0] == 1:
                inputs = inputs.repeat(batch_size, 1)
                attention_mask = attention_mask.repeat(batch_size, 1)
        else:
            # Use BOS token as default if no prompt provided
            inputs = torch.full(
                (batch_size, 1), 
                self.tokenizer.bos_token_id, 
                dtype=torch.long, 
                device=device
            )
            attention_mask = torch.ones_like(inputs)

        # Set default generation parameters from config
        default_generate_args = {
            "max_new_tokens": self.config.max_length,
            "num_beams": self.config.num_beams,
            "early_stopping": self.config.early_stopping,
            "no_repeat_ngram_size": self.config.no_repeat_ngram_size,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }

        for k, v in default_generate_args.items():
            generate_kwargs.setdefault(k, v)

        # Generate using the updated generate method
        output_ids = self.generate(
            inputs=inputs,
            attention_mask=attention_mask,
            cell_embeddings=cell_embeddings,
            **generate_kwargs
        )

        # Decode the output
        decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        if hasattr(self.tokenizer, "additional_special_tokens"):
            for token in self.tokenizer.additional_special_tokens:
                decoded = [text.replace(token, "") for text in decoded]

        return decoded[0] if batch_size == 1 else decoded

