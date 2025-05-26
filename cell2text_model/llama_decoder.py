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
        input_ids: torch.LongTensor,
        cell_embeddings: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        cell_attention_mask: Optional[torch.LongTensor] = None, 
    ):
        batch_size, seq_len = input_ids.size()
        _, cell_seq_len, _ = cell_embeddings.size()
        
        if attention_mask is None: 
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=input_ids.device)
        if cell_attention_mask is None: 
            cell_attention_mask = torch.ones((batch_size, cell_seq_len), dtype=torch.long, device=cell_embeddings.device)

        print("Vocab size:", self.llama.config.vocab_size)
        print("Pad token ID:", self.tokenizer.pad_token_id)


        print(f"input_ids:{input_ids}")
        print(f"input_ids.shape: {input_ids.shape}")
        # Get text embeddings
        inputs_embeds = self.llama.get_input_embeddings()(input_ids)
        
        # Replace placeholders with cell embeddings
        placeholder_mask = input_ids == self.config.placeholder_id
        cell_mask = cell_attention_mask.bool()
        inputs_embeds[placeholder_mask] = cell_embeddings[cell_mask]
        
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

       

        # Forward through LLaMA with prepared embeddings
        return self.llama(
            input_ids=None,  # We use inputs_embeds instead
            attention_mask=attention_mask, 
            inputs_embeds=inputs_embeds,
            **kwargs
        )
    
    def generate(self, **kwargs):
        """Delegate generation to LLaMA after processing cell embeddings if provided."""
        if 'cell_embeddings' in kwargs:
            cell_embeddings = kwargs.pop('cell_embeddings')
            cell_attention_mask = kwargs.pop('cell_attention_mask', None)
            inputs = kwargs.pop('input_ids', kwargs.pop('inputs', None))
            attention_mask = kwargs.pop('attention_mask', None)
            
            if inputs is None:
                batch_size = cell_embeddings.shape[0]
                inputs = torch.full(
                    (batch_size, 1), 
                    self.tokenizer.bos_token_id, 
                    dtype=torch.long, 
                    device=cell_embeddings.device
                )
            
            # Get prepared embeddings
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
            # Standard generation
            return self.llama.generate(**kwargs)
    
    # Delegate essential properties and methods to LLaMA
    def get_input_embeddings(self):
        print("PAD token ID:", self.tokenizer.pad_token_id)
        print("Vocab size:", self.decoder.config.vocab_size)
        print("Pad token ID:", self.tokenizer.pad_token_id)
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
        prompt_template: Optional[str] = None,
        device: str = 'cpu',
        **generate_kwargs
    ):
        batch_size = cell_embeddings.shape[0]
        cell_embeddings = cell_embeddings.to(device)
        
        if prompt_template is not None:
            prompt_inputs = self.tokenizer(
                prompt_template,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            input_ids = prompt_inputs.input_ids.to(device)
            attention_mask = prompt_inputs.attention_mask.to(device)
            
            if batch_size > 1:
                input_ids = input_ids.repeat(batch_size, 1)
                attention_mask = attention_mask.repeat(batch_size, 1)
        else:
            input_ids = torch.full(
                (batch_size, 1), 
                self.tokenizer.bos_token_id, 
                dtype=torch.long, 
                device=device
            )
            attention_mask = torch.ones_like(input_ids)

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

        output_ids = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cell_embeddings=cell_embeddings,
            **generate_kwargs
        )

        decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        if hasattr(self.tokenizer, "additional_special_tokens"):
            for token in self.tokenizer.additional_special_tokens:
                decoded = [text.replace(token, "") for text in decoded]

        return decoded[0] if batch_size == 1 else decoded