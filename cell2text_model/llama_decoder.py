#https://github.com/ColinFX/Prot2Text-V2 This code was really helpful

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM, AutoConfig, AutoTokenizer
from typing import Optional, Tuple, Union, List, Dict, Any
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Cache
import os 

def resolve_device() -> torch.device:
    """
    Decide which torch.device to use, with minimal fuss.

    Order of preference
    -------------------
    1. GPU whose index equals LOCAL_RANK (DDP case)
    2. First visible GPU ('cuda:0') if any
    3. CPU
    """
    if torch.cuda.is_available():
        # --- Distributed run: honour per‑process local rank ---------------
        local_rank = int(os.getenv("LOCAL_RANK", os.getenv("RANK", 0)))
        if local_rank < torch.cuda.device_count():
            return torch.device(f"cuda:{local_rank}")

        # --- Non‑DDP or out‑of‑range rank: fall back to first GPU ---------
        return torch.device("cuda:0")

    # --- No CUDA ----------------------------------------------------------
    return torch.device("cpu")


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
    
    def load_from_state_dict(self, state_dict, strict=True):
        """
        Generalized loader that inspects state dict and handles any model configuration
        """
        # Initialize the LLaMA model if it doesn't exist
        if self.llama is None:
            # Replace with your actual model path
            default_llama_path = "meta-llama/Llama-3.2-3B-Instruct"
            
            # First, initialize the base model to avoid NoneType errors
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            # Extract llama-related keys first
            llama_state_dict = {}
            prefix = "decoder."
            
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    model_key = key[len(prefix):]
                    llama_state_dict[model_key] = value
            
            if not llama_state_dict:
                print("Warning: No llama weights found in state dict, initializing with default weights")
                # Initialize with default weights if no state dict found
                self.llama = LlamaForCausalLM.from_pretrained(
                    default_llama_path,
                    torch_dtype=torch_dtype,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(default_llama_path)
                return
            
            # Inspect the state dict structure to determine model type
            has_lora = any('lora_A' in key or 'lora_B' in key for key in llama_state_dict.keys())
            has_base_model = any(key.startswith('base_model.') for key in llama_state_dict.keys())
            
            if has_lora or has_base_model:
                print("Detected LoRA/PEFT model structure, loading with PEFT...")
                self._load_peft_model(default_llama_path, llama_state_dict)
            else:
                print("Standard model structure detected, loading normally...")
                self._load_standard_model(default_llama_path, llama_state_dict)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(default_llama_path)
            print("Model loaded successfully!")

    def _load_standard_model(self, model_path, llama_state_dict):
        """Load standard LLaMA model"""
        self.llama = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        
        # Load weights
        missing_keys, unexpected_keys = self.llama.load_state_dict(llama_state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys: {len(missing_keys)} keys")
        if unexpected_keys:
            print(f"Unexpected keys: {len(unexpected_keys)} keys")

    def _load_peft_model(self, model_path, llama_state_dict):
        """Load PEFT/LoRA model by inspecting the state dict structure"""
        try:
            from peft import PeftModel, LoraConfig, get_peft_model
            
            # Load base model first
            base_model = LlamaForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            
            # Auto-detect LoRA configuration from state dict
            lora_config = self._detect_lora_config(llama_state_dict)
            
            # Create PEFT model
            self.llama = get_peft_model(base_model, lora_config)
            
            # Load weights
            missing_keys, unexpected_keys = self.llama.load_state_dict(llama_state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys: {len(missing_keys)} keys")
            if unexpected_keys:
                print(f"Unexpected keys: {len(unexpected_keys)} keys")
                
        except ImportError:
            print("PEFT not installed. Installing: pip install peft")
            raise
        except Exception as e:
            print(f"Error loading PEFT model: {e}")
            print("Falling back to standard loading...")
            self._load_standard_model(model_path, llama_state_dict)

    def _detect_lora_config(self, state_dict):
        """Automatically detect LoRA configuration from state dict"""
        from peft import LoraConfig
        
        # Find LoRA modules and extract config
        lora_keys = [k for k in state_dict.keys() if 'lora_A' in k or 'lora_B' in k]
        
        if not lora_keys:
            # Default config if no LoRA keys found
            return LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM",
            )
        
        # Extract target modules
        target_modules = set()
        r_value = None
        
        for key in lora_keys:
            # Extract module name (e.g., "q_proj" from "base_model.model.layers.0.self_attn.q_proj.lora_A.weight")
            parts = key.split('.')
            for i, part in enumerate(parts):
                if 'lora_A' in part or 'lora_B' in part:
                    if i > 0:
                        target_modules.add(parts[i-1])
                    break
            
            # Extract rank from lora_A weight shape
            if 'lora_A' in key and r_value is None:
                tensor = state_dict[key]
                if tensor.dim() == 2:
                    r_value = tensor.shape[0]
        
        target_modules = list(target_modules) if target_modules else ["q_proj", "v_proj"]
        r_value = r_value if r_value else 16
        
        print(f"Auto-detected LoRA config: r={r_value}, target_modules={target_modules}")
        
        return LoraConfig(
            r=r_value,
            lora_alpha=r_value * 2,  # Common convention
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
    
    def prepare_decoder_inputs(
        self,
        input_ids: torch.LongTensor,         # Tokenized text prompt (batch_size, seq_len)
        cell_embeddings: torch.FloatTensor,  # Gene expression embeddings (batch_size, cell_seq_len, hidden_dim)
        attention_mask: Optional[torch.LongTensor] = None,    # Attention mask for input_ids (batch_size, seq_len)
        cell_attention_mask: Optional[torch.LongTensor] = None, # Attention mask for cell_embeddings (batch_size, cell_seq_len)
    ):
        batch_size, seq_len = input_ids.size()
        _, cell_seq_len_dim, _ = cell_embeddings.size() # Note: Renamed cell_seq_len to avoid conflict if it's a class member

        # Get the target device from the model
        target_device = next(self.llama.parameters()).device
        
        # Ensure all tensors are on the same device
        input_ids = input_ids.to(target_device)
        cell_embeddings = cell_embeddings.to(target_device)
        
        # Default attention masks if not provided (all ones, meaning attend to all tokens)
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=target_device)
        else:
            attention_mask = attention_mask.to(target_device)
            
        if cell_attention_mask is None:
            cell_attention_mask = torch.ones((batch_size, cell_seq_len_dim), dtype=torch.long, device=target_device)
        else:
            cell_attention_mask = cell_attention_mask.to(target_device)

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
            # Ensure embeddings are on the same device and have the correct dtype
            embeddings_to_insert = embeddings_to_insert.to(device=target_device, dtype=inputs_embeds.dtype)

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
        
        # Get the target device from the model
        target_device = next(self.llama.parameters()).device

        # Move all tensors to the target device
        if input_ids is not None:
            input_ids = input_ids.to(target_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(target_device)
        if cell_embeddings is not None:
            cell_embeddings = cell_embeddings.to(target_device)
        if cell_attention_mask is not None:
            cell_attention_mask = cell_attention_mask.to(target_device)

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
                device=target_device
            )

        # Prepare inputs with placeholder replacement
        inputs_embeds, attention_mask = self.prepare_decoder_inputs(
            input_ids=input_ids, 
            cell_embeddings=cell_embeddings, 
            attention_mask=attention_mask, 
            cell_attention_mask=cell_attention_mask, 
        )
        
        if return_decoder_inputs:
            return inputs_embeds, attention_mask
        
        # Remove inputs_embeds from kwargs to avoid duplicate argument error
        kwargs.pop('inputs_embeds', None)

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
        
        # Get the actual device from the model instead of using the passed device parameter
        target_device = next(self.llama.parameters()).device
        cell_embeddings = cell_embeddings.to(target_device)
        
        if inputs is not None:
            # Use provided tokenized prompt
            inputs = inputs.to(target_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(target_device)
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
                device=target_device
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