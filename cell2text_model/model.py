import torch
import torch.nn as nn
from transformers import GPT2Config, PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from typing import Optional, Tuple, Union, Callable, List

# Import from GeneformerModel implementation
from .geneformer_encoder import GeneformerModel

# Import from Prot2Text implementation
from .utils import CABlock, _GPT2LMHeadModel




class Cell2TextModel(PreTrainedModel):
    config_class = PretrainedConfig
    _keys_to_ignore_on_load_missing = [r"transformer"]
    base_model_prefix = "decoder"
    
    def __init__(self, config):
        super().__init__(config)
        
        # GPT2 configuration for the decoder
        self.gpt_config = GPT2Config.from_dict(config.gpt_config)
        
        # Initialize the Geneformer encoder
        self.encoder = GeneformerModel(
            num_classes=config.num_classes if hasattr(config, "num_classes") else 0,
            emb_mode=config.emb_mode if hasattr(config, "emb_mode") else "cell",
            hidden_size=config.encoder_hidden_size if hasattr(config, "encoder_hidden_size") else 512,
            max_ncells=config.max_ncells if hasattr(config, "max_ncells") else 200,
            emb_layer=config.emb_layer if hasattr(config, "emb_layer") else -1,
            emb_label=config.emb_label if hasattr(config, "emb_label") else ["sample_name", "cell type rough", "cell type"],
            forward_batch_size=config.forward_batch_size if hasattr(config, "forward_batch_size") else -1,
            nproc=config.nproc if hasattr(config, "nproc") else 4,
            summary_stat=config.summary_stat if hasattr(config, "summary_stat") else None
        )
        
        # GPT2 decoder with cross-attention
        self.decoder = _GPT2LMHeadModel(self.gpt_config)
        
        # If fusion of embeddings is needed
        if hasattr(config, "fusion_method") and config.fusion_method == "cross_attention":
            self.h = nn.ModuleList([CABlock(self.gpt_config, layer_idx=i) for i in range(4)])
            self.ln_f = nn.LayerNorm(self.gpt_config.n_embd, eps=self.gpt_config.layer_norm_epsilon)
        
        # Linear projection to match dimensions if needed
        if hasattr(config, "encoder_hidden_size") and config.encoder_hidden_size != self.gpt_config.n_embd:
            self.to_embedding = nn.Linear(config.encoder_hidden_size, self.gpt_config.n_embd)
        else:
            self.to_embedding = nn.Identity()
        
        self.config = config
    
    def get_encoder(self):
        return self.encoder
    
    def get_decoder(self):
        return self.decoder
    
    def get_input_embeddings(self):
        if hasattr(self, "transformer"):
            return self.transformer.wte
        return self.decoder.transformer.wte
    
    def warm_up(self, gpt_model=None, geneformer_model=None):
        """
        Load pre-trained weights for the model components
        """
        if geneformer_model is not None:
            self.encoder = GeneformerModel.from_pretrained(geneformer_model)
        if gpt_model is not None:
            self.decoder = _GPT2LMHeadModel.from_pretrained(gpt_model, add_cross_attention=True, use_cache=False)
            self.decoder.resize_token_embeddings(self.gpt_config.vocab_size)
            self.decoder.config = self.gpt_config
    
    def forward(
        self,
        expression_tokens: Optional[torch.LongTensor] = None,
        expression_token_lengths: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values_fusion: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        cell_attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        get_cell_emb: Optional[bool] = False,
        **kwargs
    ):
        use_cache = use_cache if use_cache is not None else self.gpt_config.use_cache
        return_dict = return_dict if return_dict is not None else self.gpt_config.use_return_dict
        
        # Handle batch dimension reshaping if needed
        if decoder_input_ids is not None and len(decoder_input_ids.size()) == 3:
            decoder_input_ids = decoder_input_ids.squeeze(0)
        
        # Process gene expression data through Geneformer
        if expression_tokens is not None:
            # Use GeneformerModel to get cell embeddings
            encoder_outputs = self.encoder(
                expression_tokens=expression_tokens,
                expression_token_lengths=expression_token_lengths,
                return_dict=return_dict
            )
            
            cell_emb = encoder_outputs[1]  # Get embeddings from output tuple
            
            # Apply projection if needed
            cell_emb = self.to_embedding(cell_emb)
            
            # Apply fusion if configured
            if hasattr(self.config, "fusion_method") and self.config.fusion_method == "cross_attention":
                if past_key_values_fusion is None:
                    past_key_values_fusion = tuple([None] * len(self.h))
                
                output_shape = cell_emb.size()
                all_self_attentions = () if output_attentions else None
                all_cross_attentions = () if output_attentions and self.gpt_config.add_cross_attention else None
                all_hidden_states = () if output_hidden_states else None
                
                for i, (block, layer_past) in enumerate(zip(self.h, past_key_values_fusion)):
                    outputs = block(
                        cell_emb,
                        layer_past=layer_past,
                        attention_mask=cell_attention_mask,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                    )
                    cell_emb = outputs[0]
                
                cell_emb = self.ln_f(cell_emb)
                cell_emb = cell_emb.view(output_shape)
        
        else:
            # If no expression tokens provided, use pre-computed encoder states if available
            cell_emb = encoder_hidden_states
            cell_attention_mask = encoder_attention_mask
        
        if get_cell_emb:
            return cell_emb
        
        # Process through GPT2 decoder
        transformer_outputs = self.decoder(
            input_ids=decoder_input_ids,
            past_key_values=past_key_values,
            attention_mask=decoder_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=cell_emb,
            encoder_attention_mask=cell_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        return transformer_outputs
    
    @torch.no_grad()
    def generate_cell_description(
        self,
        expression_tokens: Optional[torch.LongTensor] = None,
        expression_token_lengths: Optional[torch.LongTensor] = None,
        tokenizer=None,
        device='cpu'
    ):
        """
        Generate text description for cell expression data
        """
        if expression_tokens is None:
            raise ValueError(
                "You need to provide expression_tokens representing gene expression data"
            )
        
        self.eval()
        self.to(device)
        
        # Prepare inputs
        inputs = {
            'expression_tokens': expression_tokens.to(device),
            'expression_token_lengths': expression_token_lengths.to(device)
        }
        
        # Create decoder input with BOS token
        inputs['decoder_input_ids'] = torch.ones((expression_tokens.shape[0], 1), 
                                               dtype=torch.long, 
                                               device=device) * tokenizer.bos_token_id
        inputs['decoder_attention_mask'] = torch.ones_like(inputs['decoder_input_ids'])
        
        # Get encoder outputs
        encoder_state = self(**inputs, get_cell_emb=True, output_attentions=True)
        
        # Generate text with the decoder
        encoder_outputs = {'hidden_states': encoder_state, 'attentions': None}
        
        # Create attention mask for encoder outputs if needed
        if hasattr(self.config, "use_cell_attention_mask") and self.config.use_cell_attention_mask:
            # Use a default mask covering all encoder outputs
            encoder_attention_mask = torch.ones((expression_tokens.shape[0], encoder_state.shape[1]), 
                                             device=device)
        else:
            encoder_attention_mask = None
        
        # Generate sequence
        generated_ids = self.decoder.generate(
            input_ids=inputs['decoder_input_ids'],
            encoder_outputs=encoder_outputs,
            use_cache=True,
            encoder_attention_mask=encoder_attention_mask,
            max_length=self.config.max_generation_length if hasattr(self.config, "max_generation_length") else 100,
            num_beams=self.config.num_beams if hasattr(self.config, "num_beams") else 4,
            early_stopping=True,
            length_penalty=1.0,
            no_repeat_ngram_size=3
        )
        
        # Decode generated tokens
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Clean up any special tokens that might remain
        if hasattr(tokenizer, "additional_special_tokens"):
            for token in tokenizer.additional_special_tokens:
                generated_text = [text.replace(token, "") for text in generated_text]
        
        return generated_text[0]
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer = None,
        **kwargs
    ):
        """
        General generate method compatible with Hugging Face generation pipeline
        """
        # Get encoder embeddings
        encoder_state = self(**kwargs, get_cell_emb=True)
        
        # Extract required arguments
        input_ids = kwargs.get('decoder_input_ids')
        attention_mask = kwargs.get('decoder_attention_mask')
        
        # Create encoder_attention_mask if needed
        encoder_attention_mask = kwargs.get('cell_attention_mask')
        
        # Create clean kwargs for generation
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in [
            'expression_tokens', 'expression_token_lengths', 'decoder_input_ids', 
            'decoder_attention_mask', 'cell_attention_mask', 'get_cell_emb'
        ]}
        
        # Call decoder's generate method
        return self.decoder.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            encoder_outputs={'hidden_states': encoder_state, 'attentions': None},
            encoder_attention_mask=encoder_attention_mask,
            **clean_kwargs
        )