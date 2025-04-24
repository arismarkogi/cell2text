import torch
import torch.nn as nn
from transformers import GPT2Config, PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from typing import Optional, Tuple, Callable, List

# Import encoders
from .geneformer_encoder import GeneformerModel, GeneformerConfig

# Import from Prot2Text implementation
from .utils import  _GPT2LMHeadModel


class Cell2TextModel(PreTrainedModel):
    config_class = PretrainedConfig
    _keys_to_ignore_on_load_missing = [r"transformer"]
    base_model_prefix = "decoder"
    
    def __init__(self, config):
        super().__init__(config)
        

        
        # GPT2 configuration for the decoder
        self.gpt_config = GPT2Config.from_dict(config.gpt_config)


        
       # Initialize the Geneformer encoder for cell data
        geneformer_config = GeneformerConfig(
            emb_mode=config.emb_mode if hasattr(config, "emb_mode") else "cell",
            max_ncells=config.max_ncells if hasattr(config, "max_ncells") else 1000,
            emb_layer=config.emb_layer if hasattr(config, "emb_layer") else -1,
            emb_label=config.emb_label if hasattr(config, "emb_label") else None,
            nproc=config.nproc if hasattr(config, "nproc") else -1,
            forward_batch_size=config.forward_batch_size if hasattr(config, "forward_batch_size") else 100,
            summary_stat=config.summary_stat if hasattr(config, "summary_stat") else None,
            token_dictionary_path=config.token_dictionary_path if hasattr(config, "token_dictionary_path") else "/home/arismarkog/Desktop/cell2text/Geneformer/geneformer/token_dictionary_gc95M.pkl"
        )

        self.geneformer_model = GeneformerModel(geneformer_config)
        
        # GPT2 decoder with cross-attention
        self.gpt_model = _GPT2LMHeadModel(self.gpt_config)
        

        self.warm_up()
        
        # Linear projections to match dimensions
        self.cell_to_embedding = nn.Linear(config.cell_encoder_hidden_size, self.gpt_config.n_embd)

            
        # # debugging
        # for name, param in self.named_parameters():
        #     print(f"{name}: requires_grad = {param.requires_grad}")

        self.config = config
    
    def get_cell_encoder(self):
        return self.cell_encoder
    
    def get_decoder(self):
        return self.decoder
    
    def get_input_embeddings(self):
        return self.decoder.transformer.wte
    
    def warm_up(self):
        """
        Load pre-trained weights for the model components
        """
   
        if self.gpt_model is not None:
            self.decoder = _GPT2LMHeadModel.from_pretrained("gpt2", add_cross_attention=True, use_cache=False)
            self.decoder.resize_token_embeddings(self.gpt_config.vocab_size)
            self.decoder.config = self.gpt_config
        
        if self.geneformer_model is not None:
            self.cell_encoder = GeneformerModel.from_pretrained(pretrained_model_name_or_path=self.config.geneformer_path)
    
    def forward(
        self,
        expression_tokens: Optional[torch.LongTensor] = None,
        expression_token_lengths: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
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
        get_embeddings: Optional[bool] = False,
        **kwargs
    ):
        use_cache = use_cache if use_cache is not None else self.gpt_config.use_cache
        return_dict = return_dict if return_dict is not None else self.gpt_config.use_return_dict

        # Run the cell encoder if input is provided
        if expression_tokens is not None:
            encoder_outputs = self.cell_encoder(
                expression_tokens=expression_tokens,
                expression_token_lengths=expression_token_lengths,
                return_dict=return_dict
            )

            cell_emb = encoder_outputs[1]  # assume this is [batch_size, hidden_dim]

            # Linear projection
            cell_emb = self.cell_to_embedding(cell_emb)

            encoder_emb = cell_emb.unsqueeze(1)  # [batch_size, 1, n_embd]

            encoder_attention_mask = torch.ones(
                (encoder_emb.size(0), encoder_emb.size(1)), device=encoder_emb.device
            )

        else:
            raise ValueError("You must provide expression_tokens.")

        if get_embeddings:
            return encoder_emb

        # Run decoder
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            past_key_values=past_key_values,
            attention_mask=decoder_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_emb,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return decoder_outputs


    
    @torch.no_grad()
    def generate_cell_description(
        self,
        expression_tokens: Optional[torch.LongTensor] = None,
        expression_token_lengths: Optional[torch.LongTensor] = None,
        tokenizer=None,
        device='cpu',
        use_both_encoders=True
    ):
        """
        Generate text description for cell expression data, optionally using text context
        """
        if expression_tokens is None:
            raise ValueError(
                "You need to provide expression_tokens"
            )
        
        self.eval()
        self.to(device)
        
        # Prepare inputs
        inputs = {}
        
        inputs['expression_tokens'] = expression_tokens.to(device)
        inputs['expression_token_lengths'] = expression_token_lengths.to(device)
        
        
        
        # Create decoder input with BOS token
        batch_size = expression_tokens.shape[0] 
        inputs['decoder_input_ids'] = torch.ones((batch_size, 1), 
                                               dtype=torch.long, 
                                               device=device) * tokenizer.bos_token_id
        
        inputs['decoder_attention_mask'] = torch.ones_like(inputs['decoder_input_ids'])
        
        # Get encoder outputs
        encoder_state = self(**inputs, get_embeddings=True, output_attentions=True)
        
        # Generate text with the decoder
        encoder_outputs = {'hidden_states': encoder_state, 'attentions': None}
        
        # Create attention mask for encoder outputs if needed
        if hasattr(self.config, "use_encoder_attention_mask") and self.config.use_encoder_attention_mask:
            # Use a default mask covering all encoder outputs
            encoder_attention_mask = torch.ones((batch_size, encoder_state.shape[1]), 
                                             device=device)
        else:
            encoder_attention_mask = None
        
        # # Generate sequence
        # generated_ids = self.decoder.generate(
        #     input_ids=inputs['decoder_input_ids'],
        #     encoder_outputs=encoder_outputs,
        #     use_cache=True,
        #     encoder_attention_mask=encoder_attention_mask,
        #     max_length=self.config.max_generation_length if hasattr(self.config, "max_generation_length") else 100,
        #     num_beams=self.config.num_beams if hasattr(self.config, "num_beams") else 4,
        #     early_stopping=True,
        #     length_penalty=1.0,
        #     no_repeat_ngram_size=3
        # )

        generated_ids = self.decoder.generate(
            input_ids=inputs['decoder_input_ids'],
            encoder_outputs=encoder_outputs,
            encoder_attention_mask=encoder_attention_mask,
            max_length=100,  # or whatever you need
            num_beams=1,     # <-- switch to greedy
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            attention_mask=inputs['decoder_attention_mask'],
            early_stopping=True,
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
        encoder_state = self(**kwargs, get_embeddings=True)
        
        # Extract required arguments
        input_ids = kwargs.get('decoder_input_ids')
        attention_mask = kwargs.get('decoder_attention_mask')
        
        # Create encoder_attention_mask if needed
        encoder_attention_mask = kwargs.get('cell_attention_mask', kwargs.get('text_attention_mask'))
        
        # Create clean kwargs for generation
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in [
            'expression_tokens', 'expression_token_lengths', 
            'decoder_input_ids', 'decoder_attention_mask', 'cell_attention_mask', 
            'text_attention_mask', 'get_embeddings'
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
