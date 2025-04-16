import torch
import torch.nn as nn
from transformers import GPT2Config, PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from typing import Optional, Tuple, Union, Callable, List
import os
# Import encoders
from .geneformer_encoder import GeneformerModel, GeneformerConfig
from .pubmedbert_encoder import PubMedBertEncoder

from .cell2text_encoder import Cell2TextEncoder

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

 
        self.encoder = Cell2TextEncoder(config)
        
        # GPT2 decoder with cross-attention
        self.gpt_model = _GPT2LMHeadModel(self.gpt_config)

        self.warm_up()
        
        # If fusion of embeddings is needed
        if hasattr(config, "fusion_method") and config.fusion_method == "cross_attention":
            self.h = nn.ModuleList([CABlock(self.gpt_config, layer_idx=i) for i in range(4)])
            self.ln_f = nn.LayerNorm(self.gpt_config.n_embd, eps=self.gpt_config.layer_norm_epsilon)
        
        # Linear projections to match dimensions if needed
        if hasattr(config, "cell_encoder_hidden_size") and config.cell_encoder_hidden_size != self.gpt_config.n_embd:
            self.cell_to_embedding = nn.Linear(config.cell_encoder_hidden_size, self.gpt_config.n_embd)
        else:
            self.cell_to_embedding = nn.Identity()
            
        if hasattr(config, "text_encoder_hidden_size") and config.text_encoder_hidden_size != self.gpt_config.n_embd:
            self.text_to_embedding = nn.Linear(config.text_encoder_hidden_size, self.gpt_config.n_embd)
        else:
            self.text_to_embedding = nn.Identity()
        

        
        # We will probably keep the attention here
        # Embedding fusion module to combine cell and text embeddings (if both are present)
        if hasattr(config, "embedding_fusion_method"):
            self.embedding_fusion_method = config.embedding_fusion_method
            if self.embedding_fusion_method == "concatenate":
                self.fusion_layer = nn.Linear(self.gpt_config.n_embd * 2, self.gpt_config.n_embd)
            elif self.embedding_fusion_method == "attention":
                self.fusion_query = nn.Linear(self.gpt_config.n_embd, self.gpt_config.n_embd)
                self.fusion_key = nn.Linear(self.gpt_config.n_embd, self.gpt_config.n_embd)
                self.fusion_value = nn.Linear(self.gpt_config.n_embd, self.gpt_config.n_embd)
                self.fusion_layer_norm = nn.LayerNorm(self.gpt_config.n_embd)
            elif self.embedding_fusion_method == "sum":
                # Simple summation, no parameters needed
                pass
            elif self.embedding_fusion_method == "linear":
                self.cell_weight = nn.Parameter(torch.tensor(0.5))
                self.text_weight = nn.Parameter(torch.tensor(0.5))
            else:
                self.embedding_fusion_method = "cell_only"  # Default to cell only if invalid method
        else:
            self.embedding_fusion_method = "cell_only"  # Default to cell only if not specified
        
        self.config = config
    
    def get_cell_encoder(self):
        return self.cell_encoder
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_decoder(self):
        return self.decoder
    
    def get_input_embeddings(self):
        if hasattr(self, "transformer"):
            return self.transformer.wte
        return self.decoder.transformer.wte
    
    def fuse_embeddings(self, cell_emb, text_emb):
        """
        Fuse cell and text embeddings based on the specified fusion method
        """
        if self.embedding_fusion_method == "concatenate":
            # Concatenate along feature dimension and project back to original size
            combined = torch.cat([cell_emb, text_emb], dim=-1)
            return self.fusion_layer(combined)
        
        elif self.embedding_fusion_method == "attention":
            # Use self-attention mechanism to fuse embeddings
            query = self.fusion_query(cell_emb)
            key = self.fusion_key(text_emb)
            value = self.fusion_value(text_emb)
            
            # Compute attention scores
            attention_scores = torch.matmul(query, key.transpose(-1, -2)) / (self.gpt_config.n_embd ** 0.5)
            attention_probs = torch.softmax(attention_scores, dim=-1)
            
            # Apply attention weights to values
            context_layer = torch.matmul(attention_probs, value)
            
            # Add residual connection and normalization
            fused_emb = self.fusion_layer_norm(cell_emb + context_layer)
            return fused_emb
        
        elif self.embedding_fusion_method == "sum":
            # Simple element-wise sum
            return cell_emb + text_emb
        
        elif self.embedding_fusion_method == "linear":
            # Weighted sum using learnable weights
            return self.cell_weight * cell_emb + self.text_weight * text_emb
        
        else:
            # Default to cell embeddings only
            return cell_emb
    
    def warm_up(self):
        """
        Load pre-trained weights for the model components
        """
        print("HERE")
        if self.pubmedbert_model is not None:
            self.text_encoder = PubMedBertEncoder.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
        if self.gpt_model is not None:
            self.decoder = _GPT2LMHeadModel.from_pretrained("gpt2", add_cross_attention=True, use_cache=False)
            self.decoder.resize_token_embeddings(self.gpt_config.vocab_size)
            self.decoder.config = self.gpt_config
        
        if self.geneformer_model is not None:
            
            
            self.cell_encoder = GeneformerModel.from_pretrained("geneformer_model")
    
    def forward(
        self,
        expression_tokens: Optional[torch.LongTensor] = None,
        expression_token_lengths: Optional[torch.LongTensor] = None,
        input_tokens: Optional[torch.LongTensor] = None,
        input_token_lengths: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values_fusion: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        cell_attention_mask: Optional[torch.FloatTensor] = None,
        text_attention_mask: Optional[torch.FloatTensor] = None,
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
        print(">>> [Forward] Starting forward pass")
        use_cache = use_cache if use_cache is not None else self.gpt_config.use_cache
        return_dict = return_dict if return_dict is not None else self.gpt_config.use_return_dict

        if decoder_input_ids is not None and len(decoder_input_ids.size()) == 3:
            decoder_input_ids = decoder_input_ids.squeeze(0)

        cell_emb = None
        text_emb = None

        try:
            if expression_tokens is not None:
                print(">>> [Forward] Running cell encoder")
                encoder_outputs = self.cell_encoder(
                    expression_tokens=expression_tokens,
                    expression_token_lengths=expression_token_lengths,
                    return_dict=return_dict
                )
                cell_emb = encoder_outputs[1]
                print(">>> [Forward] Cell encoder output shape:", cell_emb.shape)

                cell_emb = self.cell_to_embedding(cell_emb)

                if hasattr(self.config, "fusion_method") and self.config.fusion_method == "cross_attention":
                    print(">>> [Forward] Applying cross-attention fusion to cell embeddings")
                    if past_key_values_fusion is None:
                        past_key_values_fusion = tuple([None] * len(self.h))

                    output_shape = cell_emb.size()
                    for i, (block, layer_past) in enumerate(zip(self.h, past_key_values_fusion)):
                        print(f">>> [Forward] CABlock {i}")
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
            elif encoder_hidden_states is not None and kwargs.get('encoder_type', 'cell') == 'cell':
                print(">>> [Forward] Using precomputed cell encoder hidden states")
                cell_emb = encoder_hidden_states
                cell_attention_mask = encoder_attention_mask
        except Exception as e:
            print(">>> [Forward] Error in cell encoder:", str(e))
            raise

        try:
            if input_tokens is not None:
                print(">>> [Forward] Running text encoder")
                text_encoder_outputs = self.text_encoder(
                    input_tokens=input_tokens,
                    input_token_lengths=input_token_lengths,
                    return_dict=return_dict
                )
                text_emb = text_encoder_outputs[1]
                print(">>> [Forward] Text encoder output shape:", text_emb.shape)

                text_emb = self.text_to_embedding(text_emb)
            elif encoder_hidden_states is not None and kwargs.get('encoder_type', 'cell') == 'text':
                print(">>> [Forward] Using precomputed text encoder hidden states")
                text_emb = encoder_hidden_states
                text_attention_mask = encoder_attention_mask
        except Exception as e:
            print(">>> [Forward] Error in text encoder:", str(e))
            raise

        try:
            print(">>> [Forward] Fusing encoder outputs")
            if cell_emb is not None and text_emb is not None:
                encoder_emb = self.fuse_embeddings(cell_emb, text_emb)
                print(">>> [Forward] Combined embedding shape:", encoder_emb.shape)
                encoder_attention_mask = (
                    cell_attention_mask if cell_attention_mask is not None else text_attention_mask
                )
            elif cell_emb is not None:
                encoder_emb = cell_emb
                encoder_attention_mask = cell_attention_mask
            elif text_emb is not None:
                encoder_emb = text_emb
                encoder_attention_mask = text_attention_mask
            else:
                encoder_emb = encoder_hidden_states
                encoder_attention_mask = encoder_attention_mask
        except Exception as e:
            print(">>> [Forward] Error during embedding fusion:", str(e))
            raise

        if get_embeddings:
            print(">>> [Forward] Returning encoder embeddings only")
            return encoder_emb

        try:
            print(">>> [Forward] Running decoder")

            if encoder_emb is not None and encoder_emb.dim() == 2:
                encoder_emb = encoder_emb.unsqueeze(1)
            transformer_outputs = self.decoder(
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
            print(">>> [Forward] Decoder ran successfully")
        except Exception as e:
            print(">>> [Forward] Error in decoder:", str(e))
            raise

        return transformer_outputs

    
    @torch.no_grad()
    def generate_cell_description(
        self,
        expression_tokens: Optional[torch.LongTensor] = None,
        expression_token_lengths: Optional[torch.LongTensor] = None,
        input_tokens: Optional[torch.LongTensor] = None,
        input_token_lengths: Optional[torch.LongTensor] = None,
        tokenizer=None,
        device='cpu',
        use_both_encoders=True
    ):
        """
        Generate text description for cell expression data, optionally using text context
        """
        if expression_tokens is None and input_tokens is None:
            raise ValueError(
                "You need to provide at least one of expression_tokens or input_tokens"
            )
        
        self.eval()
        self.to(device)
        
        # Prepare inputs
        inputs = {}
        
        if expression_tokens is not None:
            inputs['expression_tokens'] = expression_tokens.to(device)
            inputs['expression_token_lengths'] = expression_token_lengths.to(device)
        
        if input_tokens is not None:
            inputs['input_tokens'] = input_tokens.to(device)
            inputs['input_token_lengths'] = input_token_lengths.to(device)
        
        # Create decoder input with BOS token
        batch_size = expression_tokens.shape[0] if expression_tokens is not None else input_tokens.shape[0]
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
        encoder_state = self(**kwargs, get_embeddings=True)
        
        # Extract required arguments
        input_ids = kwargs.get('decoder_input_ids')
        attention_mask = kwargs.get('decoder_attention_mask')
        
        # Create encoder_attention_mask if needed
        encoder_attention_mask = kwargs.get('cell_attention_mask', kwargs.get('text_attention_mask'))
        
        # Create clean kwargs for generation
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in [
            'expression_tokens', 'expression_token_lengths', 'input_tokens', 'input_token_lengths', 
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