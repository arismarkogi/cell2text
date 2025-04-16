import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from typing import Optional, Tuple, Union, Dict, Any
from .geneformer_encoder import GeneformerModel, GeneformerConfig
from .pubmedbert_encoder import PubMedBertEncoder



class Cell2TextEncoderConfig(PretrainedConfig):
    """Configuration class for Cell2TextEncoder."""
    
    model_type = "cell2text_encoder"
    
    def __init__(
        self,
        num_classes: int = 0,
        cell_emb_mode: str = "cell",
        cell_encoder_hidden_size: int = 512,
        text_encoder_hidden_size: int = 768,
        max_ncells: int = 200,
        cell_emb_layer: int = -1,
        text_emb_layer: int = -1,
        cell_emb_label: list = ["sample_name", "cell type rough", "cell type"],
        text_emb_mode: str = "cls",
        forward_batch_size: int = -1,
        nproc: int = 4,
        projection_dim: int = 512,  # Unified dimension for cell and text projections
        projection_dropout: float = 0.1,
        use_self_attention: bool = True,
        num_attention_heads: int = 4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.cell_emb_mode = cell_emb_mode
        self.cell_encoder_hidden_size = cell_encoder_hidden_size
        self.text_encoder_hidden_size = text_encoder_hidden_size
        self.max_ncells = max_ncells
        self.cell_emb_layer = cell_emb_layer
        self.text_emb_layer = text_emb_layer
        self.cell_emb_label = cell_emb_label
        self.text_emb_mode = text_emb_mode
        self.forward_batch_size = forward_batch_size
        self.nproc = nproc
        self.projection_dim = projection_dim
        self.projection_dropout = projection_dropout
        self.use_self_attention = use_self_attention
        self.num_attention_heads = num_attention_heads


class SelfAttentionProjection(nn.Module):
    """Self-attention based projection layer."""
    
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        
        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Projection layers
        self.linear1 = nn.Linear(input_dim, input_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(input_dim * 2, output_dim)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)
        self.activation = nn.GELU()
    
    def forward(self, x):
        # For single vectors, expand dimensions to use self-attention
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, hidden_size]
            
        # Self-attention
        residual = x
        x, _ = self.self_attention(x, x, x)
        x = self.layer_norm1(residual + x)
        
        # Feed-forward projection
        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        
        if residual.size(-1) == self.output_dim:
            x = self.layer_norm2(residual + x)
        else:
            x = self.layer_norm2(x)  # Skip residual if dimensions don't match
        
        # If input was a single vector, convert back
        if x.size(1) == 1:
            x = x.squeeze(1)
            
        return x


class Cell2TextEncoder(PreTrainedModel):
    """Encoder model for cell and text data with contrastive learning capabilities."""
    
    config_class = Cell2TextEncoderConfig
    base_model_prefix = "cell_text_encoder"
    
    def __init__(self, config):
        super().__init__(config)
        
        # Geneformer encoder
        geneformer_config = GeneformerConfig(
            num_classes=config.num_classes,
            emb_mode=config.cell_emb_mode,
            hidden_size=config.cell_encoder_hidden_size,
            max_ncells=config.max_ncells,
            emb_layer=config.cell_emb_layer,
            emb_label=config.cell_emb_label,
            forward_batch_size=config.forward_batch_size,
            nproc=config.nproc,
        )
        self.cell_encoder = GeneformerModel(geneformer_config)
        
        # PubMedBERT encoder
        self.text_encoder = PubMedBertEncoder(
            num_classes=config.num_classes,
            emb_mode=config.text_emb_mode,
            hidden_size=config.text_encoder_hidden_size,
            emb_layer=config.text_emb_layer,
            forward_batch_size=config.forward_batch_size,
            nproc=config.nproc,
        )
        
        # Projection layers with self-attention
        if config.use_self_attention:
            self.cell_projection = SelfAttentionProjection(
                input_dim=config.cell_encoder_hidden_size,
                output_dim=config.projection_dim,
                num_heads=config.num_attention_heads,
                dropout=config.projection_dropout
            )
            self.text_projection = SelfAttentionProjection(
                input_dim=config.text_encoder_hidden_size,
                output_dim=config.projection_dim,
                num_heads=config.num_attention_heads,
                dropout=config.projection_dropout
            )
        else:
            # Simple projection layers without self-attention
            self.cell_projection = nn.Sequential(
                nn.Linear(config.cell_encoder_hidden_size, config.projection_dim),
                nn.LayerNorm(config.projection_dim),
                nn.Dropout(config.projection_dropout),
                nn.GELU(),
                nn.Linear(config.projection_dim, config.projection_dim),
                nn.LayerNorm(config.projection_dim)
            )
            
            self.text_projection = nn.Sequential(
                nn.Linear(config.text_encoder_hidden_size, config.projection_dim),
                nn.LayerNorm(config.projection_dim),
                nn.Dropout(config.projection_dropout),
                nn.GELU(),
                nn.Linear(config.projection_dim, config.projection_dim),
                nn.LayerNorm(config.projection_dim)
            )
        
    def forward(
        self,
        cell_tokens: Optional[torch.LongTensor] = None,
        cell_tokens_augmented: Optional[torch.LongTensor] = None,
        cell_lengths: Optional[torch.LongTensor] = None,
        cell_lengths_augmented: Optional[torch.LongTensor] = None,
        text_tokens: Optional[torch.LongTensor] = None,
        text_tokens_augmented: Optional[torch.LongTensor] = None,
        text_lengths: Optional[torch.LongTensor] = None,
        text_lengths_augmented: Optional[torch.LongTensor] = None,
        return_dict: bool = True,
    ):
        outputs = {}
        
        # Process cell data if available
        if cell_tokens is not None:
            cell_output = self.cell_encoder(
                expression_tokens=cell_tokens,
                expression_token_lengths=cell_lengths,
                return_dict=return_dict
            )
            cell_emb = cell_output[1]  # second element is embedding
            outputs["cell_embeds"] = self.cell_projection(cell_emb)
            
            # Process augmented cell data if available
            if cell_tokens_augmented is not None:
                cell_aug_output = self.cell_encoder(
                    expression_tokens=cell_tokens_augmented,
                    expression_token_lengths=cell_lengths_augmented,
                    return_dict=return_dict
                )
                cell_emb_aug = cell_aug_output[1]
                outputs["cell_embeds_aug"] = self.cell_projection(cell_emb_aug)
        
        # Process text data if available
        if text_tokens is not None:
            text_output = self.text_encoder(
                input_tokens=text_tokens,
                input_token_lengths=text_lengths,
                return_dict=return_dict
            )
            text_emb = text_output[1]  # second element is embedding
            outputs["text_embeds"] = self.text_projection(text_emb)
            
            # Process augmented text data if available
            if text_tokens_augmented is not None:
                text_aug_output = self.text_encoder(
                    input_tokens=text_tokens_augmented,
                    input_token_lengths=text_lengths_augmented,
                    return_dict=return_dict
                )
                text_emb_aug = text_aug_output[1]
                outputs["text_embeds_aug"] = self.text_projection(text_emb_aug)
        
        return outputs