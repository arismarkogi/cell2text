import torch
import torch.nn as nn
from einops import repeat
import math


class MLPProjectionLayer(nn.Module):
    """
    Creates a multi-layer perceptron (MLP) projection layer with GELU activation.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_prob: float = 0.0, bias: bool = True) -> None:
        super().__init__()

        layers = [
            nn.Linear(input_dim, hidden_dim, bias=bias),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        ]
        if dropout_prob > 0:
            layers.append(nn.Dropout(p=dropout_prob))
        layers.extend([
            nn.Linear(hidden_dim, output_dim, bias=bias),
            nn.LayerNorm(output_dim)
        ])
        self.projection = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class PerceiverIO(nn.Module):
    """
    Perceiver IO implementation with both cross-attention and self-attention
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_latents: int = 128,
        num_cross_attn_layers: int = 1,  
        num_self_attn_layers: int = 6,   
        num_heads: int = 8,
        ff_mult: int = 2,
        dropout: float = 0.1,
        use_position_encoding: bool = True,
        max_seq_len: int = 10000,
    ):
        """
        Args:
            input_dim: Dimension of input embeddings
            output_dim: Dimension of output embeddings and latents
            num_latents: Number of learnable latent queries
            num_cross_attn_layers: Number of cross-attention layers (usually 1)
            num_self_attn_layers: Number of self-attention layers (the main processing)
            num_heads: Number of attention heads
            ff_mult: Feedforward dimension multiplier
            dropout: Dropout rate
            use_position_encoding: Whether to add positional encodings to inputs
            max_seq_len: Maximum sequence length for position encodings
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_latents = num_latents
        self.num_cross_attn_layers = num_cross_attn_layers
        self.num_self_attn_layers = num_self_attn_layers
        self.use_position_encoding = use_position_encoding
        
        # Learnable latent queries (the "bottleneck")
        self.latents = nn.Parameter(torch.randn(num_latents, output_dim))
        
        # Input projection to match latent dimension
        self.input_projection = nn.Linear(input_dim, output_dim)
        
        # Position encoding for inputs (optional but often helpful)
        if use_position_encoding:
            self.position_encoding = PositionalEncoding(output_dim, max_seq_len)
        
        # Cross-attention layers (encode inputs into latents)
        self.cross_attention_layers = nn.ModuleList([
            PerceiverCrossAttentionLayer(
                dim=output_dim,
                num_heads=num_heads,
                ff_mult=ff_mult,
                dropout=dropout
            ) for _ in range(num_cross_attn_layers)
        ])
        
        # Self-attention layers (process latents - this is the key Perceiver innovation)
        self.self_attention_layers = nn.ModuleList([
            PerceiverSelfAttentionLayer(
                dim=output_dim,
                num_heads=num_heads,
                ff_mult=ff_mult,
                dropout=dropout
            ) for _ in range(num_self_attn_layers)
        ])
        
        # Final output normalization
        self.final_norm = nn.LayerNorm(output_dim)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters"""
        nn.init.normal_(self.latents, mean=0.0, std=0.02)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, inputs: torch.Tensor, mask: torch.Tensor = None):
        """
        Forward pass through Perceiver
        
        Args:
            inputs: [batch_size, seq_len, input_dim] - input embeddings
            mask: [batch_size, seq_len] - attention mask (True = valid, False = padding)
        
        Returns:
            [batch_size, num_latents, output_dim] - processed latent representation
        """
        batch_size = inputs.size(0)
        
        # Project input to output dimension
        projected_inputs = self.input_projection(inputs)
        
        # Add position encoding if enabled
        if self.use_position_encoding:
            projected_inputs = self.position_encoding(projected_inputs)
        
        # Initialize latents for this batch
        latents = repeat(self.latents, 'q d -> b q d', b=batch_size)
        
        # Phase 1: Cross-attention (encode inputs into latents)
        for cross_layer in self.cross_attention_layers:
            latents = cross_layer(latents, projected_inputs, mask)
        
        # Phase 2: Self-attention (process latents - the main computation)
        for self_layer in self.self_attention_layers:
            latents = self_layer(latents)
        
        # Final normalization
        output = self.final_norm(latents)
        
        return output


class PerceiverCrossAttentionLayer(nn.Module):
    """
    Cross-attention layer: latents attend to inputs
    """
    
    def __init__(self, dim: int, num_heads: int = 8, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        
        # Cross-attention: latents (queries) attend to inputs (keys, values)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(dim, int(dim * ff_mult)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * ff_mult), dim),
            nn.Dropout(dropout)
        )
        
        # Layer norms (pre-norm style)
        self.norm1 = nn.LayerNorm(dim)  # Before cross-attention
        self.norm2 = nn.LayerNorm(dim)  # Before feedforward
        
    def forward(self, latents: torch.Tensor, inputs: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            latents: [batch_size, num_latents, dim] - latent queries
            inputs: [batch_size, seq_len, dim] - input sequence to attend to
            mask: [batch_size, seq_len] - attention mask (True = valid, False = padding)
        
        Returns:
            [batch_size, num_latents, dim] - updated latents
        """
        # Cross-attention with residual connection
        norm_latents = self.norm1(latents)
        
        # Convert mask format if provided
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask  # Invert for PyTorch convention
        
        attn_out, _ = self.cross_attn(
            query=norm_latents,
            key=inputs,
            value=inputs,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        
        latents = latents + attn_out
        
        # Feedforward with residual connection
        ff_out = self.feedforward(self.norm2(latents))
        latents = latents + ff_out
        
        return latents


class PerceiverSelfAttentionLayer(nn.Module):
    """
    Self-attention layer: latents attend to other latents
    This is the key innovation of the Perceiver - processing in the latent space
    """
    
    def __init__(self, dim: int, num_heads: int = 8, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        
        # Self-attention: latents attend to latents
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(dim, int(dim * ff_mult)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * ff_mult), dim),
            nn.Dropout(dropout)
        )
        
        # Layer norms (pre-norm style)
        self.norm1 = nn.LayerNorm(dim)  # Before self-attention
        self.norm2 = nn.LayerNorm(dim)  # Before feedforward
        
    def forward(self, latents: torch.Tensor):
        """
        Args:
            latents: [batch_size, num_latents, dim] - latent queries
        
        Returns:
            [batch_size, num_latents, dim] - updated latents
        """
        # Self-attention with residual connection
        norm_latents = self.norm1(latents)
        
        attn_out, _ = self.self_attn(
            query=norm_latents,
            key=norm_latents,
            value=norm_latents,
            need_weights=False
        )
        
        latents = latents + attn_out
        
        # Feedforward with residual connection
        ff_out = self.feedforward(self.norm2(latents))
        latents = latents + ff_out
        
        return latents


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding
    """
    
    def __init__(self, dim: int, max_len: int = 10000):
        super().__init__()
        
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, dim]
        Returns:
            [batch_size, seq_len, dim] with positional encoding added
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

