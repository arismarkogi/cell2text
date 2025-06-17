import torch.nn as nn
import torch
from einops import repeat


class MLPProjectionLayer(nn.Module):
    """
    Creates a multi-layer perceptron (MLP) projection layer with GELU activation.

    Args:
        input_dim (int): The dimension of the input to the projection layer.
        hidden_dim (int): The dimension of the hidden layer in the MLP.
        output_dim (int): The dimension of the output from the projection layer.
        dropout_prob (float, optional): The probability of dropout. If 0, dropout is not applied.
            Default: 0.0
        bias (bool, optional): If set to False, the linear layers will not learn an additive bias.
            Default: True
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
        """
        Performs the MLP projection.

        Args:
            x (torch.Tensor): The input tensor to project.

        Returns:
            torch.Tensor: The projected tensor.
        """
        return self.projection(x)


class SimplifiedPerceiverResampler(nn.Module):
    """
    Simplified Perceiver Resampler using standard PyTorch components
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_latents: int = 128,
        depth: int = 4,
        num_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Dimension of input embeddings (e.g., gene embeddings)
            output_dim: Dimension of output embeddings and latents
            num_latents: Number of learnable latent queries
            depth: Number of cross-attention + feedforward layers
            num_heads: Number of attention heads
            ff_mult: Feedforward dimension multiplier
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_latents = num_latents
        self.depth = depth
        
        # Learnable latent queries
        self.latents = nn.Parameter(torch.randn(num_latents, output_dim))
        
        # Input projection to match latent dimension
        self.input_projection = nn.Linear(input_dim, output_dim)
        
        # Stack of Perceiver layers
        self.layers = nn.ModuleList([
            PerceiverLayer(
                dim=output_dim,
                num_heads=num_heads,
                ff_mult=ff_mult,
                dropout=dropout
            ) for _ in range(depth)
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
        Forward pass
        
        Args:
            inputs: [batch_size, seq_len, input_dim] - input embeddings
            mask: [batch_size, seq_len] - attention mask (True = valid, False = padding)
        
        Returns:
            [batch_size, num_latents, output_dim] - compressed representation
        """
        batch_size = inputs.size(0)
        
        # Project input to output dimension
        projected_inputs = self.input_projection(inputs)
        
        # Prepare latents for batch
        latents = repeat(self.latents, 'q d -> b q d', b=batch_size)
        
        # Apply Perceiver layers
        for layer in self.layers:
            latents = layer(latents, projected_inputs, mask)
        
        # Final normalization
        output = self.final_norm(latents)
        
        return output


class PerceiverLayer(nn.Module):
    """
    Single Perceiver layer with cross-attention and feedforward
    Uses PyTorch's optimized scaled dot-product attention when available
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        
        # Cross-attention using PyTorch's MultiheadAttention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout)
        )
        
        # Layer norms
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
        # Normalize before attention
        norm_latents = self.norm1(latents)
        
        # Convert mask format if provided
        # PyTorch MultiheadAttention expects key_padding_mask where True = ignore
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask  # Invert: True = valid -> False = valid
        
        # Cross-attention: latents attend to inputs
        attn_out, _ = self.cross_attn(
            query=norm_latents,
            key=inputs,
            value=inputs,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        
        # Residual connection
        latents = latents + attn_out
        
        # Feedforward with residual connection
        ff_out = self.feedforward(self.norm2(latents))
        latents = latents + ff_out
        
        return latents