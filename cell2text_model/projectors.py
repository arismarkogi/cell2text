import torch.nn as nn
import torch
from einops import repeat
from flash_attn import flash_attn_func
from flash_attn.modules.mha import MHA


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
                dropout=dropout,
                attention_backend="auto"  # Will auto-select best available
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
    Supports multiple attention backends: standard, PyTorch SDPA, Flash Attention
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
        attention_backend: str = "auto"  # "standard", "pytorch_sdpa", "flash_attn", "auto"
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attention_backend = self._select_backend(attention_backend)
        
        # Attention projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        # Alternative: use standard MultiheadAttention for simplicity
        self.standard_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Flash Attention module
        self.flash_mha = MHA(
                embed_dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
                causal=False  # Cross-attention is not causal
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
        
        self.dropout = nn.Dropout(dropout)
        
        
   
    
    def _cross_attention_flash(self, latents: torch.Tensor, inputs: torch.Tensor, mask: torch.Tensor = None):
        """Flash Attention implementation"""
        # Flash Attention expects concatenated QKV for self-attention
        # For cross-attention, we need to handle Q and KV separately
        batch_size, num_latents, _ = latents.shape
        _, seq_len, _ = inputs.shape
        
        # Project
        q = self.q_proj(latents).view(batch_size, num_latents, self.num_heads, self.head_dim)
        k = self.k_proj(inputs).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(inputs).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Flash attention expects (batch, seq_len, num_heads, head_dim)
        # Use flash_attn_func for cross-attention
        attn_out = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            causal=False
        )
        
        # Reshape and project
        attn_out = attn_out.view(batch_size, num_latents, self.dim)
        return self.out_proj(attn_out)
        
    def forward(self, latents: torch.Tensor, inputs: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            latents: [batch_size, num_latents, dim] - latent queries
            inputs: [batch_size, seq_len, dim] - input sequence to attend to
            mask: [batch_size, seq_len] - attention mask
        
        Returns:
            [batch_size, num_latents, dim] - updated latents
        """
        # Normalize before attention
        norm_latents = self.norm1(latents)
        
        # Cross-attention with selected backend
      
        attn_out = self._cross_attention_flash(norm_latents, inputs, mask)
        
        # Residual connection
        latents = latents + attn_out
        
        # Feedforward with residual connection
        ff_out = self.feedforward(self.norm2(latents))
        latents = latents + ff_out
        
        return latents


      