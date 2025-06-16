import torch.nn as nn
import torch


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



class SimplePerceiverResampler(nn.Module):
    """
    Simple Perceiver resampler without question conditioning.
    Just compresses Gene Expression Embeddings to fixed-size representation.
    """
    def __init__(self, cell_embedding_dim, output_dim, num_latents, num_layers, num_heads):
        super().__init__()
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(num_latents, output_dim))
        
        self.input_projection = nn.Linear(cell_embedding_dim, output_dim)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=output_dim,
                nhead=num_heads,
                dim_feedforward=output_dim * 4,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, cell_embeddings):
        """
        cell_embeddings: [B, L, cell_embedding_dim]
        Returns: [B, num_latents, output_dim]
        """
        B = cell_embeddings.size(0)
        
        # Project cell embeddings
        projected_inputs = self.input_projection(cell_embeddings)
        
        # Prepare latents
        latents = self.latents.unsqueeze(0).repeat(B, 1, 1)
        
        # Cross-attention: latents attend to cell
        latents, _ = self.cross_attn(latents, projected_inputs, projected_inputs)
        
        # Self-attention layers
        for layer in self.layers:
            latents = layer(latents)
            
        return latents


class EnglishAwarePerceiverResampler(nn.Module):
    """
    Perceiver resampler with question conditioning.
    Modulates latents based on the input question.
    """
    def __init__(self, cell_embedding_dim, lm_embedding_dim, num_latents, num_layers, num_heads, question_dim):
        super().__init__()
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(num_latents, lm_embedding_dim))
        
        # Question conditioning
        self.latent_modulation = nn.Linear(question_dim, lm_embedding_dim)
        
        self.input_projection = nn.Linear(cell_embedding_dim, lm_embedding_dim)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=lm_embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=lm_embedding_dim,
                nhead=num_heads,
                dim_feedforward=lm_embedding_dim * 4,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, cell_embeddings, question_embedding):
        """
        cell_embeddings: [B, L, cell_embedding_dim]
        question_embedding: [B, question_dim]
        Returns: [B, num_latents, lm_embedding_dim]
        """
        B = cell_embeddings.size(0)
        
        # Project Cell embeddings
        projected_inputs = self.input_projection(cell_embeddings)
        
        # Prepare and modulate latents with question
        latents = self.latents.unsqueeze(0).repeat(B, 1, 1)
        question_modulation = self.latent_modulation(question_embedding).unsqueeze(1)
        latents = latents + question_modulation
        
        # Cross-attention: latents attend to Cell
        latents, _ = self.cross_attn(latents, projected_inputs, projected_inputs)
        
        # Self-attention layers
        for layer in self.layers:
            latents = layer(latents)
            
        return latents


class FiLMConditionedMLPProjector(nn.Module):
    """
    MLP projector with FiLM (Feature-wise Linear Modulation) conditioning.
    Uses question to modulate features at each layer via scale and shift.
    """
    def __init__(self, cell_embedding_dim, hidden_dim, output_dim, question_dim, 
                 num_layers=2, dropout_prob=0.0, bias=True):
        super().__init__()
        self.num_layers = num_layers
        
        # Main MLP layers
        self.layers = nn.ModuleList()
        dims = [cell_embedding_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        
        for i in range(num_layers):
            self.layers.append(nn.Linear(dims[i], dims[i + 1], bias=bias))
        
        # FiLM conditioning networks - generate scale and shift for each layer
        self.film_networks = nn.ModuleList()
        for i in range(num_layers):
            # Each FiLM network outputs both scale (gamma) and shift (beta)
            film_dim = dims[i + 1]
            self.film_networks.append(
                nn.Sequential(
                    nn.Linear(question_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, film_dim * 2)  # *2 for gamma and beta
                )
            )
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(dims[i + 1]) for i in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()
        
    def forward(self, cell_embeddings, question_embedding):
        """
        cell_embeddings: [B, L, cell_embedding_dim]
        question_embedding: [B, question_dim]
        Returns: [B, L, output_dim]
        """
        x = cell_embeddings
        
        for i, (layer, film_net, layer_norm) in enumerate(zip(self.layers, self.film_networks, self.layer_norms)):
            # Apply linear transformation
            x = layer(x)
            
            # Generate FiLM parameters
            film_params = film_net(question_embedding)  # [B, film_dim * 2]
            film_dim = film_params.size(-1) // 2
            gamma = film_params[..., :film_dim].unsqueeze(1)  # [B, 1, film_dim]
            beta = film_params[..., film_dim:].unsqueeze(1)   # [B, 1, film_dim]
            
            # Apply layer norm
            x = layer_norm(x)
            
            # Apply FiLM modulation: x = gamma * x + beta
            x = gamma * x + beta
            
            # Apply activation (except for last layer)
            if i < len(self.layers) - 1:
                x = torch.nn.functional.gelu(x)
                x = self.dropout(x)
        
        return x